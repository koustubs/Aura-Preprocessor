"""
LLM Client module for Groq API interaction.

This module provides a wrapper for the Groq API to handle authentication,
request formatting, response parsing, and error handling.
"""

import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union
import jsonschema
from jsonschema import validate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from . import config
from . import utils

class GroqClient:
    """
    Client for interacting with the Groq API.
    
    This class handles authentication, request formatting, response parsing,
    error handling, and caching for Groq API interactions.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Groq API client.
        
        Args:
            api_key: Groq API key (defaults to config.GROQ_API_KEY)
            base_url: Groq API URL (defaults to config.GROQ_BASE_URL)
            model: LLM model to use (defaults to config.DEFAULT_MODEL)
            
        Raises:
            ValueError: If no API key is provided and none is found in config
        """
        self.logger = logging.getLogger('auto_preprocessor.llm_client')
        
        self.api_key = api_key or config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env file or pass directly.")
            
        self.base_url = base_url or config.GROQ_BASE_URL
        self.model = model or config.DEFAULT_MODEL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
    def _prepare_messages(
        self, 
        system_prompt: str, 
        user_content: str
    ) -> List[Dict[str, str]]:
        """
        Prepare message format for Groq API.
        
        Args:
            system_prompt: The system prompt to guide the LLM behavior
            user_content: The user message content
            
        Returns:
            List of formatted message dictionaries
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def _prepare_response_format(self) -> Dict[str, Any]:
        """
        Prepare response format specification for structured JSON output.
        
        Returns:
            Response format dictionary for API request
        """
        return {
            "type": "json_object",
            "schema": config.LLM_RESPONSE_SCHEMA
        }
    
    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=config.RETRY_DELAY, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError))
    )
    def _make_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Groq API with retry logic.
        
        Args:
            payload: The request payload
            
        Returns:
            API response as a dictionary
            
        Raises:
            requests.exceptions.RequestException: For request-related errors
            ValueError: For invalid API responses
        """
        try:
            response = self.session.post(
                self.base_url,
                json=payload,
                timeout=config.REQUEST_TIMEOUT
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Response content: {e.response.text}")
            raise
            
    def _parse_and_validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate the Groq API response.
        
        Args:
            response: Raw API response
            
        Returns:
            Parsed and validated JSON content
            
        Raises:
            ValueError: If response validation fails
            KeyError: If expected response structure is missing
        """
        try:
            # Extract the content from the response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Parse the JSON content
            parsed_content = json.loads(content)
            
            # Validate against schema
            validate(instance=parsed_content, schema=config.LLM_RESPONSE_SCHEMA)
            
            return parsed_content
            
        except (json.JSONDecodeError, jsonschema.exceptions.ValidationError, KeyError, IndexError) as e:
            self.logger.error(f"Response validation failed: {str(e)}")
            self.logger.error(f"Raw response content: {response}")
            raise ValueError(f"Invalid response from LLM: {str(e)}")
    
    def analyze_dataset(
        self, 
        dataset_description: str, 
        data_sample: str, 
        statistics: Dict[str, Any], 
        system_prompt: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a dataset using the Groq API.
        
        Args:
            dataset_description: Description of the dataset
            data_sample: String representation of the dataset sample
            statistics: Dictionary of dataset statistics
            system_prompt: Custom system prompt (defaults to config.DEFAULT_SYSTEM_PROMPT)
            custom_instructions: Additional instructions for the LLM
            
        Returns:
            Parsed and validated response from the LLM
        """
        system_prompt = system_prompt or config.DEFAULT_SYSTEM_PROMPT
        
        # Create cache key
        cache_data = {
            "model": self.model,
            "system_prompt": system_prompt,
            "dataset_description": dataset_description,
            "data_sample": data_sample,
            "statistics": statistics,
            "custom_instructions": custom_instructions
        }
        cache_key = utils.create_cache_key(cache_data)
        
        # Check cache first
        cached_result = utils.get_from_cache(cache_key)
        if cached_result:
            self.logger.info("Using cached LLM response")
            return cached_result
        
        # Format user message
        user_content = self._format_analysis_prompt(dataset_description, data_sample, statistics, custom_instructions)
        
        # Prepare API request
        messages = self._prepare_messages(system_prompt, user_content)
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": self._prepare_response_format(),
            "temperature": 0.2,  # Low temperature for more deterministic results
        }
        
        self.logger.info(f"Sending analysis request to Groq API (model: {self.model})")
        
        # Make API request
        start_time = time.time()
        response = self._make_api_request(payload)
        elapsed_time = time.time() - start_time
        self.logger.info(f"Received response in {elapsed_time:.2f} seconds")
        
        # Parse and validate response
        result = self._parse_and_validate_response(response)
        
        # Cache result
        utils.save_to_cache(cache_key, result)
        
        return result
    
    def _format_analysis_prompt(
        self, 
        dataset_description: str, 
        data_sample: str, 
        statistics: Dict[str, Any],
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Format the analysis prompt for the LLM.
        
        Args:
            dataset_description: Description of the dataset
            data_sample: String representation of the dataset sample
            statistics: Dictionary of dataset statistics
            custom_instructions: Additional instructions for the LLM
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
# Dataset Analysis Task

## Dataset Description
{dataset_description}

## Expected Response Format
Please provide analysis results in the following JSON schema:
```json
{{
  "dataset_overview": {{
    "title": "string",
    "n_rows": number,
    "n_cols": number, 
    "missing_overall_pct": number,
    "target_candidates": ["column_names"]
  }},
  "column_reports": [
    {{
      "name": "string",
      "dtype_guess": "string",
      "role_guess": "categorical|numerical|datetime|id|target",
      "missing_pct": number,
      "unique_count_est": number,
      "encoding_suggestion": "string",
      "scaling_suggestion": "string",
      "warnings": ["list of issues"]
    }}
  ],
  "quality_issues": [
    {{
      "type": "missing_values|duplicates|outliers|multicollinearity",
      "severity": "low|medium|high", 
      "evidence": "string",
      "suggested_fix": "string",
      "columns_involved": ["column_names"]
    }}
  ],
  "actions_plan": [
    {{
      "action": "drop|impute|encode|scale|transform",
      "columns": ["column_names"],
      "params": object,
      "rationale": "string",
      "expected_effect": "string"
    }}
  ]
}}
```

## Dataset Statistics
```json
{json.dumps(statistics, indent=2)}
```

## Dataset Sample
```
{data_sample}
```

{custom_instructions or ''}

Please analyze this dataset for ML preprocessing requirements. Return ONLY the JSON response according to the schema above.
"""
        return prompt