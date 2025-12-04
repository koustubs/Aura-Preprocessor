"""
Configuration module for AutoDataPreprocessor.

This module contains all configuration variables for the project, including API keys,
data processing settings, and output configurations.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_BASE_URL = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = 'llama-3.1-8b-instant'  # Free tier model

# Data Processing Settings
MAX_SAMPLE_ROWS = 50
MAX_COLUMNS_PER_REQUEST = 12
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
RANDOM_SEED = 42

# LLM Request Settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
REQUEST_TIMEOUT = 30  # seconds
CACHE_ENABLED = True
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'cache')

# Output Settings
ENABLE_LOGGING = True
SAVE_INTERMEDIATE_RESULTS = True
LOG_LEVEL = 'INFO'
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'reports')

# Paths
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
CLEANED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned')

# Default LLM System Prompt
DEFAULT_SYSTEM_PROMPT = """
You are a senior data scientist specializing in ML data preprocessing. 
Analyze the provided dataset sample and statistics.
Return ONLY valid JSON following the exact schema provided.
Focus on ML-specific preprocessing needs: missing values, encoding, scaling, feature engineering.
Be concise but specific in your recommendations.
"""

# Validation schema for LLM responses
LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "dataset_overview": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "n_rows": {"type": "number"},
                "n_cols": {"type": "number"},
                "missing_overall_pct": {"type": "number"},
                "target_candidates": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["title", "n_rows", "n_cols", "missing_overall_pct"]
        },
        "column_reports": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dtype_guess": {"type": "string"},
                    "role_guess": {"type": "string"},
                    "missing_pct": {"type": "number"},
                    "unique_count_est": {"type": "number"},
                    "encoding_suggestion": {"type": "string"},
                    "scaling_suggestion": {"type": "string"},
                    "warnings": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "dtype_guess", "missing_pct"]
            }
        },
        "quality_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "severity": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "columns_involved": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["type", "severity", "suggested_fix"]
            }
        },
        "actions_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "params": {"type": "object"},
                    "rationale": {"type": "string"},
                    "expected_effect": {"type": "string"}
                },
                "required": ["action", "columns", "rationale"]
            }
        }
    },
    "required": ["dataset_overview", "column_reports", "quality_issues", "actions_plan"]
}