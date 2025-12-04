"""
Utility functions for AutoDataPreprocessor.

This module contains helper functions used across the project for tasks like
file handling, data validation, logging, and other common operations.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Union, Tuple, Optional
from datetime import datetime
import hashlib
import pickle
from pathlib import Path

from . import config

# Set up logging
def setup_logging(level: str = config.LOG_LEVEL) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        A configured logger instance
    """
    log_level = getattr(logging, level.upper())
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up file with timestamp
    log_file = os.path.join(logs_dir, f'auto_preprocessor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('auto_preprocessor')
    return logger

# File handling utilities
def get_file_extension(file_path: str) -> str:
    """
    Extract the extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The file extension including the dot (e.g., '.csv')
    """
    return os.path.splitext(file_path)[1].lower()

def is_supported_file(file_path: str) -> bool:
    """
    Check if the file format is supported.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file format is supported, False otherwise
    """
    return get_file_extension(file_path) in config.SUPPORTED_FORMATS

def generate_output_filepath(input_filepath: str, directory: str, suffix: str = "") -> str:
    """
    Generate output file path based on input file path.
    
    Args:
        input_filepath: Path to the input file
        directory: Target directory for the output file
        suffix: String to append to the filename (before extension)
        
    Returns:
        Path to the output file
    """
    filename = os.path.basename(input_filepath)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}{suffix}{ext}"
    return os.path.join(directory, output_filename)

# Cache utilities
def create_cache_key(data: Any) -> str:
    """
    Create a unique key for caching based on the content of the data.
    
    Args:
        data: The data to create a cache key for
        
    Returns:
        A string hash to use as the cache key
    """
    if isinstance(data, pd.DataFrame):
        # For dataframes, use a hash of the column names, shape, and sample data
        df_info = {
            'columns': list(data.columns),
            'shape': data.shape,
            'head_hash': hashlib.md5(pd.util.hash_pandas_object(data.head()).values).hexdigest(),
            'tail_hash': hashlib.md5(pd.util.hash_pandas_object(data.tail()).values).hexdigest()
        }
        data_for_hash = json.dumps(df_info, sort_keys=True).encode()
    elif isinstance(data, dict):
        # For dictionaries, convert to a sorted JSON string
        data_for_hash = json.dumps(data, sort_keys=True).encode()
    elif isinstance(data, str):
        # For strings, encode directly
        data_for_hash = data.encode()
    else:
        # For other types, convert to string first
        data_for_hash = str(data).encode()
    
    return hashlib.md5(data_for_hash).hexdigest()

def get_from_cache(cache_key: str) -> Optional[Any]:
    """
    Retrieve data from cache.
    
    Args:
        cache_key: The cache key to look up
        
    Returns:
        The cached data if found, None otherwise
    """
    if not config.CACHE_ENABLED:
        return None
    
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(config.CACHE_DIR, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger = logging.getLogger('auto_preprocessor')
            logger.warning(f"Failed to load from cache: {e}")
    
    return None

def save_to_cache(cache_key: str, data: Any) -> None:
    """
    Save data to cache.
    
    Args:
        cache_key: The cache key to store the data under
        data: The data to cache
    """
    if not config.CACHE_ENABLED:
        return
    
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(config.CACHE_DIR, f"{cache_key}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger = logging.getLogger('auto_preprocessor')
        logger.warning(f"Failed to save to cache: {e}")

# JSON utilities
def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save
        filepath: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        The loaded data as a dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

# Data validation utilities
def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate a pandas DataFrame for basic issues.
    
    Args:
        df: The DataFrame to validate
        
    Returns:
        A tuple containing (is_valid, list_of_issues)
    """
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    if df.shape[1] == 0:
        issues.append("DataFrame has no columns")
        return False, issues
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        dup_cols = [col for col in df.columns if list(df.columns).count(col) > 1]
        issues.append(f"DataFrame contains duplicate column names: {set(dup_cols)}")
    
    # Check if any column has all missing values
    all_na_cols = df.columns[df.isna().all()].tolist()
    if all_na_cols:
        issues.append(f"The following columns have all missing values: {all_na_cols}")
    
    return len(issues) == 0, issues

# Initialize the logger
logger = setup_logging()