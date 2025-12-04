"""
Test suite for AutoDataPreprocessor.

This module provides unit tests for the key components of AutoDataPreprocessor.
"""

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np
from unittest import mock
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils
from src.data_analyzer import DataAnalyzer
from src.data_cleaner import DataCleaner
from src.llm_client import GroqClient


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_file_extension(self):
        """Test file extension extraction."""
        self.assertEqual(utils.get_file_extension('/path/to/file.csv'), '.csv')
        self.assertEqual(utils.get_file_extension('/path/to/file.CSV'), '.csv')
        self.assertEqual(utils.get_file_extension('/path/to/file.xlsx'), '.xlsx')
        
    def test_is_supported_file(self):
        """Test file format validation."""
        self.assertTrue(utils.is_supported_file('/path/to/file.csv'))
        self.assertTrue(utils.is_supported_file('/path/to/file.xlsx'))
        self.assertTrue(utils.is_supported_file('/path/to/file.xls'))
        self.assertFalse(utils.is_supported_file('/path/to/file.txt'))
        self.assertFalse(utils.is_supported_file('/path/to/file.json'))
        
    def test_generate_output_filepath(self):
        """Test output filepath generation."""
        self.assertEqual(
            utils.generate_output_filepath('/path/to/file.csv', '/output/dir'),
            '/output/dir/file.csv'
        )
        self.assertEqual(
            utils.generate_output_filepath('/path/to/file.csv', '/output/dir', '_clean'),
            '/output/dir/file_clean.csv'
        )


class TestDataCleaner(unittest.TestCase):
    """Test data cleaning functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner()
        
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', np.nan, 'Eve'],
            'age': [25, 30, np.nan, 40, 35],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
    def test_drop_columns(self):
        """Test dropping columns."""
        result = self.cleaner.drop_columns(self.df, ['id'])
        self.assertNotIn('id', result.columns)
        self.assertEqual(len(result.columns), len(self.df.columns) - 1)
        
        # Test with non-existent column
        result = self.cleaner.drop_columns(self.df, ['id', 'non_existent'])
        self.assertNotIn('id', result.columns)
        self.assertEqual(len(result.columns), len(self.df.columns) - 1)
        
    def test_impute_missing(self):
        """Test missing value imputation."""
        # Test mean imputation
        result = self.cleaner.impute_missing(self.df, ['age'], method='mean')
        self.assertFalse(result['age'].isna().any())
        self.assertEqual(result['age'][2], self.df['age'].mean())
        
        # Test mode imputation
        result = self.cleaner.impute_missing(self.df, ['name'], method='mode')
        self.assertFalse(result['name'].isna().any())
        
        # Test with all columns
        result = self.cleaner.impute_missing(self.df, ['age', 'name'], method='mean')
        self.assertFalse(result['age'].isna().any())
        self.assertFalse(result['name'].isna().any())
        
    def test_encode_categorical(self):
        """Test categorical encoding."""
        # Test one-hot encoding
        result = self.cleaner.encode_categorical(self.df, ['category'], method='one-hot')
        self.assertIn('category_A', result.columns)
        self.assertIn('category_B', result.columns)
        self.assertIn('category_C', result.columns)
        self.assertNotIn('category', result.columns)
        
        # Test label encoding
        result = self.cleaner.encode_categorical(self.df, ['category'], method='label')
        self.assertIn('category_encoded', result.columns)
        self.assertEqual(len(result['category_encoded'].unique()), 3)
        
    def test_scale_numerical(self):
        """Test numerical scaling."""
        # Test standard scaling
        result = self.cleaner.scale_numerical(self.df, ['age'], method='standard')
        self.assertTrue(abs(result['age'].mean()) < 1e-10)  # Mean should be close to 0
        self.assertTrue(abs(result['age'].std() - 1.0) < 1e-10)  # Std should be close to 1
        
        # Test min-max scaling
        result = self.cleaner.scale_numerical(self.df, ['age'], method='minmax')
        self.assertEqual(result['age'].min(), 0.0)  # Min should be 0
        self.assertEqual(result['age'].max(), 1.0)  # Max should be 1


class TestDataAnalyzer(unittest.TestCase):
    """Test data analyzer functions."""
    
    @mock.patch('src.data_analyzer.DataAnalyzer.load_dataset')
    def test_create_dataset_sample(self, mock_load_dataset):
        """Test dataset sampling."""
        # Create a mock analyzer
        analyzer = DataAnalyzer()
        
        # Create a sample DataFrame
        df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.rand(1000)
        })
        
        # Test with small DataFrame (should return copy of original)
        small_df = df.head(100)
        sample = analyzer.create_dataset_sample(small_df, max_rows=200)
        self.assertEqual(len(sample), len(small_df))
        
        # Test with large DataFrame (should return sample)
        sample = analyzer.create_dataset_sample(df, max_rows=200)
        self.assertEqual(len(sample), 200)
        
        # Test with stratified sampling
        df['category'] = ['A', 'B'] * 500  # Add a categorical column
        sample = analyzer.create_dataset_sample(df, max_rows=200)
        self.assertEqual(len(sample), 200)
        # Should have both categories
        self.assertGreater(len(sample[sample['category'] == 'A']), 0)
        self.assertGreater(len(sample[sample['category'] == 'B']), 0)


@mock.patch('src.llm_client.requests.Session')
class TestGroqClient(unittest.TestCase):
    """Test Groq API client."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        os.environ['GROQ_API_KEY'] = 'test_api_key'
        
    def test_initialization(self, mock_session):
        """Test client initialization."""
        client = GroqClient()
        self.assertEqual(client.api_key, 'test_api_key')
        mock_session.return_value.headers.update.assert_called_once()
        
    def test_prepare_messages(self, mock_session):
        """Test message preparation."""
        client = GroqClient()
        system_prompt = "You are a data scientist."
        user_content = "Analyze this dataset."
        
        messages = client._prepare_messages(system_prompt, user_content)
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], system_prompt)
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], user_content)
        
    @mock.patch('src.llm_client.GroqClient._make_api_request')
    def test_analyze_dataset(self, mock_api_request, mock_session):
        """Test dataset analysis."""
        client = GroqClient()
        
        # Mock response from API
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "dataset_overview": {
                            "title": "Test Dataset",
                            "n_rows": 100,
                            "n_cols": 5,
                            "missing_overall_pct": 2.5,
                            "target_candidates": ["target"]
                        },
                        "column_reports": [{
                            "name": "col1",
                            "dtype_guess": "numeric",
                            "missing_pct": 0
                        }],
                        "quality_issues": [{
                            "type": "missing_values",
                            "severity": "low",
                            "suggested_fix": "impute",
                            "columns_involved": ["col2"]
                        }],
                        "actions_plan": [{
                            "action": "impute",
                            "columns": ["col2"],
                            "rationale": "Fill missing values"
                        }]
                    })
                }
            }]
        }
        
        mock_api_request.return_value = mock_response
        
        # Call the method
        result = client.analyze_dataset(
            dataset_description="Test dataset",
            data_sample="col1,col2\n1,2\n3,4",
            statistics={"test": "stats"}
        )
        
        # Check results
        self.assertEqual(result["dataset_overview"]["title"], "Test Dataset")
        self.assertEqual(result["column_reports"][0]["name"], "col1")
        self.assertEqual(result["quality_issues"][0]["type"], "missing_values")
        self.assertEqual(result["actions_plan"][0]["action"], "impute")
        
        # Ensure API request was made
        mock_api_request.assert_called_once()


if __name__ == '__main__':
    unittest.main()