"""
Data Cleaner module for implementing preprocessing suggestions from LLM.

This module provides a set of data cleaning and preprocessing functions
based on the suggestions provided by the LLM analysis, including handling
missing values, encoding categorical variables, scaling numerical features,
handling outliers, and feature engineering.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Tuple, Optional, Callable
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.ensemble import IsolationForest

from . import config
from . import utils

class DataCleaner:
    """
    Implement data cleaning and preprocessing based on LLM suggestions.
    
    This class provides functions to apply various preprocessing operations
    to a DataFrame based on the suggestions provided by the LLM analysis.
    """
    
    def __init__(self):
        """Initialize the DataCleaner with default configurations."""
        self.logger = logging.getLogger('auto_preprocessor.data_cleaner')
        self.scaler = StandardScaler()
        self.encoders = {}
        self.imputers = {}
        
        # Define available actions for each column type
        self.available_actions = {
            'numerical': [
                {'action': 'skip', 'label': 'Skip - No Action', 'description': 'Leave column unchanged'},
                {'action': 'impute_mean', 'label': 'Fill Missing - Mean', 'description': 'Replace missing values with column mean'},
                {'action': 'impute_median', 'label': 'Fill Missing - Median', 'description': 'Replace missing values with column median'},
                {'action': 'impute_mode', 'label': 'Fill Missing - Mode', 'description': 'Replace missing values with most frequent value'},
                {'action': 'impute_constant', 'label': 'Fill Missing - Custom Value', 'description': 'Replace missing values with a specific value'},
                {'action': 'drop_missing', 'label': 'Remove Missing Rows', 'description': 'Delete rows with missing values in this column'},
                {'action': 'scale_standard', 'label': 'Standardize (Z-score)', 'description': 'Scale to mean=0, std=1'},
                {'action': 'scale_minmax', 'label': 'Min-Max Scaling', 'description': 'Scale to range [0,1]'},
                {'action': 'scale_robust', 'label': 'Robust Scaling', 'description': 'Scale using median and IQR'},
                {'action': 'outlier_clip_iqr', 'label': 'Clip Outliers (IQR)', 'description': 'Cap extreme values using IQR method'},
                {'action': 'outlier_clip_zscore', 'label': 'Clip Outliers (Z-score)', 'description': 'Cap extreme values using Z-score method'},
                {'action': 'outlier_remove_iqr', 'label': 'Remove Outliers (IQR)', 'description': 'Delete rows with extreme values'},
                {'action': 'log_transform', 'label': 'Log Transform', 'description': 'Apply log transformation'},
                {'action': 'sqrt_transform', 'label': 'Square Root Transform', 'description': 'Apply square root transformation'}
            ],
            'categorical': [
                {'action': 'skip', 'label': 'Skip - No Action', 'description': 'Leave column unchanged'},
                {'action': 'impute_mode', 'label': 'Fill Missing - Most Frequent', 'description': 'Replace missing values with mode'},
                {'action': 'impute_constant', 'label': 'Fill Missing - Custom Value', 'description': 'Replace missing values with a specific value'},
                {'action': 'drop_missing', 'label': 'Remove Missing Rows', 'description': 'Delete rows with missing values'},
                {'action': 'encode_onehot', 'label': 'One-Hot Encoding', 'description': 'Create binary columns for each category'},
                {'action': 'encode_label', 'label': 'Label Encoding', 'description': 'Convert categories to numbers'},
                {'action': 'encode_target', 'label': 'Target Encoding', 'description': 'Encode based on target variable'},
                {'action': 'group_rare', 'label': 'Group Rare Categories', 'description': 'Combine infrequent categories into "Other"'},
                {'action': 'text_clean', 'label': 'Clean Text', 'description': 'Remove special characters and normalize'}
            ],
            'datetime': [
                {'action': 'skip', 'label': 'Skip - No Action', 'description': 'Leave column unchanged'},
                {'action': 'extract_features', 'label': 'Extract Date Features', 'description': 'Create year, month, day, weekday columns'},
                {'action': 'convert_timestamp', 'label': 'Convert to Timestamp', 'description': 'Convert to Unix timestamp'},
                {'action': 'calculate_age', 'label': 'Calculate Age/Duration', 'description': 'Calculate time difference from reference date'},
                {'action': 'drop_column', 'label': 'Remove Column', 'description': 'Delete this column completely'}
            ],
            'text': [
                {'action': 'skip', 'label': 'Skip - No Action', 'description': 'Leave column unchanged'},
                {'action': 'text_clean', 'label': 'Clean Text', 'description': 'Remove special characters, normalize case'},
                {'action': 'extract_length', 'label': 'Extract Text Length', 'description': 'Create new column with text length'},
                {'action': 'extract_keywords', 'label': 'Extract Keywords', 'description': 'Extract important keywords or patterns'},
                {'action': 'drop_column', 'label': 'Remove Column', 'description': 'Delete this column completely'}
            ]
        }
    
    def process_dataset(
        self, 
        df: pd.DataFrame, 
        llm_response: Dict[str, Any] = None,
        custom_actions: Optional[Dict[str, Dict]] = None,
        actions_to_apply: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process the dataset using cleaning recommendations from LLM or custom actions.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            llm_response (Dict[str, Any], optional): LLM response with cleaning recommendations
            custom_actions (Dict[str, Dict], optional): User-specified actions per column
            actions_to_apply (List[str], optional): Specific actions to apply
            save_path (str, optional): Path to save cleaned data
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        self.logger.info("Starting dataset processing")
        df_cleaned = df.copy()
        
        # Use custom actions if provided, otherwise use LLM recommendations
        if custom_actions:
            self.logger.info("Using custom user-specified actions")
            for column_name, column_actions in custom_actions.items():
                if column_name in df_cleaned.columns:
                    self.logger.info(f"Processing column: {column_name} with custom actions")
                    for action_name, action_config in column_actions.items():
                        if action_name != 'skip':  # Skip columns marked as 'skip'
                            self.logger.info(f"Applying {action_name} to {column_name}")
                            action_dict = {'action': action_name, **action_config}
                            df_cleaned = self._apply_column_action(df_cleaned, column_name, action_dict)
        
        elif llm_response:
            # Apply global preprocessing actions
            if 'global_actions' in llm_response:
                for action in llm_response['global_actions']:
                    self.logger.info(f"Applying global action: {action}")
                    df_cleaned = self._apply_global_action(df_cleaned, action)
            
            # Apply column-specific actions
            if 'column_recommendations' in llm_response:
                for column_name, column_info in llm_response['column_recommendations'].items():
                    if column_name in df_cleaned.columns:
                        self.logger.info(f"Processing column: {column_name}")
                        
                        # Apply each recommended action for this column
                        if 'actions' in column_info:
                            for action in column_info['actions']:
                                if actions_to_apply is None or action['action'] in actions_to_apply:
                                    self.logger.info(f"Applying {action['action']} to {column_name}")
                                    df_cleaned = self._apply_column_action(df_cleaned, column_name, action)
        
        # Save if path provided
        if save_path:
            df_cleaned.to_csv(save_path, index=False)
            self.logger.info(f"Cleaned data saved to {save_path}")
        
        return df_cleaned
    
    def _apply_column_action(self, df: pd.DataFrame, column_name: str, action_dict: Dict[str, Any]) -> pd.DataFrame:
        """Apply a single action to a specific column."""
        action = action_dict.get('action', '')
        
        if action == 'skip':
            return df
        
        try:
            # Missing value handling
            if action.startswith('impute_'):
                method = action.replace('impute_', '')
                if method == 'constant':
                    value = action_dict.get('value', 0)
                    return self.impute_missing(df, [column_name], method='constant', fill_value=value)
                else:
                    return self.impute_missing(df, [column_name], method=method)
            
            elif action == 'drop_missing':
                return self.drop_rows(df, [column_name])
            
            # Scaling actions
            elif action.startswith('scale_'):
                method = action.replace('scale_', '')
                return self.scale_numerical(df, [column_name], method=method)
            
            # Outlier handling
            elif action.startswith('outlier_'):
                if 'clip' in action:
                    method = action.replace('outlier_clip_', '')
                    factor = action_dict.get('factor', 1.5)
                    return self.handle_outliers(df, [column_name], method='clip', outlier_method=method, factor=factor)
                elif 'remove' in action:
                    method = action.replace('outlier_remove_', '')
                    factor = action_dict.get('factor', 1.5)
                    return self.handle_outliers(df, [column_name], method='remove', outlier_method=method, factor=factor)
            
            # Encoding actions
            elif action.startswith('encode_'):
                method = action.replace('encode_', '')
                return self.encode_categorical(df, [column_name], method=method)
            
            # Transform actions
            elif action == 'log_transform':
                df_copy = df.copy()
                df_copy[column_name] = np.log1p(df_copy[column_name].fillna(0))
                return df_copy
            
            elif action == 'sqrt_transform':
                df_copy = df.copy()
                df_copy[column_name] = np.sqrt(df_copy[column_name].fillna(0).abs())
                return df_copy
            
            # Drop column action
            elif action == 'drop_column':
                return self.drop_columns(df, [column_name])
            
            else:
                self.logger.warning(f"Unknown action: {action} for column {column_name}")
                return df
                
        except Exception as e:
            self.logger.error(f"Error applying action {action} to column {column_name}: {str(e)}")
            return df
    
    def _apply_global_action(self, df: pd.DataFrame, action_dict: Dict[str, Any]) -> pd.DataFrame:
        """Apply a global action to the entire dataset."""
        action = action_dict.get('action', '')
        
        try:
            if action == 'remove_duplicates':
                return df.drop_duplicates()
            
            elif action == 'reset_index':
                return df.reset_index(drop=True)
            
            elif action == 'remove_empty_rows':
                return df.dropna(how='all')
            
            elif action == 'remove_empty_columns':
                return df.dropna(axis=1, how='all')
            
            else:
                self.logger.warning(f"Unknown global action: {action}")
                return df
                
        except Exception as e:
            self.logger.error(f"Error applying global action {action}: {str(e)}")
            return df
    
    def get_column_type(self, df: pd.DataFrame, column: str) -> str:
        """Determine the type of a column for action recommendations."""
        if column not in df.columns:
            return 'unknown'
        
        col_data = df[column]
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return 'datetime'
        
        # Try to parse as datetime
        if col_data.dtype == 'object':
            sample_non_null = col_data.dropna().head(10)
            if len(sample_non_null) > 0:
                try:
                    pd.to_datetime(sample_non_null.iloc[0])
                    return 'datetime'
                except:
                    pass
        
        # Check for numerical
        if pd.api.types.is_numeric_dtype(col_data):
            return 'numerical'
        
        # Check for categorical vs text
        if col_data.dtype == 'object':
            unique_ratio = col_data.nunique() / len(col_data.dropna())
            avg_length = col_data.dropna().astype(str).str.len().mean()
            
            # If many unique values and long strings, likely text
            if unique_ratio > 0.5 and avg_length > 50:
                return 'text'
            else:
                return 'categorical'
        
        return 'categorical'
    
    def get_available_actions(self, df: pd.DataFrame, column: str) -> List[Dict]:
        """Get available preprocessing actions for a specific column."""
        col_type = self.get_column_type(df, column)
        return self.available_actions.get(col_type, self.available_actions['categorical'])
    
    def get_column_analysis(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get analysis and available actions for all columns."""
        analysis = {}
        
        for column in df.columns:
            col_type = self.get_column_type(df, column)
            col_data = df[column]
            
            analysis[column] = {
                'type': col_type,
                'missing_count': col_data.isnull().sum(),
                'missing_percentage': (col_data.isnull().sum() / len(df)) * 100,
                'unique_count': col_data.nunique(),
                'available_actions': self.get_available_actions(df, column),
                'sample_values': col_data.dropna().head(5).tolist() if len(col_data.dropna()) > 0 else []
            }
            
            # Add type-specific analysis
            if col_type == 'numerical':
                analysis[column].update({
                    'mean': col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else None,
                    'std': col_data.std() if pd.api.types.is_numeric_dtype(col_data) else None,
                    'min': col_data.min() if pd.api.types.is_numeric_dtype(col_data) else None,
                    'max': col_data.max() if pd.api.types.is_numeric_dtype(col_data) else None
                })
            elif col_type == 'categorical':
                value_counts = col_data.value_counts().head(10)
                analysis[column]['top_values'] = value_counts.to_dict()
        
        return analysis
    
    # ===== CLEANING METHODS =====
    
    def drop_columns(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Drop columns from the DataFrame.
        
        Args:
            df: Input DataFrame
            columns: List of column names to drop
            **kwargs: Additional parameters (unused)
            
        Returns:
            DataFrame with columns dropped
        """
        # Filter columns to only those that exist in the DataFrame
        existing_columns = [col for col in columns if col in df.columns]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist: {missing_columns}")
        
        if not existing_columns:
            self.logger.warning("No valid columns to drop")
            return df
            
        self.logger.info(f"Dropping columns: {existing_columns}")
        return df.drop(columns=existing_columns)
    
    def drop_rows(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Drop rows based on conditions.
        
        Args:
            df: Input DataFrame
            columns: List of column names to consider for row dropping
            **kwargs: Additional parameters including:
                - threshold: Minimum number of non-NA values required (default: None)
                - subset: Columns to consider for NA values (default: None)
                - how: {'any', 'all'} - Drop if any or all values are NA (default: 'any')
                - duplicate_subset: Columns to consider for duplicates (default: None)
                - keep: {'first', 'last', False} - Which duplicates to keep (default: 'first')
            
        Returns:
            DataFrame with rows dropped
        """
        result_df = df.copy()
        
        # Check if we need to drop rows with NA values
        if kwargs.get('threshold') is not None or kwargs.get('subset') is not None or kwargs.get('how') is not None:
            # Prepare parameters
            drop_na_params = {}
            
            if kwargs.get('threshold') is not None:
                drop_na_params['thresh'] = kwargs['threshold']
                
            if kwargs.get('subset') is not None:
                subset = kwargs['subset']
                # Filter to existing columns
                subset = [col for col in subset if col in df.columns]
                if subset:
                    drop_na_params['subset'] = subset
                    
            if kwargs.get('how') is not None:
                drop_na_params['how'] = kwargs['how']
            
            # Drop NA rows
            old_shape = result_df.shape
            result_df = result_df.dropna(**drop_na_params)
            new_shape = result_df.shape
            
            rows_dropped = old_shape[0] - new_shape[0]
            self.logger.info(f"Dropped {rows_dropped} rows with NA values")
        
        # Check if we need to drop duplicate rows
        if kwargs.get('duplicate_subset') is not None or kwargs.get('keep') is not None:
            # Prepare parameters
            drop_dup_params = {}
            
            if kwargs.get('duplicate_subset') is not None:
                subset = kwargs['duplicate_subset']
                # Filter to existing columns
                subset = [col for col in subset if col in df.columns]
                if subset:
                    drop_dup_params['subset'] = subset
            
            if kwargs.get('keep') is not None:
                drop_dup_params['keep'] = kwargs['keep']
            
            # Drop duplicate rows
            old_shape = result_df.shape
            result_df = result_df.drop_duplicates(**drop_dup_params)
            new_shape = result_df.shape
            
            rows_dropped = old_shape[0] - new_shape[0]
            self.logger.info(f"Dropped {rows_dropped} duplicate rows")
        
        return result_df
    
    def impute_missing(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Impute missing values.
        
        Args:
            df: Input DataFrame
            columns: List of column names to impute
            **kwargs: Additional parameters including:
                - method: Imputation method ('mean', 'median', 'mode', 'constant', 'knn')
                - value: Value to use for constant imputation
                - n_neighbors: Number of neighbors for KNN imputation
            
        Returns:
            DataFrame with imputed values
        """
        result_df = df.copy()
        
        # Filter columns to only those that exist in the DataFrame
        existing_columns = [col for col in columns if col in df.columns]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist: {missing_columns}")
        
        if not existing_columns:
            self.logger.warning("No valid columns to impute")
            return df
        
        # Get imputation method, default to 'mean' for numeric and 'mode' for non-numeric
        method = kwargs.get('method')
        
        # Separate numeric and non-numeric columns
        numeric_cols = [col for col in existing_columns if pd.api.types.is_numeric_dtype(df[col])]
        non_numeric_cols = [col for col in existing_columns if col not in numeric_cols]
        
        # For each set of columns, apply appropriate imputation
        if numeric_cols and (method in ['mean', 'median', 'knn'] or method is None):
            actual_method = method or 'mean'
            self.logger.info(f"Imputing numeric columns {numeric_cols} using {actual_method}")
            
            if actual_method == 'knn':
                # KNN imputation for numeric columns
                n_neighbors = kwargs.get('n_neighbors', 5)
                imputer = KNNImputer(n_neighbors=n_neighbors)
                result_df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                # Mean or median imputation for numeric columns
                imputer = SimpleImputer(strategy=actual_method)
                result_df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        if non_numeric_cols:
            if method == 'constant' and 'value' in kwargs:
                # Constant imputation for non-numeric columns
                value = kwargs['value']
                self.logger.info(f"Imputing non-numeric columns {non_numeric_cols} with constant value '{value}'")
                
                for col in non_numeric_cols:
                    result_df[col] = result_df[col].fillna(value)
            else:
                # Mode imputation for non-numeric columns
                self.logger.info(f"Imputing non-numeric columns {non_numeric_cols} using mode")
                
                imputer = SimpleImputer(strategy='most_frequent')
                
                # Impute each non-numeric column separately
                for col in non_numeric_cols:
                    result_df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1)).ravel()
        
        return result_df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: List of column names to encode
            **kwargs: Additional parameters including:
                - method: Encoding method ('one-hot', 'label', 'ordinal')
                - drop_first: Whether to drop the first category in one-hot encoding
                - handle_unknown: How to handle unknown categories ('error', 'ignore')
                - categories: List of category orders for ordinal encoding
            
        Returns:
            DataFrame with encoded categorical variables
        """
        result_df = df.copy()
        
        # Filter columns to only those that exist in the DataFrame
        existing_columns = [col for col in columns if col in df.columns]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist: {missing_columns}")
        
        if not existing_columns:
            self.logger.warning("No valid columns to encode")
            return df
        
        # Get encoding method
        method = kwargs.get('method', 'one-hot')
        
        if method == 'one-hot':
            # One-hot encoding
            drop_first = kwargs.get('drop_first', False)
            handle_unknown = kwargs.get('handle_unknown', 'ignore')
            
            self.logger.info(f"One-hot encoding columns: {existing_columns}")
            
            # Create encoder
            encoder = OneHotEncoder(
                sparse=False,
                drop='first' if drop_first else None,
                handle_unknown=handle_unknown
            )
            
            # Fit and transform
            encoded = encoder.fit_transform(result_df[existing_columns])
            
            # Create feature names
            feature_names = encoder.get_feature_names_out(existing_columns)
            
            # Add encoded columns to result
            encoded_df = pd.DataFrame(encoded, index=result_df.index, columns=feature_names)
            
            # Drop original columns and add encoded ones
            result_df = result_df.drop(columns=existing_columns)
            result_df = pd.concat([result_df, encoded_df], axis=1)
            
        elif method == 'label':
            # Label encoding
            self.logger.info(f"Label encoding columns: {existing_columns}")
            
            # Apply label encoding to each column
            for col in existing_columns:
                encoder = LabelEncoder()
                # Handle NaNs by converting to a string representation
                filled_series = result_df[col].fillna('nan')
                result_df[f"{col}_encoded"] = encoder.fit_transform(filled_series)
            
            # Drop original columns if specified
            if kwargs.get('drop_original', True):
                result_df = result_df.drop(columns=existing_columns)
                
        elif method == 'ordinal':
            # Ordinal encoding
            self.logger.info(f"Ordinal encoding columns: {existing_columns}")
            
            # Get categories for each column
            categories = kwargs.get('categories', None)
            
            if categories:
                # Use specified category orders
                encoder = OrdinalEncoder(categories=categories)
                result_df[existing_columns] = encoder.fit_transform(result_df[existing_columns])
            else:
                # Use alphabetical ordering
                encoder = OrdinalEncoder()
                result_df[existing_columns] = encoder.fit_transform(result_df[existing_columns])
                
        else:
            self.logger.warning(f"Unsupported encoding method: {method}")
        
        return result_df
    
    def scale_numerical(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            columns: List of column names to scale
            **kwargs: Additional parameters including:
                - method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        result_df = df.copy()
        
        # Filter columns to only those that exist in the DataFrame and are numeric
        existing_columns = [
            col for col in columns 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist or are not numeric: {missing_columns}")
        
        if not existing_columns:
            self.logger.warning("No valid columns to scale")
            return df
        
        # Get scaling method
        method = kwargs.get('method', 'standard')
        
        if method == 'standard':
            # Standardization (z-score normalization)
            scaler = StandardScaler()
            self.logger.info(f"Standardizing columns: {existing_columns}")
            
        elif method == 'minmax':
            # Min-max scaling
            feature_range = kwargs.get('feature_range', (0, 1))
            scaler = MinMaxScaler(feature_range=feature_range)
            self.logger.info(f"Min-max scaling columns: {existing_columns}")
            
        elif method == 'robust':
            # Robust scaling (using quantiles)
            quantile_range = kwargs.get('quantile_range', (25.0, 75.0))
            scaler = RobustScaler(quantile_range=quantile_range)
            self.logger.info(f"Robust scaling columns: {existing_columns}")
            
        else:
            self.logger.warning(f"Unsupported scaling method: {method}")
            return result_df
        
        # Apply scaling
        result_df[existing_columns] = scaler.fit_transform(result_df[existing_columns])
        
        return result_df
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names to process
            **kwargs: Additional parameters including:
                - method: Method to handle outliers ('clip', 'remove', 'isolation_forest')
                - strategy: Strategy for clipping ('iqr', 'zscore')
                - factor: Factor for IQR or z-score thresholds
            
        Returns:
            DataFrame with outliers handled
        """
        result_df = df.copy()
        
        # Filter columns to only those that exist in the DataFrame and are numeric
        existing_columns = [
            col for col in columns 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist or are not numeric: {missing_columns}")
        
        if not existing_columns:
            self.logger.warning("No valid columns to process for outliers")
            return df
        
        # Get method for handling outliers
        method = kwargs.get('method', 'clip')
        
        if method == 'clip':
            # Clip outliers to bounds
            strategy = kwargs.get('strategy', 'iqr')
            
            if strategy == 'iqr':
                # IQR-based clipping
                factor = kwargs.get('factor', 1.5)
                self.logger.info(f"Clipping outliers using IQR (factor={factor}) for columns: {existing_columns}")
                
                for col in existing_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
                    
            elif strategy == 'zscore':
                # Z-score based clipping
                factor = kwargs.get('factor', 3.0)
                self.logger.info(f"Clipping outliers using z-score (factor={factor}) for columns: {existing_columns}")
                
                for col in existing_columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    lower_bound = mean - factor * std
                    upper_bound = mean + factor * std
                    
                    result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            else:
                self.logger.warning(f"Unsupported outlier strategy: {strategy}")
                
        elif method == 'remove':
            # Remove rows with outliers
            strategy = kwargs.get('strategy', 'iqr')
            mask = pd.Series(True, index=df.index)
            
            if strategy == 'iqr':
                # IQR-based removal
                factor = kwargs.get('factor', 1.5)
                self.logger.info(f"Removing outliers using IQR (factor={factor}) for columns: {existing_columns}")
                
                for col in existing_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                    mask = mask & col_mask
                    
            elif strategy == 'zscore':
                # Z-score based removal
                factor = kwargs.get('factor', 3.0)
                self.logger.info(f"Removing outliers using z-score (factor={factor}) for columns: {existing_columns}")
                
                for col in existing_columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    lower_bound = mean - factor * std
                    upper_bound = mean + factor * std
                    
                    col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                    mask = mask & col_mask
            
            else:
                self.logger.warning(f"Unsupported outlier strategy: {strategy}")
                return result_df
            
            # Apply the mask
            old_shape = result_df.shape
            result_df = result_df[mask]
            new_shape = result_df.shape
            
            rows_removed = old_shape[0] - new_shape[0]
            self.logger.info(f"Removed {rows_removed} rows with outliers")
            
        elif method == 'isolation_forest':
            # Isolation Forest for outlier detection
            contamination = kwargs.get('contamination', 'auto')
            self.logger.info(f"Using Isolation Forest (contamination={contamination}) for outlier detection")
            
            # Create and fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=config.RANDOM_SEED)
            outlier_predictions = iso_forest.fit_predict(df[existing_columns])
            
            # Convert predictions to boolean mask (1 for inliers, -1 for outliers)
            mask = outlier_predictions == 1
            
            # Apply the mask
            old_shape = result_df.shape
            result_df = result_df[mask]
            new_shape = result_df.shape
            
            rows_removed = old_shape[0] - new_shape[0]
            self.logger.info(f"Removed {rows_removed} rows identified as outliers by Isolation Forest")
            
        else:
            self.logger.warning(f"Unsupported outlier handling method: {method}")
        
        return result_df
    
    def transform_datetime(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Transform datetime columns into useful features.
        
        Args:
            df: Input DataFrame
            columns: List of column names to transform
            **kwargs: Additional parameters including:
                - format: Datetime format string
                - extract: List of features to extract ('year', 'month', 'day', 'hour', etc.)
                - drop_original: Whether to drop the original columns
            
        Returns:
            DataFrame with transformed datetime features
        """
        result_df = df.copy()
        
        # Filter columns to only those that exist in the DataFrame
        existing_columns = [col for col in columns if col in df.columns]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist: {missing_columns}")
        
        if not existing_columns:
            self.logger.warning("No valid columns to transform")
            return df
        
        # Get parameters
        dt_format = kwargs.get('format')
        features_to_extract = kwargs.get('extract', ['year', 'month', 'day'])
        drop_original = kwargs.get('drop_original', False)
        
        # Process each column
        for col in existing_columns:
            self.logger.info(f"Transforming datetime column: {col}, extracting {features_to_extract}")
            
            # Convert to datetime
            try:
                if dt_format:
                    dt_series = pd.to_datetime(result_df[col], format=dt_format, errors='coerce')
                else:
                    dt_series = pd.to_datetime(result_df[col], errors='coerce')
                    
                # Extract specified features
                for feature in features_to_extract:
                    if feature == 'year':
                        result_df[f"{col}_year"] = dt_series.dt.year
                    elif feature == 'month':
                        result_df[f"{col}_month"] = dt_series.dt.month
                    elif feature == 'day':
                        result_df[f"{col}_day"] = dt_series.dt.day
                    elif feature == 'hour':
                        result_df[f"{col}_hour"] = dt_series.dt.hour
                    elif feature == 'minute':
                        result_df[f"{col}_minute"] = dt_series.dt.minute
                    elif feature == 'weekday':
                        result_df[f"{col}_weekday"] = dt_series.dt.weekday
                    elif feature == 'quarter':
                        result_df[f"{col}_quarter"] = dt_series.dt.quarter
                    elif feature == 'is_weekend':
                        result_df[f"{col}_is_weekend"] = (dt_series.dt.weekday >= 5).astype(int)
                    elif feature == 'is_month_start':
                        result_df[f"{col}_is_month_start"] = dt_series.dt.is_month_start.astype(int)
                    elif feature == 'is_month_end':
                        result_df[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(int)
                    elif feature == 'day_of_year':
                        result_df[f"{col}_day_of_year"] = dt_series.dt.dayofyear
                    else:
                        self.logger.warning(f"Unsupported datetime feature: {feature}")
                        
                # Drop original column if specified
                if drop_original:
                    result_df = result_df.drop(columns=[col])
                    
            except Exception as e:
                self.logger.warning(f"Error transforming datetime column {col}: {str(e)}")
        
        return result_df
    
    def bin_numerical(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Bin numerical features into categories.
        
        Args:
            df: Input DataFrame
            columns: List of column names to bin
            **kwargs: Additional parameters including:
                - n_bins: Number of bins (default: 5)
                - strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
                - labels: Labels for the bins
                - drop_original: Whether to drop the original columns
            
        Returns:
            DataFrame with binned features
        """
        result_df = df.copy()
        
        # Filter columns to only those that exist in the DataFrame and are numeric
        existing_columns = [
            col for col in columns 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist or are not numeric: {missing_columns}")
        
        if not existing_columns:
            self.logger.warning("No valid columns to bin")
            return df
        
        # Get parameters
        n_bins = kwargs.get('n_bins', 5)
        strategy = kwargs.get('strategy', 'quantile')
        labels = kwargs.get('labels')
        drop_original = kwargs.get('drop_original', False)
        
        # Process each column
        for col in existing_columns:
            self.logger.info(f"Binning column {col} into {n_bins} bins using {strategy} strategy")
            
            try:
                # Create bin edges based on strategy
                if strategy == 'uniform':
                    bins = np.linspace(df[col].min(), df[col].max(), n_bins + 1)
                elif strategy == 'quantile':
                    bins = np.percentile(df[col].dropna(), np.linspace(0, 100, n_bins + 1))
                elif strategy == 'kmeans':
                    from sklearn.cluster import KMeans
                    
                    # Reshape the data for KMeans
                    X = df[col].dropna().values.reshape(-1, 1)
                    
                    # Apply KMeans
                    kmeans = KMeans(n_clusters=n_bins, random_state=config.RANDOM_SEED).fit(X)
                    centers = sorted(kmeans.cluster_centers_.flatten())
                    
                    # Use midpoints between centers as bin edges
                    bins = [df[col].min()] + [
                        (centers[i] + centers[i+1]) / 2 for i in range(len(centers) - 1)
                    ] + [df[col].max()]
                else:
                    self.logger.warning(f"Unsupported binning strategy: {strategy}")
                    continue
                
                # Apply binning
                result_df[f"{col}_binned"] = pd.cut(
                    result_df[col], 
                    bins=bins, 
                    labels=labels,
                    include_lowest=True
                )
                
                # Drop original column if specified
                if drop_original:
                    result_df = result_df.drop(columns=[col])
                    
            except Exception as e:
                self.logger.warning(f"Error binning column {col}: {str(e)}")
        
        return result_df
    
    def create_interaction_features(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Create interaction features between numerical columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names to create interactions for
            **kwargs: Additional parameters including:
                - operations: List of operations to apply ('multiply', 'divide', 'add', 'subtract')
            
        Returns:
            DataFrame with interaction features
        """
        result_df = df.copy()
        
        # Filter columns to only those that exist in the DataFrame and are numeric
        existing_columns = [
            col for col in columns 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist or are not numeric: {missing_columns}")
        
        if len(existing_columns) < 2:
            self.logger.warning("Need at least 2 columns to create interaction features")
            return df
        
        # Get operations
        operations = kwargs.get('operations', ['multiply'])
        
        # Define operation functions
        op_funcs = {
            'multiply': lambda x, y: x * y,
            'divide': lambda x, y: x / y,
            'add': lambda x, y: x + y,
            'subtract': lambda x, y: x - y
        }
        
        # Create interaction features
        for i, col1 in enumerate(existing_columns):
            for j in range(i + 1, len(existing_columns)):
                col2 = existing_columns[j]
                
                for op_name, op_func in op_funcs.items():
                    if op_name in operations:
                        # Skip division if second column has zeros
                        if op_name == 'divide' and (result_df[col2] == 0).any():
                            self.logger.warning(f"Skipping {col1}/{col2} due to zeros in denominator")
                            continue
                            
                        try:
                            new_col_name = f"{col1}_{op_name}_{col2}"
                            result_df[new_col_name] = op_func(result_df[col1], result_df[col2])
                            self.logger.info(f"Created interaction feature: {new_col_name}")
                            
                            # Handle infinity values if present (e.g., in division)
                            if not np.isfinite(result_df[new_col_name]).all():
                                self.logger.warning(f"Infinite values found in {new_col_name}, replacing with NaN")
                                result_df[new_col_name] = result_df[new_col_name].replace([np.inf, -np.inf], np.nan)
                                
                        except Exception as e:
                            self.logger.warning(f"Error creating interaction {col1} {op_name} {col2}: {str(e)}")
        
        return result_df
    
    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
        """
        Create polynomial features for numerical columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names to create polynomials for
            **kwargs: Additional parameters including:
                - degree: Polynomial degree (default: 2)
                - include_bias: Whether to include a bias column (default: False)
                - interaction_only: Whether to include only interaction features (default: False)
            
        Returns:
            DataFrame with polynomial features
        """
        result_df = df.copy()
        
        # Filter columns to only those that exist in the DataFrame and are numeric
        existing_columns = [
            col for col in columns 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if len(existing_columns) < len(columns):
            missing_columns = set(columns) - set(existing_columns)
            self.logger.warning(f"Some columns do not exist or are not numeric: {missing_columns}")
        
        if not existing_columns:
            self.logger.warning("No valid columns to create polynomial features for")
            return df
        
        # Get parameters
        degree = kwargs.get('degree', 2)
        include_bias = kwargs.get('include_bias', False)
        interaction_only = kwargs.get('interaction_only', False)
        
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            self.logger.info(
                f"Creating polynomial features of degree {degree} for columns: {existing_columns}"
                f" (interaction_only={interaction_only}, include_bias={include_bias})"
            )
            
            # Create polynomial features
            poly = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=include_bias
            )
            
            # Generate feature names
            poly.fit(df[existing_columns])
            feature_names = poly.get_feature_names_out(existing_columns)
            
            # Transform data
            poly_features = poly.transform(df[existing_columns])
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(
                poly_features,
                columns=feature_names,
                index=df.index
            )
            
            # Remove original features from polynomial result if they exist
            for col in existing_columns:
                if col in poly_df.columns:
                    poly_df = poly_df.drop(columns=[col])
            
            # Add polynomial features to result
            result_df = pd.concat([result_df, poly_df], axis=1)
            
            self.logger.info(f"Added {poly_df.shape[1]} polynomial features")
            
        except Exception as e:
            self.logger.warning(f"Error creating polynomial features: {str(e)}")
        
        return result_df