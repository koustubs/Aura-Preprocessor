"""
Data Analyzer module for dataset analysis and LLM communication.

This module provides functionality to analyze datasets, extract statistical 
information, create samples for LLM analysis, and manage the communication
with the LLM API.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Tuple, Optional
from pathlib import Path
from datetime import datetime

from . import config
from . import utils
from .llm_client import GroqClient

class DataAnalyzer:
    """
    Analyze datasets and interact with the LLM for preprocessing insights.
    
    This class handles dataset loading, statistical analysis, sample generation,
    and LLM communication to get preprocessing recommendations.
    """
    
    def __init__(self, llm_client: Optional[GroqClient] = None):
        """
        Initialize the DataAnalyzer.
        
        Args:
            llm_client: A GroqClient instance (created if not provided)
        """
        self.logger = logging.getLogger('auto_preprocessor.data_analyzer')
        self.llm_client = llm_client or GroqClient()
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load a dataset from a file.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Loaded dataset as a pandas DataFrame
            
        Raises:
            ValueError: If file format is not supported or file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
            
        if not utils.is_supported_file(file_path):
            raise ValueError(f"Unsupported file format: {utils.get_file_extension(file_path)}")
        
        self.logger.info(f"Loading dataset from {file_path}")
        
        file_ext = utils.get_file_extension(file_path)
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        # Validate the dataframe
        is_valid, issues = utils.validate_dataframe(df)
        if not is_valid:
            for issue in issues:
                self.logger.warning(issue)
        
        self.logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def create_dataset_sample(
        self, 
        df: pd.DataFrame, 
        max_rows: int = config.MAX_SAMPLE_ROWS,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a representative sample of the dataset for LLM analysis.
        
        Args:
            df: Input DataFrame
            max_rows: Maximum number of rows in the sample
            save_path: Optional path to save the sample
            
        Returns:
            DataFrame containing the sample data
        """
        if df.shape[0] <= max_rows:
            # If dataset is already small, use the entire dataset
            sample = df.copy()
            self.logger.info(f"Using entire dataset as sample ({sample.shape[0]} rows)")
        else:
            # For larger datasets, use stratified sampling if categorical columns exist
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_cols) > 0 and len(categorical_cols) <= 3:
                # Use stratified sampling based on categorical columns (if we don't have too many)
                self.logger.info("Using stratified sampling based on categorical columns")
                
                # Choose a categorical column with moderate cardinality for stratification
                strat_col = None
                for col in categorical_cols:
                    n_unique = df[col].nunique()
                    if 2 <= n_unique <= 50:
                        strat_col = col
                        break
                
                if strat_col:
                    # Drop NA values in stratification column for sampling
                    df_for_sampling = df.dropna(subset=[strat_col])
                    if df_for_sampling.shape[0] > 0:
                        # Compute fraction to maintain max_rows constraint
                        frac = min(max_rows / df_for_sampling.shape[0], 1.0)
                        try:
                            sample = df_for_sampling.groupby(strat_col, group_keys=False).apply(
                                lambda x: x.sample(frac=frac, random_state=config.RANDOM_SEED)
                            )
                            
                            # If sample is too large, take a random subsample
                            if sample.shape[0] > max_rows:
                                sample = sample.sample(max_rows, random_state=config.RANDOM_SEED)
                                
                        except Exception as e:
                            self.logger.warning(f"Stratified sampling failed: {str(e)}. Falling back to random sampling.")
                            sample = df.sample(min(max_rows, df.shape[0]), random_state=config.RANDOM_SEED)
                    else:
                        self.logger.warning(f"No non-NA values in stratification column {strat_col}. Using random sampling.")
                        sample = df.sample(min(max_rows, df.shape[0]), random_state=config.RANDOM_SEED)
                else:
                    self.logger.info("No suitable categorical column for stratification. Using random sampling.")
                    sample = df.sample(min(max_rows, df.shape[0]), random_state=config.RANDOM_SEED)
            else:
                # Use random sampling
                self.logger.info("Using random sampling")
                sample = df.sample(min(max_rows, df.shape[0]), random_state=config.RANDOM_SEED)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file_ext = utils.get_file_extension(save_path)
            if file_ext == '.csv':
                sample.to_csv(save_path, index=False)
            elif file_ext in ['.xlsx', '.xls']:
                sample.to_excel(save_path, index=False)
            self.logger.info(f"Saved dataset sample to {save_path}")
        
        return sample
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing dataset statistics
        """
        self.logger.info("Computing dataset statistics")
        
        # General dataset information
        dataset_info = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        }
        
        # Missing value information
        missing_counts = df.isna().sum()
        missing_percentages = (missing_counts / df.shape[0]) * 100
        
        # Column-specific statistics
        column_stats = {}
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "missing_count": int(missing_counts[col]),
                "missing_percentage": float(missing_percentages[col]),
                "unique_count": int(df[col].nunique()),
            }
            
            # Type-specific statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns
                col_stats.update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "median": float(df[col].median()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "skewness": float(df[col].skew()) if not df[col].isna().all() else None,
                    "kurtosis": float(df[col].kurtosis()) if not df[col].isna().all() else None,
                    "is_integer": all(df[col].dropna().apply(lambda x: float(x).is_integer())),
                })
                
                # Check for potential outliers using IQR method
                if not df[col].isna().all():
                    Q1 = float(df[col].quantile(0.25))
                    Q3 = float(df[col].quantile(0.75))
                    IQR = Q3 - Q1
                    outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    col_stats["potential_outliers"] = int(outlier_count)
                    col_stats["potential_outliers_pct"] = float((outlier_count / df[col].count()) * 100)
                
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                # For categorical/object columns
                value_counts = df[col].value_counts(normalize=True)
                top_values = value_counts.head(5).to_dict()
                formatted_top_values = {str(k): float(v) for k, v in top_values.items()}
                
                col_stats.update({
                    "top_values": formatted_top_values,
                    "is_binary": df[col].nunique() == 2,
                    "is_high_cardinality": df[col].nunique() > 100,
                })
                
                # Check if it might be a datetime column
                if df[col].nunique() > 0:
                    sample_val = df[col].dropna().iloc[0] if not df[col].isna().all() else None
                    if sample_val and isinstance(sample_val, str):
                        try:
                            pd.to_datetime(df[col].dropna().iloc[0])
                            col_stats["potential_datetime"] = True
                        except (ValueError, TypeError):
                            col_stats["potential_datetime"] = False
            
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # For datetime columns
                if not df[col].isna().all():
                    col_stats.update({
                        "min": str(df[col].min()),
                        "max": str(df[col].max()),
                        "time_span_days": (df[col].max() - df[col].min()).days,
                    })
            
            column_stats[col] = col_stats
        
        # Correlation matrix for numerical columns
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        correlation_matrix = None
        if len(numerical_cols) > 1:
            # Compute correlation matrix, handle NaN values
            corr_matrix = df[numerical_cols].corr().fillna(0)
            # Convert to dict with only significant correlations (abs > 0.7)
            high_corr = []
            for i in range(len(numerical_cols)):
                for j in range(i+1, len(numerical_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            "col1": numerical_cols[i],
                            "col2": numerical_cols[j],
                            "correlation": float(corr_matrix.iloc[i, j])
                        })
                        
            if high_corr:
                correlation_matrix = high_corr
        
        # Combine all statistics
        statistics = {
            "dataset_info": dataset_info,
            "missing_summary": {
                "total_missing": int(missing_counts.sum()),
                "total_missing_percentage": float((missing_counts.sum() / (df.shape[0] * df.shape[1])) * 100),
                "columns_with_missing": int((missing_counts > 0).sum()),
                "columns_all_missing": int((missing_counts == df.shape[0]).sum()),
            },
            "column_stats": column_stats,
        }
        
        if correlation_matrix:
            statistics["high_correlations"] = correlation_matrix
            
        return statistics
    
    def analyze_dataset(
        self, 
        file_path: str,
        custom_instructions: Optional[str] = None,
        system_prompt: Optional[str] = None,
        save_intermediate: bool = config.SAVE_INTERMEDIATE_RESULTS
    ) -> Dict[str, Any]:
        """
        Complete workflow to analyze a dataset using LLM.
        
        Args:
            file_path: Path to the dataset file
            custom_instructions: Additional instructions for the LLM
            system_prompt: Custom system prompt for the LLM
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary containing LLM analysis results
        """
        # Load dataset
        df = self.load_dataset(file_path)
        
        # Create sample
        file_name = os.path.basename(file_path)
        sample_path = None
        if save_intermediate:
            sample_path = os.path.join(config.SAMPLES_DIR, f"sample_{file_name}")
            os.makedirs(config.SAMPLES_DIR, exist_ok=True)
            
        sample_df = self.create_dataset_sample(df, save_path=sample_path)
        
        # Compute statistics
        statistics = self.compute_statistics(df)
        
        # Save statistics
        if save_intermediate:
            stats_file = os.path.join(
                config.PROCESSED_DIR,
                f"stats_{os.path.splitext(file_name)[0]}.json"
            )
            os.makedirs(config.PROCESSED_DIR, exist_ok=True)
            utils.save_json(statistics, stats_file)
            self.logger.info(f"Saved statistics to {stats_file}")
        
        # Create dataset description
        dataset_description = f"Dataset filename: {file_name}\n"
        dataset_description += f"Full dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        dataset_description += f"Sample shape: {sample_df.shape[0]} rows × {sample_df.shape[1]} columns"
        
        # Convert sample to string representation
        data_sample = sample_df.to_string(index=False, max_rows=config.MAX_SAMPLE_ROWS, max_cols=config.MAX_COLUMNS_PER_REQUEST)
        
        # Get LLM analysis
        self.logger.info("Requesting LLM analysis of dataset")
        llm_response = self.llm_client.analyze_dataset(
            dataset_description=dataset_description,
            data_sample=data_sample,
            statistics=statistics,
            custom_instructions=custom_instructions,
            system_prompt=system_prompt
        )
        
        # Save LLM response
        if save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(
                config.REPORTS_DIR,
                f"analysis_{os.path.splitext(file_name)[0]}_{timestamp}.json"
            )
            os.makedirs(config.REPORTS_DIR, exist_ok=True)
            utils.save_json(llm_response, report_file)
            self.logger.info(f"Saved LLM analysis report to {report_file}")
        
        return llm_response