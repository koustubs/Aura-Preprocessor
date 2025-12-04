"""
Main module for AutoDataPreprocessor.

This script provides a command line interface for the AutoDataPreprocessor,
allowing users to analyze datasets, generate preprocessing suggestions,
and apply those suggestions to clean the data.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, utils
from src.data_analyzer import DataAnalyzer
from src.data_cleaner import DataCleaner
from src.llm_client import GroqClient

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoDataPreprocessor - ML data preprocessing with LLM suggestions"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a dataset and get preprocessing suggestions"
    )
    analyze_parser.add_argument(
        "input_file", help="Path to the input dataset file (.csv, .xlsx, .xls)"
    )
    analyze_parser.add_argument(
        "--custom_instructions", help="Additional instructions for the LLM"
    )
    
    # Clean command
    clean_parser = subparsers.add_parser(
        "clean", help="Clean a dataset based on an analysis report"
    )
    clean_parser.add_argument(
        "input_file", help="Path to the input dataset file (.csv, .xlsx, .xls)"
    )
    clean_parser.add_argument(
        "analysis_file", help="Path to the analysis JSON file"
    )
    clean_parser.add_argument(
        "--actions", help="Comma-separated list of action IDs to apply (e.g., action_0,action_2)"
    )
    clean_parser.add_argument(
        "--output_file", help="Path to save the cleaned dataset"
    )
    
    # Full workflow command
    full_parser = subparsers.add_parser(
        "full", help="Run the full workflow (analyze and clean)"
    )
    full_parser.add_argument(
        "input_file", help="Path to the input dataset file (.csv, .xlsx, .xls)"
    )
    full_parser.add_argument(
        "--custom_instructions", help="Additional instructions for the LLM"
    )
    full_parser.add_argument(
        "--interactive", action="store_true", 
        help="Interactive mode to choose which actions to apply"
    )
    full_parser.add_argument(
        "--output_file", help="Path to save the cleaned dataset"
    )
    
    return parser.parse_args()

def setup_environment():
    """Set up the environment for the application."""
    # Create required directories
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.CLEANED_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    
    # Set up logging
    logger = utils.setup_logging()
    
    return logger

def analyze_dataset(input_file: str, custom_instructions: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a dataset and get preprocessing suggestions.
    
    Args:
        input_file: Path to the input dataset file
        custom_instructions: Additional instructions for the LLM
        
    Returns:
        Dictionary containing LLM analysis results
    """
    # Create a DataAnalyzer instance
    analyzer = DataAnalyzer()
    
    # Analyze the dataset
    analysis_result = analyzer.analyze_dataset(
        file_path=input_file,
        custom_instructions=custom_instructions
    )
    
    return analysis_result

def clean_dataset(
    input_file: str, 
    analysis_result: Dict[str, Any],
    actions_to_apply: Optional[List[str]] = None,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Clean a dataset based on an analysis result.
    
    Args:
        input_file: Path to the input dataset file
        analysis_result: LLM analysis results
        actions_to_apply: List of action IDs to apply
        output_file: Path to save the cleaned dataset
        
    Returns:
        Cleaned DataFrame
    """
    # Create a DataAnalyzer instance to load the dataset
    analyzer = DataAnalyzer()
    
    # Load the dataset
    df = analyzer.load_dataset(input_file)
    
    # Create a DataCleaner instance
    cleaner = DataCleaner()
    
    # Clean the dataset
    cleaned_df = cleaner.process_dataset(
        df=df,
        llm_response=analysis_result,
        actions_to_apply=actions_to_apply,
        save_path=output_file
    )
    
    return cleaned_df

def interactive_action_selection(analysis_result: Dict[str, Any]) -> List[str]:
    """
    Allow user to interactively select actions to apply.
    
    Args:
        analysis_result: LLM analysis results
        
    Returns:
        List of selected action IDs
    """
    if 'actions_plan' not in analysis_result:
        print("No actions plan found in analysis result")
        return []
    
    actions_plan = analysis_result['actions_plan']
    selected_actions = []
    
    print("\n===== SUGGESTED ACTIONS =====")
    for i, action in enumerate(actions_plan):
        action_id = f"action_{i}"
        action_type = action.get('action')
        columns = action.get('columns', [])
        rationale = action.get('rationale', 'No rationale provided')
        
        print(f"\n[{action_id}] {action_type.upper()} on {columns}")
        print(f"Rationale: {rationale}")
        
        choice = input("Apply this action? (y/n): ").strip().lower()
        if choice == 'y':
            selected_actions.append(action_id)
    
    print(f"\nSelected {len(selected_actions)} actions to apply")
    return selected_actions

def run_full_workflow(
    input_file: str, 
    custom_instructions: Optional[str] = None,
    interactive: bool = False,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the full workflow (analyze and clean).
    
    Args:
        input_file: Path to the input dataset file
        custom_instructions: Additional instructions for the LLM
        interactive: Whether to interactively select actions
        output_file: Path to save the cleaned dataset
        
    Returns:
        Cleaned DataFrame
    """
    # Analyze the dataset
    print("Analyzing dataset...")
    analysis_result = analyze_dataset(input_file, custom_instructions)
    
    # Save the analysis result
    timestamp = utils.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = os.path.join(
        config.REPORTS_DIR,
        f"analysis_{os.path.splitext(os.path.basename(input_file))[0]}_{timestamp}.json"
    )
    utils.save_json(analysis_result, analysis_file)
    print(f"Analysis results saved to: {analysis_file}")
    
    # Select actions
    actions_to_apply = None
    if interactive:
        actions_to_apply = interactive_action_selection(analysis_result)
    
    # Set default output file if not provided
    if not output_file:
        output_file = os.path.join(
            config.CLEANED_DIR,
            f"cleaned_{os.path.basename(input_file)}"
        )
    
    # Clean the dataset
    print("Cleaning dataset...")
    cleaned_df = clean_dataset(
        input_file=input_file,
        analysis_result=analysis_result,
        actions_to_apply=actions_to_apply,
        output_file=output_file
    )
    
    print(f"Cleaned dataset saved to: {output_file}")
    
    return cleaned_df

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment
    logger = setup_environment()
    logger.info("Starting AutoDataPreprocessor")
    
    try:
        if args.command == "analyze":
            # Analyze a dataset
            analysis_result = analyze_dataset(
                input_file=args.input_file,
                custom_instructions=args.custom_instructions
            )
            
            # Print summary
            if 'dataset_overview' in analysis_result:
                overview = analysis_result['dataset_overview']
                print("\n===== DATASET OVERVIEW =====")
                print(f"Title: {overview.get('title', 'N/A')}")
                print(f"Rows: {overview.get('n_rows', 'N/A')}")
                print(f"Columns: {overview.get('n_cols', 'N/A')}")
                print(f"Missing: {overview.get('missing_overall_pct', 'N/A')}%")
                
                if 'target_candidates' in overview:
                    print(f"Potential target columns: {overview['target_candidates']}")
            
            if 'quality_issues' in analysis_result:
                print("\n===== QUALITY ISSUES =====")
                for issue in analysis_result['quality_issues']:
                    print(f"- {issue.get('type', 'Unknown')} ({issue.get('severity', 'unknown')})")
                    print(f"  Columns: {issue.get('columns_involved', [])}")
                    print(f"  Fix: {issue.get('suggested_fix', 'N/A')}")
            
            if 'actions_plan' in analysis_result:
                print("\n===== SUGGESTED ACTIONS =====")
                for i, action in enumerate(analysis_result['actions_plan']):
                    print(f"[{i}] {action.get('action', 'Unknown')} on {action.get('columns', [])}")
                    print(f"    Rationale: {action.get('rationale', 'N/A')}")
                    
        elif args.command == "clean":
            # Load analysis result
            analysis_result = utils.load_json(args.analysis_file)
            
            # Parse actions to apply
            actions_to_apply = None
            if args.actions:
                actions_to_apply = args.actions.split(",")
            
            # Set default output file if not provided
            output_file = args.output_file
            if not output_file:
                output_file = os.path.join(
                    config.CLEANED_DIR,
                    f"cleaned_{os.path.basename(args.input_file)}"
                )
            
            # Clean the dataset
            cleaned_df = clean_dataset(
                input_file=args.input_file,
                analysis_result=analysis_result,
                actions_to_apply=actions_to_apply,
                output_file=output_file
            )
            
            print(f"Cleaned dataset saved to: {output_file}")
            print(f"Shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
            
        elif args.command == "full":
            # Run the full workflow
            cleaned_df = run_full_workflow(
                input_file=args.input_file,
                custom_instructions=args.custom_instructions,
                interactive=args.interactive,
                output_file=args.output_file
            )
            
            print(f"Final dataset shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
            
        else:
            print("No command specified. Use --help for usage information.")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)
    
    logger.info("AutoDataPreprocessor completed successfully")

if __name__ == "__main__":
    main()