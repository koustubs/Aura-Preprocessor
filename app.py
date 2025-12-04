"""
Flask Web Application for AutoDataPreprocessor.

This module provides a web interface for the AutoDataPreprocessor,
allowing users to upload datasets, view analysis results, and apply preprocessing actions.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import pandas as pd

# Import our existing modules
from src.data_analyzer import DataAnalyzer
from src.data_cleaner import DataCleaner
from src import config, utils

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production
app.config['UPLOAD_FOLDER'] = config.RAW_DATA_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Set up logging
logger = utils.setup_logging()

# Global variables to store current analysis
current_analysis = {}
current_dataset_path = ""

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['csv', 'xlsx', 'xls']

@app.route('/')
def index():
    """Home page with file upload."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    global current_dataset_path
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        current_dataset_path = filepath
        
        # Get custom instructions
        instructions = request.form.get('instructions', '').strip()
        
        flash(f'File {filename} uploaded successfully!')
        
        # Redirect with instructions as query parameter
        if instructions:
            return redirect(url_for('analyze_dataset', instructions=instructions))
        else:
            return redirect(url_for('analyze_dataset'))
    else:
        flash('Invalid file type. Please upload CSV, XLS, or XLSX files.')
        return redirect(url_for('index'))

@app.route('/analyze')
def analyze_dataset():
    """Analyze the uploaded dataset."""
    global current_analysis, current_dataset_path
    
    if not current_dataset_path or not os.path.exists(current_dataset_path):
        flash('No dataset uploaded. Please upload a file first.')
        return redirect(url_for('index'))
    
    try:
        # Get custom instructions from query parameter
        custom_instructions = request.args.get('instructions', '')
        
        # Create analyzer and run analysis
        analyzer = DataAnalyzer()
        analysis_result = analyzer.analyze_dataset(
            file_path=current_dataset_path,
            custom_instructions=custom_instructions if custom_instructions else None
        )
        
        current_analysis = analysis_result
        
        # Load the dataset for preview
        df = analyzer.load_dataset(current_dataset_path)
        
        # Get column-wise analysis and available actions
        from src.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        column_analysis = cleaner.get_column_analysis(df)
        
        dataset_preview = df.head(10).to_dict('records')
        dataset_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'filename': os.path.basename(current_dataset_path)
        }
        
        # Extract LLM recommendations for display
        llm_recommendations = {}
        if analysis_result and 'column_recommendations' in analysis_result:
            for col_name, col_info in analysis_result['column_recommendations'].items():
                if 'actions' in col_info and col_info['actions']:
                    llm_recommendations[col_name] = {
                        'issues': col_info.get('issues', []),
                        'primary_action': col_info['actions'][0] if col_info['actions'] else None,
                        'all_actions': col_info['actions'],
                        'reasoning': col_info.get('reasoning', '')
                    }
        
        return render_template('column_actions.html', 
                             analysis=analysis_result,
                             column_analysis=column_analysis,
                             llm_recommendations=llm_recommendations,
                             dataset_preview=dataset_preview,
                             dataset_info=dataset_info)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        flash(f'Error analyzing dataset: {str(e)}')
        return redirect(url_for('index'))

@app.route('/clean', methods=['POST'])
def clean_dataset():
    """Clean the dataset based on selected actions."""
    global current_analysis, current_dataset_path
    
    if not current_analysis or not current_dataset_path:
        flash('No analysis available. Please analyze a dataset first.')
        return redirect(url_for('index'))
    
    try:
        # Get selected actions from the form
        selected_actions = request.form.getlist('selected_actions')
        
        if not selected_actions:
            flash('No actions selected. Please select at least one preprocessing action.')
            return redirect(url_for('analyze_dataset'))
        
        # Load the dataset
        analyzer = DataAnalyzer()
        df = analyzer.load_dataset(current_dataset_path)
        
        # Create cleaner and process dataset
        cleaner = DataCleaner()
        cleaned_df = cleaner.process_dataset(
            df=df,
            llm_response=current_analysis,
            actions_to_apply=selected_actions
        )
        
        # Save cleaned dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(current_dataset_path)
        name, ext = os.path.splitext(filename)
        cleaned_filename = f"{name}_cleaned_{timestamp}{ext}"
        cleaned_path = os.path.join(config.CLEANED_DIR, cleaned_filename)
        
        if ext.lower() == '.csv':
            cleaned_df.to_csv(cleaned_path, index=False)
        else:
            cleaned_df.to_excel(cleaned_path, index=False)
        
        # Get cleaning summary
        cleaning_summary = {
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape,
            'actions_applied': len(selected_actions),
            'filename': cleaned_filename,
            'download_path': cleaned_path
        }
        
        # Preview cleaned data
        cleaned_preview = cleaned_df.head(10).to_dict('records')
        
        return render_template('results.html',
                             cleaning_summary=cleaning_summary,
                             cleaned_preview=cleaned_preview,
                             original_columns=list(df.columns),
                             cleaned_columns=list(cleaned_df.columns))
        
    except Exception as e:
        logger.error(f"Error during cleaning: {str(e)}", exc_info=True)
        flash(f'Error cleaning dataset: {str(e)}')
        return redirect(url_for('analyze_dataset'))

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download cleaned dataset."""
    try:
        file_path = os.path.join(config.CLEANED_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found.')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error downloading file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/dataset-info')
def api_dataset_info():
    """API endpoint to get current dataset information."""
    global current_dataset_path
    
    if not current_dataset_path or not os.path.exists(current_dataset_path):
        return jsonify({'error': 'No dataset loaded'})
    
    try:
        df = pd.read_csv(current_dataset_path)
        info = {
            'filename': os.path.basename(current_dataset_path),
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/custom-clean', methods=['POST'])
def custom_clean_dataset():
    """Clean the dataset using custom column-specific actions."""
    global current_dataset_path
    
    if not current_dataset_path:
        flash('No dataset available. Please upload a dataset first.')
        return redirect(url_for('index'))
    
    try:
        # Parse the custom actions from the form
        custom_actions = {}
        
        # Get all form data
        form_data = request.form.to_dict()
        
        # Parse column actions
        for key, value in form_data.items():
            if key.startswith('action_'):
                column_name = key.replace('action_', '')
                if value != 'skip':
                    custom_actions[column_name] = {value: {}}
                    
                    # Get parameters for the action if any
                    param_key = f'param_{column_name}_{value}'
                    if param_key in form_data and form_data[param_key]:
                        if value == 'impute_constant':
                            custom_actions[column_name][value]['value'] = form_data[param_key]
                        elif value in ['outlier_clip_iqr', 'outlier_clip_zscore', 'outlier_remove_iqr']:
                            custom_actions[column_name][value]['factor'] = float(form_data[param_key])
        
        if not custom_actions:
            flash('No preprocessing actions selected.')
            return redirect(url_for('analyze_dataset'))
        
        # Load and clean the dataset
        from src.data_analyzer import DataAnalyzer
        from src.data_cleaner import DataCleaner
        
        analyzer = DataAnalyzer()
        df = analyzer.load_dataset(current_dataset_path)
        
        cleaner = DataCleaner()
        cleaned_df = cleaner.process_dataset(df, custom_actions=custom_actions)
        
        # Save cleaned dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(current_dataset_path)
        base_name = os.path.splitext(filename)[0]
        cleaned_filename = f"cleaned_{base_name}_{timestamp}.csv"
        cleaned_path = os.path.join(config.CLEANED_DIR, cleaned_filename)
        
        cleaned_df.to_csv(cleaned_path, index=False)
        
        # Generate cleaning summary
        cleaning_summary = {
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape,
            'actions_applied': len(custom_actions),
            'filename': cleaned_filename,
            'download_path': cleaned_path
        }
        
        # Preview cleaned data
        cleaned_preview = cleaned_df.head(10).to_dict('records')
        
        flash(f'Dataset cleaned successfully! Saved as {cleaned_filename}')
        
        return render_template('results.html',
                             cleaning_summary=cleaning_summary,
                             cleaned_preview=cleaned_preview,
                             original_columns=list(df.columns),
                             cleaned_columns=list(cleaned_df.columns))
        
    except Exception as e:
        logger.error(f"Error during custom cleaning: {str(e)}", exc_info=True)
        flash(f'Error cleaning dataset: {str(e)}')
        return redirect(url_for('analyze_dataset'))

@app.route('/reports')
def view_reports():
    """View all analysis reports."""
    try:
        reports_dir = config.REPORTS_DIR
        report_files = [f for f in os.listdir(reports_dir) if f.endswith('.json')]
        
        reports = []
        for report_file in sorted(report_files, reverse=True):  # Most recent first
            file_path = os.path.join(reports_dir, report_file)
            with open(file_path, 'r') as f:
                report_data = json.load(f)
                reports.append({
                    'filename': report_file,
                    'created': datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S'),
                    'dataset_name': report_data.get('dataset_overview', {}).get('title', 'Unknown'),
                    'rows': report_data.get('dataset_overview', {}).get('n_rows', 0),
                    'columns': report_data.get('dataset_overview', {}).get('n_cols', 0)
                })
        
        return render_template('reports.html', reports=reports)
        
    except Exception as e:
        logger.error(f"Error loading reports: {str(e)}", exc_info=True)
        flash(f'Error loading reports: {str(e)}')
        return redirect(url_for('index'))

@app.route('/reports/<filename>')
def get_report(filename):
    """API endpoint to get a specific report."""
    try:
        file_path = os.path.join(config.REPORTS_DIR, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.CLEANED_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)