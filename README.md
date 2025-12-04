# AutoDataPreprocessor

An ML data preprocessing pipeline that uses LLMs (Groq API) to analyze raw datasets and provide intelligent preprocessing suggestions.

## Overview

AutoDataPreprocessor is a semi-automated data cleaning pipeline that:

1. Takes raw CSV/Excel datasets as input
2. Uses LLM (Groq API) to analyze and explain the dataset
3. Surfaces data quality issues and preprocessing suggestions
4. Allows human intervention to choose which actions to take
5. Outputs cleaned datasets ready for ML model training
6. Optionally suggests appropriate ML models for the cleaned data

## Architecture

```
Raw Dataset → LLM Analysis → Issue Detection → Human Choices → Data Cleaning → Clean Dataset + Model Suggestions
     ↓              ↓              ↓              ↓              ↓                    ↓
   CSV/Excel    Groq API    JSON Response   User Interface   Python Code        Final Output
```

## Project Structure

```
AutoDataPreprocessor/
├── data/
│   ├── raw/              # Original datasets
│   ├── samples/          # Small samples for LLM analysis
│   ├── processed/        # Intermediate data
│   └── cleaned/          # Final cleaned datasets
├── src/
│   ├── config.py         # API keys, settings
│   ├── data_analyzer.py  # Dataset analysis & LLM communication
│   ├── data_cleaner.py   # Cleaning functions
│   ├── llm_client.py     # Groq API wrapper
│   ├── main.py           # CLI interface
│   └── utils.py          # Helper functions
├── postman/              # API testing collections
├── outputs/reports/      # LLM analysis reports
├── .env                  # Environment variables
└── requirements.txt      # Project dependencies
```

## Requirements

- Python 3.8+
- Groq API key
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AutoDataPreprocessor.git
   cd AutoDataPreprocessor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.template .env
   ```
   
5. Edit `.env` to add your Groq API key

## Usage

### Command Line Interface

The package provides a command-line interface with several commands:

#### Analyze a dataset

```bash
python -m src.main analyze path/to/your/dataset.csv
```

#### Clean a dataset using an existing analysis

```bash
python -m src.main clean path/to/your/dataset.csv path/to/analysis.json --output_file=path/to/output.csv
```

#### Run the full workflow (analyze + clean)

```bash
python -m src.main full path/to/your/dataset.csv --interactive --output_file=path/to/output.csv
```

### Options

- `--custom_instructions`: Additional instructions for the LLM
- `--interactive`: Choose which actions to apply interactively
- `--actions`: Comma-separated list of action IDs to apply (e.g., action_0,action_2)
- `--output_file`: Path to save the cleaned dataset

## Example

```bash
# Analyze a dataset
python -m src.main analyze data/raw/customer_data.csv

# Clean with interactive selection of actions
python -m src.main full data/raw/customer_data.csv --interactive --output_file=data/cleaned/customer_data_clean.csv
```

## Groq API Integration

The system uses the Groq API to analyze datasets and generate preprocessing suggestions. The API client is configured to:

1. Send dataset samples and statistics
2. Request structured JSON responses
3. Cache results to avoid redundant API calls
4. Handle errors and retry logic

## Development

### Running Tests

```bash
pytest
```

### Adding New Cleaning Methods

To add a new cleaning method:

1. Add the method to `DataCleaner` class in `data_cleaner.py`
2. Update the `action_map` dictionary in the `__init__` method
3. Ensure the method follows the same signature as existing methods

## License

[MIT License](LICENSE)

## Contributors

- [Koustub S](https://github.com/koustubs)
- [Santosh Kumaar](https://github.com/Santosh7131)
- [Contributor 2](https://github.com/contributor2)
- [Contributor 3](https://github.com/contributor3)
