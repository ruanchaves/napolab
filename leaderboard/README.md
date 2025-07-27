# Napolab Leaderboard - Gradio App

A comprehensive Gradio web application for exploring and benchmarking Portuguese language models using the Napolab dataset collection.

## Features

- **🏆 Benchmark Results**: Single comprehensive table with one column per dataset and clickable model links
- **📈 Model Analysis**: Radar chart showing model performance across all datasets

## Installation

1. Navigate to the leaderboard directory:
```bash
cd dev/napolab/leaderboard
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Extract data from external sources (optional but recommended):
```bash
# Extract data from Portuguese LLM Leaderboard
python extract_portuguese_leaderboard.py

# Download external models data
python download_external_models.py
```

4. Run the Gradio app:
```bash
python app.py
```

The app will be available at `http://localhost:7860`

## Data Management

The app uses a YAML configuration file (`data.yaml`) for adding new data, making it easy to edit and maintain.

### Data Extraction Scripts

The leaderboard includes scripts to automatically extract and update data from external sources:

#### `extract_portuguese_leaderboard.py`
This script extracts benchmark results from the Open Portuguese LLM Leaderboard:
- Fetches data from the Hugging Face Spaces leaderboard
- Updates the `portuguese_leaderboard.csv` file
- Includes both open-source and proprietary models
- Automatically handles data formatting and validation

#### `download_external_models.py`
This script downloads additional model data:
- Fetches model metadata from various sources
- Updates the `external_models.csv` file
- Includes model links and performance metrics
- Ensures data consistency with the main leaderboard

**Note**: These scripts require internet connection and may take a few minutes to complete. Run them periodically to keep the leaderboard data up to date.