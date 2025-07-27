---
title: Napolab Leaderboard
emoji: 🌎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: true
python_version: "3.10"
tags:
  - nlp
  - portuguese
  - benchmarking
  - language-models
  - gradio
datasets:
  - ruanchaves/napolab
  - assin
  - assin2
  - ruanchaves/hatebr
  - ruanchaves/faquad-nli
short_description: "The Natural Portuguese Language Benchmark"
---

# Napolab Leaderboard - Gradio App

A comprehensive Gradio web application for exploring and benchmarking Portuguese language models using the Napolab dataset collection.

## Features

- **🏆 Benchmark Results**: Single comprehensive table with one column per dataset and clickable model links
- **📈 Model Analysis**: Radar chart showing model performance across all datasets
- **ℹ️ About**: Information about Napolab and citation details

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

## Usage

### Benchmark Results Tab
- **Single Comprehensive Table**: Shows all models with one column per dataset
- **Dataset Columns**: Each dataset has its own column showing model performance scores
- **Average Column**: Shows the average performance across all datasets for each model
- **Model Column**: Clickable links to Hugging Face model pages
- **Sorted Results**: Models are sorted by overall average performance (descending)

### Model Analysis Tab
- Radar chart showing each model's performance across all datasets
- **Default view**: Shows only bertimbau-large and mdeberta-v3-base models
- **Interactive legend**: Click to show/hide models, double-click to isolate
- Each line represents one model, each point represents one dataset
- Color-coded by model architecture
- Interactive hover information with detailed performance metrics

### Model Hub Tab
- Access links to pre-trained models on Hugging Face
- Models are organized by dataset and architecture type
- Direct links to model repositories

## Supported Datasets

The app includes all Napolab datasets:

- **ASSIN**: Semantic Similarity and Textual Entailment
- **ASSIN 2**: Semantic Similarity and Textual Entailment (v2)
- **Rerelem**: Relational Reasoning
- **HateBR**: Hate Speech Detection
- **Reli-SA**: Religious Sentiment Analysis
- **FaQUaD-NLI**: Factual Question Answering and NLI
- **PorSimplesSent**: Simple Sentences Sentiment Analysis

## Model Architectures

The benchmark includes models based on:
- **mDeBERTa v3**: Multilingual DeBERTa v3
- **BERT Large**: Large Portuguese BERT
- **BERT Base**: Base Portuguese BERT

## Data Management

The app now uses a YAML configuration file (`data.yaml`) for all data, making it easy to edit and maintain.

### Editing Data

Simply edit the `data.yaml` file to:
- Add new datasets
- Update benchmark results
- Add new models
- Modify model metadata

### Data Structure

The YAML file contains four main sections:

1. **datasets**: Information about each dataset
2. **benchmark_results**: Performance metrics for models on datasets
3. **model_metadata**: Model information (parameters, architecture, etc.)
4. **additional_models**: Additional models for the Model Hub

### Data Management Tools

Use the `manage_data.py` script for data operations:

```bash
# Validate the data structure
python manage_data.py validate

# Add a new dataset
python manage_data.py add-dataset \
  --dataset-name "new_dataset" \
  --dataset-display-name "New Dataset" \
  --dataset-description "Description of the dataset" \
  --dataset-tasks "Classification" "Sentiment Analysis" \
  --dataset-url "https://huggingface.co/datasets/new_dataset"

# Add benchmark results
python manage_data.py add-benchmark \
  --dataset-name "assin" \
  --model-name "new-model" \
  --metrics "accuracy=0.92" "f1=0.91"

# Add model metadata
python manage_data.py add-model \
  --model-name "new-model" \
  --parameters 110000000 \
  --architecture "BERT Base" \
  --base-model "bert-base-uncased" \
  --task "Classification" \
  --huggingface-url "https://huggingface.co/new-model"
```

### Customization

To add new datasets or benchmark results:

1. Edit the `data.yaml` file directly, or
2. Use the `manage_data.py` script for structured additions
3. The app will automatically reload the data when restarted

## Troubleshooting

- **Dataset loading errors**: Ensure you have internet connection to access Hugging Face datasets
- **Memory issues**: Reduce the number of samples in the Dataset Explorer
- **Port conflicts**: Change the port in the `app.launch()` call

## Contributing

Feel free to contribute by:
- Adding new datasets
- Improving visualizations
- Adding new features
- Reporting bugs

## License

This project follows the same license as the main Napolab repository.