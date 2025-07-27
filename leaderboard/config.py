"""
Configuration file for the Napolab Leaderboard Gradio App
"""

# App Configuration
APP_TITLE = "Napolab Leaderboard"
APP_DESCRIPTION = "Natural Portuguese Language Benchmark Leaderboard"
APP_THEME = "soft"
APP_PORT = 7860
APP_HOST = "0.0.0.0"
APP_SHARE = True

# Dataset Configuration
DEFAULT_DATASET = "assin"
DEFAULT_SPLIT = "test"
DEFAULT_SAMPLES = 5
MAX_SAMPLES = 20

# Chart Configuration
CHART_HEIGHT = 400
OVERVIEW_CHART_HEIGHT = 600
CHART_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "warning": "#d62728"
}

# Model Configuration
DEFAULT_MODELS_TO_COMPARE = 2

# Cache Configuration
CACHE_DURATION = 3600  # 1 hour in seconds

# Error Messages
ERROR_MESSAGES = {
    "dataset_load": "Error loading dataset. Please check your internet connection.",
    "no_benchmark": "No benchmark data available for this dataset.",
    "no_models": "No models found for comparison.",
    "invalid_selection": "Invalid selection. Please try again."
}

# Links
LINKS = {
    "github": "https://github.com/ruanchaves/napolab",
    "huggingface_dataset": "https://huggingface.co/datasets/ruanchaves/napolab",
    "open_pt_llm_leaderboard": "https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard"
} 