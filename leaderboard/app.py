import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Import data loader
from data_loader import data_loader, get_napolab_datasets, get_sample_benchmark_results, get_model_metadata

# Load data from YAML file
NAPOLAB_DATASETS = get_napolab_datasets()
SAMPLE_BENCHMARK_RESULTS = get_sample_benchmark_results()
MODEL_METADATA = get_model_metadata()

def load_portuguese_leaderboard_data() -> pd.DataFrame:
    """Load data from the Portuguese leaderboard CSV file."""
    try:
        csv_path = "portuguese_leaderboard.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Select only the relevant columns
            relevant_columns = ['model_name', 'model_num_parameters', 'assin2_rte', 'assin2_sts', 'faquad_nli', 'hatebr_offensive']
            df = df[relevant_columns].copy()
            
            # Rename columns to match the existing format
            df = df.rename(columns={
                'assin2_rte': 'ASSIN2 RTE',
                'assin2_sts': 'ASSIN2 STS', 
                'faquad_nli': 'FaQUaD-NLI',
                'hatebr_offensive': 'HateBR'
            })
            
            # Add source information
            df['source'] = 'portuguese_leaderboard'
            
            print(f"Loaded {len(df)} models from Portuguese leaderboard")
            return df
        else:
            print(f"Portuguese leaderboard CSV not found: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading Portuguese leaderboard data: {e}")
        return pd.DataFrame()

def load_external_models_data() -> pd.DataFrame:
    """Load data from the external models CSV file."""
    try:
        csv_path = "external_models.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Select only the relevant columns
            relevant_columns = ['model', 'link', 'assin2_rte', 'assin2_sts', 'faquad_nli', 'hatebr_offensive']
            df = df[relevant_columns].copy()
            
            # Rename columns to match the existing format
            df = df.rename(columns={
                'model': 'model_name',
                'assin2_rte': 'ASSIN2 RTE',
                'assin2_sts': 'ASSIN2 STS', 
                'faquad_nli': 'FaQUaD-NLI',
                'hatebr_offensive': 'HateBR'
            })
            
            # Add source information
            df['source'] = 'external_models'
            
            # Add model_num_parameters column with 0 for external models
            df['model_num_parameters'] = 0
            
            print(f"Loaded {len(df)} external models")
            return df
        else:
            print(f"External models CSV not found: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading external models data: {e}")
        return pd.DataFrame()

# Load Portuguese leaderboard data
PORTUGUESE_LEADERBOARD_DATA = load_portuguese_leaderboard_data()

# Load external models data
EXTERNAL_MODELS_DATA = load_external_models_data()

def create_simplified_benchmark_table(selected_datasets: List[str] = None, show_napolab_thesis: bool = True, show_teenytinyllama: bool = True, show_portuguese_leaderboard: bool = True, show_external_models: bool = True, hide_incomplete_models: bool = False, min_average_performance: float = 0.0, search_query: str = "", max_num_parameters: int = 0) -> pd.DataFrame:
    """Create a simplified benchmark table with one column per dataset."""
    # Get all dataset names
    dataset_names = sorted(NAPOLAB_DATASETS.keys())
    dataset_display_names = [NAPOLAB_DATASETS[name].get('name', name) for name in dataset_names]
    
    # Use selected datasets if provided, otherwise use all datasets
    if selected_datasets is None:
        selected_datasets = dataset_names
    
    # Collect data for each model
    model_data = {}
    
    # Process existing benchmark results
    for dataset_name, models in SAMPLE_BENCHMARK_RESULTS.items():
        for model_name, metrics in models.items():
            if model_name not in model_data:
                model_data[model_name] = {
                    'dataset_scores': {},
                    'url': None,
                    'source': 'existing'
                }
            
            # Calculate average performance for this dataset
            avg_performance = np.mean(list(metrics.values()))
            model_data[model_name]['dataset_scores'][dataset_name] = avg_performance
    
    # Process Portuguese leaderboard data
    if show_portuguese_leaderboard and not PORTUGUESE_LEADERBOARD_DATA.empty:
        for _, row in PORTUGUESE_LEADERBOARD_DATA.iterrows():
            model_name = row['model_name']
            
            if model_name not in model_data:
                model_data[model_name] = {
                    'dataset_scores': {},
                    'url': None,
                    'source': 'portuguese_leaderboard',
                    'num_parameters': row.get('model_num_parameters', 0)
                }
            
            # Map Portuguese leaderboard columns to dataset names
            column_mapping = {
                'ASSIN2 RTE': 'assin2_rte',
                'ASSIN2 STS': 'assin2_sts',
                'FaQUaD-NLI': 'faquad-nli',
                'HateBR': 'hatebr'
            }
            
            for display_name, dataset_name in column_mapping.items():
                if dataset_name in NAPOLAB_DATASETS:
                    score = row[display_name]
                    if pd.notna(score) and score > 0:
                        model_data[model_name]['dataset_scores'][dataset_name] = score
    
    # Process external models data
    if show_external_models and not EXTERNAL_MODELS_DATA.empty:
        for _, row in EXTERNAL_MODELS_DATA.iterrows():
            model_name = row['model_name']
            
            if model_name not in model_data:
                model_data[model_name] = {
                    'dataset_scores': {},
                    'url': row.get('link', ''),
                    'source': 'external_models',
                    'num_parameters': row.get('model_num_parameters', 0)
                }
            
            # Map external models columns to dataset names
            column_mapping = {
                'ASSIN2 RTE': 'assin2_rte',
                'ASSIN2 STS': 'assin2_sts',
                'FaQUaD-NLI': 'faquad-nli',
                'HateBR': 'hatebr'
            }
            
            for display_name, dataset_name in column_mapping.items():
                if dataset_name in NAPOLAB_DATASETS:
                    score = row[display_name]
                    if pd.notna(score) and score > 0:
                        model_data[model_name]['dataset_scores'][dataset_name] = score
    
    # Get model URLs and source information for existing models
    additional_models = data_loader.get_additional_models()
    for model_name in model_data.keys():
        if model_data[model_name]['source'] == 'existing':
            # Get URL
            for arch_models in additional_models.values():
                if model_name in arch_models:
                    model_data[model_name]['url'] = arch_models[model_name].get('huggingface_url', '')
                    break
            
            # Get source information
            model_metadata = MODEL_METADATA.get(model_name, {})
            source = model_metadata.get('source', 'unknown')
            model_data[model_name]['source'] = source
            
            # Add num_parameters for existing models (set to 0 as they don't have this info)
            model_data[model_name]['num_parameters'] = 0
    
    # Create table data
    table_data = []
    
    for model_name, data in model_data.items():
        # Apply source filtering
        source = data['source']
        
        # Apply show filters - only show models from sources that are checked
        if source == 'napolab_thesis' and not show_napolab_thesis:
            continue
        if source == 'teenytinyllama_paper' and not show_teenytinyllama:
            continue
        if source == 'portuguese_leaderboard' and not show_portuguese_leaderboard:
            continue
        if source == 'external_models' and not show_external_models:
            continue
        # Hide models with unknown source (should not happen with proper data)
        if source == 'unknown':
            continue
        
        # Apply parameter filtering (only for Portuguese leaderboard models)
        if max_num_parameters > 0 and source == 'portuguese_leaderboard':
            num_parameters = data.get('num_parameters', 0)
            if num_parameters > max_num_parameters:
                continue
        
        # Create clickable link for model name
        if data['url']:
            model_display = f"[{model_name}]({data['url']})"
        elif source == 'portuguese_leaderboard' and '/' in model_name:
            # Create Hugging Face link for Portuguese leaderboard models with slashes
            huggingface_url = f"https://huggingface.co/{model_name}"
            model_display = f"[{model_name}]({huggingface_url})"
        else:
            model_display = model_name
        
        # Create row with dataset scores
        row_data = {'Model': model_display}
        
        # Calculate average only over selected datasets
        selected_scores = []
        for dataset_name in selected_datasets:
            score = data['dataset_scores'].get(dataset_name, 0)
            if score > 0:  # Only include non-zero scores in average
                selected_scores.append(score)
        
        overall_avg = np.mean(selected_scores) if selected_scores else 0
        row_data['Average'] = round(overall_avg, 4)
        
        # Add scores for each dataset (only selected ones)
        for dataset_name in dataset_names:
            score = data['dataset_scores'].get(dataset_name, 0)
            display_name = dataset_display_names[dataset_names.index(dataset_name)]
            # Only add columns for selected datasets
            if dataset_name in selected_datasets:
                row_data[display_name] = round(score, 4)
        
        table_data.append(row_data)
    
    df = pd.DataFrame(table_data)
    
    # Filter to show only models that have scores for at least one selected dataset
    if selected_datasets and not df.empty:
        # Get display names for selected datasets
        selected_display_names = [NAPOLAB_DATASETS[name].get('name', name) for name in selected_datasets]
        
        # Filter models based on selection criteria
        models_to_keep = []
        for _, row in df.iterrows():
            has_score = False
            has_all_scores = True
            
            # Only check the datasets that are actually selected for display
            for dataset_name in selected_datasets:
                display_name = NAPOLAB_DATASETS[dataset_name].get('name', dataset_name)
                if display_name in df.columns:
                    score = row[display_name]
                    if score > 0:
                        has_score = True
                    else:
                        has_all_scores = False
            
            # Keep model if it has at least one score
            if has_score:
                # If hide_incomplete_models is True, only keep models with all scores in selected datasets
                if not hide_incomplete_models or has_all_scores:
                    models_to_keep.append(row['Model'])
        
        # Filter dataframe to only include selected models
        if models_to_keep:
            df = df[df['Model'].isin(models_to_keep)]
        else:
            # If no models to keep, create empty DataFrame with proper structure
            # Create columns list first
            columns = ['Model']
            for dataset_name in dataset_names:
                display_name = dataset_display_names[dataset_names.index(dataset_name)]
                if dataset_name in selected_datasets:
                    columns.append(display_name)
            columns.append('Average')
            
            # Create empty DataFrame with correct columns
            df = pd.DataFrame(columns=columns)
    
    # Filter by minimum average performance
    if min_average_performance > 0 and not df.empty:
        df = df[df['Average'] >= min_average_performance]
    
    # Filter by search query
    if search_query and not df.empty:
        # Extract model names from markdown links for searching
        df_filtered = df.copy()
        df_filtered['model_name_clean'] = df_filtered['Model'].str.replace(r'\[([^\]]+)\]\([^)]+\)', r'\1', regex=True)
        try:
            # Use regex pattern matching
            df_filtered = df_filtered[df_filtered['model_name_clean'].str.contains(search_query, case=False, na=False, regex=True)]
        except re.error:
            # Fallback to simple string matching if regex is invalid
            df_filtered = df_filtered[df_filtered['model_name_clean'].str.contains(search_query, case=False, na=False)]
        df = df_filtered.drop('model_name_clean', axis=1)
    
    # Sort by Average (descending)
    if not df.empty:
        df = df.sort_values('Average', ascending=False)
    
    # Add rank column with medal emojis for top 3 and color-coded emojis for others
    if not df.empty:
        df = df.reset_index(drop=True)
        df.index = df.index + 1  # Start ranking from 1
        
        # Create rank column with medal emojis and color-coded emojis
        rank_column = []
        total_models = len(df)
        
        for rank in df.index:
            if rank == 1:
                rank_column.append("🥇 1")
            elif rank == 2:
                rank_column.append("🥈 2")
            elif rank == 3:
                rank_column.append("🥉 3")
            else:
                # Color-code based on position relative to total
                position_ratio = rank / total_models
                if position_ratio <= 0.33:  # Top third
                    rank_column.append("🟢 " + str(rank))
                elif position_ratio <= 0.67:  # Middle third
                    rank_column.append("🟡 " + str(rank))
                else:  # Bottom third
                    rank_column.append("🔴 " + str(rank))
        
        df.insert(0, 'Rank', rank_column)
    
    return df


# Global variable to track the current CSV file
current_csv_file = None

def export_csv(df: pd.DataFrame):
    """Export the benchmark table to CSV."""
    global current_csv_file
    
    print(f"Export function called with dataframe shape: {df.shape}")
    
    if df.empty:
        print("Dataframe is empty, returning None")
        return None
    
    # Clean up previous file if it exists
    if current_csv_file:
        try:
            import os
            if os.path.exists(current_csv_file):
                os.remove(current_csv_file)
                print(f"Deleted previous CSV file: {current_csv_file}")
        except Exception as e:
            print(f"Error deleting previous file {current_csv_file}: {e}")
    
    # Clean the dataframe for CSV export
    df_clean = df.copy()
    
    # Remove markdown formatting from model names for cleaner CSV
    df_clean['Model'] = df_clean['Model'].str.replace(r'\[([^\]]+)\]\([^)]+\)', r'\1', regex=True)
    
    # Create filename with timestamp
    from datetime import datetime
    import tempfile
    import os
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"napolab_benchmark_results_{timestamp}.csv"
    
    # Create file in current directory (simpler approach)
    file_path = filename
    
    print(f"Creating CSV file at: {file_path}")
    
    # Save to CSV file
    df_clean.to_csv(file_path, index=False)
    
    print(f"CSV file created successfully. File exists: {os.path.exists(file_path)}")
    
    # Update current file tracking
    current_csv_file = file_path
    
    return file_path

def cleanup_current_csv():
    """Clean up the current CSV file after download."""
    global current_csv_file
    import os
    
    if current_csv_file and os.path.exists(current_csv_file):
        try:
            os.remove(current_csv_file)
            print(f"Deleted CSV file after download: {current_csv_file}")
            current_csv_file = None
        except Exception as e:
            print(f"Error deleting file {current_csv_file}: {e}")


def create_model_performance_radar(selected_datasets: List[str] = None, show_napolab_thesis: bool = True, show_teenytinyllama: bool = True, show_portuguese_leaderboard: bool = True, show_external_models: bool = True, hide_incomplete_models: bool = False, min_average_performance: float = 0.0, search_query: str = "", max_num_parameters: int = 0) -> go.Figure:
    """Create a radar chart showing model performance across all datasets."""
    # Use selected datasets if provided, otherwise use all datasets
    if selected_datasets is None:
        selected_datasets = list(NAPOLAB_DATASETS.keys())
    
    # Get dataset names for the radar axes (only selected ones)
    dataset_names = selected_datasets
    dataset_display_names = [NAPOLAB_DATASETS[name].get('name', name) for name in dataset_names]
    
    # Collect data for each model
    model_data = {}
    
    # Process existing benchmark results
    for dataset_name, models in SAMPLE_BENCHMARK_RESULTS.items():
        if dataset_name in selected_datasets:
            for model_name, metrics in models.items():
                if model_name not in model_data:
                    model_data[model_name] = {
                        'performances': {},
                        'architecture': MODEL_METADATA.get(model_name, {}).get('architecture', 'Unknown'),
                        'source': 'existing'
                    }
                
                # Calculate average performance for this dataset
                avg_performance = np.mean(list(metrics.values()))
                model_data[model_name]['performances'][dataset_name] = avg_performance
    
    # Process Portuguese leaderboard data
    if show_portuguese_leaderboard and not PORTUGUESE_LEADERBOARD_DATA.empty:
        for _, row in PORTUGUESE_LEADERBOARD_DATA.iterrows():
            model_name = row['model_name']
            
            if model_name not in model_data:
                model_data[model_name] = {
                    'performances': {},
                    'architecture': 'Unknown',
                    'source': 'portuguese_leaderboard',
                    'num_parameters': row.get('model_num_parameters', 0)
                }
            
            # Map Portuguese leaderboard columns to dataset names
            column_mapping = {
                'ASSIN2 RTE': 'assin2_rte',
                'ASSIN2 STS': 'assin2_sts',
                'FaQUaD-NLI': 'faquad-nli',
                'HateBR': 'hatebr'
            }
            
            for display_name, dataset_name in column_mapping.items():
                if dataset_name in selected_datasets:
                    score = row[display_name]
                    if pd.notna(score) and score > 0:
                        model_data[model_name]['performances'][dataset_name] = score
    
    # Process external models data
    if show_external_models and not EXTERNAL_MODELS_DATA.empty:
        for _, row in EXTERNAL_MODELS_DATA.iterrows():
            model_name = row['model_name']
            
            if model_name not in model_data:
                model_data[model_name] = {
                    'performances': {},
                    'architecture': 'Unknown',
                    'source': 'external_models',
                    'num_parameters': row.get('model_num_parameters', 0)
                }
            
            # Map external models columns to dataset names
            column_mapping = {
                'ASSIN2 RTE': 'assin2_rte',
                'ASSIN2 STS': 'assin2_sts',
                'FaQUaD-NLI': 'faquad-nli',
                'HateBR': 'hatebr'
            }
            
            for display_name, dataset_name in column_mapping.items():
                if dataset_name in selected_datasets:
                    score = row[display_name]
                    if pd.notna(score) and score > 0:
                        model_data[model_name]['performances'][dataset_name] = score
    
    # Get model URLs and source information for existing models
    additional_models = data_loader.get_additional_models()
    for model_name in model_data.keys():
        if model_data[model_name]['source'] == 'existing':
            # Get URL
            for arch_models in additional_models.values():
                if model_name in arch_models:
                    model_data[model_name]['url'] = arch_models[model_name].get('huggingface_url', '')
                    break
            
            # Get source information
            model_metadata = MODEL_METADATA.get(model_name, {})
            source = model_metadata.get('source', 'unknown')
            model_data[model_name]['source'] = source
            
            # Add num_parameters for existing models (set to 0 as they don't have this info)
            model_data[model_name]['num_parameters'] = 0
    
    # Apply source filtering
    filtered_model_data = {}
    for model_name, data in model_data.items():
        source = data.get('source', 'existing')
        
        # Apply show filters - only show models from sources that are checked
        if source == 'napolab_thesis' and not show_napolab_thesis:
            continue
        if source == 'teenytinyllama_paper' and not show_teenytinyllama:
            continue
        if source == 'portuguese_leaderboard' and not show_portuguese_leaderboard:
            continue
        if source == 'external_models' and not show_external_models:
            continue
        # Hide models with unknown source (should not happen with proper data)
        if source == 'unknown':
            continue
        
        # Apply parameter filtering (only for Portuguese leaderboard models)
        if max_num_parameters > 0 and source == 'portuguese_leaderboard':
            num_parameters = data.get('num_parameters', 0)
            if num_parameters > max_num_parameters:
                continue
        
        filtered_model_data[model_name] = data
    
    # Apply incomplete model filtering
    if hide_incomplete_models and selected_datasets:
        final_filtered_data = {}
        for model_name, data in filtered_model_data.items():
            has_all_scores = True
            for dataset_name in selected_datasets:
                if data['performances'].get(dataset_name, 0) == 0:
                    has_all_scores = False
                    break
            if has_all_scores:
                final_filtered_data[model_name] = data
        filtered_model_data = final_filtered_data
    
    # Apply minimum average performance filtering
    if min_average_performance > 0 and selected_datasets:
        final_filtered_data = {}
        for model_name, data in filtered_model_data.items():
            # Calculate average performance for selected datasets
            scores = []
            for dataset_name in selected_datasets:
                score = data['performances'].get(dataset_name, 0)
                if score > 0:  # Only include non-zero scores
                    scores.append(score)
            
            if scores:
                avg_performance = np.mean(scores)
                if avg_performance >= min_average_performance:
                    final_filtered_data[model_name] = data
        filtered_model_data = final_filtered_data
    
    # Apply search query filtering
    if search_query:
        final_filtered_data = {}
        try:
            # Use regex pattern matching
            import re
            pattern = re.compile(search_query, re.IGNORECASE)
            for model_name, data in filtered_model_data.items():
                if pattern.search(model_name):
                    final_filtered_data[model_name] = data
        except re.error:
            # Fallback to simple string matching if regex is invalid
            for model_name, data in filtered_model_data.items():
                if search_query.lower() in model_name.lower():
                    final_filtered_data[model_name] = data
        filtered_model_data = final_filtered_data
    
    # Sort models by average performance (descending)
    model_performances = []
    for model_name, data in filtered_model_data.items():
        # Calculate average performance for selected datasets
        scores = []
        for dataset_name in selected_datasets:
            score = data['performances'].get(dataset_name, 0)
            if score > 0:  # Only include non-zero scores
                scores.append(score)
        
        avg_performance = np.mean(scores) if scores else 0
        model_performances.append((model_name, data, avg_performance))
    
    # Sort by average performance (descending)
    model_performances.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate dynamic range based on actual data
    all_performance_values = []
    for model_name, data, avg_performance in model_performances:
        for dataset_name in dataset_names:
            score = data['performances'].get(dataset_name, 0)
            if score > 0:  # Only include non-zero scores
                all_performance_values.append(score)
    
    # Set dynamic range with some padding
    if all_performance_values:
        min_score = min(all_performance_values)
        max_score = max(all_performance_values)
        # Add 5% padding below minimum and ensure minimum is not below 0.5
        range_min = max(0.5, min_score - (max_score - min_score) * 0.05)
        range_max = 1.0
    else:
        # Fallback to default range if no data
        range_min = 0.6
        range_max = 1.0
    
    # Create radar chart
    fig = go.Figure()
    
    # Generate a more distinguishable color palette
    num_models = len(model_performances)
    
    # Create a list of line styles for better differentiation
    line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
    
    # Use highly contrasting colors for better differentiation
    base_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#ff9896',  # Light Red
        '#98df8a',  # Light Green
        '#ffbb78',  # Light Orange
        '#aec7e8',  # Light Blue
        '#c5b0d5',  # Light Purple
    ]
    
    # Ensure we have enough colors
    while len(base_colors) < num_models:
        base_colors.extend(base_colors)
    
    colors = base_colors[:num_models]
    
    for i, (model_name, data, avg_performance) in enumerate(model_performances):
        # Get performance values for all datasets (fill with 0 if missing)
        performance_values = []
        for dataset_name in dataset_names:
            performance_values.append(data['performances'].get(dataset_name, 0))
        
        # Close the polygon by adding the first value at the end
        if performance_values:
            performance_values.append(performance_values[0])
        
        # Assign color and line style based on model index for better differentiation
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        # Show first two models by default, hide the rest
        visible = True if i < 2 else 'legendonly'
        
        # Create theta values that close the polygon
        theta_values = dataset_display_names + [dataset_display_names[0]] if dataset_display_names else []
        
        fig.add_trace(go.Scatterpolar(
            r=performance_values,
            theta=theta_values,
            fill=None,
            name=model_name,
            line_color=color,
            line_dash=line_style,
            line_width=3,
            opacity=0.8,
            visible=visible,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "Dataset: %{theta}<br>" +
                "Performance: %{r:.3f}<br>" +
                "Architecture: " + data['architecture'] + "<br>" +
                "<extra></extra>"
            )
        ))
    
    # Update layout
    fig.update_layout(
        title="Model Performance Radar Chart",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[range_min, range_max],
                gridcolor='rgba(0, 0, 0, 0.2)',
                linecolor='rgba(0, 0, 0, 0.5)',
                tickcolor='rgba(0, 0, 0, 0.7)',
                tickfont=dict(color='rgba(0, 0, 0, 0.8)')
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=list(range(len(dataset_display_names))),
                ticktext=dataset_display_names,
                gridcolor='rgba(0, 0, 0, 0.2)',
                linecolor='rgba(0, 0, 0, 0.5)',
                tickcolor='rgba(0, 0, 0, 0.7)',
                tickfont=dict(color='rgba(0, 0, 0, 0.8)')
            ),
            bgcolor='rgba(255, 255, 255, 0)'
        ),
        height=700,
        showlegend=True,
        plot_bgcolor='rgba(255, 255, 255, 0)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        legend=dict(
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            orientation="h",
            font=dict(color='rgba(0, 0, 0, 0.8)')
        ),
        margin=dict(l=50, r=50, t=100, b=100),
        font=dict(color='rgba(0, 0, 0, 0.8)')
    )
    
    return fig

# Gradio Interface
with gr.Blocks(title="Napolab Leaderboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🌎 Napolab Leaderboard
    
    Stay up to date with the latest advancements in Portuguese language models and their performance across carefully curated Portuguese language tasks.
    
    [⭐ Star us on GitHub](https://github.com/ruanchaves/napolab)
    """)
    
    with gr.Tabs():
        
        # Benchmark Results Tab
        with gr.Tab("🏆 Benchmark Results"):
            gr.Markdown("### Model Performance Benchmarks")
            
            with gr.Accordion("Select Datasets to Include: (Click to expand)", open=False):
                with gr.Row():
                    # Create checkboxes for each dataset
                    dataset_checkboxes = []
                    for dataset_name in sorted(NAPOLAB_DATASETS.keys()):
                        display_name = NAPOLAB_DATASETS[dataset_name].get('name', dataset_name)
                        # Default to selected only for ASSIN 2 STS, FaQUaD-NLI, and HateBR
                        default_value = display_name in ['ASSIN 2 STS', 'FaQUaD-NLI', 'HateBR']
                        checkbox = gr.Checkbox(
                            label=display_name,
                            value=default_value
                        )
                        dataset_checkboxes.append((dataset_name, checkbox))
            
            with gr.Accordion("Filter by Score: (Click to expand)", open=False):
                with gr.Row():
                    hide_incomplete_models = gr.Checkbox(
                        label="Hide models with zero scores in selected datasets",
                        value=True
                    )
                    
                    min_average_performance = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=80,
                        step=1,
                        label="Minimum Average Performance (%)"
                    )
            
            with gr.Accordion("Filter by Data Source: (Click to expand)", open=False):
                with gr.Row():
                    show_napolab_thesis = gr.Checkbox(
                        label="Napolab Thesis models",
                        value=True
                    )
                    show_teenytinyllama = gr.Checkbox(
                        label="TeenyTinyLlama models",
                        value=True
                    )
                    show_portuguese_leaderboard = gr.Checkbox(
                        label="Open Portuguese LLM Leaderboard models (open-source)",
                        value=True
                    )
                    
                    show_external_models = gr.Checkbox(
                        label="Open Portuguese LLM Leaderboard models (proprietary)",
                        value=True
                    )
            
            # Calculate max parameters for slider
            max_params = 0
            if not PORTUGUESE_LEADERBOARD_DATA.empty:
                max_params = int(PORTUGUESE_LEADERBOARD_DATA['model_num_parameters'].max())
            
            with gr.Accordion("Filter by Model Size: (Click to expand)", open=False):
                with gr.Row():
                    max_num_parameters = gr.Slider(
                        minimum=0,
                        maximum=max_params,
                        value=0,
                        step=1,
                        label="Maximum Number of Parameters",
                        info="This slider is applicable only to Open PT LLM Leaderboard models. For other models, it will have no effect."
                    )
            
            # Search bar for filtering models
            search_query = gr.Textbox(
                label="Search models by name (supports regex)",
                placeholder="Enter model name or regex pattern to filter...",
                value="",
                info="Supports regular expressions. Examples: 'bert.*large', 'gemini|gpt', 'mdeberta.*', '^bert'"
            )
            
            benchmark_table = gr.DataFrame(
                label="Model Performance Benchmarks", 
                wrap=[True, False, False, False, False, False, False, False, False, False], 
                interactive=False,
                datatype=["str", "markdown", "number", "number", "number", "number", "number", "number", "number", "number"],
                column_widths=["80px", "200px", "100px", "120px", "120px", "120px", "120px", "120px", "120px", "120px"]
            )
            
            gr.Markdown("*🥇🥈🥉 = Top 3 | 🟢 = Top 33% | 🟡 = Middle 33% | 🔴 = Bottom 33%*")
            
            # Export to CSV button and file component
            export_button = gr.Button("📥 Export to CSV", variant="secondary")
            csv_file = gr.File(label="Download CSV", interactive=False, visible=True)
        
        # Model Analysis Tab
        with gr.Tab("📈 Model Analysis"):
            gr.Markdown("### Model Performance Radar Chart")
            
            # Dataset Selection Controls
            with gr.Accordion("Select Datasets to Display: (Click to expand)", open=False):
                with gr.Row():
                    # Create checkboxes for each dataset
                    analysis_dataset_checkboxes = []
                    for dataset_name in sorted(NAPOLAB_DATASETS.keys()):
                        display_name = NAPOLAB_DATASETS[dataset_name].get('name', dataset_name)
                        # Default to selected only for ASSIN 2 STS, FaQUaD-NLI, and HateBR
                        default_value = display_name in ['ASSIN 2 STS', 'FaQUaD-NLI', 'HateBR']
                        checkbox = gr.Checkbox(
                            label=display_name,
                            value=default_value
                        )
                        analysis_dataset_checkboxes.append((dataset_name, checkbox))
            
            # Filter Controls
            with gr.Accordion("Filter by Score: (Click to expand)", open=False):
                with gr.Row():
                    hide_incomplete_models_analysis = gr.Checkbox(
                        label="Hide models with zero scores in selected datasets",
                        value=True
                    )
                    
                    min_average_performance_analysis = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=80,
                        step=1,
                        label="Minimum Average Performance (%)"
                    )
            
            with gr.Accordion("Filter by Data Source: (Click to expand)", open=False):
                with gr.Row():
                    show_napolab_thesis_analysis = gr.Checkbox(
                        label="Napolab Thesis models",
                        value=True
                    )
                    
                    show_teenytinyllama_analysis = gr.Checkbox(
                        label="TeenyTinyLlama models",
                        value=True
                    )
                    
                    show_portuguese_leaderboard_analysis = gr.Checkbox(
                        label="Open Portuguese LLM Leaderboard models (open-source)",
                        value=True
                    )
                    
                    show_external_models_analysis = gr.Checkbox(
                        label="Open Portuguese LLM Leaderboard models (proprietary)",
                        value=True
                    )
            
            # Parameter slider for Model Analysis tab
            with gr.Accordion("Filter by Model Size: (Click to expand)", open=False):
                with gr.Row():
                    max_num_parameters_analysis = gr.Slider(
                        minimum=0,
                        maximum=max_params,
                        value=0,
                        step=1,
                        label="Maximum Number of Parameters",
                        info="This slider is applicable only to Open PT LLM Leaderboard models. For other models, it will have no effect."
                    )

            # Search bar for filtering models in radar chart
            search_query_analysis = gr.Textbox(
                label="Search models by name (supports regex)",
                placeholder="Enter model name or regex pattern to filter...",
                value="",
                info="Supports regular expressions. Examples: 'bert.*large', 'gemini|gpt', 'mdeberta.*', '^bert'"
            )
            
            model_analysis_chart = gr.Plot(label="Model Performance Radar Chart")
            
            # Add scatter plot below radar chart
            model_scatter_plot = gr.Plot(label="Model Performance vs Number of Parameters")
            
            gr.Markdown("""
            **How to interact with the chart:**
            - **Click on legend items** to show/hide specific models.
            - **Double-click on a legend item** to isolate that model (hide all others).
            - **Double-click again** to show all models.

            Models in the legend are sorted in descending order based on their average performance across your chosen datasets.
            """)
        

        
        # About Tab
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About Napolab
            
            **Natural Portuguese Language Benchmark (Napolab)** is a comprehensive collection of Portuguese datasets designed for evaluating Large Language Models.
            
            For more information, please visit the [GitHub repository](https://github.com/ruanchaves/napolab) and the [Hugging Face Dataset](https://huggingface.co/datasets/ruanchaves/napolab).
            
            ### Data Sources:
            The benchmark results and model evaluations presented in this leaderboard are compiled from multiple sources:
            
            **1. "Lessons learned from the evaluation of Portuguese language models"** by Ruan Chaves Rodrigues (2023). Available at: [University of Malta OAR@UM Repository](https://www.um.edu.mt/library/oar/handle/123456789/120557)

            **2. Open PT LLM Leaderboard** by Eduardo Garcia (2025). Available at: [Hugging Face Spaces](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard).
            
            **3. "TeenyTinyLlama: Open-source tiny language models trained in Brazilian Portuguese"** by Corrêa et al. (2024). Available at: [arXiv](https://arxiv.org/abs/2401.16640).
            
            ### Thesis Citation:
            ```bibtex
            @mastersthesis{chaves2023lessons,
            title={Lessons learned from the evaluation of Portuguese language models},
            author={Chaves Rodrigues, Ruan},
            year={2023},
            school={University of Malta},
            url={https://www.um.edu.mt/library/oar/handle/123456789/120557}
            }
            ```
            
            ### Napolab Citation:
            ```bibtex
            @software{Chaves_Rodrigues_napolab_2023,
            author = {Chaves Rodrigues, Ruan and Tanti, Marc and Agerri, Rodrigo},
            doi = {10.5281/zenodo.7781848},
            month = {3},
            title = {{Natural Portuguese Language Benchmark (Napolab)}},
            url = {https://github.com/ruanchaves/napolab},
            version = {1.0.0},
            year = {2023}
            }
            ```

            """)
    
    def create_model_performance_scatter(selected_datasets: List[str] = None, show_napolab_thesis: bool = True, show_teenytinyllama: bool = True, show_portuguese_leaderboard: bool = True, show_external_models: bool = True, hide_incomplete_models: bool = False, min_average_performance: float = 0.0, search_query: str = "", max_num_parameters: int = 0) -> go.Figure:
        """Create a scatter plot showing model performance vs number of parameters."""
        # Use selected datasets if provided, otherwise use all datasets
        if selected_datasets is None:
            selected_datasets = list(NAPOLAB_DATASETS.keys())
        
        # Collect data for each model
        model_data = {}
        
        # Process existing benchmark results
        for dataset_name, models in SAMPLE_BENCHMARK_RESULTS.items():
            if dataset_name in selected_datasets:
                for model_name, metrics in models.items():
                    if model_name not in model_data:
                        # Get actual source from MODEL_METADATA
                        model_metadata = MODEL_METADATA.get(model_name, {})
                        actual_source = model_metadata.get('source', 'unknown')
                        
                        model_data[model_name] = {
                            'performances': {},
                            'architecture': model_metadata.get('architecture', 'Unknown'),
                            'source': actual_source,
                            'num_parameters': 0
                        }
                    
                    # Calculate average performance for this dataset
                    avg_performance = np.mean(list(metrics.values()))
                    model_data[model_name]['performances'][dataset_name] = avg_performance
        
        # Process Portuguese leaderboard data
        if show_portuguese_leaderboard and not PORTUGUESE_LEADERBOARD_DATA.empty:
            for _, row in PORTUGUESE_LEADERBOARD_DATA.iterrows():
                model_name = row['model_name']
                
                if model_name not in model_data:
                    model_data[model_name] = {
                        'performances': {},
                        'architecture': 'Unknown',
                        'source': 'portuguese_leaderboard',
                        'num_parameters': row.get('model_num_parameters', 0)
                    }
                
                # Map Portuguese leaderboard columns to dataset names
                column_mapping = {
                    'ASSIN2 RTE': 'assin2_rte',
                    'ASSIN2 STS': 'assin2_sts',
                    'FaQUaD-NLI': 'faquad-nli',
                    'HateBR': 'hatebr'
                }
                
                for display_name, dataset_name in column_mapping.items():
                    if dataset_name in selected_datasets:
                        score = row[display_name]
                        if pd.notna(score) and score > 0:
                            model_data[model_name]['performances'][dataset_name] = score
        
        # Process external models data
        if show_external_models and not EXTERNAL_MODELS_DATA.empty:
            for _, row in EXTERNAL_MODELS_DATA.iterrows():
                model_name = row['model_name']
                
                if model_name not in model_data:
                    model_data[model_name] = {
                        'performances': {},
                        'architecture': 'Unknown',
                        'source': 'external_models',
                        'num_parameters': row.get('model_num_parameters', 0)
                    }
                
                # Map external models columns to dataset names
                column_mapping = {
                    'ASSIN2 RTE': 'assin2_rte',
                    'ASSIN2 STS': 'assin2_sts',
                    'FaQUaD-NLI': 'faquad-nli',
                    'HateBR': 'hatebr'
                }
                
                for display_name, dataset_name in column_mapping.items():
                    if dataset_name in selected_datasets:
                        score = row[display_name]
                        if pd.notna(score) and score > 0:
                            model_data[model_name]['performances'][dataset_name] = score
        
        # Apply source filtering
        filtered_model_data = {}
        for model_name, data in model_data.items():
            source = data.get('source', 'existing')
            
            # Apply show filters - only show models from sources that are checked
            if source == 'napolab_thesis' and not show_napolab_thesis:
                continue
            if source == 'teenytinyllama_paper' and not show_teenytinyllama:
                continue
            if source == 'portuguese_leaderboard' and not show_portuguese_leaderboard:
                continue
            if source == 'external_models' and not show_external_models:
                continue
            # Hide models with unknown source (should not happen with proper data)
            if source == 'unknown':
                continue
            
            # Apply parameter filtering (only for Portuguese leaderboard models)
            if max_num_parameters > 0 and source == 'portuguese_leaderboard':
                num_parameters = data.get('num_parameters', 0)
                if num_parameters > max_num_parameters:
                    continue
            
            filtered_model_data[model_name] = data
        
        # Apply incomplete model filtering
        if hide_incomplete_models and selected_datasets:
            final_filtered_data = {}
            for model_name, data in filtered_model_data.items():
                has_all_scores = True
                for dataset_name in selected_datasets:
                    if data['performances'].get(dataset_name, 0) == 0:
                        has_all_scores = False
                        break
                if has_all_scores:
                    final_filtered_data[model_name] = data
            filtered_model_data = final_filtered_data
        
        # Apply minimum average performance filtering
        if min_average_performance > 0 and selected_datasets:
            final_filtered_data = {}
            for model_name, data in filtered_model_data.items():
                # Calculate average performance for selected datasets
                scores = []
                for dataset_name in selected_datasets:
                    score = data['performances'].get(dataset_name, 0)
                    if score > 0:  # Only include non-zero scores
                        scores.append(score)
                
                if scores:
                    avg_performance = np.mean(scores)
                    if avg_performance >= min_average_performance:
                        final_filtered_data[model_name] = data
            filtered_model_data = final_filtered_data
        
        # Apply search query filtering
        if search_query:
            final_filtered_data = {}
            try:
                # Use regex pattern matching
                import re
                pattern = re.compile(search_query, re.IGNORECASE)
                for model_name, data in filtered_model_data.items():
                    if pattern.search(model_name):
                        final_filtered_data[model_name] = data
            except re.error:
                # Fallback to simple string matching if regex is invalid
                for model_name, data in filtered_model_data.items():
                    if search_query.lower() in model_name.lower():
                        final_filtered_data[model_name] = data
            filtered_model_data = final_filtered_data
        
        # Prepare data for scatter plot
        scatter_data = []
        for model_name, data in filtered_model_data.items():
            # Calculate average performance for selected datasets
            scores = []
            for dataset_name in selected_datasets:
                score = data['performances'].get(dataset_name, 0)
                if score > 0:  # Only include non-zero scores
                    scores.append(score)
            
            if scores:
                avg_performance = np.mean(scores)
                num_parameters = data.get('num_parameters', 0)
                source = data.get('source', 'unknown')
                
                scatter_data.append({
                    'model_name': model_name,
                    'avg_performance': avg_performance,
                    'num_parameters': num_parameters,
                    'source': source
                })
        
        if not scatter_data:
            # Create empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for the selected filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Model Performance vs Number of Parameters",
                xaxis_title="Number of Parameters",
                yaxis_title="Average Performance Score",
                height=500
            )
            return fig
        
        # Create scatter plot
        df_scatter = pd.DataFrame(scatter_data)
        
        # Create color mapping for sources
        color_map = {
            'portuguese_leaderboard': '#1f77b4',
            'external_models': '#ff7f0e',
            'napolab_thesis': '#2ca02c',
            'teenytinyllama_paper': '#d62728',
            'unknown': '#9467bd'
        }
        
        # Create display name mapping for sources
        display_name_map = {
            'portuguese_leaderboard': 'Open PT LLM Leaderboard',
            'external_models': 'Proprietary Models',
            'napolab_thesis': 'Napolab Thesis',
            'teenytinyllama_paper': 'TeenyTinyLlama Paper',
            'unknown': 'Unknown Source'
        }
        
        fig = go.Figure()
        
        for source in df_scatter['source'].unique():
            source_data = df_scatter[df_scatter['source'] == source]
            color = color_map.get(source, '#7f7f7f')
            display_name = display_name_map.get(source, source.replace('_', ' ').title())
            
            fig.add_trace(go.Scatter(
                x=source_data['num_parameters'],
                y=source_data['avg_performance'],
                mode='markers',
                name=display_name,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.7
                ),
                text=source_data['model_name'],
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Average Performance: %{y:.3f}<br>" +
                    "Number of Parameters: %{x:,}<br>" +
                    "Source: " + display_name + "<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title="Model Performance vs Number of Parameters",
            xaxis_title="Number of Parameters",
            yaxis_title="Average Performance Score",
            height=500,
            showlegend=True,
            plot_bgcolor='rgba(255, 255, 255, 0)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            legend=dict(
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                orientation="h"
            ),
            margin=dict(l=50, r=50, t=100, b=100)
        )
        
        return fig
    
    # Event handlers
    def update_radar_chart(*args):
        # Extract arguments for radar chart
        dataset_values = args[:len(analysis_dataset_checkboxes)]
        hide_incomplete_models = args[len(analysis_dataset_checkboxes)]
        min_average_performance = args[len(analysis_dataset_checkboxes) + 1] / 100.0  # Convert percentage to decimal
        show_napolab_thesis = args[len(analysis_dataset_checkboxes) + 2]
        show_teenytinyllama = args[len(analysis_dataset_checkboxes) + 3]
        show_portuguese_leaderboard = args[len(analysis_dataset_checkboxes) + 4]
        show_external_models = args[len(analysis_dataset_checkboxes) + 5]
        search_query = args[len(analysis_dataset_checkboxes) + 6]
        max_num_parameters = args[len(analysis_dataset_checkboxes) + 7]
        
        # Convert dataset selections to list of selected dataset names
        selected_datasets = []
        for i, (dataset_name, _) in enumerate(analysis_dataset_checkboxes):
            if dataset_values[i]:
                selected_datasets.append(dataset_name)
        
        return create_model_performance_radar(selected_datasets, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, hide_incomplete_models, min_average_performance, search_query, max_num_parameters)
    
    def update_benchmark_table(*args):
        # Extract arguments
        dataset_values = args[:len(dataset_checkboxes)]
        hide_incomplete_models = args[len(dataset_checkboxes)]
        min_average_performance = args[len(dataset_checkboxes) + 1] / 100.0  # Convert percentage to decimal
        show_napolab_thesis = args[len(dataset_checkboxes) + 2]
        show_teenytinyllama = args[len(dataset_checkboxes) + 3]
        show_portuguese_leaderboard = args[len(dataset_checkboxes) + 4]
        show_external_models = args[len(dataset_checkboxes) + 5]
        search_query = args[len(dataset_checkboxes) + 6]
        max_num_parameters = args[len(dataset_checkboxes) + 7]
        
        # Convert dataset selections to list of selected dataset names
        selected_datasets = []
        for i, (dataset_name, _) in enumerate(dataset_checkboxes):
            if dataset_values[i]:
                selected_datasets.append(dataset_name)
        
        df = create_simplified_benchmark_table(selected_datasets, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, hide_incomplete_models, min_average_performance, search_query, max_num_parameters)
        
        return df
    
    def update_scatter_plot(*args):
        # Extract arguments for scatter plot
        dataset_values = args[:len(analysis_dataset_checkboxes)]
        hide_incomplete_models = args[len(analysis_dataset_checkboxes)]
        min_average_performance = args[len(analysis_dataset_checkboxes) + 1] / 100.0  # Convert percentage to decimal
        show_napolab_thesis = args[len(analysis_dataset_checkboxes) + 2]
        show_teenytinyllama = args[len(analysis_dataset_checkboxes) + 3]
        show_portuguese_leaderboard = args[len(analysis_dataset_checkboxes) + 4]
        show_external_models = args[len(analysis_dataset_checkboxes) + 5]
        search_query = args[len(analysis_dataset_checkboxes) + 6]
        max_num_parameters = args[len(analysis_dataset_checkboxes) + 7]
        
        # Convert dataset selections to list of selected dataset names
        selected_datasets = []
        for i, (dataset_name, _) in enumerate(analysis_dataset_checkboxes):
            if dataset_values[i]:
                selected_datasets.append(dataset_name)
        
        return create_model_performance_scatter(selected_datasets, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, hide_incomplete_models, min_average_performance, search_query, max_num_parameters)
    
    # Connect dataset checkboxes to update table
    for dataset_name, checkbox in dataset_checkboxes:
        checkbox.change(
            update_benchmark_table,
            inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
            outputs=benchmark_table
        )
    
    hide_incomplete_models.change(
        update_benchmark_table,
        inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
        outputs=benchmark_table
    )
    
    min_average_performance.change(
        update_benchmark_table,
        inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
        outputs=benchmark_table
    )
    
    show_napolab_thesis.change(
        update_benchmark_table,
        inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
        outputs=benchmark_table
    )
    
    show_teenytinyllama.change(
        update_benchmark_table,
        inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
        outputs=benchmark_table
    )

    show_portuguese_leaderboard.change(
        update_benchmark_table,
        inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
        outputs=benchmark_table
    )
    
    show_external_models.change(
        update_benchmark_table,
        inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
        outputs=benchmark_table
    )
    
    # Connect search query to update table
    search_query.change(
        update_benchmark_table,
        inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
        outputs=benchmark_table
    )
    
    # Connect max_num_parameters to update table
    max_num_parameters.change(
        update_benchmark_table,
        inputs=[cb for _, cb in dataset_checkboxes] + [hide_incomplete_models, min_average_performance, show_napolab_thesis, show_teenytinyllama, show_portuguese_leaderboard, show_external_models, search_query, max_num_parameters],
        outputs=benchmark_table
    )
    
    # Connect export button
    export_button.click(
        export_csv,
        inputs=benchmark_table,
        outputs=csv_file
    )
    
    # Connect file download to cleanup
    csv_file.change(
        cleanup_current_csv,
        inputs=None,
        outputs=None
    )
    
    # Connect analysis chart events
    # Connect dataset checkboxes to update radar chart
    for dataset_name, checkbox in analysis_dataset_checkboxes:
        checkbox.change(
            update_radar_chart,
            inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
            outputs=model_analysis_chart
        )
    
    hide_incomplete_models_analysis.change(
        update_radar_chart,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_analysis_chart
    )
    
    min_average_performance_analysis.change(
        update_radar_chart,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_analysis_chart
    )
    
    show_napolab_thesis_analysis.change(
        update_radar_chart,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_analysis_chart
    )
    
    show_teenytinyllama_analysis.change(
        update_radar_chart,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_analysis_chart
    )

    show_portuguese_leaderboard_analysis.change(
        update_radar_chart,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_analysis_chart
    )
    
    show_external_models_analysis.change(
        update_radar_chart,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_analysis_chart
    )
    
    # Connect search query to update radar chart
    search_query_analysis.change(
        update_radar_chart,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_analysis_chart
    )
    
    # Connect max_num_parameters_analysis to update radar chart
    max_num_parameters_analysis.change(
        update_radar_chart,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_analysis_chart
    )
    
    # Connect all analysis controls to update scatter plot
    for dataset_name, checkbox in analysis_dataset_checkboxes:
        checkbox.change(
            update_scatter_plot,
            inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
            outputs=model_scatter_plot
        )
    
    hide_incomplete_models_analysis.change(
        update_scatter_plot,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_scatter_plot
    )
    
    min_average_performance_analysis.change(
        update_scatter_plot,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_scatter_plot
    )
    
    show_napolab_thesis_analysis.change(
        update_scatter_plot,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_scatter_plot
    )
    
    show_teenytinyllama_analysis.change(
        update_scatter_plot,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_scatter_plot
    )

    show_portuguese_leaderboard_analysis.change(
        update_scatter_plot,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_scatter_plot
    )
    
    show_external_models_analysis.change(
        update_scatter_plot,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_scatter_plot
    )
    
    search_query_analysis.change(
        update_scatter_plot,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_scatter_plot
    )
    
    max_num_parameters_analysis.change(
        update_scatter_plot,
        inputs=[cb for _, cb in analysis_dataset_checkboxes] + [hide_incomplete_models_analysis, min_average_performance_analysis, show_napolab_thesis_analysis, show_teenytinyllama_analysis, show_portuguese_leaderboard_analysis, show_external_models_analysis, search_query_analysis, max_num_parameters_analysis],
        outputs=model_scatter_plot
    )
    
    # Connect events
    # Load model analysis chart on app start
    app.load(lambda: update_radar_chart(*([display_name in ['ASSIN 2 STS', 'FaQUaD-NLI', 'HateBR'] for _, display_name in [(name, NAPOLAB_DATASETS[name].get('name', name)) for name in sorted(NAPOLAB_DATASETS.keys())]] + [True, 80, True, True, True, True, "", 0])), outputs=model_analysis_chart)
    
    # Load scatter plot on app start
    app.load(lambda: update_scatter_plot(*([display_name in ['ASSIN 2 STS', 'FaQUaD-NLI', 'HateBR'] for _, display_name in [(name, NAPOLAB_DATASETS[name].get('name', name)) for name in sorted(NAPOLAB_DATASETS.keys())]] + [True, 80, True, True, True, True, "", 0])), outputs=model_scatter_plot)
    
    # Load benchmark table on app start
    app.load(lambda: update_benchmark_table(*([display_name in ['ASSIN 2 STS', 'FaQUaD-NLI', 'HateBR'] for _, display_name in [(name, NAPOLAB_DATASETS[name].get('name', name)) for name in sorted(NAPOLAB_DATASETS.keys())]] + [True, 80, True, True, True, True, "", 0])), outputs=benchmark_table)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860) 