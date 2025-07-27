#!/usr/bin/env python3
"""
Script to extract data from JSON files in a repository folder
and save it as a CSV file for import into the benchmark.
"""

import pandas as pd
import json
import os
import sys
import argparse
from pathlib import Path

def is_valid_json_file(file_path):
    """
    Check if a file is a valid JSON file containing a dict.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        bool: True if valid JSON dict, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return isinstance(data, dict)
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
        return False

def find_json_files(repo_path):
    """
    Recursively find all JSON files in the repository folder.
    
    Args:
        repo_path (str): Path to the repository folder
        
    Returns:
        list: List of paths to valid JSON files
    """
    json_files = []
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        print(f"Error: Repository path '{repo_path}' does not exist.")
        return []
    
    if not repo_path.is_dir():
        print(f"Error: Repository path '{repo_path}' is not a directory.")
        return []
    
    print(f"Scanning repository: {repo_path}")
    
    for file_path in repo_path.rglob("*.json"):
        if is_valid_json_file(file_path):
            json_files.append(file_path)
            print(f"Found valid JSON file: {file_path}")
    
    print(f"Total valid JSON files found: {len(json_files)}")
    return json_files

def extract_data_from_json(json_file_path):
    """
    Extract data from a single JSON file.
    
    Args:
        json_file_path (Path): Path to the JSON file
        
    Returns:
        dict or None: Extracted data or None if extraction failed
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if required fields exist
        if 'config_general' not in data or 'results' not in data:
            return None
        
        config_general = data['config_general']
        results = data['results']
        
        # Extract model information
        model_name = config_general.get('model_name', '')
        model_private = config_general.get('model_private', False)
        
        # Extract results
        all_grouped = results.get('all_grouped', {})
        
        # Extract metrics
        assin2_rte = all_grouped.get('assin2_rte', 0.0)
        assin2_sts = all_grouped.get('assin2_sts', 0.0)
        faquad_nli = all_grouped.get('faquad_nli', 0.0)
        hatebr_offensive = all_grouped.get('hatebr_offensive', 0.0)
        
        # Create row data
        row_data = {
            'json_file': str(json_file_path),
            'model_name': model_name,
            'model_private': model_private,
            'assin2_rte': assin2_rte,
            'assin2_sts': assin2_sts,
            'faquad_nli': faquad_nli,
            'hatebr_offensive': hatebr_offensive
        }
        
        return row_data
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return None

def extract_portuguese_leaderboard(repo_path):
    """
    Extract data from JSON files in the repository folder and save as CSV.
    
    Args:
        repo_path (str): Path to the repository folder
    """
    
    print("Scanning repository for JSON files...")
    
    # Find all JSON files
    json_files = find_json_files(repo_path)
    
    if not json_files:
        print("No valid JSON files found in the repository.")
        return
    
    # Prepare data for DataFrame
    data = []
    
    # Process each JSON file
    for i, json_file in enumerate(json_files):
        print(f"Processing file {i+1}/{len(json_files)}: {json_file.name}")
        
        row_data = extract_data_from_json(json_file)
        if row_data:
            data.append(row_data)
        
        # Print progress every 10 files
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} files...")
    
    if not data:
        print("No valid data extracted from JSON files.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Write to CSV
    output_file = 'portuguese_leaderboard.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully extracted {len(df)} models to {output_file}")
    
    # Show first few entries as preview
    print("\nFirst 5 entries:")
    print(df.head().to_string(index=False))
    
    # Show some statistics
    if not df.empty:
        print(f"\nStatistics:")
        print(f"Total models: {len(df)}")
        print(f"Private models: {df['model_private'].sum()}")
        print(f"Public models: {(~df['model_private']).sum()}")
        
        # Average scores
        print(f"\nAverage scores:")
        print(df[['assin2_rte', 'assin2_sts', 'faquad_nli', 'hatebr_offensive']].mean().round(2))
        
        # Show data types and info
        print(f"\nDataFrame info:")
        print(df.info())

def main():
    """Main function to run the extraction."""
    parser = argparse.ArgumentParser(description='Extract Portuguese LLM Leaderboard data from JSON files')
    parser.add_argument('repo_path', help='Path to the repository folder containing JSON files')
    
    args = parser.parse_args()
    
    print("Portuguese LLM Leaderboard Data Extractor")
    print("=" * 50)
    
    try:
        extract_portuguese_leaderboard(args.repo_path)
        print("\nExtraction completed successfully!")
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 