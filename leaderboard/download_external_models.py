#!/usr/bin/env python3
"""
Script to download external models data from the Open Portuguese LLM Leaderboard
and convert it to CSV format for import into the benchmark.
"""

import requests
import pandas as pd
import json
import sys

def download_external_models():
    """Download external models data and convert to CSV."""
    
    url = "https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard/raw/main/external_models_results.json"
    
    print("Downloading external models data...")
    
    try:
        # Download the JSON file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse JSON
        data = response.json()
        
        if not isinstance(data, list):
            print("Error: Expected JSON array, got:", type(data))
            return
        
        print(f"Downloaded {len(data)} external models")
        
        # Extract data for each model
        extracted_data = []
        
        for item in data:
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dict item: {type(item)}")
                continue
            
            # Extract required fields
            model = item.get('model', '')
            link = item.get('link', '')
            result_metrics = item.get('result_metrics', {})
            
            if not isinstance(result_metrics, dict):
                print(f"Warning: Skipping model '{model}' - result_metrics is not a dict")
                continue
            
            # Extract metrics
            assin2_sts = result_metrics.get('assin2_sts', 0.0)
            assin2_rte = result_metrics.get('assin2_rte', 0.0)
            faquad_nli = result_metrics.get('faquad_nli', 0.0)
            hatebr_offensive = result_metrics.get('hatebr_offensive', 0.0)
            
            # Create row data
            row_data = {
                'model': model,
                'link': link,
                'assin2_sts': assin2_sts,
                'assin2_rte': assin2_rte,
                'faquad_nli': faquad_nli,
                'hatebr_offensive': hatebr_offensive
            }
            
            extracted_data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(extracted_data)
        
        # Save to CSV
        output_file = 'external_models.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully extracted {len(df)} models to {output_file}")
        
        # Show first few entries as preview
        print("\nFirst 5 entries:")
        print(df.head().to_string(index=False))
        
        # Show some statistics
        if not df.empty:
            print(f"\nStatistics:")
            print(f"Total models: {len(df)}")
            
            # Count models with non-zero scores for each metric
            print(f"\nModels with scores:")
            print(f"ASSIN2 STS: {(df['assin2_sts'] > 0).sum()}")
            print(f"ASSIN2 RTE: {(df['assin2_rte'] > 0).sum()}")
            print(f"FaQuAD-NLI: {(df['faquad_nli'] > 0).sum()}")
            print(f"HateBR: {(df['hatebr_offensive'] > 0).sum()}")
            
            # Average scores
            print(f"\nAverage scores:")
            print(df[['assin2_sts', 'assin2_rte', 'faquad_nli', 'hatebr_offensive']].mean().round(3))
            
            # Show data types and info
            print(f"\nDataFrame info:")
            print(df.info())
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

def main():
    """Main function to run the download."""
    print("External Models Data Downloader")
    print("=" * 40)
    
    try:
        download_external_models()
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"Error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 