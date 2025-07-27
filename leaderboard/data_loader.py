"""
Data loader for Napolab Leaderboard
Loads datasets, benchmark results, and model metadata from YAML configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class NapolabDataLoader:
    """Loads and manages Napolab data from YAML configuration files."""
    
    def __init__(self, data_file: str = "data.yaml"):
        """
        Initialize the data loader.
        
        Args:
            data_file: Path to the YAML data file
        """
        self.data_file = data_file
        self.data = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load data from the YAML file."""
        try:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            data_path = script_dir / self.data_file
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as file:
                self.data = yaml.safe_load(file)
                
        except Exception as e:
            print(f"Error loading data from {self.data_file}: {e}")
            # Fallback to empty data structure
            self.data = {
                'datasets': {},
                'benchmark_results': {},
                'model_metadata': {},
                'additional_models': {}
            }
    
    def get_datasets(self) -> Dict[str, Any]:
        """Get all datasets information."""
        return self.data.get('datasets', {})
    
    def get_benchmark_results(self) -> Dict[str, Any]:
        """Get all benchmark results."""
        return self.data.get('benchmark_results', {})
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get all model metadata."""
        return self.data.get('model_metadata', {})
    
    def get_additional_models(self) -> Dict[str, Any]:
        """Get additional models for the Model Hub."""
        return self.data.get('additional_models', {})
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information for a specific dataset."""
        return self.data.get('datasets', {}).get(dataset_name)
    
    def get_benchmark_for_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get benchmark results for a specific dataset."""
        return self.data.get('benchmark_results', {}).get(dataset_name)
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model."""
        return self.data.get('model_metadata', {}).get(model_name)
    
    def get_available_datasets(self) -> list:
        """Get list of available dataset names."""
        return list(self.data.get('datasets', {}).keys())
    
    def get_available_models_for_dataset(self, dataset_name: str) -> list:
        """Get list of available models for a specific dataset."""
        benchmark = self.get_benchmark_for_dataset(dataset_name)
        if benchmark:
            return list(benchmark.keys())
        return []
    
    def get_all_models(self) -> list:
        """Get list of all available models."""
        return list(self.data.get('model_metadata', {}).keys())
    
    def validate_data(self) -> bool:
        """Validate the loaded data structure."""
        required_keys = ['datasets', 'benchmark_results', 'model_metadata']
        
        for key in required_keys:
            if key not in self.data:
                print(f"Missing required key: {key}")
                return False
        
        return True
    
    def reload_data(self) -> None:
        """Reload data from the YAML file."""
        self.load_data()
    
    def export_data(self, output_file: str = "exported_data.yaml") -> None:
        """Export the current data to a YAML file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                yaml.dump(self.data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"Data exported to {output_file}")
        except Exception as e:
            print(f"Error exporting data: {e}")

# Global data loader instance
data_loader = NapolabDataLoader()

# Convenience functions for backward compatibility
def get_napolab_datasets() -> Dict[str, Any]:
    """Get Napolab datasets (for backward compatibility)."""
    return data_loader.get_datasets()

def get_sample_benchmark_results() -> Dict[str, Any]:
    """Get benchmark results (for backward compatibility)."""
    return data_loader.get_benchmark_results()

def get_model_metadata() -> Dict[str, Any]:
    """Get model metadata (for backward compatibility)."""
    return data_loader.get_model_metadata()

def get_additional_models() -> Dict[str, Any]:
    """Get additional models (for backward compatibility)."""
    return data_loader.get_additional_models() 