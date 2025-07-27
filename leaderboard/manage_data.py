#!/usr/bin/env python3
"""
Data Management Utility for Napolab Leaderboard

This script provides utilities to manage, validate, and update the YAML data file.
"""

import yaml
import argparse
from pathlib import Path
from data_loader import NapolabDataLoader
from typing import Dict, Any

def validate_yaml_structure(data: Dict[str, Any]) -> bool:
    """Validate the YAML data structure."""
    print("🔍 Validating YAML structure...")
    
    required_sections = ['datasets', 'benchmark_results', 'model_metadata']
    
    for section in required_sections:
        if section not in data:
            print(f"❌ Missing required section: {section}")
            return False
        print(f"✅ Found section: {section}")
    
    # Validate datasets
    print("\n📊 Validating datasets...")
    for dataset_name, dataset_info in data['datasets'].items():
        required_fields = ['name', 'description', 'tasks', 'url']
        for field in required_fields:
            if field not in dataset_info:
                print(f"❌ Dataset '{dataset_name}' missing field: {field}")
                return False
        print(f"✅ Dataset '{dataset_name}' is valid")
    
    # Validate benchmark results
    print("\n🏆 Validating benchmark results...")
    for dataset_name, models in data['benchmark_results'].items():
        if dataset_name not in data['datasets']:
            print(f"⚠️  Warning: Benchmark for '{dataset_name}' but no dataset info found")
        
        for model_name, metrics in models.items():
            if not isinstance(metrics, dict):
                print(f"❌ Invalid metrics format for model '{model_name}'")
                return False
            print(f"✅ Model '{model_name}' has {len(metrics)} metrics")
    
    # Validate model metadata
    print("\n🤖 Validating model metadata...")
    for model_name, metadata in data['model_metadata'].items():
        required_fields = ['parameters', 'architecture', 'base_model', 'task']
        for field in required_fields:
            if field not in metadata:
                print(f"❌ Model '{model_name}' missing field: {field}")
                return False
        print(f"✅ Model '{model_name}' is valid")
    
    print("\n🎉 All validations passed!")
    return True

def create_sample_data() -> Dict[str, Any]:
    """Create a sample data structure."""
    return {
        'datasets': {
            'sample_dataset': {
                'name': 'Sample Dataset',
                'description': 'A sample dataset for testing',
                'tasks': ['Classification'],
                'url': 'https://huggingface.co/datasets/sample'
            }
        },
        'benchmark_results': {
            'sample_dataset': {
                'sample-model': {
                    'accuracy': 0.85,
                    'f1': 0.84
                }
            }
        },
        'model_metadata': {
            'sample-model': {
                'parameters': 100000000,
                'architecture': 'BERT Base',
                'base_model': 'bert-base-uncased',
                'task': 'Classification',
                'huggingface_url': 'https://huggingface.co/sample/model'
            }
        },
        'additional_models': {}
    }

def add_dataset(data: Dict[str, Any], dataset_name: str, name: str, description: str, 
                tasks: list, url: str) -> Dict[str, Any]:
    """Add a new dataset to the data structure."""
    data['datasets'][dataset_name] = {
        'name': name,
        'description': description,
        'tasks': tasks,
        'url': url
    }
    print(f"✅ Added dataset: {dataset_name}")
    return data

def add_benchmark_result(data: Dict[str, Any], dataset_name: str, model_name: str, 
                        metrics: Dict[str, float]) -> Dict[str, Any]:
    """Add benchmark results for a model on a dataset."""
    if dataset_name not in data['benchmark_results']:
        data['benchmark_results'][dataset_name] = {}
    
    data['benchmark_results'][dataset_name][model_name] = metrics
    print(f"✅ Added benchmark result for {model_name} on {dataset_name}")
    return data

def add_model_metadata(data: Dict[str, Any], model_name: str, parameters: int, 
                      architecture: str, base_model: str, task: str, 
                      huggingface_url: str = None) -> Dict[str, Any]:
    """Add model metadata."""
    data['model_metadata'][model_name] = {
        'parameters': parameters,
        'architecture': architecture,
        'base_model': base_model,
        'task': task
    }
    
    if huggingface_url:
        data['model_metadata'][model_name]['huggingface_url'] = huggingface_url
    
    print(f"✅ Added model metadata: {model_name}")
    return data

def export_data(data: Dict[str, Any], output_file: str) -> None:
    """Export data to a YAML file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"✅ Data exported to {output_file}")
    except Exception as e:
        print(f"❌ Error exporting data: {e}")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Manage Napolab Leaderboard Data')
    parser.add_argument('action', choices=['validate', 'create-sample', 'add-dataset', 'add-benchmark', 'add-model'],
                       help='Action to perform')
    parser.add_argument('--data-file', default='data.yaml', help='Path to data file')
    parser.add_argument('--output', help='Output file for export')
    
    # Dataset arguments
    parser.add_argument('--dataset-name', help='Dataset name')
    parser.add_argument('--dataset-display-name', help='Dataset display name')
    parser.add_argument('--dataset-description', help='Dataset description')
    parser.add_argument('--dataset-tasks', nargs='+', help='Dataset tasks')
    parser.add_argument('--dataset-url', help='Dataset URL')
    
    # Benchmark arguments
    parser.add_argument('--model-name', help='Model name')
    parser.add_argument('--metrics', nargs='+', help='Metrics as key=value pairs')
    
    # Model metadata arguments
    parser.add_argument('--parameters', type=int, help='Number of parameters')
    parser.add_argument('--architecture', help='Model architecture')
    parser.add_argument('--base-model', help='Base model name')
    parser.add_argument('--task', help='Task type')
    parser.add_argument('--huggingface-url', help='Hugging Face URL')
    
    args = parser.parse_args()
    
    # Load existing data or create new
    data_loader = NapolabDataLoader(args.data_file)
    data = data_loader.data
    
    if args.action == 'validate':
        if validate_yaml_structure(data):
            print("✅ Data validation successful!")
        else:
            print("❌ Data validation failed!")
            return 1
    
    elif args.action == 'create-sample':
        data = create_sample_data()
        export_data(data, args.output or 'sample_data.yaml')
    
    elif args.action == 'add-dataset':
        if not all([args.dataset_name, args.dataset_display_name, args.dataset_description, 
                   args.dataset_tasks, args.dataset_url]):
            print("❌ All dataset arguments are required")
            return 1
        
        data = add_dataset(data, args.dataset_name, args.dataset_display_name, 
                          args.dataset_description, args.dataset_tasks, args.dataset_url)
        export_data(data, args.data_file)
    
    elif args.action == 'add-benchmark':
        if not all([args.dataset_name, args.model_name, args.metrics]):
            print("❌ All benchmark arguments are required")
            return 1
        
        # Parse metrics
        metrics = {}
        for metric in args.metrics:
            if '=' in metric:
                key, value = metric.split('=', 1)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    print(f"❌ Invalid metric value: {metric}")
                    return 1
        
        data = add_benchmark_result(data, args.dataset_name, args.model_name, metrics)
        export_data(data, args.data_file)
    
    elif args.action == 'add-model':
        if not all([args.model_name, args.parameters, args.architecture, 
                   args.base_model, args.task]):
            print("❌ All model metadata arguments are required")
            return 1
        
        data = add_model_metadata(data, args.model_name, args.parameters, 
                                args.architecture, args.base_model, args.task, 
                                args.huggingface_url)
        export_data(data, args.data_file)
    
    return 0

if __name__ == "__main__":
    exit(main()) 