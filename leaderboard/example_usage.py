#!/usr/bin/env python3
"""
Example Usage of Napolab Leaderboard Data Management

This script demonstrates how to use the YAML-based data management system.
"""

from data_loader import NapolabDataLoader
from manage_data import validate_yaml_structure, add_dataset, add_benchmark_result, add_model_metadata, export_data
import yaml

def example_usage():
    """Demonstrate the data management functionality."""
    
    print("🚀 Napolab Leaderboard Data Management Example")
    print("=" * 50)
    
    # 1. Load existing data
    print("\n1. Loading existing data...")
    data_loader = NapolabDataLoader()
    data = data_loader.data
    
    print(f"✅ Loaded {len(data['datasets'])} datasets")
    print(f"✅ Loaded {len(data['model_metadata'])} models")
    
    # 2. Validate the data structure
    print("\n2. Validating data structure...")
    if validate_yaml_structure(data):
        print("✅ Data structure is valid!")
    else:
        print("❌ Data structure has issues!")
        return
    
    # 3. Add a new dataset
    print("\n3. Adding a new dataset...")
    data = add_dataset(
        data=data,
        dataset_name="example_dataset",
        name="Example Dataset",
        description="An example dataset for demonstration",
        tasks=["Classification", "Sentiment Analysis"],
        url="https://huggingface.co/datasets/example"
    )
    
    # 4. Add a new model
    print("\n4. Adding a new model...")
    data = add_model_metadata(
        data=data,
        model_name="example-model",
        parameters=125000000,
        architecture="BERT Large",
        base_model="bert-large-uncased",
        task="Classification",
        huggingface_url="https://huggingface.co/example/model"
    )
    
    # 5. Add benchmark results
    print("\n5. Adding benchmark results...")
    data = add_benchmark_result(
        data=data,
        dataset_name="example_dataset",
        model_name="example-model",
        metrics={
            "accuracy": 0.89,
            "f1": 0.88,
            "precision": 0.90,
            "recall": 0.87
        }
    )
    
    # 6. Export the updated data
    print("\n6. Exporting updated data...")
    export_data(data, "example_updated_data.yaml")
    
    # 7. Demonstrate data access
    print("\n7. Demonstrating data access...")
    
    # Get dataset info
    dataset_info = data_loader.get_dataset_info("assin")
    if dataset_info:
        print(f"📊 ASSIN dataset: {dataset_info['name']}")
        print(f"   Tasks: {', '.join(dataset_info['tasks'])}")
    
    # Get available models for a dataset
    models = data_loader.get_available_models_for_dataset("assin")
    print(f"🤖 Available models for ASSIN: {len(models)} models")
    
    # Get model info
    model_info = data_loader.get_model_info("mdeberta-v3-base-assin-similarity")
    if model_info:
        print(f"🔧 Model parameters: {model_info['parameters']:,}")
        print(f"   Architecture: {model_info['architecture']}")
    
    print("\n✅ Example completed successfully!")
    print("📁 Check 'example_updated_data.yaml' for the updated data")

def demonstrate_yaml_structure():
    """Show the YAML structure."""
    print("\n📋 YAML Data Structure Example:")
    print("-" * 30)
    
    example_data = {
        'datasets': {
            'my_dataset': {
                'name': 'My Dataset',
                'description': 'A custom dataset',
                'tasks': ['Classification'],
                'url': 'https://huggingface.co/datasets/my_dataset'
            }
        },
        'benchmark_results': {
            'my_dataset': {
                'my-model': {
                    'accuracy': 0.92,
                    'f1': 0.91
                }
            }
        },
        'model_metadata': {
            'my-model': {
                'parameters': 110000000,
                'architecture': 'BERT Base',
                'base_model': 'bert-base-uncased',
                'task': 'Classification',
                'huggingface_url': 'https://huggingface.co/my-model'
            }
        }
    }
    
    print(yaml.dump(example_data, default_flow_style=False, allow_unicode=True))

if __name__ == "__main__":
    example_usage()
    demonstrate_yaml_structure() 