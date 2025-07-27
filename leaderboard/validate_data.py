#!/usr/bin/env python3
"""
Validation script for the updated Napolab data structure
"""

from data_loader import NapolabDataLoader
from manage_data import validate_yaml_structure
import pandas as pd

def main():
    """Validate the updated data structure."""
    print("🔍 Validating Updated Napolab Data Structure")
    print("=" * 50)
    print("📚 Data Source: Master's thesis 'Lessons learned from the evaluation of Portuguese language models'")
    print("   by Ruan Chaves Rodrigues (2023) - University of Malta")
    print("   Available at: https://www.um.edu.mt/library/oar/handle/123456789/120557")
    print("=" * 50)
    
    # Load data
    data_loader = NapolabDataLoader()
    data = data_loader.data
    
    # Validate structure
    print("\n1. Validating YAML structure...")
    if validate_yaml_structure(data):
        print("✅ YAML structure is valid!")
    else:
        print("❌ YAML structure has issues!")
        return
    
    # Check datasets
    print("\n2. Checking datasets...")
    datasets = data_loader.get_datasets()
    print(f"📊 Found {len(datasets)} datasets:")
    for name, info in datasets.items():
        print(f"   - {name}: {info['name']} ({', '.join(info['tasks'])})")
    
    # Check benchmark results
    print("\n3. Checking benchmark results...")
    benchmark_results = data_loader.get_benchmark_results()
    print(f"🏆 Found {len(benchmark_results)} benchmark datasets:")
    for dataset_name, models in benchmark_results.items():
        print(f"   - {dataset_name}: {len(models)} models")
    
    # Check model metadata
    print("\n4. Checking model metadata...")
    model_metadata = data_loader.get_model_metadata()
    print(f"🤖 Found {len(model_metadata)} models:")
    
    # Group models by architecture
    architectures = {}
    for model_name, metadata in model_metadata.items():
        arch = metadata['architecture']
        if arch not in architectures:
            architectures[arch] = []
        architectures[arch].append(model_name)
    
    for arch, models in architectures.items():
        print(f"   - {arch}: {len(models)} models")
        for model in models[:3]:  # Show first 3 models
            print(f"     * {model}")
        if len(models) > 3:
            print(f"     ... and {len(models) - 3} more")
    
    # Test data access functions
    print("\n5. Testing data access functions...")
    
    # Test getting available models for a dataset
    test_dataset = list(benchmark_results.keys())[0]
    models = data_loader.get_available_models_for_dataset(test_dataset)
    print(f"   Available models for {test_dataset}: {len(models)} models")
    
    # Test getting model info
    if models:
        test_model = models[0]
        model_info = data_loader.get_model_info(test_model)
        if model_info:
            print(f"   Model {test_model}: {model_info['parameters']:,} parameters")
    
    # Create a summary table
    print("\n6. Creating summary table...")
    summary_data = []
    
    for dataset_name, models in benchmark_results.items():
        for model_name, metrics in models.items():
            if model_name in model_metadata:
                summary_data.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Architecture': model_metadata[model_name]['architecture'],
                    'Parameters': model_metadata[model_name]['parameters'],
                    'Performance': metrics.get('accuracy', 0)
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(f"📋 Summary: {len(df)} model-dataset combinations")
        print(f"   Average performance: {df['Performance'].mean():.3f}")
        print(f"   Best performance: {df['Performance'].max():.3f}")
        print(f"   Models with >0.9 performance: {(df['Performance'] > 0.9).sum()}")
    
    print("\n✅ Validation completed successfully!")
    print("🚀 The updated data structure is ready to use!")

if __name__ == "__main__":
    main() 