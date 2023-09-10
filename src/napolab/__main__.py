import os
import argparse
from napolab import export_napolab_benchmark

def main():
    parser = argparse.ArgumentParser(description="Export the Napolab benchmark datasets as CSV")
    
    # Define command line arguments
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=os.getcwd(), 
        help='The path where datasets will be saved. Default is the current directory.'
    )
    
    parser.add_argument(
        '--include_translations', 
        type=bool, 
        default=True, 
        help='Whether to include translated versions of the datasets. Defaults to True.'
    )
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Call the function
    export_napolab_benchmark(args.output_path, args.include_translations)

if __name__ == "__main__":
    main()