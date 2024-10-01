import argparse
from pipeline import run_pipeline

def main():
    """
    Main function to parse command-line arguments and initiate the pipeline process.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="End-to-end AI/ML pipeline for autoencoder.")
    
    # Add argument for dataset path
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        required=True, 
        help="Path to the dataset directory. The directory must contain 'train' and 'test' folders."
    )
    
    # Parse the arguments
    args = parser.parse_args()

    # Run the pipeline with the provided dataset path
    run_pipeline(args.dataset_path)

if __name__ == "__main__":
    main()
