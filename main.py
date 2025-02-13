import argparse  # For parsing command-line arguments
from pathlib import Path  # To handle file system paths in a platform-independent way
from loguru import logger  # For logging messages (info, error, success, etc.)
import pandas as pd  # For DataFrame operations
import ydata_profiling  # For generating data profile reports
from core.cleaner import DataCleaner  # Custom class for data cleaning
from core.preprocessor import DataPreprocessor  # Custom class for data preprocessing
from config.settings import settings  # Project-specific settings and configurations

def main():
    # Set up the argument parser with a description and default formatting
    parser = argparse.ArgumentParser(
        description="Enterprise Data Cleaning Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Define required and optional arguments
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", default=settings.OUTPUT_DIR, help="Output directory")
    parser.add_argument("-c", "--config", default="config/cleaning_rules.yaml", help="Config file")
    parser.add_argument("--dask", action="store_true", help="Use Dask for parallel processing")
    parser.add_argument("--profile", action="store_true", help="Generate data profile report")
    args = parser.parse_args()  # Parse the command-line arguments

    # Update settings based on parsed arguments
    settings.USE_DASK = args.dask

    # Convert the input and output paths from string to Path objects
    input_path = Path(args.input)
    output_dir = Path(args.output)
    # Create the output directory if it doesn't exist (including any necessary parent directories)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the DataCleaner with the provided configuration file
    cleaner = DataCleaner(args.config)
    # Initialize the DataPreprocessor
    preprocessor = DataPreprocessor()
    
    # Check if the input path is a directory or a single file
    if input_path.is_dir():
        process_directory(input_path, output_dir, cleaner, preprocessor)
    else:
        process_file(input_path, output_dir, cleaner, preprocessor)
    
    # If the --profile flag is set, generate a data profile report
    if args.profile:
        generate_profile_report(output_dir)

def process_file(input_path: Path, output_dir: Path, cleaner: DataCleaner, preprocessor: DataPreprocessor):
    """Process a single file: clean, preprocess, and save it."""
    try:
        # Log the file being processed
        logger.info(f"Processing {input_path.name}")
        # Clean the data using the DataCleaner
        df = cleaner.clean_data(input_path)
        # Preprocess the cleaned data using the DataPreprocessor
        df = preprocessor.preprocess(df)
        
        # Build the output file path by prefixing the original filename with 'cleaned_'
        output_path = output_dir / f"cleaned_{input_path.name}"
        # Save the processed DataFrame to the specified file format
        save_data(df, output_path)
        logger.success(f"Saved cleaned data to {output_path}")
    except Exception as e:
        # Log any error that occurs during processing and re-raise the exception
        logger.error(f"Error processing {input_path}: {str(e)}")
        raise

def process_directory(input_dir: Path, output_dir: Path, cleaner: DataCleaner, preprocessor: DataPreprocessor):
    """Process all supported files in a directory."""
    logger.info(f"Processing directory {input_dir}")
    # Loop through all files in the directory
    for file_path in input_dir.glob("*"):
        # Check if the file extension is one of the supported formats
        if file_path.suffix.lower() in [".csv", ".parquet", ".feather", ".xlsx", ".xls"]:
            process_file(file_path, output_dir, cleaner, preprocessor)

def generate_profile_report(output_dir: Path):
    """Generate a comprehensive data profile report from processed files."""
    from ydata_profiling import ProfileReport  # Import here to limit scope
    
    logger.info("Generating data profile report")
    all_dfs = []
    
    # Read all processed files (CSV and Parquet formats) from the output directory
    for file in output_dir.glob("*"):
        if file.suffix == ".parquet":
            all_dfs.append(pd.read_parquet(file))
        elif file.suffix == ".csv":
            all_dfs.append(pd.read_csv(file))
    
    # Combine all DataFrames into a single DataFrame for profiling
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Generate the profile report with a title and explorative analysis enabled
    profile = ProfileReport(combined_df, title="Data Cleaning Report", explorative=True)
    # Define the path to save the HTML report
    profile_path = output_dir / "data_profile.html"
    # Save the profile report to a file
    profile.to_file(profile_path)
    logger.success(f"Saved profile report to {profile_path}")

def save_data(df: pd.DataFrame, output_path: Path):
    """Save the DataFrame to the specified output file based on its extension."""
    # Save as CSV if the extension is .csv
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    # Save as Parquet if the extension is .parquet
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    # Save as Excel if the extension is .xlsx or .xls
    elif output_path.suffix in [".xlsx", ".xls"]:
        df.to_excel(output_path, index=False)
    else:
        # Raise an error if the file format is unsupported
        raise ValueError(f"Unsupported file format: {output_path.suffix}")

# Entry point of the script: execute main() if this file is run directly
if __name__ == "__main__":
    main()
