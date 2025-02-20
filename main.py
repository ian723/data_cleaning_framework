import argparse # for parsing command-line arguments
from pathlib import Path
from loguru import logger
import pandas as pd
# import ydata_profiling
from core.cleaner import DataCleaner
from core.preprocessor import DataPreprocessor
from config.settings import settings

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Enterprise Data Cleaning Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", default=settings.OUTPUT_DIR, help="Output directory")
    parser.add_argument("-c", "--config", default="config/cleaning_rules.yaml", help="Config file")
    parser.add_argument("--dask", action="store_true", help="Enable Dask for parallel processing")
    parser.add_argument("--profile", action="store_true", help="Generate a data profile report")
    args = parser.parse_args()

    # Update settings based on CLI options
    settings.USE_DASK = args.dask

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize our core components
    cleaner = DataCleaner(args.config)
    preprocessor = DataPreprocessor()

    # Process input depending on whether it's a single file or directory
    if input_path.is_dir():
        process_directory(input_path, output_dir, cleaner, preprocessor)
    else:
        process_file(input_path, output_dir, cleaner, preprocessor)

    # Generate a profile report if requested
    if args.profile:
        generate_profile_report(output_dir)

def process_file(input_path: Path, output_dir: Path, cleaner: DataCleaner, preprocessor: DataPreprocessor):
    try:
        logger.info(f"Processing file: {input_path.name}")
        df = cleaner.clean_data(input_path)
        df = preprocessor.preprocess(df)
        output_path = output_dir / f"cleaned_{input_path.name}"
        save_data(df, output_path)
        logger.success(f"Saved cleaned data to {output_path}")
    except Exception as err:
        logger.error(f"Error processing {input_path.name}: {err}")
        raise

def process_directory(input_dir: Path, output_dir: Path, cleaner: DataCleaner, preprocessor: DataPreprocessor):
    # Process all files in the directory
    logger.info(f"Scanning directory: {input_dir}")
    for file_path in input_dir.glob("*"):
        if file_path.suffix.lower() in [".csv", ".parquet", ".feather", ".xlsx", ".xls"]:
            process_file(file_path, output_dir, cleaner, preprocessor)

def generate_profile_report(output_dir: Path):
    # Generate a data profile report
    from ydata_profiling import ProfileReport
    logger.info("Generating data profile report...")

    dfs = []
    for file in output_dir.glob("*"):
        if file.suffix == ".parquet":
            dfs.append(pd.read_parquet(file))
        elif file.suffix == ".csv":
            dfs.append(pd.read_csv(file))

    if not dfs:
        logger.warning("No processed files found for profiling.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    profile = ProfileReport(combined_df, title="Data Cleaning Report", explorative=True)
    report_path = output_dir / "data_profile.html"
    profile.to_file(report_path)
    logger.success(f"Profile report saved to {report_path}")

def save_data(df: pd.DataFrame, output_path: Path):
    # Save the cleaned data to different file formats.
    ext = output_path.suffix.lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False)
    elif ext == ".parquet":
        df.to_parquet(output_path, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

if __name__ == "__main__":
    main()
