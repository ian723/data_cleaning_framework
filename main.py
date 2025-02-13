import argparse
from pathlib import Path
from loguru import logger
import pandas as pd
import ydata_profiling
from core.cleaner import DataCleaner
from core.preprocessor import DataPreprocessor
from config import settings

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Data Cleaning Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", default=settings.OUTPUT_DIR, help="Output directory")
    parser.add_argument("-c", "--config", default="config/cleaning_rules.yaml", help="Config file")
    parser.add_argument("--dask", action="store_true", help="Use Dask for parallel processing")
    parser.add_argument("--profile", action="store_true", help="Generate data profile report")
    args = parser.parse_args()
    
    settings.USE_DASK = args.dask
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cleaner = DataCleaner(args.config)
    preprocessor = DataPreprocessor()
    
    if input_path.is_dir():
        process_directory(input_path, output_dir, cleaner, preprocessor)
    else:
        process_file(input_path, output_dir, cleaner, preprocessor)
    
    if args.profile:
        generate_profile_report(output_dir)

def process_file(input_path: Path, output_dir: Path, cleaner: DataCleaner, preprocessor: DataPreprocessor):
    """Process single file"""
    try:
        logger.info(f"Processing {input_path.name}")
        df = cleaner.clean_data(input_path)
        df = preprocessor.preprocess(df)
        
        output_path = output_dir / f"cleaned_{input_path.name}"
        save_data(df, output_path)
        logger.success(f"Saved cleaned data to {output_path}")
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        raise

def process_directory(input_dir: Path, output_dir: Path, cleaner: DataCleaner, preprocessor: DataPreprocessor):
    """Process all files in directory"""
    logger.info(f"Processing directory {input_dir}")
    for file_path in input_dir.glob("*"):
        if file_path.suffix.lower() in [".csv", ".parquet", ".feather", ".xlsx", ".xls"]:
            process_file(file_path, output_dir, cleaner, preprocessor)

def generate_profile_report(output_dir: Path):
    """Generate comprehensive data profile report"""
    from ydata_profiling import ProfileReport
    
    logger.info("Generating data profile report")
    all_dfs = []
    
    for file in output_dir.glob("*"):
        if file.suffix == ".parquet":
            all_dfs.append(pd.read_parquet(file))
        elif file.suffix == ".csv":
            all_dfs.append(pd.read_csv(file))
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    profile = ProfileReport(combined_df, title="Data Cleaning Report", explorative=True)
    profile_path = output_dir / "data_profile.html"
    profile.to_file(profile_path)
    logger.success(f"Saved profile report to {profile_path}")

if __name__ == "__main__":
    main()