import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
from typing import Union
from .utils import memory_optimize  # Handy utilities to slim down memory usage
from config.settings import settings  # Our trusted config settings

class DataCleaner:
    def __init__(self, config_path: Union[str, Path] = "config/cleaning_rules.yaml"):
        """
        Initialize the DataCleaner.
        Loads our YAML configuration and sets up the imputer and scaler.
        """
        self.config = self._load_config(config_path)
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = PowerTransformer(method='yeo-johnson')

    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load cleaning rules from a YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def clean_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """
        Main routine: reads, cleans, engineers features, validates, and optimizes data.
        """
        logger.info(f"Cleaning data from {input_path}")
        df = self._read_data(input_path)
        df = self._handle_missing_values(df)
        df = self._clean_text_data(df)
        df = self._handle_outliers(df)
        df = self._engineer_features(df)
        df = self._validate_data(df)
        df = memory_optimize(df)
        logger.info("Data cleaning completed.")
        return df

    def _read_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """Read data from file based on its extension."""
        ext = Path(input_path).suffix.lower()
        readers = {
            '.csv': pd.read_csv,
            '.parquet': pd.read_parquet,
            '.feather': pd.read_feather,
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
        }
        if ext in readers:
            return readers[ext](input_path)
        raise ValueError(f"Unsupported file format: {ext}")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with too many missing values and fill in gaps based on config.
        """
        missing = df.isnull().mean()
        to_drop = missing[missing > settings.MISSING_THRESHOLD].index
        if len(to_drop):
            logger.info(f"Dropping columns: {list(to_drop)}")
            df = df.drop(columns=to_drop)
        for col, strategy in self.config.get('missing_values', {}).items():
            if col not in df.columns:
                continue
            if strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == "knn":
                df[col] = self.imputer.fit_transform(df[[col]])
            elif strategy == "drop_column":
                df = df.drop(columns=[col])
        return df

    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean text columns by trimming, lowercasing, and stripping unwanted punctuation.
        """
        if not settings.TEXT_CLEANING:
            return df
        text_cols = self.config.get('text_columns', [])
        for col in text_cols:
            if col in df.columns:
                df[col] = (df[col].astype(str)
                           .str.strip()
                           .str.lower()
                           .str.replace(r'\s+', ' ', regex=True)
                           .str.replace(r'[^\w\s]', '', regex=True))
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numeric columns using winsorization or isolation forest.
        """
        num_cols = df.select_dtypes(include=np.number).columns
        method = self.config.get('outlier_handling', {}).get('method', 'winsorize')
        for col in num_cols:
            if method == 'winsorize':
                # Cap values at 45 * 1.5 (feel free to adjust this hard-coded limit)
                df[col] = df[col].clip(upper=45 * 1.5)
            elif method == 'isolation_forest':
                df = self._iso_forest_outlier_handling(df, col)
        return df

    def _iso_forest_outlier_handling(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Detect outliers with Isolation Forest, mark them as NaN, then impute with median.
        """
        clf = IsolationForest(contamination=0.05, random_state=42)
        outliers = clf.fit_predict(df[[col]])
        df[col] = np.where(outliers == -1, np.nan, df[col])
        df[col].fillna(df[col].median(), inplace=True)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features such as date parts and interactions.
        """
        df = self._extract_date_features(df)
        df = self._create_interactions(df)
        return df

    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year, month, and day-of-week from date columns."""
        date_cols = self.config.get('date_columns', [])
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                for feat in self.config.get('feature_engineering', {}).get('date_features', []):
                    if feat == 'year':
                        df[f'{col}_year'] = df[col].dt.year
                    elif feat == 'month':
                        df[f'{col}_month'] = df[col].dt.month
                    elif feat == 'day_of_week':
                        df[f'{col}_day_of_week'] = df[col].dt.dayofweek
        return df

    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new interaction features as defined in the configuration."""
        interactions = self.config.get('feature_engineering', {}).get('interactions', [])
        for interaction in interactions:
            cols = interaction.get('columns', [])
            if all(col in df.columns for col in cols) and interaction.get('operation') == 'multiply' and len(cols) == 2:
                new_col = '_x_'.join(cols)
                df[new_col] = df[cols[0]] * df[cols[1]]
        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the cleaned data. If issues are found, either raise an error or log warnings.
        """
        from .validator import DataValidator
        if not settings.VALIDATE_DATA:
            return df
        validator = DataValidator(self.config)
        report = validator.validate(df)
        if not report.get('is_valid', True):
            errors = "\n".join(report.get('errors', []))
            if settings.STRICT_MODE:
                raise ValueError(f"Validation failed:\n{errors}")
            else:
                logger.warning(f"Validation issues:\n{errors}")
        return df

if __name__ == '__main__':
    sample_input = "data/sample_data.csv"  # Adjust to your actual file
    cleaner = DataCleaner()
    try:
        cleaned_df = cleaner.clean_data(sample_input)
        logger.info("Data cleaning succeeded.")
    except Exception as e:
        logger.error(f"Error during cleaning: {e}")
