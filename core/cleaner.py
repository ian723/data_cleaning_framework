import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from loguru import logger
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
from typing import Union
from .utils import memory_optimize, convert_dtypes
from ..config import settings

class DataCleaner:
    def __init__(self, config_path: Union[str, Path] = "config/cleaning_rules.yaml"):
        self.config = self._load_config(config_path)
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = PowerTransformer(method='yeo-johnson')
        
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load cleaning rules from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def clean_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """Main cleaning entry point"""
        logger.info(f"Starting cleaning process for {input_path}")
        df = self._read_data(input_path)
        df = self._handle_missing_values(df)
        df = self._clean_text_data(df)
        df = self._handle_outliers(df)
        df = self._engineer_features(df)
        df = self._validate_data(df)
        df = memory_optimize(df)
        logger.success(f"Completed cleaning for {input_path}")
        return df
    
    def _read_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """Read data with appropriate engine"""
        file_extension = Path(input_path).suffix.lower()
        
        read_functions = {
            '.csv': pd.read_csv,
            '.parquet': pd.read_parquet,
            '.feather': pd.read_feather,
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel
        }
        
        if file_extension in read_functions:
            return read_functions[file_extension](input_path)
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing value imputation"""
        logger.info("Handling missing values")
        
        # Drop columns with high missingness
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > settings.MISSING_THRESHOLD].index
        df = df.drop(columns=cols_to_drop)
        
        # Apply configured imputation strategies
        for col, strategy in self.config.get('missing_values', {}).items():
            if col in df.columns:
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strategy == "knn":
                    df[col] = self.imputer.fit_transform(df[[col]])
                elif strategy == "drop_column":
                    df = df.drop(columns=[col])
        
        return df
    
    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize text data"""
        if not settings.TEXT_CLEANING:
            return df
            
        logger.info("Cleaning text data")
        text_cols = self.config.get('text_columns', [])
        
        for col in text_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.replace(r'[^\w\s]', '', regex=True)
                )
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced outlier detection and handling"""
        logger.info("Handling outliers")
        num_cols = df.select_dtypes(include=np.number).columns
        
        for col in num_cols:
            if self.config['outlier_handling']['method'] == 'winsorize':
                df[col] = self._winsorize(df[col])
            elif self.config['outlier_handling']['method'] == 'isolation_forest':
                df = self._iso_forest_outlier_handling(df, col)
        
        return df
    
    def _winsorize(self, series: pd.Series) -> pd.Series:
        """Winsorize outliers"""
        q1 = series.quantile(0.05)
        q3 = series.quantile(0.95)
        return series.clip(lower=q1, upper=q3)
    
    def _iso_forest_outlier_handling(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Isolation Forest based outlier handling"""
        clf = IsolationForest(contamination=0.05, random_state=42)
        outliers = clf.fit_predict(df[[col]])
        df[col] = np.where(outliers == -1, np.nan, df[col])
        return df.fillna({col: df[col].median()})
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering pipeline"""
        logger.info("Engineering features")
        df = self._extract_date_features(df)
        df = self._create_interactions(df)
        return df
    
    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from date columns"""
        date_cols = self.config.get('date_columns', [])
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                for feature in self.config['feature_engineering']['date_features']:
                    if feature == 'year':
                        df[f'{col}_year'] = df[col].dt.year
                    elif feature == 'month':
                        df[f'{col}_month'] = df[col].dt.month
                    elif feature == 'day_of_week':
                        df[f'{col}_day_of_week'] = df[col].dt.dayofweek
        return df
    
    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        interactions = self.config['feature_engineering'].get('interactions', [])
        
        for interaction in interactions:
            cols = interaction['columns']
            if all(c in df.columns for c in cols):
                if interaction['operation'] == 'multiply':
                    df['_x_'.join(cols)] = df[cols[0]] * df[cols[1]]
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data validation checks"""
        from .validator import DataValidator
        
        if not settings.VALIDATE_DATA:
            return df
            
        logger.info("Validating data")
        validator = DataValidator(self.config)
        report = validator.validate(df)
        
        if not report['is_valid']:
            error_msg = "\n".join(report['errors'])
            if settings.STRICT_MODE:
                raise ValueError(f"Data validation failed:\n{error_msg}")
            else:
                logger.warning(f"Data validation issues:\n{error_msg}")
        
        return df