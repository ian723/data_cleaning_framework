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
from .utils import memory_optimize, convert_dtypes  # Utility functions to optimize memory usage and convert data types
from ..config import settings  # Configuration settings (e.g., thresholds, flags) for the cleaning process

class DataCleaner:
    def __init__(self, config_path: Union[str, Path] = "config/cleaning_rules.yaml"):
        """
        Initializes the DataCleaner.
        Loads the cleaning configuration from a YAML file and sets up
        imputation and scaling methods.
        
        :param config_path: Path to the YAML configuration file.
        """
        # Load cleaning rules from the specified configuration file
        self.config = self._load_config(config_path)
        # Initialize a KNN imputer for handling missing values
        self.imputer = KNNImputer(n_neighbors=5)
        # Initialize a power transformer for scaling numerical features
        self.scaler = PowerTransformer(method='yeo-johnson')
        
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """
        Loads cleaning rules from a YAML configuration file.
        
        :param config_path: Path to the YAML file.
        :return: A dictionary containing cleaning configuration settings.
        """
        import yaml  # Import here to limit the scope
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def clean_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """
        The main entry point for cleaning data.
        Reads the data from the provided file, applies missing value handling,
        cleans text data, handles outliers, engineers features, validates the data,
        and finally optimizes the DataFrame's memory usage.
        
        :param input_path: Path to the input data file.
        :return: A cleaned pandas DataFrame.
        """
        logger.info(f"Starting cleaning process for {input_path}")
        # Read the data based on file type
        df = self._read_data(input_path)
        # Process missing values
        df = self._handle_missing_values(df)
        # Clean text fields if required
        df = self._clean_text_data(df)
        # Detect and handle outliers in numeric columns
        df = self._handle_outliers(df)
        # Create additional features from existing data
        df = self._engineer_features(df)
        # Validate the data against predefined rules
        df = self._validate_data(df)
        # Optimize the DataFrame to use less memory
        df = memory_optimize(df)
        logger.success(f"Completed cleaning for {input_path}")
        return df
    
    def _read_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """
        Reads the input data from various file formats (CSV, Parquet, Excel, etc.)
        based on the file extension.
        
        :param input_path: The path to the data file.
        :return: A DataFrame containing the loaded data.
        :raises ValueError: If the file format is unsupported.
        """
        file_extension = Path(input_path).suffix.lower()  # Extract the file extension
        
        # Mapping of file extensions to their corresponding pandas read functions
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
        """
        Handles missing values in the DataFrame by dropping columns with
        a high proportion of missing values and applying various imputation
        strategies defined in the configuration.
        
        :param df: The input DataFrame.
        :return: A DataFrame with missing values handled.
        """
        logger.info("Handling missing values")
        
        # Calculate the ratio of missing values for each column
        missing_ratio = df.isnull().mean()
        # Drop columns that exceed the missing threshold from settings
        cols_to_drop = missing_ratio[missing_ratio > settings.MISSING_THRESHOLD].index
        df = df.drop(columns=cols_to_drop)
        
        # Loop through each column and its imputation strategy from the config
        for col, strategy in self.config.get('missing_values', {}).items():
            if col in df.columns:
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strategy == "knn":
                    # Use the KNN imputer (reshapes the data as needed)
                    df[col] = self.imputer.fit_transform(df[[col]])
                elif strategy == "drop_column":
                    df = df.drop(columns=[col])
        
        return df
    
    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans text data in specified columns by stripping whitespace,
        converting to lowercase, normalizing spaces, and removing punctuation.
        
        :param df: The input DataFrame.
        :return: A DataFrame with cleaned text data.
        """
        # Only perform text cleaning if the corresponding setting is True
        if not settings.TEXT_CLEANING:
            return df
            
        logger.info("Cleaning text data")
        # Retrieve text columns from the config
        text_cols = self.config.get('text_columns', [])
        
        for col in text_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)  # Ensure the column is treated as string
                    .str.strip()  # Remove leading/trailing whitespace
                    .str.lower()  # Convert to lowercase for consistency
                    .str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with one
                    .str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
                )
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects and handles outliers in numeric columns using methods
        specified in the configuration. Supported methods include winsorization
        and isolation forest.
        
        :param df: The input DataFrame.
        :return: A DataFrame with outliers processed.
        """
        logger.info("Handling outliers")
        # Identify all numeric columns in the DataFrame
        num_cols = df.select_dtypes(include=np.number).columns
        
        for col in num_cols:
            if self.config['outlier_handling']['method'] == 'winsorize':
                df[col] = self._winsorize(df[col])
            elif self.config['outlier_handling']['method'] == 'isolation_forest':
                df = self._iso_forest_outlier_handling(df, col)
        
        return df
    
    def _winsorize(self, series: pd.Series) -> pd.Series:
        """
        Applies winsorization to a pandas Series by clipping values
        outside the 5th and 95th percentiles.
        
        :param series: A pandas Series representing a numerical column.
        :return: A Series with its extreme values clipped.
        """
        q1 = series.quantile(0.05)  # 5th percentile
        q3 = series.quantile(0.95)  # 95th percentile
        return series.clip(lower=q1, upper=q3)
    
    def _iso_forest_outlier_handling(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Uses the Isolation Forest algorithm to detect outliers in a given column.
        Outliers are replaced with NaN and then imputed with the median value.
        
        :param df: The input DataFrame.
        :param col: The column name to process.
        :return: A DataFrame with outliers handled in the specified column.
        """
        clf = IsolationForest(contamination=0.05, random_state=42)
        # Fit the model and predict outliers (-1 indicates an outlier)
        outliers = clf.fit_predict(df[[col]])
        # Replace outliers with NaN
        df[col] = np.where(outliers == -1, np.nan, df[col])
        # Impute the NaN values (from outliers) with the median of the column
        return df.fillna({col: df[col].median()})
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering by extracting date features and creating
        interactions between existing features.
        
        :param df: The input DataFrame.
        :return: A DataFrame with additional engineered features.
        """
        logger.info("Engineering features")
        # Extract additional features from date columns
        df = self._extract_date_features(df)
        # Create new features based on interactions between columns
        df = self._create_interactions(df)
        return df
    
    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features such as year, month, and day of the week from date columns,
        as specified in the configuration.
        
        :param df: The input DataFrame.
        :return: The DataFrame with new date-related feature columns.
        """
        # Get the list of date columns from the configuration
        date_cols = self.config.get('date_columns', [])
        
        for col in date_cols:
            if col in df.columns:
                # Convert the column to a datetime type
                df[col] = pd.to_datetime(df[col])
                # Loop through each date feature to extract (e.g., year, month)
                for feature in self.config['feature_engineering']['date_features']:
                    if feature == 'year':
                        df[f'{col}_year'] = df[col].dt.year
                    elif feature == 'month':
                        df[f'{col}_month'] = df[col].dt.month
                    elif feature == 'day_of_week':
                        df[f'{col}_day_of_week'] = df[col].dt.dayofweek
        return df
    
    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates interaction features by combining columns (for example, by multiplying them)
        based on the configuration.
        
        :param df: The input DataFrame.
        :return: The DataFrame with new interaction features.
        """
        # Get interaction configurations from the config file
        interactions = self.config['feature_engineering'].get('interactions', [])
        
        for interaction in interactions:
            cols = interaction['columns']
            # Only create the interaction if all specified columns are present in the DataFrame
            if all(c in df.columns for c in cols):
                if interaction['operation'] == 'multiply':
                    # Create a new column with a name combining the two columns
                    df['_x_'.join(cols)] = df[cols[0]] * df[cols[1]]
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the DataFrame using a separate DataValidator class.
        If validation fails, it either raises an error (in strict mode) or logs warnings.
        
        :param df: The input DataFrame.
        :return: The validated DataFrame.
        :raises ValueError: If validation fails in strict mode.
        """
        from .validator import DataValidator  # Import the validator module from the same package
        # Skip validation if the setting is disabled
        if not settings.VALIDATE_DATA:
            return df
            
        logger.info("Validating data")
        validator = DataValidator(self.config)
        report = validator.validate(df)
        
        # If the report indicates validation issues, process them based on STRICT_MODE setting
        if not report['is_valid']:
            error_msg = "\n".join(report['errors'])
            if settings.STRICT_MODE:
                raise ValueError(f"Data validation failed:\n{error_msg}")
            else:
                logger.warning(f"Data validation issues:\n{error_msg}")
        
        return df
