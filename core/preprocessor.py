import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from loguru import logger
from .utils import memory_optimize

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = []
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing pipeline"""
        logger.info("Starting data preprocessing")
        df = self._encode_categoricals(df)
        df = self._normalize_numerical(df)
        df = self._handle_skewness(df)
        df = memory_optimize(df)
        logger.success("Completed preprocessing")
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced categorical encoding"""
        cat_cols = df.select_dtypes(include=['category', 'object']).columns
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform data
        processed_array = preprocessor.fit_transform(df)
        self.feature_names = preprocessor.get_feature_names_out()
        
        return pd.DataFrame(processed_array, columns=self.feature_names)
    
    def _normalize_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features"""
        num_cols = df.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df
    
    def _handle_skewness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle skewed numerical features"""
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1.0:
                df[col] = np.log1p(df[col])
        return df