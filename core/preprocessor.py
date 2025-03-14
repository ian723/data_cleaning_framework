import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from loguru import logger
from .utils import memory_optimize

class DataPreprocessor:
    def __init__(self):
        # Initialize variables for the preprocessor and feature names.
        self.preprocessor = None
        # Store the feature names generated by OneHotEncoder  
        self.feature_names = [] 
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Main preprocessing method
        logger.info("Starting data preprocessing")
        # Encode categorical features using one-hot encoding (sparse DataFrame output)
        df = self._encode_categoricals(df)
        # Normalize numerical features using StandardScaler
        df = self._normalize_numerical(df)
        # Apply logarithmic transformation to highly skewed numerical features
        df = self._handle_skewness(df)
        # Optimize DataFrame memory usage (e.g., reducing data types)
        df = memory_optimize(df)
        logger.success("Completed preprocessing")
        return df
    
    # Encode categorical features using one-hot encoding
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Select columns of type 'object' or 'category'
        cat_cols = df.select_dtypes(include=['category', 'object']).columns
        
        # Create a ColumnTransformer that applies OneHotEncoder to the categorical columns.
        # Use `sparse_output=True` (for scikit-learn >=1.2) to keep output sparse.
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cols)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform the data into a sparse matrix
        processed_array = preprocessor.fit_transform(df)
        # Retrieve and store the feature names generated by OneHotEncoder
        self.feature_names = preprocessor.get_feature_names_out()
        
        # Return a sparse DataFrame created from the sparse matrix
        return pd.DataFrame.sparse.from_spmatrix(processed_array, columns=self.feature_names)
    
    def _normalize_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identify numeric columns
        num_cols = df.select_dtypes(include=np.number).columns
        # Initialize the scaler with with_mean=False (to avoid centering sparse data)
        scaler = StandardScaler(with_mean=False)
        
        # Extract numeric columns; if they are sparse, convert them to dense.
        df_num = df[num_cols]
        if hasattr(df_num, "sparse"):
            df_num = df_num.sparse.to_dense()
        
        # Fit and transform the numeric data
        scaled_array = scaler.fit_transform(df_num)
        # Create a new DataFrame from the scaled data with the same index and column names
        df_scaled = pd.DataFrame(scaled_array, index=df.index, columns=num_cols)
        
        # Replace each numeric column in the original DataFrame with the scaled values
        for col in num_cols:
            df[col] = df_scaled[col]
            
        return df
    
    def _handle_skewness(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identify numeric columns
        num_cols = df.select_dtypes(include=np.number).columns
        # Iterate through each numeric column
        for col in num_cols:
            skewness = df[col].skew()  # Calculate skewness
            if abs(skewness) > 1.0:
                df[col] = np.log1p(df[col])
        return df
