import pandas as pd
import numpy as np
from loguru import logger

def memory_optimize(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage"""
    logger.info("Optimizing memory usage")
    
    # Downcast numerical columns
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
    
    # Convert object columns to category
    obj_cols = df.select_dtypes(include='object').columns
    for col in obj_cols:
        if df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    
    return df

def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to proper dtypes"""
    date_cols = df.select_dtypes(include='datetime').columns
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    return df