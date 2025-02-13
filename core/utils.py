import pandas as pd
import numpy as np
from loguru import logger

def memory_optimize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric columns and converting
    object columns to category type when appropriate.
    
    :param df: Input DataFrame.
    :return: DataFrame with optimized memory usage.
    """
    logger.info("Optimizing memory usage")
    
    # Downcast numerical columns to more efficient types.
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # If the column is a float, downcast to a smaller float type.
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            # For integer columns, downcast based on the range of values.
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
    
    # Convert object (string) columns to category if the number of unique values is low.
    obj_cols = df.select_dtypes(include='object').columns
    for col in obj_cols:
        # Check if the ratio of unique values to total values is below 50%.
        if df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    
    return df

def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns in the DataFrame to appropriate data types.
    This example focuses on converting datetime-like columns using pandas.to_datetime.
    
    :param df: Input DataFrame.
    :return: DataFrame with converted data types.
    """
    # Select columns that are already of datetime type (if any).
    date_cols = df.select_dtypes(include='datetime').columns
    for col in date_cols:
        # Convert each column to datetime to ensure consistency.
        df[col] = pd.to_datetime(df[col])
    
    return df
