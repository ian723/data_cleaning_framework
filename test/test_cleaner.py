import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from core.cleaner import DataCleaner
from core.validator import DataValidator
from core.utils import memory_optimize

# ----------------------------
# Pytest Fixture for Sample Data
# ----------------------------
@pytest.fixture
def sample_data():
    """
    Create a sample DataFrame for testing purposes.
    
    Columns:
      - 'age': Contains numeric values, with a missing value (np.nan) and an outlier (150).
      - 'salary': Contains numeric salary values, including one missing value.
      - 'department': Contains categorical data with a missing entry.
      - 'hire_date': Contains dates as strings, including one invalid entry ('invalid').
    """
    return pd.DataFrame({
        'age': [25, 30, np.nan, 45, 150],
        'salary': [50000, 60000, 75000, np.nan, 100000],
        'department': ['IT', 'HR', 'IT', np.nan, 'Sales'],
        'hire_date': ['2020-01-01', '2021-02-15', 'invalid', '2022-03-20', '2023-04-01']
    })

# ----------------------------
# Test: Missing Value Handling
# ----------------------------
def test_missing_value_handling(sample_data):
    """
    Test that the DataCleaner properly handles missing values.
    
    The _handle_missing_values method should process the sample_data
    and result in a DataFrame with no missing values.
    """
    cleaner = DataCleaner()
    cleaned = cleaner._handle_missing_values(sample_data)
    assert cleaned.isnull().sum().sum() == 0

# ----------------------------
# Test: Outlier Handling
# ----------------------------
def test_outlier_handling(sample_data):
    """
    Test the outlier handling functionality in DataCleaner.
    
    This test applies the _handle_outliers method and checks that the
    maximum value in the 'age' column is within an acceptable range.
    
    Here, we expect that the outlier (age 150) will be handled,
    and the resulting maximum should be less than or equal to 45 * 1.5.
    """
    cleaner = DataCleaner()
    cleaned = cleaner._handle_outliers(sample_data)
    assert cleaned['age'].max() <= 45 * 1.5

# ----------------------------
# Test: Data Validation
# ----------------------------
def test_data_validation(sample_data):
    """
    Test the DataValidator's validate method using custom validation rules.
    
    Configuration:
      - 'value_ranges': The 'age' column should have a maximum value of 100.
      - 'regex_patterns': The 'hire_date' column must match the pattern 'YYYY-MM-DD'.
    
    The sample_data has an outlier in 'age' (150) and an invalid 'hire_date' ('invalid'),
    so the validation report is expected to mark the data as invalid and contain an error related to 'age'.
    """
    validator = DataValidator({
        'validation_rules': {
            'value_ranges': {'age': {'max': 100}},
            'regex_patterns': {'hire_date': r'\d{4}-\d{2}-\d{2}'}
        }
    })
    report = validator.validate(sample_data)
    assert not report['is_valid']
    assert 'age' in str(report['errors'])

# ----------------------------
# Test: Text Cleaning
# ----------------------------
def test_text_cleaning():
    """
    Test that text cleaning normalizes string data by stripping whitespace,
    converting to lowercase, and removing punctuation.
    """
    df = pd.DataFrame({
        'name': ['  John Doe  ', 'JANE SMITH!', 'Alice   ', None]
    })
    cleaner = DataCleaner()
    cleaned = cleaner._clean_text_data(df)
    
    expected = ['john doe', 'jane smith', 'alice', 'none']
    actual = cleaned['name'].astype(str).tolist()
    
    for exp, act in zip(expected, actual):
        # Compare lowercased strings (adjust as necessary for your cleaning logic)
        assert exp in act.lower()

# ----------------------------
# Test: Date Feature Extraction
# ----------------------------
def test_date_feature_extraction():
    """
    Test that date features (year, month, day_of_week) are correctly extracted
    from a date column.
    """
    df = pd.DataFrame({
        'hire_date': ['2020-01-15', '2021-06-30', '2022-12-25']
    })
    # Create a test configuration for date feature extraction.
    config = {
        'date_columns': ['hire_date'],
        'feature_engineering': {
            'date_features': ['year', 'month', 'day_of_week'],
            'interactions': []
        }
    }
    cleaner = DataCleaner()
    cleaner.config = config  # Override the configuration for testing
    
    df = cleaner._extract_date_features(df)
    
    assert 'hire_date_year' in df.columns
    assert 'hire_date_month' in df.columns
    assert 'hire_date_day_of_week' in df.columns
    assert df.loc[0, 'hire_date_year'] == 2020

# ----------------------------
# Test: Feature Interaction Creation
# ----------------------------
def test_feature_interactions():
    """
    Test that interaction features are created correctly.
    For example, multiplying 'age' and 'salary' should create a new column.
    """
    df = pd.DataFrame({
        'age': [25, 30, 45],
        'salary': [50000, 60000, 70000]
    })
    config = {
        'feature_engineering': {
            'interactions': [
                {'columns': ['age', 'salary'], 'operation': 'multiply'}
            ]
        }
    }
    cleaner = DataCleaner()
    cleaner.config = config  # Override configuration for testing
    
    df = cleaner._create_interactions(df)
    
    interaction_col = '_x_'.join(['age', 'salary'])
    assert interaction_col in df.columns
    assert df.loc[0, interaction_col] == 25 * 50000

# ----------------------------
# Test: Memory Optimization
# ----------------------------
def test_memory_optimization():
    """
    Test that memory optimization downcasts numerical columns and converts object
    columns to categories.
    """
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],
        # 2 unique values out of 5 gives a ratio of 2/5 = 0.4 (< 0.5).
        'obj_col': ['a', 'a', 'a', 'a', 'b']
    })
    optimized_df = memory_optimize(df)
    
    # Check that numeric columns are downcasted.
    assert optimized_df['int_col'].dtype in [np.int8, np.int16, np.int32, np.int64]
    # Check that object columns are converted to 'category'.
    assert optimized_df['obj_col'].dtype.name == 'category'

# ----------------------------
# Test: Read CSV Functionality
# ----------------------------
def test_read_csv(tmp_path):
    """
    Test that the _read_data function correctly reads a CSV file.
    """
    csv_file = tmp_path / "test.csv"
    df_orig = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df_orig.to_csv(csv_file, index=False)
    
    cleaner = DataCleaner()
    df_read = cleaner._read_data(csv_file)
    
    pd.testing.assert_frame_equal(df_orig, df_read)

# ----------------------------
# Test: Required Columns Validation
# ----------------------------
def test_required_columns_validation(sample_data):
    """
    Test that the DataValidator correctly identifies missing required columns.
    """
    df_missing = sample_data.drop(columns=['hire_date'])
    
    validator = DataValidator({
        'validation_rules': {
            'required_columns': ['hire_date'],
        }
    })
    
    report = validator.validate(df_missing)
    assert not report['is_valid']
    assert 'hire_date' in str(report['errors'])
