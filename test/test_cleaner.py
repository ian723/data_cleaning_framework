import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from core.cleaner import DataCleaner
from core.validator import DataValidator

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
    cleaner = DataCleaner()  # Initialize a DataCleaner instance.
    cleaned = cleaner._handle_missing_values(sample_data)  # Process the data.
    
    # Assert that there are zero missing values in the entire DataFrame.
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
    cleaner = DataCleaner()  # Initialize a DataCleaner instance.
    cleaned = cleaner._handle_outliers(sample_data)  # Process the data.
    
    # Assert that the maximum age is capped to 45 * 1.5 (example threshold).
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
    # Create a DataValidator with custom validation rules.
    validator = DataValidator({
        'validation_rules': {
            'value_ranges': {'age': {'max': 100}},  # Set maximum allowed age to 100.
            'regex_patterns': {'hire_date': r'\d{4}-\d{2}-\d{2}'}  # Regex pattern for valid date format.
        }
    })
    
    # Validate the sample data.
    report = validator.validate(sample_data)
    
    # Assert that the validation report indicates invalid data.
    assert not report['is_valid']
    # Assert that the error message mentions an issue with 'age'.
    assert 'age' in str(report['errors'])
