import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from core.cleaner import DataCleaner
from core.validator import DataValidator

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [25, 30, np.nan, 45, 150],
        'salary': [50000, 60000, 75000, np.nan, 100000],
        'department': ['IT', 'HR', 'IT', np.nan, 'Sales'],
        'hire_date': ['2020-01-01', '2021-02-15', 'invalid', '2022-03-20', '2023-04-01']
    })

def test_missing_value_handling(sample_data):
    cleaner = DataCleaner()
    cleaned = cleaner._handle_missing_values(sample_data)
    assert cleaned.isnull().sum().sum() == 0

def test_outlier_handling(sample_data):
    cleaner = DataCleaner()
    cleaned = cleaner._handle_outliers(sample_data)
    assert cleaned['age'].max() <= 45 * 1.5

def test_data_validation(sample_data):
    validator = DataValidator({
        'validation_rules': {
            'value_ranges': {'age': {'max': 100}},
            'regex_patterns': {'hire_date': r'\d{4}-\d{2}-\d{2}'}
        }
    })
    report = validator.validate(sample_data)
    assert not report['is_valid']
    assert 'age' in str(report['errors'])