import re
import pandas as pd
from loguru import logger

class DataValidator:
    def __init__(self, config: dict):
        self.config = config
        self.required_cols = config['validation_rules'].get('required_columns', [])
        self.value_ranges = config['validation_rules'].get('value_ranges', {})
        self.regex_patterns = config['validation_rules'].get('regex_patterns', {})
    
    def validate(self, df: pd.DataFrame) -> dict:
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        missing_cols = [col for col in self.required_cols if col not in df.columns]
        if missing_cols:
            report['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Validate value ranges
        for col, rules in self.value_ranges.items():
            if col in df.columns:
                if 'min' in rules and df[col].min() < rules['min']:
                    report['errors'].append(
                        f"Values in {col} below minimum {rules['min']}"
                    )
                if 'max' in rules and df[col].max() > rules['max']:
                    report['errors'].append(
                        f"Values in {col} above maximum {rules['max']}"
                    )
        
        # Validate regex patterns
        for col, pattern in self.regex_patterns.items():
            if col in df.columns:
                invalid = ~df[col].astype(str).str.match(pattern)
                if invalid.any():
                    count = invalid.sum()
                    report['errors'].append(
                        f"{count} invalid entries in {col} violating pattern {pattern}"
                    )
        
        # Check for remaining missing values
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                report['warnings'].append(
                    f"{col} has {count} missing values"
                )
        
        report['is_valid'] = len(report['errors']) == 0
        return report