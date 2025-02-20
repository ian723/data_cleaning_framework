import pandas as pd

class DataValidator:
    def __init__(self, config: dict):
        # Store the configuration dictionary.
        self.config = config
        # Extract the list of required columns; defaults to empty list if not provided.
        self.required_cols = config['validation_rules'].get('required_columns', [])
        # Extract the value ranges for columns; defaults to empty dict if not provided.
        self.value_ranges = config['validation_rules'].get('value_ranges', {})
        # Extract regex patterns for columns; defaults to empty dict if not provided.
        self.regex_patterns = config['validation_rules'].get('regex_patterns', {})
    
    def validate(self, df: pd.DataFrame) -> dict:
        # Initialize a report dictionary with default values.
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 1. Check for missing required columns.
        missing_cols = [col for col in self.required_cols if col not in df.columns]
        if missing_cols:
            report['errors'].append(f"Missing required columns: {missing_cols}")
        
        # 2. Validate that numerical columns are within the specified value ranges.
        for col, rules in self.value_ranges.items():
            if col in df.columns:
                # Check minimum value constraint.
                if 'min' in rules and df[col].min() < rules['min']:
                    report['errors'].append(
                        f"Values in {col} below minimum {rules['min']}"
                    )
                # Check maximum value constraint.
                if 'max' in rules and df[col].max() > rules['max']:
                    report['errors'].append(
                        f"Values in {col} above maximum {rules['max']}"
                    )
        
        # 3. Validate string columns against regex patterns.
        for col, pattern in self.regex_patterns.items():
            if col in df.columns:
                # Convert column to string and test against the regex pattern.
                # The '~' operator inverts the boolean mask (True if does NOT match).
                invalid = ~df[col].astype(str).str.match(pattern)
                if invalid.any():
                    # Count the number of invalid entries.
                    count = invalid.sum()
                    report['errors'].append(
                        f"{count} invalid entries in {col} violating pattern {pattern}"
                    )
        
        # 4. Check for any missing values in the DataFrame and issue warnings.
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                report['warnings'].append(
                    f"{col} has {count} missing values"
                )
        
        # Update overall validity: valid if there are no errors.
        report['is_valid'] = len(report['errors']) == 0
        return report
