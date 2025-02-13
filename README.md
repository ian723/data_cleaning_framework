# Enterprise Data Cleaning Framework

The **Enterprise Data Cleaning Framework** is a Python-based solution designed to clean, preprocess, and validate data from various file formats. It leverages a configurable pipeline to handle missing values, outliers, text cleaning, feature engineering, and even generate comprehensive data profile reports.

---

## Features

- **Data Cleaning:**  
  Handle missing values with configurable strategies (median, mean, mode, KNN, or drop columns), outlier detection (winsorization, isolation forest), and text cleaning.
  
- **Data Preprocessing:**  
  Perform categorical encoding, numerical normalization, and handle skewness using logarithmic transformations.
  
- **Data Validation:**  
  Validate data against custom rules (required columns, value ranges, regex patterns).
  
- **Memory Optimization:**  
  Downcast numeric columns and convert object columns to categories to reduce memory usage.
  
- **Configurable Pipeline:**  
  Use a YAML configuration file (`config/cleaning_rules.yaml`) to control cleaning rules and processing parameters.
  
- **Command-Line Interface:**  
  Easily process single files or entire directories with options for parallel processing (via Dask) and data profiling (via ydata_profiling).
  
- **Logging:**  
  Built-in logging using [Loguru](https://github.com/Delgan/loguru) to track processing steps and errors.

---

## Installation
```bash
git clone https://github.com/ian723/data_cleaning_framework.git
pip install -r requirements.txt


## Running Tests

Make sure your virtual environment is activated, then run:

```bash
pytest




Usage
Running the Data Cleaning Pipeline
You can run the data cleaning pipeline using the CLI. For example, to process a single file:
```bash
python main.py data/raw/mydata.csv -o data/processed/ --dask --profile

CLI Options
input: Path to an input fil
e or directory.
-o/--output: Directory to save cleaned data (default is defined in settings).
-c/--config: Path to the YAML configuration file (default: config/cleaning_rules.yaml).
--dask: Enable parallel processing using Dask.
--profile: Generate a data profile report using ydata_profiling.



Configuration
The cleaning rules and processing parameters are defined in the YAML configuration file located at config/cleaning_rules.yaml.