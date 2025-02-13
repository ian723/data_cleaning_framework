from pathlib import Path
from loguru import logger

class Settings:
    def __init__(self):
        # Directory configurations as instance attributes
        self.INPUT_DIR = Path("data/raw")
        self.OUTPUT_DIR = Path("data/processed")
        self.CACHE_DIR = Path("data/cache")
        self.LOG_DIR = Path("logs")
        
        # Processing configurations
        self.CHUNK_SIZE = 100_000
        self.MAX_MEMORY_USAGE = "16GB"
        self.FILE_FORMAT = "csv"  # Options: csv, parquet, feather
        self.USE_DASK = True
        self.N_WORKERS = 4
        
        # Data cleaning configurations
        self.MISSING_THRESHOLD = 0.7
        self.OUTLIER_METHOD = "isolation_forest"  # Options: zscore, isolation_forest, iqr
        self.TEXT_CLEANING = True
        self.VALIDATE_DATA = True
        
        # Logging configuration
        self.LOG_LEVEL = "INFO"
        self.LOG_RETENTION = "30 days"
        
        # Create required directories and configure logging
        self._create_directories()
        self._configure_logging()
    
    def _create_directories(self):
        self.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _configure_logging(self):
        logger.add(
            self.LOG_DIR / "cleaning_{time}.log",
            level=self.LOG_LEVEL,
            retention=self.LOG_RETENTION,
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

# Create an instance of Settings that can be imported
settings = Settings()
