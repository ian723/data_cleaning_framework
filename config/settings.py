from pathlib import Path
from loguru import logger

class Settings:
    # Directory configurations
    INPUT_DIR = Path("data/raw")
    OUTPUT_DIR = Path("data/processed")
    CACHE_DIR = Path("data/cache")
    LOG_DIR = Path("logs")
    
    # Processing configurations
    CHUNK_SIZE = 100_000
    MAX_MEMORY_USAGE = "16GB"
    FILE_FORMAT = "parquet"  # csv, parquet, feather
    USE_DASK = True
    N_WORKERS = 4
    
    # Data cleaning configurations
    MISSING_THRESHOLD = 0.7
    OUTLIER_METHOD = "iqr"  # zscore, isolation_forest
    TEXT_CLEANING = True
    VALIDATE_DATA = True
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_RETENTION = "30 days"
    
    def __init__(self):
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

settings = Settings()