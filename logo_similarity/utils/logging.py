import sys
from loguru import logger
from logo_similarity.config.paths import LOGS_DIR

def setup_logging(level="INFO"):
    """Configure loguru logging to console and file."""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Add file handler with rotation
    logger.add(
        LOGS_DIR / "project.log",
        rotation="10 MB",
        retention="10 days",
        level=level,
        compression="zip"
    )
    
    # Performance logging for training/indexing
    logger.add(
        LOGS_DIR / "performance.log",
        filter=lambda record: "performance" in record["extra"],
        format="{time} | {message}",
        level="INFO"
    )

    return logger

# Initialize default logger
setup_logging()
