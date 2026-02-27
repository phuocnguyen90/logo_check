import os
import sys
import psutil
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from logo_similarity.api.app import APIContext
from logo_similarity.utils.logging import logger

def get_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_mem():
    logger.info(f"Start memory: {get_mem():.2f} MB")
    
    ctx = APIContext()
    # Mocking bucket and download to use local files
    ctx.bucket = "local"
    
    # We need to make sure paths are correct. 
    # APIContext.initialize uses paths.MODELS_DIR
    
    start_time = time.time()
    ctx.initialize()
    end_time = time.time()
    
    logger.info(f"Initialization took {end_time - start_time:.2f} seconds")
    logger.info(f"End memory: {get_mem():.2f} MB")
    
    # Check if id_map is the culprit
    if ctx.id_map:
        import sys
        id_map_mem = sys.getsizeof(ctx.id_map) / (1024 * 1024)
        logger.info(f"id_map sys.getsizeof: {id_map_mem:.2f} MB")
        
        # Actual size is harder to compute in Python, let's just estimate
        # by deleting it and gc-ing
        import gc
        del ctx.id_map
        gc.collect()
        logger.info(f"Memory after deleting id_map: {get_mem():.2f} MB")

if __name__ == "__main__":
    test_mem()
