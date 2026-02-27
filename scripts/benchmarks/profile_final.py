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
    ctx.bucket = "local"
    
    start_time = time.time()
    ctx.initialize()
    end_time = time.time()
    
    logger.info(f"Initialization took {end_time - start_time:.2f} seconds")
    logger.info(f"End memory: {get_mem():.2f} MB")
    
    # Test search speed
    import numpy as np
    q = np.random.random((1, 256)).astype('float32')
    s = time.time()
    res = ctx.vector_store.search(q, 10)
    logger.info(f"Search took {(time.time()-s)*1000:.2f} ms")

if __name__ == "__main__":
    test_mem()
