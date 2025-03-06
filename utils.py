
# General utility functions that can be called from any script

import cv2
import os
import time
from functools import wraps 


def timeit():
    stats = {"count": 0, "total_time": 0}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            duration = end_time - start_time
            stats["count"] += 1
            stats["total_time"] += duration

            print(f"\nFunction '{func.__name__}' Call #{stats['count']}: {duration:.6f}s")
            print(f"Total Execution Time: {stats['total_time']:.6f}s\n")

            return result
        return wrapper
    return decorator