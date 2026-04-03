"""
Retry decorator with exponential backoff and jitter.
Handles flaky API calls gracefully.
"""
import asyncio
import functools
import random
import logging
from typing import Callable, Type

logger = logging.getLogger(__name__)

def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True
):
    """
    Decorator for async functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay cap in seconds
        exceptions: Exception types to catch and retry on
        jitter: Add random jitter to prevent thundering herd
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} "
                            f"attempts. Final error: {e}"
                        )
                        raise
                    
                    # Exponential backoff: 1s, 2s, 4s, 8s...
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)  # nosec B311
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        # Support sync functions too
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)  # nosec B311
                    import time
                    time.sleep(delay)
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
