import functools
import random
import time
from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
):
    """Retry sync function with exponential backoff and jitter."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt == max_attempts:
                        break
                    delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
                    delay += random.uniform(0, delay * 0.1)  # nosec B311
                    time.sleep(delay)
            raise last_error

        return wrapper  # type: ignore[return-value]

    return decorator
