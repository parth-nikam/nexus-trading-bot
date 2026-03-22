"""
utils/retry.py — Exponential backoff retry decorator for async functions.

Usage:
    @retry(max_attempts=3, base_delay=1.0, exceptions=(aiohttp.ClientError,))
    async def fetch_data():
        ...
"""

from __future__ import annotations
import asyncio
import functools
import random
from typing import Type
from utils.logger import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
):
    """
    Exponential backoff with optional jitter.
    delay = min(base_delay * 2^attempt + jitter, max_delay)
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts - 1:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)
                    logger.warning(
                        f"[retry] {fn.__name__} attempt {attempt+1}/{max_attempts} "
                        f"failed: {e} — retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
            raise last_exc
        return wrapper
    return decorator
