"""
utils/circuit_breaker.py — Circuit breaker pattern for exchange calls.

States: CLOSED (normal) → OPEN (halted) → HALF_OPEN (testing recovery)

Usage:
    cb = CircuitBreaker("binance", max_failures=5, cooldown=300)
    async with cb:
        result = await risky_call()
"""

from __future__ import annotations
import asyncio
import time
from enum import Enum
from utils.logger import get_logger

logger = get_logger(__name__)


class CBState(Enum):
    CLOSED    = "CLOSED"     # normal operation
    OPEN      = "OPEN"       # halted — too many failures
    HALF_OPEN = "HALF_OPEN"  # testing if service recovered


class CircuitBreakerOpen(Exception):
    pass


class CircuitBreaker:
    def __init__(self, name: str, max_failures: int = 5, cooldown: float = 300):
        self.name         = name
        self.max_failures = max_failures
        self.cooldown     = cooldown
        self._failures    = 0
        self._state       = CBState.CLOSED
        self._opened_at   = 0.0
        self._lock        = asyncio.Lock()

    @property
    def state(self) -> CBState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CBState.OPEN

    async def __aenter__(self):
        async with self._lock:
            if self._state == CBState.OPEN:
                elapsed = time.time() - self._opened_at
                if elapsed >= self.cooldown:
                    self._state = CBState.HALF_OPEN
                    logger.info(f"[CB:{self.name}] HALF_OPEN — testing recovery")
                else:
                    raise CircuitBreakerOpen(
                        f"[CB:{self.name}] OPEN — {self.cooldown - elapsed:.0f}s remaining"
                    )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            if exc_type is None:
                # Success
                if self._state == CBState.HALF_OPEN:
                    logger.info(f"[CB:{self.name}] CLOSED — recovered")
                self._failures = 0
                self._state    = CBState.CLOSED
            elif exc_type is not CircuitBreakerOpen:
                # Failure
                self._failures += 1
                if self._failures >= self.max_failures:
                    self._state     = CBState.OPEN
                    self._opened_at = time.time()
                    logger.error(
                        f"[CB:{self.name}] OPEN after {self._failures} failures — "
                        f"cooldown {self.cooldown}s"
                    )
                else:
                    logger.warning(
                        f"[CB:{self.name}] failure {self._failures}/{self.max_failures}"
                    )
        return False  # don't suppress exceptions

    def reset(self):
        self._failures  = 0
        self._state     = CBState.CLOSED
        self._opened_at = 0.0

    def status(self) -> dict:
        return {
            "name":     self.name,
            "state":    self._state.value,
            "failures": self._failures,
            "cooldown_remaining": max(0, self.cooldown - (time.time() - self._opened_at))
            if self._state == CBState.OPEN else 0,
        }
