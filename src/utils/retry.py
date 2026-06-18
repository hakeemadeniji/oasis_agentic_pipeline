"""
Retry logic utilities for external service calls.

Implements exponential backoff with jitter for resilient external service communication,
circuit breaker pattern for failing services, and configurable retry strategies.
"""

import asyncio
import time
import random
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    # Exceptions to retry on
    retry_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    # Exceptions to never retry on
    no_retry_exceptions: Tuple[Type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
    )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter."""
        delay = min(self.initial_delay * (self.exponential_base ** (attempt - 1)), self.max_delay)

        if self.jitter:
            jitter_amount = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for failing external services.

    Prevents cascading failures by stopping calls to a failing service after
    a threshold of failures, allowing it time to recover.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                logger.info("Circuit breaker attempting reset")
            else:
                raise Exception(f"Circuit breaker is open for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful function call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed function call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def retry_with_backoff(
    config: Optional[RetryConfig] = None, circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        config: Retry configuration (uses default if not provided)
        circuit_breaker: Optional circuit breaker for failure protection

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_attempts=3, initial_delay=1.0)
        def external_api_call():
            # API call that may fail
            pass
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Apply circuit breaker if provided
            if circuit_breaker is not None:
                return circuit_breaker.call(func, *args, **kwargs)

            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.no_retry_exceptions as e:
                    # Don't retry on these exceptions
                    logger.error(
                        f"Function {func.__name__} raised non-retryable exception: {type(e).__name__}"
                    )
                    raise
                except config.retry_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {attempt} attempts: {str(e)}"
                        )
                        raise

                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Function {func.__name__} attempt {attempt} failed: {str(e)}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                except Exception as e:
                    # Unexpected exception - don't retry
                    logger.error(
                        f"Function {func.__name__} raised unexpected exception: {type(e).__name__}"
                    )
                    raise

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def async_retry_with_backoff(
    config: Optional[RetryConfig] = None, circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    Async decorator for retrying async function calls with exponential backoff.

    Args:
        config: Retry configuration (uses default if not provided)
        circuit_breaker: Optional circuit breaker for failure protection

    Returns:
        Decorated async function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Apply circuit breaker if provided
            if circuit_breaker is not None:
                return circuit_breaker.call(func, *args, **kwargs)

            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.no_retry_exceptions as e:
                    logger.error(
                        f"Async function {func.__name__} raised non-retryable exception: {type(e).__name__}"
                    )
                    raise
                except config.retry_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        logger.error(
                            f"Async function {func.__name__} failed after {attempt} attempts: {str(e)}"
                        )
                        raise

                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Async function {func.__name__} attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(
                        f"Async function {func.__name__} raised unexpected exception: {type(e).__name__}"
                    )
                    raise

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


# Predefined retry configurations for common use cases
LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    retry_exceptions=(ConnectionError, TimeoutError, OSError),
)

API_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retry_exceptions=(ConnectionError, TimeoutError, OSError),
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=False,
    retry_exceptions=(ConnectionError, TimeoutError, OSError),
)


# Convenience decorators
def retry_llm_call(func: Callable) -> Callable:
    """Decorator optimized for LLM API calls with longer timeouts."""
    return retry_with_backoff(LLM_RETRY_CONFIG)(func)


def retry_api_call(func: Callable) -> Callable:
    """Decorator optimized for external API calls."""
    return retry_with_backoff(API_RETRY_CONFIG)(func)


def retry_database_call(func: Callable) -> Callable:
    """Decorator optimized for database operations."""
    return retry_with_backoff(DATABASE_RETRY_CONFIG)(func)


if __name__ == "__main__":
    # Example usage and testing
    import asyncio

    @retry_with_backoff(max_attempts=3, initial_delay=0.5)
    def failing_function():
        """Function that fails before succeeding."""
        if random.random() < 0.7:
            raise ConnectionError("Random connection error")
        return "Success!"

    @retry_with_backoff(max_attempts=2, initial_delay=0.1)
    def non_retryable_function():
        """Function with non-retryable exception."""
        raise ValueError("This should not be retried")

    # Test retry logic
    print("Testing retry logic...")
    try:
        result = failing_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Final error: {e}")

    # Test non-retryable exception
    print("\nTesting non-retryable exception...")
    try:
        result = non_retryable_function()
    except ValueError as e:
        print(f"Correctly raised ValueError without retry: {e}")

    # Test circuit breaker
    print("\nTesting circuit breaker...")
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=2.0)

    @retry_with_backoff(max_attempts=1, circuit_breaker=cb)
    def always_failing_function():
        raise ConnectionError("Always fails")

    try:
        for i in range(3):
            always_failing_function()
    except Exception as e:
        print(f"Circuit breaker test completed: {e}")
