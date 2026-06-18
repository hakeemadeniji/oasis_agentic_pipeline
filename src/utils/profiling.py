"""
Performance profiling utilities for OASIS Agentic Pipeline.

Provides tools for profiling, benchmarking, and identifying performance bottlenecks
in the diagnostic pipeline and API operations.
"""

import time
import functools
import logging
import cProfile
import pstats
import io
import sys
import os
from typing import Callable, Any, Dict, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    memory_usage: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def update(self, execution_time: float):
        """Update metrics with new execution time."""
        self.call_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time if self.min_time != float("inf") else 0.0,
            "max_time": self.max_time,
            "memory_usage": self.memory_usage,
            "timestamp": self.timestamp,
        }


class PerformanceProfiler:
    """Performance profiler for monitoring function execution times."""

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.enabled = True

    def profile(self, func: Callable) -> Callable:
        """
        Decorator to profile function execution.

        Args:
            func: Function to profile

        Returns:
            Wrapped function with profiling
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                func_name = func.__name__
                if func_name not in self.metrics:
                    self.metrics[func_name] = PerformanceMetrics(function_name=func_name)

                self.metrics[func_name].update(execution_time)

        return wrapper

    def get_metrics(self, function_name: Optional[str] = None) -> Dict[str, PerformanceMetrics]:
        """
        Get performance metrics.

        Args:
            function_name: Specific function name (optional)

        Returns:
            Dictionary of metrics
        """
        if function_name:
            return {function_name: self.metrics.get(function_name)}
        return self.metrics

    def print_summary(self):
        """Print performance summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE PROFILING SUMMARY")
        print("=" * 60)

        if not self.metrics:
            print("No performance data collected.")
            return

        # Sort by total time
        sorted_metrics = sorted(self.metrics.items(), key=lambda x: x[1].total_time, reverse=True)

        print(f"{'Function':<30} {'Calls':<8} {'Total (s)':<12} {'Avg (ms)':<12} {'Max (ms)':<12}")
        print("-" * 60)

        for func_name, metrics in sorted_metrics:
            print(
                f"{func_name:<30} "
                f"{metrics.call_count:<8} "
                f"{metrics.total_time:<12.4f} "
                f"{metrics.avg_time * 1000:<12.2f} "
                f"{metrics.max_time * 1000:<12.2f}"
            )

        print("=" * 60)

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


# Global profiler instance
profiler = PerformanceProfiler()


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution using global profiler.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with profiling

    Example:
        @profile_function
        def my_function():
            # Function code
            pass
    """
    return profiler.profile(func)


@contextmanager
def profile_context(name: str):
    """
    Context manager for profiling code blocks.

    Args:
        name: Name for the profiling block

    Example:
        with profile_context("image_processing"):
            # Code to profile
            pass
    """
    if not profiler.enabled:
        yield
        return

    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if name not in profiler.metrics:
            profiler.metrics[name] = PerformanceMetrics(function_name=name)

        profiler.metrics[name].update(execution_time)


class DetailedProfiler:
    """Detailed profiler using cProfile for deep analysis."""

    @staticmethod
    def profile_function(func: Callable, *args, **kwargs) -> Any:
        """
        Profile a function with detailed cProfile output.

        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        profiler_obj = cProfile.Profile()
        profiler_obj.enable()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler_obj.disable()

            # Print statistics
            s = io.StringIO()
            ps = pstats.Stats(profiler_obj, stream=s).sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions
            print(s.getvalue())

    @staticmethod
    def profile_to_file(func: Callable, filename: str, *args, **kwargs) -> Any:
        """
        Profile a function and save results to file.

        Args:
            func: Function to profile
            filename: Output filename
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        profiler_obj = cProfile.Profile()
        profiler_obj.enable()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler_obj.disable()

            # Save statistics to file
            profiler_obj.dump_stats(filename)
            logger.info(f"Profile data saved to {filename}")


class MemoryProfiler:
    """Memory usage profiler for detecting memory leaks and optimization."""

    @staticmethod
    def get_memory_usage() -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            logger.warning("psutil not available, memory profiling disabled")
            return 0.0

    @staticmethod
    @contextmanager
    def profile_memory(name: str):
        """
        Context manager for profiling memory usage.

        Args:
            name: Name for the profiling block

        Example:
            with MemoryProfiler.profile_memory("image_loading"):
                # Code to profile
                pass
        """
        start_memory = MemoryProfiler.get_memory_usage()

        try:
            yield
        finally:
            end_memory = MemoryProfiler.get_memory_usage()
            memory_delta = end_memory - start_memory

            logger.info(
                f"Memory profile [{name}]: "
                f"Start: {start_memory:.2f}MB, "
                f"End: {end_memory:.2f}MB, "
                f"Delta: {memory_delta:.2f}MB"
            )


class BatchProcessor:
    """Optimized batch processor for handling multiple items efficiently."""

    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items to process in each batch
            max_workers: Maximum number of parallel workers
        """
        self.batch_size = batch_size
        self.max_workers = max_workers

    def process_batch(
        self, items: List[Any], process_func: Callable, use_parallel: bool = False
    ) -> List[Any]:
        """
        Process items in batches for better performance.

        Args:
            items: Items to process
            process_func: Function to process each item
            use_parallel: Whether to use parallel processing

        Returns:
            List of processed results
        """
        results = []

        if use_parallel and self.max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_func, item): item for item in items}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing item: {e}")
        else:
            # Sequential processing
            for item in items:
                try:
                    result = process_func(item)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")

        return results

    def process_in_batches(
        self,
        items: List[Any],
        process_func: Callable,
        batch_process_func: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Process items in batches with optional batch-level processing.

        Args:
            items: Items to process
            process_func: Function to process each item
            batch_process_func: Optional function to process entire batch

        Returns:
            List of processed results
        """
        results = []

        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]

            if batch_process_func:
                # Process entire batch
                batch_results = batch_process_func(batch)
                results.extend(batch_results)
            else:
                # Process items in batch individually
                for item in batch:
                    try:
                        result = process_func(item)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing item: {e}")

        return results


class CacheManager:
    """Simple in-memory cache manager for performance optimization."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of items in cache
            ttl: Time-to-live in seconds
        """
        self.cache: Dict[str, tuple] = {}  # key: (value, timestamp)
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            return None

        value, timestamp = self.cache[key]

        # Check if expired
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None

        return value

    def set(self, key: str, value: Any):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove oldest item if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (value, time.time())

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "utilization": len(self.cache) / self.max_size,
        }


# Global cache instance
cache_manager = CacheManager()


def cached(ttl: int = 3600):
    """
    Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds

    Returns:
        Decorated function with caching

    Example:
        @cached(ttl=1800)
        def expensive_function(arg1, arg2):
            # Expensive computation
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}_{str(args)}_{str(kwargs)}"

            # Check cache
            cached_value = cache_manager.get(key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache_manager.set(key, result)

            return result

        return wrapper

    return decorator


class PerformanceMonitor:
    """Real-time performance monitoring for production systems."""

    def __init__(self):
        self.alert_thresholds = {
            "function_time": 5.0,  # seconds
            "memory_usage": 1000.0,  # MB
            "error_rate": 0.1,  # 10%
        }
        self.error_count = 0
        self.total_requests = 0

    def monitor_function(self, func: Callable) -> Callable:
        """
        Decorator to monitor function performance with alerts.

        Args:
            func: Function to monitor

        Returns:
            Wrapped function with monitoring
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.total_requests += 1
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.error_count += 1
                raise e
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Check for performance issues
                if execution_time > self.alert_thresholds["function_time"]:
                    logger.warning(
                        f"Performance alert: {func.__name__} took {execution_time:.2f}s "
                        f"(threshold: {self.alert_thresholds['function_time']}s)"
                    )

                # Check memory usage
                memory_usage = MemoryProfiler.get_memory_usage()
                if memory_usage > self.alert_thresholds["memory_usage"]:
                    logger.warning(
                        f"Memory alert: Current usage {memory_usage:.2f}MB "
                        f"(threshold: {self.alert_thresholds['memory_usage']}MB)"
                    )

        return wrapper

    def get_error_rate(self) -> float:
        """Calculate current error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    def reset(self):
        """Reset monitoring statistics."""
        self.error_count = 0
        self.total_requests = 0


# Global performance monitor
performance_monitor = PerformanceMonitor()


if __name__ == "__main__":
    # Test profiling utilities
    print("Testing performance profiling utilities...")

    @profile_function
    def test_function(n: int = 1000000):
        """Test function for profiling."""
        total = 0
        for i in range(n):
            total += i
        return total

    # Run profiled function
    result = test_function(1000000)
    print(f"Result: {result}")

    # Print summary
    profiler.print_summary()

    # Test cache manager
    print("\nTesting cache manager...")
    cache = CacheManager(max_size=10, ttl=60)

    @cached(ttl=60)
    def expensive_computation(x: int):
        """Expensive computation for caching test."""
        time.sleep(0.1)  # Simulate expensive operation
        return x * x

    # First call (not cached)
    start = time.time()
    result1 = expensive_computation(5)
    time1 = time.time() - start

    # Second call (cached)
    start = time.time()
    result2 = expensive_computation(5)
    time2 = time.time() - start

    print(f"First call: {time1:.4f}s, Second call: {time2:.4f}s")
    print(f"Cache speedup: {time1 / time2:.2f}x")

    print("\nPerformance profiling utilities test complete!")
