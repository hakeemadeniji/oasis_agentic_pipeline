"""
Asynchronous processing system for long-running tasks in OASIS Agentic Pipeline.

Implements background task processing, job queues, and async workflows
to improve API responsiveness and handle resource-intensive operations.
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time,
        }


class AsyncTask:
    """Asynchronous task wrapper."""

    def __init__(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        callback: Optional[Callable] = None,
    ):
        """
        Initialize async task.

        Args:
            task_id: Unique task identifier
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            callback: Callback function on completion
        """
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.callback = callback

        self.status = TaskStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0

        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    async def execute(self):
        """Execute the task asynchronously."""
        try:
            self.status = TaskStatus.RUNNING
            self.started_at = datetime.now()

            # Check if function is async
            if asyncio.iscoroutinefunction(self.func):
                self.result = await self.func(*self.args, **self.kwargs)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                self.result = await loop.run_in_executor(
                    None, lambda: self.func(*self.args, **self.kwargs)
                )

            self.status = TaskStatus.COMPLETED
            self.completed_at = datetime.now()

            # Call callback if provided
            if self.callback:
                await self._run_callback()

        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = str(e)
            self.completed_at = datetime.now()
            logger.error(f"Task {self.task_id} failed: {e}")

    async def _run_callback(self):
        """Run callback function."""
        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(self.result)
            else:
                self.callback(self.result)
        except Exception as e:
            logger.error(f"Callback failed for task {self.task_id}: {e}")

    def get_result(self) -> TaskResult:
        """Get task result."""
        execution_time = None
        if self.started_at and self.completed_at:
            execution_time = (self.completed_at - self.started_at).total_seconds()

        return TaskResult(
            task_id=self.task_id,
            status=self.status,
            result=self.result,
            error=self.error,
            progress=self.progress,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            execution_time=execution_time,
        )


class TaskQueue:
    """Thread-safe task queue for background processing."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize task queue.

        Args:
            max_size: Maximum queue size
        """
        self.queue = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
        self.size = 0

    def put(self, task: AsyncTask, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Add task to queue.

        Args:
            task: Task to add
            block: Block if queue is full
            timeout: Timeout for blocking

        Returns:
            True if successful
        """
        try:
            self.queue.put(task, block=block, timeout=timeout)
            with self.lock:
                self.size += 1
            return True
        except queue.Full:
            logger.warning("Task queue is full")
            return False

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[AsyncTask]:
        """
        Get task from queue.

        Args:
            block: Block if queue is empty
            timeout: Timeout for blocking

        Returns:
            Task or None
        """
        try:
            task = self.queue.get(block=block, timeout=timeout)
            with self.lock:
                self.size -= 1
            return task
        except queue.Empty:
            return None

    def qsize(self) -> int:
        """Get queue size."""
        return self.queue.qsize()


class AsyncTaskProcessor:
    """
    Asynchronous task processor with thread pool and job queue.

    Handles long-running tasks in background to improve API responsiveness.
    """

    def __init__(
        self, max_workers: int = 4, max_queue_size: int = 1000, use_process_pool: bool = False
    ):
        """
        Initialize async task processor.

        Args:
            max_workers: Maximum number of worker threads/processes
            max_queue_size: Maximum queue size
            use_process_pool: Use process pool instead of thread pool
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.use_process_pool = use_process_pool

        self.tasks: Dict[str, AsyncTask] = {}
        self.task_queue = TaskQueue(max_size=max_queue_size)

        # Initialize executor
        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        logger.info(
            f"Async task processor initialized - "
            f"Workers: {max_workers}, "
            f"Queue: {max_queue_size}, "
            f"Type: {'process' if use_process_pool else 'thread'}"
        )

    def start(self):
        """Start the task processor."""
        if self.running:
            logger.warning("Task processor already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("Task processor started")

    def stop(self):
        """Stop the task processor."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Task processor stopped")

    def _process_queue(self):
        """Process tasks from queue (runs in worker thread)."""
        while self.running:
            task = self.task_queue.get(block=True, timeout=1)
            if task is None:
                continue

            # Submit task to executor
            future = self.executor.submit(self._execute_task_sync, task)

            # Store future for result retrieval
            task.future = future

    def _execute_task_sync(self, task: AsyncTask):
        """Execute task synchronously (called by executor)."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

            # Execute function
            task.result = task.func(*task.args, **task.kwargs)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # Call callback if provided
            if task.callback:
                try:
                    task.callback(task.result)
                except Exception as e:
                    logger.error(f"Callback failed: {e}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error(f"Task execution failed: {e}")

    def submit_task(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        callback: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Submit task for background processing.

        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            callback: Callback function
            task_id: Optional task ID (auto-generated if not provided)

        Returns:
            Task ID
        """
        task_id = task_id or str(uuid.uuid4())

        task = AsyncTask(task_id=task_id, func=func, args=args, kwargs=kwargs, callback=callback)

        # Store task
        self.tasks[task_id] = task

        # Add to queue
        success = self.task_queue.put(task)
        if not success:
            logger.error(f"Failed to add task {task_id} to queue")
            del self.tasks[task_id]
            raise Exception("Task queue is full")

        logger.info(f"Task {task_id} submitted for background processing")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """
        Get status of a task.

        Args:
            task_id: Task ID

        Returns:
            Task result or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        return task.get_result()

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get result of a task (blocks if not complete).

        Args:
            task_id: Task ID
            timeout: Timeout in seconds

        Returns:
            Task result

        Raises:
            Exception: If task failed or timeout
        """
        task = self.tasks.get(task_id)
        if task is None:
            raise Exception(f"Task {task_id} not found")

        # Wait for completion
        if hasattr(task, "future") and task.future:
            try:
                task.future.result(timeout=timeout)
            except Exception as e:
                raise Exception(f"Task execution failed: {e}")

        if task.status == TaskStatus.FAILED:
            raise Exception(task.error or "Task failed")

        if task.status != TaskStatus.COMPLETED:
            raise Exception(f"Task not complete (status: {task.status})")

        return task.result

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            True if successful
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False

        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            return True

        return False

    def cleanup_completed_tasks(self, max_age_seconds: int = 3600):
        """
        Clean up completed tasks older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds
        """
        current_time = datetime.now()
        to_remove = []

        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task.completed_at:
                    age = (current_time - task.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(task_id)

        for task_id in to_remove:
            del self.tasks[task_id]

        logger.info(f"Cleaned up {len(to_remove)} completed tasks")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary with statistics
        """
        status_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "running": self.running,
            "total_tasks": len(self.tasks),
            "queue_size": self.task_queue.qsize(),
            "status_counts": status_counts,
            "max_workers": self.max_workers,
            "max_queue_size": self.max_queue_size,
        }


class BatchProcessor:
    """Optimized batch processor for parallel execution."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_batch(
        self, items: List[Any], process_func: Callable, show_progress: bool = False
    ) -> List[Any]:
        """
        Process batch of items in parallel.

        Args:
            items: Items to process
            process_func: Function to process each item
            show_progress: Show progress indicator

        Returns:
            List of results
        """
        futures = []
        results = []

        # Submit all tasks
        for item in items:
            future = self.executor.submit(process_func, item)
            futures.append(future)

        # Collect results
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)

                if show_progress and (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(items)} items")

            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                results.append(None)

        return results

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


# Global task processor instance
task_processor = AsyncTaskProcessor(max_workers=4)


def async_task(func: Optional[Callable] = None, callback: Optional[Callable] = None):
    """
    Decorator to run function as background task.

    Args:
        func: Function to run as task
        callback: Callback function

    Returns:
        Decorated function or task ID

    Example:
        @async_task
        def long_running_function():
            time.sleep(10)
            return "completed"

        task_id = long_running_function()  # Returns immediately with task ID
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Submit as background task
            task_id = task_processor.submit_task(
                func=f, args=args, kwargs=kwargs, callback=callback
            )
            return task_id

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# Import wraps
from functools import wraps


if __name__ == "__main__":
    # Test async processing
    print("Testing asynchronous processing...")

    # Start task processor
    processor = AsyncTaskProcessor(max_workers=2)
    processor.start()

    # Submit tasks
    def long_task(n: int):
        time.sleep(1)
        return n * n

    task_id1 = processor.submit_task(long_task, args=(5,))
    task_id2 = processor.submit_task(long_task, args=(10,))

    print(f"Submitted tasks: {task_id1}, {task_id2}")

    # Wait for completion
    time.sleep(3)

    # Check status
    status1 = processor.get_task_status(task_id1)
    status2 = processor.get_task_status(task_id2)

    print(f"Task 1 status: {status1.status.value}")
    print(f"Task 2 status: {status2.status.value}")

    if status1.status == TaskStatus.COMPLETED:
        print(f"Task 1 result: {status1.result}")

    # Get stats
    stats = processor.get_stats()
    print(f"Processor stats: {stats}")

    # Stop processor
    processor.stop()

    # Test batch processor
    print("\nTesting batch processor...")
    batch_processor = BatchProcessor(max_workers=4)

    def process_item(item):
        time.sleep(0.1)
        return item * 2

    items = list(range(10))
    results = batch_processor.process_batch(items, process_item)
    print(f"Batch results: {results}")

    batch_processor.shutdown()

    print("\nAsynchronous processing test complete!")
