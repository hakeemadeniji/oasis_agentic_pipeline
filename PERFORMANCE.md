# Performance Optimization Guide

This guide documents the performance optimizations implemented in the OASIS Agentic Pipeline and provides usage examples.

## Overview

Phase 3 implemented comprehensive performance optimizations including:

- **Profiling Tools**: Performance monitoring and bottleneck identification
- **Caching Strategies**: Multi-layered caching with Redis and in-memory fallback
- **Image Processing**: Optimized preprocessing pipeline with GPU acceleration
- **Async Processing**: Background task processing for improved responsiveness
- **Memory Optimization**: Memory-efficient loading and processing

## Performance Profiling

### Using the Performance Profiler

```python
from src.utils.profiling import profile_function, profiler

@profile_function
def my_function():
    # Function code
    pass

# Print performance summary
profiler.print_summary()
```

### Context Manager Profiling

```python
from src.utils.profiling import profile_context

with profile_context("image_processing"):
    # Code to profile
    process_image(image_path)
```

### Detailed Profiling

```python
from src.utils.profiling import DetailedProfiler

result = DetailedProfiler.profile_function(
    expensive_function,
    arg1, arg2
)

# Save profile to file
DetailedProfiler.profile_to_file(
    expensive_function,
    "profile.prof",
    arg1, arg2
)
```

## Caching Strategies

### Multi-Layer Cache

```python
from src.utils.caching import MultiLayerCache, cached

# Initialize cache
cache = MultiLayerCache(
    use_redis=True,  # Enable Redis backend
    redis_config={"host": "localhost", "port": 6379},
    in_memory_config={"max_size": 10000, "default_ttl": 3600}
)

# Use cache manually
cache.set("key", value, ttl=1800)
result = cache.get("key")
```

### Decorator-Based Caching

```python
@cached(ttl=1800, key_prefix="vision_")
def process_image(image_path):
    # Expensive processing
    return result

# First call: executes function
result1 = process_image("image.jpg")

# Second call: returns cached result
result2 = process_image("image.jpg")
```

### Query Caching

```python
from src.utils.caching import query_cache

def get_patient_data(patient_id):
    # Database query
    return db.query(patient_id)

result = query_cache.get_query_result(
    query_type="patient_data",
    query_params={"patient_id": patient_id},
    execute_func=lambda: get_patient_data(patient_id),
    ttl=300
)
```

## Image Processing Optimization

### Optimized Preprocessing

```python
from src.utils.image_preprocessing import OptimizedImagePreprocessor

# Initialize preprocessor
preprocessor = OptimizedImagePreprocessor(
    target_size=(224, 224),
    use_cache=True,
    use_gpu=True
)

# Preprocess single image
tensor = preprocessor.preprocess_image("image.jpg")

# Preprocess batch
batch_tensor = preprocessor.preprocess_batch(
    image_paths,
    batch_size=32
)
```

### Memory-Efficient Loading

```python
from src.utils.image_preprocessing import MemoryEfficientImageLoader

loader = MemoryEfficientImageLoader(max_cache_size=100)

# Load with lazy loading
image = loader.load_image_lazy("large_image.jpg")

# Clear cache when done
loader.clear_cache()
```

### Image Augmentation

```python
from src.utils.image_preprocessing import ImageAugmentation

augmenter = ImageAugmentation(
    rotation_range=10.0,
    brightness_range=(0.9, 1.1),
    contrast_range=(0.9, 1.1)
)

# Augment single image
augmented = augmenter.augment(image)

# Augment batch
augmented_batch = augmenter.augment_batch(images)
```

## Asynchronous Processing

### Background Task Processing

```python
from src.utils.async_processing import task_processor

# Start processor
task_processor.start()

# Submit background task
def long_running_task(patient_id):
    # Long processing
    return diagnosis_result

task_id = task_processor.submit_task(
    func=long_running_task,
    args=("OAS2_0001",)
)

# Check status
status = task_processor.get_task_status(task_id)
print(f"Status: {status.status}")

# Get result (blocks if not complete)
result = task_processor.get_task_result(task_id, timeout=60)
```

### Decorator-Based Background Tasks

```python
from src.utils.async_processing import async_task

@async_task
def process_diagnosis(patient_id):
    # Long-running diagnosis
    return result

# Submit as background task
task_id = process_diagnosis("OAS2_0001")
```

### Batch Processing

```python
from src.utils.async_processing import BatchProcessor

processor = BatchProcessor(max_workers=4)

results = processor.process_batch(
    items=patient_list,
    process_func=process_single_patient,
    show_progress=True
)
```

## Performance Monitoring

### Real-Time Monitoring

```python
from src.utils.profiling import performance_monitor

@performance_monitor.monitor_function
def api_endpoint():
    # API endpoint code
    pass

# Check error rate
error_rate = performance_monitor.get_error_rate()
```

### Memory Profiling

```python
from src.utils.profiling import MemoryProfiler

with MemoryProfiler.profile_memory("image_loading"):
    # Code to profile memory usage
    load_large_dataset()
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Caching Configuration
ENABLE_REDIS_CACHE=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=3600
IN_MEMORY_CACHE_SIZE=10000

# Performance Configuration
MAX_WORKERS=4
BATCH_SIZE=32
ENABLE_GPU=true

# Task Processing
TASK_QUEUE_SIZE=1000
TASK_TIMEOUT=300
```

### Performance Settings

```python
# src/config.py additions
class PerformanceSettings:
    # Caching
    enable_redis_cache: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl: int = 3600
    in_memory_cache_size: int = 10000
    
    # Processing
    max_workers: int = 4
    batch_size: int = 32
    enable_gpu: bool = True
    
    # Task Processing
    task_queue_size: int = 1000
    task_timeout: int = 300
```

## Performance Benchmarks

### Expected Performance Improvements

With optimizations enabled:

- **Image Preprocessing**: 2-3x faster with caching
- **Vision Inference**: 1.5-2x faster with GPU acceleration
- **Database Queries**: 5-10x faster with query caching
- **API Response Time**: 50-70% faster with async processing
- **Memory Usage**: 30-40% reduction with memory-efficient loading

### Benchmarking

```python
import time

# Benchmark image preprocessing
start = time.time()
for _ in range(100):
    preprocessor.preprocess_image("test.jpg")
elapsed = time.time() - start
print(f"Average: {elapsed/100:.4f}s per image")

# Benchmark caching
@cached(ttl=3600)
def expensive_function():
    time.sleep(1)
    return result

# First call (not cached)
start = time.time()
result1 = expensive_function()
time1 = time.time() - start

# Second call (cached)
start = time.time()
result2 = expensive_function()
time2 = time.time() - start

print(f"Speedup: {time1/time2:.2f}x")
```

## Best Practices

### 1. Use Caching Strategically

- Cache expensive operations (database queries, model inference)
- Set appropriate TTL values
- Use cache invalidation for data updates
- Monitor cache hit rates

### 2. Optimize Critical Paths

- Profile before optimizing
- Focus on bottlenecks identified by profiling
- Test performance improvements
- Monitor production performance

### 3. Use Async Processing

- Offload long-running tasks to background
- Use async for I/O-bound operations
- Implement proper error handling
- Monitor task queue sizes

### 4. Memory Management

- Use memory-efficient loading for large datasets
- Clear caches when not needed
- Monitor memory usage in production
- Implement proper cleanup

### 5. Batch Processing

- Process items in batches where possible
- Use parallel processing for independent tasks
- Optimize batch sizes for your hardware
- Monitor resource usage

## Troubleshooting

### Cache Not Working

- Check Redis connection
- Verify cache key generation
- Monitor cache hit rates
- Check TTL settings

### Performance Degradation

- Profile to identify new bottlenecks
- Check memory usage
- Monitor task queue sizes
- Review cache effectiveness

### Async Tasks Failing

- Check task status
- Review error logs
- Verify function signatures
- Monitor executor health

## Monitoring and Metrics

### Key Performance Indicators

- **Cache Hit Rate**: Should be >70%
- **Average Response Time**: <2s for API endpoints
- **Task Queue Size**: <50% of max capacity
- **Memory Usage**: <80% of available memory
- **CPU Usage**: <70% during normal operation

### Monitoring Commands

```python
# Get cache statistics
stats = cache.get_stats()
print(f"Cache utilization: {stats['memory']['utilization']:.2%}")

# Get task processor statistics
stats = task_processor.get_stats()
print(f"Queue size: {stats['queue_size']}")
print(f"Total tasks: {stats['total_tasks']}")

# Get performance summary
profiler.print_summary()
```

## Further Optimization

### Advanced Optimizations

1. **Model Quantization**: Reduce model size and improve inference speed
2. **ONNX Export**: Optimize models for production deployment
3. **Connection Pooling**: Reuse database connections
4. **Lazy Loading**: Load components only when needed
5. **Database Optimization**: Index tuning and query optimization

### Implementation Status

- ✅ Profiling tools
- ✅ Caching strategies
- ✅ Image processing optimization
- ✅ Async processing
- ⏳ Model quantization (planned)
- ⏳ ONNX export (planned)
- ⏳ Connection pooling (planned)
- ⏳ Database optimization (planned)

## Support

For performance issues:
1. Enable profiling to identify bottlenecks
2. Check cache statistics
3. Monitor task processor health
4. Review logs for errors
5. Consult this guide for optimization strategies

The performance optimization infrastructure is now in place and can be extended as needed based on production monitoring and requirements.