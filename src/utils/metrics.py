"""
Performance Monitoring and Metrics Collection for OASIS Agentic Pipeline
Provides Prometheus-compatible metrics and custom performance tracking
"""

import time
import functools
from typing import Callable, Any, Optional, Dict
from datetime import datetime
import threading
from collections import defaultdict

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info
    from prometheus_client import generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Metrics will be collected but not exported.")


class MetricsCollector:
    """Centralized metrics collection system"""
    
    def __init__(self, app_name: str = "oasis_pipeline"):
        self.app_name = app_name
        self._lock = threading.Lock()
        self._custom_metrics: Dict[str, list] = defaultdict(list)
        
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        else:
            self._setup_fallback_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Request metrics
        self.request_count = Counter(
            'oasis_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'oasis_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        # Diagnosis metrics
        self.diagnosis_count = Counter(
            'oasis_diagnoses_total',
            'Total number of diagnoses performed',
            ['diagnosis_class', 'approved']
        )
        
        self.diagnosis_duration = Histogram(
            'oasis_diagnosis_duration_seconds',
            'Diagnosis processing duration in seconds',
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
        )
        
        self.diagnosis_confidence = Histogram(
            'oasis_diagnosis_confidence',
            'Diagnosis confidence scores',
            ['diagnosis_class'],
            buckets=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
        )
        
        # Agent metrics
        self.agent_execution_count = Counter(
            'oasis_agent_executions_total',
            'Total number of agent executions',
            ['agent_name', 'status']
        )
        
        self.agent_execution_duration = Histogram(
            'oasis_agent_execution_duration_seconds',
            'Agent execution duration in seconds',
            ['agent_name'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
        )
        
        # Batch processing metrics
        self.batch_size = Histogram(
            'oasis_batch_size',
            'Number of patients in batch requests',
            buckets=(1, 5, 10, 20, 50, 100)
        )
        
        self.batch_duration = Histogram(
            'oasis_batch_duration_seconds',
            'Batch processing duration in seconds',
            buckets=(10, 30, 60, 120, 300, 600)
        )
        
        # System metrics
        self.active_connections = Gauge(
            'oasis_active_connections',
            'Number of active connections',
            ['connection_type']
        )
        
        self.model_loaded = Gauge(
            'oasis_model_loaded',
            'Whether models are loaded (1=loaded, 0=not loaded)',
            ['model_name']
        )
        
        self.memory_usage = Gauge(
            'oasis_memory_usage_bytes',
            'Memory usage in bytes',
            ['component']
        )
        
        # Error metrics
        self.error_count = Counter(
            'oasis_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # Ethics audit metrics
        self.ethics_audit_count = Counter(
            'oasis_ethics_audits_total',
            'Total number of ethics audits',
            ['result']
        )
        
        # Info metrics
        self.app_info = Info(
            'oasis_app',
            'Application information'
        )
        self.app_info.info({
            'version': '1.0.0',
            'name': self.app_name
        })
    
    def _setup_fallback_metrics(self):
        """Setup fallback metrics when Prometheus is not available"""
        self._metrics = {
            'requests': defaultdict(int),
            'diagnoses': defaultdict(int),
            'agents': defaultdict(int),
            'errors': defaultdict(int)
        }
    
    # Request tracking
    def track_request(self, method: str, endpoint: str, status: int, duration: float):
        """Track API request"""
        if PROMETHEUS_AVAILABLE:
            self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        else:
            key = f"{method}_{endpoint}_{status}"
            self._metrics['requests'][key] += 1
    
    # Diagnosis tracking
    def track_diagnosis(
        self,
        diagnosis_class: str,
        confidence: float,
        approved: bool,
        duration: float
    ):
        """Track diagnosis result"""
        if PROMETHEUS_AVAILABLE:
            self.diagnosis_count.labels(
                diagnosis_class=diagnosis_class,
                approved=str(approved)
            ).inc()
            self.diagnosis_duration.observe(duration)
            self.diagnosis_confidence.labels(diagnosis_class=diagnosis_class).observe(confidence / 100.0)
        else:
            key = f"{diagnosis_class}_{approved}"
            self._metrics['diagnoses'][key] += 1
    
    # Agent tracking
    def track_agent_execution(
        self,
        agent_name: str,
        duration: float,
        status: str = "success"
    ):
        """Track agent execution"""
        if PROMETHEUS_AVAILABLE:
            self.agent_execution_count.labels(agent_name=agent_name, status=status).inc()
            self.agent_execution_duration.labels(agent_name=agent_name).observe(duration)
        else:
            key = f"{agent_name}_{status}"
            self._metrics['agents'][key] += 1
    
    # Batch tracking
    def track_batch(self, batch_size: int, duration: float):
        """Track batch processing"""
        if PROMETHEUS_AVAILABLE:
            self.batch_size.observe(batch_size)
            self.batch_duration.observe(duration)
    
    # Connection tracking
    def set_active_connections(self, connection_type: str, count: int):
        """Set active connection count"""
        if PROMETHEUS_AVAILABLE:
            self.active_connections.labels(connection_type=connection_type).set(count)
    
    # Model status
    def set_model_loaded(self, model_name: str, loaded: bool):
        """Set model loaded status"""
        if PROMETHEUS_AVAILABLE:
            self.model_loaded.labels(model_name=model_name).set(1 if loaded else 0)
    
    # Memory tracking
    def set_memory_usage(self, component: str, bytes_used: int):
        """Set memory usage"""
        if PROMETHEUS_AVAILABLE:
            self.memory_usage.labels(component=component).set(bytes_used)
    
    # Error tracking
    def track_error(self, error_type: str, component: str):
        """Track error occurrence"""
        if PROMETHEUS_AVAILABLE:
            self.error_count.labels(error_type=error_type, component=component).inc()
        else:
            key = f"{error_type}_{component}"
            self._metrics['errors'][key] += 1
    
    # Ethics audit tracking
    def track_ethics_audit(self, approved: bool):
        """Track ethics audit result"""
        if PROMETHEUS_AVAILABLE:
            result = "approved" if approved else "blocked"
            self.ethics_audit_count.labels(result=result).inc()
    
    # Custom metrics
    def record_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record custom metric"""
        with self._lock:
            self._custom_metrics[name].append({
                'value': value,
                'labels': labels or {},
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(REGISTRY).decode('utf-8')
        else:
            # Return simple text format
            lines = ["# Fallback metrics (Prometheus not available)\n"]
            for category, metrics in self._metrics.items():
                lines.append(f"\n# {category}\n")
                for key, value in metrics.items():
                    lines.append(f"{category}_{key} {value}\n")
            return ''.join(lines)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Decorator for timing functions
def track_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track function execution time"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                collector = get_metrics_collector()
                collector.record_custom_metric(
                    f"{metric_name}_duration_seconds",
                    duration,
                    labels
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                collector = get_metrics_collector()
                collector.track_error(type(e).__name__, func.__name__)
                raise
        return wrapper
    return decorator


# Context manager for timing code blocks
class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        
        collector = get_metrics_collector()
        collector.record_custom_metric(
            f"{self.name}_duration_seconds",
            self.duration,
            self.labels
        )
        
        if exc_type is not None:
            collector.track_error(exc_type.__name__, self.name)
        
        return False


# Example usage
if __name__ == "__main__":
    # Initialize metrics collector
    collector = get_metrics_collector()
    
    # Track some metrics
    collector.track_request("POST", "/diagnose", 200, 1.5)
    collector.track_diagnosis("Very Mild Dementia", 87.5, True, 2.3)
    collector.track_agent_execution("VisionAgent", 0.5, "success")
    collector.track_batch(10, 25.5)
    collector.set_model_loaded("ResNet18", True)
    collector.track_ethics_audit(True)
    
    # Use decorator
    @track_time("test_function", {"component": "test"})
    def test_function():
        time.sleep(0.1)
        return "done"
    
    test_function()
    
    # Use context manager
    with Timer("test_block", {"component": "test"}):
        time.sleep(0.2)
    
    # Get metrics
    print("\nCollected Metrics:")
    print(collector.get_metrics())
