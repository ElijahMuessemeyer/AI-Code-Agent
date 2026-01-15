"""
Performance optimization module for AI Code Agent.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc

from src.performance.cache_manager import get_cache_manager


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    request_count: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    errors: int = 0
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / max(1, self.request_count)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total)
    
    def update_response_time(self, response_time: float):
        """Update response time metrics."""
        self.request_count += 1
        self.total_response_time += response_time
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_count": self.request_count,
            "average_response_time": self.average_response_time,
            "min_response_time": self.min_response_time if self.min_response_time != float('inf') else 0,
            "max_response_time": self.max_response_time,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "active_connections": self.active_connections,
            "errors": self.errors
        }


@dataclass
class BatchRequest:
    """Represents a batched request."""
    id: str
    function: Callable
    args: tuple
    kwargs: dict
    future: asyncio.Future
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 1


class RequestBatcher:
    """Batches similar requests for efficient processing."""
    
    def __init__(self, batch_size: int = 10, max_wait_time: float = 0.5):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batches: Dict[str, List[BatchRequest]] = defaultdict(list)
        self.processing = False
        self.batch_processors: Dict[str, Callable] = {}
    
    def register_batch_processor(self, operation_type: str, processor: Callable):
        """Register a batch processor for a specific operation type."""
        self.batch_processors[operation_type] = processor
    
    async def add_request(self, operation_type: str, function: Callable, *args, **kwargs) -> Any:
        """Add a request to the batch."""
        future = asyncio.Future()
        request = BatchRequest(
            id=f"{operation_type}_{len(self.batches[operation_type])}",
            function=function,
            args=args,
            kwargs=kwargs,
            future=future
        )
        
        self.batches[operation_type].append(request)
        
        # Start processing if batch is full or this is the first request
        if len(self.batches[operation_type]) >= self.batch_size or not self.processing:
            asyncio.create_task(self._process_batches())
        
        return await future
    
    async def _process_batches(self):
        """Process accumulated batches."""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            # Wait for max_wait_time to accumulate more requests
            await asyncio.sleep(self.max_wait_time)
            
            # Process each operation type
            for operation_type, requests in list(self.batches.items()):
                if not requests:
                    continue
                
                # Remove requests from queue
                batch_requests = requests[:self.batch_size]
                self.batches[operation_type] = requests[self.batch_size:]
                
                # Process batch
                if operation_type in self.batch_processors:
                    await self._process_batch_with_processor(operation_type, batch_requests)
                else:
                    await self._process_batch_individually(batch_requests)
        
        finally:
            self.processing = False
            
            # Check if more batches need processing
            if any(self.batches.values()):
                asyncio.create_task(self._process_batches())
    
    async def _process_batch_with_processor(self, operation_type: str, requests: List[BatchRequest]):
        """Process batch using registered processor."""
        try:
            processor = self.batch_processors[operation_type]
            
            # Extract arguments for batch processing
            batch_args = [req.args for req in requests]
            batch_kwargs = [req.kwargs for req in requests]
            
            # Process entire batch
            results = await processor(batch_args, batch_kwargs)
            
            # Set results for individual futures
            for request, result in zip(requests, results):
                if not request.future.done():
                    request.future.set_result(result)
        
        except Exception as e:
            # Set exception for all futures in batch
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _process_batch_individually(self, requests: List[BatchRequest]):
        """Process batch by calling individual functions."""
        tasks = []
        
        for request in requests:
            if asyncio.iscoroutinefunction(request.function):
                task = request.function(*request.args, **request.kwargs)
            else:
                task = asyncio.get_event_loop().run_in_executor(
                    None, request.function, *request.args, **request.kwargs
                )
            tasks.append((request, task))
        
        # Wait for all tasks to complete
        for request, task in tasks:
            try:
                result = await task
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as e:
                if not request.future.done():
                    request.future.set_exception(e)


class ResourceMonitor:
    """Monitors system resources and adjusts performance parameters."""
    
    def __init__(self, monitoring_interval: float = 30.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.metrics_history: deque = deque(maxlen=100)
        self.callbacks: List[Callable] = []
        
        # Thresholds for auto-scaling
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.response_time_threshold = 5.0
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for resource threshold events."""
        self.callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start resource monitoring."""
        self.is_monitoring = True
        
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds and trigger callbacks
                await self._check_thresholds(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
            
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        process = psutil.Process()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": process.memory_info().rss / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "active_threads": threading.active_count(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
    
    async def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check if metrics exceed thresholds."""
        alerts = []
        
        if metrics["cpu_percent"] > self.cpu_threshold:
            alerts.append({
                "type": "cpu_high",
                "value": metrics["cpu_percent"],
                "threshold": self.cpu_threshold,
                "recommendation": "Consider scaling horizontally or optimizing CPU usage"
            })
        
        if metrics["memory_percent"] > self.memory_threshold:
            alerts.append({
                "type": "memory_high", 
                "value": metrics["memory_percent"],
                "threshold": self.memory_threshold,
                "recommendation": "Consider increasing memory or implementing memory optimization"
            })
        
        # Trigger callbacks for alerts
        for alert in alerts:
            for callback in self.callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        cpu_values = [m["cpu_percent"] for m in recent_metrics]
        memory_values = [m["memory_percent"] for m in recent_metrics]
        
        return {
            "current_cpu": recent_metrics[-1]["cpu_percent"],
            "average_cpu": sum(cpu_values) / len(cpu_values),
            "max_cpu": max(cpu_values),
            "current_memory": recent_metrics[-1]["memory_percent"],
            "average_memory": sum(memory_values) / len(memory_values),
            "max_memory": max(memory_values),
            "active_threads": recent_metrics[-1]["active_threads"],
            "samples_count": len(recent_metrics)
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.request_batcher = RequestBatcher()
        self.resource_monitor = ResourceMonitor()
        self.metrics = PerformanceMetrics()
        
        # Thread pools for different types of work
        self.io_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="io")
        self.cpu_executor = ProcessPoolExecutor(max_workers=4)
        
        # Setup batch processors
        self._setup_batch_processors()
        
        # Setup resource callbacks
        self.resource_monitor.add_callback(self._handle_resource_alert)
    
    def _setup_batch_processors(self):
        """Setup batch processors for common operations."""
        
        async def batch_file_analysis(batch_args: List[tuple], batch_kwargs: List[dict]):
            """Process multiple file analyses together."""
            results = []
            
            # Group by similar analysis types
            for args, kwargs in zip(batch_args, batch_kwargs):
                # This would normally call the actual analysis function
                # For now, return placeholder
                results.append(f"Analysis result for {args[0] if args else 'unknown'}")
            
            return results
        
        async def batch_llm_requests(batch_args: List[tuple], batch_kwargs: List[dict]):
            """Process multiple LLM requests together."""
            results = []
            
            # Combine prompts for batch processing
            for args, kwargs in zip(batch_args, batch_kwargs):
                # This would normally call the LLM with batched prompts
                results.append(f"LLM response for batch request")
            
            return results
        
        self.request_batcher.register_batch_processor("file_analysis", batch_file_analysis)
        self.request_batcher.register_batch_processor("llm_request", batch_llm_requests)
    
    async def _handle_resource_alert(self, alert: Dict[str, Any]):
        """Handle resource threshold alerts."""
        alert_type = alert["type"]
        
        if alert_type == "cpu_high":
            # Reduce concurrent operations
            print(f"High CPU usage detected: {alert['value']:.1f}%")
            print("Recommendation: Reducing concurrent operations")
            
        elif alert_type == "memory_high":
            # Trigger garbage collection and cache cleanup
            print(f"High memory usage detected: {alert['value']:.1f}%")
            print("Recommendation: Cleaning up memory")
            
            # Force garbage collection
            gc.collect()
            
            # Clear some cache entries
            if hasattr(self.cache_manager.cache, 'clear'):
                # Clear oldest 25% of cache entries (simplified)
                pass
    
    def performance_monitor(self, operation_name: str = "operation"):
        """Decorator to monitor function performance."""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    # Check cache first
                    cache_key = f"{operation_name}:{str(args)}:{str(kwargs)}"
                    cached_result = await self.cache_manager.get("llm_response", prompt=cache_key)
                    
                    if cached_result is not None:
                        self.metrics.cache_hits += 1
                        return cached_result
                    
                    self.metrics.cache_misses += 1
                    
                    # Execute function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.io_executor, func, *args, **kwargs
                        )
                    
                    # Cache result
                    await self.cache_manager.set("llm_response", result, prompt=cache_key)
                    
                    return result
                
                except Exception as e:
                    self.metrics.errors += 1
                    raise
                
                finally:
                    # Update metrics
                    response_time = time.time() - start_time
                    self.metrics.update_response_time(response_time)
            
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(async_wrapper(*args, **kwargs))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    async def optimize_request(self, operation_type: str, function: Callable, *args, **kwargs):
        """Optimize a request using batching and caching."""
        # Check cache first
        cache_result = await self.cache_manager.get(
            operation_type, 
            function=function.__name__,
            args=str(args),
            kwargs=str(kwargs)
        )
        
        if cache_result is not None:
            self.metrics.cache_hits += 1
            return cache_result
        
        self.metrics.cache_misses += 1
        
        # Use request batcher for potentially batchable operations
        if operation_type in ["file_analysis", "llm_request"]:
            result = await self.request_batcher.add_request(operation_type, function, *args, **kwargs)
        else:
            # Execute directly
            if asyncio.iscoroutinefunction(function):
                result = await function(*args, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.io_executor, function, *args, **kwargs
                )
        
        # Cache the result
        await self.cache_manager.set(
            operation_type,
            result,
            function=function.__name__,
            args=str(args),
            kwargs=str(kwargs)
        )
        
        return result
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        await self.resource_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.resource_monitor.stop_monitoring()
        
        # Cleanup executors
        self.io_executor.shutdown(wait=False)
        self.cpu_executor.shutdown(wait=False)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        cache_stats = self.cache_manager.get_stats()
        resource_summary = self.resource_monitor.get_metrics_summary()
        
        return {
            "performance_metrics": self.metrics.to_dict(),
            "cache_statistics": cache_stats,
            "resource_summary": resource_summary,
            "recommendations": self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Cache hit rate recommendations
        if self.metrics.cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate detected. Consider adjusting cache TTL or warming strategy.")
        
        # Response time recommendations
        if self.metrics.average_response_time > 3.0:
            recommendations.append("High average response time. Consider optimizing slow operations or scaling resources.")
        
        # Memory recommendations
        resource_summary = self.resource_monitor.get_metrics_summary()
        if resource_summary.get("average_memory", 0) > 80:
            recommendations.append("High memory usage detected. Consider implementing memory optimization or scaling.")
        
        # CPU recommendations
        if resource_summary.get("average_cpu", 0) > 70:
            recommendations.append("High CPU usage detected. Consider optimizing CPU-intensive operations or horizontal scaling.")
        
        if not recommendations:
            recommendations.append("Performance looks good! System is running optimally.")
        
        return recommendations


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


# Convenience decorators
def optimize_performance(operation_type: str = "general"):
    """Decorator for automatic performance optimization."""
    optimizer = get_performance_optimizer()
    return optimizer.performance_monitor(operation_type)


def cache_and_batch(operation_type: str):
    """Decorator for caching and batching operations."""
    async def decorator(func):
        async def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            return await optimizer.optimize_request(operation_type, func, *args, **kwargs)
        return wrapper
    return decorator