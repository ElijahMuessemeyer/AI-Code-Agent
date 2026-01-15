"""
Tests for the performance optimization system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from src.performance.optimizer import (
    PerformanceOptimizer, PerformanceMetrics, RequestBatcher, 
    ResourceMonitor, BatchRequest
)


class TestPerformanceMetrics:
    """Test cases for performance metrics tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = PerformanceMetrics()
    
    def test_initial_state(self):
        """Test initial metrics state."""
        assert self.metrics.request_count == 0
        assert self.metrics.total_response_time == 0.0
        assert self.metrics.min_response_time == float('inf')
        assert self.metrics.max_response_time == 0.0
        assert self.metrics.cache_hits == 0
        assert self.metrics.cache_misses == 0
        assert self.metrics.errors == 0
    
    def test_update_response_time(self):
        """Test updating response time metrics."""
        self.metrics.update_response_time(1.5)
        assert self.metrics.request_count == 1
        assert self.metrics.total_response_time == 1.5
        assert self.metrics.min_response_time == 1.5
        assert self.metrics.max_response_time == 1.5
        assert self.metrics.average_response_time == 1.5
        
        self.metrics.update_response_time(2.5)
        assert self.metrics.request_count == 2
        assert self.metrics.total_response_time == 4.0
        assert self.metrics.min_response_time == 1.5
        assert self.metrics.max_response_time == 2.5
        assert self.metrics.average_response_time == 2.0
    
    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        assert self.metrics.cache_hit_rate == 0.0
        
        self.metrics.cache_hits = 7
        self.metrics.cache_misses = 3
        assert self.metrics.cache_hit_rate == 0.7
        
        self.metrics.cache_hits = 0
        self.metrics.cache_misses = 0
        assert self.metrics.cache_hit_rate == 0.0
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        self.metrics.update_response_time(1.0)
        self.metrics.cache_hits = 5
        self.metrics.cache_misses = 2
        
        result = self.metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["request_count"] == 1
        assert result["average_response_time"] == 1.0
        assert result["cache_hit_rate"] == 5/7
        assert "memory_usage_mb" in result
        assert "cpu_usage_percent" in result


class TestRequestBatcher:
    """Test cases for request batching."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batcher = RequestBatcher(batch_size=3, max_wait_time=0.1)
    
    def test_register_batch_processor(self):
        """Test registering batch processors."""
        async def test_processor(batch_args, batch_kwargs):
            return ["result1", "result2"]
        
        self.batcher.register_batch_processor("test_operation", test_processor)
        assert "test_operation" in self.batcher.batch_processors
    
    @pytest.mark.asyncio
    async def test_individual_processing(self):
        """Test processing requests individually when no batch processor exists."""
        async def test_function(x, y=1):
            await asyncio.sleep(0.01)  # Simulate work
            return x + y
        
        # Submit requests
        task1 = asyncio.create_task(
            self.batcher.add_request("unknown_operation", test_function, 1, y=2)
        )
        task2 = asyncio.create_task(
            self.batcher.add_request("unknown_operation", test_function, 3, y=4)
        )
        
        # Wait for results
        result1 = await task1
        result2 = await task2
        
        assert result1 == 3
        assert result2 == 7
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing with registered processor."""
        async def batch_processor(batch_args, batch_kwargs):
            # Sum all first arguments
            results = []
            for args, kwargs in zip(batch_args, batch_kwargs):
                result = args[0] + kwargs.get('y', 0)
                results.append(result)
            return results
        
        self.batcher.register_batch_processor("test_batch", batch_processor)
        
        # Submit requests
        tasks = [
            asyncio.create_task(
                self.batcher.add_request("test_batch", lambda x, y=0: x + y, 1, y=1)
            ),
            asyncio.create_task(
                self.batcher.add_request("test_batch", lambda x, y=0: x + y, 2, y=2)
            ),
            asyncio.create_task(
                self.batcher.add_request("test_batch", lambda x, y=0: x + y, 3, y=3)
            )
        ]
        
        # Wait for results
        results = await asyncio.gather(*tasks)
        
        assert results == [2, 4, 6]
    
    @pytest.mark.asyncio
    async def test_batch_size_trigger(self):
        """Test that batch processing triggers when batch size is reached."""
        async def batch_processor(batch_args, batch_kwargs):
            return [f"processed_{args[0]}" for args in batch_args]
        
        self.batcher.register_batch_processor("size_test", batch_processor)
        
        # Submit exactly batch_size requests
        tasks = []
        for i in range(3):  # batch_size = 3
            task = asyncio.create_task(
                self.batcher.add_request("size_test", lambda x: x, i)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all("processed_" in str(result) for result in results)


class TestResourceMonitor:
    """Test cases for resource monitoring."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ResourceMonitor(monitoring_interval=0.1)
    
    def test_initial_state(self):
        """Test initial monitor state."""
        assert self.monitor.is_monitoring is False
        assert len(self.monitor.metrics_history) == 0
        assert len(self.monitor.callbacks) == 0
    
    def test_add_callback(self):
        """Test adding monitoring callbacks."""
        def test_callback(alert):
            pass
        
        self.monitor.add_callback(test_callback)
        assert len(self.monitor.callbacks) == 1
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.Process')
    def test_collect_metrics(self, mock_process, mock_disk, mock_memory, mock_cpu):
        """Test metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value.percent = 60.0
        mock_disk.return_value.percent = 70.0
        
        # Mock process metrics
        mock_proc_instance = Mock()
        mock_proc_instance.memory_info.return_value.rss = 128 * 1024 * 1024  # 128MB
        mock_proc_instance.open_files.return_value = []
        mock_proc_instance.connections.return_value = []
        mock_process.return_value = mock_proc_instance
        
        metrics = self.monitor._collect_metrics()
        
        assert metrics["cpu_percent"] == 45.0
        assert metrics["memory_percent"] == 60.0
        assert metrics["disk_usage_percent"] == 70.0
        assert metrics["memory_used_mb"] == 128.0
        assert "timestamp" in metrics
    
    @pytest.mark.asyncio
    async def test_threshold_checking(self):
        """Test threshold checking and alerts."""
        alerts_received = []
        
        async def test_callback(alert):
            alerts_received.append(alert)
        
        self.monitor.add_callback(test_callback)
        
        # Create metrics that exceed thresholds
        high_cpu_metrics = {
            "cpu_percent": 85.0,  # Above 80% threshold
            "memory_percent": 50.0,
            "timestamp": "2023-01-01T00:00:00"
        }
        
        await self.monitor._check_thresholds(high_cpu_metrics)
        
        assert len(alerts_received) == 1
        assert alerts_received[0]["type"] == "cpu_high"
        assert alerts_received[0]["value"] == 85.0
    
    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        # Add some test metrics
        for i in range(5):
            self.monitor.metrics_history.append({
                "cpu_percent": 40.0 + i,
                "memory_percent": 50.0 + i,
                "active_threads": 10 + i,
                "timestamp": f"2023-01-01T00:0{i}:00"
            })
        
        summary = self.monitor.get_metrics_summary()
        
        assert summary["current_cpu"] == 44.0
        assert summary["average_cpu"] == 42.0
        assert summary["max_cpu"] == 44.0
        assert summary["current_memory"] == 54.0
        assert summary["samples_count"] == 5
    
    def test_empty_metrics_summary(self):
        """Test metrics summary with no data."""
        summary = self.monitor.get_metrics_summary()
        assert summary == {}


class TestPerformanceOptimizer:
    """Test cases for the main performance optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = PerformanceOptimizer()
    
    def test_initial_state(self):
        """Test initial optimizer state."""
        assert self.optimizer.cache_manager is not None
        assert self.optimizer.request_batcher is not None
        assert self.optimizer.resource_monitor is not None
        assert self.optimizer.metrics is not None
        assert self.optimizer.io_executor is not None
        assert self.optimizer.cpu_executor is not None
    
    @pytest.mark.asyncio
    @patch('src.performance.cache_manager.get_cache_manager')
    async def test_optimize_request_with_cache_hit(self, mock_cache_manager):
        """Test request optimization with cache hit."""
        # Mock cache manager
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value="cached_result")
        mock_cache_manager.return_value = mock_cache
        
        def test_function(x):
            return x * 2
        
        result = await self.optimizer.optimize_request(
            "test_operation", test_function, 5
        )
        
        assert result == "cached_result"
        assert self.optimizer.metrics.cache_hits == 1
        assert self.optimizer.metrics.cache_misses == 0
    
    @pytest.mark.asyncio
    @patch('src.performance.cache_manager.get_cache_manager')
    async def test_optimize_request_with_cache_miss(self, mock_cache_manager):
        """Test request optimization with cache miss."""
        # Mock cache manager
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock(return_value=True)
        mock_cache_manager.return_value = mock_cache
        
        def test_function(x):
            return x * 2
        
        result = await self.optimizer.optimize_request(
            "other_operation", test_function, 5
        )
        
        assert result == 10
        assert self.optimizer.metrics.cache_hits == 0
        assert self.optimizer.metrics.cache_misses == 1
    
    @pytest.mark.asyncio
    async def test_optimize_request_batchable(self):
        """Test optimization of batchable requests."""
        def test_function(x):
            return x * 3
        
        # Mock the request batcher
        self.optimizer.request_batcher.add_request = AsyncMock(return_value=15)
        
        result = await self.optimizer.optimize_request(
            "file_analysis", test_function, 5
        )
        
        assert result == 15
        self.optimizer.request_batcher.add_request.assert_called_once()
    
    def test_performance_monitor_decorator(self):
        """Test performance monitoring decorator."""
        @self.optimizer.performance_monitor("test_operation")
        def test_sync_function(x):
            return x + 1
        
        result = test_sync_function(5)
        assert result == 6
        assert self.optimizer.metrics.request_count > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitor_async_decorator(self):
        """Test performance monitoring decorator with async function."""
        @self.optimizer.performance_monitor("test_async_operation")
        async def test_async_function(x):
            await asyncio.sleep(0.01)
            return x + 1
        
        result = await test_async_function(5)
        assert result == 6
        assert self.optimizer.metrics.request_count > 0
    
    @pytest.mark.asyncio
    async def test_resource_alert_handling(self):
        """Test handling of resource alerts."""
        # Test CPU alert
        cpu_alert = {
            "type": "cpu_high",
            "value": 85.0,
            "threshold": 80.0,
            "recommendation": "Reduce concurrent operations"
        }
        
        await self.optimizer._handle_resource_alert(cpu_alert)
        # Should not raise exception
        
        # Test memory alert
        memory_alert = {
            "type": "memory_high",
            "value": 90.0,
            "threshold": 85.0,
            "recommendation": "Clean up memory"
        }
        
        await self.optimizer._handle_resource_alert(memory_alert)
        # Should not raise exception
    
    @patch('src.performance.cache_manager.get_cache_manager')
    def test_get_performance_report(self, mock_cache_manager):
        """Test performance report generation."""
        # Mock cache manager
        mock_cache = Mock()
        mock_cache.get_stats = Mock(return_value={
            "hits": 100,
            "misses": 20,
            "hit_rate": 0.83
        })
        mock_cache_manager.return_value = mock_cache
        
        # Add some metrics
        self.optimizer.metrics.update_response_time(1.5)
        self.optimizer.metrics.update_response_time(2.0)
        self.optimizer.metrics.cache_hits = 100
        self.optimizer.metrics.cache_misses = 20
        
        # Add some resource metrics
        self.optimizer.resource_monitor.metrics_history.append({
            "cpu_percent": 45.0,
            "memory_percent": 60.0,
            "active_threads": 10
        })
        
        report = self.optimizer.get_performance_report()
        
        assert "performance_metrics" in report
        assert "cache_statistics" in report
        assert "resource_summary" in report
        assert "recommendations" in report
        assert report["performance_metrics"]["request_count"] == 2
    
    def test_generate_performance_recommendations(self):
        """Test performance recommendation generation."""
        # Set up metrics that trigger recommendations
        self.optimizer.metrics.cache_hits = 10
        self.optimizer.metrics.cache_misses = 40  # Low hit rate
        self.optimizer.metrics.update_response_time(4.0)  # High response time
        
        # Add resource metrics
        self.optimizer.resource_monitor.metrics_history.append({
            "cpu_percent": 75.0,
            "memory_percent": 85.0
        })
        
        recommendations = self.optimizer._generate_performance_recommendations()
        
        assert len(recommendations) > 0
        assert any("cache hit rate" in rec.lower() for rec in recommendations)
        assert any("response time" in rec.lower() for rec in recommendations)
    
    def test_cleanup(self):
        """Test optimizer cleanup."""
        self.optimizer.stop_monitoring()
        
        # Should not raise exception
        assert not self.optimizer.resource_monitor.is_monitoring


@pytest.mark.asyncio
async def test_convenience_decorators():
    """Test convenience decorators."""
    from src.performance.optimizer import optimize_performance, cache_and_batch
    
    @optimize_performance("test_op")
    async def test_function(x):
        return x * 2
    
    result = await test_function(5)
    assert result == 10


if __name__ == "__main__":
    pytest.main([__file__])