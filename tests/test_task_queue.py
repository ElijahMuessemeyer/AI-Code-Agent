"""
Tests for the distributed task queue system.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.distributed.task_queue import (
    TaskQueue, Task, TaskStatus, TaskPriority, TaskResult,
    get_task_queue
)


class TestTask:
    """Test cases for Task dataclass."""
    
    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            id="test_task_123",
            function_name="process_data",
            args=(1, 2, 3),
            kwargs={"param": "value"},
            priority=TaskPriority.HIGH,
            max_retries=5,
            timeout=600
        )
        
        assert task.id == "test_task_123"
        assert task.function_name == "process_data"
        assert task.args == (1, 2, 3)
        assert task.kwargs == {"param": "value"}
        assert task.priority == TaskPriority.HIGH
        assert task.max_retries == 5
        assert task.timeout == 600
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
        assert isinstance(task.created_at, datetime)
    
    def test_task_comparison(self):
        """Test task comparison for priority queue."""
        high_priority_task = Task(
            id="high", function_name="test", priority=TaskPriority.HIGH
        )
        low_priority_task = Task(
            id="low", function_name="test", priority=TaskPriority.LOW
        )
        
        # High priority should be "less than" low priority (for min-heap)
        assert high_priority_task < low_priority_task
    
    def test_task_serialization(self):
        """Test task serialization to/from dictionary."""
        original_task = Task(
            id="serialize_test",
            function_name="test_function",
            args=(1, "test"),
            kwargs={"key": "value"},
            priority=TaskPriority.URGENT,
            dependencies=["dep1", "dep2"],
            scheduled_at=datetime.now() + timedelta(hours=1)
        )
        
        # Convert to dict
        task_dict = original_task.to_dict()
        assert isinstance(task_dict, dict)
        assert task_dict["id"] == "serialize_test"
        assert task_dict["priority"] == TaskPriority.URGENT.value
        assert task_dict["status"] == TaskStatus.PENDING.value
        
        # Convert back to Task
        restored_task = Task.from_dict(task_dict)
        assert restored_task.id == original_task.id
        assert restored_task.function_name == original_task.function_name
        assert restored_task.priority == original_task.priority
        assert restored_task.status == original_task.status


class TestTaskResult:
    """Test cases for TaskResult dataclass."""
    
    def test_task_result_creation(self):
        """Test creating a task result."""
        result = TaskResult(
            task_id="test_task",
            status=TaskStatus.COMPLETED,
            result={"output": "success"},
            execution_time=1.5,
            worker_id="worker_001"
        )
        
        assert result.task_id == "test_task"
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"output": "success"}
        assert result.execution_time == 1.5
        assert result.worker_id == "worker_001"
        assert result.error is None


class TestTaskQueue:
    """Test cases for task queue."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.task_queue = TaskQueue(backend="memory")
    
    @pytest.mark.asyncio
    async def test_task_queue_initialization(self):
        """Test task queue initialization."""
        success = await self.task_queue.initialize()
        assert success is True
        assert self.task_queue.backend == "memory"
        assert len(self.task_queue.tasks) == 0
        assert len(self.task_queue.pending_queue) == 0
    
    def test_register_function(self):
        """Test registering functions for execution."""
        def test_function(x, y):
            return x + y
        
        self.task_queue.register_function("add", test_function)
        assert "add" in self.task_queue.task_functions
        assert self.task_queue.task_functions["add"] == test_function
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test submitting a task."""
        task_id = await self.task_queue.submit_task(
            "test_function",
            1, 2, 3,
            priority=TaskPriority.HIGH,
            timeout=300,
            param="value"
        )
        
        assert task_id is not None
        assert task_id in self.task_queue.tasks
        
        task = self.task_queue.tasks[task_id]
        assert task.function_name == "test_function"
        assert task.args == (1, 2, 3)
        assert task.kwargs == {"param": "value"}
        assert task.priority == TaskPriority.HIGH
        assert task.timeout == 300
    
    @pytest.mark.asyncio
    async def test_submit_scheduled_task(self):
        """Test submitting a scheduled task."""
        future_time = datetime.now() + timedelta(seconds=1)
        
        task_id = await self.task_queue.submit_task(
            "scheduled_function",
            scheduled_at=future_time
        )
        
        task = self.task_queue.tasks[task_id]
        assert task.scheduled_at == future_time
        assert len(self.task_queue.pending_queue) == 0  # Not in queue yet
    
    @pytest.mark.asyncio
    async def test_submit_task_with_dependencies(self):
        """Test submitting a task with dependencies."""
        # Submit first task
        dep_task_id = await self.task_queue.submit_task("dependency_task")
        
        # Submit task that depends on first task
        task_id = await self.task_queue.submit_task(
            "dependent_task",
            dependencies=[dep_task_id]
        )
        
        task = self.task_queue.tasks[task_id]
        assert dep_task_id in task.dependencies
        assert len(self.task_queue.pending_queue) == 1  # Only independent task in queue
    
    @pytest.mark.asyncio
    async def test_get_next_task(self):
        """Test getting the next task for execution."""
        # Submit multiple tasks with different priorities
        low_task_id = await self.task_queue.submit_task("low_task", priority=TaskPriority.LOW)
        high_task_id = await self.task_queue.submit_task("high_task", priority=TaskPriority.HIGH)
        normal_task_id = await self.task_queue.submit_task("normal_task", priority=TaskPriority.NORMAL)
        
        # Should get high priority task first
        task = await self.task_queue.get_next_task("worker_001")
        assert task.id == high_task_id
        assert task.status == TaskStatus.RUNNING
        assert task.assigned_worker == "worker_001"
        
        # Should get normal priority task next
        task = await self.task_queue.get_next_task("worker_002")
        assert task.id == normal_task_id
        
        # Should get low priority task last
        task = await self.task_queue.get_next_task("worker_003")
        assert task.id == low_task_id
    
    @pytest.mark.asyncio
    async def test_get_next_task_with_expired_task(self):
        """Test getting next task when some tasks are expired."""
        # Submit task that expires immediately
        task_id = await self.task_queue.submit_task(
            "expired_task",
            priority=TaskPriority.HIGH
        )
        
        # Manually expire the task
        task = self.task_queue.tasks[task_id]
        task.expires_at = datetime.now() - timedelta(hours=1)
        
        # Should not get the expired task
        next_task = await self.task_queue.get_next_task("worker_001")
        assert next_task is None
        
        # Task should be cancelled
        assert task.status == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self):
        """Test successful task execution."""
        # Register a test function
        def add_numbers(x, y):
            return x + y
        
        self.task_queue.register_function("add", add_numbers)
        
        # Create and execute task
        task = Task(
            id="execute_test",
            function_name="add",
            args=(5, 3),
            assigned_worker="worker_001"
        )
        
        result = await self.task_queue.execute_task(task)
        
        assert result.status == TaskStatus.COMPLETED
        assert result.result == 8
        assert result.error is None
        assert result.execution_time > 0
        assert result.worker_id == "worker_001"
    
    @pytest.mark.asyncio
    async def test_execute_task_async_function(self):
        """Test executing an async function."""
        async def async_add(x, y):
            await asyncio.sleep(0.01)  # Simulate async work
            return x + y
        
        self.task_queue.register_function("async_add", async_add)
        
        task = Task(
            id="async_test",
            function_name="async_add",
            args=(10, 20),
            assigned_worker="worker_001"
        )
        
        result = await self.task_queue.execute_task(task)
        
        assert result.status == TaskStatus.COMPLETED
        assert result.result == 30
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self):
        """Test task execution failure."""
        def failing_function():
            raise ValueError("Test error")
        
        self.task_queue.register_function("fail", failing_function)
        
        task = Task(
            id="fail_test",
            function_name="fail",
            max_retries=2,
            assigned_worker="worker_001"
        )
        
        result = await self.task_queue.execute_task(task)
        
        # Should be marked for retry
        assert result.status == TaskStatus.RETRY
        assert "Test error" in result.error
        assert task.retry_count == 1
        assert task.status == TaskStatus.RETRY
    
    @pytest.mark.asyncio
    async def test_execute_task_max_retries_exceeded(self):
        """Test task execution when max retries are exceeded."""
        def failing_function():
            raise ValueError("Persistent error")
        
        self.task_queue.register_function("persistent_fail", failing_function)
        
        task = Task(
            id="max_retry_test",
            function_name="persistent_fail",
            max_retries=2,
            assigned_worker="worker_001"
        )
        
        # Execute task multiple times to exceed retry limit
        task.retry_count = 2  # Already at max retries
        
        result = await self.task_queue.execute_task(task)
        
        assert result.status == TaskStatus.FAILED
        assert task.status == TaskStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_execute_task_timeout(self):
        """Test task execution timeout."""
        async def slow_function():
            await asyncio.sleep(1.0)  # Longer than timeout
            return "completed"
        
        self.task_queue.register_function("slow", slow_function)
        
        task = Task(
            id="timeout_test",
            function_name="slow",
            timeout=0.1,  # Very short timeout
            assigned_worker="worker_001"
        )
        
        result = await self.task_queue.execute_task(task)
        
        assert result.status in [TaskStatus.FAILED, TaskStatus.RETRY]
        assert "timed out" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_task_unregistered_function(self):
        """Test executing a task with unregistered function."""
        task = Task(
            id="unregistered_test",
            function_name="nonexistent_function",
            assigned_worker="worker_001"
        )
        
        result = await self.task_queue.execute_task(task)
        
        assert result.status == TaskStatus.FAILED
        assert "not registered" in result.error
    
    @pytest.mark.asyncio
    async def test_get_task_result(self):
        """Test getting task results."""
        task_id = "result_test"
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result="test_output"
        )
        
        self.task_queue.results[task_id] = result
        
        retrieved_result = await self.task_queue.get_task_result(task_id)
        assert retrieved_result == result
        
        # Test non-existent result
        none_result = await self.task_queue.get_task_result("nonexistent")
        assert none_result is None
    
    @pytest.mark.asyncio
    async def test_get_task_status(self):
        """Test getting task status."""
        task = Task(id="status_test", function_name="test", status=TaskStatus.RUNNING)
        self.task_queue.tasks["status_test"] = task
        
        status = await self.task_queue.get_task_status("status_test")
        assert status == TaskStatus.RUNNING
        
        # Test non-existent task
        none_status = await self.task_queue.get_task_status("nonexistent")
        assert none_status is None
    
    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """Test cancelling a pending task."""
        task_id = await self.task_queue.submit_task("cancel_test")
        
        success = await self.task_queue.cancel_task(task_id)
        assert success is True
        
        task = self.task_queue.tasks[task_id]
        assert task.status == TaskStatus.CANCELLED
        
        # Should have a cancelled result
        result = await self.task_queue.get_task_result(task_id)
        assert result.status == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_cancel_running_task(self):
        """Test that running tasks cannot be cancelled."""
        task = Task(id="running_test", function_name="test", status=TaskStatus.RUNNING)
        self.task_queue.tasks["running_test"] = task
        
        success = await self.task_queue.cancel_task("running_test")
        assert success is False
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        # Create completed dependency
        dep_task = Task(id="dependency", function_name="test", status=TaskStatus.COMPLETED)
        self.task_queue.completed_tasks["dependency"] = dep_task
        self.task_queue.results["dependency"] = TaskResult(
            task_id="dependency",
            status=TaskStatus.COMPLETED
        )
        
        # Test task with satisfied dependencies
        task_with_deps = Task(
            id="dependent",
            function_name="test",
            dependencies=["dependency"]
        )
        
        assert self.task_queue._check_dependencies(task_with_deps) is True
        
        # Test task with unsatisfied dependencies
        task_with_bad_deps = Task(
            id="bad_dependent",
            function_name="test",
            dependencies=["nonexistent"]
        )
        
        assert self.task_queue._check_dependencies(task_with_bad_deps) is False
    
    def test_get_queue_stats(self):
        """Test getting queue statistics."""
        # Add some test data
        self.task_queue.execution_stats["total_tasks"] = 10
        self.task_queue.execution_stats["completed_tasks"] = 7
        self.task_queue.execution_stats["failed_tasks"] = 2
        self.task_queue.execution_stats["average_execution_time"] = 1.5
        
        # Add some tasks to queues
        task1 = Task(id="pending1", function_name="test", priority=TaskPriority.HIGH)
        task2 = Task(id="running1", function_name="test", status=TaskStatus.RUNNING)
        task3 = Task(id="completed1", function_name="test", status=TaskStatus.COMPLETED)
        
        self.task_queue.pending_queue.append(task1)
        self.task_queue.running_tasks["running1"] = task2
        self.task_queue.completed_tasks["completed1"] = task3
        
        stats = self.task_queue.get_queue_stats()
        
        assert stats["total_tasks"] == 10
        assert stats["pending_tasks"] == 1
        assert stats["running_tasks"] == 1
        assert stats["completed_tasks"] == 1
        assert stats["failed_tasks"] == 2
        assert stats["success_rate"] == 7/9  # 7 completed out of 9 finished
        assert stats["backend"] == "memory"
        assert "priority_breakdown" in stats
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_tasks(self):
        """Test cleanup of expired tasks."""
        # Create expired task
        expired_task = Task(
            id="expired",
            function_name="test",
            expires_at=datetime.now() - timedelta(hours=1)
        )
        self.task_queue.tasks["expired"] = expired_task
        
        # Create non-expired task
        valid_task = Task(
            id="valid",
            function_name="test",
            expires_at=datetime.now() + timedelta(hours=1)
        )
        self.task_queue.tasks["valid"] = valid_task
        
        await self.task_queue._cleanup_expired_tasks()
        
        # Expired task should be cancelled
        assert expired_task.status == TaskStatus.CANCELLED
        assert valid_task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_dependent_task_scheduling(self):
        """Test that dependent tasks are scheduled when dependencies complete."""
        # Submit dependency task
        dep_task_id = await self.task_queue.submit_task("dependency")
        
        # Submit dependent task
        dep_task_id2 = await self.task_queue.submit_task(
            "dependent",
            dependencies=[dep_task_id]
        )
        
        # Initially, only independent task should be in queue
        assert len(self.task_queue.pending_queue) == 1
        
        # Complete the dependency
        dep_task = self.task_queue.tasks[dep_task_id]
        dep_task.status = TaskStatus.COMPLETED
        self.task_queue.completed_tasks[dep_task_id] = dep_task
        self.task_queue.results[dep_task_id] = TaskResult(
            task_id=dep_task_id,
            status=TaskStatus.COMPLETED
        )
        
        # Trigger dependency check
        await self.task_queue._check_dependent_tasks(dep_task_id)
        
        # Now dependent task should be in queue
        assert len(self.task_queue.pending_queue) == 1
        
        # Get the task and verify it's the dependent one
        next_task = await self.task_queue.get_next_task("worker_001")
        assert next_task.id == dep_task_id2


@patch('src.distributed.task_queue.redis')
class TestTaskQueueRedisBackend:
    """Test cases for Redis backend (mocked)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.task_queue = TaskQueue(backend="redis", redis_url="redis://localhost:6379")
    
    @pytest.mark.asyncio
    async def test_redis_initialization(self, mock_redis):
        """Test Redis backend initialization."""
        # Mock Redis client
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_redis.from_url.return_value = mock_client
        
        success = await self.task_queue.initialize()
        assert success is True
        assert self.task_queue.redis_client == mock_client
    
    @pytest.mark.asyncio
    async def test_redis_submit_task(self, mock_redis):
        """Test submitting task to Redis backend."""
        # Mock Redis client
        mock_client = AsyncMock()
        mock_client.hset = AsyncMock(return_value=True)
        mock_client.lpush = AsyncMock(return_value=1)
        self.task_queue.redis_client = mock_client
        
        task_id = await self.task_queue.submit_task("redis_test")
        
        # Verify Redis operations were called
        mock_client.hset.assert_called()
        mock_client.lpush.assert_called()
    
    @pytest.mark.asyncio
    async def test_redis_get_next_task(self, mock_redis):
        """Test getting next task from Redis backend."""
        # Mock Redis client with task data
        mock_client = AsyncMock()
        
        # Mock task data
        task = Task(id="redis_task", function_name="test")
        task_json = task.to_dict()
        
        mock_client.rpop = AsyncMock(return_value="redis_task")
        mock_client.hget = AsyncMock(return_value=json.dumps(task_json))
        mock_client.hset = AsyncMock(return_value=True)
        self.task_queue.redis_client = mock_client
        
        retrieved_task = await self.task_queue._get_next_task_redis("worker_001")
        
        assert retrieved_task is not None
        assert retrieved_task.id == "redis_task"
        assert retrieved_task.assigned_worker == "worker_001"


def test_global_task_queue():
    """Test global task queue singleton."""
    queue1 = get_task_queue()
    queue2 = get_task_queue()
    
    assert queue1 is queue2  # Should be same instance


if __name__ == "__main__":
    pytest.main([__file__])