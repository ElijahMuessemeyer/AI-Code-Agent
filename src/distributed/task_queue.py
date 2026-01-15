"""
Distributed task queue system for AI Code Agent scaling.
"""

import asyncio
import uuid
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import heapq
from collections import defaultdict

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    import celery
    from celery import Celery
except ImportError:
    celery = None
    Celery = None


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a distributed task."""
    id: str
    function_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout: int = 300  # 5 minutes
    retry_delay: int = 60  # 1 minute
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime fields
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    assigned_worker: Optional[str] = None
    
    def __lt__(self, other):
        """Compare tasks for priority queue ordering."""
        if not isinstance(other, Task):
            return NotImplemented
        
        # Higher priority values come first
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        
        # Earlier creation time comes first for same priority
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        data = asdict(self)
        # Convert enums and datetime objects
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['scheduled_at'] = self.scheduled_at.isoformat() if self.scheduled_at else None
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        # Convert back to proper types
        data['priority'] = TaskPriority(data['priority'])
        data['status'] = TaskStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['scheduled_at']:
            data['scheduled_at'] = datetime.fromisoformat(data['scheduled_at'])
        if data['expires_at']:
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        
        return cls(**data)


class TaskQueue:
    """Distributed task queue with multiple backend support."""
    
    def __init__(self, backend: str = "memory", redis_url: str = None, 
                 celery_broker: str = None):
        self.backend = backend
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.pending_queue = []  # Priority queue
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_functions: Dict[str, Callable] = {}
        
        # Task execution tracking
        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0
        }
        
        # Initialize backend
        if backend == "redis" and redis:
            self.redis_client = None
            self.redis_url = redis_url or "redis://localhost:6379"
        elif backend == "celery" and Celery:
            self.celery_app = None
            self.celery_broker = celery_broker or "redis://localhost:6379"
        
        # Scheduled task checking
        self._scheduler_running = False
    
    async def initialize(self) -> bool:
        """Initialize the task queue backend."""
        try:
            if self.backend == "redis" and redis:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                print("Redis task queue initialized")
                
            elif self.backend == "celery" and Celery:
                self.celery_app = Celery('ai_code_agent', broker=self.celery_broker)
                print("Celery task queue initialized")
                
            else:
                print("Memory task queue initialized")
            
            # Start scheduler for periodic tasks
            asyncio.create_task(self._task_scheduler())
            return True
            
        except Exception as e:
            print(f"Failed to initialize task queue: {e}")
            return False
    
    def register_function(self, name: str, function: Callable):
        """Register a function that can be executed as a task."""
        self.task_functions[name] = function
    
    async def submit_task(self, function_name: str, *args, 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: int = 300, max_retries: int = 3,
                         scheduled_at: Optional[datetime] = None,
                         dependencies: List[str] = None,
                         **kwargs) -> str:
        """Submit a task for execution."""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            scheduled_at=scheduled_at,
            dependencies=dependencies or [],
            expires_at=datetime.now() + timedelta(hours=24)  # Default expiry
        )
        
        # Store task
        self.tasks[task_id] = task
        self.execution_stats["total_tasks"] += 1
        
        # Add to appropriate queue based on backend
        if self.backend == "redis" and self.redis_client:
            await self._submit_task_redis(task)
        elif self.backend == "celery" and self.celery_app:
            await self._submit_task_celery(task)
        else:
            await self._submit_task_memory(task)
        
        return task_id
    
    async def _submit_task_memory(self, task: Task):
        """Submit task to memory queue."""
        if task.scheduled_at and task.scheduled_at > datetime.now():
            # Don't add to pending queue yet, scheduler will handle it
            pass
        elif self._check_dependencies(task):
            heapq.heappush(self.pending_queue, task)
    
    async def _submit_task_redis(self, task: Task):
        """Submit task to Redis queue."""
        try:
            # Store task data
            await self.redis_client.hset(
                f"task:{task.id}", 
                mapping={"data": json.dumps(task.to_dict())}
            )
            
            # Add to appropriate queue
            if task.scheduled_at and task.scheduled_at > datetime.now():
                # Add to delayed queue
                score = task.scheduled_at.timestamp()
                await self.redis_client.zadd("delayed_tasks", {task.id: score})
            elif self._check_dependencies(task):
                # Add to priority queue
                await self.redis_client.lpush(f"queue:{task.priority.name.lower()}", task.id)
        
        except Exception as e:
            print(f"Error submitting task to Redis: {e}")
            # Fallback to memory queue
            await self._submit_task_memory(task)
    
    async def _submit_task_celery(self, task: Task):
        """Submit task to Celery."""
        try:
            # This would use Celery's delay/apply_async methods
            print(f"Would submit task {task.id} to Celery")
        except Exception as e:
            print(f"Error submitting task to Celery: {e}")
            # Fallback to memory queue
            await self._submit_task_memory(task)
    
    def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            
            dep_result = self.results.get(dep_id)
            if not dep_result or dep_result.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def get_next_task(self, worker_id: str) -> Optional[Task]:
        """Get the next task for execution."""
        if self.backend == "redis" and self.redis_client:
            return await self._get_next_task_redis(worker_id)
        else:
            return await self._get_next_task_memory(worker_id)
    
    async def _get_next_task_memory(self, worker_id: str) -> Optional[Task]:
        """Get next task from memory queue."""
        while self.pending_queue:
            task = heapq.heappop(self.pending_queue)
            
            # Check if task is still valid
            if task.expires_at and datetime.now() > task.expires_at:
                task.status = TaskStatus.CANCELLED
                continue
            
            # Check dependencies again
            if not self._check_dependencies(task):
                # Put back in queue if dependencies not ready
                heapq.heappush(self.pending_queue, task)
                break
            
            # Assign task to worker
            task.status = TaskStatus.RUNNING
            task.assigned_worker = worker_id
            self.running_tasks[task.id] = task
            
            return task
        
        return None
    
    async def _get_next_task_redis(self, worker_id: str) -> Optional[Task]:
        """Get next task from Redis queue."""
        try:
            # Try priority queues in order
            for priority in [TaskPriority.CRITICAL, TaskPriority.URGENT, 
                           TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
                
                queue_name = f"queue:{priority.name.lower()}"
                task_id = await self.redis_client.rpop(queue_name)
                
                if task_id:
                    # Get task data
                    task_data = await self.redis_client.hget(f"task:{task_id}", "data")
                    if task_data:
                        task_dict = json.loads(task_data)
                        task = Task.from_dict(task_dict)
                        
                        # Assign to worker
                        task.status = TaskStatus.RUNNING
                        task.assigned_worker = worker_id
                        self.running_tasks[task.id] = task
                        
                        # Update in Redis
                        await self.redis_client.hset(
                            f"task:{task.id}",
                            mapping={"data": json.dumps(task.to_dict())}
                        )
                        
                        return task
            
            return None
            
        except Exception as e:
            print(f"Error getting task from Redis: {e}")
            return await self._get_next_task_memory(worker_id)
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task and return the result."""
        start_time = time.time()
        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(),
            worker_id=task.assigned_worker
        )
        
        try:
            # Get function
            if task.function_name not in self.task_functions:
                raise ValueError(f"Function '{task.function_name}' not registered")
            
            function = self.task_functions[task.function_name]
            
            # Execute with timeout
            try:
                if asyncio.iscoroutinefunction(function):
                    task_result = await asyncio.wait_for(
                        function(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    # Run in executor to avoid blocking
                    task_result = await asyncio.get_event_loop().run_in_executor(
                        None, function, *task.args, **task.kwargs
                    )
                
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                
            except asyncio.TimeoutError:
                raise Exception(f"Task timed out after {task.timeout} seconds")
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            
            # Check if should retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRY
                
                # Schedule retry
                retry_at = datetime.now() + timedelta(seconds=task.retry_delay)
                task.scheduled_at = retry_at
                
                result.status = TaskStatus.RETRY
                
                # Re-submit for retry
                await self._submit_task_memory(task) if self.backend == "memory" else await self._submit_task_redis(task)
        
        finally:
            result.execution_time = time.time() - start_time
            result.completed_at = datetime.now()
            
            # Update task status
            if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.status = result.status
                
                # Move from running to completed
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                self.completed_tasks[task.id] = task
                
                # Update stats
                if result.status == TaskStatus.COMPLETED:
                    self.execution_stats["completed_tasks"] += 1
                else:
                    self.execution_stats["failed_tasks"] += 1
                
                # Update average execution time
                total_completed = self.execution_stats["completed_tasks"]
                if total_completed > 0:
                    current_avg = self.execution_stats["average_execution_time"]
                    self.execution_stats["average_execution_time"] = (
                        (current_avg * (total_completed - 1) + result.execution_time) / total_completed
                    )
                
                # Check and submit dependent tasks
                await self._check_dependent_tasks(task.id)
            
            # Store result
            self.results[task.id] = result
        
        return result
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Check for tasks that were waiting for this dependency."""
        for task in list(self.tasks.values()):
            if (completed_task_id in task.dependencies and 
                task.status == TaskStatus.PENDING and
                self._check_dependencies(task)):
                
                # Dependencies are now satisfied, add to queue
                if self.backend == "redis":
                    await self._submit_task_redis(task)
                else:
                    heapq.heappush(self.pending_queue, task)
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get the result of a task."""
        return self.results.get(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            
            # Remove from pending queue (for memory backend)
            self.pending_queue = [t for t in self.pending_queue if t.id != task_id]
            heapq.heapify(self.pending_queue)
            
            # Store cancelled result
            self.results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                completed_at=datetime.now()
            )
            
            return True
        
        return False
    
    async def _task_scheduler(self):
        """Background scheduler for delayed and periodic tasks."""
        self._scheduler_running = True
        
        while self._scheduler_running:
            try:
                now = datetime.now()
                
                # Check for delayed tasks that are ready
                if self.backend == "redis" and self.redis_client:
                    # Get tasks from delayed queue that are ready
                    ready_tasks = await self.redis_client.zrangebyscore(
                        "delayed_tasks", 0, now.timestamp()
                    )
                    
                    for task_id in ready_tasks:
                        # Move to appropriate priority queue
                        task_data = await self.redis_client.hget(f"task:{task_id}", "data")
                        if task_data:
                            task_dict = json.loads(task_data)
                            task = Task.from_dict(task_dict)
                            
                            if self._check_dependencies(task):
                                await self.redis_client.lpush(
                                    f"queue:{task.priority.name.lower()}", 
                                    task_id
                                )
                            
                            # Remove from delayed queue
                            await self.redis_client.zrem("delayed_tasks", task_id)
                
                else:
                    # Memory backend - check tasks with scheduled_at
                    scheduled_tasks = [
                        task for task in self.tasks.values()
                        if (task.scheduled_at and 
                            task.scheduled_at <= now and
                            task.status == TaskStatus.PENDING)
                    ]
                    
                    for task in scheduled_tasks:
                        if self._check_dependencies(task):
                            task.scheduled_at = None
                            heapq.heappush(self.pending_queue, task)
                
                # Clean up expired tasks
                await self._cleanup_expired_tasks()
                
                # Sleep before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def _cleanup_expired_tasks(self):
        """Clean up expired tasks."""
        now = datetime.now()
        expired_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.expires_at and now > task.expires_at and task.status == TaskStatus.PENDING
        ]
        
        for task_id in expired_tasks:
            await self.cancel_task(task_id)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending_count = len(self.pending_queue)
        running_count = len(self.running_tasks)
        completed_count = len(self.completed_tasks)
        
        # Group pending tasks by priority
        priority_breakdown = defaultdict(int)
        for task in self.pending_queue:
            priority_breakdown[task.priority.name] += 1
        
        return {
            "total_tasks": self.execution_stats["total_tasks"],
            "pending_tasks": pending_count,
            "running_tasks": running_count,
            "completed_tasks": completed_count,
            "failed_tasks": self.execution_stats["failed_tasks"],
            "average_execution_time": self.execution_stats["average_execution_time"],
            "success_rate": (
                self.execution_stats["completed_tasks"] / 
                max(1, self.execution_stats["completed_tasks"] + self.execution_stats["failed_tasks"])
            ),
            "priority_breakdown": dict(priority_breakdown),
            "backend": self.backend
        }
    
    async def shutdown(self):
        """Shutdown the task queue."""
        self._scheduler_running = False
        
        if self.backend == "redis" and self.redis_client:
            await self.redis_client.close()


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create global task queue."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue