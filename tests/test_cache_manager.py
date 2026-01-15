"""
Tests for the intelligent caching system.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.performance.cache_manager import (
    CacheManager, MemoryCache, CacheStats, CacheEntry, CacheBackend,
    get_cache_manager
)


class TestCacheEntry:
    """Test cases for cache entry."""
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time()
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert entry.last_accessed is None
        assert entry.compressed is False


class TestCacheStats:
    """Test cases for cache statistics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = CacheStats()
    
    def test_initial_stats(self):
        """Test initial statistics state."""
        assert self.stats.hits == 0
        assert self.stats.misses == 0
        assert self.stats.sets == 0
        assert self.stats.deletes == 0
        assert self.stats.evictions == 0
        assert self.stats.hit_rate == 0.0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        self.stats.hits = 8
        self.stats.misses = 2
        assert self.stats.hit_rate == 0.8
        
        # Test with no requests
        self.stats.hits = 0
        self.stats.misses = 0
        assert self.stats.hit_rate == 0.0
    
    def test_uptime_tracking(self):
        """Test uptime calculation."""
        initial_time = self.stats.start_time
        time.sleep(0.01)  # Small delay
        uptime = self.stats.uptime_seconds
        assert uptime > 0
        assert uptime >= time.time() - initial_time
    
    def test_to_dict(self):
        """Test converting stats to dictionary."""
        self.stats.hits = 10
        self.stats.misses = 5
        self.stats.sets = 15
        
        result = self.stats.to_dict()
        
        assert isinstance(result, dict)
        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["sets"] == 15
        assert result["hit_rate"] == 2/3
        assert "uptime_seconds" in result


class TestMemoryCache:
    """Test cases for memory cache implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = MemoryCache(max_size=3, default_ttl=1)  # Small size for testing
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        # Test set and get
        success = self.cache.set("key1", "value1")
        assert success is True
        
        value = self.cache.get("key1")
        assert value == "value1"
        
        # Test miss
        value = self.cache.get("nonexistent")
        assert value is None
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        # Set with short TTL
        success = self.cache.set("temp_key", "temp_value", ttl=0.1)
        assert success is True
        
        # Should be available immediately
        value = self.cache.get("temp_key")
        assert value == "temp_value"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        value = self.cache.get("temp_key")
        assert value is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        self.cache.get("key1")
        
        # Add another key, should evict key2 (least recently used)
        self.cache.set("key4", "value4")
        
        assert self.cache.get("key1") == "value1"  # Still there
        assert self.cache.get("key2") is None      # Evicted
        assert self.cache.get("key3") == "value3"  # Still there
        assert self.cache.get("key4") == "value4"  # Newly added
    
    def test_delete_operation(self):
        """Test cache deletion."""
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"
        
        success = self.cache.delete("key1")
        assert success is True
        assert self.cache.get("key1") is None
        
        # Delete non-existent key
        success = self.cache.delete("nonexistent")
        assert success is False
    
    def test_clear_operation(self):
        """Test cache clearing."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.cache.clear()
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None
        assert self.cache.size() == 0
    
    def test_size_tracking(self):
        """Test cache size tracking."""
        assert self.cache.size() == 0
        
        self.cache.set("key1", "value1")
        assert self.cache.size() == 1
        
        self.cache.set("key2", "value2")
        assert self.cache.size() == 2
        
        self.cache.delete("key1")
        assert self.cache.size() == 1
    
    def test_statistics_tracking(self):
        """Test cache statistics tracking."""
        # Test hits and misses
        self.cache.get("nonexistent")  # Miss
        assert self.cache.stats.misses == 1
        assert self.cache.stats.hits == 0
        
        self.cache.set("key1", "value1")  # Set
        assert self.cache.stats.sets == 1
        
        self.cache.get("key1")  # Hit
        assert self.cache.stats.hits == 1
        
        self.cache.delete("key1")  # Delete
        assert self.cache.stats.deletes == 1


class TestCacheManager:
    """Test cases for the main cache manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = CacheManager(CacheBackend.MEMORY)
    
    def test_initial_state(self):
        """Test initial cache manager state."""
        assert self.cache_manager.backend == CacheBackend.MEMORY
        assert self.cache_manager.cache is not None
        assert len(self.cache_manager.cache_strategies) > 0
    
    def test_hash_code_content(self):
        """Test code content hashing strategy."""
        code1 = "def hello(): return 'world'"
        code2 = "def goodbye(): return 'world'"
        
        hash1 = self.cache_manager._hash_code_content(code1, language="python")
        hash2 = self.cache_manager._hash_code_content(code2, language="python")
        hash3 = self.cache_manager._hash_code_content(code1, language="python")
        
        assert hash1 != hash2  # Different code should have different hashes
        assert hash1 == hash3  # Same code should have same hash
        assert hash1.startswith("code:")
    
    def test_hash_llm_prompt(self):
        """Test LLM prompt hashing strategy."""
        prompt1 = "Review this code"
        prompt2 = "Generate documentation"
        
        hash1 = self.cache_manager._hash_llm_prompt(prompt1, model="gpt-4")
        hash2 = self.cache_manager._hash_llm_prompt(prompt2, model="gpt-4")
        hash3 = self.cache_manager._hash_llm_prompt(prompt1, model="claude-3")
        
        assert hash1 != hash2  # Different prompts
        assert hash1 != hash3  # Different models
        assert hash1.startswith("llm:")
    
    def test_hash_file_path_and_mtime(self):
        """Test file path and modification time hashing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            hash1 = self.cache_manager._hash_file_path_and_mtime(temp_file)
            
            # Modify file
            time.sleep(0.01)
            with open(temp_file, 'a') as f:
                f.write("more content")
            
            hash2 = self.cache_manager._hash_file_path_and_mtime(temp_file)
            
            assert hash1 != hash2  # Different modification times
            assert hash1.startswith("file:")
            assert hash2.startswith("file:")
            
        finally:
            Path(temp_file).unlink()
    
    def test_hash_workflow_config(self):
        """Test workflow configuration hashing."""
        files1 = ["file1.py", "file2.py"]
        files2 = ["file2.py", "file1.py"]  # Same files, different order
        files3 = ["file1.py", "file3.py"]  # Different files
        
        hash1 = self.cache_manager._hash_workflow_config("analysis", files1)
        hash2 = self.cache_manager._hash_workflow_config("analysis", files2)
        hash3 = self.cache_manager._hash_workflow_config("analysis", files3)
        
        assert hash1 == hash2  # Order shouldn't matter (sorted internally)
        assert hash1 != hash3  # Different files
        assert hash1.startswith("workflow:")
    
    @pytest.mark.asyncio
    async def test_get_and_set_operations(self):
        """Test cache get and set operations."""
        # Test cache miss
        result = await self.cache_manager.get("code_analysis", content="test code")
        assert result is None
        
        # Test cache set and hit
        success = await self.cache_manager.set(
            "code_analysis", 
            {"issues": [], "metrics": {"complexity": 1}},
            ttl=60,
            content="test code"
        )
        assert success is True
        
        result = await self.cache_manager.get("code_analysis", content="test code")
        assert result is not None
        assert "issues" in result
        assert "metrics" in result
    
    @pytest.mark.asyncio
    async def test_invalidate_operation(self):
        """Test cache invalidation."""
        # Set a cache entry
        await self.cache_manager.set(
            "llm_response",
            "Generated response",
            prompt="test prompt"
        )
        
        # Verify it's there
        result = await self.cache_manager.get("llm_response", prompt="test prompt")
        assert result == "Generated response"
        
        # Invalidate it
        success = await self.cache_manager.invalidate("llm_response", prompt="test prompt")
        assert success is True
        
        # Verify it's gone
        result = await self.cache_manager.get("llm_response", prompt="test prompt")
        assert result is None
    
    def test_get_stats(self):
        """Test cache statistics retrieval."""
        # Perform some operations
        self.cache_manager.cache.set("test_key", "test_value")
        self.cache_manager.cache.get("test_key")
        self.cache_manager.cache.get("nonexistent")
        
        stats = self.cache_manager.get_stats()
        
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "sets" in stats
    
    @pytest.mark.asyncio
    async def test_warm_cache(self):
        """Test cache warming functionality."""
        warmup_data = [
            {
                "type": "llm_response",
                "value": "Cached response 1",
                "ttl": 3600,
                "kwargs": {"prompt": "prompt1", "model": "gpt-4"}
            },
            {
                "type": "code_analysis", 
                "value": {"complexity": 2},
                "ttl": 1800,
                "kwargs": {"content": "def test(): pass"}
            }
        ]
        
        await self.cache_manager.warm_cache(warmup_data)
        
        # Verify cached data
        result1 = await self.cache_manager.get("llm_response", prompt="prompt1", model="gpt-4")
        assert result1 == "Cached response 1"
        
        result2 = await self.cache_manager.get("code_analysis", content="def test(): pass")
        assert result2 == {"complexity": 2}
    
    def test_cache_decorator_sync(self):
        """Test cache decorator for synchronous functions."""
        call_count = 0
        
        @self.cache_manager.cache_decorator("test_cache", ttl=60)
        def expensive_function(x, y=1):
            nonlocal call_count
            call_count += 1
            return x * y * 2
        
        # First call should execute function
        result1 = expensive_function(5, y=3)
        assert result1 == 30
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = expensive_function(5, y=3)
        assert result2 == 30
        assert call_count == 1  # Function not called again
        
        # Call with different args should execute function
        result3 = expensive_function(7, y=2)
        assert result3 == 28
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_decorator_async(self):
        """Test cache decorator for async functions."""
        call_count = 0
        
        @self.cache_manager.cache_decorator("async_test_cache", ttl=60)
        async def expensive_async_function(x, y=1):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return x * y * 3
        
        # First call should execute function
        result1 = await expensive_async_function(4, y=2)
        assert result1 == 24
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = await expensive_async_function(4, y=2)
        assert result2 == 24
        assert call_count == 1  # Function not called again
    
    def test_unknown_cache_type(self):
        """Test behavior with unknown cache type."""
        # get with unknown type should return None
        result = asyncio.run(self.cache_manager.get("unknown_type", key="value"))
        assert result is None
        
        # set with unknown type should return False
        success = asyncio.run(self.cache_manager.set("unknown_type", "data", key="value"))
        assert success is False
        
        # invalidate with unknown type should return False
        success = asyncio.run(self.cache_manager.invalidate("unknown_type", key="value"))
        assert success is False


class TestCacheBackendInitialization:
    """Test cache backend initialization."""
    
    def test_memory_backend(self):
        """Test memory backend initialization."""
        manager = CacheManager(CacheBackend.MEMORY)
        assert isinstance(manager.cache, MemoryCache)
    
    @patch('src.performance.cache_manager.aioredis')
    def test_redis_backend_fallback(self, mock_aioredis):
        """Test Redis backend fallback to memory when Redis unavailable."""
        # Mock Redis as None to simulate import failure
        mock_aioredis = None
        
        with patch('src.performance.cache_manager.aioredis', None):
            manager = CacheManager(CacheBackend.REDIS)
            assert isinstance(manager.cache, MemoryCache)
    
    @patch('src.performance.cache_manager.diskcache')
    def test_disk_backend(self, mock_diskcache):
        """Test disk backend initialization."""
        manager = CacheManager(CacheBackend.DISK)
        # Should create DiskCache instance


def test_convenience_decorators():
    """Test convenience decorator functions."""
    from src.performance.cache_manager import (
        cache_code_analysis, cache_llm_response, cache_file_analysis
    )
    
    @cache_code_analysis(ttl=1800)
    def analyze_code(code):
        return {"complexity": len(code)}
    
    result = analyze_code("def test(): pass")
    assert isinstance(result, dict)
    assert "complexity" in result


def test_global_cache_manager():
    """Test global cache manager singleton."""
    from src.performance.cache_manager import get_cache_manager
    
    manager1 = get_cache_manager()
    manager2 = get_cache_manager()
    
    assert manager1 is manager2  # Should be same instance


if __name__ == "__main__":
    pytest.main([__file__])