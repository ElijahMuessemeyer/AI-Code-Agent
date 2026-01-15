"""
Intelligent caching system for AI Code Agent performance optimization.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import zlib
from enum import Enum

try:
    import redis
    import aioredis
except ImportError:
    redis = None
    aioredis = None

try:
    import diskcache
except ImportError:
    diskcache = None


class CacheBackend(Enum):
    """Supported cache backends."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Represents a cached entry."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheStats:
    """Cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.total_size_bytes = 0
        self.start_time = time.time()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def uptime_seconds(self) -> float:
        """Get cache uptime in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "total_size_bytes": self.total_size_bytes,
            "uptime_seconds": self.uptime_seconds
        }


class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.expires_at is None:
            return False
        return datetime.now() > entry.expires_at
    
    def _evict_if_needed(self):
        """Evict entries if cache is full."""
        if len(self.cache) >= self.max_size:
            # LRU eviction
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed or self.cache[k].created_at
            )
            del self.cache[oldest_key]
            self.stats.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.stats.misses += 1
            return None
        
        entry = self.cache[key]
        
        if self._is_expired(entry):
            del self.cache[key]
            self.stats.misses += 1
            return None
        
        # Update access info
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        self.stats.hits += 1
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            self._evict_if_needed()
            
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                last_accessed=datetime.now()
            )
            
            self.cache[key] = entry
            self.stats.sets += 1
            return True
            
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            self.stats.deletes += 1
            return True
        return False
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self.cache)


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 default_ttl: int = 3600, key_prefix: str = "aicode:"):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.stats = CacheStats()
        self.redis_client: Optional[aioredis.Redis] = None
    
    async def _get_redis(self):
        """Get Redis client."""
        if self.redis_client is None:
            self.redis_client = aioredis.from_url(self.redis_url)
        return self.redis_client
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis_client = await self._get_redis()
            data = await redis_client.get(self._make_key(key))
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Decompress and deserialize
            decompressed = zlib.decompress(data)
            value = pickle.loads(decompressed)
            
            self.stats.hits += 1
            return value
            
        except Exception:
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            redis_client = await self._get_redis()
            
            # Serialize and compress
            serialized = pickle.dumps(value)
            compressed = zlib.compress(serialized)
            
            # Set with TTL
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            await redis_client.setex(self._make_key(key), ttl_seconds, compressed)
            
            self.stats.sets += 1
            return True
            
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.delete(self._make_key(key))
            
            if result > 0:
                self.stats.deletes += 1
                return True
            return False
            
        except Exception:
            return False
    
    async def clear(self):
        """Clear all cache entries with prefix."""
        try:
            redis_client = await self._get_redis()
            keys = await redis_client.keys(f"{self.key_prefix}*")
            if keys:
                await redis_client.delete(*keys)
        except Exception:
            pass


class DiskCache:
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir: str = ".cache", max_size_gb: float = 1.0, default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.default_ttl = default_ttl
        self.stats = CacheStats()
        
        if diskcache:
            self.cache = diskcache.Cache(str(self.cache_dir), size_limit=self.max_size_bytes)
        else:
            self.cache = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        if not self.cache:
            return None
        
        try:
            value = self.cache.get(key)
            if value is not None:
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            return value
        except Exception:
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        if not self.cache:
            return False
        
        try:
            expire_time = ttl if ttl is not None else self.default_ttl
            self.cache.set(key, value, expire=expire_time)
            self.stats.sets += 1
            return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        if not self.cache:
            return False
        
        try:
            result = self.cache.delete(key)
            if result:
                self.stats.deletes += 1
            return result
        except Exception:
            return False
    
    def clear(self):
        """Clear all cache entries."""
        if self.cache:
            self.cache.clear()


class HybridCache:
    """Hybrid cache using multiple backends."""
    
    def __init__(self, 
                 memory_cache: MemoryCache,
                 redis_cache: Optional[RedisCache] = None,
                 disk_cache: Optional[DiskCache] = None):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.disk_cache = disk_cache
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache (memory -> redis -> disk)."""
        # Try memory first
        value = self.memory_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            return value
        
        # Try Redis second
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.set(key, value)
                self.stats.hits += 1
                return value
        
        # Try disk last
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # Promote to memory and Redis
                self.memory_cache.set(key, value)
                if self.redis_cache:
                    await self.redis_cache.set(key, value)
                self.stats.hits += 1
                return value
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all cache layers."""
        success = True
        
        # Set in memory
        if not self.memory_cache.set(key, value, ttl):
            success = False
        
        # Set in Redis
        if self.redis_cache:
            if not await self.redis_cache.set(key, value, ttl):
                success = False
        
        # Set in disk
        if self.disk_cache:
            if not self.disk_cache.set(key, value, ttl):
                success = False
        
        if success:
            self.stats.sets += 1
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache layers."""
        success = True
        
        if not self.memory_cache.delete(key):
            success = False
        
        if self.redis_cache:
            if not await self.redis_cache.delete(key):
                success = False
        
        if self.disk_cache:
            if not self.disk_cache.delete(key):
                success = False
        
        if success:
            self.stats.deletes += 1
        
        return success


class CacheManager:
    """Main cache manager with intelligent caching strategies."""
    
    def __init__(self, backend: CacheBackend = CacheBackend.HYBRID):
        self.backend = backend
        self.cache = self._initialize_cache()
        self.cache_strategies: Dict[str, Callable] = {
            "code_analysis": self._hash_code_content,
            "llm_response": self._hash_llm_prompt,
            "file_analysis": self._hash_file_path_and_mtime,
            "workflow_result": self._hash_workflow_config
        }
    
    def _initialize_cache(self):
        """Initialize cache backend."""
        if self.backend == CacheBackend.MEMORY:
            return MemoryCache(max_size=10000, default_ttl=3600)
        
        elif self.backend == CacheBackend.REDIS:
            if aioredis is None:
                print("Redis not available, falling back to memory cache")
                return MemoryCache()
            return RedisCache()
        
        elif self.backend == CacheBackend.DISK:
            return DiskCache()
        
        elif self.backend == CacheBackend.HYBRID:
            memory = MemoryCache(max_size=1000, default_ttl=1800)  # 30 min
            redis_cache = None
            disk_cache = None
            
            if aioredis:
                try:
                    redis_cache = RedisCache(default_ttl=7200)  # 2 hours
                except:
                    pass
            
            if diskcache:
                disk_cache = DiskCache(default_ttl=86400)  # 24 hours
            
            return HybridCache(memory, redis_cache, disk_cache)
        
        return MemoryCache()
    
    def _hash_code_content(self, content: str, **kwargs) -> str:
        """Generate cache key for code content."""
        hasher = hashlib.sha256()
        hasher.update(content.encode('utf-8'))
        
        # Include additional parameters
        for key, value in sorted(kwargs.items()):
            hasher.update(f"{key}:{value}".encode('utf-8'))
        
        return f"code:{hasher.hexdigest()[:16]}"
    
    def _hash_llm_prompt(self, prompt: str, model: str = "default", **kwargs) -> str:
        """Generate cache key for LLM prompts."""
        hasher = hashlib.sha256()
        hasher.update(prompt.encode('utf-8'))
        hasher.update(model.encode('utf-8'))
        
        # Include model parameters
        for key, value in sorted(kwargs.items()):
            hasher.update(f"{key}:{value}".encode('utf-8'))
        
        return f"llm:{hasher.hexdigest()[:16]}"
    
    def _hash_file_path_and_mtime(self, file_path: str, **kwargs) -> str:
        """Generate cache key for file analysis including modification time."""
        try:
            mtime = Path(file_path).stat().st_mtime
            hasher = hashlib.sha256()
            hasher.update(file_path.encode('utf-8'))
            hasher.update(str(mtime).encode('utf-8'))
            
            for key, value in sorted(kwargs.items()):
                hasher.update(f"{key}:{value}".encode('utf-8'))
            
            return f"file:{hasher.hexdigest()[:16]}"
        except:
            # Fallback to just file path
            return f"file:{hashlib.sha256(file_path.encode()).hexdigest()[:16]}"
    
    def _hash_workflow_config(self, workflow_type: str, files: List[str], **kwargs) -> str:
        """Generate cache key for workflow results."""
        hasher = hashlib.sha256()
        hasher.update(workflow_type.encode('utf-8'))
        
        # Hash sorted file list
        for file_path in sorted(files):
            hasher.update(file_path.encode('utf-8'))
        
        for key, value in sorted(kwargs.items()):
            hasher.update(f"{key}:{value}".encode('utf-8'))
        
        return f"workflow:{hasher.hexdigest()[:16]}"
    
    async def get(self, cache_type: str, **kwargs) -> Optional[Any]:
        """Get cached value using appropriate strategy."""
        if cache_type not in self.cache_strategies:
            return None
        
        key = self.cache_strategies[cache_type](**kwargs)
        
        if isinstance(self.cache, (RedisCache, HybridCache)):
            return await self.cache.get(key)
        else:
            return self.cache.get(key)
    
    async def set(self, cache_type: str, value: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """Set cached value using appropriate strategy."""
        if cache_type not in self.cache_strategies:
            return False
        
        key = self.cache_strategies[cache_type](**kwargs)
        
        if isinstance(self.cache, (RedisCache, HybridCache)):
            return await self.cache.set(key, value, ttl)
        else:
            return self.cache.set(key, value, ttl)
    
    async def invalidate(self, cache_type: str, **kwargs) -> bool:
        """Invalidate cached value."""
        if cache_type not in self.cache_strategies:
            return False
        
        key = self.cache_strategies[cache_type](**kwargs)
        
        if isinstance(self.cache, (RedisCache, HybridCache)):
            return await self.cache.delete(key)
        else:
            return self.cache.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if hasattr(self.cache, 'stats'):
            return self.cache.stats.to_dict()
        return {}
    
    async def warm_cache(self, warmup_data: List[Dict[str, Any]]):
        """Pre-populate cache with frequently accessed data."""
        for item in warmup_data:
            cache_type = item.get('type')
            value = item.get('value')
            ttl = item.get('ttl')
            kwargs = item.get('kwargs', {})
            
            if cache_type and value is not None:
                await self.set(cache_type, value, ttl, **kwargs)
    
    def cache_decorator(self, cache_type: str, ttl: Optional[int] = None):
        """Decorator for automatic caching of function results."""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                # Generate cache key from function arguments
                cache_key_data = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(sorted(kwargs.items()))
                }
                
                # Try to get cached result
                cached_result = await self.get(cache_type, **cache_key_data)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(cache_type, result, ttl, **cache_key_data)
                return result
            
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions, we need to run async operations
                import asyncio
                
                cache_key_data = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(sorted(kwargs.items()))
                }
                
                # Try to get cached result
                try:
                    loop = asyncio.get_event_loop()
                    cached_result = loop.run_until_complete(self.get(cache_type, **cache_key_data))
                    if cached_result is not None:
                        return cached_result
                except:
                    pass
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self.set(cache_type, result, ttl, **cache_key_data))
                except:
                    pass
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# Convenience decorators
def cache_code_analysis(ttl: int = 3600):
    """Cache code analysis results."""
    return get_cache_manager().cache_decorator("code_analysis", ttl)


def cache_llm_response(ttl: int = 7200):
    """Cache LLM responses."""
    return get_cache_manager().cache_decorator("llm_response", ttl)


def cache_file_analysis(ttl: int = 1800):
    """Cache file analysis results."""
    return get_cache_manager().cache_decorator("file_analysis", ttl)