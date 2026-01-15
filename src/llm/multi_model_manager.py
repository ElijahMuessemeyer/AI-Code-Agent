"""
Multi-model management system with intelligent model routing and switching.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import statistics

from src.llm.model_interface import ModelManager, ModelProvider, ModelResponse, ModelConfig
from src.performance.cache_manager import get_cache_manager


class ModelCapability(Enum):
    """Model capabilities for intelligent routing."""
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    BUG_DETECTION = "bug_detection"
    DOCUMENTATION = "documentation"
    GENERAL_CHAT = "general_chat"
    REASONING = "reasoning"
    MATH = "math"
    CREATIVE_WRITING = "creative_writing"


class RoutingStrategy(Enum):
    """Model routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    HIGHEST_QUALITY = "highest_quality"
    COST_OPTIMIZED = "cost_optimized"
    CAPABILITY_BASED = "capability_based"
    ADAPTIVE = "adaptive"


@dataclass
class ModelMetrics:
    """Performance metrics for a model."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    error_rate_window: List[bool] = field(default_factory=list)  # Recent success/failure
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_requests / max(1, self.total_requests)
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / max(1, self.successful_requests)
    
    @property
    def average_quality_score(self) -> float:
        """Calculate average quality score."""
        return statistics.mean(self.quality_scores) if self.quality_scores else 0.0
    
    @property
    def recent_error_rate(self) -> float:
        """Calculate recent error rate from sliding window."""
        if not self.error_rate_window:
            return 0.0
        return (len(self.error_rate_window) - sum(self.error_rate_window)) / len(self.error_rate_window)
    
    def update_request(self, success: bool, response_time: float, tokens: int = 0, 
                      cost: float = 0.0, quality_score: Optional[float] = None):
        """Update metrics with new request data."""
        self.total_requests += 1
        self.last_used = datetime.now()
        
        if success:
            self.successful_requests += 1
            self.total_response_time += response_time
            self.min_response_time = min(self.min_response_time, response_time)
            self.max_response_time = max(self.max_response_time, response_time)
            self.total_tokens += tokens
            self.total_cost += cost
            
            if quality_score is not None:
                self.quality_scores.append(quality_score)
                # Keep only recent 100 scores
                if len(self.quality_scores) > 100:
                    self.quality_scores = self.quality_scores[-100:]
        else:
            self.failed_requests += 1
        
        # Update error rate window (keep last 50 requests)
        self.error_rate_window.append(success)
        if len(self.error_rate_window) > 50:
            self.error_rate_window = self.error_rate_window[-50:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "min_response_time": self.min_response_time if self.min_response_time != float('inf') else 0,
            "max_response_time": self.max_response_time,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_quality_score": self.average_quality_score,
            "recent_error_rate": self.recent_error_rate,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class ModelConfig:
    """Extended model configuration."""
    name: str
    provider: ModelProvider
    model_name: str
    api_key: str
    capabilities: List[ModelCapability] = field(default_factory=list)
    max_tokens: int = 4000
    temperature: float = 0.1
    cost_per_token: float = 0.0
    priority: int = 1  # Higher priority = preferred
    max_concurrent: int = 10
    enabled: bool = True
    timeout: int = 30


class ModelLoadBalancer:
    """Load balancer for distributing requests across models."""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.round_robin_counter = 0
        self.active_requests: Dict[str, int] = {}  # model_name -> active_request_count
    
    def select_model(self, 
                    available_models: List[str],
                    model_metrics: Dict[str, ModelMetrics],
                    model_configs: Dict[str, ModelConfig],
                    capability: Optional[ModelCapability] = None) -> Optional[str]:
        """Select the best model based on strategy."""
        
        # Filter by capability if specified
        if capability:
            capable_models = [
                name for name in available_models 
                if capability in model_configs[name].capabilities
            ]
            if capable_models:
                available_models = capable_models
        
        # Filter out disabled models and those with high error rates
        available_models = [
            name for name in available_models
            if (model_configs[name].enabled and 
                model_metrics[name].recent_error_rate < 0.5)
        ]
        
        if not available_models:
            return None
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_models)
        
        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_select(available_models)
        
        elif self.strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(available_models, model_metrics)
        
        elif self.strategy == RoutingStrategy.HIGHEST_QUALITY:
            return self._highest_quality_select(available_models, model_metrics)
        
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_select(available_models, model_configs, model_metrics)
        
        elif self.strategy == RoutingStrategy.CAPABILITY_BASED:
            return self._capability_based_select(available_models, model_configs, capability)
        
        elif self.strategy == RoutingStrategy.ADAPTIVE:
            return self._adaptive_select(available_models, model_metrics, model_configs, capability)
        
        else:
            return random.choice(available_models)
    
    def _round_robin_select(self, models: List[str]) -> str:
        """Round-robin selection."""
        self.round_robin_counter = (self.round_robin_counter + 1) % len(models)
        return models[self.round_robin_counter]
    
    def _least_loaded_select(self, models: List[str]) -> str:
        """Select model with least active requests."""
        return min(models, key=lambda m: self.active_requests.get(m, 0))
    
    def _fastest_response_select(self, models: List[str], metrics: Dict[str, ModelMetrics]) -> str:
        """Select model with fastest average response time."""
        return min(models, key=lambda m: metrics[m].average_response_time)
    
    def _highest_quality_select(self, models: List[str], metrics: Dict[str, ModelMetrics]) -> str:
        """Select model with highest quality scores."""
        return max(models, key=lambda m: metrics[m].average_quality_score)
    
    def _cost_optimized_select(self, models: List[str], configs: Dict[str, ModelConfig], 
                              metrics: Dict[str, ModelMetrics]) -> str:
        """Select most cost-effective model."""
        def cost_efficiency(model_name: str) -> float:
            config = configs[model_name]
            metric = metrics[model_name]
            
            # Calculate cost per successful request
            if metric.successful_requests == 0:
                return float('inf')
            
            avg_cost_per_request = metric.total_cost / metric.successful_requests
            quality_factor = max(0.1, metric.average_quality_score / 10.0)
            
            return avg_cost_per_request / quality_factor
        
        return min(models, key=cost_efficiency)
    
    def _capability_based_select(self, models: List[str], configs: Dict[str, ModelConfig],
                                capability: Optional[ModelCapability]) -> str:
        """Select based on model capabilities and priority."""
        if capability:
            # Prefer models with the specific capability
            capable_models = [m for m in models if capability in configs[m].capabilities]
            if capable_models:
                # Among capable models, select by priority
                return max(capable_models, key=lambda m: configs[m].priority)
        
        # Fallback to priority-based selection
        return max(models, key=lambda m: configs[m].priority)
    
    def _adaptive_select(self, models: List[str], metrics: Dict[str, ModelMetrics],
                        configs: Dict[str, ModelConfig], 
                        capability: Optional[ModelCapability]) -> str:
        """Adaptive selection combining multiple factors."""
        def adaptive_score(model_name: str) -> float:
            metric = metrics[model_name]
            config = configs[model_name]
            
            # Base score from success rate
            score = metric.success_rate * 100
            
            # Quality factor
            if metric.average_quality_score > 0:
                score += metric.average_quality_score * 10
            
            # Response time factor (lower is better)
            if metric.average_response_time > 0:
                score -= min(metric.average_response_time, 10) * 5
            
            # Load factor (lower active requests is better)
            active = self.active_requests.get(model_name, 0)
            score -= active * 2
            
            # Priority factor
            score += config.priority * 5
            
            # Capability match bonus
            if capability and capability in config.capabilities:
                score += 20
            
            # Recent performance penalty
            score -= metric.recent_error_rate * 50
            
            return score
        
        return max(models, key=adaptive_score)
    
    def track_request_start(self, model_name: str):
        """Track start of request for load balancing."""
        self.active_requests[model_name] = self.active_requests.get(model_name, 0) + 1
    
    def track_request_end(self, model_name: str):
        """Track end of request for load balancing."""
        if model_name in self.active_requests:
            self.active_requests[model_name] = max(0, self.active_requests[model_name] - 1)


class CircuitBreaker:
    """Circuit breaker pattern for model resilience."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).total_seconds() > self.timeout:
                self.state = "half-open"
                return True
            return False
        
        if self.state == "half-open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
        elif self.state == "half-open":
            self.state = "open"


class MultiModelManager:
    """Advanced multi-model manager with intelligent routing."""
    
    def __init__(self, default_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE):
        self.models: Dict[str, ModelManager] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.load_balancer = ModelLoadBalancer(default_strategy)
        self.cache_manager = get_cache_manager()
        
        # Fallback chain for resilience
        self.fallback_chain: List[str] = []
        
    def add_model(self, config: ModelConfig) -> bool:
        """Add a model to the manager."""
        try:
            # Create model manager instance
            model_config = ModelConfig(
                provider=config.provider,
                model_name=config.model_name,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout
            )
            
            model_manager = ModelManager()
            
            # Test the model
            if config.provider == ModelProvider.OPENAI:
                from src.llm.model_interface import OpenAIInterface
                interface = OpenAIInterface(model_config)
                model_manager.add_model(config.name, interface)
                
            elif config.provider == ModelProvider.ANTHROPIC:
                from src.llm.model_interface import AnthropicInterface
                interface = AnthropicInterface(model_config)
                model_manager.add_model(config.name, interface)
            
            # Store configuration and initialize metrics
            self.models[config.name] = model_manager
            self.model_configs[config.name] = config
            self.model_metrics[config.name] = ModelMetrics()
            self.circuit_breakers[config.name] = CircuitBreaker()
            
            print(f"Added model: {config.name} ({config.provider.value})")
            return True
            
        except Exception as e:
            print(f"Failed to add model {config.name}: {e}")
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the manager."""
        if model_name in self.models:
            del self.models[model_name]
            del self.model_configs[model_name]
            del self.model_metrics[model_name]
            del self.circuit_breakers[model_name]
            
            # Remove from fallback chain
            if model_name in self.fallback_chain:
                self.fallback_chain.remove(model_name)
            
            return True
        return False
    
    def set_fallback_chain(self, model_names: List[str]):
        """Set fallback chain for resilience."""
        self.fallback_chain = [name for name in model_names if name in self.models]
    
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              capability: Optional[ModelCapability] = None,
                              preferred_models: Optional[List[str]] = None,
                              **kwargs) -> ModelResponse:
        """Generate response using intelligent model selection."""
        
        # Check cache first
        cache_key = f"{prompt}:{system_prompt}:{capability}"
        cached_result = await self.cache_manager.get("llm_response", prompt=cache_key)
        if cached_result:
            return cached_result
        
        # Get available models
        available_models = preferred_models or list(self.models.keys())
        available_models = [m for m in available_models if m in self.models]
        
        # Try models in order (selected -> fallback chain)
        models_to_try = []
        
        # Add intelligently selected model first
        selected_model = self.load_balancer.select_model(
            available_models, self.model_metrics, self.model_configs, capability
        )
        if selected_model:
            models_to_try.append(selected_model)
        
        # Add fallback models
        for fallback_model in self.fallback_chain:
            if fallback_model not in models_to_try and fallback_model in available_models:
                models_to_try.append(fallback_model)
        
        # Try remaining models
        for model in available_models:
            if model not in models_to_try:
                models_to_try.append(model)
        
        last_error = None
        
        for model_name in models_to_try:
            # Check circuit breaker
            if not self.circuit_breakers[model_name].can_execute():
                continue
            
            try:
                # Track request start
                self.load_balancer.track_request_start(model_name)
                start_time = time.time()
                
                # Generate response
                response = await self.models[model_name].generate_response(
                    prompt, system_prompt, **kwargs
                )
                
                # Calculate metrics
                response_time = time.time() - start_time
                tokens_used = response.tokens_used or 0
                cost = self._calculate_cost(model_name, tokens_used)
                quality_score = self._estimate_quality_score(response)
                
                # Update metrics and circuit breaker
                self.model_metrics[model_name].update_request(
                    True, response_time, tokens_used, cost, quality_score
                )
                self.circuit_breakers[model_name].record_success()
                
                # Cache successful response
                await self.cache_manager.set("llm_response", response, ttl=3600, prompt=cache_key)
                
                return response
                
            except Exception as e:
                # Update metrics and circuit breaker
                response_time = time.time() - start_time
                self.model_metrics[model_name].update_request(False, response_time)
                self.circuit_breakers[model_name].record_failure()
                
                last_error = e
                continue
                
            finally:
                # Track request end
                self.load_balancer.track_request_end(model_name)
        
        # All models failed
        raise Exception(f"All models failed. Last error: {last_error}")
    
    def _calculate_cost(self, model_name: str, tokens: int) -> float:
        """Calculate cost for model usage."""
        config = self.model_configs[model_name]
        return tokens * config.cost_per_token
    
    def _estimate_quality_score(self, response: ModelResponse) -> float:
        """Estimate quality score for response."""
        # Simple heuristic based on response length and finish reason
        score = 5.0  # Base score
        
        if response.finish_reason == "stop":
            score += 2.0
        elif response.finish_reason == "length":
            score += 1.0
        
        # Length factor
        content_length = len(response.content)
        if 50 <= content_length <= 2000:
            score += 2.0
        elif content_length > 2000:
            score += 1.0
        
        # Token efficiency
        if response.tokens_used and response.tokens_used > 0:
            chars_per_token = len(response.content) / response.tokens_used
            if 3 <= chars_per_token <= 5:  # Good efficiency
                score += 1.0
        
        return min(10.0, score)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            "total_models": len(self.models),
            "active_models": len([m for m in self.model_configs.values() if m.enabled]),
            "fallback_chain": self.fallback_chain,
            "load_balancer_strategy": self.load_balancer.strategy.value,
            "models": {}
        }
        
        for model_name in self.models:
            config = self.model_configs[model_name]
            metrics = self.model_metrics[model_name]
            circuit_breaker = self.circuit_breakers[model_name]
            
            status["models"][model_name] = {
                "provider": config.provider.value,
                "model_name": config.model_name,
                "enabled": config.enabled,
                "circuit_breaker_state": circuit_breaker.state,
                "active_requests": self.load_balancer.active_requests.get(model_name, 0),
                "capabilities": [cap.value for cap in config.capabilities],
                "metrics": metrics.to_dict()
            }
        
        return status
    
    def set_routing_strategy(self, strategy: RoutingStrategy):
        """Change routing strategy."""
        self.load_balancer.strategy = strategy
    
    def enable_model(self, model_name: str) -> bool:
        """Enable a model."""
        if model_name in self.model_configs:
            self.model_configs[model_name].enabled = True
            return True
        return False
    
    def disable_model(self, model_name: str) -> bool:
        """Disable a model."""
        if model_name in self.model_configs:
            self.model_configs[model_name].enabled = False
            return True
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models."""
        health_results = {}
        
        for model_name in self.models:
            try:
                start_time = time.time()
                
                # Simple health check prompt
                response = await self.models[model_name].generate_response(
                    "Say 'OK' if you are working properly.",
                    timeout=10
                )
                
                response_time = time.time() - start_time
                
                health_results[model_name] = {
                    "status": "healthy",
                    "response_time": response_time,
                    "response": response.content[:20] if response.content else ""
                }
                
            except Exception as e:
                health_results[model_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health_results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        total_requests = sum(m.total_requests for m in self.model_metrics.values())
        total_cost = sum(m.total_cost for m in self.model_metrics.values())
        
        return {
            "summary": {
                "total_requests": total_requests,
                "total_cost": total_cost,
                "active_models": len([m for m in self.model_configs.values() if m.enabled])
            },
            "model_performance": {
                name: metrics.to_dict() 
                for name, metrics in self.model_metrics.items()
            },
            "recommendations": self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze model performance
        for name, metrics in self.model_metrics.items():
            if metrics.recent_error_rate > 0.2:
                recommendations.append(f"Model {name} has high error rate ({metrics.recent_error_rate:.1%})")
            
            if metrics.average_response_time > 5.0:
                recommendations.append(f"Model {name} has slow response time ({metrics.average_response_time:.1f}s)")
        
        # Cost optimization
        if len(self.models) > 1:
            costs = [(name, m.total_cost / max(1, m.successful_requests)) 
                    for name, m in self.model_metrics.items() 
                    if m.successful_requests > 0]
            
            if costs:
                costs.sort(key=lambda x: x[1])
                cheapest = costs[0][0]
                recommendations.append(f"Consider using {cheapest} more often for cost optimization")
        
        return recommendations


# Global multi-model manager
_multi_model_manager: Optional[MultiModelManager] = None


def get_multi_model_manager() -> MultiModelManager:
    """Get or create global multi-model manager."""
    global _multi_model_manager
    if _multi_model_manager is None:
        _multi_model_manager = MultiModelManager()
    return _multi_model_manager