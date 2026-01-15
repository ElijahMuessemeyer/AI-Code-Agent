"""
Tests for the multi-model manager system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.llm.multi_model_manager import (
    MultiModelManager, ModelConfig, ModelMetrics, ModelCapability,
    RoutingStrategy, LoadBalancer, CircuitBreaker
)
from src.llm.model_interface import ModelProvider, ModelResponse


class TestModelMetrics:
    """Test cases for model metrics tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = ModelMetrics()
    
    def test_initial_metrics(self):
        """Test initial metrics state."""
        assert self.metrics.total_requests == 0
        assert self.metrics.successful_requests == 0
        assert self.metrics.failed_requests == 0
        assert self.metrics.success_rate == 0.0
        assert self.metrics.average_response_time == 0.0
    
    def test_update_successful_request(self):
        """Test updating metrics for successful requests."""
        self.metrics.update_request(
            success=True,
            response_time=1.5,
            tokens=100,
            cost=0.01,
            quality_score=8.5
        )
        
        assert self.metrics.total_requests == 1
        assert self.metrics.successful_requests == 1
        assert self.metrics.failed_requests == 0
        assert self.metrics.success_rate == 1.0
        assert self.metrics.average_response_time == 1.5
        assert self.metrics.total_tokens == 100
        assert self.metrics.total_cost == 0.01
        assert self.metrics.average_quality_score == 8.5
    
    def test_update_failed_request(self):
        """Test updating metrics for failed requests."""
        self.metrics.update_request(success=False, response_time=0.5)
        
        assert self.metrics.total_requests == 1
        assert self.metrics.successful_requests == 0
        assert self.metrics.failed_requests == 1
        assert self.metrics.success_rate == 0.0
        assert self.metrics.average_response_time == 0.0  # Only successful requests count
    
    def test_error_rate_window(self):
        """Test error rate window tracking."""
        # Add successful requests
        for _ in range(5):
            self.metrics.update_request(success=True, response_time=1.0)
        
        # Add failed requests
        for _ in range(3):
            self.metrics.update_request(success=False, response_time=0.5)
        
        # Error rate should be 3/8 = 0.375
        assert abs(self.metrics.recent_error_rate - 0.375) < 0.001
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        self.metrics.update_request(success=True, response_time=2.0, tokens=50)
        result = self.metrics.to_dict()
        
        assert isinstance(result, dict)
        assert "total_requests" in result
        assert "success_rate" in result
        assert "average_response_time" in result
        assert result["total_requests"] == 1


class TestLoadBalancer:
    """Test cases for load balancer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.load_balancer = LoadBalancer(RoutingStrategy.ROUND_ROBIN)
        self.model_configs = {
            "model1": ModelConfig(
                name="model1",
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                api_key="test-key",
                capabilities=[ModelCapability.CODE_ANALYSIS],
                priority=1
            ),
            "model2": ModelConfig(
                name="model2", 
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet",
                api_key="test-key",
                capabilities=[ModelCapability.CODE_GENERATION],
                priority=2
            )
        }
        self.model_metrics = {
            "model1": ModelMetrics(),
            "model2": ModelMetrics()
        }
    
    def test_round_robin_selection(self):
        """Test round-robin model selection."""
        available_models = ["model1", "model2"]
        
        # First selection should be model1
        selected = self.load_balancer.select_model(
            available_models, self.model_metrics, self.model_configs
        )
        assert selected == "model1"
        
        # Second selection should be model2
        selected = self.load_balancer.select_model(
            available_models, self.model_metrics, self.model_configs
        )
        assert selected == "model2"
        
        # Third selection should be model1 again
        selected = self.load_balancer.select_model(
            available_models, self.model_metrics, self.model_configs
        )
        assert selected == "model1"
    
    def test_capability_filtering(self):
        """Test filtering models by capability."""
        available_models = ["model1", "model2"]
        
        # Select model for code analysis (only model1 supports this)
        selected = self.load_balancer.select_model(
            available_models, self.model_metrics, self.model_configs,
            capability=ModelCapability.CODE_ANALYSIS
        )
        assert selected == "model1"
        
        # Select model for code generation (only model2 supports this)
        selected = self.load_balancer.select_model(
            available_models, self.model_metrics, self.model_configs,
            capability=ModelCapability.CODE_GENERATION
        )
        assert selected == "model2"
    
    def test_no_available_models(self):
        """Test behavior when no models are available."""
        selected = self.load_balancer.select_model(
            [], self.model_metrics, self.model_configs
        )
        assert selected is None
    
    def test_request_tracking(self):
        """Test tracking of active requests."""
        self.load_balancer.track_request_start("model1")
        assert self.load_balancer.active_requests["model1"] == 1
        
        self.load_balancer.track_request_start("model1")
        assert self.load_balancer.active_requests["model1"] == 2
        
        self.load_balancer.track_request_end("model1")
        assert self.load_balancer.active_requests["model1"] == 1
        
        self.load_balancer.track_request_end("model1")
        assert self.load_balancer.active_requests["model1"] == 0


class TestCircuitBreaker:
    """Test cases for circuit breaker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)
    
    def test_initial_state(self):
        """Test initial circuit breaker state."""
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.can_execute() is True
        assert self.circuit_breaker.failure_count == 0
    
    def test_failure_tracking(self):
        """Test failure count tracking."""
        # Record failures below threshold
        self.circuit_breaker.record_failure()
        self.circuit_breaker.record_failure()
        
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.can_execute() is True
        assert self.circuit_breaker.failure_count == 2
    
    def test_circuit_opens_on_threshold(self):
        """Test circuit opens when failure threshold is reached."""
        # Record failures to reach threshold
        for _ in range(3):
            self.circuit_breaker.record_failure()
        
        assert self.circuit_breaker.state == "open"
        assert self.circuit_breaker.can_execute() is False
    
    def test_success_resets_failures(self):
        """Test success resets failure count."""
        self.circuit_breaker.record_failure()
        self.circuit_breaker.record_failure()
        
        self.circuit_breaker.record_success()
        
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.state == "closed"


class TestMultiModelManager:
    """Test cases for multi-model manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MultiModelManager()
    
    def test_initial_state(self):
        """Test initial manager state."""
        assert len(self.manager.models) == 0
        assert len(self.manager.model_configs) == 0
        assert len(self.manager.model_metrics) == 0
        assert len(self.manager.circuit_breakers) == 0
    
    @patch('src.llm.model_interface.ModelManager')
    def test_add_model_success(self, mock_model_manager):
        """Test successfully adding a model."""
        # Mock the model manager
        mock_instance = Mock()
        mock_model_manager.return_value = mock_instance
        
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        result = self.manager.add_model(config)
        
        assert result is True
        assert "test_model" in self.manager.models
        assert "test_model" in self.manager.model_configs
        assert "test_model" in self.manager.model_metrics
        assert "test_model" in self.manager.circuit_breakers
    
    def test_remove_model(self):
        """Test removing a model."""
        # First add a model manually
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        self.manager.model_configs["test_model"] = config
        self.manager.model_metrics["test_model"] = ModelMetrics()
        self.manager.circuit_breakers["test_model"] = CircuitBreaker()
        self.manager.models["test_model"] = Mock()
        
        result = self.manager.remove_model("test_model")
        
        assert result is True
        assert "test_model" not in self.manager.models
        assert "test_model" not in self.manager.model_configs
        assert "test_model" not in self.manager.model_metrics
        assert "test_model" not in self.manager.circuit_breakers
    
    def test_remove_nonexistent_model(self):
        """Test removing a model that doesn't exist."""
        result = self.manager.remove_model("nonexistent_model")
        assert result is False
    
    def test_enable_disable_model(self):
        """Test enabling and disabling models."""
        # Add a model first
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key"
        )
        self.manager.model_configs["test_model"] = config
        
        # Test disabling
        result = self.manager.disable_model("test_model")
        assert result is True
        assert self.manager.model_configs["test_model"].enabled is False
        
        # Test enabling
        result = self.manager.enable_model("test_model")
        assert result is True
        assert self.manager.model_configs["test_model"].enabled is True
    
    def test_fallback_chain(self):
        """Test setting and using fallback chain."""
        # Add some models
        self.manager.models["model1"] = Mock()
        self.manager.models["model2"] = Mock()
        self.manager.models["model3"] = Mock()
        
        self.manager.set_fallback_chain(["model1", "model2", "model3"])
        
        assert self.manager.fallback_chain == ["model1", "model2", "model3"]
        
        # Test with non-existent model
        self.manager.set_fallback_chain(["model1", "nonexistent", "model2"])
        assert self.manager.fallback_chain == ["model1", "model2"]
    
    @pytest.mark.asyncio
    async def test_generate_response_no_models(self):
        """Test response generation with no models available."""
        with pytest.raises(Exception, match="All models failed"):
            await self.manager.generate_response("test prompt")
    
    @pytest.mark.asyncio
    @patch('src.performance.cache_manager.get_cache_manager')
    async def test_generate_response_success(self, mock_cache_manager):
        """Test successful response generation."""
        # Mock cache manager
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock(return_value=True)
        mock_cache_manager.return_value = mock_cache
        
        # Add a mock model
        mock_model_manager = Mock()
        mock_response = ModelResponse(
            content="Generated response",
            finish_reason="stop",
            tokens_used=50,
            model="test-model"
        )
        mock_model_manager.generate_response = AsyncMock(return_value=mock_response)
        
        self.manager.models["test_model"] = mock_model_manager
        self.manager.model_configs["test_model"] = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key",
            enabled=True
        )
        self.manager.model_metrics["test_model"] = ModelMetrics()
        self.manager.circuit_breakers["test_model"] = CircuitBreaker()
        
        response = await self.manager.generate_response("test prompt")
        
        assert response.content == "Generated response"
        assert response.finish_reason == "stop"
        assert response.tokens_used == 50
    
    def test_get_model_status(self):
        """Test getting model status."""
        # Add a mock model
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key",
            capabilities=[ModelCapability.CODE_ANALYSIS]
        )
        
        self.manager.models["test_model"] = Mock()
        self.manager.model_configs["test_model"] = config
        self.manager.model_metrics["test_model"] = ModelMetrics()
        self.manager.circuit_breakers["test_model"] = CircuitBreaker()
        
        status = self.manager.get_model_status()
        
        assert status["total_models"] == 1
        assert status["active_models"] == 1
        assert "test_model" in status["models"]
        assert status["models"]["test_model"]["provider"] == "openai"
        assert status["models"]["test_model"]["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        # Add a mock model
        mock_model_manager = Mock()
        mock_response = ModelResponse(
            content="OK",
            finish_reason="stop",
            tokens_used=1,
            model="test-model"
        )
        mock_model_manager.generate_response = AsyncMock(return_value=mock_response)
        
        self.manager.models["test_model"] = mock_model_manager
        
        health_results = await self.manager.health_check()
        
        assert "test_model" in health_results
        assert health_results["test_model"]["status"] == "healthy"
        assert "response_time" in health_results["test_model"]
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check with model failure."""
        # Add a mock model that fails
        mock_model_manager = Mock()
        mock_model_manager.generate_response = AsyncMock(
            side_effect=Exception("Model error")
        )
        
        self.manager.models["test_model"] = mock_model_manager
        
        health_results = await self.manager.health_check()
        
        assert "test_model" in health_results
        assert health_results["test_model"]["status"] == "unhealthy"
        assert "error" in health_results["test_model"]
    
    def test_performance_report(self):
        """Test performance report generation."""
        # Add some mock data
        metrics = ModelMetrics()
        metrics.update_request(success=True, response_time=1.5, tokens=100, cost=0.01)
        metrics.update_request(success=True, response_time=2.0, tokens=150, cost=0.015)
        
        self.manager.model_metrics["test_model"] = metrics
        self.manager.model_configs["test_model"] = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key",
            enabled=True
        )
        
        report = self.manager.get_performance_report()
        
        assert "summary" in report
        assert "model_performance" in report
        assert "recommendations" in report
        assert report["summary"]["total_requests"] == 2
        assert report["summary"]["total_cost"] == 0.025


if __name__ == "__main__":
    pytest.main([__file__])