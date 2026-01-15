"""
LLM model interface for integrating with various AI providers.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelResponse:
    """Response from an LLM model."""
    content: str
    model: str
    provider: ModelProvider
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    provider: ModelProvider
    model_name: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 30


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    async def generate_response(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the model configuration."""
        pass


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if openai is None:
            raise ImportError("OpenAI package not installed")
        
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        return (
            self.config.api_key and 
            self.config.model_name and
            self.config.provider == ModelProvider.OPENAI
        )
    
    async def generate_response(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              **kwargs) -> ModelResponse:
        """Generate response using OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                **kwargs
            )
            
            choice = response.choices[0]
            
            return ModelResponse(
                content=choice.message.content or "",
                model=self.config.model_name,
                provider=ModelProvider.OPENAI,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=choice.finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class AnthropicInterface(LLMInterface):
    """Interface for Anthropic models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if anthropic is None:
            raise ImportError("Anthropic package not installed")
        
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration."""
        return (
            self.config.api_key and 
            self.config.model_name and
            self.config.provider == ModelProvider.ANTHROPIC
        )
    
    async def generate_response(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              **kwargs) -> ModelResponse:
        """Generate response using Anthropic API."""
        try:
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                timeout=self.config.timeout,
                **kwargs
            )
            
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
            
            return ModelResponse(
                content=content,
                model=self.config.model_name,
                provider=ModelProvider.ANTHROPIC,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens if response.usage else None,
                finish_reason=response.stop_reason,
                metadata={
                    "input_tokens": response.usage.input_tokens if response.usage else None,
                    "output_tokens": response.usage.output_tokens if response.usage else None
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")


class ModelManager:
    """Manager for multiple LLM models."""
    
    def __init__(self):
        self.models: Dict[str, LLMInterface] = {}
        self.default_model: Optional[str] = None
    
    def add_model(self, name: str, interface: LLMInterface) -> None:
        """Add a model interface."""
        if not interface.validate_config():
            raise ValueError(f"Invalid configuration for model {name}")
        
        self.models[name] = interface
        
        if self.default_model is None:
            self.default_model = name
    
    def set_default_model(self, name: str) -> None:
        """Set the default model."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        self.default_model = name
    
    async def generate_response(self, 
                              prompt: str,
                              model_name: Optional[str] = None,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> ModelResponse:
        """Generate response using specified or default model."""
        model_name = model_name or self.default_model
        
        if not model_name or model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        return await self.models[model_name].generate_response(
            prompt, system_prompt, **kwargs
        )
    
    def list_models(self) -> List[str]:
        """List available model names."""
        return list(self.models.keys())
    
    @classmethod
    def from_env(cls) -> 'ModelManager':
        """Create ModelManager from environment variables."""
        manager = cls()
        
        # Add OpenAI model if configured
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai:
            config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name=os.getenv('OPENAI_MODEL', 'gpt-4'),
                api_key=openai_key,
                temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '4000'))
            )
            try:
                manager.add_model('openai', OpenAIInterface(config))
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI model: {e}")
        
        # Add Anthropic model if configured
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key and anthropic:
            config = ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name=os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
                api_key=anthropic_key,
                temperature=float(os.getenv('ANTHROPIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('ANTHROPIC_MAX_TOKENS', '4000'))
            )
            try:
                manager.add_model('anthropic', AnthropicInterface(config))
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic model: {e}")
        
        return manager


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager.from_env()
    return _model_manager


async def generate_response(prompt: str, 
                          model_name: Optional[str] = None,
                          system_prompt: Optional[str] = None,
                          **kwargs) -> ModelResponse:
    """Convenience function to generate a response."""
    manager = get_model_manager()
    return await manager.generate_response(prompt, model_name, system_prompt, **kwargs)