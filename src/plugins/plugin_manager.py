"""
Plugin management system for AI Code Agent extensibility.
"""

import asyncio
import importlib
import importlib.util
import sys
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import traceback
from enum import Enum

from .plugin_interface import (
    BasePlugin, PluginResult, PluginContext, PluginType,
    AnalysisPlugin, GeneratorPlugin, IntegrationPlugin, 
    ReportingPlugin, WorkflowPlugin
)


class HookType(Enum):
    """Types of hooks available in the plugin system."""
    PRE_ANALYSIS = "pre_analysis"
    POST_ANALYSIS = "post_analysis"
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"
    PRE_WORKFLOW = "pre_workflow"
    POST_WORKFLOW = "post_workflow"
    ERROR_HANDLER = "error_handler"
    VALIDATION = "validation"
    CUSTOM = "custom"


@dataclass
class PluginMetadata:
    """Metadata about a plugin."""
    name: str
    version: str
    description: str
    author: str = ""
    plugin_type: PluginType = PluginType.CUSTOM
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    installed_at: datetime = field(default_factory=datetime.now)
    file_path: Optional[str] = None


@dataclass
class PluginHook:
    """Represents a plugin hook."""
    name: str
    hook_type: HookType
    callback: Callable
    priority: int = 50  # Lower numbers = higher priority
    conditions: Dict[str, Any] = field(default_factory=dict)
    plugin_name: str = ""


class Plugin:
    """Wrapper class for loaded plugins."""
    
    def __init__(self, plugin_instance: BasePlugin, metadata: PluginMetadata):
        self.instance = plugin_instance
        self.metadata = metadata
        self.loaded_at = datetime.now()
        self.execution_count = 0
        self.last_execution = None
        self.total_execution_time = 0.0
        self.error_count = 0
        self.last_error = None
    
    async def execute(self, context: PluginContext) -> PluginResult:
        """Execute the plugin and track statistics."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await self.instance.execute(context)
            
            # Update statistics
            self.execution_count += 1
            self.last_execution = datetime.now()
            self.total_execution_time += result.execution_time
            
            if not result.success:
                self.error_count += 1
                self.last_error = result.error_message
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.error_count += 1
            self.last_error = str(e)
            
            return PluginResult(
                plugin_name=self.metadata.name,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin execution statistics."""
        avg_execution_time = (self.total_execution_time / self.execution_count 
                             if self.execution_count > 0 else 0)
        
        return {
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.execution_count),
            "average_execution_time": avg_execution_time,
            "total_execution_time": self.total_execution_time,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_error": self.last_error,
            "loaded_at": self.loaded_at.isoformat()
        }


class PluginManager:
    """Manages plugin lifecycle and execution."""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[HookType, List[PluginHook]] = {hook_type: [] for hook_type in HookType}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.enabled_plugins: set = set()
        
        # Create plugins directory if it doesn't exist
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Load plugin configurations
        self._load_plugin_configs()
    
    def _load_plugin_configs(self):
        """Load plugin configurations from file."""
        config_file = self.plugins_dir / "plugins.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.plugin_configs = config_data.get("plugins", {})
                    self.enabled_plugins = set(config_data.get("enabled", []))
            except Exception as e:
                print(f"Error loading plugin configs: {e}")
    
    def _save_plugin_configs(self):
        """Save plugin configurations to file."""
        config_file = self.plugins_dir / "plugins.json"
        try:
            config_data = {
                "plugins": self.plugin_configs,
                "enabled": list(self.enabled_plugins),
                "last_updated": datetime.now().isoformat()
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving plugin configs: {e}")
    
    async def load_plugin_from_file(self, file_path: str) -> bool:
        """Load a plugin from a Python file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"Plugin file not found: {file_path}")
                return False
            
            # Load module from file
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                print(f"Could not load spec for {file_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin classes in the module
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    
                    # Create plugin instance
                    plugin_instance = obj()
                    
                    # Create metadata
                    metadata = PluginMetadata(
                        name=plugin_instance.name,
                        version=plugin_instance.version,
                        description=plugin_instance.description,
                        file_path=str(file_path),
                        plugin_type=self._determine_plugin_type(plugin_instance)
                    )
                    
                    # Initialize plugin
                    config = self.plugin_configs.get(plugin_instance.name, {})
                    if await plugin_instance.initialize(config):
                        # Store plugin
                        plugin = Plugin(plugin_instance, metadata)
                        self.plugins[plugin_instance.name] = plugin
                        
                        # Enable plugin if configured
                        if plugin_instance.name in self.enabled_plugins:
                            plugin.metadata.enabled = True
                        
                        print(f"Loaded plugin: {plugin_instance.name}")
                        return True
            
            print(f"No valid plugin classes found in {file_path}")
            return False
            
        except Exception as e:
            print(f"Error loading plugin from {file_path}: {e}")
            traceback.print_exc()
            return False
    
    def _determine_plugin_type(self, plugin_instance: BasePlugin) -> PluginType:
        """Determine the type of a plugin based on its class."""
        if isinstance(plugin_instance, AnalysisPlugin):
            return PluginType.ANALYSIS
        elif isinstance(plugin_instance, GeneratorPlugin):
            return PluginType.GENERATOR
        elif isinstance(plugin_instance, IntegrationPlugin):
            return PluginType.INTEGRATION
        elif isinstance(plugin_instance, ReportingPlugin):
            return PluginType.REPORTING
        elif isinstance(plugin_instance, WorkflowPlugin):
            return PluginType.WORKFLOW
        else:
            return PluginType.CUSTOM
    
    async def load_plugins_from_directory(self, directory: str = None) -> int:
        """Load all plugins from a directory."""
        plugin_dir = Path(directory) if directory else self.plugins_dir
        loaded_count = 0
        
        if not plugin_dir.exists():
            print(f"Plugin directory not found: {plugin_dir}")
            return 0
        
        # Load .py files
        for file_path in plugin_dir.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            
            if await self.load_plugin_from_file(str(file_path)):
                loaded_count += 1
        
        print(f"Loaded {loaded_count} plugins from {plugin_dir}")
        return loaded_count
    
    def register_plugin(self, plugin_instance: BasePlugin, 
                       metadata: Optional[PluginMetadata] = None) -> bool:
        """Register a plugin instance directly."""
        try:
            if metadata is None:
                metadata = PluginMetadata(
                    name=plugin_instance.name,
                    version=plugin_instance.version,
                    description=plugin_instance.description,
                    plugin_type=self._determine_plugin_type(plugin_instance)
                )
            
            plugin = Plugin(plugin_instance, metadata)
            self.plugins[plugin_instance.name] = plugin
            
            print(f"Registered plugin: {plugin_instance.name}")
            return True
            
        except Exception as e:
            print(f"Error registering plugin {plugin_instance.name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            # Call cleanup
            asyncio.create_task(plugin.instance.cleanup())
            
            # Remove from enabled set
            self.enabled_plugins.discard(plugin_name)
            
            # Remove plugin
            del self.plugins[plugin_name]
            
            # Remove associated hooks
            for hook_type in self.hooks:
                self.hooks[hook_type] = [
                    hook for hook in self.hooks[hook_type] 
                    if hook.plugin_name != plugin_name
                ]
            
            print(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            print(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].metadata.enabled = True
            self.enabled_plugins.add(plugin_name)
            self._save_plugin_configs()
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].metadata.enabled = False
            self.enabled_plugins.discard(plugin_name)
            self._save_plugin_configs()
            return True
        return False
    
    def register_hook(self, hook: PluginHook) -> bool:
        """Register a plugin hook."""
        try:
            self.hooks[hook.hook_type].append(hook)
            # Sort by priority (lower number = higher priority)
            self.hooks[hook.hook_type].sort(key=lambda h: h.priority)
            return True
        except Exception as e:
            print(f"Error registering hook {hook.name}: {e}")
            return False
    
    async def execute_hooks(self, hook_type: HookType, 
                           context: PluginContext) -> List[Any]:
        """Execute all hooks of a specific type."""
        results = []
        
        for hook in self.hooks[hook_type]:
            # Check if plugin is enabled
            if hook.plugin_name and hook.plugin_name not in self.enabled_plugins:
                continue
            
            # Check conditions
            if not self._check_hook_conditions(hook, context):
                continue
            
            try:
                if asyncio.iscoroutinefunction(hook.callback):
                    result = await hook.callback(context)
                else:
                    result = hook.callback(context)
                results.append(result)
            except Exception as e:
                print(f"Error executing hook {hook.name}: {e}")
        
        return results
    
    def _check_hook_conditions(self, hook: PluginHook, 
                              context: PluginContext) -> bool:
        """Check if hook conditions are met."""
        if not hook.conditions:
            return True
        
        for key, expected_value in hook.conditions.items():
            if key == "file_extension":
                if context.file_path:
                    actual_value = Path(context.file_path).suffix.lstrip('.')
                    if actual_value != expected_value:
                        return False
            elif key in context.metadata:
                if context.metadata[key] != expected_value:
                    return False
            elif key in context.user_config:
                if context.user_config[key] != expected_value:
                    return False
        
        return True
    
    async def execute_plugin(self, plugin_name: str, 
                            context: PluginContext) -> PluginResult:
        """Execute a specific plugin."""
        if plugin_name not in self.plugins:
            return PluginResult(
                plugin_name=plugin_name,
                success=False,
                error_message=f"Plugin '{plugin_name}' not found"
            )
        
        plugin = self.plugins[plugin_name]
        
        if not plugin.metadata.enabled:
            return PluginResult(
                plugin_name=plugin_name,
                success=False,
                error_message=f"Plugin '{plugin_name}' is disabled"
            )
        
        return await plugin.execute(context)
    
    async def execute_plugins_by_type(self, plugin_type: PluginType, 
                                     context: PluginContext) -> List[PluginResult]:
        """Execute all enabled plugins of a specific type."""
        results = []
        
        for plugin_name, plugin in self.plugins.items():
            if (plugin.metadata.plugin_type == plugin_type and 
                plugin.metadata.enabled):
                
                result = await plugin.execute(context)
                results.append(result)
        
        return results
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific plugin."""
        if plugin_name not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_name]
        
        return {
            "metadata": {
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "description": plugin.metadata.description,
                "author": plugin.metadata.author,
                "plugin_type": plugin.metadata.plugin_type.value,
                "enabled": plugin.metadata.enabled,
                "dependencies": plugin.metadata.dependencies,
                "requirements": plugin.metadata.requirements,
                "file_path": plugin.metadata.file_path
            },
            "statistics": plugin.get_statistics(),
            "config": self.plugin_configs.get(plugin_name, {})
        }
    
    def list_plugins(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List all plugins with their information."""
        plugins_info = []
        
        for plugin_name, plugin in self.plugins.items():
            if enabled_only and not plugin.metadata.enabled:
                continue
            
            info = self.get_plugin_info(plugin_name)
            if info:
                plugins_info.append(info)
        
        return plugins_info
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """Get list of plugin names by type."""
        return [
            name for name, plugin in self.plugins.items()
            if plugin.metadata.plugin_type == plugin_type
        ]
    
    def update_plugin_config(self, plugin_name: str, 
                            config: Dict[str, Any]) -> bool:
        """Update plugin configuration."""
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        
        # Validate config
        if not plugin.instance.validate_config(config):
            return False
        
        # Update config
        self.plugin_configs[plugin_name] = config
        plugin.instance.config = config
        
        # Save configs
        self._save_plugin_configs()
        
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall plugin system status."""
        total_plugins = len(self.plugins)
        enabled_plugins = len([p for p in self.plugins.values() if p.metadata.enabled])
        
        plugin_types = {}
        for plugin in self.plugins.values():
            plugin_type = plugin.metadata.plugin_type.value
            plugin_types[plugin_type] = plugin_types.get(plugin_type, 0) + 1
        
        total_executions = sum(p.execution_count for p in self.plugins.values())
        total_errors = sum(p.error_count for p in self.plugins.values())
        
        return {
            "total_plugins": total_plugins,
            "enabled_plugins": enabled_plugins,
            "disabled_plugins": total_plugins - enabled_plugins,
            "plugin_types": plugin_types,
            "total_executions": total_executions,
            "total_errors": total_errors,
            "error_rate": total_errors / max(1, total_executions),
            "plugins_directory": str(self.plugins_dir),
            "hooks_registered": sum(len(hooks) for hooks in self.hooks.values())
        }


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get or create global plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager