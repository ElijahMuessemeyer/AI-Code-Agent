"""
Tests for the plugin management system.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.plugins.plugin_manager import (
    PluginManager, Plugin, PluginMetadata, PluginHook, HookType,
    get_plugin_manager
)
from src.plugins.plugin_interface import (
    BasePlugin, PluginContext, PluginResult, PluginType,
    AnalysisPlugin, GeneratorPlugin, SampleAnalysisPlugin
)


class MockPlugin(BasePlugin):
    """Mock plugin for testing."""
    
    def __init__(self, name="mock_plugin", should_fail=False):
        super().__init__(name, "1.0.0", "Mock plugin for testing")
        self.should_fail = should_fail
        self.initialization_called = False
        self.execution_called = False
        self.cleanup_called = False
    
    async def initialize(self, config=None):
        """Mock initialization."""
        self.initialization_called = True
        self.config = config or {}
        return not self.should_fail
    
    async def execute(self, context):
        """Mock execution."""
        self.execution_called = True
        if self.should_fail:
            raise Exception("Mock plugin execution failed")
        
        return PluginResult(
            plugin_name=self.name,
            success=True,
            data={"result": "mock_success", "context_data": context.metadata},
            execution_time=0.1
        )
    
    async def cleanup(self):
        """Mock cleanup."""
        self.cleanup_called = True
        return True


class TestPluginMetadata:
    """Test cases for plugin metadata."""
    
    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.2.3",
            description="A test plugin",
            author="Test Author",
            plugin_type=PluginType.ANALYSIS,
            dependencies=["numpy", "pandas"],
            requirements=["python>=3.8"],
            enabled=True
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.2.3"
        assert metadata.description == "A test plugin"
        assert metadata.author == "Test Author"
        assert metadata.plugin_type == PluginType.ANALYSIS
        assert "numpy" in metadata.dependencies
        assert "python>=3.8" in metadata.requirements
        assert metadata.enabled is True


class TestPluginHook:
    """Test cases for plugin hooks."""
    
    def test_plugin_hook_creation(self):
        """Test creating a plugin hook."""
        def test_callback(context):
            return "hook_result"
        
        hook = PluginHook(
            name="test_hook",
            hook_type=HookType.PRE_ANALYSIS,
            callback=test_callback,
            priority=10,
            conditions={"file_extension": "py"},
            plugin_name="test_plugin"
        )
        
        assert hook.name == "test_hook"
        assert hook.hook_type == HookType.PRE_ANALYSIS
        assert hook.callback == test_callback
        assert hook.priority == 10
        assert hook.conditions["file_extension"] == "py"
        assert hook.plugin_name == "test_plugin"


class TestPlugin:
    """Test cases for Plugin wrapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_plugin = MockPlugin("test_plugin")
        self.metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            plugin_type=PluginType.CUSTOM
        )
        self.plugin = Plugin(self.mock_plugin, self.metadata)
    
    def test_plugin_creation(self):
        """Test creating a plugin wrapper."""
        assert self.plugin.instance == self.mock_plugin
        assert self.plugin.metadata == self.metadata
        assert self.plugin.execution_count == 0
        assert self.plugin.error_count == 0
        assert self.plugin.total_execution_time == 0.0
    
    @pytest.mark.asyncio
    async def test_plugin_execution_success(self):
        """Test successful plugin execution."""
        context = PluginContext(
            file_path="test.py",
            code_content="def test(): pass",
            metadata={"test": "data"}
        )
        
        result = await self.plugin.execute(context)
        
        assert result.success is True
        assert result.plugin_name == "test_plugin"
        assert "mock_success" in str(result.data)
        assert self.plugin.execution_count == 1
        assert self.plugin.error_count == 0
        assert self.plugin.total_execution_time > 0
    
    @pytest.mark.asyncio
    async def test_plugin_execution_failure(self):
        """Test plugin execution failure."""
        failing_plugin = MockPlugin("failing_plugin", should_fail=True)
        metadata = PluginMetadata(
            name="failing_plugin",
            version="1.0.0", 
            description="Failing plugin",
            plugin_type=PluginType.CUSTOM
        )
        plugin = Plugin(failing_plugin, metadata)
        
        context = PluginContext()
        result = await plugin.execute(context)
        
        assert result.success is False
        assert "Mock plugin execution failed" in result.error_message
        assert plugin.execution_count == 0  # Failure doesn't count as execution
        assert plugin.error_count == 1
    
    def test_plugin_statistics(self):
        """Test plugin statistics generation."""
        # Simulate some executions
        self.plugin.execution_count = 5
        self.plugin.error_count = 1
        self.plugin.total_execution_time = 2.5
        
        stats = self.plugin.get_statistics()
        
        assert stats["execution_count"] == 5
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.2
        assert stats["average_execution_time"] == 0.5
        assert stats["total_execution_time"] == 2.5


class TestPluginManager:
    """Test cases for plugin manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = Path(temp_dir)
            self.plugin_manager = PluginManager(str(self.temp_dir))
    
    def test_plugin_manager_creation(self):
        """Test creating a plugin manager."""
        assert len(self.plugin_manager.plugins) == 0
        assert len(self.plugin_manager.hooks) == len(HookType)
        assert len(self.plugin_manager.plugin_configs) == 0
        assert len(self.plugin_manager.enabled_plugins) == 0
    
    def test_register_plugin_success(self):
        """Test successfully registering a plugin."""
        mock_plugin = MockPlugin("registered_plugin")
        
        result = self.plugin_manager.register_plugin(mock_plugin)
        
        assert result is True
        assert "registered_plugin" in self.plugin_manager.plugins
        assert mock_plugin.initialization_called is False  # Not initialized yet
    
    def test_register_plugin_with_metadata(self):
        """Test registering a plugin with custom metadata."""
        mock_plugin = MockPlugin("custom_plugin")
        metadata = PluginMetadata(
            name="custom_plugin",
            version="2.0.0",
            description="Custom plugin",
            author="Test Author",
            plugin_type=PluginType.ANALYSIS
        )
        
        result = self.plugin_manager.register_plugin(mock_plugin, metadata)
        
        assert result is True
        plugin = self.plugin_manager.plugins["custom_plugin"]
        assert plugin.metadata.author == "Test Author"
        assert plugin.metadata.plugin_type == PluginType.ANALYSIS
    
    def test_unload_plugin(self):
        """Test unloading a plugin."""
        mock_plugin = MockPlugin("unload_test")
        self.plugin_manager.register_plugin(mock_plugin)
        
        assert "unload_test" in self.plugin_manager.plugins
        
        result = self.plugin_manager.unload_plugin("unload_test")
        
        assert result is True
        assert "unload_test" not in self.plugin_manager.plugins
        assert mock_plugin.cleanup_called is False  # Cleanup is async, called in background
    
    def test_unload_nonexistent_plugin(self):
        """Test unloading a plugin that doesn't exist."""
        result = self.plugin_manager.unload_plugin("nonexistent")
        assert result is False
    
    def test_enable_disable_plugin(self):
        """Test enabling and disabling plugins."""
        mock_plugin = MockPlugin("toggle_test")
        self.plugin_manager.register_plugin(mock_plugin)
        
        # Test enabling
        result = self.plugin_manager.enable_plugin("toggle_test")
        assert result is True
        assert "toggle_test" in self.plugin_manager.enabled_plugins
        
        # Test disabling
        result = self.plugin_manager.disable_plugin("toggle_test")
        assert result is True
        assert "toggle_test" not in self.plugin_manager.enabled_plugins
    
    def test_register_hook(self):
        """Test registering plugin hooks."""
        def test_callback(context):
            return "hook_executed"
        
        hook = PluginHook(
            name="test_hook",
            hook_type=HookType.PRE_ANALYSIS,
            callback=test_callback,
            priority=5
        )
        
        result = self.plugin_manager.register_hook(hook)
        assert result is True
        assert len(self.plugin_manager.hooks[HookType.PRE_ANALYSIS]) == 1
    
    def test_hook_priority_ordering(self):
        """Test that hooks are ordered by priority."""
        def callback1(context): return "1"
        def callback2(context): return "2"
        def callback3(context): return "3"
        
        # Register hooks with different priorities
        hook1 = PluginHook("hook1", HookType.PRE_ANALYSIS, callback1, priority=30)
        hook2 = PluginHook("hook2", HookType.PRE_ANALYSIS, callback2, priority=10)
        hook3 = PluginHook("hook3", HookType.PRE_ANALYSIS, callback3, priority=20)
        
        self.plugin_manager.register_hook(hook1)
        self.plugin_manager.register_hook(hook2)
        self.plugin_manager.register_hook(hook3)
        
        hooks = self.plugin_manager.hooks[HookType.PRE_ANALYSIS]
        
        # Should be ordered by priority (lower number = higher priority)
        assert hooks[0].priority == 10
        assert hooks[1].priority == 20
        assert hooks[2].priority == 30
    
    @pytest.mark.asyncio
    async def test_execute_hooks(self):
        """Test executing hooks."""
        results = []
        
        def hook1_callback(context):
            results.append("hook1")
            return "result1"
        
        async def hook2_callback(context):
            results.append("hook2")
            return "result2"
        
        # Register hooks
        hook1 = PluginHook("hook1", HookType.PRE_ANALYSIS, hook1_callback)
        hook2 = PluginHook("hook2", HookType.PRE_ANALYSIS, hook2_callback)
        
        self.plugin_manager.register_hook(hook1)
        self.plugin_manager.register_hook(hook2)
        
        # Execute hooks
        context = PluginContext()
        hook_results = await self.plugin_manager.execute_hooks(HookType.PRE_ANALYSIS, context)
        
        assert len(hook_results) == 2
        assert "result1" in hook_results
        assert "result2" in hook_results
        assert "hook1" in results
        assert "hook2" in results
    
    @pytest.mark.asyncio
    async def test_execute_hooks_with_conditions(self):
        """Test executing hooks with conditions."""
        def python_hook(context):
            return "python_hook_executed"
        
        def javascript_hook(context):
            return "javascript_hook_executed"
        
        # Register hooks with conditions
        python_hook_obj = PluginHook(
            "python_hook", HookType.PRE_ANALYSIS, python_hook,
            conditions={"file_extension": "py"}
        )
        javascript_hook_obj = PluginHook(
            "javascript_hook", HookType.PRE_ANALYSIS, javascript_hook,
            conditions={"file_extension": "js"}
        )
        
        self.plugin_manager.register_hook(python_hook_obj)
        self.plugin_manager.register_hook(javascript_hook_obj)
        
        # Test with Python file
        python_context = PluginContext(file_path="test.py")
        results = await self.plugin_manager.execute_hooks(HookType.PRE_ANALYSIS, python_context)
        
        assert "python_hook_executed" in results
        assert "javascript_hook_executed" not in results
        
        # Test with JavaScript file
        js_context = PluginContext(file_path="test.js")
        results = await self.plugin_manager.execute_hooks(HookType.PRE_ANALYSIS, js_context)
        
        assert "javascript_hook_executed" in results
        assert "python_hook_executed" not in results
    
    @pytest.mark.asyncio
    async def test_execute_plugin(self):
        """Test executing a specific plugin."""
        mock_plugin = MockPlugin("execution_test")
        self.plugin_manager.register_plugin(mock_plugin)
        self.plugin_manager.enable_plugin("execution_test")
        
        context = PluginContext(metadata={"test": "data"})
        result = await self.plugin_manager.execute_plugin("execution_test", context)
        
        assert result.success is True
        assert result.plugin_name == "execution_test"
        assert mock_plugin.execution_called is True
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_plugin(self):
        """Test executing a plugin that doesn't exist."""
        context = PluginContext()
        result = await self.plugin_manager.execute_plugin("nonexistent", context)
        
        assert result.success is False
        assert "not found" in result.error_message
    
    @pytest.mark.asyncio
    async def test_execute_disabled_plugin(self):
        """Test executing a disabled plugin."""
        mock_plugin = MockPlugin("disabled_test")
        self.plugin_manager.register_plugin(mock_plugin)
        # Plugin is not enabled
        
        context = PluginContext()
        result = await self.plugin_manager.execute_plugin("disabled_test", context)
        
        assert result.success is False
        assert "disabled" in result.error_message
    
    @pytest.mark.asyncio
    async def test_execute_plugins_by_type(self):
        """Test executing all plugins of a specific type."""
        # Register analysis plugins
        analysis_plugin1 = MockPlugin("analysis1")
        analysis_plugin2 = MockPlugin("analysis2")
        
        metadata1 = PluginMetadata("analysis1", "1.0.0", "Analysis 1", plugin_type=PluginType.ANALYSIS)
        metadata2 = PluginMetadata("analysis2", "1.0.0", "Analysis 2", plugin_type=PluginType.ANALYSIS)
        
        self.plugin_manager.register_plugin(analysis_plugin1, metadata1)
        self.plugin_manager.register_plugin(analysis_plugin2, metadata2)
        self.plugin_manager.enable_plugin("analysis1")
        self.plugin_manager.enable_plugin("analysis2")
        
        # Register a generator plugin (should not be executed)
        generator_plugin = MockPlugin("generator1")
        generator_metadata = PluginMetadata("generator1", "1.0.0", "Generator", plugin_type=PluginType.GENERATOR)
        self.plugin_manager.register_plugin(generator_plugin, generator_metadata)
        self.plugin_manager.enable_plugin("generator1")
        
        # Execute analysis plugins
        context = PluginContext()
        results = await self.plugin_manager.execute_plugins_by_type(PluginType.ANALYSIS, context)
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert analysis_plugin1.execution_called is True
        assert analysis_plugin2.execution_called is True
        assert generator_plugin.execution_called is False
    
    def test_get_plugin_info(self):
        """Test getting plugin information."""
        mock_plugin = MockPlugin("info_test")
        metadata = PluginMetadata(
            name="info_test",
            version="1.5.0",
            description="Info test plugin",
            author="Test Author",
            plugin_type=PluginType.ANALYSIS
        )
        self.plugin_manager.register_plugin(mock_plugin, metadata)
        
        info = self.plugin_manager.get_plugin_info("info_test")
        
        assert info is not None
        assert info["metadata"]["name"] == "info_test"
        assert info["metadata"]["version"] == "1.5.0"
        assert info["metadata"]["author"] == "Test Author"
        assert info["metadata"]["plugin_type"] == "analysis"
        assert "statistics" in info
        assert "config" in info
    
    def test_get_plugin_info_nonexistent(self):
        """Test getting info for nonexistent plugin."""
        info = self.plugin_manager.get_plugin_info("nonexistent")
        assert info is None
    
    def test_list_plugins(self):
        """Test listing all plugins."""
        # Register some plugins
        plugin1 = MockPlugin("list_test1")
        plugin2 = MockPlugin("list_test2")
        
        self.plugin_manager.register_plugin(plugin1)
        self.plugin_manager.register_plugin(plugin2)
        self.plugin_manager.enable_plugin("list_test1")
        
        # List all plugins
        all_plugins = self.plugin_manager.list_plugins()
        assert len(all_plugins) == 2
        
        # List only enabled plugins
        enabled_plugins = self.plugin_manager.list_plugins(enabled_only=True)
        assert len(enabled_plugins) == 1
        assert enabled_plugins[0]["metadata"]["name"] == "list_test1"
    
    def test_get_plugins_by_type(self):
        """Test getting plugins by type."""
        # Register plugins of different types
        analysis_plugin = MockPlugin("analysis_type")
        generator_plugin = MockPlugin("generator_type")
        
        analysis_metadata = PluginMetadata("analysis_type", "1.0.0", "Analysis", plugin_type=PluginType.ANALYSIS)
        generator_metadata = PluginMetadata("generator_type", "1.0.0", "Generator", plugin_type=PluginType.GENERATOR)
        
        self.plugin_manager.register_plugin(analysis_plugin, analysis_metadata)
        self.plugin_manager.register_plugin(generator_plugin, generator_metadata)
        
        analysis_plugins = self.plugin_manager.get_plugins_by_type(PluginType.ANALYSIS)
        generator_plugins = self.plugin_manager.get_plugins_by_type(PluginType.GENERATOR)
        
        assert "analysis_type" in analysis_plugins
        assert "generator_type" in generator_plugins
        assert len(analysis_plugins) == 1
        assert len(generator_plugins) == 1
    
    def test_update_plugin_config(self):
        """Test updating plugin configuration."""
        mock_plugin = MockPlugin("config_test")
        self.plugin_manager.register_plugin(mock_plugin)
        
        config = {"setting1": "value1", "setting2": 42}
        result = self.plugin_manager.update_plugin_config("config_test", config)
        
        assert result is True
        assert mock_plugin.config == config
        assert "config_test" in self.plugin_manager.plugin_configs
        assert self.plugin_manager.plugin_configs["config_test"] == config
    
    def test_get_system_status(self):
        """Test getting system status."""
        # Register some plugins
        plugin1 = MockPlugin("status1")
        plugin2 = MockPlugin("status2")
        
        analysis_metadata = PluginMetadata("status1", "1.0.0", "Status 1", plugin_type=PluginType.ANALYSIS)
        generator_metadata = PluginMetadata("status2", "1.0.0", "Status 2", plugin_type=PluginType.GENERATOR)
        
        self.plugin_manager.register_plugin(plugin1, analysis_metadata)
        self.plugin_manager.register_plugin(plugin2, generator_metadata)
        self.plugin_manager.enable_plugin("status1")
        
        status = self.plugin_manager.get_system_status()
        
        assert status["total_plugins"] == 2
        assert status["enabled_plugins"] == 1
        assert status["disabled_plugins"] == 1
        assert "plugin_types" in status
        assert status["plugin_types"]["analysis"] == 1
        assert status["plugin_types"]["generator"] == 1


class TestSamplePlugins:
    """Test cases for sample plugin implementations."""
    
    @pytest.mark.asyncio
    async def test_sample_analysis_plugin(self):
        """Test the sample analysis plugin."""
        plugin = SampleAnalysisPlugin()
        
        # Initialize plugin
        success = await plugin.initialize()
        assert success is True
        
        # Test code analysis
        code_content = "def hello_world():\n    print('Hello, World!')\n    return True"
        file_path = "test.py"
        context = PluginContext(code_content=code_content, file_path=file_path)
        
        result = await plugin.execute(context)
        
        assert result.success is True
        assert result.plugin_name == "sample_analysis"
        assert "line_count" in result.data
        assert "char_count" in result.data
        assert result.data["line_count"] == 3
        assert result.data["file_extension"] == "py"
    
    @pytest.mark.asyncio
    async def test_sample_analysis_plugin_non_python(self):
        """Test sample analysis plugin with non-Python file."""
        plugin = SampleAnalysisPlugin()
        await plugin.initialize()
        
        context = PluginContext(
            code_content="function test() { return true; }",
            file_path="test.js"
        )
        
        result = await plugin.execute(context)
        
        assert result.success is True
        assert result.data["file_extension"] == "js"


def test_global_plugin_manager():
    """Test global plugin manager singleton."""
    manager1 = get_plugin_manager()
    manager2 = get_plugin_manager()
    
    assert manager1 is manager2  # Should be same instance


if __name__ == "__main__":
    pytest.main([__file__])