"""
Plugin system for AI Code Agent extensibility.
"""

from .plugin_manager import (
    PluginManager,
    Plugin,
    PluginMetadata,
    PluginHook,
    HookType,
    get_plugin_manager
)

from .plugin_interface import (
    AnalysisPlugin,
    GeneratorPlugin,
    IntegrationPlugin,
    ReportingPlugin
)

__all__ = [
    "PluginManager",
    "Plugin",
    "PluginMetadata", 
    "PluginHook",
    "HookType",
    "get_plugin_manager",
    "AnalysisPlugin",
    "GeneratorPlugin",
    "IntegrationPlugin",
    "ReportingPlugin"
]