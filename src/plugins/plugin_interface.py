"""
Plugin interfaces and base classes for AI Code Agent extensibility.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PluginType(Enum):
    """Types of plugins supported."""
    ANALYSIS = "analysis"
    GENERATOR = "generator"
    INTEGRATION = "integration"
    REPORTING = "reporting"
    WORKFLOW = "workflow"
    CUSTOM = "custom"


@dataclass
class PluginResult:
    """Represents the result of a plugin execution."""
    plugin_name: str
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PluginContext:
    """Context information passed to plugins."""
    file_path: Optional[str] = None
    code_content: Optional[str] = None
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    user_config: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePlugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.enabled = True
        self.config = {}
        self.dependencies = []
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    async def execute(self, context: PluginContext) -> PluginResult:
        """Execute the plugin's main functionality."""
        pass
    
    async def cleanup(self) -> bool:
        """Cleanup resources when plugin is unloaded."""
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        return True
    
    def get_required_config_keys(self) -> List[str]:
        """Get list of required configuration keys."""
        return []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
            "dependencies": self.dependencies,
            "config_keys": self.get_required_config_keys()
        }


class AnalysisPlugin(BasePlugin):
    """Base class for code analysis plugins."""
    
    @abstractmethod
    async def analyze_code(self, code_content: str, file_path: str, 
                          context: PluginContext) -> Dict[str, Any]:
        """Analyze code and return analysis results."""
        pass
    
    async def execute(self, context: PluginContext) -> PluginResult:
        """Execute code analysis."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not context.code_content or not context.file_path:
                return PluginResult(
                    plugin_name=self.name,
                    success=False,
                    error_message="Code content and file path required for analysis"
                )
            
            analysis_result = await self.analyze_code(
                context.code_content, 
                context.file_path, 
                context
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return PluginResult(
                plugin_name=self.name,
                success=True,
                data=analysis_result,
                execution_time=execution_time,
                metadata={"analysis_type": "code_analysis"}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return PluginResult(
                plugin_name=self.name,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )


class GeneratorPlugin(BasePlugin):
    """Base class for code generation plugins."""
    
    @abstractmethod
    async def generate_code(self, prompt: str, language: str, 
                           context: PluginContext) -> str:
        """Generate code based on prompt and language."""
        pass
    
    async def execute(self, context: PluginContext) -> PluginResult:
        """Execute code generation."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            prompt = context.metadata.get("prompt", "")
            language = context.metadata.get("language", "python")
            
            if not prompt:
                return PluginResult(
                    plugin_name=self.name,
                    success=False,
                    error_message="Prompt required for code generation"
                )
            
            generated_code = await self.generate_code(prompt, language, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return PluginResult(
                plugin_name=self.name,
                success=True,
                data={"generated_code": generated_code, "language": language},
                execution_time=execution_time,
                metadata={"generation_type": "code_generation"}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return PluginResult(
                plugin_name=self.name,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )


class IntegrationPlugin(BasePlugin):
    """Base class for external integration plugins."""
    
    @abstractmethod
    async def send_data(self, data: Dict[str, Any], 
                       destination: str, context: PluginContext) -> bool:
        """Send data to external system."""
        pass
    
    @abstractmethod
    async def receive_data(self, source: str, 
                          context: PluginContext) -> Dict[str, Any]:
        """Receive data from external system."""
        pass
    
    async def execute(self, context: PluginContext) -> PluginResult:
        """Execute integration operation."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            operation = context.metadata.get("operation", "send")
            
            if operation == "send":
                data = context.metadata.get("data", {})
                destination = context.metadata.get("destination", "")
                
                success = await self.send_data(data, destination, context)
                result_data = {"operation": "send", "success": success}
                
            elif operation == "receive":
                source = context.metadata.get("source", "")
                received_data = await self.receive_data(source, context)
                result_data = {"operation": "receive", "data": received_data}
                success = True
                
            else:
                return PluginResult(
                    plugin_name=self.name,
                    success=False,
                    error_message=f"Unknown operation: {operation}"
                )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return PluginResult(
                plugin_name=self.name,
                success=success,
                data=result_data,
                execution_time=execution_time,
                metadata={"integration_type": operation}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return PluginResult(
                plugin_name=self.name,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )


class ReportingPlugin(BasePlugin):
    """Base class for reporting and visualization plugins."""
    
    @abstractmethod
    async def generate_report(self, data: Dict[str, Any], 
                             report_type: str, context: PluginContext) -> str:
        """Generate a report from data."""
        pass
    
    @abstractmethod
    async def create_visualization(self, data: Dict[str, Any], 
                                  viz_type: str, context: PluginContext) -> str:
        """Create a visualization from data."""
        pass
    
    async def execute(self, context: PluginContext) -> PluginResult:
        """Execute reporting operation."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            operation = context.metadata.get("operation", "report")
            data = context.metadata.get("data", {})
            
            if operation == "report":
                report_type = context.metadata.get("report_type", "summary")
                report_content = await self.generate_report(data, report_type, context)
                result_data = {"report": report_content, "type": report_type}
                
            elif operation == "visualization":
                viz_type = context.metadata.get("viz_type", "chart")
                viz_content = await self.create_visualization(data, viz_type, context)
                result_data = {"visualization": viz_content, "type": viz_type}
                
            else:
                return PluginResult(
                    plugin_name=self.name,
                    success=False,
                    error_message=f"Unknown operation: {operation}"
                )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return PluginResult(
                plugin_name=self.name,
                success=True,
                data=result_data,
                execution_time=execution_time,
                metadata={"reporting_type": operation}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return PluginResult(
                plugin_name=self.name,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )


class WorkflowPlugin(BasePlugin):
    """Base class for workflow orchestration plugins."""
    
    @abstractmethod
    async def execute_workflow(self, workflow_config: Dict[str, Any], 
                              context: PluginContext) -> Dict[str, Any]:
        """Execute a custom workflow."""
        pass
    
    async def execute(self, context: PluginContext) -> PluginResult:
        """Execute workflow."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            workflow_config = context.metadata.get("workflow_config", {})
            
            if not workflow_config:
                return PluginResult(
                    plugin_name=self.name,
                    success=False,
                    error_message="Workflow configuration required"
                )
            
            workflow_result = await self.execute_workflow(workflow_config, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return PluginResult(
                plugin_name=self.name,
                success=True,
                data=workflow_result,
                execution_time=execution_time,
                metadata={"workflow_type": "custom"}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return PluginResult(
                plugin_name=self.name,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )


# Example plugin implementations
class SampleAnalysisPlugin(AnalysisPlugin):
    """Sample analysis plugin for demonstration."""
    
    def __init__(self):
        super().__init__(
            name="sample_analysis",
            version="1.0.0",
            description="Sample plugin for basic code analysis"
        )
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the sample plugin."""
        self.config = config or {}
        return True
    
    async def analyze_code(self, code_content: str, file_path: str, 
                          context: PluginContext) -> Dict[str, Any]:
        """Perform basic code analysis."""
        lines = code_content.split('\n')
        
        return {
            "line_count": len(lines),
            "char_count": len(code_content),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "file_extension": file_path.split('.')[-1] if '.' in file_path else "unknown"
        }


class SampleGeneratorPlugin(GeneratorPlugin):
    """Sample code generator plugin for demonstration."""
    
    def __init__(self):
        super().__init__(
            name="sample_generator",
            version="1.0.0",
            description="Sample plugin for basic code generation"
        )
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the sample plugin."""
        self.config = config or {}
        return True
    
    async def generate_code(self, prompt: str, language: str, 
                           context: PluginContext) -> str:
        """Generate simple code based on prompt."""
        if language.lower() == "python":
            return f'"""\n{prompt}\n"""\n\ndef generated_function():\n    pass\n'
        elif language.lower() == "javascript":
            return f'/*\n{prompt}\n*/\n\nfunction generatedFunction() {{\n    // Implementation here\n}}\n'
        else:
            return f'// Generated from prompt: {prompt}\n// Language: {language}\n'


class SampleIntegrationPlugin(IntegrationPlugin):
    """Sample integration plugin for demonstration."""
    
    def __init__(self):
        super().__init__(
            name="sample_integration",
            version="1.0.0",
            description="Sample plugin for external integration"
        )
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the sample plugin."""
        self.config = config or {}
        return True
    
    async def send_data(self, data: Dict[str, Any], 
                       destination: str, context: PluginContext) -> bool:
        """Mock sending data to external system."""
        # In a real plugin, this would send data to an external API
        print(f"Sending data to {destination}: {data}")
        return True
    
    async def receive_data(self, source: str, 
                          context: PluginContext) -> Dict[str, Any]:
        """Mock receiving data from external system."""
        # In a real plugin, this would fetch data from an external API
        return {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "data": {"sample": "data", "from": source}
        }


class SampleReportingPlugin(ReportingPlugin):
    """Sample reporting plugin for demonstration."""
    
    def __init__(self):
        super().__init__(
            name="sample_reporting",
            version="1.0.0",
            description="Sample plugin for generating reports"
        )
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the sample plugin."""
        self.config = config or {}
        return True
    
    async def generate_report(self, data: Dict[str, Any], 
                             report_type: str, context: PluginContext) -> str:
        """Generate a simple text report."""
        report = f"# {report_type.title()} Report\n\n"
        report += f"Generated at: {datetime.now().isoformat()}\n\n"
        
        for key, value in data.items():
            report += f"**{key}**: {value}\n"
        
        return report
    
    async def create_visualization(self, data: Dict[str, Any], 
                                  viz_type: str, context: PluginContext) -> str:
        """Create a simple ASCII visualization."""
        if viz_type == "bar":
            viz = f"# {viz_type.title()} Chart\n\n"
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    bar = "█" * int(value / 10)  # Scale for display
                    viz += f"{key:15} |{bar} {value}\n"
            return viz
        else:
            return f"Visualization type '{viz_type}' not supported by sample plugin"