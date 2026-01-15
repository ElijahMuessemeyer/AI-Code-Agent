"""
Sample plugins demonstrating the AI Code Agent plugin system.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from src.plugins.plugin_interface import (
    AnalysisPlugin, GeneratorPlugin, IntegrationPlugin, 
    ReportingPlugin, PluginContext
)


class PylintAnalysisPlugin(AnalysisPlugin):
    """Plugin that performs Pylint analysis on Python code."""
    
    def __init__(self):
        super().__init__(
            name="pylint_analysis",
            version="1.0.0",
            description="Advanced Python code analysis using Pylint"
        )
        self.dependencies = ["pylint"]
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the Pylint plugin."""
        self.config = config or {}
        self.pylint_options = self.config.get("pylint_options", [
            "--disable=C0111",  # Missing docstring
            "--disable=R0903",  # Too few public methods
        ])
        return True
    
    def get_required_config_keys(self) -> List[str]:
        """Get required configuration keys."""
        return ["pylint_options"]
    
    async def analyze_code(self, code_content: str, file_path: str, 
                          context: PluginContext) -> Dict[str, Any]:
        """Analyze Python code using Pylint."""
        if not file_path.endswith('.py'):
            return {"error": "Pylint analysis only supports Python files"}
        
        try:
            # In a real implementation, this would run pylint
            # For demo purposes, we'll simulate pylint results
            issues = self._simulate_pylint_analysis(code_content)
            
            return {
                "tool": "pylint",
                "file_path": file_path,
                "issues": issues,
                "issue_count": len(issues),
                "severity_breakdown": self._categorize_issues(issues)
            }
            
        except Exception as e:
            return {"error": f"Pylint analysis failed: {str(e)}"}
    
    def _simulate_pylint_analysis(self, code_content: str) -> List[Dict[str, Any]]:
        """Simulate Pylint analysis results."""
        issues = []
        lines = code_content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for long lines
            if len(line) > 100:
                issues.append({
                    "line": i,
                    "column": 100,
                    "message": "Line too long",
                    "severity": "warning",
                    "code": "C0301"
                })
            
            # Check for unused imports (simplified)
            if line.strip().startswith('import ') and 'unused' in line.lower():
                issues.append({
                    "line": i,
                    "column": 1,
                    "message": "Unused import",
                    "severity": "warning", 
                    "code": "W0611"
                })
            
            # Check for missing docstrings (simplified)
            if line.strip().startswith('def ') and i < len(lines) - 1:
                next_line = lines[i].strip() if i < len(lines) else ""
                if not next_line.startswith('"""') and not next_line.startswith("'''"):
                    issues.append({
                        "line": i,
                        "column": 1,
                        "message": "Missing function docstring",
                        "severity": "info",
                        "code": "C0111"
                    })
        
        return issues
    
    def _categorize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize issues by severity."""
        categories = {"error": 0, "warning": 0, "info": 0}
        
        for issue in issues:
            severity = issue.get("severity", "info")
            categories[severity] = categories.get(severity, 0) + 1
        
        return categories


class DocstringGeneratorPlugin(GeneratorPlugin):
    """Plugin that generates Python docstrings."""
    
    def __init__(self):
        super().__init__(
            name="docstring_generator",
            version="1.0.0",
            description="Generates Python docstrings for functions and classes"
        )
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the docstring generator."""
        self.config = config or {}
        self.docstring_style = self.config.get("style", "google")  # google, numpy, sphinx
        return True
    
    async def generate_code(self, prompt: str, language: str, 
                           context: PluginContext) -> str:
        """Generate Python docstrings."""
        if language.lower() != "python":
            return f"# Docstring generation only supports Python"
        
        # Extract function/class signature from prompt
        function_match = re.search(r'def\s+(\w+)\s*\((.*?)\)', prompt)
        class_match = re.search(r'class\s+(\w+)\s*\((.*?)\):', prompt)
        
        if function_match:
            func_name = function_match.group(1)
            params = function_match.group(2)
            return self._generate_function_docstring(func_name, params)
        
        elif class_match:
            class_name = class_match.group(1)
            base_classes = class_match.group(2)
            return self._generate_class_docstring(class_name, base_classes)
        
        else:
            return self._generate_generic_docstring(prompt)
    
    def _generate_function_docstring(self, func_name: str, params: str) -> str:
        """Generate function docstring."""
        if self.docstring_style == "google":
            docstring = f'"""{func_name.replace("_", " ").title()}\n\n'
            
            if params.strip():
                docstring += "    Args:\n"
                for param in params.split(','):
                    param = param.strip()
                    if param and '=' not in param:
                        docstring += f"        {param}: Description of {param}\n"
            
            docstring += "\n    Returns:\n"
            docstring += "        Description of return value\n"
            docstring += '    """'
            
            return docstring
        
        else:
            return f'"""{func_name.replace("_", " ").title()}."""'
    
    def _generate_class_docstring(self, class_name: str, base_classes: str) -> str:
        """Generate class docstring."""
        docstring = f'"""{class_name} class.\n\n'
        docstring += f"    This class implements {class_name.lower()} functionality.\n"
        
        if base_classes.strip():
            docstring += f"\n    Inherits from: {base_classes}\n"
        
        docstring += '    """'
        return docstring
    
    def _generate_generic_docstring(self, prompt: str) -> str:
        """Generate generic docstring."""
        return f'"""{prompt}\n\n    Additional documentation here.\n    """'


class SlackIntegrationPlugin(IntegrationPlugin):
    """Plugin for Slack integration."""
    
    def __init__(self):
        super().__init__(
            name="slack_integration",
            version="1.0.0",
            description="Send notifications and reports to Slack"
        )
        self.dependencies = ["slack_sdk"]
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize Slack integration."""
        self.config = config or {}
        self.webhook_url = self.config.get("webhook_url", "")
        self.default_channel = self.config.get("default_channel", "#general")
        return bool(self.webhook_url)
    
    def get_required_config_keys(self) -> List[str]:
        """Get required configuration keys."""
        return ["webhook_url"]
    
    async def send_data(self, data: Dict[str, Any], 
                       destination: str, context: PluginContext) -> bool:
        """Send data to Slack."""
        try:
            channel = destination or self.default_channel
            
            # Format message based on data type
            if "analysis_results" in data:
                message = self._format_analysis_message(data["analysis_results"])
            elif "report" in data:
                message = self._format_report_message(data["report"])
            else:
                message = f"Update from AI Code Agent:\n```{json.dumps(data, indent=2)}```"
            
            # In a real implementation, this would use Slack SDK
            print(f"Sending to Slack {channel}: {message}")
            
            return True
            
        except Exception as e:
            print(f"Slack integration error: {e}")
            return False
    
    async def receive_data(self, source: str, 
                          context: PluginContext) -> Dict[str, Any]:
        """Receive data from Slack (webhook events)."""
        # In a real implementation, this would parse Slack webhook data
        return {
            "source": "slack",
            "channel": source,
            "timestamp": datetime.now().isoformat(),
            "data": {"message": "Sample Slack webhook data"}
        }
    
    def _format_analysis_message(self, analysis_results: Dict[str, Any]) -> str:
        """Format analysis results for Slack."""
        message = "🔍 **Code Analysis Complete**\n\n"
        
        if "issues" in analysis_results:
            issues = analysis_results["issues"]
            message += f"Found {len(issues)} issues:\n"
            
            for issue in issues[:5]:  # Show first 5 issues
                severity_emoji = {"error": "🚨", "warning": "⚠️", "info": "ℹ️"}
                emoji = severity_emoji.get(issue.get("severity", "info"), "📝")
                message += f"{emoji} Line {issue.get('line', '?')}: {issue.get('message', 'Unknown issue')}\n"
            
            if len(issues) > 5:
                message += f"... and {len(issues) - 5} more issues\n"
        
        return message
    
    def _format_report_message(self, report: Dict[str, Any]) -> str:
        """Format report for Slack."""
        message = "📊 **Analysis Report**\n\n"
        message += f"**Generated**: {report.get('timestamp', datetime.now().isoformat())}\n"
        message += f"**Overall Score**: {report.get('overall_score', 'N/A')}\n"
        
        if "key_insights" in report:
            message += "\n**Key Insights:**\n"
            for insight in report["key_insights"][:3]:
                message += f"• {insight}\n"
        
        return message


class HTMLReportPlugin(ReportingPlugin):
    """Plugin that generates HTML reports."""
    
    def __init__(self):
        super().__init__(
            name="html_report",
            version="1.0.0",
            description="Generate HTML reports with charts and visualizations"
        )
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize HTML report generator."""
        self.config = config or {}
        self.template_dir = self.config.get("template_dir", "templates")
        self.output_dir = self.config.get("output_dir", "reports")
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        return True
    
    async def generate_report(self, data: Dict[str, Any], 
                             report_type: str, context: PluginContext) -> str:
        """Generate HTML report."""
        if report_type == "analysis":
            return self._generate_analysis_report(data)
        elif report_type == "summary":
            return self._generate_summary_report(data)
        else:
            return self._generate_generic_report(data, report_type)
    
    async def create_visualization(self, data: Dict[str, Any], 
                                  viz_type: str, context: PluginContext) -> str:
        """Create HTML visualization."""
        if viz_type == "chart":
            return self._create_chart_visualization(data)
        elif viz_type == "dashboard":
            return self._create_dashboard_visualization(data)
        else:
            return f"<p>Visualization type '{viz_type}' not supported</p>"
    
    def _generate_analysis_report(self, data: Dict[str, Any]) -> str:
        """Generate analysis report HTML."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Code Agent - Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f0f8ff; padding: 20px; border-radius: 8px; }
                .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }
                .issue { margin: 10px 0; padding: 10px; background: #fff5f5; border-radius: 4px; }
                .severity-error { border-left: 4px solid #ff4444; }
                .severity-warning { border-left: 4px solid #ff8800; }
                .severity-info { border-left: 4px solid #0088ff; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 AI Code Agent Analysis Report</h1>
                <p>Generated: {timestamp}</p>
            </div>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Add summary section
        if "summary" in data:
            html += f"""
            <div class="section">
                <h2>📊 Summary</h2>
                <p>{data['summary']}</p>
            </div>
            """
        
        # Add issues section
        if "issues" in data:
            html += '<div class="section"><h2>🔍 Issues Found</h2>'
            
            for issue in data["issues"]:
                severity = issue.get("severity", "info")
                html += f"""
                <div class="issue severity-{severity}">
                    <strong>Line {issue.get('line', '?')}</strong>: {issue.get('message', 'Unknown issue')}
                    <br><small>Severity: {severity.title()}</small>
                </div>
                """
            
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        # Save to file
        report_file = Path(self.output_dir) / f"analysis_report_{int(datetime.now().timestamp())}.html"
        with open(report_file, 'w') as f:
            f.write(html)
        
        return str(report_file)
    
    def _generate_summary_report(self, data: Dict[str, Any]) -> str:
        """Generate summary report HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Code Agent - Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          background: #f9f9f9; border-radius: 8px; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
            </style>
        </head>
        <body>
            <h1>📈 Summary Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="metrics">
        """
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                html += f"""
                <div class="metric">
                    <div class="metric-value">{value}</div>
                    <div>{key.replace('_', ' ').title()}</div>
                </div>
                """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        # Save to file
        report_file = Path(self.output_dir) / f"summary_report_{int(datetime.now().timestamp())}.html"
        with open(report_file, 'w') as f:
            f.write(html)
        
        return str(report_file)
    
    def _generate_generic_report(self, data: Dict[str, Any], report_type: str) -> str:
        """Generate generic HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Code Agent - {report_type.title()} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                pre {{ background: #f5f5f5; padding: 15px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>{report_type.title()} Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <pre>{json.dumps(data, indent=2)}</pre>
        </body>
        </html>
        """
        
        # Save to file
        report_file = Path(self.output_dir) / f"{report_type}_report_{int(datetime.now().timestamp())}.html"
        with open(report_file, 'w') as f:
            f.write(html)
        
        return str(report_file)
    
    def _create_chart_visualization(self, data: Dict[str, Any]) -> str:
        """Create chart visualization."""
        # This would use a charting library like Chart.js in a real implementation
        chart_html = """
        <div id="chart" style="width: 100%; height: 400px; border: 1px solid #ccc;">
            <p>Chart visualization would be rendered here using Chart.js or similar library</p>
            <pre>""" + json.dumps(data, indent=2) + """</pre>
        </div>
        """
        return chart_html
    
    def _create_dashboard_visualization(self, data: Dict[str, Any]) -> str:
        """Create dashboard visualization."""
        dashboard_html = """
        <div class="dashboard" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div class="widget" style="border: 1px solid #ccc; padding: 15px;">
                <h3>📊 Metrics Overview</h3>
                <p>Dashboard widgets would be rendered here</p>
            </div>
            <div class="widget" style="border: 1px solid #ccc; padding: 15px;">
                <h3>🎯 Key Insights</h3>
                <p>Interactive insights would be displayed here</p>
            </div>
        </div>
        """
        return dashboard_html