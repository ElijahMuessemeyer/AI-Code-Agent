"""
Multi-agent coordination system that orchestrates different AI agents to work together.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
from datetime import datetime

from src.agents.code_reviewer import CodeReviewerAgent, ReviewResult
from src.agents.bug_detector import BugDetectorAgent, BugDetectionResult
from src.agents.code_generator import CodeGeneratorAgent, CodeGenerationResult, CodeRequirement
from src.agents.test_generator import TestGeneratorAgent, TestGenerationResult


class WorkflowType(Enum):
    """Types of multi-agent workflows."""
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"
    CODE_IMPROVEMENT = "code_improvement"
    FULL_DEVELOPMENT = "full_development"
    SECURITY_AUDIT = "security_audit"
    QUALITY_ASSURANCE = "quality_assurance"
    REFACTORING = "refactoring"


class TaskPriority(Enum):
    """Priority levels for agent tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentTask:
    """Represents a task for a specific agent."""
    agent_type: str
    task_id: str
    priority: TaskPriority
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 2
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class WorkflowResult:
    """Result of a multi-agent workflow."""
    workflow_type: WorkflowType
    target_files: List[str]
    task_results: Dict[str, Any]
    execution_time_ms: float
    success_rate: float
    summary: str
    recommendations: List[str]
    artifacts_generated: List[str] = field(default_factory=list)


class AgentCoordinator:
    """Coordinates multiple AI agents to work together on complex tasks."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize all agents
        self.code_reviewer = CodeReviewerAgent(model_name)
        self.bug_detector = BugDetectorAgent(model_name)
        self.code_generator = CodeGeneratorAgent(model_name)
        self.test_generator = TestGeneratorAgent(model_name)
        
        # Task management
        self.tasks: Dict[str, AgentTask] = {}
        self.task_counter = 0
        
        # Workflow templates
        self.workflow_templates = self._initialize_workflow_templates()
    
    def _initialize_workflow_templates(self) -> Dict[WorkflowType, List[Dict[str, Any]]]:
        """Initialize predefined workflow templates."""
        return {
            WorkflowType.COMPREHENSIVE_ANALYSIS: [
                {"agent": "code_reviewer", "priority": TaskPriority.HIGH},
                {"agent": "bug_detector", "priority": TaskPriority.HIGH},
                {"agent": "test_generator", "priority": TaskPriority.MEDIUM}
            ],
            
            WorkflowType.CODE_IMPROVEMENT: [
                {"agent": "code_reviewer", "priority": TaskPriority.HIGH},
                {"agent": "bug_detector", "priority": TaskPriority.HIGH},
                {"agent": "code_generator", "priority": TaskPriority.MEDIUM, 
                 "depends_on": ["code_reviewer", "bug_detector"]},
                {"agent": "test_generator", "priority": TaskPriority.MEDIUM,
                 "depends_on": ["code_generator"]}
            ],
            
            WorkflowType.FULL_DEVELOPMENT: [
                {"agent": "code_generator", "priority": TaskPriority.CRITICAL},
                {"agent": "code_reviewer", "priority": TaskPriority.HIGH,
                 "depends_on": ["code_generator"]},
                {"agent": "bug_detector", "priority": TaskPriority.HIGH,
                 "depends_on": ["code_generator"]},
                {"agent": "test_generator", "priority": TaskPriority.HIGH,
                 "depends_on": ["code_generator"]}
            ],
            
            WorkflowType.SECURITY_AUDIT: [
                {"agent": "code_reviewer", "priority": TaskPriority.HIGH},
                {"agent": "bug_detector", "priority": TaskPriority.CRITICAL}
            ],
            
            WorkflowType.QUALITY_ASSURANCE: [
                {"agent": "code_reviewer", "priority": TaskPriority.HIGH},
                {"agent": "test_generator", "priority": TaskPriority.HIGH},
                {"agent": "bug_detector", "priority": TaskPriority.MEDIUM}
            ]
        }
    
    async def execute_workflow(self, 
                             workflow_type: WorkflowType,
                             target_files: List[str],
                             custom_config: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute a multi-agent workflow."""
        import time
        start_time = time.time()
        
        # Create tasks based on workflow template
        tasks = self._create_workflow_tasks(workflow_type, target_files, custom_config)
        
        # Execute tasks with dependency management
        task_results = await self._execute_tasks_with_dependencies(tasks)
        
        # Analyze results and generate summary
        summary, recommendations = self._analyze_workflow_results(workflow_type, task_results)
        
        # Calculate success rate
        successful_tasks = sum(1 for task in tasks.values() if task.status == "completed")
        success_rate = successful_tasks / len(tasks) if tasks else 0.0
        
        execution_time = (time.time() - start_time) * 1000
        
        return WorkflowResult(
            workflow_type=workflow_type,
            target_files=target_files,
            task_results=task_results,
            execution_time_ms=execution_time,
            success_rate=success_rate,
            summary=summary,
            recommendations=recommendations
        )
    
    async def analyze_codebase(self, directory_path: str) -> WorkflowResult:
        """Perform comprehensive analysis of an entire codebase."""
        # Find all supported files
        target_files = self._find_analyzable_files(directory_path)
        
        if not target_files:
            return WorkflowResult(
                workflow_type=WorkflowType.COMPREHENSIVE_ANALYSIS,
                target_files=[],
                task_results={},
                execution_time_ms=0.0,
                success_rate=0.0,
                summary="No analyzable files found in the directory.",
                recommendations=["Add supported code files (.py, .js, .ts, etc.)"]
            )
        
        return await self.execute_workflow(
            WorkflowType.COMPREHENSIVE_ANALYSIS,
            target_files[:10]  # Limit to first 10 files for performance
        )
    
    async def improve_code_quality(self, 
                                 file_paths: List[str],
                                 focus_areas: Optional[List[str]] = None) -> WorkflowResult:
        """Improve code quality using multiple agents."""
        custom_config = {
            "focus_areas": focus_areas or ["bugs", "performance", "maintainability"],
            "generate_improvements": True
        }
        
        return await self.execute_workflow(
            WorkflowType.CODE_IMPROVEMENT,
            file_paths,
            custom_config
        )
    
    async def develop_from_requirements(self, 
                                      requirements: List[CodeRequirement]) -> WorkflowResult:
        """Develop code from requirements using all agents."""
        custom_config = {
            "requirements": requirements,
            "quality_level": "production"
        }
        
        # Create a temporary file list for the workflow
        target_files = [f"generated_{i}.py" for i in range(len(requirements))]
        
        return await self.execute_workflow(
            WorkflowType.FULL_DEVELOPMENT,
            target_files,
            custom_config
        )
    
    def _create_workflow_tasks(self, 
                             workflow_type: WorkflowType,
                             target_files: List[str],
                             custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, AgentTask]:
        """Create tasks based on workflow template."""
        template = self.workflow_templates.get(workflow_type, [])
        tasks = {}
        
        for step in template:
            agent_type = step["agent"]
            priority = step["priority"]
            dependencies = step.get("depends_on", [])
            
            # Create task for each target file or as single task
            if agent_type == "code_generator" and custom_config and "requirements" in custom_config:
                # Special case: code generation from requirements
                for i, requirement in enumerate(custom_config["requirements"]):
                    task_id = f"{agent_type}_{i}"
                    tasks[task_id] = AgentTask(
                        agent_type=agent_type,
                        task_id=task_id,
                        priority=priority,
                        input_data={"requirement": requirement},
                        dependencies=[f"{dep}_{i}" for dep in dependencies]
                    )
            else:
                # Standard case: one task per file
                for i, file_path in enumerate(target_files):
                    task_id = f"{agent_type}_{i}"
                    
                    input_data = {"file_path": file_path}
                    if custom_config:
                        input_data.update(custom_config)
                    
                    tasks[task_id] = AgentTask(
                        agent_type=agent_type,
                        task_id=task_id,
                        priority=priority,
                        input_data=input_data,
                        dependencies=[f"{dep}_{i}" for dep in dependencies]
                    )
        
        return tasks
    
    async def _execute_tasks_with_dependencies(self, tasks: Dict[str, AgentTask]) -> Dict[str, Any]:
        """Execute tasks respecting dependencies."""
        completed_tasks = set()
        results = {}
        
        # Create a task execution queue
        while len(completed_tasks) < len(tasks):
            # Find tasks that can be executed (dependencies met)
            ready_tasks = []
            
            for task_id, task in tasks.items():
                if (task_id not in completed_tasks and 
                    task.status == "pending" and
                    all(dep in completed_tasks for dep in task.dependencies)):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Check for circular dependencies or failed dependencies
                remaining_tasks = [t for t in tasks.values() if t.task_id not in completed_tasks]
                for task in remaining_tasks:
                    if task.status == "pending":
                        task.status = "failed"
                        task.error = "Dependency resolution failed"
                        completed_tasks.add(task.task_id)
                break
            
            # Execute ready tasks concurrently
            task_coroutines = []
            for task in ready_tasks:
                task.status = "running"
                task.start_time = datetime.now()
                coroutine = self._execute_single_task(task)
                task_coroutines.append(coroutine)
            
            # Wait for all ready tasks to complete
            task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Process results
            for task, result in zip(ready_tasks, task_results):
                task.end_time = datetime.now()
                
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.error = str(result)
                else:
                    task.status = "completed"
                    task.result = result
                    results[task.task_id] = result
                
                completed_tasks.add(task.task_id)
        
        return results
    
    async def _execute_single_task(self, task: AgentTask) -> Any:
        """Execute a single agent task."""
        try:
            if task.agent_type == "code_reviewer":
                return await self._execute_code_review_task(task)
            elif task.agent_type == "bug_detector":
                return await self._execute_bug_detection_task(task)
            elif task.agent_type == "code_generator":
                return await self._execute_code_generation_task(task)
            elif task.agent_type == "test_generator":
                return await self._execute_test_generation_task(task)
            else:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
                
        except Exception as e:
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = "pending"
                return await self._execute_single_task(task)
            else:
                raise e
    
    async def _execute_code_review_task(self, task: AgentTask) -> ReviewResult:
        """Execute code review task."""
        file_path = task.input_data["file_path"]
        return await self.code_reviewer.review_file(file_path)
    
    async def _execute_bug_detection_task(self, task: AgentTask) -> BugDetectionResult:
        """Execute bug detection task."""
        file_path = task.input_data["file_path"]
        context = task.input_data.get("context", "")
        return await self.bug_detector.detect_bugs(file_path, context)
    
    async def _execute_code_generation_task(self, task: AgentTask) -> CodeGenerationResult:
        """Execute code generation task."""
        requirement = task.input_data["requirement"]
        return await self.code_generator.generate_code(requirement)
    
    async def _execute_test_generation_task(self, task: AgentTask) -> TestGenerationResult:
        """Execute test generation task."""
        file_path = task.input_data["file_path"]
        return await self.test_generator.generate_tests(file_path)
    
    def _analyze_workflow_results(self, 
                                workflow_type: WorkflowType,
                                task_results: Dict[str, Any]) -> tuple[str, List[str]]:
        """Analyze workflow results and generate summary."""
        summary_parts = []
        recommendations = []
        
        # Count results by agent type
        agent_counts = {}
        total_issues = 0
        total_tests = 0
        avg_quality = 0.0
        
        for task_id, result in task_results.items():
            agent_type = task_id.split('_')[0]
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            
            if isinstance(result, ReviewResult):
                total_issues += len(result.ai_feedback.issues)
                avg_quality += result.ai_feedback.overall_score
                
                if result.ai_feedback.security_concerns:
                    recommendations.append("Address security concerns identified in code review")
                
                if result.ai_feedback.overall_score < 7.0:
                    recommendations.append("Improve code quality based on review feedback")
            
            elif isinstance(result, BugDetectionResult):
                total_issues += len(result.bugs_detected)
                
                critical_bugs = [bug for bug in result.bugs_detected 
                               if bug.severity.value == "critical"]
                if critical_bugs:
                    recommendations.append("Fix critical bugs immediately")
            
            elif isinstance(result, TestGenerationResult):
                total_tests += len(result.test_suite.test_cases)
                
                if result.test_suite.coverage_estimate < 0.8:
                    recommendations.append("Increase test coverage to at least 80%")
            
            elif isinstance(result, CodeGenerationResult):
                if result.confidence < 0.8:
                    recommendations.append("Review generated code carefully due to lower confidence")
        
        # Generate summary based on workflow type
        if workflow_type == WorkflowType.COMPREHENSIVE_ANALYSIS:
            summary_parts.append(f"Analyzed {sum(agent_counts.values())} files")
            summary_parts.append(f"Found {total_issues} potential issues")
            summary_parts.append(f"Generated {total_tests} test cases")
            
        elif workflow_type == WorkflowType.CODE_IMPROVEMENT:
            summary_parts.append(f"Improved {agent_counts.get('code_reviewer', 0)} files")
            summary_parts.append(f"Fixed {total_issues} issues")
            summary_parts.append(f"Enhanced test coverage with {total_tests} new tests")
            
        elif workflow_type == WorkflowType.FULL_DEVELOPMENT:
            summary_parts.append(f"Generated {agent_counts.get('code_generator', 0)} code modules")
            summary_parts.append(f"Created {total_tests} comprehensive tests")
            summary_parts.append(f"Quality assurance completed")
        
        # Add general recommendations
        if total_issues > 10:
            recommendations.append("Consider refactoring to reduce complexity")
        
        if not recommendations:
            recommendations.append("Code quality appears good - continue current practices")
        
        summary = ". ".join(summary_parts) if summary_parts else "Workflow completed"
        
        return summary, recommendations
    
    def _find_analyzable_files(self, directory_path: str) -> List[str]:
        """Find all files that can be analyzed by the agents."""
        analyzable_files = []
        directory = Path(directory_path)
        
        if not directory.exists():
            return analyzable_files
        
        # Supported extensions
        supported_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
        
        for file_path in directory.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions):
                analyzable_files.append(str(file_path))
        
        return sorted(analyzable_files)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current status of all tasks."""
        status = {
            "total_tasks": len(self.tasks),
            "by_status": {},
            "by_agent": {},
            "active_tasks": []
        }
        
        for task in self.tasks.values():
            # Count by status
            status["by_status"][task.status] = status["by_status"].get(task.status, 0) + 1
            
            # Count by agent type
            status["by_agent"][task.agent_type] = status["by_agent"].get(task.agent_type, 0) + 1
            
            # Add active tasks
            if task.status in ["running", "pending"]:
                status["active_tasks"].append({
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "status": task.status,
                    "priority": task.priority.value
                })
        
        return status
    
    def generate_workflow_report(self, result: WorkflowResult) -> str:
        """Generate a comprehensive workflow report."""
        report = f"""# Multi-Agent Workflow Report

## Workflow: {result.workflow_type.value.replace('_', ' ').title()}

### Summary
{result.summary}

**Execution Details:**
- **Files Processed:** {len(result.target_files)}
- **Success Rate:** {result.success_rate:.1%}
- **Total Execution Time:** {result.execution_time_ms:.0f}ms

### Recommendations
"""
        
        for i, recommendation in enumerate(result.recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        report += "\n### Detailed Results\n\n"
        
        # Group results by agent type
        agent_results = {}
        for task_id, task_result in result.task_results.items():
            agent_type = task_id.split('_')[0]
            if agent_type not in agent_results:
                agent_results[agent_type] = []
            agent_results[agent_type].append(task_result)
        
        # Generate sections for each agent type
        for agent_type, results in agent_results.items():
            report += f"#### {agent_type.replace('_', ' ').title()} Results\n\n"
            
            if agent_type == "code_reviewer":
                for result_item in results:
                    if isinstance(result_item, ReviewResult):
                        report += f"**File:** {result_item.file_path}\n"
                        report += f"**Quality Score:** {result_item.ai_feedback.overall_score:.1f}/10\n"
                        report += f"**Issues Found:** {len(result_item.ai_feedback.issues)}\n\n"
            
            elif agent_type == "bug_detector":
                total_bugs = sum(len(r.bugs_detected) for r in results if isinstance(r, BugDetectionResult))
                report += f"**Total Bugs Detected:** {total_bugs}\n"
                
                for result_item in results:
                    if isinstance(result_item, BugDetectionResult):
                        report += f"**File:** {result_item.file_path}\n"
                        report += f"**Bugs:** {len(result_item.bugs_detected)}\n\n"
            
            elif agent_type == "test_generator":
                total_tests = sum(len(r.test_suite.test_cases) for r in results if isinstance(r, TestGenerationResult))
                report += f"**Total Tests Generated:** {total_tests}\n"
                
                for result_item in results:
                    if isinstance(result_item, TestGenerationResult):
                        report += f"**File:** {result_item.source_file}\n"
                        report += f"**Tests:** {len(result_item.test_suite.test_cases)}\n"
                        report += f"**Coverage:** {result_item.test_suite.coverage_estimate:.1%}\n\n"
            
            elif agent_type == "code_generator":
                for result_item in results:
                    if isinstance(result_item, CodeGenerationResult):
                        report += f"**Generated:** {result_item.requirement.code_type.value}\n"
                        report += f"**Quality Score:** {result_item.generated_code.quality_score:.1f}/10\n"
                        report += f"**Confidence:** {result_item.confidence:.1%}\n\n"
        
        return report