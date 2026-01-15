"""
Command-line interface for the AI Code Agent system.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import json

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
except ImportError:
    print("Please install required dependencies: pip install typer rich")
    sys.exit(1)

from src.agents.agent_coordinator import AgentCoordinator, WorkflowType
from src.agents.code_generator import CodeRequirement, CodeType, CodeQuality


app = typer.Typer(help="AI Code Agent - Autonomous code analysis and development")
console = Console()


@app.command()
def analyze(
    paths: List[str] = typer.Argument(..., help="File or directory paths to analyze"),
    workflow: str = typer.Option("comprehensive_analysis", help="Workflow type"),
    output: Optional[str] = typer.Option(None, help="Output file for report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Analyze code files or directories."""
    asyncio.run(_analyze_async(paths, workflow, output, verbose))


@app.command()
def generate(
    description: str = typer.Argument(..., help="Code description/requirements"),
    language: str = typer.Option("python", help="Programming language"),
    code_type: str = typer.Option("function", help="Type of code to generate"),
    quality: str = typer.Option("production", help="Quality level"),
    output: Optional[str] = typer.Option(None, help="Output file for generated code"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Generate code from natural language description."""
    asyncio.run(_generate_async(description, language, code_type, quality, output, verbose))


@app.command()
def improve(
    paths: List[str] = typer.Argument(..., help="File paths to improve"),
    focus: Optional[List[str]] = typer.Option(None, help="Focus areas (bugs, performance, etc.)"),
    output: Optional[str] = typer.Option(None, help="Output directory for improved code"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Improve code quality using multiple agents."""
    asyncio.run(_improve_async(paths, focus, output, verbose))


@app.command()
def test(
    paths: List[str] = typer.Argument(..., help="File paths to generate tests for"),
    framework: str = typer.Option("pytest", help="Test framework to use"),
    output: Optional[str] = typer.Option(None, help="Output directory for test files"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Generate comprehensive tests for code files."""
    asyncio.run(_test_async(paths, framework, output, verbose))


@app.command()
def review(
    paths: List[str] = typer.Argument(..., help="File paths to review"),
    output: Optional[str] = typer.Option(None, help="Output file for review report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Perform code review on files."""
    asyncio.run(_review_async(paths, output, verbose))


@app.command()
def bugs(
    paths: List[str] = typer.Argument(..., help="File paths to scan for bugs"),
    output: Optional[str] = typer.Option(None, help="Output file for bug report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Detect bugs and security issues in code."""
    asyncio.run(_bugs_async(paths, output, verbose))


@app.command()
def status():
    """Show system status and capabilities."""
    _show_status()


async def _analyze_async(paths: List[str], workflow: str, output: Optional[str], verbose: bool):
    """Async implementation of analyze command."""
    try:
        coordinator = AgentCoordinator()
        
        # Validate workflow type
        try:
            workflow_type = WorkflowType(workflow)
        except ValueError:
            console.print(f"[red]Invalid workflow type: {workflow}[/red]")
            console.print(f"Valid types: {[wf.value for wf in WorkflowType]}")
            return
        
        # Determine if analyzing files or directories
        target_files = []
        for path_str in paths:
            path = Path(path_str)
            if path.is_file():
                target_files.append(str(path))
            elif path.is_dir():
                console.print(f"[blue]Analyzing directory: {path}[/blue]")
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                    task = progress.add_task("Finding files...", total=None)
                    dir_files = coordinator._find_analyzable_files(str(path))
                    target_files.extend(dir_files[:10])  # Limit for performance
                    progress.update(task, completed=True)
                    
                console.print(f"Found {len(dir_files)} analyzable files (processing first 10)")
            else:
                console.print(f"[red]Path not found: {path}[/red]")
                return
        
        if not target_files:
            console.print("[red]No analyzable files found[/red]")
            return
        
        # Execute workflow
        console.print(f"[green]Starting {workflow_type.value} workflow...[/green]")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Analyzing files...", total=None)
            result = await coordinator.execute_workflow(workflow_type, target_files)
            progress.update(task, completed=True)
        
        # Display results
        _display_workflow_result(result, verbose)
        
        # Save report if requested
        if output:
            report = coordinator.generate_workflow_report(result)
            Path(output).write_text(report)
            console.print(f"[green]Report saved to: {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        if verbose:
            console.print_exception()


async def _generate_async(description: str, language: str, code_type: str, quality: str, output: Optional[str], verbose: bool):
    """Async implementation of generate command."""
    try:
        coordinator = AgentCoordinator()
        
        # Create requirement
        try:
            req = CodeRequirement(
                description=description,
                language=language,
                code_type=CodeType(code_type),
                quality_level=CodeQuality(quality)
            )
        except ValueError as e:
            console.print(f"[red]Invalid parameter: {e}[/red]")
            return
        
        console.print(f"[green]Generating {code_type} in {language}...[/green]")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Generating code...", total=None)
            result = await coordinator.code_generator.generate_code(req)
            progress.update(task, completed=True)
        
        # Display generated code
        console.print(Panel(f"[bold]Generated Code[/bold] (Confidence: {result.confidence:.1%})"))
        
        if result.generated_code.explanation:
            console.print(f"[blue]{result.generated_code.explanation}[/blue]\n")
        
        syntax = Syntax(result.generated_code.code, language, theme="monokai", line_numbers=True)
        console.print(syntax)
        
        if result.generated_code.documentation:
            console.print(Panel(result.generated_code.documentation, title="Documentation"))
        
        if verbose and result.generated_code.examples:
            console.print(Panel("\n".join(result.generated_code.examples), title="Examples"))
        
        # Save code if requested
        if output:
            saved_path = coordinator.code_generator.save_generated_code(result, str(Path(output).parent))
            console.print(f"[green]Code saved to: {saved_path}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during code generation: {e}[/red]")
        if verbose:
            console.print_exception()


async def _improve_async(paths: List[str], focus: Optional[List[str]], output: Optional[str], verbose: bool):
    """Async implementation of improve command."""
    try:
        coordinator = AgentCoordinator()
        
        # Validate files
        target_files = []
        for path_str in paths:
            path = Path(path_str)
            if path.exists():
                target_files.append(str(path))
            else:
                console.print(f"[red]File not found: {path}[/red]")
        
        if not target_files:
            return
        
        console.print(f"[green]Improving {len(target_files)} files...[/green]")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Improving code quality...", total=None)
            result = await coordinator.improve_code_quality(target_files, focus)
            progress.update(task, completed=True)
        
        _display_workflow_result(result, verbose)
        
        if output:
            report = coordinator.generate_workflow_report(result)
            Path(output).write_text(report)
            console.print(f"[green]Report saved to: {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during improvement: {e}[/red]")
        if verbose:
            console.print_exception()


async def _test_async(paths: List[str], framework: str, output: Optional[str], verbose: bool):
    """Async implementation of test command."""
    try:
        coordinator = AgentCoordinator()
        
        target_files = [str(Path(p)) for p in paths if Path(p).exists()]
        if not target_files:
            console.print("[red]No valid files provided[/red]")
            return
        
        console.print(f"[green]Generating tests for {len(target_files)} files...[/green]")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Generating tests...", total=None)
            results = await coordinator.test_generator.generate_tests_for_directory(".")
            progress.update(task, completed=True)
        
        # Display results
        for result in results:
            if result.source_file in target_files:
                console.print(f"\n[blue]File: {result.source_file}[/blue]")
                console.print(f"Tests generated: {len(result.test_suite.test_cases)}")
                console.print(f"Coverage estimate: {result.test_suite.coverage_estimate:.1%}")
                
                if verbose:
                    for test_case in result.test_suite.test_cases[:3]:  # Show first 3 tests
                        console.print(f"  - {test_case.name}: {test_case.description}")
        
        if output and results:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
            for result in results:
                if result.source_file in target_files:
                    test_file = coordinator.test_generator.generate_test_file(result, str(output_dir))
                    console.print(f"[green]Test file saved: {test_file}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during test generation: {e}[/red]")
        if verbose:
            console.print_exception()


async def _review_async(paths: List[str], output: Optional[str], verbose: bool):
    """Async implementation of review command."""
    try:
        coordinator = AgentCoordinator()
        
        target_files = [str(Path(p)) for p in paths if Path(p).exists()]
        if not target_files:
            console.print("[red]No valid files provided[/red]")
            return
        
        console.print(f"[green]Reviewing {len(target_files)} files...[/green]")
        
        results = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Performing code review...", total=len(target_files))
            
            for file_path in target_files:
                result = await coordinator.code_reviewer.review_file(file_path)
                if result:
                    results.append(result)
                progress.advance(task)
        
        # Display results
        for result in results:
            console.print(f"\n[blue]File: {result.file_path}[/blue]")
            console.print(f"Overall Score: {result.ai_feedback.overall_score:.1f}/10")
            console.print(f"Issues: {len(result.ai_feedback.issues)}")
            
            if verbose and result.ai_feedback.issues:
                for issue in result.ai_feedback.issues[:3]:  # Show first 3 issues
                    console.print(f"  - {issue.get('description', 'Unknown issue')}")
        
        if output:
            report = coordinator.code_reviewer.generate_review_report(results)
            Path(output).write_text(report)
            console.print(f"[green]Review report saved to: {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during review: {e}[/red]")
        if verbose:
            console.print_exception()


async def _bugs_async(paths: List[str], output: Optional[str], verbose: bool):
    """Async implementation of bugs command."""
    try:
        coordinator = AgentCoordinator()
        
        target_files = [str(Path(p)) for p in paths if Path(p).exists()]
        if not target_files:
            console.print("[red]No valid files provided[/red]")
            return
        
        console.print(f"[green]Scanning {len(target_files)} files for bugs...[/green]")
        
        results = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Detecting bugs...", total=len(target_files))
            
            for file_path in target_files:
                result = await coordinator.bug_detector.detect_bugs(file_path)
                if result:
                    results.append(result)
                progress.advance(task)
        
        # Display results
        total_bugs = sum(len(r.bugs_detected) for r in results)
        console.print(f"\n[yellow]Total bugs detected: {total_bugs}[/yellow]")
        
        for result in results:
            if result.bugs_detected:
                console.print(f"\n[blue]File: {result.file_path}[/blue]")
                console.print(f"Bugs found: {len(result.bugs_detected)}")
                
                if verbose:
                    for bug in result.bugs_detected[:3]:  # Show first 3 bugs
                        console.print(f"  - Line {bug.line_number}: {bug.description}")
        
        if output:
            report = coordinator.bug_detector.generate_bug_report(results)
            Path(output).write_text(report)
            console.print(f"[green]Bug report saved to: {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during bug detection: {e}[/red]")
        if verbose:
            console.print_exception()


def _display_workflow_result(result, verbose: bool):
    """Display workflow result in a nice format."""
    # Summary panel
    summary_text = f"""
Success Rate: {result.success_rate:.1%}
Execution Time: {result.execution_time_ms:.0f}ms
Files Processed: {len(result.target_files)}

{result.summary}
"""
    console.print(Panel(summary_text.strip(), title=f"Workflow: {result.workflow_type.value}"))
    
    # Recommendations
    if result.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(result.recommendations, 1):
            console.print(f"{i}. {rec}")
    
    # Detailed results if verbose
    if verbose and result.task_results:
        console.print(f"\n[bold]Detailed Results ({len(result.task_results)} tasks):[/bold]")
        
        table = Table()
        table.add_column("Task")
        table.add_column("Status")
        table.add_column("Details")
        
        for task_id, task_result in result.task_results.items():
            status = "✅ Completed" if task_result else "❌ Failed"
            details = type(task_result).__name__ if task_result else "No result"
            table.add_row(task_id, status, details)
        
        console.print(table)


def _show_status():
    """Show system status and capabilities."""
    console.print(Panel("[bold]AI Code Agent System Status[/bold]", style="green"))
    
    # Agent capabilities table
    table = Table(title="Available Agents")
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Capabilities")
    
    agents = [
        ("Code Reviewer", "Active", "Quality analysis, security checks, maintainability"),
        ("Bug Detector", "Active", "Runtime errors, logic bugs, edge cases"),
        ("Code Generator", "Active", "Code from requirements, multiple languages"),
        ("Test Generator", "Active", "Unit tests, edge cases, coverage analysis")
    ]
    
    for name, status, capabilities in agents:
        table.add_row(name, status, capabilities)
    
    console.print(table)
    
    # Workflow types
    console.print("\n[bold]Available Workflows:[/bold]")
    for workflow in WorkflowType:
        console.print(f"  - {workflow.value}")
    
    # Supported languages
    console.print("\n[bold]Supported Languages:[/bold]")
    languages = ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust"]
    console.print(f"  {', '.join(languages)}")


if __name__ == "__main__":
    app()