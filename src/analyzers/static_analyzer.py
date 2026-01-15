"""
Static code analysis module for parsing and analyzing code files.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from radon.raw import analyze


class IssueType(Enum):
    """Types of code issues that can be detected."""
    COMPLEXITY = "complexity"
    STYLE = "style"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"


class Severity(Enum):
    """Severity levels for detected issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeIssue:
    """Represents a code issue found during analysis."""
    type: IssueType
    severity: Severity
    message: str
    line_number: int
    column_number: int = 0
    suggestion: Optional[str] = None
    confidence: float = 1.0


@dataclass
class FileMetrics:
    """Metrics for a single file."""
    lines_of_code: int
    blank_lines: int
    comment_lines: int
    cyclomatic_complexity: float
    maintainability_index: float
    function_count: int
    class_count: int


@dataclass
class AnalysisResult:
    """Result of static code analysis."""
    file_path: str
    metrics: FileMetrics
    issues: List[CodeIssue]
    ast_tree: Optional[ast.AST] = None
    analysis_time_ms: float = 0.0


class PythonAnalyzer:
    """Analyzer for Python code files."""
    
    def __init__(self):
        self.supported_extensions = {'.py'}
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can handle the given file."""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyze a Python file and return results."""
        import time
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content, filename=file_path)
            except SyntaxError as e:
                return AnalysisResult(
                    file_path=file_path,
                    metrics=FileMetrics(0, 0, 0, 0, 0, 0, 0),
                    issues=[CodeIssue(
                        type=IssueType.STYLE,
                        severity=Severity.CRITICAL,
                        message=f"Syntax error: {e.msg}",
                        line_number=e.lineno or 1,
                        column_number=e.offset or 0
                    )],
                    analysis_time_ms=(time.time() - start_time) * 1000
                )
            
            # Calculate metrics
            metrics = self._calculate_metrics(content, tree)
            
            # Detect issues
            issues = self._detect_issues(content, tree)
            
            analysis_time = (time.time() - start_time) * 1000
            
            return AnalysisResult(
                file_path=file_path,
                metrics=metrics,
                issues=issues,
                ast_tree=tree,
                analysis_time_ms=analysis_time
            )
            
        except Exception as e:
            return AnalysisResult(
                file_path=file_path,
                metrics=FileMetrics(0, 0, 0, 0, 0, 0, 0),
                issues=[CodeIssue(
                    type=IssueType.STYLE,
                    severity=Severity.CRITICAL,
                    message=f"Analysis error: {str(e)}",
                    line_number=1
                )],
                analysis_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_metrics(self, content: str, tree: ast.AST) -> FileMetrics:
        """Calculate various code metrics."""
        # Use radon for basic metrics
        raw_metrics = analyze(content)
        
        # Calculate complexity
        try:
            complexity_data = radon_cc.cc_visit(content)
            avg_complexity = sum(item.complexity for item in complexity_data) / len(complexity_data) if complexity_data else 0
        except:
            avg_complexity = 0
        
        # Calculate maintainability index
        try:
            mi = radon_metrics.mi_visit(content, multi=True)
            maintainability = mi if isinstance(mi, (int, float)) else 0
        except:
            maintainability = 0
        
        # Count functions and classes from AST
        function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        
        return FileMetrics(
            lines_of_code=raw_metrics.loc,
            blank_lines=raw_metrics.blank,
            comment_lines=raw_metrics.comments,
            cyclomatic_complexity=avg_complexity,
            maintainability_index=maintainability,
            function_count=function_count,
            class_count=class_count
        )
    
    def _detect_issues(self, content: str, tree: ast.AST) -> List[CodeIssue]:
        """Detect various code issues."""
        issues = []
        
        # Check for high complexity functions
        try:
            complexity_data = radon_cc.cc_visit(content)
            for item in complexity_data:
                if item.complexity > 10:  # McCabe complexity threshold
                    issues.append(CodeIssue(
                        type=IssueType.COMPLEXITY,
                        severity=Severity.HIGH if item.complexity > 15 else Severity.MEDIUM,
                        message=f"High cyclomatic complexity ({item.complexity}) in {item.name}",
                        line_number=item.lineno,
                        suggestion="Consider breaking this function into smaller functions"
                    ))
        except:
            pass
        
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    length = node.end_lineno - node.lineno
                    if length > 50:  # Long function threshold
                        issues.append(CodeIssue(
                            type=IssueType.MAINTAINABILITY,
                            severity=Severity.MEDIUM,
                            message=f"Long function '{node.name}' ({length} lines)",
                            line_number=node.lineno,
                            suggestion="Consider breaking this function into smaller functions"
                        ))
        
        # Check for missing docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append(CodeIssue(
                        type=IssueType.DOCUMENTATION,
                        severity=Severity.LOW,
                        message=f"Missing docstring for {type(node).__name__.lower()} '{node.name}'",
                        line_number=node.lineno,
                        suggestion="Add a docstring to describe the purpose and parameters"
                    ))
        
        # Check for too many arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                arg_count = len(node.args.args)
                if hasattr(node.args, 'posonlyargs'):
                    arg_count += len(node.args.posonlyargs)
                if hasattr(node.args, 'kwonlyargs'):
                    arg_count += len(node.args.kwonlyargs)
                
                if arg_count > 5:  # Too many arguments threshold
                    issues.append(CodeIssue(
                        type=IssueType.MAINTAINABILITY,
                        severity=Severity.MEDIUM,
                        message=f"Function '{node.name}' has too many arguments ({arg_count})",
                        line_number=node.lineno,
                        suggestion="Consider using a configuration object or breaking the function down"
                    ))
        
        return issues


class StaticAnalyzer:
    """Main static analyzer that coordinates different language analyzers."""
    
    def __init__(self):
        self.analyzers = {
            'python': PythonAnalyzer()
        }
    
    def analyze_file(self, file_path: str) -> Optional[AnalysisResult]:
        """Analyze a file using the appropriate analyzer."""
        if not os.path.exists(file_path):
            return None
        
        # Find appropriate analyzer
        for analyzer in self.analyzers.values():
            if analyzer.can_analyze(file_path):
                return analyzer.analyze_file(file_path)
        
        return None
    
    def analyze_directory(self, directory_path: str, 
                         extensions: Optional[List[str]] = None) -> List[AnalysisResult]:
        """Analyze all supported files in a directory."""
        results = []
        directory = Path(directory_path)
        
        if not directory.exists():
            return results
        
        # Get all files to analyze
        files_to_analyze = []
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                if extensions:
                    if file_path.suffix.lower() in extensions:
                        files_to_analyze.append(str(file_path))
                else:
                    # Check if any analyzer can handle this file
                    for analyzer in self.analyzers.values():
                        if analyzer.can_analyze(str(file_path)):
                            files_to_analyze.append(str(file_path))
                            break
        
        # Analyze each file
        for file_path in files_to_analyze:
            result = self.analyze_file(file_path)
            if result:
                results.append(result)
        
        return results
    
    def get_summary_statistics(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results."""
        if not results:
            return {}
        
        total_files = len(results)
        total_issues = sum(len(result.issues) for result in results)
        total_loc = sum(result.metrics.lines_of_code for result in results)
        
        # Issue distribution by type
        issue_types = {}
        severity_counts = {}
        
        for result in results:
            for issue in result.issues:
                issue_types[issue.type.value] = issue_types.get(issue.type.value, 0) + 1
                severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
        
        # Average metrics
        avg_complexity = sum(result.metrics.cyclomatic_complexity for result in results) / total_files
        avg_maintainability = sum(result.metrics.maintainability_index for result in results) / total_files
        
        return {
            'total_files': total_files,
            'total_issues': total_issues,
            'total_lines_of_code': total_loc,
            'issues_per_file': total_issues / total_files if total_files > 0 else 0,
            'issue_types': issue_types,
            'severity_distribution': severity_counts,
            'average_complexity': avg_complexity,
            'average_maintainability': avg_maintainability,
            'analysis_time_total_ms': sum(result.analysis_time_ms for result in results)
        }