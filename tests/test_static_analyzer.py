"""
Tests for the static analyzer module.
"""

import tempfile
import os
from pathlib import Path

import pytest

from src.analyzers.static_analyzer import (
    StaticAnalyzer, 
    PythonAnalyzer, 
    IssueType, 
    Severity
)


class TestPythonAnalyzer:
    """Test cases for Python code analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PythonAnalyzer()
    
    def test_can_analyze_python_files(self):
        """Test that analyzer can identify Python files."""
        assert self.analyzer.can_analyze("test.py")
        assert self.analyzer.can_analyze("module.py")
        assert not self.analyzer.can_analyze("test.js")
        assert not self.analyzer.can_analyze("README.md")
    
    def test_analyze_simple_file(self):
        """Test analysis of a simple Python file."""
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def hello_world():
    """A simple function."""
    print("Hello, world!")
    return "success"

class SimpleClass:
    """A simple class."""
    def __init__(self):
        self.value = 42
''')
            temp_file = f.name
        
        try:
            result = self.analyzer.analyze_file(temp_file)
            
            # Check basic properties
            assert result is not None
            assert result.file_path == temp_file
            assert result.metrics.function_count == 2  # hello_world + __init__
            assert result.metrics.class_count == 1
            assert result.metrics.lines_of_code > 0
            
            # Should have minimal issues for well-written code
            assert len(result.issues) >= 0
            
        finally:
            os.unlink(temp_file)
    
    def test_detect_complexity_issues(self):
        """Test detection of high complexity code."""
        complex_code = '''
def complex_function(a, b, c, d, e, f):
    """A complex function with many branches."""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            return a + b + c + d + e + f
                        else:
                            return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_code)
            temp_file = f.name
        
        try:
            result = self.analyzer.analyze_file(temp_file)
            
            # Should detect complexity and argument count issues
            complexity_issues = [issue for issue in result.issues 
                               if issue.type == IssueType.COMPLEXITY]
            maintainability_issues = [issue for issue in result.issues 
                                    if issue.type == IssueType.MAINTAINABILITY]
            
            assert len(complexity_issues) > 0 or len(maintainability_issues) > 0
            
        finally:
            os.unlink(temp_file)
    
    def test_detect_missing_docstring(self):
        """Test detection of missing docstrings."""
        undocumented_code = '''
def undocumented_function():
    return "no docstring"

class UndocumentedClass:
    def method_without_docs(self):
        pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(undocumented_code)
            temp_file = f.name
        
        try:
            result = self.analyzer.analyze_file(temp_file)
            
            # Should detect missing docstrings
            doc_issues = [issue for issue in result.issues 
                         if issue.type == IssueType.DOCUMENTATION]
            
            assert len(doc_issues) >= 2  # Function and class missing docstrings
            
        finally:
            os.unlink(temp_file)
    
    def test_syntax_error_handling(self):
        """Test handling of files with syntax errors."""
        invalid_code = '''
def broken_function(
    # Missing closing parenthesis and colon
    return "broken"
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(invalid_code)
            temp_file = f.name
        
        try:
            result = self.analyzer.analyze_file(temp_file)
            
            # Should handle syntax error gracefully
            assert result is not None
            assert len(result.issues) > 0
            assert any(issue.severity == Severity.CRITICAL for issue in result.issues)
            
        finally:
            os.unlink(temp_file)


class TestStaticAnalyzer:
    """Test cases for the main static analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StaticAnalyzer()
    
    def test_analyze_nonexistent_file(self):
        """Test handling of nonexistent files."""
        result = self.analyzer.analyze_file("nonexistent.py")
        assert result is None
    
    def test_analyze_directory(self):
        """Test analysis of a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some Python files
            py_file1 = Path(temp_dir) / "file1.py"
            py_file2 = Path(temp_dir) / "subdir" / "file2.py"
            js_file = Path(temp_dir) / "file.js"
            
            # Create subdirectory
            (Path(temp_dir) / "subdir").mkdir()
            
            # Write Python files
            py_file1.write_text('def test1(): pass')
            py_file2.write_text('def test2(): pass')
            js_file.write_text('function test() {}')
            
            results = self.analyzer.analyze_directory(temp_dir)
            
            # Should analyze Python files but not JS
            assert len(results) == 2
            assert all(result.file_path.endswith('.py') for result in results)
    
    def test_get_summary_statistics(self):
        """Test generation of summary statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a Python file
            py_file = Path(temp_dir) / "test.py"
            py_file.write_text('''
def test_function():
    """A test function."""
    return True

def undocumented():
    return False
''')
            
            results = self.analyzer.analyze_directory(temp_dir)
            summary = self.analyzer.get_summary_statistics(results)
            
            assert 'total_files' in summary
            assert 'total_issues' in summary
            assert 'total_lines_of_code' in summary
            assert 'issue_types' in summary
            assert 'severity_distribution' in summary
            assert summary['total_files'] == 1
    
    def test_empty_results_summary(self):
        """Test summary statistics with empty results."""
        summary = self.analyzer.get_summary_statistics([])
        assert summary == {}


if __name__ == "__main__":
    pytest.main([__file__])