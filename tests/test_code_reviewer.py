"""
Tests for the code reviewer agent.
"""

import tempfile
import os
from pathlib import Path
import asyncio

import pytest

from src.agents.code_reviewer import CodeReviewerAgent, ReviewResult


class TestCodeReviewerAgent:
    """Test cases for the code reviewer agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = CodeReviewerAgent()
    
    def test_detect_language(self):
        """Test language detection from file extensions."""
        assert self.agent._detect_language("test.py") == "python"
        assert self.agent._detect_language("app.js") == "javascript"
        assert self.agent._detect_language("main.cpp") == "cpp"
        assert self.agent._detect_language("unknown.xyz") == "unknown"
    
    def test_calculate_scores(self):
        """Test score calculation methods."""
        # Create a mock static result
        from src.analyzers.static_analyzer import AnalysisResult, FileMetrics, CodeIssue, IssueType, Severity
        
        metrics = FileMetrics(
            lines_of_code=100,
            blank_lines=10,
            comment_lines=20,
            cyclomatic_complexity=5.0,
            maintainability_index=85.0,
            function_count=5,
            class_count=1
        )
        
        issues = [
            CodeIssue(
                type=IssueType.STYLE,
                severity=Severity.LOW,
                message="Minor style issue",
                line_number=10
            )
        ]
        
        static_result = AnalysisResult(
            file_path="test.py",
            metrics=metrics,
            issues=issues,
            analysis_time_ms=100.0
        )
        
        # Test maintainability score calculation
        maintainability_score = self.agent._calculate_maintainability_score(static_result)
        assert 0.0 <= maintainability_score <= 10.0
        
        # Test overall score calculation with mock feedback
        from src.agents.code_reviewer import ReviewFeedback
        feedback = ReviewFeedback(
            overall_score=0,
            summary="Test",
            strengths=[],
            issues=[],
            suggestions=[],
            security_concerns=[],
            performance_notes=[],
            maintainability_score=0,
            confidence=0.8
        )
        
        overall_score = self.agent._calculate_overall_score(static_result, feedback)
        assert 0.0 <= overall_score <= 10.0
    
    @pytest.mark.asyncio
    async def test_review_simple_file(self):
        """Test reviewing a simple Python file without AI."""
        # Create a temporary Python file
        simple_code = '''
def greet(name):
    """Greet someone by name."""
    if name:
        return f"Hello, {name}!"
    return "Hello, stranger!"

class Person:
    """A simple person class."""
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        """Introduce the person."""
        return f"My name is {self.name} and I'm {self.age} years old."
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(simple_code)
            temp_file = f.name
        
        try:
            # Mock the AI response to avoid API calls in tests
            original_generate = self.agent._generate_ai_review
            
            async def mock_generate_ai_review(file_path, code_content, language, static_result):
                return self.agent._fallback_feedback(static_result)
            
            self.agent._generate_ai_review = mock_generate_ai_review
            
            # Test the review
            result = await self.agent.review_file(temp_file)
            
            # Verify result
            assert result is not None
            assert isinstance(result, ReviewResult)
            assert result.file_path == temp_file
            assert result.language == "python"
            assert result.static_analysis is not None
            assert result.ai_feedback is not None
            assert result.review_time_ms > 0
            
            # Restore original method
            self.agent._generate_ai_review = original_generate
            
        finally:
            os.unlink(temp_file)
    
    def test_review_nonexistent_file(self):
        """Test handling of nonexistent files."""
        async def run_test():
            result = await self.agent.review_file("nonexistent.py")
            assert result is None
        
        asyncio.run(run_test())
    
    def test_generate_review_report(self):
        """Test generation of review reports."""
        from src.agents.code_reviewer import ReviewResult, ReviewFeedback
        from src.analyzers.static_analyzer import AnalysisResult, FileMetrics
        
        # Create mock results
        feedback = ReviewFeedback(
            overall_score=8.5,
            summary="Good code quality with minor issues",
            strengths=["Clear function names", "Good documentation"],
            issues=[
                {"description": "Long function detected", "severity": "medium", "line": 15}
            ],
            suggestions=["Consider breaking down long functions"],
            security_concerns=[],
            performance_notes=["Consider caching results"],
            maintainability_score=8.0,
            confidence=0.9
        )
        
        metrics = FileMetrics(50, 5, 10, 3.0, 80.0, 3, 1)
        static_result = AnalysisResult("test.py", metrics, [], None, 100.0)
        
        result = ReviewResult(
            file_path="test.py",
            language="python",
            static_analysis=static_result,
            ai_feedback=feedback,
            review_time_ms=250.0,
            model_used="test_model"
        )
        
        report = self.agent.generate_review_report([result])
        
        # Verify report content
        assert "Code Review Report" in report
        assert "test.py" in report
        assert "8.5/10" in report
        assert "Good code quality with minor issues" in report
        assert "Long function detected" in report
    
    def test_empty_review_report(self):
        """Test generation of report with no results."""
        report = self.agent.generate_review_report([])
        assert "No files were reviewed" in report


if __name__ == "__main__":
    pytest.main([__file__])