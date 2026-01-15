"""
Code reviewer agent that analyzes code and provides intelligent feedback.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

from src.analyzers.static_analyzer import StaticAnalyzer, AnalysisResult
from src.llm.model_interface import get_model_manager, ModelResponse
from src.llm.prompt_manager import get_prompt_manager, PromptType


@dataclass
class ReviewFeedback:
    """Structured feedback from code review."""
    overall_score: float  # 0-10 scale
    summary: str
    strengths: List[str]
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    security_concerns: List[str]
    performance_notes: List[str]
    maintainability_score: float  # 0-10 scale
    confidence: float  # 0-1 scale


@dataclass
class ReviewResult:
    """Complete result of a code review."""
    file_path: str
    language: str
    static_analysis: AnalysisResult
    ai_feedback: ReviewFeedback
    review_time_ms: float
    model_used: str


class CodeReviewerAgent:
    """AI agent that performs comprehensive code reviews."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.static_analyzer = StaticAnalyzer()
        self.prompt_manager = get_prompt_manager()
        self.model_manager = get_model_manager()
        self.model_name = model_name
    
    async def review_file(self, file_path: str) -> Optional[ReviewResult]:
        """Review a single code file."""
        import time
        start_time = time.time()
        
        # Check if file exists and is supported
        if not Path(file_path).exists():
            return None
        
        # Perform static analysis first
        static_result = self.static_analyzer.analyze_file(file_path)
        if not static_result:
            return None
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        
        # Determine language
        language = self._detect_language(file_path)
        
        # Generate AI review
        ai_feedback = await self._generate_ai_review(
            file_path, code_content, language, static_result
        )
        
        review_time = (time.time() - start_time) * 1000
        
        return ReviewResult(
            file_path=file_path,
            language=language,
            static_analysis=static_result,
            ai_feedback=ai_feedback,
            review_time_ms=review_time,
            model_used=self.model_name or "default"
        )
    
    async def review_directory(self, directory_path: str) -> List[ReviewResult]:
        """Review all supported files in a directory."""
        static_results = self.static_analyzer.analyze_directory(directory_path)
        review_results = []
        
        # Process files concurrently (but limit concurrency)
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent reviews
        
        async def review_with_semaphore(file_path: str) -> Optional[ReviewResult]:
            async with semaphore:
                return await self.review_file(file_path)
        
        # Create tasks for all files
        tasks = [
            review_with_semaphore(result.file_path) 
            for result in static_results
        ]
        
        # Execute reviews
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        for result in results:
            if isinstance(result, ReviewResult):
                review_results.append(result)
            elif isinstance(result, Exception):
                print(f"Review error: {result}")
        
        return review_results
    
    async def _generate_ai_review(self, 
                                file_path: str,
                                code_content: str,
                                language: str,
                                static_result: AnalysisResult) -> ReviewFeedback:
        """Generate AI-powered review feedback."""
        try:
            # Generate prompt
            system_prompt, user_prompt = self.prompt_manager.generate_code_review_prompt(
                file_path, code_content, language, static_result
            )
            
            # Get AI response
            response = await self.model_manager.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_name=self.model_name
            )
            
            # Parse response into structured feedback
            feedback = self._parse_review_response(response.content, static_result)
            return feedback
            
        except Exception as e:
            print(f"Error generating AI review: {e}")
            # Return basic feedback based on static analysis
            return self._fallback_feedback(static_result)
    
    def _parse_review_response(self, 
                             response_content: str, 
                             static_result: AnalysisResult) -> ReviewFeedback:
        """Parse AI response into structured feedback."""
        # This is a simplified parser - in production, you'd want more robust parsing
        lines = response_content.split('\n')
        
        feedback = ReviewFeedback(
            overall_score=7.0,  # Default score
            summary="",
            strengths=[],
            issues=[],
            suggestions=[],
            security_concerns=[],
            performance_notes=[],
            maintainability_score=7.0,
            confidence=0.8
        )
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if "overall assessment" in line.lower():
                current_section = "summary"
            elif "strengths" in line.lower() or "positive" in line.lower():
                current_section = "strengths"
            elif "issues" in line.lower() or "problems" in line.lower():
                current_section = "issues"
            elif "suggestions" in line.lower() or "recommendations" in line.lower():
                current_section = "suggestions"
            elif "security" in line.lower():
                current_section = "security"
            elif "performance" in line.lower():
                current_section = "performance"
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                # List item
                content = line.lstrip('12345.-* ')
                if current_section == "strengths":
                    feedback.strengths.append(content)
                elif current_section == "issues":
                    feedback.issues.append({
                        "description": content,
                        "severity": "medium",
                        "line": None
                    })
                elif current_section == "suggestions":
                    feedback.suggestions.append(content)
                elif current_section == "security":
                    feedback.security_concerns.append(content)
                elif current_section == "performance":
                    feedback.performance_notes.append(content)
            elif current_section == "summary" and len(line) > 20:
                feedback.summary = line
        
        # Calculate scores based on static analysis and content
        feedback.overall_score = self._calculate_overall_score(static_result, feedback)
        feedback.maintainability_score = self._calculate_maintainability_score(static_result)
        
        return feedback
    
    def _calculate_overall_score(self, 
                               static_result: AnalysisResult, 
                               feedback: ReviewFeedback) -> float:
        """Calculate overall code quality score."""
        base_score = 8.0
        
        # Deduct points for issues
        for issue in static_result.issues:
            if issue.severity.value == "critical":
                base_score -= 2.0
            elif issue.severity.value == "high":
                base_score -= 1.0
            elif issue.severity.value == "medium":
                base_score -= 0.5
            elif issue.severity.value == "low":
                base_score -= 0.2
        
        # Deduct for complexity
        if static_result.metrics.cyclomatic_complexity > 15:
            base_score -= 1.0
        elif static_result.metrics.cyclomatic_complexity > 10:
            base_score -= 0.5
        
        # Deduct for AI-identified issues
        base_score -= len(feedback.issues) * 0.3
        base_score -= len(feedback.security_concerns) * 0.5
        
        return max(0.0, min(10.0, base_score))
    
    def _calculate_maintainability_score(self, static_result: AnalysisResult) -> float:
        """Calculate maintainability score."""
        score = 8.0
        
        # Factor in complexity
        if static_result.metrics.cyclomatic_complexity > 10:
            score -= 2.0
        elif static_result.metrics.cyclomatic_complexity > 5:
            score -= 1.0
        
        # Factor in maintainability index if available
        if static_result.metrics.maintainability_index > 0:
            # Maintainability index is typically 0-100
            mi_score = static_result.metrics.maintainability_index / 10
            score = (score + mi_score) / 2
        
        return max(0.0, min(10.0, score))
    
    def _fallback_feedback(self, static_result: AnalysisResult) -> ReviewFeedback:
        """Create fallback feedback when AI review fails."""
        issues = []
        for issue in static_result.issues:
            issues.append({
                "description": issue.message,
                "severity": issue.severity.value,
                "line": issue.line_number,
                "suggestion": issue.suggestion
            })
        
        return ReviewFeedback(
            overall_score=self._calculate_overall_score(static_result, ReviewFeedback(
                overall_score=0, summary="", strengths=[], issues=issues,
                suggestions=[], security_concerns=[], performance_notes=[],
                maintainability_score=0, confidence=0
            )),
            summary="Automated analysis completed. AI review unavailable.",
            strengths=["Code compiles successfully"] if not any(
                issue.severity.value == "critical" for issue in static_result.issues
            ) else [],
            issues=issues,
            suggestions=[issue.suggestion for issue in static_result.issues if issue.suggestion],
            security_concerns=[],
            performance_notes=[],
            maintainability_score=self._calculate_maintainability_score(static_result),
            confidence=0.6
        )
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        suffix = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        return language_map.get(suffix, 'unknown')
    
    def generate_review_report(self, results: List[ReviewResult]) -> str:
        """Generate a comprehensive review report."""
        if not results:
            return "No files were reviewed."
        
        total_files = len(results)
        total_issues = sum(len(result.ai_feedback.issues) for result in results)
        avg_score = sum(result.ai_feedback.overall_score for result in results) / total_files
        
        report = f"""# Code Review Report
        
## Summary
- **Files Reviewed:** {total_files}
- **Total Issues Found:** {total_issues}
- **Average Quality Score:** {avg_score:.1f}/10
- **Total Review Time:** {sum(r.review_time_ms for r in results):.0f}ms

## File Reviews

"""
        
        for result in results:
            report += f"""### {result.file_path}
**Language:** {result.language}
**Overall Score:** {result.ai_feedback.overall_score:.1f}/10
**Maintainability:** {result.ai_feedback.maintainability_score:.1f}/10

**Summary:** {result.ai_feedback.summary}

**Issues ({len(result.ai_feedback.issues)}):**
"""
            for i, issue in enumerate(result.ai_feedback.issues, 1):
                report += f"{i}. {issue['description']}\n"
            
            if result.ai_feedback.suggestions:
                report += "\n**Suggestions:**\n"
                for i, suggestion in enumerate(result.ai_feedback.suggestions, 1):
                    report += f"{i}. {suggestion}\n"
            
            if result.ai_feedback.security_concerns:
                report += "\n**Security Concerns:**\n"
                for i, concern in enumerate(result.ai_feedback.security_concerns, 1):
                    report += f"{i}. {concern}\n"
            
            report += "\n---\n\n"
        
        return report