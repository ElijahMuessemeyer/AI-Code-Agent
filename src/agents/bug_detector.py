"""
Bug detector agent that specializes in finding runtime errors, logic bugs, and edge cases.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from src.analyzers.static_analyzer import StaticAnalyzer, AnalysisResult
from src.llm.model_interface import get_model_manager, ModelResponse
from src.llm.prompt_manager import get_prompt_manager, PromptType


class BugSeverity(Enum):
    """Severity levels for detected bugs."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BugCategory(Enum):
    """Categories of bugs that can be detected."""
    LOGIC_ERROR = "logic_error"
    NULL_POINTER = "null_pointer"
    RACE_CONDITION = "race_condition"
    MEMORY_LEAK = "memory_leak"
    BOUNDARY_ERROR = "boundary_error"
    ERROR_HANDLING = "error_handling"
    RESOURCE_LEAK = "resource_leak"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    TYPE_ERROR = "type_error"


@dataclass
class DetectedBug:
    """Represents a detected bug."""
    id: str
    line_number: int
    column_number: int = 0
    category: BugCategory = BugCategory.LOGIC_ERROR
    severity: BugSeverity = BugSeverity.MEDIUM
    title: str = ""
    description: str = ""
    impact: str = ""
    suggested_fix: str = ""
    confidence: float = 0.8
    code_snippet: str = ""
    test_case: Optional[str] = None


@dataclass
class BugDetectionResult:
    """Result of bug detection analysis."""
    file_path: str
    language: str
    bugs_detected: List[DetectedBug]
    static_analysis: AnalysisResult
    analysis_time_ms: float
    model_used: str
    confidence_score: float


class BugDetectorAgent:
    """AI agent specialized in detecting bugs and runtime issues."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.static_analyzer = StaticAnalyzer()
        self.prompt_manager = get_prompt_manager()
        self.model_manager = get_model_manager()
        self.model_name = model_name
        self.bug_counter = 0
    
    async def detect_bugs(self, file_path: str, context: str = "") -> Optional[BugDetectionResult]:
        """Detect bugs in a single code file."""
        import time
        start_time = time.time()
        
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
        
        # Generate AI bug detection
        bugs = await self._detect_bugs_with_ai(
            file_path, code_content, language, context
        )
        
        # Calculate overall confidence
        confidence = sum(bug.confidence for bug in bugs) / len(bugs) if bugs else 0.8
        
        analysis_time = (time.time() - start_time) * 1000
        
        return BugDetectionResult(
            file_path=file_path,
            language=language,
            bugs_detected=bugs,
            static_analysis=static_result,
            analysis_time_ms=analysis_time,
            model_used=self.model_name or "default",
            confidence_score=confidence
        )
    
    async def detect_bugs_in_directory(self, directory_path: str) -> List[BugDetectionResult]:
        """Detect bugs in all supported files in a directory."""
        static_results = self.static_analyzer.analyze_directory(directory_path)
        bug_results = []
        
        # Process files concurrently (but limit concurrency)
        semaphore = asyncio.Semaphore(3)
        
        async def detect_with_semaphore(file_path: str) -> Optional[BugDetectionResult]:
            async with semaphore:
                return await self.detect_bugs(file_path)
        
        # Create tasks for all files
        tasks = [
            detect_with_semaphore(result.file_path) 
            for result in static_results
        ]
        
        # Execute detection
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        for result in results:
            if isinstance(result, BugDetectionResult):
                bug_results.append(result)
            elif isinstance(result, Exception):
                print(f"Bug detection error: {result}")
        
        return bug_results
    
    async def _detect_bugs_with_ai(self,
                                 file_path: str,
                                 code_content: str,
                                 language: str,
                                 context: str) -> List[DetectedBug]:
        """Use AI to detect bugs in code."""
        try:
            # Generate prompt
            system_prompt, user_prompt = self.prompt_manager.generate_bug_detection_prompt(
                file_path, code_content, language, context
            )
            
            # Get AI response
            response = await self.model_manager.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_name=self.model_name
            )
            
            # Parse response into bug objects
            bugs = self._parse_bug_response(response.content, code_content)
            return bugs
            
        except Exception as e:
            print(f"Error in AI bug detection: {e}")
            # Return bugs from static analysis as fallback
            return self._fallback_bug_detection(code_content)
    
    def _parse_bug_response(self, response_content: str, code_content: str) -> List[DetectedBug]:
        """Parse AI response into DetectedBug objects."""
        bugs = []
        
        try:
            # Try to parse as JSON first
            if response_content.strip().startswith('{'):
                data = json.loads(response_content)
                if 'bugs' in data:
                    for bug_data in data['bugs']:
                        bug = self._create_bug_from_json(bug_data, code_content)
                        if bug:
                            bugs.append(bug)
                return bugs
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse structured text
        return self._parse_text_response(response_content, code_content)
    
    def _create_bug_from_json(self, bug_data: Dict[str, Any], code_content: str) -> Optional[DetectedBug]:
        """Create DetectedBug from JSON data."""
        try:
            line_num = int(bug_data.get('line', 1))
            
            # Extract code snippet around the bug
            code_lines = code_content.split('\n')
            snippet_start = max(0, line_num - 3)
            snippet_end = min(len(code_lines), line_num + 2)
            snippet = '\n'.join(code_lines[snippet_start:snippet_end])
            
            # Map severity
            severity_map = {
                'low': BugSeverity.LOW,
                'medium': BugSeverity.MEDIUM,
                'high': BugSeverity.HIGH,
                'critical': BugSeverity.CRITICAL
            }
            
            # Map category based on bug type or description
            category = self._determine_bug_category(
                bug_data.get('type', ''),
                bug_data.get('description', '')
            )
            
            self.bug_counter += 1
            
            return DetectedBug(
                id=f"bug_{self.bug_counter}",
                line_number=line_num,
                category=category,
                severity=severity_map.get(bug_data.get('severity', 'medium'), BugSeverity.MEDIUM),
                title=bug_data.get('type', 'Unknown bug'),
                description=bug_data.get('description', ''),
                impact=bug_data.get('impact', ''),
                suggested_fix=bug_data.get('fix', ''),
                confidence=0.85,
                code_snippet=snippet,
                test_case=self._generate_test_case(bug_data, code_content)
            )
            
        except Exception as e:
            print(f"Error creating bug from JSON: {e}")
            return None
    
    def _parse_text_response(self, response_content: str, code_content: str) -> List[DetectedBug]:
        """Parse structured text response into bugs."""
        bugs = []
        lines = response_content.split('\n')
        
        current_bug = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for bug indicators
            if 'line' in line.lower() and any(char.isdigit() for char in line):
                # Extract line number
                import re
                line_match = re.search(r'line\s*(\d+)', line, re.IGNORECASE)
                if line_match:
                    current_bug['line'] = int(line_match.group(1))
            
            elif 'bug:' in line.lower() or 'issue:' in line.lower():
                current_bug['description'] = line.split(':', 1)[1].strip()
            
            elif 'fix:' in line.lower() or 'solution:' in line.lower():
                current_bug['fix'] = line.split(':', 1)[1].strip()
                
                # Create bug when we have enough info
                if 'line' in current_bug and 'description' in current_bug:
                    self.bug_counter += 1
                    bug = DetectedBug(
                        id=f"bug_{self.bug_counter}",
                        line_number=current_bug['line'],
                        category=BugCategory.LOGIC_ERROR,
                        severity=BugSeverity.MEDIUM,
                        title="Potential Issue",
                        description=current_bug['description'],
                        suggested_fix=current_bug.get('fix', ''),
                        confidence=0.7,
                        code_snippet=self._extract_code_snippet(
                            current_bug['line'], code_content
                        )
                    )
                    bugs.append(bug)
                    current_bug = {}
        
        return bugs
    
    def _determine_bug_category(self, bug_type: str, description: str) -> BugCategory:
        """Determine bug category from type and description."""
        text = (bug_type + " " + description).lower()
        
        if any(word in text for word in ['null', 'none', 'undefined', 'pointer']):
            return BugCategory.NULL_POINTER
        elif any(word in text for word in ['race', 'concurrent', 'thread', 'lock']):
            return BugCategory.RACE_CONDITION
        elif any(word in text for word in ['memory', 'leak', 'allocation']):
            return BugCategory.MEMORY_LEAK
        elif any(word in text for word in ['boundary', 'index', 'range', 'overflow']):
            return BugCategory.BOUNDARY_ERROR
        elif any(word in text for word in ['error', 'exception', 'catch', 'handle']):
            return BugCategory.ERROR_HANDLING
        elif any(word in text for word in ['resource', 'file', 'connection', 'close']):
            return BugCategory.RESOURCE_LEAK
        elif any(word in text for word in ['security', 'injection', 'xss', 'csrf']):
            return BugCategory.SECURITY_ISSUE
        elif any(word in text for word in ['performance', 'slow', 'optimization']):
            return BugCategory.PERFORMANCE_ISSUE
        elif any(word in text for word in ['type', 'cast', 'conversion']):
            return BugCategory.TYPE_ERROR
        else:
            return BugCategory.LOGIC_ERROR
    
    def _extract_code_snippet(self, line_number: int, code_content: str) -> str:
        """Extract code snippet around a specific line."""
        lines = code_content.split('\n')
        start = max(0, line_number - 3)
        end = min(len(lines), line_number + 2)
        return '\n'.join(lines[start:end])
    
    def _generate_test_case(self, bug_data: Dict[str, Any], code_content: str) -> Optional[str]:
        """Generate a simple test case to reproduce the bug."""
        description = bug_data.get('description', '')
        
        if 'null' in description.lower() or 'none' in description.lower():
            return "# Test with None/null input\nresult = function_name(None)\nassert result is not None"
        elif 'boundary' in description.lower() or 'index' in description.lower():
            return "# Test boundary conditions\nresult = function_name([])\nresult = function_name([1] * 1000)"
        elif 'empty' in description.lower():
            return "# Test with empty input\nresult = function_name('')\nassert result != ''"
        
        return None
    
    def _fallback_bug_detection(self, code_content: str) -> List[DetectedBug]:
        """Fallback bug detection using simple heuristics."""
        bugs = []
        lines = code_content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower().strip()
            
            # Check for common bug patterns
            if 'todo' in line_lower or 'fixme' in line_lower:
                self.bug_counter += 1
                bugs.append(DetectedBug(
                    id=f"bug_{self.bug_counter}",
                    line_number=i,
                    category=BugCategory.LOGIC_ERROR,
                    severity=BugSeverity.LOW,
                    title="TODO/FIXME comment",
                    description=f"Code contains TODO or FIXME comment: {line.strip()}",
                    suggested_fix="Complete the implementation or remove the comment",
                    confidence=0.9,
                    code_snippet=line.strip()
                ))
            
            # Check for potential null pointer issues (Python)
            if '.get(' not in line and '[' in line and ']' in line and 'if' not in line_lower:
                if i < len(lines) - 1 and 'if' not in lines[i].lower():
                    self.bug_counter += 1
                    bugs.append(DetectedBug(
                        id=f"bug_{self.bug_counter}",
                        line_number=i,
                        category=BugCategory.NULL_POINTER,
                        severity=BugSeverity.MEDIUM,
                        title="Potential KeyError",
                        description="Dictionary/list access without bounds checking",
                        suggested_fix="Use .get() method or add bounds checking",
                        confidence=0.6,
                        code_snippet=line.strip()
                    ))
        
        return bugs
    
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
    
    def generate_bug_report(self, results: List[BugDetectionResult]) -> str:
        """Generate a comprehensive bug detection report."""
        if not results:
            return "No files were analyzed for bugs."
        
        total_files = len(results)
        total_bugs = sum(len(result.bugs_detected) for result in results)
        
        # Categorize bugs by severity
        severity_counts = {severity: 0 for severity in BugSeverity}
        category_counts = {category: 0 for category in BugCategory}
        
        for result in results:
            for bug in result.bugs_detected:
                severity_counts[bug.severity] += 1
                category_counts[bug.category] += 1
        
        # Calculate average confidence
        all_bugs = [bug for result in results for bug in result.bugs_detected]
        avg_confidence = sum(bug.confidence for bug in all_bugs) / len(all_bugs) if all_bugs else 0
        
        report = f"""# Bug Detection Report

## Summary
- **Files Analyzed:** {total_files}
- **Total Bugs Found:** {total_bugs}
- **Average Confidence:** {avg_confidence:.1%}
- **Analysis Time:** {sum(r.analysis_time_ms for r in results):.0f}ms

## Severity Distribution
- **Critical:** {severity_counts[BugSeverity.CRITICAL]}
- **High:** {severity_counts[BugSeverity.HIGH]}
- **Medium:** {severity_counts[BugSeverity.MEDIUM]}
- **Low:** {severity_counts[BugSeverity.LOW]}

## Bug Categories
"""
        
        for category, count in category_counts.items():
            if count > 0:
                report += f"- **{category.value.replace('_', ' ').title()}:** {count}\n"
        
        report += "\n## Detailed Findings\n\n"
        
        for result in results:
            if result.bugs_detected:
                report += f"### {result.file_path}\n"
                report += f"**Language:** {result.language}\n"
                report += f"**Bugs Found:** {len(result.bugs_detected)}\n\n"
                
                for bug in result.bugs_detected:
                    report += f"#### {bug.title} (Line {bug.line_number})\n"
                    report += f"**Severity:** {bug.severity.value.title()}\n"
                    report += f"**Category:** {bug.category.value.replace('_', ' ').title()}\n"
                    report += f"**Confidence:** {bug.confidence:.0%}\n\n"
                    report += f"**Description:** {bug.description}\n\n"
                    
                    if bug.impact:
                        report += f"**Impact:** {bug.impact}\n\n"
                    
                    if bug.suggested_fix:
                        report += f"**Suggested Fix:** {bug.suggested_fix}\n\n"
                    
                    if bug.code_snippet:
                        report += f"**Code Snippet:**\n```{result.language}\n{bug.code_snippet}\n```\n\n"
                    
                    if bug.test_case:
                        report += f"**Test Case:**\n```{result.language}\n{bug.test_case}\n```\n\n"
                    
                    report += "---\n\n"
        
        return report
    
    def get_bug_statistics(self, results: List[BugDetectionResult]) -> Dict[str, Any]:
        """Get statistical summary of bug detection results."""
        if not results:
            return {}
        
        all_bugs = [bug for result in results for bug in result.bugs_detected]
        
        severity_dist = {}
        category_dist = {}
        confidence_scores = []
        
        for bug in all_bugs:
            severity_dist[bug.severity.value] = severity_dist.get(bug.severity.value, 0) + 1
            category_dist[bug.category.value] = category_dist.get(bug.category.value, 0) + 1
            confidence_scores.append(bug.confidence)
        
        return {
            'total_files': len(results),
            'total_bugs': len(all_bugs),
            'bugs_per_file': len(all_bugs) / len(results) if results else 0,
            'severity_distribution': severity_dist,
            'category_distribution': category_dist,
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'confidence_range': {
                'min': min(confidence_scores) if confidence_scores else 0,
                'max': max(confidence_scores) if confidence_scores else 0
            },
            'high_confidence_bugs': len([c for c in confidence_scores if c > 0.8]),
            'total_analysis_time_ms': sum(r.analysis_time_ms for r in results)
        }