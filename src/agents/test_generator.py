"""
Test generator agent that creates comprehensive test suites for existing code.
"""

import asyncio
import ast
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from src.analyzers.static_analyzer import StaticAnalyzer, AnalysisResult
from src.llm.model_interface import get_model_manager, ModelResponse
from src.llm.prompt_manager import get_prompt_manager, PromptType


class TestType(Enum):
    """Types of tests that can be generated."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    FUNCTIONAL_TEST = "functional_test"
    PERFORMANCE_TEST = "performance_test"
    SECURITY_TEST = "security_test"
    EDGE_CASE_TEST = "edge_case_test"


class TestFramework(Enum):
    """Supported testing frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    GTEST = "gtest"
    GOTEST = "gotest"
    CARGO_TEST = "cargo_test"


@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    description: str
    test_code: str
    test_type: TestType
    target_function: str
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None
    expected_outcome: str = ""
    covers_edge_case: bool = False
    mock_dependencies: List[str] = None
    
    def __post_init__(self):
        if self.mock_dependencies is None:
            self.mock_dependencies = []


@dataclass
class TestSuite:
    """Collection of test cases for a code unit."""
    file_path: str
    target_code: str
    language: str
    framework: TestFramework
    test_cases: List[TestCase]
    setup_code: str = ""
    imports: List[str] = None
    fixtures: List[str] = None
    coverage_estimate: float = 0.0
    
    def __post_init__(self):
        if self.imports is None:
            self.imports = []
        if self.fixtures is None:
            self.fixtures = []


@dataclass
class TestGenerationResult:
    """Result of test generation process."""
    source_file: str
    test_suite: TestSuite
    static_analysis: AnalysisResult
    generation_time_ms: float
    model_used: str
    confidence: float
    coverage_analysis: Dict[str, Any]


class TestGeneratorAgent:
    """AI agent that generates comprehensive test suites."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.static_analyzer = StaticAnalyzer()
        self.prompt_manager = get_prompt_manager()
        self.model_manager = get_model_manager()
        self.model_name = model_name
    
    async def generate_tests(self, file_path: str, 
                           framework: Optional[TestFramework] = None,
                           test_types: Optional[List[TestType]] = None) -> Optional[TestGenerationResult]:
        """Generate comprehensive tests for a code file."""
        import time
        start_time = time.time()
        
        if not Path(file_path).exists():
            return None
        
        # Perform static analysis
        static_result = self.static_analyzer.analyze_file(file_path)
        if not static_result:
            return None
        
        # Read source code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        
        # Determine language and framework
        language = self._detect_language(file_path)
        if framework is None:
            framework = self._determine_framework(language)
        
        # Set default test types
        if test_types is None:
            test_types = [TestType.UNIT_TEST, TestType.EDGE_CASE_TEST]
        
        # Analyze code structure
        code_analysis = self._analyze_code_structure(source_code, language)
        
        # Generate test suite
        test_suite = await self._generate_test_suite(
            file_path, source_code, language, framework, test_types, code_analysis
        )
        
        # Calculate coverage estimate
        coverage_analysis = self._analyze_coverage(test_suite, code_analysis)
        test_suite.coverage_estimate = coverage_analysis.get('estimated_coverage', 0.0)
        
        # Calculate confidence
        confidence = self._calculate_confidence(test_suite, code_analysis)
        
        generation_time = (time.time() - start_time) * 1000
        
        return TestGenerationResult(
            source_file=file_path,
            test_suite=test_suite,
            static_analysis=static_result,
            generation_time_ms=generation_time,
            model_used=self.model_name or "default",
            confidence=confidence,
            coverage_analysis=coverage_analysis
        )
    
    async def generate_tests_for_directory(self, directory_path: str,
                                         framework: Optional[TestFramework] = None) -> List[TestGenerationResult]:
        """Generate tests for all supported files in a directory."""
        static_results = self.static_analyzer.analyze_directory(directory_path)
        test_results = []
        
        # Process files concurrently
        semaphore = asyncio.Semaphore(3)
        
        async def generate_with_semaphore(file_path: str) -> Optional[TestGenerationResult]:
            async with semaphore:
                return await self.generate_tests(file_path, framework)
        
        tasks = [generate_with_semaphore(result.file_path) for result in static_results]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, TestGenerationResult):
                test_results.append(result)
            elif isinstance(result, Exception):
                print(f"Test generation error: {result}")
        
        return test_results
    
    async def _generate_test_suite(self,
                                 file_path: str,
                                 source_code: str,
                                 language: str,
                                 framework: TestFramework,
                                 test_types: List[TestType],
                                 code_analysis: Dict[str, Any]) -> TestSuite:
        """Generate a complete test suite using AI."""
        try:
            # Build requirements for test generation
            requirements = self._build_test_requirements(test_types, code_analysis)
            
            # Generate prompt
            system_prompt, user_prompt = self.prompt_manager.generate_test_generation_prompt(
                file_path=file_path,
                code_content=source_code,
                language=language,
                requirements=requirements
            )
            
            # Get AI response
            response = await self.model_manager.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_name=self.model_name
            )
            
            # Parse response into test suite
            test_suite = self._parse_test_response(
                response.content, file_path, source_code, language, framework, code_analysis
            )
            
            return test_suite
            
        except Exception as e:
            print(f"Error generating test suite: {e}")
            return self._create_fallback_test_suite(file_path, source_code, language, framework)
    
    def _analyze_code_structure(self, source_code: str, language: str) -> Dict[str, Any]:
        """Analyze code structure to identify testable units."""
        analysis = {
            'functions': [],
            'classes': [],
            'methods': [],
            'imports': [],
            'complexity': 'low',
            'dependencies': [],
            'async_functions': [],
            'error_handling': False
        }
        
        if language == 'python':
            try:
                tree = ast.parse(source_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_info = {
                            'name': node.name,
                            'line': node.lineno,
                            'args': [arg.arg for arg in node.args.args],
                            'returns': hasattr(node, 'returns'),
                            'is_async': isinstance(node, ast.AsyncFunctionDef),
                            'docstring': ast.get_docstring(node)
                        }
                        
                        if isinstance(node, ast.AsyncFunctionDef):
                            analysis['async_functions'].append(func_info)
                        
                        analysis['functions'].append(func_info)
                    
                    elif isinstance(node, ast.ClassDef):
                        class_info = {
                            'name': node.name,
                            'line': node.lineno,
                            'methods': [],
                            'docstring': ast.get_docstring(node)
                        }
                        
                        # Find methods in class
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_info = {
                                    'name': item.name,
                                    'line': item.lineno,
                                    'args': [arg.arg for arg in item.args.args],
                                    'is_property': any(
                                        isinstance(dec, ast.Name) and dec.id == 'property'
                                        for dec in item.decorator_list
                                    )
                                }
                                class_info['methods'].append(method_info)
                                analysis['methods'].append(method_info)
                        
                        analysis['classes'].append(class_info)
                    
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                analysis['imports'].append(alias.name)
                        else:
                            module = node.module or ''
                            for alias in node.names:
                                analysis['imports'].append(f"{module}.{alias.name}")
                    
                    elif isinstance(node, (ast.Try, ast.Raise, ast.ExceptHandler)):
                        analysis['error_handling'] = True
                
            except SyntaxError:
                # Fallback for syntax errors
                pass
        
        # Determine complexity
        total_functions = len(analysis['functions'])
        total_classes = len(analysis['classes'])
        
        if total_functions + total_classes > 10:
            analysis['complexity'] = 'high'
        elif total_functions + total_classes > 5:
            analysis['complexity'] = 'medium'
        
        return analysis
    
    def _build_test_requirements(self, test_types: List[TestType], code_analysis: Dict[str, Any]) -> str:
        """Build requirements string for test generation."""
        requirements = []
        
        # Basic requirements
        requirements.append("Generate comprehensive test coverage including:")
        
        for test_type in test_types:
            if test_type == TestType.UNIT_TEST:
                requirements.append("- Unit tests for all functions and methods")
            elif test_type == TestType.EDGE_CASE_TEST:
                requirements.append("- Edge case tests (empty inputs, boundary values, None/null)")
            elif test_type == TestType.INTEGRATION_TEST:
                requirements.append("- Integration tests for component interactions")
            elif test_type == TestType.SECURITY_TEST:
                requirements.append("- Security tests for input validation and injection prevention")
            elif test_type == TestType.PERFORMANCE_TEST:
                requirements.append("- Performance tests for time-critical functions")
        
        # Specific requirements based on code analysis
        if code_analysis.get('async_functions'):
            requirements.append("- Async/await testing for asynchronous functions")
        
        if code_analysis.get('error_handling'):
            requirements.append("- Error handling and exception testing")
        
        if code_analysis.get('classes'):
            requirements.append("- Class instantiation and method testing")
        
        # Mock requirements
        if code_analysis.get('dependencies'):
            requirements.append("- Mock external dependencies appropriately")
        
        return '\n'.join(requirements)
    
    def _parse_test_response(self,
                           response_content: str,
                           file_path: str,
                           source_code: str,
                           language: str,
                           framework: TestFramework,
                           code_analysis: Dict[str, Any]) -> TestSuite:
        """Parse AI response into TestSuite object."""
        # Split response into sections
        sections = self._split_test_response(response_content)
        
        # Extract imports
        imports = self._extract_test_imports(sections.get('imports', ''), framework)
        
        # Extract setup code
        setup_code = sections.get('setup', '')
        
        # Extract test cases
        test_cases = self._extract_test_cases(sections, language, code_analysis)
        
        # Extract fixtures if any
        fixtures = self._extract_fixtures(sections.get('fixtures', ''))
        
        return TestSuite(
            file_path=file_path,
            target_code=source_code,
            language=language,
            framework=framework,
            test_cases=test_cases,
            setup_code=setup_code,
            imports=imports,
            fixtures=fixtures
        )
    
    def _split_test_response(self, response: str) -> Dict[str, str]:
        """Split test response into logical sections."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check for section headers
            if any(header in line_lower for header in ['## ', '### ', '# ']):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Determine section type
                if 'import' in line_lower:
                    current_section = 'imports'
                elif 'setup' in line_lower or 'fixture' in line_lower:
                    current_section = 'setup'
                elif 'test' in line_lower:
                    current_section = 'tests'
                elif 'mock' in line_lower:
                    current_section = 'mocks'
                else:
                    current_section = 'general'
                
                current_content = []
            else:
                current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If no clear sections, treat as tests
        if not sections and response.strip():
            sections['tests'] = response.strip()
        
        return sections
    
    def _extract_test_imports(self, imports_text: str, framework: TestFramework) -> List[str]:
        """Extract test imports from text."""
        imports = []
        
        # Add framework-specific imports
        framework_imports = {
            TestFramework.PYTEST: ['import pytest'],
            TestFramework.UNITTEST: ['import unittest'],
            TestFramework.JEST: ['const { test, expect } = require("@jest/globals");'],
            TestFramework.MOCHA: ['const { describe, it } = require("mocha");', 'const { expect } = require("chai");']
        }
        
        if framework in framework_imports:
            imports.extend(framework_imports[framework])
        
        # Extract custom imports from response
        if imports_text:
            lines = imports_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('import ', 'from ', 'const ', 'require(')):
                    imports.append(line)
        
        return imports
    
    def _extract_test_cases(self, sections: Dict[str, str], language: str, code_analysis: Dict[str, Any]) -> List[TestCase]:
        """Extract test cases from response sections."""
        test_cases = []
        tests_content = sections.get('tests', '')
        
        if not tests_content:
            # Fallback: generate basic test cases from code analysis
            return self._generate_fallback_tests(code_analysis, language)
        
        # Split into individual test functions
        test_functions = self._split_test_functions(tests_content, language)
        
        for i, test_func in enumerate(test_functions):
            test_case = self._parse_single_test(test_func, language, i)
            if test_case:
                test_cases.append(test_case)
        
        return test_cases
    
    def _split_test_functions(self, tests_content: str, language: str) -> List[str]:
        """Split test content into individual test functions."""
        functions = []
        
        if language == 'python':
            # Look for function definitions
            lines = tests_content.split('\n')
            current_function = []
            in_function = False
            indent_level = 0
            
            for line in lines:
                if line.strip().startswith('def test_') or line.strip().startswith('async def test_'):
                    # Start of new test function
                    if current_function:
                        functions.append('\n'.join(current_function))
                    current_function = [line]
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                elif in_function:
                    current_line_indent = len(line) - len(line.lstrip())
                    if line.strip() and current_line_indent <= indent_level and not line.startswith(' '):
                        # End of function
                        functions.append('\n'.join(current_function))
                        current_function = []
                        in_function = False
                        if line.strip().startswith('def test_'):
                            current_function = [line]
                            in_function = True
                            indent_level = current_line_indent
                    else:
                        current_function.append(line)
            
            # Add last function
            if current_function:
                functions.append('\n'.join(current_function))
        
        return functions
    
    def _parse_single_test(self, test_func: str, language: str, index: int) -> Optional[TestCase]:
        """Parse a single test function into TestCase object."""
        if not test_func.strip():
            return None
        
        # Extract test name
        name_match = re.search(r'def\s+(test_\w+)', test_func)
        name = name_match.group(1) if name_match else f"test_{index}"
        
        # Extract description from docstring or comments
        description = self._extract_test_description(test_func)
        
        # Determine test type
        test_type = self._determine_test_type(test_func, name)
        
        # Extract target function
        target_function = self._extract_target_function(test_func)
        
        # Check for edge case indicators
        covers_edge_case = any(word in test_func.lower() for word in 
                              ['edge', 'boundary', 'empty', 'none', 'null', 'zero', 'negative'])
        
        return TestCase(
            name=name,
            description=description,
            test_code=test_func.strip(),
            test_type=test_type,
            target_function=target_function,
            covers_edge_case=covers_edge_case
        )
    
    def _extract_test_description(self, test_func: str) -> str:
        """Extract description from test function."""
        # Look for docstring
        docstring_match = re.search(r'"""(.*?)"""', test_func, re.DOTALL)
        if docstring_match:
            return docstring_match.group(1).strip()
        
        # Look for single-line docstring
        docstring_match = re.search(r'"([^"]+)"', test_func)
        if docstring_match:
            return docstring_match.group(1).strip()
        
        # Look for comment
        comment_match = re.search(r'#\s*(.+)', test_func)
        if comment_match:
            return comment_match.group(1).strip()
        
        return "Test case"
    
    def _determine_test_type(self, test_func: str, name: str) -> TestType:
        """Determine the type of test based on content and name."""
        func_lower = test_func.lower()
        name_lower = name.lower()
        
        if 'edge' in name_lower or 'boundary' in name_lower:
            return TestType.EDGE_CASE_TEST
        elif 'integration' in name_lower or 'end_to_end' in name_lower:
            return TestType.INTEGRATION_TEST
        elif 'performance' in name_lower or 'speed' in name_lower:
            return TestType.PERFORMANCE_TEST
        elif 'security' in name_lower or 'injection' in func_lower:
            return TestType.SECURITY_TEST
        else:
            return TestType.UNIT_TEST
    
    def _extract_target_function(self, test_func: str) -> str:
        """Extract the target function being tested."""
        # Look for function calls in the test
        func_calls = re.findall(r'(\w+)\s*\(', test_func)
        
        # Filter out test framework functions
        test_keywords = {'assert', 'assertEqual', 'assertTrue', 'assertFalse', 'expect', 'test', 'describe', 'it'}
        
        for call in func_calls:
            if call not in test_keywords and not call.startswith('test_'):
                return call
        
        return "unknown"
    
    def _extract_fixtures(self, fixtures_text: str) -> List[str]:
        """Extract test fixtures from text."""
        fixtures = []
        
        if fixtures_text:
            # Look for fixture functions
            fixture_matches = re.findall(r'@pytest\.fixture[^\n]*\ndef\s+(\w+)', fixtures_text)
            fixtures.extend(fixture_matches)
        
        return fixtures
    
    def _generate_fallback_tests(self, code_analysis: Dict[str, Any], language: str) -> List[TestCase]:
        """Generate basic test cases as fallback."""
        test_cases = []
        
        # Generate tests for functions
        for i, func in enumerate(code_analysis.get('functions', [])):
            test_case = TestCase(
                name=f"test_{func['name']}",
                description=f"Test {func['name']} function",
                test_code=self._generate_basic_test_code(func, language),
                test_type=TestType.UNIT_TEST,
                target_function=func['name']
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_basic_test_code(self, func_info: Dict[str, Any], language: str) -> str:
        """Generate basic test code for a function."""
        if language == 'python':
            return f"""def test_{func_info['name']}():
    \"\"\"Test {func_info['name']} function.\"\"\"
    # TODO: Implement test for {func_info['name']}
    pass"""
        
        return f"// TODO: Implement test for {func_info['name']}"
    
    def _analyze_coverage(self, test_suite: TestSuite, code_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test coverage."""
        total_functions = len(code_analysis.get('functions', []))
        total_methods = len(code_analysis.get('methods', []))
        total_testable = total_functions + total_methods
        
        if total_testable == 0:
            return {'estimated_coverage': 0.0, 'tested_functions': 0, 'total_functions': 0}
        
        # Count how many functions are tested
        tested_functions = set()
        for test_case in test_suite.test_cases:
            if test_case.target_function != 'unknown':
                tested_functions.add(test_case.target_function)
        
        coverage = len(tested_functions) / total_testable
        
        return {
            'estimated_coverage': coverage,
            'tested_functions': len(tested_functions),
            'total_functions': total_testable,
            'untested_functions': total_testable - len(tested_functions),
            'edge_case_coverage': sum(1 for tc in test_suite.test_cases if tc.covers_edge_case)
        }
    
    def _calculate_confidence(self, test_suite: TestSuite, code_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the generated tests."""
        confidence = 0.7  # Base confidence
        
        # Factor in number of test cases
        if len(test_suite.test_cases) > 5:
            confidence += 0.1
        elif len(test_suite.test_cases) > 10:
            confidence += 0.15
        
        # Factor in coverage
        if test_suite.coverage_estimate > 0.8:
            confidence += 0.1
        elif test_suite.coverage_estimate > 0.5:
            confidence += 0.05
        
        # Factor in test quality indicators
        quality_indicators = 0
        for test_case in test_suite.test_cases:
            if test_case.description and len(test_case.description) > 10:
                quality_indicators += 1
            if 'assert' in test_case.test_code.lower():
                quality_indicators += 1
            if test_case.covers_edge_case:
                quality_indicators += 1
        
        if quality_indicators > len(test_suite.test_cases):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _create_fallback_test_suite(self,
                                  file_path: str,
                                  source_code: str,
                                  language: str,
                                  framework: TestFramework) -> TestSuite:
        """Create a fallback test suite when AI generation fails."""
        return TestSuite(
            file_path=file_path,
            target_code=source_code,
            language=language,
            framework=framework,
            test_cases=[],
            imports=['# Test generation failed - manual implementation required']
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
            '.rs': 'rust'
        }
        
        return language_map.get(suffix, 'python')
    
    def _determine_framework(self, language: str) -> TestFramework:
        """Determine appropriate testing framework for language."""
        framework_map = {
            'python': TestFramework.PYTEST,
            'javascript': TestFramework.JEST,
            'typescript': TestFramework.JEST,
            'java': TestFramework.JUNIT,
            'cpp': TestFramework.GTEST,
            'c': TestFramework.GTEST,
            'go': TestFramework.GOTEST,
            'rust': TestFramework.CARGO_TEST
        }
        
        return framework_map.get(language, TestFramework.PYTEST)
    
    def generate_test_file(self, result: TestGenerationResult, output_dir: str) -> str:
        """Generate a complete test file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create test filename
        source_file = Path(result.source_file)
        test_filename = f"test_{source_file.stem}.py"  # Assume Python for now
        test_file_path = output_path / test_filename
        
        # Build test file content
        content = []
        
        # Add header
        content.append(f"# Generated tests for {result.source_file}")
        content.append(f"# Framework: {result.test_suite.framework.value}")
        content.append(f"# Generated by AI Code Agent")
        content.append(f"# Coverage estimate: {result.test_suite.coverage_estimate:.1%}")
        content.append("")
        
        # Add imports
        for import_line in result.test_suite.imports:
            content.append(import_line)
        content.append("")
        
        # Add setup code
        if result.test_suite.setup_code:
            content.append("# Setup")
            content.append(result.test_suite.setup_code)
            content.append("")
        
        # Add fixtures
        for fixture in result.test_suite.fixtures:
            content.append(f"# Fixture: {fixture}")
        
        if result.test_suite.fixtures:
            content.append("")
        
        # Add test cases
        for test_case in result.test_suite.test_cases:
            content.append(f"# {test_case.description}")
            content.append(test_case.test_code)
            content.append("")
        
        # Write to file
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        return str(test_file_path)
    
    def generate_test_report(self, results: List[TestGenerationResult]) -> str:
        """Generate a comprehensive test generation report."""
        if not results:
            return "No test generation results to report."
        
        total_files = len(results)
        total_tests = sum(len(r.test_suite.test_cases) for r in results)
        avg_coverage = sum(r.test_suite.coverage_estimate for r in results) / total_files
        avg_confidence = sum(r.confidence for r in results) / total_files
        
        report = f"""# Test Generation Report

## Summary
- **Files Processed:** {total_files}
- **Total Tests Generated:** {total_tests}
- **Average Coverage:** {avg_coverage:.1%}
- **Average Confidence:** {avg_confidence:.1%}
- **Total Generation Time:** {sum(r.generation_time_ms for r in results):.0f}ms

## Test Statistics
"""
        
        # Count by test type
        test_type_counts = {}
        framework_counts = {}
        
        for result in results:
            framework = result.test_suite.framework.value
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
            
            for test_case in result.test_suite.test_cases:
                test_type = test_case.test_type.value
                test_type_counts[test_type] = test_type_counts.get(test_type, 0) + 1
        
        report += "\n**Test Types:**\n"
        for test_type, count in test_type_counts.items():
            report += f"- {test_type.replace('_', ' ').title()}: {count}\n"
        
        report += "\n**Frameworks Used:**\n"
        for framework, count in framework_counts.items():
            report += f"- {framework}: {count} files\n"
        
        report += "\n## Detailed Results\n\n"
        
        for result in results:
            suite = result.test_suite
            report += f"### {result.source_file}\n"
            report += f"**Framework:** {suite.framework.value}\n"
            report += f"**Tests Generated:** {len(suite.test_cases)}\n"
            report += f"**Coverage Estimate:** {suite.coverage_estimate:.1%}\n"
            report += f"**Confidence:** {result.confidence:.1%}\n"
            report += f"**Generation Time:** {result.generation_time_ms:.0f}ms\n\n"
            
            if suite.test_cases:
                report += "**Test Cases:**\n"
                for test_case in suite.test_cases:
                    report += f"- `{test_case.name}`: {test_case.description}\n"
                report += "\n"
            
            report += "---\n\n"
        
        return report