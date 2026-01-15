"""
Prompt templates and management for code analysis tasks.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from src.analyzers.static_analyzer import AnalysisResult, CodeIssue, FileMetrics


class PromptType(Enum):
    """Types of prompts for different tasks."""
    CODE_REVIEW = "code_review"
    BUG_DETECTION = "bug_detection"
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    system_prompt: str
    user_template: str
    expected_format: Optional[str] = None
    max_tokens: int = 4000


class PromptManager:
    """Manages prompt templates for different code analysis tasks."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Initialize all prompt templates."""
        return {
            PromptType.CODE_REVIEW: PromptTemplate(
                system_prompt="""You are an expert code reviewer with deep knowledge of software engineering best practices, security, performance, and maintainability. Your role is to analyze code and provide constructive, actionable feedback.

Focus Areas:
- Code quality and best practices
- Security vulnerabilities
- Performance optimization opportunities
- Maintainability and readability
- Potential bugs or edge cases
- Design patterns and architecture

Provide specific, actionable suggestions with examples when possible.""",
                
                user_template="""Please review the following code and provide detailed feedback:

**File:** {file_path}

**Code:**
```{language}
{code_content}
```

**Static Analysis Results:**
- Lines of Code: {loc}
- Cyclomatic Complexity: {complexity}
- Functions: {function_count}
- Classes: {class_count}

**Existing Issues Found:**
{existing_issues}

Please provide:
1. Overall assessment of code quality
2. Specific issues and improvements
3. Security concerns (if any)
4. Performance optimization opportunities
5. Maintainability suggestions

Format your response as structured feedback with clear sections.""",
                
                expected_format="structured_feedback",
                max_tokens=3000
            ),
            
            PromptType.BUG_DETECTION: PromptTemplate(
                system_prompt="""You are a software bug detection specialist. Your expertise lies in identifying potential bugs, edge cases, and runtime errors in code. Focus on logic errors, null pointer exceptions, race conditions, memory leaks, and other runtime issues that static analysis might miss.

Analysis Approach:
- Trace through code execution paths
- Identify potential null/undefined values
- Check boundary conditions
- Look for race conditions in concurrent code
- Verify error handling completeness
- Check for resource leaks""",
                
                user_template="""Analyze the following code for potential bugs and runtime issues:

**File:** {file_path}

**Code:**
```{language}
{code_content}
```

**Context:**
{context}

Please identify:
1. Potential runtime errors
2. Logic bugs
3. Edge cases not handled
4. Error handling gaps
5. Resource management issues

For each issue, provide:
- Line number(s)
- Bug description
- Potential impact
- Suggested fix

Format as JSON with this structure:
```json
{
  "bugs": [
    {
      "line": number,
      "type": "bug_type",
      "severity": "low|medium|high|critical",
      "description": "detailed description",
      "impact": "potential impact",
      "fix": "suggested fix"
    }
  ]
}
```""",
                
                expected_format="json",
                max_tokens=2500
            ),
            
            PromptType.CODE_GENERATION: PromptTemplate(
                system_prompt="""You are an expert software developer who writes clean, efficient, and well-documented code. You follow best practices, write readable code, include proper error handling, and add appropriate comments and docstrings.

Code Standards:
- Follow language-specific conventions
- Include comprehensive error handling
- Write clear, descriptive variable names
- Add docstrings for functions and classes
- Include type hints where applicable
- Follow SOLID principles
- Write testable code""",
                
                user_template="""Generate code based on the following requirements:

**Requirements:**
{requirements}

**Language:** {language}

**Context:**
{context}

**Constraints:**
{constraints}

Please provide:
1. Complete, working code
2. Proper documentation
3. Error handling
4. Type hints (if applicable)
5. Example usage (if appropriate)

The code should be production-ready and follow best practices.""",
                
                expected_format="code_with_documentation",
                max_tokens=4000
            ),
            
            PromptType.TEST_GENERATION: PromptTemplate(
                system_prompt="""You are a test automation specialist who creates comprehensive test suites. You understand testing best practices including unit tests, integration tests, edge cases, mocking, and test data management.

Testing Principles:
- Write clear, readable test cases
- Cover happy path, edge cases, and error conditions
- Use descriptive test names
- Include setup and teardown as needed
- Mock external dependencies appropriately
- Follow AAA pattern (Arrange, Act, Assert)""",
                
                user_template="""Generate comprehensive tests for the following code:

**Code to Test:**
```{language}
{code_content}
```

**File:** {file_path}

**Requirements:**
{requirements}

Please generate:
1. Unit tests covering all functions/methods
2. Edge case tests
3. Error condition tests
4. Integration tests (if applicable)
5. Mock setups for external dependencies

Use the appropriate testing framework for {language} and include:
- Clear test names
- Setup/teardown if needed
- Assertions with meaningful messages
- Test data/fixtures""",
                
                expected_format="test_code",
                max_tokens=4000
            ),
            
            PromptType.REFACTORING: PromptTemplate(
                system_prompt="""You are a code refactoring expert who improves code structure, readability, and maintainability while preserving functionality. You understand design patterns, SOLID principles, and clean code practices.

Refactoring Focus:
- Extract methods/functions for better modularity
- Eliminate code duplication
- Improve naming conventions
- Simplify complex logic
- Apply appropriate design patterns
- Enhance error handling
- Improve performance where applicable""",
                
                user_template="""Refactor the following code to improve its structure and maintainability:

**Current Code:**
```{language}
{code_content}
```

**File:** {file_path}

**Issues to Address:**
{issues_to_fix}

**Goals:**
{refactoring_goals}

Please provide:
1. Refactored code with improvements
2. Explanation of changes made
3. Benefits of the refactoring
4. Any potential breaking changes

Maintain the same functionality while improving code quality.""",
                
                expected_format="refactored_code_with_explanation",
                max_tokens=4000
            ),
            
            PromptType.DOCUMENTATION: PromptTemplate(
                system_prompt="""You are a technical documentation specialist who creates clear, comprehensive documentation for code. You understand how to write effective docstrings, API documentation, and code comments that help developers understand and use the code effectively.

Documentation Standards:
- Write clear, concise descriptions
- Document parameters, return values, and exceptions
- Include usage examples
- Explain complex algorithms or business logic
- Use consistent formatting
- Follow language-specific documentation conventions""",
                
                user_template="""Generate comprehensive documentation for the following code:

**Code:**
```{language}
{code_content}
```

**File:** {file_path}

**Documentation Type:** {doc_type}

Please provide:
1. Function/class docstrings
2. Parameter and return value documentation
3. Usage examples
4. Any necessary inline comments
5. API documentation (if applicable)

Follow {language} documentation conventions and make it developer-friendly.""",
                
                expected_format="documented_code",
                max_tokens=3000
            )
        }
    
    def get_template(self, prompt_type: PromptType) -> PromptTemplate:
        """Get a prompt template by type."""
        return self.templates[prompt_type]
    
    def generate_code_review_prompt(self, 
                                  file_path: str,
                                  code_content: str,
                                  language: str,
                                  analysis_result: AnalysisResult) -> tuple[str, str]:
        """Generate a code review prompt."""
        template = self.get_template(PromptType.CODE_REVIEW)
        
        # Format existing issues
        issues_text = ""
        if analysis_result.issues:
            issues_text = "\n".join([
                f"- Line {issue.line_number}: {issue.message} ({issue.severity.value})"
                for issue in analysis_result.issues
            ])
        else:
            issues_text = "No significant issues detected by static analysis."
        
        user_prompt = template.user_template.format(
            file_path=file_path,
            code_content=code_content,
            language=language,
            loc=analysis_result.metrics.lines_of_code,
            complexity=analysis_result.metrics.cyclomatic_complexity,
            function_count=analysis_result.metrics.function_count,
            class_count=analysis_result.metrics.class_count,
            existing_issues=issues_text
        )
        
        return template.system_prompt, user_prompt
    
    def generate_bug_detection_prompt(self,
                                    file_path: str,
                                    code_content: str,
                                    language: str,
                                    context: str = "") -> tuple[str, str]:
        """Generate a bug detection prompt."""
        template = self.get_template(PromptType.BUG_DETECTION)
        
        user_prompt = template.user_template.format(
            file_path=file_path,
            code_content=code_content,
            language=language,
            context=context or "No additional context provided."
        )
        
        return template.system_prompt, user_prompt
    
    def generate_code_generation_prompt(self,
                                      requirements: str,
                                      language: str,
                                      context: str = "",
                                      constraints: str = "") -> tuple[str, str]:
        """Generate a code generation prompt."""
        template = self.get_template(PromptType.CODE_GENERATION)
        
        user_prompt = template.user_template.format(
            requirements=requirements,
            language=language,
            context=context or "No additional context provided.",
            constraints=constraints or "No specific constraints."
        )
        
        return template.system_prompt, user_prompt
    
    def generate_test_generation_prompt(self,
                                      file_path: str,
                                      code_content: str,
                                      language: str,
                                      requirements: str = "") -> tuple[str, str]:
        """Generate a test generation prompt."""
        template = self.get_template(PromptType.TEST_GENERATION)
        
        user_prompt = template.user_template.format(
            file_path=file_path,
            code_content=code_content,
            language=language,
            requirements=requirements or "Generate comprehensive test coverage."
        )
        
        return template.system_prompt, user_prompt
    
    def generate_refactoring_prompt(self,
                                  file_path: str,
                                  code_content: str,
                                  language: str,
                                  issues_to_fix: List[str],
                                  refactoring_goals: List[str]) -> tuple[str, str]:
        """Generate a refactoring prompt."""
        template = self.get_template(PromptType.REFACTORING)
        
        issues_text = "\n".join([f"- {issue}" for issue in issues_to_fix])
        goals_text = "\n".join([f"- {goal}" for goal in refactoring_goals])
        
        user_prompt = template.user_template.format(
            file_path=file_path,
            code_content=code_content,
            language=language,
            issues_to_fix=issues_text,
            refactoring_goals=goals_text
        )
        
        return template.system_prompt, user_prompt
    
    def generate_documentation_prompt(self,
                                    file_path: str,
                                    code_content: str,
                                    language: str,
                                    doc_type: str = "comprehensive") -> tuple[str, str]:
        """Generate a documentation prompt."""
        template = self.get_template(PromptType.DOCUMENTATION)
        
        user_prompt = template.user_template.format(
            file_path=file_path,
            code_content=code_content,
            language=language,
            doc_type=doc_type
        )
        
        return template.system_prompt, user_prompt


# Global instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager