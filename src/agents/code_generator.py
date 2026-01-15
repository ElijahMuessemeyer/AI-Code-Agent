"""
Code generator agent that creates code from natural language requirements.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from src.llm.model_interface import get_model_manager, ModelResponse
from src.llm.prompt_manager import get_prompt_manager, PromptType


class CodeType(Enum):
    """Types of code that can be generated."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    SCRIPT = "script"
    API_ENDPOINT = "api_endpoint"
    DATA_MODEL = "data_model"
    UTILITY = "utility"
    ALGORITHM = "algorithm"


class CodeQuality(Enum):
    """Quality levels for generated code."""
    BASIC = "basic"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


@dataclass
class CodeRequirement:
    """Requirements for code generation."""
    description: str
    language: str
    code_type: CodeType
    quality_level: CodeQuality = CodeQuality.PRODUCTION
    constraints: List[str] = None
    dependencies: List[str] = None
    examples: List[str] = None
    context: str = ""
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.dependencies is None:
            self.dependencies = []
        if self.examples is None:
            self.examples = []


@dataclass
class GeneratedCode:
    """Container for generated code with metadata."""
    code: str
    language: str
    code_type: CodeType
    documentation: str
    examples: List[str]
    tests: Optional[str] = None
    dependencies: List[str] = None
    explanation: str = ""
    quality_score: float = 0.0
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class CodeGenerationResult:
    """Result of code generation process."""
    requirement: CodeRequirement
    generated_code: GeneratedCode
    generation_time_ms: float
    model_used: str
    confidence: float
    alternative_solutions: List[GeneratedCode] = None
    
    def __post_init__(self):
        if self.alternative_solutions is None:
            self.alternative_solutions = []


class CodeGeneratorAgent:
    """AI agent that generates code from natural language requirements."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.prompt_manager = get_prompt_manager()
        self.model_manager = get_model_manager()
        self.model_name = model_name
    
    async def generate_code(self, requirement: CodeRequirement) -> CodeGenerationResult:
        """Generate code from a requirement specification."""
        import time
        start_time = time.time()
        
        try:
            # Generate the main code solution
            generated_code = await self._generate_code_with_ai(requirement)
            
            # Calculate confidence based on requirement completeness and code quality
            confidence = self._calculate_confidence(requirement, generated_code)
            
            # Optionally generate alternative solutions for complex requirements
            alternatives = []
            if requirement.quality_level == CodeQuality.ENTERPRISE:
                alternatives = await self._generate_alternatives(requirement, max_alternatives=2)
            
            generation_time = (time.time() - start_time) * 1000
            
            return CodeGenerationResult(
                requirement=requirement,
                generated_code=generated_code,
                generation_time_ms=generation_time,
                model_used=self.model_name or "default",
                confidence=confidence,
                alternative_solutions=alternatives
            )
            
        except Exception as e:
            print(f"Error in code generation: {e}")
            # Return fallback result
            return self._create_fallback_result(requirement, str(e))
    
    async def generate_multiple_codes(self, requirements: List[CodeRequirement]) -> List[CodeGenerationResult]:
        """Generate code for multiple requirements concurrently."""
        semaphore = asyncio.Semaphore(3)  # Limit concurrent generations
        
        async def generate_with_semaphore(req: CodeRequirement) -> CodeGenerationResult:
            async with semaphore:
                return await self.generate_code(req)
        
        tasks = [generate_with_semaphore(req) for req in requirements]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, CodeGenerationResult):
                successful_results.append(result)
            elif isinstance(result, Exception):
                print(f"Code generation error: {result}")
        
        return successful_results
    
    async def _generate_code_with_ai(self, requirement: CodeRequirement) -> GeneratedCode:
        """Generate code using AI model."""
        # Prepare context and constraints
        context = self._build_generation_context(requirement)
        constraints = self._format_constraints(requirement)
        
        # Generate prompt
        system_prompt, user_prompt = self.prompt_manager.generate_code_generation_prompt(
            requirements=requirement.description,
            language=requirement.language,
            context=context,
            constraints=constraints
        )
        
        # Get AI response
        response = await self.model_manager.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model_name=self.model_name
        )
        
        # Parse response into structured code
        return self._parse_code_response(response.content, requirement)
    
    def _build_generation_context(self, requirement: CodeRequirement) -> str:
        """Build context string for code generation."""
        context_parts = []
        
        if requirement.context:
            context_parts.append(f"Context: {requirement.context}")
        
        if requirement.code_type:
            context_parts.append(f"Code type: {requirement.code_type.value}")
        
        if requirement.quality_level:
            context_parts.append(f"Quality level: {requirement.quality_level.value}")
        
        if requirement.dependencies:
            deps = ", ".join(requirement.dependencies)
            context_parts.append(f"Dependencies: {deps}")
        
        if requirement.examples:
            examples = "\n".join(requirement.examples)
            context_parts.append(f"Examples:\n{examples}")
        
        return "\n".join(context_parts)
    
    def _format_constraints(self, requirement: CodeRequirement) -> str:
        """Format constraints for the prompt."""
        if not requirement.constraints:
            return "No specific constraints."
        
        return "\n".join([f"- {constraint}" for constraint in requirement.constraints])
    
    def _parse_code_response(self, response_content: str, requirement: CodeRequirement) -> GeneratedCode:
        """Parse AI response into GeneratedCode object."""
        # Initialize default values
        code = ""
        documentation = ""
        examples = []
        tests = None
        dependencies = []
        explanation = ""
        
        # Split response into sections
        sections = self._split_response_sections(response_content)
        
        # Extract main code
        code = sections.get('code', sections.get('implementation', ''))
        
        # Clean up code (remove markdown formatting)
        code = self._clean_code_block(code, requirement.language)
        
        # Extract other sections
        documentation = sections.get('documentation', sections.get('docstring', ''))
        explanation = sections.get('explanation', sections.get('description', ''))
        
        # Extract examples
        if 'examples' in sections:
            examples = self._extract_examples(sections['examples'])
        elif 'usage' in sections:
            examples = self._extract_examples(sections['usage'])
        
        # Extract tests
        tests = sections.get('tests', sections.get('test', None))
        if tests:
            tests = self._clean_code_block(tests, requirement.language)
        
        # Extract dependencies
        deps_text = sections.get('dependencies', sections.get('imports', ''))
        if deps_text:
            dependencies = self._extract_dependencies(deps_text)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(code, documentation, tests)
        
        return GeneratedCode(
            code=code,
            language=requirement.language,
            code_type=requirement.code_type,
            documentation=documentation,
            examples=examples,
            tests=tests,
            dependencies=dependencies,
            explanation=explanation,
            quality_score=quality_score
        )
    
    def _split_response_sections(self, response: str) -> Dict[str, str]:
        """Split response into logical sections."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            # Check for section headers
            line_lower = line.lower().strip()
            
            if any(header in line_lower for header in ['## ', '### ', '# ']):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                if 'code' in line_lower or 'implementation' in line_lower:
                    current_section = 'code'
                elif 'documentation' in line_lower or 'docstring' in line_lower:
                    current_section = 'documentation'
                elif 'example' in line_lower or 'usage' in line_lower:
                    current_section = 'examples'
                elif 'test' in line_lower:
                    current_section = 'tests'
                elif 'dependencies' in line_lower or 'import' in line_lower:
                    current_section = 'dependencies'
                elif 'explanation' in line_lower or 'description' in line_lower:
                    current_section = 'explanation'
                else:
                    current_section = 'general'
                
                current_content = []
            else:
                current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If no clear sections found, treat as code
        if not sections and response.strip():
            sections['code'] = response.strip()
        
        return sections
    
    def _clean_code_block(self, code_text: str, language: str) -> str:
        """Clean code block by removing markdown formatting."""
        if not code_text:
            return ""
        
        # Remove markdown code block markers
        code_text = re.sub(r'```\w*\n?', '', code_text)
        code_text = re.sub(r'```', '', code_text)
        
        # Remove excessive whitespace
        lines = code_text.split('\n')
        
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)
    
    def _extract_examples(self, examples_text: str) -> List[str]:
        """Extract examples from text."""
        examples = []
        
        # Split by common separators
        potential_examples = re.split(r'\n\s*\n|Example \d+|Usage \d+', examples_text)
        
        for example in potential_examples:
            example = example.strip()
            if example and len(example) > 10:  # Filter out very short snippets
                # Clean up example
                example = self._clean_code_block(example, 'python')  # Assume python for examples
                if example:
                    examples.append(example)
        
        return examples
    
    def _extract_dependencies(self, deps_text: str) -> List[str]:
        """Extract dependencies from text."""
        dependencies = []
        
        # Look for import statements or dependency lists
        lines = deps_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Python imports
            if line.startswith(('import ', 'from ')):
                dependencies.append(line)
            
            # Package names (simple format)
            elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', line):
                dependencies.append(line)
            
            # Requirements.txt format
            elif '==' in line or '>=' in line:
                dependencies.append(line)
        
        return dependencies
    
    def _calculate_quality_score(self, code: str, documentation: str, tests: Optional[str]) -> float:
        """Calculate quality score for generated code."""
        score = 0.0
        
        # Code presence and length
        if code:
            score += 3.0
            if len(code) > 100:
                score += 1.0
        
        # Documentation presence
        if documentation:
            score += 2.0
            if len(documentation) > 50:
                score += 1.0
        
        # Tests presence
        if tests:
            score += 2.0
        
        # Code quality indicators
        if code:
            # Check for docstrings
            if '"""' in code or "'''" in code:
                score += 0.5
            
            # Check for type hints (Python)
            if '->' in code or ': int' in code or ': str' in code:
                score += 0.5
            
            # Check for error handling
            if 'try:' in code or 'except' in code or 'raise' in code:
                score += 0.5
            
            # Check for comments
            if '#' in code:
                score += 0.5
        
        return min(10.0, score)  # Cap at 10.0
    
    def _calculate_confidence(self, requirement: CodeRequirement, generated_code: GeneratedCode) -> float:
        """Calculate confidence in the generated code."""
        confidence = 0.7  # Base confidence
        
        # Requirement completeness
        if requirement.description and len(requirement.description) > 20:
            confidence += 0.1
        
        if requirement.examples:
            confidence += 0.05
        
        if requirement.constraints:
            confidence += 0.05
        
        # Generated code quality
        if generated_code.code and len(generated_code.code) > 50:
            confidence += 0.1
        
        if generated_code.documentation:
            confidence += 0.05
        
        if generated_code.tests:
            confidence += 0.05
        
        if generated_code.quality_score > 7.0:
            confidence += 0.1
        elif generated_code.quality_score > 5.0:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    async def _generate_alternatives(self, requirement: CodeRequirement, max_alternatives: int = 2) -> List[GeneratedCode]:
        """Generate alternative solutions."""
        alternatives = []
        
        for i in range(max_alternatives):
            try:
                # Modify the requirement slightly for variation
                alt_requirement = CodeRequirement(
                    description=f"{requirement.description} (Alternative approach {i+1})",
                    language=requirement.language,
                    code_type=requirement.code_type,
                    quality_level=requirement.quality_level,
                    constraints=requirement.constraints + [f"Use a different approach than the main solution"],
                    dependencies=requirement.dependencies,
                    examples=requirement.examples,
                    context=requirement.context
                )
                
                alt_code = await self._generate_code_with_ai(alt_requirement)
                alternatives.append(alt_code)
                
            except Exception as e:
                print(f"Error generating alternative {i+1}: {e}")
                break
        
        return alternatives
    
    def _create_fallback_result(self, requirement: CodeRequirement, error_msg: str) -> CodeGenerationResult:
        """Create a fallback result when generation fails."""
        fallback_code = GeneratedCode(
            code=f"# Error occurred during code generation: {error_msg}\n# Please refine your requirements and try again.",
            language=requirement.language,
            code_type=requirement.code_type,
            documentation="Code generation failed. See error in code comments.",
            examples=[],
            quality_score=0.0
        )
        
        return CodeGenerationResult(
            requirement=requirement,
            generated_code=fallback_code,
            generation_time_ms=0.0,
            model_used=self.model_name or "fallback",
            confidence=0.0
        )
    
    def generate_code_report(self, results: List[CodeGenerationResult]) -> str:
        """Generate a comprehensive code generation report."""
        if not results:
            return "No code generation results to report."
        
        successful_results = [r for r in results if r.confidence > 0.0]
        total_generated = len(successful_results)
        avg_confidence = sum(r.confidence for r in successful_results) / total_generated if successful_results else 0
        avg_quality = sum(r.generated_code.quality_score for r in successful_results) / total_generated if successful_results else 0
        
        report = f"""# Code Generation Report

## Summary
- **Total Requests:** {len(results)}
- **Successful Generations:** {total_generated}
- **Success Rate:** {(total_generated/len(results)*100):.1f}%
- **Average Confidence:** {avg_confidence:.1%}
- **Average Quality Score:** {avg_quality:.1f}/10
- **Total Generation Time:** {sum(r.generation_time_ms for r in results):.0f}ms

## Generated Code

"""
        
        for i, result in enumerate(successful_results, 1):
            req = result.requirement
            code = result.generated_code
            
            report += f"""### {i}. {req.code_type.value.title()} in {req.language.title()}

**Description:** {req.description}

**Quality Score:** {code.quality_score:.1f}/10
**Confidence:** {result.confidence:.1%}
**Generation Time:** {result.generation_time_ms:.0f}ms

"""
            
            if code.explanation:
                report += f"**Explanation:** {code.explanation}\n\n"
            
            # Show code
            report += f"**Generated Code:**\n```{code.language}\n{code.code}\n```\n\n"
            
            # Show documentation if available
            if code.documentation:
                report += f"**Documentation:** {code.documentation}\n\n"
            
            # Show examples if available
            if code.examples:
                report += "**Examples:**\n"
                for j, example in enumerate(code.examples, 1):
                    report += f"```{code.language}\n{example}\n```\n"
                report += "\n"
            
            # Show tests if available
            if code.tests:
                report += f"**Tests:**\n```{code.language}\n{code.tests}\n```\n\n"
            
            # Show dependencies if any
            if code.dependencies:
                report += "**Dependencies:**\n"
                for dep in code.dependencies:
                    report += f"- {dep}\n"
                report += "\n"
            
            # Show alternatives if any
            if result.alternative_solutions:
                report += f"**Alternative Solutions:** {len(result.alternative_solutions)} available\n\n"
            
            report += "---\n\n"
        
        return report
    
    def save_generated_code(self, result: CodeGenerationResult, output_dir: str) -> str:
        """Save generated code to a file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        safe_desc = re.sub(r'[^\w\s-]', '', result.requirement.description)[:50]
        safe_desc = re.sub(r'[-\s]+', '_', safe_desc)
        
        ext_map = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'java': '.java',
            'cpp': '.cpp',
            'c': '.c',
            'go': '.go',
            'rust': '.rs'
        }
        
        extension = ext_map.get(result.requirement.language, '.txt')
        filename = f"{safe_desc}_{result.requirement.code_type.value}{extension}"
        file_path = output_path / filename
        
        # Prepare content
        content = []
        
        # Add header comment
        content.append(f"# Generated by AI Code Agent")
        content.append(f"# Description: {result.requirement.description}")
        content.append(f"# Language: {result.requirement.language}")
        content.append(f"# Type: {result.requirement.code_type.value}")
        content.append(f"# Quality Score: {result.generated_code.quality_score:.1f}/10")
        content.append(f"# Confidence: {result.confidence:.1%}")
        content.append("")
        
        # Add dependencies if any
        if result.generated_code.dependencies:
            content.append("# Dependencies:")
            for dep in result.generated_code.dependencies:
                content.append(f"# {dep}")
            content.append("")
        
        # Add main code
        content.append(result.generated_code.code)
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        return str(file_path)