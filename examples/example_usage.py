"""
Example usage of the AI Code Agent system.

This demonstrates how to use the different agents and workflows.
"""

import asyncio
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.agent_coordinator import AgentCoordinator, WorkflowType
from agents.code_generator import CodeRequirement, CodeType, CodeQuality


async def example_comprehensive_analysis():
    """Example: Comprehensive analysis of a code file."""
    print("🔍 Example 1: Comprehensive Code Analysis")
    print("=" * 50)
    
    coordinator = AgentCoordinator()
    
    # Create a sample Python file for analysis
    sample_code = '''
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def process_data(data):
    results = []
    for item in data:
        if item > 0:
            results.append(item * 2)
    return results

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_data(self, value):
        self.data.append(value)
    
    def get_average(self):
        return sum(self.data) / len(self.data)
'''
    
    # Write sample file
    sample_file = Path("sample_code.py")
    sample_file.write_text(sample_code)
    
    try:
        # Run comprehensive analysis
        result = await coordinator.execute_workflow(
            WorkflowType.COMPREHENSIVE_ANALYSIS, 
            [str(sample_file)]
        )
        
        print(f"✅ Analysis completed!")
        print(f"📊 Success rate: {result.success_rate:.1%}")
        print(f"⏱️  Execution time: {result.execution_time_ms:.0f}ms")
        print(f"📝 Summary: {result.summary}")
        
        print("\n🎯 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Generate detailed report
        report = coordinator.generate_workflow_report(result)
        report_file = Path("analysis_report.md")
        report_file.write_text(report)
        print(f"\n📄 Detailed report saved to: {report_file}")
        
    finally:
        # Cleanup
        if sample_file.exists():
            sample_file.unlink()


async def example_code_generation():
    """Example: Generate code from requirements."""
    print("\n🔨 Example 2: Code Generation")
    print("=" * 50)
    
    coordinator = AgentCoordinator()
    
    # Define requirements
    requirement = CodeRequirement(
        description="Create a function that validates email addresses using regex and returns True if valid, False otherwise. Include proper error handling.",
        language="python",
        code_type=CodeType.FUNCTION,
        quality_level=CodeQuality.PRODUCTION,
        constraints=[
            "Use standard library only",
            "Include comprehensive docstring",
            "Add input validation"
        ]
    )
    
    # Generate code
    result = await coordinator.code_generator.generate_code(requirement)
    
    print(f"✅ Code generation completed!")
    print(f"🎯 Confidence: {result.confidence:.1%}")
    print(f"⭐ Quality score: {result.generated_code.quality_score:.1f}/10")
    print(f"⏱️  Generation time: {result.generation_time_ms:.0f}ms")
    
    print("\n📄 Generated Code:")
    print("-" * 40)
    print(result.generated_code.code)
    
    if result.generated_code.documentation:
        print("\n📚 Documentation:")
        print("-" * 40)
        print(result.generated_code.documentation)
    
    # Save generated code
    saved_path = coordinator.code_generator.save_generated_code(result, ".")
    print(f"\n💾 Code saved to: {saved_path}")


async def example_bug_detection():
    """Example: Bug detection in code."""
    print("\n🐛 Example 3: Bug Detection")
    print("=" * 50)
    
    coordinator = AgentCoordinator()
    
    # Create problematic code
    buggy_code = '''
def divide_numbers(a, b):
    return a / b  # No zero division check

def get_user_data(user_id):
    users = {1: "Alice", 2: "Bob"}
    return users[user_id]  # No key error handling

def process_list(items):
    for i in range(len(items) + 1):  # Off-by-one error
        print(items[i])

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    
    def withdraw(self, amount):
        self.balance -= amount  # No balance check
        return self.balance
'''
    
    # Write buggy file
    buggy_file = Path("buggy_code.py")
    buggy_file.write_text(buggy_code)
    
    try:
        # Detect bugs
        result = await coordinator.bug_detector.detect_bugs(str(buggy_file))
        
        print(f"✅ Bug detection completed!")
        print(f"🎯 Confidence: {result.confidence_score:.1%}")
        print(f"🐛 Bugs found: {len(result.bugs_detected)}")
        print(f"⏱️  Analysis time: {result.analysis_time_ms:.0f}ms")
        
        print("\n🚨 Detected Issues:")
        for i, bug in enumerate(result.bugs_detected, 1):
            print(f"  {i}. Line {bug.line_number}: {bug.title}")
            print(f"     Severity: {bug.severity.value}")
            print(f"     Description: {bug.description}")
            if bug.suggested_fix:
                print(f"     Fix: {bug.suggested_fix}")
            print()
        
        # Generate bug report
        report = coordinator.bug_detector.generate_bug_report([result])
        report_file = Path("bug_report.md")
        report_file.write_text(report)
        print(f"📄 Bug report saved to: {report_file}")
        
    finally:
        # Cleanup
        if buggy_file.exists():
            buggy_file.unlink()


async def example_test_generation():
    """Example: Generate tests for existing code."""
    print("\n🧪 Example 4: Test Generation")
    print("=" * 50)
    
    coordinator = AgentCoordinator()
    
    # Create code to test
    test_target_code = '''
def add_numbers(a, b):
    """Add two numbers and return the result."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b

def factorial(n):
    """Calculate factorial of a number."""
    if n < 0:
        raise ValueError("Cannot calculate factorial of negative number")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def calculate(self, operation, a, b):
        """Perform calculation and store in history."""
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        self.history.append((operation, a, b, result))
        return result
'''
    
    # Write target file
    target_file = Path("calculator.py")
    target_file.write_text(test_target_code)
    
    try:
        # Generate tests
        result = await coordinator.test_generator.generate_tests(str(target_file))
        
        print(f"✅ Test generation completed!")
        print(f"🎯 Confidence: {result.confidence:.1%}")
        print(f"🧪 Tests generated: {len(result.test_suite.test_cases)}")
        print(f"📊 Coverage estimate: {result.test_suite.coverage_estimate:.1%}")
        print(f"⏱️  Generation time: {result.generation_time_ms:.0f}ms")
        
        print("\n🧪 Generated Test Cases:")
        for i, test_case in enumerate(result.test_suite.test_cases, 1):
            print(f"  {i}. {test_case.name}")
            print(f"     Type: {test_case.test_type.value}")
            print(f"     Description: {test_case.description}")
            if test_case.covers_edge_case:
                print(f"     ⚠️  Covers edge case")
            print()
        
        # Generate test file
        test_file_path = coordinator.test_generator.generate_test_file(result, ".")
        print(f"💾 Test file saved to: {test_file_path}")
        
    finally:
        # Cleanup
        if target_file.exists():
            target_file.unlink()


async def example_full_workflow():
    """Example: Complete development workflow."""
    print("\n🚀 Example 5: Full Development Workflow")
    print("=" * 50)
    
    coordinator = AgentCoordinator()
    
    # Define development requirements
    requirements = [
        CodeRequirement(
            description="Create a simple REST API endpoint for user registration that validates email and password strength",
            language="python",
            code_type=CodeType.API_ENDPOINT,
            quality_level=CodeQuality.PRODUCTION
        )
    ]
    
    # Execute full development workflow
    result = await coordinator.develop_from_requirements(requirements)
    
    print(f"✅ Full development workflow completed!")
    print(f"📊 Success rate: {result.success_rate:.1%}")
    print(f"⏱️  Total execution time: {result.execution_time_ms:.0f}ms")
    print(f"📝 Summary: {result.summary}")
    
    print("\n🎯 Final Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Generate comprehensive report
    report = coordinator.generate_workflow_report(result)
    report_file = Path("full_workflow_report.md")
    report_file.write_text(report)
    print(f"\n📄 Complete workflow report saved to: {report_file}")


async def main():
    """Run all examples."""
    print("🤖 AI Code Agent - Example Usage")
    print("=" * 60)
    print("This demonstration shows the capabilities of the multi-agent system.")
    print()
    
    try:
        await example_comprehensive_analysis()
        await example_code_generation()
        await example_bug_detection()
        await example_test_generation()
        await example_full_workflow()
        
        print("\n🎉 All examples completed successfully!")
        print("\nℹ️  Note: This demo uses mock AI responses since no API keys are configured.")
        print("   To use with real AI models, set your OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("   environment variables.")
        
    except Exception as e:
        print(f"\n❌ Error during examples: {e}")
        print("This is expected if you haven't configured AI model API keys.")


if __name__ == "__main__":
    asyncio.run(main())