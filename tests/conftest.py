"""
Pytest configuration and fixtures for AI Code Agent tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def calculate_score(items, weights=None):
    """Calculate weighted score for items."""
    if not items:
        return 0.0
    
    if weights is None:
        weights = [1.0] * len(items)
    
    if len(items) != len(weights):
        raise ValueError("Items and weights must have same length")
    
    total_score = 0.0
    total_weight = 0.0
    
    for item, weight in zip(items, weights):
        if weight < 0:
            raise ValueError("Weights must be non-negative")
        total_score += item * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0.0


class DataProcessor:
    """Process and analyze data."""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.processed_count = 0
    
    def process(self, data):
        """Process a batch of data."""
        results = []
        for item in data:
            if item > self.threshold:
                results.append(item * 2)
            else:
                results.append(item)
        
        self.processed_count += len(data)
        return results
    
    def get_stats(self):
        """Get processing statistics."""
        return {
            "processed_count": self.processed_count,
            "threshold": self.threshold
        }
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return '''
function validateEmail(email) {
    // Simple email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function processUserData(userData) {
    const errors = [];
    
    if (!userData.name || userData.name.length < 2) {
        errors.push("Name must be at least 2 characters");
    }
    
    if (!validateEmail(userData.email)) {
        errors.push("Invalid email format");
    }
    
    if (userData.age && (userData.age < 13 || userData.age > 120)) {
        errors.push("Age must be between 13 and 120");
    }
    
    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

class UserManager {
    constructor() {
        this.users = new Map();
        this.nextId = 1;
    }
    
    addUser(userData) {
        const validation = processUserData(userData);
        if (!validation.isValid) {
            throw new Error(validation.errors.join(", "));
        }
        
        const user = {
            id: this.nextId++,
            ...userData,
            createdAt: new Date()
        };
        
        this.users.set(user.id, user);
        return user;
    }
    
    getUser(id) {
        return this.users.get(id);
    }
    
    getUserCount() {
        return this.users.size;
    }
}
'''


@pytest.fixture
def sample_complex_code():
    """Complex code with potential issues for testing."""
    return '''
import os
import requests

# Hardcoded credentials (security issue)
API_KEY = "sk-1234567890abcdef1234567890abcdef"
DATABASE_URL = "postgresql://user:password123@localhost/db"

def get_user_data(user_id):
    """Get user data from database (SQL injection vulnerability)."""
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = '%s'" % user_id
    return execute_query(query)

def complex_calculation(a, b, c, d, e, f, g, h, i, j):
    """Overly complex function with high cyclomatic complexity."""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                if h > 0:
                                    if i > 0:
                                        if j > 0:
                                            return a + b + c + d + e + f + g + h + i + j
                                        else:
                                            return a + b + c + d + e + f + g + h + i
                                    else:
                                        return a + b + c + d + e + f + g + h
                                else:
                                    return a + b + c + d + e + f + g
                            else:
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

# Very long function (maintainability issue)
def process_large_dataset(data):
    """Process a large dataset with many operations."""
    # This function is intentionally long and complex
    results = []
    temp_storage = {}
    counters = {"processed": 0, "errors": 0, "skipped": 0}
    
    for item in data:
        try:
            # Multiple nested operations
            if item.get("type") == "A":
                if item.get("status") == "active":
                    processed_item = {"id": item["id"], "value": item["value"] * 2}
                    if processed_item["value"] > 100:
                        processed_item["category"] = "high"
                    else:
                        processed_item["category"] = "normal"
                    results.append(processed_item)
                    counters["processed"] += 1
                else:
                    counters["skipped"] += 1
            elif item.get("type") == "B":
                if item.get("priority") == "urgent":
                    urgent_item = {"id": item["id"], "urgent_value": item["value"] * 3}
                    temp_storage[item["id"]] = urgent_item
                    counters["processed"] += 1
                else:
                    normal_item = {"id": item["id"], "normal_value": item["value"]}
                    results.append(normal_item)
                    counters["processed"] += 1
            else:
                # Handle unknown types
                unknown_item = {"id": item.get("id", "unknown"), "raw_data": item}
                results.append(unknown_item)
                counters["processed"] += 1
        except Exception as e:
            counters["errors"] += 1
            continue
    
    # More processing logic...
    final_results = []
    for result in results:
        if result.get("category") == "high":
            result["bonus"] = 50
        final_results.append(result)
    
    return final_results, counters, temp_storage

# Code duplication example
def calculate_circle_area(radius):
    """Calculate area of a circle."""
    pi = 3.14159
    return pi * radius * radius

def calculate_circle_circumference(radius):
    """Calculate circumference of a circle."""
    pi = 3.14159  # Duplicated constant
    return 2 * pi * radius
'''


@pytest.fixture
def sample_code_metrics():
    """Sample code metrics for testing."""
    return {
        "cyclomatic_complexity": 5.2,
        "halstead_difficulty": 8.5,
        "maintainability_index": 72.3,
        "lines_of_code": 150,
        "blank_lines": 15,
        "comment_lines": 25,
        "code_duplication": 3.5,
        "test_coverage": 78.5,
        "number_of_methods": 8,
        "number_of_classes": 2,
        "number_of_functions": 12,
        "number_of_loops": 4,
        "number_of_conditions": 6,
        "commits_last_month": 15,
        "authors_count": 3,
        "file_age_days": 45,
        "security_warnings": 2,
        "linting_violations": 7,
        "type_errors": 1,
        "nesting_depth": 3,
        "halstead_volume": 1250.0,
        "number_of_variables": 18,
        "number_of_arrays": 2,
        "number_of_objects": 4,
        "file_operations": 1,
        "network_operations": 2,
        "database_operations": 3
    }


@pytest.fixture(autouse=True)
def reset_global_instances():
    """Reset global instances before each test."""
    # Skip imports that may not exist or have dependency issues
    try:
        import src.performance.cache_manager
        src.performance.cache_manager._cache_manager = None
    except (ImportError, AttributeError):
        pass
    
    yield
    
    # Clean up after test
    try:
        import src.performance.cache_manager
        src.performance.cache_manager._cache_manager = None
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    from src.llm.model_interface import ModelResponse
    
    return ModelResponse(
        content="This is a mock response from the LLM model.",
        finish_reason="stop",
        tokens_used=25,
        model="mock-model-v1"
    )