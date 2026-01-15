"""
Tests for the insights engine and analytics system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analytics.insights_engine import (
    InsightsEngine, Insight, InsightType, InsightSeverity, CodeMetric,
    CodeQualityAnalyzer, PerformanceAnalyzer, SecurityAnalyzer, 
    TechnicalDebtAnalyzer, get_insights_engine
)


class TestInsight:
    """Test cases for Insight dataclass."""
    
    def test_insight_creation(self):
        """Test creating an insight."""
        insight = Insight(
            id="test_insight_1",
            type=InsightType.CODE_QUALITY_TREND,
            severity=InsightSeverity.MEDIUM,
            title="Test Insight",
            description="This is a test insight",
            evidence=["Evidence 1", "Evidence 2"],
            recommendations=["Fix this", "Improve that"],
            confidence_score=0.85
        )
        
        assert insight.id == "test_insight_1"
        assert insight.type == InsightType.CODE_QUALITY_TREND
        assert insight.severity == InsightSeverity.MEDIUM
        assert insight.title == "Test Insight"
        assert len(insight.evidence) == 2
        assert len(insight.recommendations) == 2
        assert insight.confidence_score == 0.85
        assert isinstance(insight.timestamp, datetime)
    
    def test_insight_to_dict(self):
        """Test converting insight to dictionary."""
        insight = Insight(
            id="test_insight_2",
            type=InsightType.SECURITY_RISK,
            severity=InsightSeverity.HIGH,
            title="Security Issue",
            description="Potential security vulnerability",
            evidence=["SQL injection pattern found"],
            recommendations=["Use parameterized queries"],
            confidence_score=0.92
        )
        
        result = insight.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == "test_insight_2"
        assert result["type"] == "security_risk"
        assert result["severity"] == "high"
        assert result["title"] == "Security Issue"
        assert result["confidence_score"] == 0.92
        assert "timestamp" in result


class TestCodeMetric:
    """Test cases for CodeMetric dataclass."""
    
    def test_code_metric_creation(self):
        """Test creating a code metric."""
        metric = CodeMetric(
            timestamp=datetime.now(),
            file_path="/path/to/file.py",
            metric_type="complexity",
            value=8.5,
            metadata={"function": "calculate_score"}
        )
        
        assert metric.file_path == "/path/to/file.py"
        assert metric.metric_type == "complexity"
        assert metric.value == 8.5
        assert metric.metadata["function"] == "calculate_score"
        assert isinstance(metric.timestamp, datetime)


class TestCodeQualityAnalyzer:
    """Test cases for code quality analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CodeQualityAnalyzer()
    
    def test_initial_state(self):
        """Test initial analyzer state."""
        assert len(self.analyzer.metrics_history) == 0
        assert "complexity" in self.analyzer.quality_thresholds
        assert "maintainability" in self.analyzer.quality_thresholds
    
    def test_add_metric(self):
        """Test adding metrics to history."""
        metric = CodeMetric(
            timestamp=datetime.now(),
            file_path="test.py",
            metric_type="complexity",
            value=5.0
        )
        
        self.analyzer.add_metric(metric)
        assert len(self.analyzer.metrics_history) == 1
    
    def test_metrics_cleanup(self):
        """Test that old metrics are cleaned up."""
        # Add old metrics (over 30 days old)
        old_date = datetime.now() - timedelta(days=35)
        old_metric = CodeMetric(
            timestamp=old_date,
            file_path="old.py",
            metric_type="complexity",
            value=10.0
        )
        
        # Add recent metric
        recent_metric = CodeMetric(
            timestamp=datetime.now(),
            file_path="recent.py",
            metric_type="complexity",
            value=5.0
        )
        
        self.analyzer.add_metric(old_metric)
        self.analyzer.add_metric(recent_metric)
        
        # Only recent metric should remain
        assert len(self.analyzer.metrics_history) == 1
        assert self.analyzer.metrics_history[0].file_path == "recent.py"
    
    def test_analyze_quality_trends_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        # Add only a few metrics
        for i in range(3):
            metric = CodeMetric(
                timestamp=datetime.now() - timedelta(days=i),
                file_path="test.py",
                metric_type="complexity",
                value=5.0 + i
            )
            self.analyzer.add_metric(metric)
        
        insights = self.analyzer.analyze_quality_trends()
        assert len(insights) == 0  # Need at least 10 metrics and 5 per file
    
    def test_analyze_quality_trends_increasing_complexity(self):
        """Test detection of increasing complexity trend."""
        # Add metrics showing increasing complexity
        base_time = datetime.now() - timedelta(days=10)
        for i in range(10):
            metric = CodeMetric(
                timestamp=base_time + timedelta(days=i),
                file_path="trending.py",
                metric_type="complexity",
                value=5.0 + i * 0.5  # Gradually increasing
            )
            self.analyzer.add_metric(metric)
        
        insights = self.analyzer.analyze_quality_trends()
        
        # Should detect increasing complexity
        complexity_insights = [
            i for i in insights 
            if i.type == InsightType.CODE_QUALITY_TREND and "complexity" in i.title.lower()
        ]
        assert len(complexity_insights) > 0
    
    def test_analyze_quality_trends_declining_maintainability(self):
        """Test detection of declining maintainability."""
        base_time = datetime.now() - timedelta(days=10)
        for i in range(10):
            metric = CodeMetric(
                timestamp=base_time + timedelta(days=i),
                file_path="declining.py",
                metric_type="maintainability",
                value=90.0 - i * 2.0  # Declining maintainability
            )
            self.analyzer.add_metric(metric)
        
        insights = self.analyzer.analyze_quality_trends()
        
        # Should detect declining maintainability
        maintainability_insights = [
            i for i in insights 
            if i.type == InsightType.MAINTENANCE_BURDEN
        ]
        assert len(maintainability_insights) > 0
    
    def test_calculate_trend(self):
        """Test trend calculation method."""
        # Test increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        trend = self.analyzer._calculate_trend(increasing_values)
        assert trend > 0
        
        # Test decreasing trend
        decreasing_values = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        trend = self.analyzer._calculate_trend(decreasing_values)
        assert trend < 0
        
        # Test stable trend
        stable_values = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        trend = self.analyzer._calculate_trend(stable_values)
        assert abs(trend) < 1.0  # Should be close to 0


class TestPerformanceAnalyzer:
    """Test cases for performance analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PerformanceAnalyzer()
    
    def test_initial_state(self):
        """Test initial analyzer state."""
        assert len(self.analyzer.performance_data) == 0
    
    def test_add_performance_data(self):
        """Test adding performance data."""
        data = {
            "response_time": 2.5,
            "memory_mb": 128.0,
            "operation": "code_analysis"
        }
        
        self.analyzer.add_performance_data(data)
        assert len(self.analyzer.performance_data) == 1
        assert "timestamp" in self.analyzer.performance_data[0]
    
    def test_data_cleanup(self):
        """Test that old performance data is cleaned up."""
        # Add old data
        old_data = {"response_time": 1.0}
        self.analyzer.performance_data.append({
            **old_data,
            "timestamp": datetime.now() - timedelta(days=8)
        })
        
        # Add recent data
        recent_data = {"response_time": 2.0}
        self.analyzer.add_performance_data(recent_data)
        
        # Only recent data should remain
        assert len(self.analyzer.performance_data) == 1
        assert self.analyzer.performance_data[0]["response_time"] == 2.0
    
    def test_analyze_performance_patterns_insufficient_data(self):
        """Test analysis with insufficient data."""
        # Add only a few data points
        for i in range(10):
            self.analyzer.add_performance_data({"response_time": 1.0 + i})
        
        insights = self.analyzer.analyze_performance_patterns()
        assert len(insights) == 0  # Need at least 20 data points
    
    def test_analyze_high_response_time(self):
        """Test detection of high response times."""
        # Add data with high response times
        for i in range(25):
            self.analyzer.add_performance_data({"response_time": 6.0 + i * 0.1})
        
        insights = self.analyzer.analyze_performance_patterns()
        
        # Should detect high response time
        response_time_insights = [
            i for i in insights 
            if "response time" in i.title.lower()
        ]
        assert len(response_time_insights) > 0
        assert any(i.severity == InsightSeverity.HIGH for i in response_time_insights)
    
    def test_analyze_memory_leak_detection(self):
        """Test detection of potential memory leaks."""
        # Add data showing increasing memory usage
        for i in range(25):
            self.analyzer.add_performance_data({
                "response_time": 2.0,
                "memory_mb": 100.0 + i * 10.0  # Steadily increasing memory
            })
        
        insights = self.analyzer.analyze_performance_patterns()
        
        # Should detect potential memory leak
        memory_insights = [
            i for i in insights 
            if "memory leak" in i.title.lower()
        ]
        assert len(memory_insights) > 0
        assert any(i.severity == InsightSeverity.CRITICAL for i in memory_insights)
    
    def test_calculate_memory_trend(self):
        """Test memory trend calculation."""
        # Test increasing memory usage
        increasing_memory = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
        trend = self.analyzer._calculate_memory_trend(increasing_memory)
        assert trend > 0
        
        # Test stable memory usage
        stable_memory = [100.0] * 10
        trend = self.analyzer._calculate_memory_trend(stable_memory)
        assert abs(trend) < 5.0


class TestSecurityAnalyzer:
    """Test cases for security analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SecurityAnalyzer()
    
    def test_initial_state(self):
        """Test initial analyzer state."""
        assert len(self.analyzer.security_events) == 0
        assert "sql_injection" in self.analyzer.vulnerability_patterns
        assert "xss" in self.analyzer.vulnerability_patterns
        assert "hardcoded_secrets" in self.analyzer.vulnerability_patterns
    
    def test_add_security_event(self):
        """Test adding security events."""
        event = {
            "type": "suspicious_login",
            "user_id": "test_user",
            "details": {"ip": "192.168.1.1"}
        }
        
        self.analyzer.add_security_event(event)
        assert len(self.analyzer.security_events) == 1
        assert "timestamp" in self.analyzer.security_events[0]
    
    def test_analyze_sql_injection_patterns(self):
        """Test detection of SQL injection patterns."""
        code_with_sql_injection = '''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = '%s'" % user_id
    return execute(query)
'''
        
        insights = self.analyzer.analyze_security_patterns(
            code_with_sql_injection, "vulnerable.py"
        )
        
        # Should detect SQL injection vulnerability
        sql_insights = [
            i for i in insights 
            if "sql injection" in i.title.lower()
        ]
        assert len(sql_insights) > 0
        assert any(i.severity == InsightSeverity.HIGH for i in sql_insights)
    
    def test_analyze_xss_patterns(self):
        """Test detection of XSS patterns."""
        code_with_xss = '''
function updateContent(userInput) {
    document.getElementById("content").innerHTML = userInput;
}
'''
        
        insights = self.analyzer.analyze_security_patterns(
            code_with_xss, "vulnerable.js"
        )
        
        # Should detect XSS vulnerability
        xss_insights = [
            i for i in insights 
            if "xss" in i.title.lower()
        ]
        assert len(xss_insights) > 0
    
    def test_analyze_hardcoded_secrets(self):
        """Test detection of hardcoded secrets."""
        code_with_secrets = '''
API_KEY = "sk-1234567890abcdef1234567890abcdef"
password = "super_secret_password"
def connect_to_db():
    return connect("user:secret123@db.host.com")
'''
        
        insights = self.analyzer.analyze_security_patterns(
            code_with_secrets, "secrets.py"
        )
        
        # Should detect hardcoded secrets
        secret_insights = [
            i for i in insights 
            if "hardcoded" in i.title.lower()
        ]
        assert len(secret_insights) > 0
        assert any(i.severity == InsightSeverity.HIGH for i in secret_insights)
    
    def test_get_security_recommendations(self):
        """Test getting security recommendations."""
        sql_recommendations = self.analyzer._get_security_recommendations("sql_injection")
        assert "parameterized queries" in " ".join(sql_recommendations).lower()
        
        xss_recommendations = self.analyzer._get_security_recommendations("xss")
        assert "sanitize" in " ".join(xss_recommendations).lower()
        
        secrets_recommendations = self.analyzer._get_security_recommendations("hardcoded_secrets")
        assert "environment variables" in " ".join(secrets_recommendations).lower()


class TestTechnicalDebtAnalyzer:
    """Test cases for technical debt analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TechnicalDebtAnalyzer()
    
    def test_initial_state(self):
        """Test initial analyzer state."""
        assert "code_duplication" in self.analyzer.debt_indicators
        assert "large_functions" in self.analyzer.debt_indicators
        assert "complex_conditions" in self.analyzer.debt_indicators
        assert "poor_naming" in self.analyzer.debt_indicators
    
    def test_analyze_low_technical_debt(self):
        """Test analysis with low technical debt."""
        analysis_results = {
            "duplication_percentage": 5,
            "average_function_length": 30,
            "complex_conditions": 2,
            "poor_naming_count": 3
        }
        
        insights = self.analyzer.analyze_technical_debt(analysis_results)
        assert len(insights) == 0  # Should not generate insights for low debt
    
    def test_analyze_high_technical_debt(self):
        """Test analysis with high technical debt."""
        analysis_results = {
            "duplication_percentage": 15,  # High duplication
            "average_function_length": 80,  # Large functions
            "complex_conditions": 8,       # Complex conditions
            "poor_naming_count": 15        # Poor naming
        }
        
        insights = self.analyzer.analyze_technical_debt(analysis_results)
        
        # Should generate technical debt insight
        debt_insights = [
            i for i in insights 
            if i.type == InsightType.TECHNICAL_DEBT
        ]
        assert len(debt_insights) > 0
        
        insight = debt_insights[0]
        assert insight.severity in [InsightSeverity.MEDIUM, InsightSeverity.HIGH]
        assert "technical debt" in insight.title.lower()
        assert len(insight.evidence) > 0
        assert len(insight.recommendations) > 0


class TestInsightsEngine:
    """Test cases for the main insights engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = InsightsEngine()
    
    def test_initial_state(self):
        """Test initial engine state."""
        assert self.engine.quality_analyzer is not None
        assert self.engine.performance_analyzer is not None
        assert self.engine.security_analyzer is not None
        assert self.engine.debt_analyzer is not None
        assert len(self.engine.insights_history) == 0
    
    @pytest.mark.asyncio
    async def test_generate_insights_basic(self):
        """Test basic insight generation."""
        analysis_data = {
            "code_content": "def test(): pass",
            "file_path": "test.py",
            "duplication_percentage": 5,
            "average_function_length": 20
        }
        
        insights = await self.engine.generate_insights(analysis_data)
        
        # Should return a list of insights
        assert isinstance(insights, list)
        # Each insight should be stored in history
        assert len(self.engine.insights_history) == len(insights)
    
    @pytest.mark.asyncio
    async def test_generate_insights_with_security_issues(self):
        """Test insight generation with security code."""
        analysis_data = {
            "code_content": "query = 'SELECT * FROM users WHERE id = %s' % user_id",
            "file_path": "vulnerable.py"
        }
        
        insights = await self.engine.generate_insights(analysis_data)
        
        # Should detect security issues
        security_insights = [
            i for i in insights 
            if i.type == InsightType.SECURITY_RISK
        ]
        assert len(security_insights) > 0
    
    @pytest.mark.asyncio
    @patch('src.analytics.insights_engine.np')
    @patch('src.analytics.insights_engine.KMeans')
    async def test_generate_ml_insights(self, mock_kmeans, mock_np):
        """Test ML-based insight generation."""
        # Mock numpy and sklearn availability
        mock_np.return_value = True
        mock_kmeans.return_value = True
        
        # Add some history for ML analysis
        for i in range(60):
            insight = Insight(
                id=f"insight_{i}",
                type=InsightType.CODE_QUALITY_TREND,
                severity=InsightSeverity.LOW,
                title="Test insight",
                description="Test",
                evidence=[],
                recommendations=[],
                confidence_score=0.5,
                metadata={"metric_value": i * 0.1}
            )
            self.engine.insights_history.append(insight)
        
        analysis_data = {"metric_value": 10.0}
        insights = await self.engine._generate_ml_insights(analysis_data)
        
        # ML insights might be generated if patterns are detected
        assert isinstance(insights, list)
    
    def test_add_metric(self):
        """Test adding code metrics."""
        metric = CodeMetric(
            timestamp=datetime.now(),
            file_path="test.py",
            metric_type="complexity",
            value=5.0
        )
        
        self.engine.add_metric(metric)
        
        # Should be added to quality analyzer
        assert len(self.engine.quality_analyzer.metrics_history) == 1
    
    def test_add_performance_data(self):
        """Test adding performance data."""
        data = {"response_time": 2.0, "memory_mb": 128.0}
        
        self.engine.add_performance_data(data)
        
        # Should be added to performance analyzer
        assert len(self.engine.performance_analyzer.performance_data) == 1
    
    def test_add_security_event(self):
        """Test adding security events."""
        event = {"type": "login_failure", "user": "test"}
        
        self.engine.add_security_event(event)
        
        # Should be added to security analyzer
        assert len(self.engine.security_analyzer.security_events) == 1
    
    def test_get_insights_summary(self):
        """Test insights summary generation."""
        # Add some test insights
        insights = [
            Insight(
                id="1", type=InsightType.SECURITY_RISK, severity=InsightSeverity.CRITICAL,
                title="Critical issue", description="Test", evidence=[], recommendations=[],
                confidence_score=0.9
            ),
            Insight(
                id="2", type=InsightType.CODE_QUALITY_TREND, severity=InsightSeverity.HIGH,
                title="Quality issue", description="Test", evidence=[], recommendations=[],
                confidence_score=0.8
            ),
            Insight(
                id="3", type=InsightType.PERFORMANCE_PATTERN, severity=InsightSeverity.MEDIUM,
                title="Performance issue", description="Test", evidence=[], recommendations=[],
                confidence_score=0.7
            )
        ]
        
        self.engine.insights_history.extend(insights)
        
        summary = self.engine.get_insights_summary()
        
        assert summary["total_insights"] == 3
        assert summary["critical_insights"] == 1
        assert "severity_breakdown" in summary
        assert "type_breakdown" in summary
        assert summary["average_confidence"] > 0
    
    def test_get_recommendations_by_priority(self):
        """Test getting prioritized recommendations."""
        # Add insights with different severities
        high_severity_insight = Insight(
            id="high", type=InsightType.SECURITY_RISK, severity=InsightSeverity.HIGH,
            title="High severity", description="Test", evidence=[], 
            recommendations=["Fix security issue", "Add validation"],
            confidence_score=0.9
        )
        
        low_severity_insight = Insight(
            id="low", type=InsightType.CODE_QUALITY_TREND, severity=InsightSeverity.LOW,
            title="Low severity", description="Test", evidence=[],
            recommendations=["Minor improvement", "Optional optimization"],
            confidence_score=0.6
        )
        
        self.engine.insights_history.extend([high_severity_insight, low_severity_insight])
        
        recommendations = self.engine.get_recommendations_by_priority()
        
        assert len(recommendations) > 0
        # High severity recommendations should be weighted higher
        assert "Fix security issue" in recommendations
    
    def test_calculate_priority_weight(self):
        """Test priority weight calculation."""
        critical_insight = Insight(
            id="critical", type=InsightType.SECURITY_RISK, severity=InsightSeverity.CRITICAL,
            title="Critical", description="Test", evidence=[], recommendations=[],
            confidence_score=0.95
        )
        
        low_insight = Insight(
            id="low", type=InsightType.CODE_QUALITY_TREND, severity=InsightSeverity.LOW,
            title="Low", description="Test", evidence=[], recommendations=[],
            confidence_score=0.5
        )
        
        critical_weight = self.engine._calculate_priority_weight(critical_insight)
        low_weight = self.engine._calculate_priority_weight(low_insight)
        
        assert critical_weight > low_weight
        assert critical_weight > 4.0  # Critical should have high weight
        assert low_weight < 2.0      # Low should have low weight
    
    def test_extract_features(self):
        """Test feature extraction for ML analysis."""
        data = {
            "complexity": 5.0,
            "lines_of_code": 100,
            "test_coverage": 85.5,
            "non_numeric": "text",
            "string_number": "42.5"
        }
        
        features = self.engine._extract_features(data)
        
        # Should extract numerical features
        assert 5.0 in features
        assert 100 in features
        assert 85.5 in features
        assert 42.5 in features
        # Should exclude non-numeric values
        assert len(features) == 4


def test_global_insights_engine():
    """Test global insights engine singleton."""
    engine1 = get_insights_engine()
    engine2 = get_insights_engine()
    
    assert engine1 is engine2  # Should be same instance


if __name__ == "__main__":
    pytest.main([__file__])