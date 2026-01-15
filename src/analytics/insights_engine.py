"""
Advanced analytics and machine learning insights for AI Code Agent.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import statistics
import json
import re
from pathlib import Path

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
except ImportError:
    np = None
    KMeans = None
    TfidfVectorizer = None
    cosine_similarity = None
    StandardScaler = None


class InsightType(Enum):
    """Types of insights that can be generated."""
    CODE_QUALITY_TREND = "code_quality_trend"
    PERFORMANCE_PATTERN = "performance_pattern"
    SECURITY_RISK = "security_risk"
    MAINTENANCE_BURDEN = "maintenance_burden"
    TEAM_PRODUCTIVITY = "team_productivity"
    TECHNICAL_DEBT = "technical_debt"
    BUG_PREDICTION = "bug_prediction"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"


class InsightSeverity(Enum):
    """Severity levels for insights."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeMetric:
    """Represents a code quality metric over time."""
    timestamp: datetime
    file_path: str
    metric_type: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Insight:
    """Represents a generated insight."""
    id: str
    type: InsightType
    severity: InsightSeverity
    title: str
    description: str
    evidence: List[str]
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class CodeQualityAnalyzer:
    """Analyzes code quality trends and patterns."""
    
    def __init__(self):
        self.metrics_history: List[CodeMetric] = []
        self.quality_thresholds = {
            "complexity": 10,
            "maintainability": 70,
            "test_coverage": 80,
            "duplication": 5
        }
    
    def add_metric(self, metric: CodeMetric):
        """Add a code metric to the history."""
        self.metrics_history.append(metric)
        
        # Keep only recent metrics (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff
        ]
    
    def analyze_quality_trends(self) -> List[Insight]:
        """Analyze code quality trends over time."""
        insights = []
        
        if len(self.metrics_history) < 10:
            return insights
        
        # Group metrics by type and file
        metrics_by_type = defaultdict(list)
        for metric in self.metrics_history:
            key = (metric.metric_type, metric.file_path)
            metrics_by_type[key].append(metric)
        
        for (metric_type, file_path), metrics in metrics_by_type.items():
            if len(metrics) < 5:
                continue
            
            # Sort by timestamp
            metrics.sort(key=lambda m: m.timestamp)
            values = [m.value for m in metrics]
            
            # Calculate trend
            trend = self._calculate_trend(values)
            
            # Generate insights based on trend
            if metric_type == "complexity" and trend > 0.5:
                insights.append(Insight(
                    id=f"complexity_trend_{hash(file_path)}",
                    type=InsightType.CODE_QUALITY_TREND,
                    severity=InsightSeverity.MEDIUM,
                    title=f"Increasing Complexity in {Path(file_path).name}",
                    description=f"Cyclomatic complexity has increased by {trend:.1f}% over the last 30 days",
                    evidence=[
                        f"Complexity increased from {values[0]:.1f} to {values[-1]:.1f}",
                        f"Trend analysis shows consistent upward pattern"
                    ],
                    recommendations=[
                        "Consider refactoring complex functions",
                        "Break down large functions into smaller, focused ones",
                        "Add unit tests to verify refactoring correctness"
                    ],
                    confidence_score=0.8,
                    metadata={"file_path": file_path, "trend": trend}
                ))
            
            elif metric_type == "maintainability" and trend < -0.3:
                insights.append(Insight(
                    id=f"maintainability_trend_{hash(file_path)}",
                    type=InsightType.MAINTENANCE_BURDEN,
                    severity=InsightSeverity.HIGH,
                    title=f"Declining Maintainability in {Path(file_path).name}",
                    description=f"Code maintainability has decreased by {abs(trend):.1f}% recently",
                    evidence=[
                        f"Maintainability score dropped from {values[0]:.1f} to {values[-1]:.1f}",
                        f"Consistent downward trend detected"
                    ],
                    recommendations=[
                        "Prioritize refactoring of this file",
                        "Improve code documentation and comments",
                        "Consider architectural improvements"
                    ],
                    confidence_score=0.9,
                    metadata={"file_path": file_path, "trend": trend}
                ))
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend as percentage change."""
        if len(values) < 2:
            return 0.0
        
        start_avg = statistics.mean(values[:len(values)//3])
        end_avg = statistics.mean(values[-len(values)//3:])
        
        if start_avg == 0:
            return 0.0
        
        return ((end_avg - start_avg) / start_avg) * 100


class PerformanceAnalyzer:
    """Analyzes performance patterns and bottlenecks."""
    
    def __init__(self):
        self.performance_data: List[Dict[str, Any]] = []
    
    def add_performance_data(self, data: Dict[str, Any]):
        """Add performance measurement data."""
        data["timestamp"] = datetime.now()
        self.performance_data.append(data)
        
        # Keep only recent data (last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.performance_data = [
            d for d in self.performance_data 
            if d["timestamp"] > cutoff
        ]
    
    def analyze_performance_patterns(self) -> List[Insight]:
        """Analyze performance patterns and identify bottlenecks."""
        insights = []
        
        if len(self.performance_data) < 20:
            return insights
        
        # Analyze response time patterns
        response_times = [d.get("response_time", 0) for d in self.performance_data]
        avg_response_time = statistics.mean(response_times)
        p95_response_time = np.percentile(response_times, 95) if np else max(response_times)
        
        if avg_response_time > 5.0:
            insights.append(Insight(
                id="high_response_time",
                type=InsightType.PERFORMANCE_PATTERN,
                severity=InsightSeverity.HIGH,
                title="High Average Response Time Detected",
                description=f"Average response time is {avg_response_time:.2f}s, which exceeds optimal threshold",
                evidence=[
                    f"Average response time: {avg_response_time:.2f}s",
                    f"95th percentile: {p95_response_time:.2f}s",
                    f"Based on {len(response_times)} measurements"
                ],
                recommendations=[
                    "Implement caching for frequently accessed data",
                    "Optimize database queries and add indexes",
                    "Consider implementing request batching",
                    "Profile code to identify specific bottlenecks"
                ],
                confidence_score=0.85,
                metadata={"avg_response_time": avg_response_time}
            ))
        
        # Analyze memory usage patterns
        memory_usage = [d.get("memory_mb", 0) for d in self.performance_data if d.get("memory_mb")]
        if memory_usage and len(memory_usage) > 10:
            memory_trend = self._calculate_memory_trend(memory_usage)
            
            if memory_trend > 20:  # 20% increase
                insights.append(Insight(
                    id="memory_leak_detection",
                    type=InsightType.PERFORMANCE_PATTERN,
                    severity=InsightSeverity.CRITICAL,
                    title="Potential Memory Leak Detected",
                    description=f"Memory usage has increased by {memory_trend:.1f}% over recent operations",
                    evidence=[
                        f"Memory usage trend: +{memory_trend:.1f}%",
                        f"Current average: {statistics.mean(memory_usage[-5:]):.1f}MB",
                        f"Initial average: {statistics.mean(memory_usage[:5]):.1f}MB"
                    ],
                    recommendations=[
                        "Investigate potential memory leaks",
                        "Review object lifecycle management",
                        "Implement memory profiling and monitoring",
                        "Consider garbage collection optimization"
                    ],
                    confidence_score=0.9,
                    metadata={"memory_trend": memory_trend}
                ))
        
        return insights
    
    def _calculate_memory_trend(self, memory_values: List[float]) -> float:
        """Calculate memory usage trend."""
        if len(memory_values) < 10:
            return 0.0
        
        start_avg = statistics.mean(memory_values[:5])
        end_avg = statistics.mean(memory_values[-5:])
        
        if start_avg == 0:
            return 0.0
        
        return ((end_avg - start_avg) / start_avg) * 100


class SecurityAnalyzer:
    """Analyzes security patterns and vulnerabilities."""
    
    def __init__(self):
        self.security_events: List[Dict[str, Any]] = []
        self.vulnerability_patterns = {
            "sql_injection": [
                r"(?i)execute\s*\(\s*[\"'].*%.*[\"']\s*\)",
                r"(?i)query\s*\(\s*[\"'].*\+.*[\"']\s*\)"
            ],
            "xss": [
                r"(?i)innerHTML\s*=\s*.*\+.*",
                r"(?i)document\.write\s*\(\s*.*\+.*\)"
            ],
            "hardcoded_secrets": [
                r"(?i)(password|secret|key|token)\s*=\s*[\"'][^\"']{8,}[\"']",
                r"(?i)(api_key|access_token)\s*:\s*[\"'][^\"']{16,}[\"']"
            ]
        }
    
    def add_security_event(self, event: Dict[str, Any]):
        """Add a security-related event."""
        event["timestamp"] = datetime.now()
        self.security_events.append(event)
        
        # Keep only recent events (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        self.security_events = [
            e for e in self.security_events 
            if e["timestamp"] > cutoff
        ]
    
    def analyze_security_patterns(self, code_content: str, file_path: str) -> List[Insight]:
        """Analyze code for security vulnerabilities."""
        insights = []
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, code_content)
                if matches:
                    severity = InsightSeverity.HIGH if vuln_type in ["sql_injection", "hardcoded_secrets"] else InsightSeverity.MEDIUM
                    
                    insights.append(Insight(
                        id=f"security_{vuln_type}_{hash(file_path)}",
                        type=InsightType.SECURITY_RISK,
                        severity=severity,
                        title=f"Potential {vuln_type.replace('_', ' ').title()} Vulnerability",
                        description=f"Code patterns suggest potential {vuln_type.replace('_', ' ')} vulnerability in {Path(file_path).name}",
                        evidence=[
                            f"Found {len(matches)} suspicious pattern(s)",
                            f"Pattern type: {vuln_type}",
                            f"File: {file_path}"
                        ],
                        recommendations=self._get_security_recommendations(vuln_type),
                        confidence_score=0.7,
                        metadata={"vulnerability_type": vuln_type, "matches": len(matches)}
                    ))
        
        return insights
    
    def _get_security_recommendations(self, vuln_type: str) -> List[str]:
        """Get security recommendations for vulnerability type."""
        recommendations = {
            "sql_injection": [
                "Use parameterized queries or prepared statements",
                "Implement input validation and sanitization",
                "Use an ORM with built-in SQL injection protection",
                "Never concatenate user input directly into SQL queries"
            ],
            "xss": [
                "Sanitize all user input before rendering",
                "Use content security policies (CSP)",
                "Escape HTML content properly",
                "Use frameworks with built-in XSS protection"
            ],
            "hardcoded_secrets": [
                "Move secrets to environment variables",
                "Use a secrets management system",
                "Never commit secrets to version control",
                "Implement secret rotation policies"
            ]
        }
        return recommendations.get(vuln_type, ["Review code for security best practices"])


class TechnicalDebtAnalyzer:
    """Analyzes technical debt and maintenance burden."""
    
    def __init__(self):
        self.debt_indicators = {
            "code_duplication": 0.3,
            "large_functions": 0.4,
            "complex_conditions": 0.2,
            "poor_naming": 0.1
        }
    
    def analyze_technical_debt(self, analysis_results: Dict[str, Any]) -> List[Insight]:
        """Analyze technical debt from code analysis results."""
        insights = []
        
        # Calculate overall debt score
        debt_score = 0
        debt_factors = []
        
        if analysis_results.get("duplication_percentage", 0) > 10:
            debt_score += 30
            debt_factors.append("High code duplication")
        
        if analysis_results.get("average_function_length", 0) > 50:
            debt_score += 25
            debt_factors.append("Large function sizes")
        
        if analysis_results.get("complex_conditions", 0) > 5:
            debt_score += 20
            debt_factors.append("Complex conditional logic")
        
        if analysis_results.get("poor_naming_count", 0) > 10:
            debt_score += 15
            debt_factors.append("Poor naming conventions")
        
        if debt_score > 40:
            severity = InsightSeverity.HIGH if debt_score > 70 else InsightSeverity.MEDIUM
            
            insights.append(Insight(
                id=f"technical_debt_{hash(str(analysis_results))}",
                type=InsightType.TECHNICAL_DEBT,
                severity=severity,
                title=f"High Technical Debt Detected (Score: {debt_score}/100)",
                description="Multiple factors are contributing to technical debt in this codebase",
                evidence=debt_factors,
                recommendations=[
                    "Prioritize refactoring based on debt factors",
                    "Implement code review processes to prevent new debt",
                    "Set up automated quality gates",
                    "Create a technical debt reduction roadmap"
                ],
                confidence_score=0.8,
                metadata={"debt_score": debt_score, "factors": debt_factors}
            ))
        
        return insights


class InsightsEngine:
    """Main insights engine that coordinates all analyzers."""
    
    def __init__(self):
        self.quality_analyzer = CodeQualityAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.debt_analyzer = TechnicalDebtAnalyzer()
        self.insights_history: List[Insight] = []
    
    async def generate_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Generate comprehensive insights from analysis data."""
        all_insights = []
        
        # Quality trend analysis
        quality_insights = self.quality_analyzer.analyze_quality_trends()
        all_insights.extend(quality_insights)
        
        # Performance pattern analysis
        performance_insights = self.performance_analyzer.analyze_performance_patterns()
        all_insights.extend(performance_insights)
        
        # Security analysis
        if "code_content" in analysis_data and "file_path" in analysis_data:
            security_insights = self.security_analyzer.analyze_security_patterns(
                analysis_data["code_content"],
                analysis_data["file_path"]
            )
            all_insights.extend(security_insights)
        
        # Technical debt analysis
        debt_insights = self.debt_analyzer.analyze_technical_debt(analysis_data)
        all_insights.extend(debt_insights)
        
        # ML-based insights (if libraries available)
        if np and KMeans:
            ml_insights = await self._generate_ml_insights(analysis_data)
            all_insights.extend(ml_insights)
        
        # Store insights
        self.insights_history.extend(all_insights)
        
        # Keep only recent insights (last 90 days)
        cutoff = datetime.now() - timedelta(days=90)
        self.insights_history = [
            i for i in self.insights_history 
            if i.timestamp > cutoff
        ]
        
        return all_insights
    
    async def _generate_ml_insights(self, analysis_data: Dict[str, Any]) -> List[Insight]:
        """Generate ML-based insights using clustering and pattern recognition."""
        insights = []
        
        try:
            # Prepare feature vector from analysis data
            features = self._extract_features(analysis_data)
            
            if len(features) > 0:
                # Simple anomaly detection using statistical methods
                if len(self.insights_history) > 50:
                    historical_features = [self._extract_features(i.metadata) for i in self.insights_history[-50:]]
                    historical_features = [f for f in historical_features if len(f) == len(features)]
                    
                    if len(historical_features) > 10:
                        # Check for anomalies
                        current_score = sum(features) / len(features)
                        historical_scores = [sum(f) / len(f) for f in historical_features]
                        
                        mean_score = statistics.mean(historical_scores)
                        std_score = statistics.stdev(historical_scores) if len(historical_scores) > 1 else 0
                        
                        if abs(current_score - mean_score) > 2 * std_score:
                            insights.append(Insight(
                                id=f"anomaly_{int(time.time())}",
                                type=InsightType.OPTIMIZATION_OPPORTUNITY,
                                severity=InsightSeverity.MEDIUM,
                                title="Anomalous Code Patterns Detected",
                                description="Statistical analysis indicates unusual patterns in code metrics",
                                evidence=[
                                    f"Current analysis score: {current_score:.2f}",
                                    f"Historical average: {mean_score:.2f}",
                                    f"Standard deviation: {std_score:.2f}"
                                ],
                                recommendations=[
                                    "Review recent code changes for quality issues",
                                    "Investigate deviations from coding standards",
                                    "Consider additional testing for unusual patterns"
                                ],
                                confidence_score=0.6,
                                metadata={"ml_analysis": True, "anomaly_score": abs(current_score - mean_score)}
                            ))
        
        except Exception as e:
            # ML analysis is optional, don't fail the entire process
            pass
        
        return insights
    
    def _extract_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from analysis data."""
        features = []
        
        # Extract numerical values from the data
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str) and value.replace('.', '').isdigit():
                features.append(float(value))
        
        return features
    
    def add_metric(self, metric: CodeMetric):
        """Add a code metric for trend analysis."""
        self.quality_analyzer.add_metric(metric)
    
    def add_performance_data(self, data: Dict[str, Any]):
        """Add performance data for pattern analysis."""
        self.performance_analyzer.add_performance_data(data)
    
    def add_security_event(self, event: Dict[str, Any]):
        """Add a security event for analysis."""
        self.security_analyzer.add_security_event(event)
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of all insights."""
        recent_insights = [
            i for i in self.insights_history 
            if i.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        severity_counts = Counter(i.severity.value for i in recent_insights)
        type_counts = Counter(i.type.value for i in recent_insights)
        
        return {
            "total_insights": len(recent_insights),
            "severity_breakdown": dict(severity_counts),
            "type_breakdown": dict(type_counts),
            "average_confidence": statistics.mean([i.confidence_score for i in recent_insights]) if recent_insights else 0,
            "most_common_types": type_counts.most_common(5),
            "critical_insights": len([i for i in recent_insights if i.severity == InsightSeverity.CRITICAL])
        }
    
    def get_recommendations_by_priority(self) -> List[str]:
        """Get prioritized recommendations across all insights."""
        recent_insights = [
            i for i in self.insights_history 
            if i.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        # Weight recommendations by severity and confidence
        weighted_recommendations = []
        
        for insight in recent_insights:
            weight = self._calculate_priority_weight(insight)
            for rec in insight.recommendations:
                weighted_recommendations.append((rec, weight))
        
        # Group by recommendation text and sum weights
        rec_weights = defaultdict(float)
        for rec, weight in weighted_recommendations:
            rec_weights[rec] += weight
        
        # Sort by weight and return top recommendations
        sorted_recs = sorted(rec_weights.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in sorted_recs[:10]]
    
    def _calculate_priority_weight(self, insight: Insight) -> float:
        """Calculate priority weight for an insight."""
        severity_weights = {
            InsightSeverity.CRITICAL: 5.0,
            InsightSeverity.HIGH: 4.0,
            InsightSeverity.MEDIUM: 3.0,
            InsightSeverity.LOW: 2.0,
            InsightSeverity.INFO: 1.0
        }
        
        base_weight = severity_weights.get(insight.severity, 1.0)
        confidence_weight = insight.confidence_score
        
        return base_weight * confidence_weight


# Global insights engine
_insights_engine: Optional[InsightsEngine] = None


def get_insights_engine() -> InsightsEngine:
    """Get or create global insights engine."""
    global _insights_engine
    if _insights_engine is None:
        _insights_engine = InsightsEngine()
    return _insights_engine