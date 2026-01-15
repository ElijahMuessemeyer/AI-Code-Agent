"""
Advanced reporting and visualization for AI Code Agent analytics.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import base64
from io import BytesIO

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
except ImportError:
    plt = None
    sns = None
    pd = None

from .insights_engine import InsightsEngine, Insight, InsightType, InsightSeverity
from .ml_models import MLModelManager, get_ml_manager


@dataclass
class ReportSection:
    """Represents a section of an analytics report."""
    title: str
    content: str
    charts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""
    title: str
    generated_at: datetime
    summary: str
    sections: List[ReportSection]
    overall_score: float
    key_insights: List[str]
    priority_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "summary": self.summary,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "charts": section.charts,
                    "recommendations": section.recommendations,
                    "metadata": section.metadata
                }
                for section in self.sections
            ],
            "overall_score": self.overall_score,
            "key_insights": self.key_insights,
            "priority_actions": self.priority_actions,
            "metadata": self.metadata
        }


class ChartGenerator:
    """Generates charts and visualizations for analytics."""
    
    def __init__(self, output_dir: str = ".cache/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if plt:
            # Set style
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
            sns.set_palette("husl") if sns else None
    
    def generate_trend_chart(self, data: Dict[str, List[float]], 
                           title: str, ylabel: str) -> Optional[str]:
        """Generate a trend line chart."""
        if not plt or not data:
            return None
        
        try:
            plt.figure(figsize=(12, 6))
            
            for label, values in data.items():
                x = list(range(len(values)))
                plt.plot(x, values, marker='o', label=label, linewidth=2)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save chart
            chart_file = self.output_dir / f"trend_{int(time.time())}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(chart_file)
            
        except Exception as e:
            plt.close()
            return None
    
    def generate_distribution_chart(self, data: List[float], 
                                  title: str, xlabel: str) -> Optional[str]:
        """Generate a distribution histogram."""
        if not plt or not data:
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            
            plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(sum(data)/len(data), color='red', linestyle='--', 
                       label=f'Mean: {sum(data)/len(data):.2f}')
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save chart
            chart_file = self.output_dir / f"dist_{int(time.time())}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(chart_file)
            
        except Exception as e:
            plt.close()
            return None
    
    def generate_heatmap(self, data: Dict[str, Dict[str, float]], 
                        title: str) -> Optional[str]:
        """Generate a correlation heatmap."""
        if not plt or not sns or not pd or not data:
            return None
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(10, 8))
            
            # Generate heatmap
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save chart
            chart_file = self.output_dir / f"heatmap_{int(time.time())}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(chart_file)
            
        except Exception as e:
            plt.close()
            return None
    
    def generate_severity_pie_chart(self, severity_counts: Dict[str, int], 
                                   title: str) -> Optional[str]:
        """Generate a pie chart for insight severities."""
        if not plt or not severity_counts:
            return None
        
        try:
            plt.figure(figsize=(8, 8))
            
            colors = {
                'critical': '#ff4444',
                'high': '#ff8800',
                'medium': '#ffdd00',
                'low': '#88dd00',
                'info': '#44aaff'
            }
            
            labels = list(severity_counts.keys())
            sizes = list(severity_counts.values())
            chart_colors = [colors.get(label.lower(), '#cccccc') for label in labels]
            
            plt.pie(sizes, labels=labels, colors=chart_colors, autopct='%1.1f%%',
                   startangle=90, explode=[0.05] * len(labels))
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.axis('equal')
            
            # Save chart
            chart_file = self.output_dir / f"pie_{int(time.time())}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(chart_file)
            
        except Exception as e:
            plt.close()
            return None
    
    def generate_bar_chart(self, data: Dict[str, float], 
                          title: str, ylabel: str) -> Optional[str]:
        """Generate a bar chart."""
        if not plt or not data:
            return None
        
        try:
            plt.figure(figsize=(12, 6))
            
            x_labels = list(data.keys())
            values = list(data.values())
            
            bars = plt.bar(x_labels, values, color='steelblue', alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                        f'{value:.2f}', ha='center', va='bottom')
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.ylabel(ylabel, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save chart
            chart_file = self.output_dir / f"bar_{int(time.time())}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(chart_file)
            
        except Exception as e:
            plt.close()
            return None


class ReportGenerator:
    """Generates comprehensive analytics reports."""
    
    def __init__(self, insights_engine: InsightsEngine, ml_manager: MLModelManager):
        self.insights_engine = insights_engine
        self.ml_manager = ml_manager
        self.chart_generator = ChartGenerator()
    
    async def generate_comprehensive_report(self, 
                                          analysis_data: Dict[str, Any],
                                          time_range: timedelta = timedelta(days=7)) -> AnalyticsReport:
        """Generate a comprehensive analytics report."""
        
        # Get recent insights
        recent_insights = [
            insight for insight in self.insights_engine.insights_history
            if insight.timestamp > datetime.now() - time_range
        ]
        
        # Generate sections
        sections = []
        
        # Executive Summary
        summary_section = await self._generate_summary_section(recent_insights, analysis_data)
        sections.append(summary_section)
        
        # Code Quality Analysis
        quality_section = await self._generate_quality_section(recent_insights, analysis_data)
        sections.append(quality_section)
        
        # Performance Analysis
        performance_section = await self._generate_performance_section(recent_insights, analysis_data)
        sections.append(performance_section)
        
        # Security Analysis
        security_section = await self._generate_security_section(recent_insights, analysis_data)
        sections.append(security_section)
        
        # ML Predictions
        ml_section = await self._generate_ml_section(analysis_data)
        sections.append(ml_section)
        
        # Recommendations
        recommendations_section = await self._generate_recommendations_section(recent_insights)
        sections.append(recommendations_section)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(recent_insights, analysis_data)
        
        # Extract key insights
        key_insights = self._extract_key_insights(recent_insights)
        
        # Get priority actions
        priority_actions = self.insights_engine.get_recommendations_by_priority()[:5]
        
        return AnalyticsReport(
            title=f"AI Code Agent Analytics Report",
            generated_at=datetime.now(),
            summary=self._generate_executive_summary(recent_insights, overall_score),
            sections=sections,
            overall_score=overall_score,
            key_insights=key_insights,
            priority_actions=priority_actions,
            metadata={
                "time_range_days": time_range.days,
                "total_insights": len(recent_insights),
                "analysis_timestamp": datetime.now().isoformat()
            }
        )
    
    async def _generate_summary_section(self, insights: List[Insight], 
                                      data: Dict[str, Any]) -> ReportSection:
        """Generate executive summary section."""
        
        # Insight statistics
        severity_counts = {}
        for insight in insights:
            severity = insight.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Generate severity pie chart
        charts = []
        if severity_counts:
            severity_chart = self.chart_generator.generate_severity_pie_chart(
                severity_counts, "Insights by Severity"
            )
            if severity_chart:
                charts.append(severity_chart)
        
        content = f"""
        **Analysis Period**: Last 7 days
        **Total Insights Generated**: {len(insights)}
        **Critical Issues**: {severity_counts.get('critical', 0)}
        **High Priority Issues**: {severity_counts.get('high', 0)}
        **Overall Health Score**: {self._calculate_overall_score(insights, data):.1f}/100
        
        This report provides a comprehensive analysis of your codebase health,
        including code quality trends, performance patterns, security risks,
        and machine learning predictions.
        """
        
        return ReportSection(
            title="Executive Summary",
            content=content.strip(),
            charts=charts,
            metadata={"severity_distribution": severity_counts}
        )
    
    async def _generate_quality_section(self, insights: List[Insight], 
                                      data: Dict[str, Any]) -> ReportSection:
        """Generate code quality analysis section."""
        
        quality_insights = [i for i in insights if i.type in [
            InsightType.CODE_QUALITY_TREND, 
            InsightType.TECHNICAL_DEBT,
            InsightType.MAINTENANCE_BURDEN
        ]]
        
        # Generate quality metrics chart
        charts = []
        quality_metrics = {
            "Complexity": data.get("complexity_score", 0),
            "Maintainability": data.get("maintainability_score", 0),
            "Test Coverage": data.get("test_coverage", 0),
            "Documentation": data.get("documentation_score", 0)
        }
        
        if quality_metrics:
            quality_chart = self.chart_generator.generate_bar_chart(
                quality_metrics, "Code Quality Metrics", "Score"
            )
            if quality_chart:
                charts.append(quality_chart)
        
        # Calculate quality score
        avg_quality = sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0
        
        content = f"""
        **Code Quality Assessment**
        
        **Average Quality Score**: {avg_quality:.1f}/100
        **Quality-related Insights**: {len(quality_insights)}
        **Technical Debt Issues**: {len([i for i in quality_insights if i.type == InsightType.TECHNICAL_DEBT])}
        
        **Key Quality Metrics:**
        - Complexity Score: {quality_metrics.get('Complexity', 0):.1f}
        - Maintainability: {quality_metrics.get('Maintainability', 0):.1f}
        - Test Coverage: {quality_metrics.get('Test Coverage', 0):.1f}%
        - Documentation: {quality_metrics.get('Documentation', 0):.1f}
        
        **Quality Trends:**
        {"Improving" if avg_quality > 70 else "Declining" if avg_quality < 50 else "Stable"}
        """
        
        recommendations = [
            "Implement automated code quality gates",
            "Increase test coverage to 80%+",
            "Refactor high-complexity functions",
            "Improve code documentation"
        ]
        
        return ReportSection(
            title="Code Quality Analysis",
            content=content.strip(),
            charts=charts,
            recommendations=recommendations,
            metadata={"quality_metrics": quality_metrics}
        )
    
    async def _generate_performance_section(self, insights: List[Insight], 
                                          data: Dict[str, Any]) -> ReportSection:
        """Generate performance analysis section."""
        
        performance_insights = [i for i in insights if i.type == InsightType.PERFORMANCE_PATTERN]
        
        # Performance metrics
        perf_data = {
            "Avg Response Time": data.get("avg_response_time", 0),
            "Memory Usage": data.get("memory_usage_mb", 0),
            "CPU Usage": data.get("cpu_usage_percent", 0),
            "Cache Hit Rate": data.get("cache_hit_rate", 0) * 100
        }
        
        charts = []
        if perf_data:
            perf_chart = self.chart_generator.generate_bar_chart(
                perf_data, "Performance Metrics", "Value"
            )
            if perf_chart:
                charts.append(perf_chart)
        
        content = f"""
        **Performance Analysis**
        
        **Performance Issues**: {len(performance_insights)}
        **Response Time**: {perf_data.get('Avg Response Time', 0):.2f}s
        **Memory Usage**: {perf_data.get('Memory Usage', 0):.1f}MB
        **CPU Usage**: {perf_data.get('CPU Usage', 0):.1f}%
        **Cache Efficiency**: {perf_data.get('Cache Hit Rate', 0):.1f}%
        
        **Performance Status**: {"Optimal" if perf_data.get('Avg Response Time', 0) < 1.0 else "Needs Attention"}
        """
        
        recommendations = [
            "Implement caching for frequently accessed data",
            "Optimize database queries",
            "Monitor memory usage patterns",
            "Set up performance alerting"
        ]
        
        return ReportSection(
            title="Performance Analysis",
            content=content.strip(),
            charts=charts,
            recommendations=recommendations,
            metadata={"performance_metrics": perf_data}
        )
    
    async def _generate_security_section(self, insights: List[Insight], 
                                       data: Dict[str, Any]) -> ReportSection:
        """Generate security analysis section."""
        
        security_insights = [i for i in insights if i.type == InsightType.SECURITY_RISK]
        
        # Security metrics
        security_metrics = {
            "Critical Vulnerabilities": len([i for i in security_insights if i.severity == InsightSeverity.CRITICAL]),
            "High Risk Issues": len([i for i in security_insights if i.severity == InsightSeverity.HIGH]),
            "Security Warnings": data.get("security_warnings", 0),
            "Hardcoded Secrets": data.get("hardcoded_secrets", 0)
        }
        
        charts = []
        if any(security_metrics.values()):
            security_chart = self.chart_generator.generate_bar_chart(
                security_metrics, "Security Issues", "Count"
            )
            if security_chart:
                charts.append(security_chart)
        
        risk_level = "High" if security_metrics["Critical Vulnerabilities"] > 0 else \
                    "Medium" if security_metrics["High Risk Issues"] > 0 else "Low"
        
        content = f"""
        **Security Assessment**
        
        **Risk Level**: {risk_level}
        **Total Security Issues**: {len(security_insights)}
        **Critical Vulnerabilities**: {security_metrics['Critical Vulnerabilities']}
        **High Risk Issues**: {security_metrics['High Risk Issues']}
        **Security Warnings**: {security_metrics['Security Warnings']}
        
        **Security Score**: {max(0, 100 - len(security_insights) * 10)}/100
        """
        
        recommendations = [
            "Address critical vulnerabilities immediately",
            "Implement security scanning in CI/CD",
            "Remove hardcoded secrets",
            "Add input validation and sanitization"
        ]
        
        return ReportSection(
            title="Security Analysis",
            content=content.strip(),
            charts=charts,
            recommendations=recommendations,
            metadata={"security_metrics": security_metrics}
        )
    
    async def _generate_ml_section(self, data: Dict[str, Any]) -> ReportSection:
        """Generate ML predictions section."""
        
        # Get ML predictions
        predictions = await self.ml_manager.get_all_predictions(data)
        
        content = f"""
        **Machine Learning Predictions**
        
        **Models Available**: {len(predictions)}
        """
        
        charts = []
        ml_scores = {}
        
        for model_name, prediction in predictions.items():
            if "error" not in prediction.metadata:
                content += f"""
        
        **{model_name.replace('_', ' ').title()}**:
        - Prediction: {prediction.prediction:.3f}
        - Confidence: {prediction.confidence:.1%}
        - Features Used: {len(prediction.features_used)}
                """
                ml_scores[model_name] = prediction.prediction
        
        if ml_scores:
            ml_chart = self.chart_generator.generate_bar_chart(
                ml_scores, "ML Model Predictions", "Score"
            )
            if ml_chart:
                charts.append(ml_chart)
        
        recommendations = [
            "Monitor ML prediction accuracy",
            "Collect more training data",
            "Validate model predictions with domain experts",
            "Implement prediction confidence thresholds"
        ]
        
        return ReportSection(
            title="Machine Learning Predictions",
            content=content.strip(),
            charts=charts,
            recommendations=recommendations,
            metadata={"predictions": {k: v.to_dict() if hasattr(v, 'to_dict') else str(v) for k, v in predictions.items()}}
        )
    
    async def _generate_recommendations_section(self, insights: List[Insight]) -> ReportSection:
        """Generate recommendations section."""
        
        # Get prioritized recommendations
        priority_recs = self.insights_engine.get_recommendations_by_priority()
        
        # Group by category
        categories = {
            "Immediate Actions": priority_recs[:3],
            "Short-term Improvements": priority_recs[3:7],
            "Long-term Strategies": priority_recs[7:10]
        }
        
        content = "**Prioritized Recommendations**\n\n"
        
        for category, recs in categories.items():
            if recs:
                content += f"**{category}:**\n"
                for i, rec in enumerate(recs, 1):
                    content += f"{i}. {rec}\n"
                content += "\n"
        
        return ReportSection(
            title="Recommendations & Action Items",
            content=content.strip(),
            metadata={"recommendation_categories": categories}
        )
    
    def _calculate_overall_score(self, insights: List[Insight], data: Dict[str, Any]) -> float:
        """Calculate overall health score."""
        base_score = 100.0
        
        # Deduct points for insights by severity
        for insight in insights:
            if insight.severity == InsightSeverity.CRITICAL:
                base_score -= 15
            elif insight.severity == InsightSeverity.HIGH:
                base_score -= 10
            elif insight.severity == InsightSeverity.MEDIUM:
                base_score -= 5
            elif insight.severity == InsightSeverity.LOW:
                base_score -= 2
        
        # Factor in data metrics
        quality_score = data.get("maintainability_score", 80)
        test_coverage = data.get("test_coverage", 70)
        
        # Weighted average
        final_score = (base_score * 0.5 + quality_score * 0.3 + test_coverage * 0.2)
        
        return max(0, min(100, final_score))
    
    def _extract_key_insights(self, insights: List[Insight]) -> List[str]:
        """Extract key insights from the analysis."""
        key_insights = []
        
        critical_insights = [i for i in insights if i.severity == InsightSeverity.CRITICAL]
        high_insights = [i for i in insights if i.severity == InsightSeverity.HIGH]
        
        if critical_insights:
            key_insights.append(f"Found {len(critical_insights)} critical issues requiring immediate attention")
        
        if high_insights:
            key_insights.append(f"Identified {len(high_insights)} high-priority improvements")
        
        # Add specific insight types
        insight_types = {}
        for insight in insights:
            insight_types[insight.type] = insight_types.get(insight.type, 0) + 1
        
        for insight_type, count in insight_types.items():
            if count > 2:
                key_insights.append(f"Multiple {insight_type.value.replace('_', ' ')} issues detected ({count} instances)")
        
        return key_insights[:5]  # Limit to top 5
    
    def _generate_executive_summary(self, insights: List[Insight], overall_score: float) -> str:
        """Generate executive summary text."""
        critical_count = len([i for i in insights if i.severity == InsightSeverity.CRITICAL])
        high_count = len([i for i in insights if i.severity == InsightSeverity.HIGH])
        
        status = "Excellent" if overall_score >= 90 else \
                "Good" if overall_score >= 75 else \
                "Fair" if overall_score >= 60 else "Needs Improvement"
        
        return f"""
        Code health assessment shows {status.lower()} overall condition with a score of {overall_score:.1f}/100.
        Analysis identified {len(insights)} insights, including {critical_count} critical and {high_count} high-priority issues.
        {"Immediate action required on critical issues." if critical_count > 0 else "Focus on continuous improvement opportunities."}
        """.strip()


# Global report generator
_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Get or create global report generator."""
    global _report_generator
    if _report_generator is None:
        from .insights_engine import get_insights_engine
        _report_generator = ReportGenerator(get_insights_engine(), get_ml_manager())
    return _report_generator