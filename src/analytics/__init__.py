"""
Analytics and insights module for AI Code Agent.
"""

from .insights_engine import (
    InsightsEngine,
    Insight,
    InsightType,
    InsightSeverity,
    CodeMetric,
    get_insights_engine
)

from .ml_models import (
    MLModelManager,
    ModelPrediction,
    BugPredictionModel,
    PerformancePredictor,
    AnomalyDetector,
    get_ml_manager
)

from .reporting import (
    ReportGenerator,
    AnalyticsReport,
    ReportSection,
    ChartGenerator,
    get_report_generator
)

__all__ = [
    "InsightsEngine",
    "Insight", 
    "InsightType",
    "InsightSeverity",
    "CodeMetric",
    "get_insights_engine",
    "MLModelManager",
    "ModelPrediction",
    "BugPredictionModel", 
    "PerformancePredictor",
    "AnomalyDetector",
    "get_ml_manager",
    "ReportGenerator",
    "AnalyticsReport",
    "ReportSection",
    "ChartGenerator",
    "get_report_generator"
]