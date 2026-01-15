"""
Machine learning models for predictive analytics in AI Code Agent.
"""

import asyncio
import time
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib
except ImportError:
    np = None
    RandomForestClassifier = None
    IsolationForest = None
    LinearRegression = None
    LogisticRegression = None
    train_test_split = None
    StandardScaler = None
    LabelEncoder = None
    accuracy_score = None
    joblib = None


@dataclass
class ModelPrediction:
    """Represents a model prediction."""
    model_name: str
    prediction: Any
    confidence: float
    features_used: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingData:
    """Represents training data for ML models."""
    features: List[List[float]]
    labels: List[Any]
    feature_names: List[str]
    created_at: datetime = field(default_factory=datetime.now)


class BugPredictionModel:
    """Machine learning model to predict potential bugs."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_path = model_path or ".cache/bug_prediction_model.pkl"
        self.training_data: List[Dict[str, Any]] = []
        
        if RandomForestClassifier:
            self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one."""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.feature_names = model_data['feature_names']
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.scaler = StandardScaler()
        except Exception:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42) if RandomForestClassifier else None
            self.scaler = StandardScaler() if StandardScaler else None
    
    def extract_features(self, code_metrics: Dict[str, Any]) -> List[float]:
        """Extract features from code metrics for bug prediction."""
        features = []
        
        # Code complexity features
        features.append(code_metrics.get('cyclomatic_complexity', 0))
        features.append(code_metrics.get('halstead_difficulty', 0))
        features.append(code_metrics.get('maintainability_index', 0))
        features.append(code_metrics.get('lines_of_code', 0))
        
        # Code quality features
        features.append(code_metrics.get('code_duplication', 0))
        features.append(code_metrics.get('test_coverage', 0))
        features.append(code_metrics.get('number_of_methods', 0))
        features.append(code_metrics.get('number_of_classes', 0))
        
        # Change frequency features
        features.append(code_metrics.get('commits_last_month', 0))
        features.append(code_metrics.get('authors_count', 0))
        features.append(code_metrics.get('file_age_days', 0))
        
        # Static analysis features
        features.append(code_metrics.get('security_warnings', 0))
        features.append(code_metrics.get('linting_violations', 0))
        features.append(code_metrics.get('type_errors', 0))
        
        self.feature_names = [
            'cyclomatic_complexity', 'halstead_difficulty', 'maintainability_index',
            'lines_of_code', 'code_duplication', 'test_coverage', 'number_of_methods',
            'number_of_classes', 'commits_last_month', 'authors_count', 'file_age_days',
            'security_warnings', 'linting_violations', 'type_errors'
        ]
        
        return features
    
    def add_training_example(self, code_metrics: Dict[str, Any], has_bug: bool):
        """Add a training example to the dataset."""
        features = self.extract_features(code_metrics)
        
        self.training_data.append({
            'features': features,
            'label': has_bug,
            'timestamp': datetime.now(),
            'metrics': code_metrics
        })
    
    def train_model(self) -> Dict[str, Any]:
        """Train the bug prediction model."""
        if not self.model or not self.training_data or len(self.training_data) < 10:
            return {"error": "Insufficient training data or model not available"}
        
        # Prepare training data
        X = [data['features'] for data in self.training_data]
        y = [data['label'] for data in self.training_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'training_samples': len(self.training_data),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        
        # Save model
        self._save_model()
        
        return metrics
    
    def predict_bug_probability(self, code_metrics: Dict[str, Any]) -> ModelPrediction:
        """Predict the probability of a bug in the code."""
        if not self.model or not self.scaler:
            return ModelPrediction(
                model_name="bug_prediction",
                prediction=0.5,
                confidence=0.0,
                features_used=[],
                metadata={"error": "Model not trained"}
            )
        
        features = self.extract_features(code_metrics)
        features_scaled = self.scaler.transform([features])
        
        # Get probability prediction
        probabilities = self.model.predict_proba(features_scaled)[0]
        bug_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Calculate confidence based on how far from 0.5 the prediction is
        confidence = abs(bug_probability - 0.5) * 2
        
        return ModelPrediction(
            model_name="bug_prediction",
            prediction=bug_probability,
            confidence=confidence,
            features_used=self.feature_names,
            metadata={
                "feature_values": features,
                "model_accuracy": getattr(self, 'last_accuracy', 0.0)
            }
        )
    
    def _save_model(self):
        """Save the trained model to disk."""
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'training_data_size': len(self.training_data),
                'saved_at': datetime.now()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"Error saving model: {e}")


class PerformancePredictor:
    """Predicts performance bottlenecks and optimization opportunities."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.model_path = model_path or ".cache/performance_model.pkl"
        self.training_data: List[Dict[str, Any]] = []
        
        if LinearRegression:
            self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one."""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
            else:
                self.model = LinearRegression()
                self.scaler = StandardScaler()
        except Exception:
            self.model = LinearRegression() if LinearRegression else None
            self.scaler = StandardScaler() if StandardScaler else None
    
    def extract_performance_features(self, code_metrics: Dict[str, Any]) -> List[float]:
        """Extract features relevant to performance prediction."""
        features = []
        
        # Code size features
        features.append(code_metrics.get('lines_of_code', 0))
        features.append(code_metrics.get('number_of_functions', 0))
        features.append(code_metrics.get('number_of_loops', 0))
        features.append(code_metrics.get('number_of_conditions', 0))
        
        # Complexity features
        features.append(code_metrics.get('cyclomatic_complexity', 0))
        features.append(code_metrics.get('nesting_depth', 0))
        features.append(code_metrics.get('halstead_volume', 0))
        
        # Data structure features
        features.append(code_metrics.get('number_of_variables', 0))
        features.append(code_metrics.get('number_of_arrays', 0))
        features.append(code_metrics.get('number_of_objects', 0))
        
        # I/O features
        features.append(code_metrics.get('file_operations', 0))
        features.append(code_metrics.get('network_operations', 0))
        features.append(code_metrics.get('database_operations', 0))
        
        return features
    
    def add_performance_example(self, code_metrics: Dict[str, Any], execution_time: float):
        """Add a performance training example."""
        features = self.extract_performance_features(code_metrics)
        
        self.training_data.append({
            'features': features,
            'execution_time': execution_time,
            'timestamp': datetime.now(),
            'metrics': code_metrics
        })
    
    def predict_execution_time(self, code_metrics: Dict[str, Any]) -> ModelPrediction:
        """Predict execution time for code."""
        if not self.model or not self.scaler:
            return ModelPrediction(
                model_name="performance_prediction",
                prediction=1.0,
                confidence=0.0,
                features_used=[],
                metadata={"error": "Model not trained"}
            )
        
        features = self.extract_performance_features(code_metrics)
        features_scaled = self.scaler.transform([features])
        
        predicted_time = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on training data variance
        confidence = 0.7  # Simplified confidence score
        
        return ModelPrediction(
            model_name="performance_prediction",
            prediction=predicted_time,
            confidence=confidence,
            features_used=[
                'lines_of_code', 'number_of_functions', 'number_of_loops',
                'number_of_conditions', 'cyclomatic_complexity', 'nesting_depth',
                'halstead_volume', 'number_of_variables', 'number_of_arrays',
                'number_of_objects', 'file_operations', 'network_operations',
                'database_operations'
            ],
            metadata={"feature_values": features}
        )


class AnomalyDetector:
    """Detects anomalies in code patterns and metrics."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.model_path = model_path or ".cache/anomaly_model.pkl"
        self.training_data: List[List[float]] = []
        
        if IsolationForest:
            self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one."""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
            else:
                self.model = IsolationForest(contamination=0.1, random_state=42)
                self.scaler = StandardScaler()
        except Exception:
            self.model = IsolationForest(contamination=0.1, random_state=42) if IsolationForest else None
            self.scaler = StandardScaler() if StandardScaler else None
    
    def add_normal_example(self, code_metrics: Dict[str, Any]):
        """Add a normal code example for training."""
        features = self._extract_anomaly_features(code_metrics)
        self.training_data.append(features)
    
    def _extract_anomaly_features(self, code_metrics: Dict[str, Any]) -> List[float]:
        """Extract features for anomaly detection."""
        features = []
        
        # Normalize all numeric features
        numeric_keys = [
            'cyclomatic_complexity', 'lines_of_code', 'number_of_functions',
            'halstead_difficulty', 'maintainability_index', 'code_duplication',
            'test_coverage', 'linting_violations', 'security_warnings'
        ]
        
        for key in numeric_keys:
            features.append(code_metrics.get(key, 0))
        
        return features
    
    def train_anomaly_detector(self) -> Dict[str, Any]:
        """Train the anomaly detection model."""
        if not self.model or len(self.training_data) < 20:
            return {"error": "Insufficient training data or model not available"}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.training_data)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Save model
        self._save_model()
        
        return {
            "training_samples": len(self.training_data),
            "model_type": "IsolationForest",
            "contamination_rate": 0.1
        }
    
    def detect_anomaly(self, code_metrics: Dict[str, Any]) -> ModelPrediction:
        """Detect if code metrics are anomalous."""
        if not self.model or not self.scaler:
            return ModelPrediction(
                model_name="anomaly_detection",
                prediction=0,
                confidence=0.0,
                features_used=[],
                metadata={"error": "Model not trained"}
            )
        
        features = self._extract_anomaly_features(code_metrics)
        features_scaled = self.scaler.transform([features])
        
        # Predict anomaly (-1 for anomaly, 1 for normal)
        prediction = self.model.predict(features_scaled)[0]
        
        # Get anomaly score (lower scores indicate more anomalous)
        anomaly_score = self.model.decision_function(features_scaled)[0]
        
        # Convert to probability-like score (0-1, where 1 is more anomalous)
        anomaly_probability = max(0, min(1, (0.5 - anomaly_score) / 0.5))
        
        confidence = abs(anomaly_score) / 0.5  # Normalized confidence
        
        return ModelPrediction(
            model_name="anomaly_detection",
            prediction=anomaly_probability,
            confidence=min(1.0, confidence),
            features_used=[
                'cyclomatic_complexity', 'lines_of_code', 'number_of_functions',
                'halstead_difficulty', 'maintainability_index', 'code_duplication',
                'test_coverage', 'linting_violations', 'security_warnings'
            ],
            metadata={
                "anomaly_score": anomaly_score,
                "is_anomaly": prediction == -1,
                "feature_values": features
            }
        )
    
    def _save_model(self):
        """Save the trained model to disk."""
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'training_data_size': len(self.training_data),
                'saved_at': datetime.now()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"Error saving anomaly model: {e}")


class MLModelManager:
    """Manages all machine learning models."""
    
    def __init__(self, models_dir: str = ".cache/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.bug_predictor = BugPredictionModel(str(self.models_dir / "bug_model.pkl"))
        self.performance_predictor = PerformancePredictor(str(self.models_dir / "perf_model.pkl"))
        self.anomaly_detector = AnomalyDetector(str(self.models_dir / "anomaly_model.pkl"))
        
        # Model metadata
        self.model_info = {
            "bug_prediction": {
                "description": "Predicts probability of bugs in code",
                "features": 14,
                "model_type": "RandomForestClassifier"
            },
            "performance_prediction": {
                "description": "Predicts execution time of code",
                "features": 13,
                "model_type": "LinearRegression"
            },
            "anomaly_detection": {
                "description": "Detects anomalous code patterns",
                "features": 9,
                "model_type": "IsolationForest"
            }
        }
    
    async def get_all_predictions(self, code_metrics: Dict[str, Any]) -> Dict[str, ModelPrediction]:
        """Get predictions from all available models."""
        predictions = {}
        
        # Bug prediction
        if np and RandomForestClassifier:
            try:
                predictions["bug_prediction"] = self.bug_predictor.predict_bug_probability(code_metrics)
            except Exception as e:
                predictions["bug_prediction"] = ModelPrediction(
                    model_name="bug_prediction",
                    prediction=0.5,
                    confidence=0.0,
                    features_used=[],
                    metadata={"error": str(e)}
                )
        
        # Performance prediction
        if np and LinearRegression:
            try:
                predictions["performance_prediction"] = self.performance_predictor.predict_execution_time(code_metrics)
            except Exception as e:
                predictions["performance_prediction"] = ModelPrediction(
                    model_name="performance_prediction",
                    prediction=1.0,
                    confidence=0.0,
                    features_used=[],
                    metadata={"error": str(e)}
                )
        
        # Anomaly detection
        if np and IsolationForest:
            try:
                predictions["anomaly_detection"] = self.anomaly_detector.detect_anomaly(code_metrics)
            except Exception as e:
                predictions["anomaly_detection"] = ModelPrediction(
                    model_name="anomaly_detection",
                    prediction=0.0,
                    confidence=0.0,
                    features_used=[],
                    metadata={"error": str(e)}
                )
        
        return predictions
    
    def add_training_data(self, data_type: str, code_metrics: Dict[str, Any], **kwargs):
        """Add training data to appropriate model."""
        if data_type == "bug_prediction" and "has_bug" in kwargs:
            self.bug_predictor.add_training_example(code_metrics, kwargs["has_bug"])
        
        elif data_type == "performance_prediction" and "execution_time" in kwargs:
            self.performance_predictor.add_performance_example(code_metrics, kwargs["execution_time"])
        
        elif data_type == "anomaly_detection":
            self.anomaly_detector.add_normal_example(code_metrics)
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Train all available models."""
        results = {}
        
        if np and RandomForestClassifier:
            results["bug_prediction"] = self.bug_predictor.train_model()
        
        if np and IsolationForest:
            results["anomaly_detection"] = self.anomaly_detector.train_anomaly_detector()
        
        return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            "available_models": list(self.model_info.keys()),
            "sklearn_available": np is not None,
            "models_directory": str(self.models_dir),
            "model_details": {}
        }
        
        for model_name, info in self.model_info.items():
            model_path = self.models_dir / f"{model_name.split('_')[0]}_model.pkl"
            status["model_details"][model_name] = {
                **info,
                "trained": model_path.exists(),
                "model_file": str(model_path)
            }
        
        return status


# Global ML model manager
_ml_manager: Optional[MLModelManager] = None


def get_ml_manager() -> MLModelManager:
    """Get or create global ML model manager."""
    global _ml_manager
    if _ml_manager is None:
        _ml_manager = MLModelManager()
    return _ml_manager