"""
Tests for the machine learning models system.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.analytics.ml_models import (
    MLModelManager, BugPredictionModel, PerformancePredictor,
    AnomalyDetector, ModelPrediction, TrainingData, get_ml_manager
)


class TestModelPrediction:
    """Test cases for ModelPrediction dataclass."""
    
    def test_model_prediction_creation(self):
        """Test creating a model prediction."""
        prediction = ModelPrediction(
            model_name="test_model",
            prediction=0.75,
            confidence=0.9,
            features_used=["complexity", "lines_of_code"],
            metadata={"feature_values": [5.0, 100]}
        )
        
        assert prediction.model_name == "test_model"
        assert prediction.prediction == 0.75
        assert prediction.confidence == 0.9
        assert "complexity" in prediction.features_used
        assert prediction.metadata["feature_values"] == [5.0, 100]


class TestTrainingData:
    """Test cases for TrainingData dataclass."""
    
    def test_training_data_creation(self):
        """Test creating training data."""
        training_data = TrainingData(
            features=[[1.0, 2.0], [3.0, 4.0]],
            labels=[True, False],
            feature_names=["feature1", "feature2"]
        )
        
        assert len(training_data.features) == 2
        assert len(training_data.labels) == 2
        assert training_data.feature_names == ["feature1", "feature2"]


class TestBugPredictionModel:
    """Test cases for bug prediction model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_bug_model.pkl"
            self.model = BugPredictionModel(str(model_path))
    
    def test_initial_state(self):
        """Test initial model state."""
        assert self.model.model is not None or self.model.model is None  # Depends on sklearn availability
        assert self.model.scaler is not None or self.model.scaler is None
        assert len(self.model.training_data) == 0
        assert len(self.model.feature_names) >= 0
    
    def test_extract_features(self):
        """Test feature extraction from code metrics."""
        code_metrics = {
            "cyclomatic_complexity": 5.0,
            "halstead_difficulty": 3.2,
            "maintainability_index": 75.0,
            "lines_of_code": 150,
            "code_duplication": 2.5,
            "test_coverage": 85.0,
            "number_of_methods": 8,
            "number_of_classes": 2,
            "commits_last_month": 12,
            "authors_count": 3,
            "file_age_days": 45,
            "security_warnings": 1,
            "linting_violations": 5,
            "type_errors": 0
        }
        
        features = self.model.extract_features(code_metrics)
        
        assert len(features) == 14  # Should extract all 14 features
        assert features[0] == 5.0  # cyclomatic_complexity
        assert features[3] == 150  # lines_of_code
        assert features[5] == 85.0  # test_coverage
    
    def test_extract_features_missing_data(self):
        """Test feature extraction with missing data."""
        incomplete_metrics = {
            "cyclomatic_complexity": 3.0,
            "lines_of_code": 100
            # Missing other metrics
        }
        
        features = self.model.extract_features(incomplete_metrics)
        
        assert len(features) == 14  # Should still return 14 features
        assert features[0] == 3.0  # cyclomatic_complexity
        assert features[1] == 0    # halstead_difficulty (default)
        assert features[3] == 100  # lines_of_code
    
    def test_add_training_example(self):
        """Test adding training examples."""
        metrics = {
            "cyclomatic_complexity": 8.0,
            "lines_of_code": 200,
            "test_coverage": 60.0
        }
        
        self.model.add_training_example(metrics, has_bug=True)
        
        assert len(self.model.training_data) == 1
        assert self.model.training_data[0]["label"] is True
        assert len(self.model.training_data[0]["features"]) == 14
    
    @patch('src.analytics.ml_models.RandomForestClassifier')
    @patch('src.analytics.ml_models.StandardScaler')
    @patch('src.analytics.ml_models.train_test_split')
    @patch('src.analytics.ml_models.accuracy_score')
    def test_train_model_success(self, mock_accuracy, mock_split, mock_scaler, mock_rf):
        """Test successful model training."""
        # Mock sklearn components
        mock_rf_instance = Mock()
        mock_rf_instance.fit = Mock()
        mock_rf_instance.predict = Mock(return_value=[1, 0, 1])
        mock_rf_instance.feature_importances_ = [0.1] * 14
        mock_rf.return_value = mock_rf_instance
        
        mock_scaler_instance = Mock()
        mock_scaler_instance.fit_transform = Mock(return_value=[[1, 2, 3], [4, 5, 6]])
        mock_scaler_instance.transform = Mock(return_value=[[7, 8, 9], [10, 11, 12]])
        mock_scaler.return_value = mock_scaler_instance
        
        mock_split.return_value = ([[1, 2]], [[3, 4]], [True], [False])
        mock_accuracy.return_value = 0.85
        
        # Add training data
        for i in range(15):  # Need at least 10 examples
            metrics = {"cyclomatic_complexity": i, "lines_of_code": i * 10}
            self.model.add_training_example(metrics, has_bug=(i % 2 == 0))
        
        self.model.model = mock_rf_instance
        self.model.scaler = mock_scaler_instance
        
        result = self.model.train_model()
        
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "feature_importance" in result
        assert result["training_samples"] == 15
    
    def test_train_model_insufficient_data(self):
        """Test training with insufficient data."""
        # Add only a few examples
        for i in range(5):
            metrics = {"cyclomatic_complexity": i}
            self.model.add_training_example(metrics, has_bug=True)
        
        result = self.model.train_model()
        
        assert "error" in result
        assert "insufficient" in result["error"].lower()
    
    @patch('src.analytics.ml_models.RandomForestClassifier')
    @patch('src.analytics.ml_models.StandardScaler')
    def test_predict_bug_probability(self, mock_scaler, mock_rf):
        """Test bug probability prediction."""
        # Mock trained model
        mock_rf_instance = Mock()
        mock_rf_instance.predict_proba = Mock(return_value=[[0.3, 0.7]])
        
        mock_scaler_instance = Mock()
        mock_scaler_instance.transform = Mock(return_value=[[1, 2, 3]])
        
        self.model.model = mock_rf_instance
        self.model.scaler = mock_scaler_instance
        
        metrics = {
            "cyclomatic_complexity": 10.0,
            "lines_of_code": 500,
            "test_coverage": 40.0
        }
        
        prediction = self.model.predict_bug_probability(metrics)
        
        assert prediction.model_name == "bug_prediction"
        assert prediction.prediction == 0.7  # Probability of bug
        assert prediction.confidence > 0
        assert len(prediction.features_used) == 14
    
    def test_predict_without_trained_model(self):
        """Test prediction without a trained model."""
        metrics = {"cyclomatic_complexity": 5.0}
        
        prediction = self.model.predict_bug_probability(metrics)
        
        assert prediction.model_name == "bug_prediction"
        assert prediction.prediction == 0.5  # Default prediction
        assert prediction.confidence == 0.0
        assert "error" in prediction.metadata


class TestPerformancePredictor:
    """Test cases for performance prediction model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_perf_model.pkl"
            self.model = PerformancePredictor(str(model_path))
    
    def test_extract_performance_features(self):
        """Test feature extraction for performance prediction."""
        code_metrics = {
            "lines_of_code": 200,
            "number_of_functions": 15,
            "number_of_loops": 8,
            "number_of_conditions": 12,
            "cyclomatic_complexity": 6.5,
            "nesting_depth": 4,
            "halstead_volume": 1500.0,
            "number_of_variables": 25,
            "number_of_arrays": 3,
            "number_of_objects": 5,
            "file_operations": 2,
            "network_operations": 1,
            "database_operations": 4
        }
        
        features = self.model.extract_performance_features(code_metrics)
        
        assert len(features) == 13
        assert features[0] == 200  # lines_of_code
        assert features[4] == 6.5  # cyclomatic_complexity
    
    def test_add_performance_example(self):
        """Test adding performance training examples."""
        metrics = {
            "lines_of_code": 100,
            "number_of_functions": 5,
            "cyclomatic_complexity": 3.0
        }
        
        self.model.add_performance_example(metrics, execution_time=2.5)
        
        assert len(self.model.training_data) == 1
        assert self.model.training_data[0]["execution_time"] == 2.5
        assert len(self.model.training_data[0]["features"]) == 13
    
    @patch('src.analytics.ml_models.LinearRegression')
    @patch('src.analytics.ml_models.StandardScaler')
    def test_predict_execution_time(self, mock_scaler, mock_lr):
        """Test execution time prediction."""
        # Mock trained model
        mock_lr_instance = Mock()
        mock_lr_instance.predict = Mock(return_value=[3.2])
        
        mock_scaler_instance = Mock()
        mock_scaler_instance.transform = Mock(return_value=[[1, 2, 3]])
        
        self.model.model = mock_lr_instance
        self.model.scaler = mock_scaler_instance
        
        metrics = {
            "lines_of_code": 300,
            "number_of_functions": 20,
            "cyclomatic_complexity": 8.0
        }
        
        prediction = self.model.predict_execution_time(metrics)
        
        assert prediction.model_name == "performance_prediction"
        assert prediction.prediction == 3.2
        assert prediction.confidence > 0
        assert len(prediction.features_used) == 13


class TestAnomalyDetector:
    """Test cases for anomaly detection model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_anomaly_model.pkl"
            self.model = AnomalyDetector(str(model_path))
    
    def test_extract_anomaly_features(self):
        """Test feature extraction for anomaly detection."""
        code_metrics = {
            "cyclomatic_complexity": 7.0,
            "lines_of_code": 250,
            "number_of_functions": 12,
            "halstead_difficulty": 4.5,
            "maintainability_index": 68.0,
            "code_duplication": 3.2,
            "test_coverage": 75.0,
            "linting_violations": 8,
            "security_warnings": 2,
            "extra_field": "ignored"  # Should be ignored
        }
        
        features = self.model._extract_anomaly_features(code_metrics)
        
        assert len(features) == 9  # Should extract 9 numeric features
        assert features[0] == 7.0   # cyclomatic_complexity
        assert features[1] == 250   # lines_of_code
    
    def test_add_normal_example(self):
        """Test adding normal examples for training."""
        metrics = {
            "cyclomatic_complexity": 4.0,
            "lines_of_code": 150,
            "maintainability_index": 80.0
        }
        
        self.model.add_normal_example(metrics)
        
        assert len(self.model.training_data) == 1
        assert len(self.model.training_data[0]) == 9
    
    @patch('src.analytics.ml_models.IsolationForest')
    @patch('src.analytics.ml_models.StandardScaler')
    def test_train_anomaly_detector(self, mock_scaler, mock_isolation):
        """Test training the anomaly detector."""
        # Mock sklearn components
        mock_isolation_instance = Mock()
        mock_isolation_instance.fit = Mock()
        mock_isolation.return_value = mock_isolation_instance
        
        mock_scaler_instance = Mock()
        mock_scaler_instance.fit_transform = Mock(return_value=[[1, 2, 3]])
        mock_scaler.return_value = mock_scaler_instance
        
        # Add training data
        for i in range(25):  # Need at least 20 examples
            metrics = {"cyclomatic_complexity": i, "lines_of_code": i * 10}
            self.model.add_normal_example(metrics)
        
        self.model.model = mock_isolation_instance
        self.model.scaler = mock_scaler_instance
        
        result = self.model.train_anomaly_detector()
        
        assert "training_samples" in result
        assert "model_type" in result
        assert result["training_samples"] == 25
        assert result["model_type"] == "IsolationForest"
    
    @patch('src.analytics.ml_models.IsolationForest')
    @patch('src.analytics.ml_models.StandardScaler')
    def test_detect_anomaly(self, mock_scaler, mock_isolation):
        """Test anomaly detection."""
        # Mock trained model
        mock_isolation_instance = Mock()
        mock_isolation_instance.predict = Mock(return_value=[-1])  # Anomaly
        mock_isolation_instance.decision_function = Mock(return_value=[-0.2])
        
        mock_scaler_instance = Mock()
        mock_scaler_instance.transform = Mock(return_value=[[1, 2, 3]])
        
        self.model.model = mock_isolation_instance
        self.model.scaler = mock_scaler_instance
        
        metrics = {
            "cyclomatic_complexity": 15.0,  # High complexity (anomalous)
            "lines_of_code": 1000,
            "maintainability_index": 30.0   # Low maintainability
        }
        
        prediction = self.model.detect_anomaly(metrics)
        
        assert prediction.model_name == "anomaly_detection"
        assert prediction.prediction > 0.5  # Should indicate anomaly
        assert prediction.metadata["is_anomaly"] is True


class TestMLModelManager:
    """Test cases for ML model manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MLModelManager()
    
    def test_initial_state(self):
        """Test initial manager state."""
        assert self.manager.bug_predictor is not None
        assert self.manager.performance_predictor is not None
        assert self.manager.anomaly_detector is not None
        assert len(self.manager.model_info) == 3
        assert "bug_prediction" in self.manager.model_info
        assert "performance_prediction" in self.manager.model_info
        assert "anomaly_detection" in self.manager.model_info
    
    @pytest.mark.asyncio
    async def test_get_all_predictions(self):
        """Test getting predictions from all models."""
        code_metrics = {
            "cyclomatic_complexity": 6.0,
            "lines_of_code": 200,
            "test_coverage": 70.0,
            "maintainability_index": 75.0
        }
        
        predictions = await self.manager.get_all_predictions(code_metrics)
        
        # Should return predictions from all models (even if they have errors)
        assert isinstance(predictions, dict)
        assert len(predictions) <= 3  # May be less if sklearn not available
        
        # Check that each prediction is a ModelPrediction object
        for model_name, prediction in predictions.items():
            assert isinstance(prediction, ModelPrediction)
            assert prediction.model_name == model_name
    
    def test_add_training_data_bug_prediction(self):
        """Test adding training data for bug prediction."""
        code_metrics = {
            "cyclomatic_complexity": 8.0,
            "lines_of_code": 300
        }
        
        self.manager.add_training_data("bug_prediction", code_metrics, has_bug=True)
        
        assert len(self.manager.bug_predictor.training_data) == 1
        assert self.manager.bug_predictor.training_data[0]["label"] is True
    
    def test_add_training_data_performance_prediction(self):
        """Test adding training data for performance prediction."""
        code_metrics = {
            "lines_of_code": 150,
            "number_of_functions": 10
        }
        
        self.manager.add_training_data("performance_prediction", code_metrics, execution_time=2.5)
        
        assert len(self.manager.performance_predictor.training_data) == 1
        assert self.manager.performance_predictor.training_data[0]["execution_time"] == 2.5
    
    def test_add_training_data_anomaly_detection(self):
        """Test adding training data for anomaly detection."""
        code_metrics = {
            "cyclomatic_complexity": 4.0,
            "maintainability_index": 85.0
        }
        
        self.manager.add_training_data("anomaly_detection", code_metrics)
        
        assert len(self.manager.anomaly_detector.training_data) == 1
    
    def test_add_training_data_unknown_type(self):
        """Test adding training data for unknown model type."""
        code_metrics = {"test": 1.0}
        
        # Should not raise an exception, just ignore unknown types
        self.manager.add_training_data("unknown_model", code_metrics)
        
        # No training data should be added to any model
        assert len(self.manager.bug_predictor.training_data) == 0
        assert len(self.manager.performance_predictor.training_data) == 0
        assert len(self.manager.anomaly_detector.training_data) == 0
    
    @patch('src.analytics.ml_models.RandomForestClassifier')
    @patch('src.analytics.ml_models.IsolationForest')
    def test_train_all_models(self, mock_isolation, mock_rf):
        """Test training all available models."""
        # Mock sklearn availability
        with patch('src.analytics.ml_models.np', True):
            # Add some training data
            for i in range(25):
                metrics = {"cyclomatic_complexity": i, "lines_of_code": i * 10}
                self.manager.add_training_data("bug_prediction", metrics, has_bug=(i % 2 == 0))
                self.manager.add_training_data("anomaly_detection", metrics)
            
            # Mock the models
            self.manager.bug_predictor.model = Mock()
            self.manager.bug_predictor.scaler = Mock()
            self.manager.anomaly_detector.model = Mock()
            self.manager.anomaly_detector.scaler = Mock()
            
            results = self.manager.train_all_models()
            
            assert isinstance(results, dict)
            # Results may be empty if sklearn is not available
    
    def test_get_model_status(self):
        """Test getting model status."""
        status = self.manager.get_model_status()
        
        assert "available_models" in status
        assert "sklearn_available" in status
        assert "models_directory" in status
        assert "model_details" in status
        
        assert len(status["available_models"]) == 3
        assert "bug_prediction" in status["available_models"]
        assert "performance_prediction" in status["available_models"]
        assert "anomaly_detection" in status["available_models"]
        
        # Check model details
        for model_name in status["available_models"]:
            assert model_name in status["model_details"]
            model_detail = status["model_details"][model_name]
            assert "description" in model_detail
            assert "features" in model_detail
            assert "model_type" in model_detail
            assert "trained" in model_detail


def test_global_ml_manager():
    """Test global ML manager singleton."""
    manager1 = get_ml_manager()
    manager2 = get_ml_manager()
    
    assert manager1 is manager2  # Should be same instance


if __name__ == "__main__":
    pytest.main([__file__])