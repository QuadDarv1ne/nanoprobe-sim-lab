"""
Unit tests for utils.ai.machine_learning module
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.ai.machine_learning import (
    ImageAnalysisPredictor,
    ProjectMLPipeline,
    SSTVPredictor,
    SurfacePredictionModel,
)


class TestSurfacePredictionModel:
    """Tests for SurfacePredictionModel class"""

    def test_init(self):
        """Test model initialization"""
        model = SurfacePredictionModel()
        assert model.is_trained is False
        assert "regressor" in model.models
        assert "classifier" in model.models
        assert model.scaler is not None
        assert model.label_encoder is not None

    def test_prepare_features_2d(self):
        """Test feature preparation for 2D surface data"""
        model = SurfacePredictionModel()
        surface_data = np.random.rand(50, 50)
        features = model.prepare_features(surface_data)

        assert isinstance(features, np.ndarray)
        # Features: 10 stats + 4 geometry + 7 gradient = 21 (but rows/cols vary)
        assert features.shape[0] == 1
        assert features.shape[1] > 0

    def test_prepare_features_3d(self):
        """Test feature preparation for 3D surface data"""
        model = SurfacePredictionModel()
        # Note: prepare_features expects 2D data, convert 3D to 2D by taking mean
        surface_data_3d = np.random.rand(50, 50, 3)
        surface_data_2d = np.mean(surface_data_3d, axis=2)
        features = model.prepare_features(surface_data_2d)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1
        assert features.shape[1] > 0

    def test_prepare_features_different_sizes(self):
        """Test feature preparation with different surface sizes"""
        model = SurfacePredictionModel()

        sizes = [(50, 50), (60, 80), (100, 100)]
        for rows, cols in sizes:
            surface_data = np.random.rand(rows, cols)
            features = model.prepare_features(surface_data)
            assert features.shape[0] == 1
            assert features.shape[1] > 0

    def test_train_regression_model(self):
        """Test training regression model"""
        model = SurfacePredictionModel()

        n_samples = 100
        # Use consistent size for all samples
        X = np.array(
            [model.prepare_features(np.random.rand(50, 50)).flatten() for _ in range(n_samples)]
        )
        y = np.random.rand(n_samples)

        metrics = model.train_regression_model(X, y)

        assert isinstance(metrics, dict)
        assert "r2_score" in metrics
        assert "mse" in metrics
        assert model.is_trained is True

    def test_train_classification_model(self):
        """Test training classification model"""
        model = SurfacePredictionModel()

        n_samples = 100
        X = np.array(
            [model.prepare_features(np.random.rand(50, 50)).flatten() for _ in range(n_samples)]
        )
        y = np.random.randint(0, 3, n_samples)

        metrics = model.train_classification_model(X, y)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert model.is_trained is True

    def test_predict_regression(self):
        """Test regression prediction"""
        model = SurfacePredictionModel()

        n_samples = 100
        X = np.array(
            [model.prepare_features(np.random.rand(50, 50)).flatten() for _ in range(n_samples)]
        )
        y = np.random.rand(n_samples)
        model.train_regression_model(X, y)

        # Use same size as training data
        surface_data = np.random.rand(50, 50)
        predictions = model.predict(surface_data, "regression")

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1

    def test_predict_classification(self):
        """Test classification prediction"""
        model = SurfacePredictionModel()

        n_samples = 100
        X = np.array(
            [model.prepare_features(np.random.rand(50, 50)).flatten() for _ in range(n_samples)]
        )
        y = np.random.randint(0, 3, n_samples)
        model.train_classification_model(X, y)

        # Use same size as training data
        surface_data = np.random.rand(50, 50)
        predictions = model.predict(surface_data, "classification")

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1

    def test_save_and_load_model(self):
        """Test model saving and loading"""
        model = SurfacePredictionModel()

        n_samples = 100
        X = np.array(
            [model.prepare_features(np.random.rand(50, 50)).flatten() for _ in range(n_samples)]
        )
        y = np.random.rand(n_samples)
        model.train_regression_model(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            model.save_model(temp_path)
            assert Path(temp_path).exists()

            # load_model is not a static method, create new instance
            loaded_model = SurfacePredictionModel()
            loaded_model.load_model(temp_path)
            assert loaded_model.is_trained is True

            # Test prediction with loaded model
            surface_data = np.random.rand(50, 50)
            original_pred = model.predict(surface_data, "regression")
            loaded_pred = loaded_model.predict(surface_data, "regression")
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestImageAnalysisPredictor:
    """Tests for ImageAnalysisPredictor class"""

    def test_init(self):
        """Test predictor initialization"""
        predictor = ImageAnalysisPredictor()
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert len(predictor.feature_names) == 12

    def test_prepare_image_features_2d(self):
        """Test feature preparation for 2D image"""
        predictor = ImageAnalysisPredictor()
        image_data = np.random.rand(50, 50)
        features = predictor.prepare_image_features(image_data)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 12)

    def test_prepare_image_features_3d(self):
        """Test feature preparation for 3D image"""
        predictor = ImageAnalysisPredictor()
        image_data = np.random.rand(50, 50, 3)
        features = predictor.prepare_image_features(image_data)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 12)

    def test_train(self):
        """Test training the image analysis model"""
        predictor = ImageAnalysisPredictor()

        n_samples = 100
        X = np.random.rand(n_samples, 12)
        y = np.random.rand(n_samples)

        metrics = predictor.train(X, y)

        assert isinstance(metrics, dict)
        assert "r2_score" in metrics
        assert "mse" in metrics

    def test_predict_quality_score(self):
        """Test quality score prediction"""
        predictor = ImageAnalysisPredictor()

        n_samples = 100
        X = np.random.rand(n_samples, 12)
        y = np.random.rand(n_samples)
        predictor.train(X, y)

        # predict_quality_score expects image_data, not features
        image_data = np.random.rand(50, 50)
        quality = predictor.predict_quality_score(image_data)

        assert isinstance(quality, float)
        assert 0 <= quality <= 1


class TestSSTVPredictor:
    """Tests for SSTVPredictor class"""

    def test_init(self):
        """Test SSTV predictor initialization"""
        predictor = SSTVPredictor()
        assert predictor.quality_model is not None
        assert predictor.error_model is not None
        assert predictor.scaler is not None

    def test_prepare_signal_features(self):
        """Test SSTV signal feature preparation"""
        predictor = SSTVPredictor()
        signal_data = np.random.rand(1000)
        features = predictor.prepare_signal_features(signal_data)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 7)

    def test_train_quality_model(self):
        """Test training SSTV quality model"""
        predictor = SSTVPredictor()

        n_samples = 100
        X = np.random.rand(n_samples, 7)
        y = np.random.rand(n_samples)

        metrics = predictor.train_quality_model(X, y)

        assert isinstance(metrics, dict)
        assert "r2_score" in metrics
        assert "mse" in metrics

    def test_predict_decoding_quality(self):
        """Test SSTV decoding quality prediction"""
        predictor = SSTVPredictor()

        n_samples = 100
        X = np.random.rand(n_samples, 7)
        y = np.random.rand(n_samples)
        predictor.train_quality_model(X, y)

        signal_data = np.random.rand(1000)
        quality = predictor.predict_decoding_quality(signal_data)

        assert isinstance(quality, float)
        assert 0 <= quality <= 1


class TestProjectMLPipeline:
    """Tests for ProjectMLPipeline class"""

    def test_init(self):
        """Test pipeline initialization"""
        pipeline = ProjectMLPipeline()
        assert pipeline.surface_predictor is not None
        assert pipeline.image_predictor is not None
        assert pipeline.sstv_predictor is not None

    def test_train_all_models_empty(self):
        """Test training all models with empty data"""
        pipeline = ProjectMLPipeline()
        results = pipeline.train_all_models({})

        assert isinstance(results, dict)
        assert len(results) == 0

    def test_train_all_models_with_surface_data(self):
        """Test training with surface data"""
        pipeline = ProjectMLPipeline()

        n_samples = 50
        # Use consistent size for all samples
        X = np.array(
            [
                pipeline.surface_predictor.prepare_features(np.random.rand(50, 50)).flatten()
                for _ in range(n_samples)
            ]
        )
        training_data = {
            "surface": {
                "features": X,
                "targets": np.random.rand(n_samples),
            }
        }

        results = pipeline.train_all_models(training_data)

        assert isinstance(results, dict)
        assert "surface_regression" in results

    def test_make_predictions_empty(self):
        """Test making predictions with empty data"""
        pipeline = ProjectMLPipeline()
        predictions = pipeline.make_predictions({})

        assert isinstance(predictions, dict)
        assert len(predictions) == 0

    def test_make_predictions_with_surface(self):
        """Test making surface predictions"""
        pipeline = ProjectMLPipeline()

        # Train first
        n_samples = 50
        X = np.array(
            [
                pipeline.surface_predictor.prepare_features(np.random.rand(50, 50)).flatten()
                for _ in range(n_samples)
            ]
        )
        training_data = {
            "surface": {
                "features": X,
                "targets": np.random.rand(n_samples),
            }
        }
        pipeline.train_all_models(training_data)

        # Make predictions - pass surface data as 2D array
        input_data = {
            "surface": np.random.rand(50, 50),
        }

        predictions = pipeline.make_predictions(input_data)

        assert isinstance(predictions, dict)

    def test_make_predictions_with_image(self):
        """Test making image predictions"""
        pipeline = ProjectMLPipeline()

        # Train first
        n_samples = 50
        X = np.array(
            [
                pipeline.image_predictor.prepare_image_features(np.random.rand(50, 50)).flatten()
                for _ in range(n_samples)
            ]
        )
        training_data = {
            "image": {
                "features": X,
                "targets": np.random.rand(n_samples),
            }
        }
        pipeline.train_all_models(training_data)

        # Make predictions
        input_data = {
            "image": np.random.rand(50, 50),
        }

        predictions = pipeline.make_predictions(input_data)

        assert isinstance(predictions, dict)

    def test_make_predictions_with_sstv(self):
        """Test making SSTV predictions"""
        pipeline = ProjectMLPipeline()

        # Train first
        n_samples = 50
        X = np.array(
            [
                pipeline.sstv_predictor.prepare_signal_features(np.random.rand(1000)).flatten()
                for _ in range(n_samples)
            ]
        )
        training_data = {
            "sstv": {
                "features": X,
                "targets": np.random.rand(n_samples),
            }
        }
        pipeline.train_all_models(training_data)

        # Make predictions
        input_data = {
            "sstv_signal": np.random.rand(1000),
        }

        predictions = pipeline.make_predictions(input_data)

        assert isinstance(predictions, dict)
