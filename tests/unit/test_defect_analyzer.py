"""
Unit tests for utils.ai.defect_analyzer module
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.ai.defect_analyzer import (
    AdvancedDefectAnalyzer,
    DefectAnalysisPipeline,
    DefectDetector,
    analyze_defects,
)


class TestDefectDetector:
    """Tests for DefectDetector class"""

    def test_init_default_model(self):
        """Test default model initialization"""
        detector = DefectDetector()
        assert detector.model_name == "isolation_forest"
        assert detector.is_trained is False

    def test_init_isolation_forest_model(self):
        """Test isolation forest model initialization"""
        detector = DefectDetector(model_name="isolation_forest")
        assert detector.model_name == "isolation_forest"

    def test_init_kmeans_model(self):
        """Test KMeans model initialization"""
        detector = DefectDetector(model_name="kmeans")
        assert detector.model_name == "kmeans"

    def test_init_dbscan_model(self):
        """Test DBSCAN model initialization"""
        detector = DefectDetector(model_name="dbscan")
        assert detector.model_name == "dbscan"

    def test_init_invalid_model(self):
        """Test invalid model name raises ValueError"""
        with pytest.raises(ValueError, match="Неизвестная модель"):
            DefectDetector(model_name="invalid_model")

    def test_extract_features_2d_image(self):
        """Test feature extraction from 2D image"""
        detector = DefectDetector()
        image = np.random.rand(100, 100)
        features, positions = detector.extract_features(image, patch_size=16)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert len(positions) > 0
        assert features.shape[1] == 15  # 8 stats + 4 gradient + 2 laplacian + 1 entropy

    def test_extract_features_3d_image(self):
        """Test feature extraction from 3D image"""
        detector = DefectDetector()
        image = np.random.rand(100, 100, 3)
        features, positions = detector.extract_features(image, patch_size=16)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert features.shape[1] == 15

    def test_detect_defects_returns_structure(self):
        """Test detect_defects returns expected structure"""
        detector = DefectDetector()
        image = np.random.rand(100, 100)
        results = detector.detect_defects(image)

        assert "defects_count" in results
        assert "defects" in results
        assert "defect_mask" in results
        assert "scores" in results
        assert "positions" in results
        assert "summary" in results

    def test_detect_defects_2d_image(self):
        """Test defect detection on 2D image"""
        detector = DefectDetector()
        image = np.random.rand(100, 100)
        results = detector.detect_defects(image)

        assert isinstance(results["defects_count"], int)
        assert isinstance(results["defects"], list)
        assert isinstance(results["summary"], str)

    def test_detect_defects_3d_image(self):
        """Test defect detection on 3D image"""
        detector = DefectDetector()
        image = np.random.rand(100, 100, 3)
        results = detector.detect_defects(image)

        assert isinstance(results["defects_count"], int)
        assert isinstance(results["defects"], list)

    def test_detect_defects_kmeans_model(self):
        """Test defect detection with KMeans model"""
        detector = DefectDetector(model_name="kmeans")
        image = np.random.rand(100, 100)
        results = detector.detect_defects(image)

        assert "defects_count" in results
        assert isinstance(results["defects"], list)

    def test_detect_defects_dbscan_model(self):
        """Test defect detection with DBSCAN model"""
        detector = DefectDetector(model_name="dbscan")
        image = np.random.rand(100, 100)
        results = detector.detect_defects(image)

        assert "defects_count" in results

    def test_train_unsupervised(self):
        """Test unsupervised training"""
        detector = DefectDetector()
        images = [np.random.rand(50, 50) for _ in range(3)]
        detector.train(images)

        assert detector.is_trained is True

    def test_train_supervised(self):
        """Test supervised training"""
        detector = DefectDetector()
        images = [np.random.rand(50, 50) for _ in range(5)]
        # Each image produces ~25 feature vectors (patch extraction)
        # So we need 5 * ~25 = ~125 labels
        labels = [0, 1] * 62 + [0]  # 125 labels to match features
        detector.train(images, labels)

        assert detector.is_trained is True
        # After supervised training, model becomes RandomForestClassifier
        # but model_name attribute stays the same
        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(detector.model, RandomForestClassifier)

    def test_save_and_load_model(self):
        """Test model saving and loading"""
        detector = DefectDetector()
        images = [np.random.rand(50, 50) for _ in range(3)]
        detector.train(images)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            detector.save_model(temp_path)
            assert Path(temp_path).exists()

            new_detector = DefectDetector()
            new_detector.load_model(temp_path)

            assert new_detector.is_trained is True
            assert new_detector.model_name == detector.model_name
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDefectAnalysisPipeline:
    """Tests for DefectAnalysisPipeline class"""

    def test_init(self):
        """Test pipeline initialization"""
        pipeline = DefectAnalysisPipeline()
        assert pipeline.detector is not None
        assert pipeline.output_dir.exists()

    def test_init_with_db_manager(self):
        """Test pipeline initialization with DB manager"""
        mock_db = MagicMock()
        pipeline = DefectAnalysisPipeline(db_manager=mock_db)
        assert pipeline.db_manager == mock_db

    def test_analyze_image_with_mock(self):
        """Test image analysis with mocked image loading"""
        pipeline = DefectAnalysisPipeline()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            # Create a simple test image
            with patch("PIL.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_img.__array_interface__ = {
                    "shape": (100, 100),
                    "typestr": "|u1",
                    "data": np.random.randint(0, 255, (100, 100), dtype=np.uint8).tobytes(),
                }
                mock_open.return_value = mock_img

                results = pipeline.analyze_image(temp_path, save_results=False)

                assert "analysis_id" in results
                assert "timestamp" in results
                assert "image_path" in results
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_batch(self):
        """Test batch analysis"""
        pipeline = DefectAnalysisPipeline()

        # Create temp files
        temp_files = []
        for _ in range(3):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_files.append(f.name)

        try:
            with patch("PIL.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_img.__array_interface__ = {
                    "shape": (50, 50),
                    "typestr": "|u1",
                    "data": np.random.randint(0, 255, (50, 50), dtype=np.uint8).tobytes(),
                }
                mock_open.return_value = mock_img

                results = pipeline.analyze_batch(temp_files, save_results=False)

                assert len(results) == 3
                for result in results:
                    assert "analysis_id" in result or "error" in result
        finally:
            for path in temp_files:
                Path(path).unlink(missing_ok=True)

    def test_batch_analyze_with_folder(self):
        """Test folder-based batch analysis"""
        pipeline = DefectAnalysisPipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temp image files
            for i in range(2):
                Path(temp_dir, f"test_{i}.png").touch()

            with patch("PIL.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_img.__array_interface__ = {
                    "shape": (50, 50),
                    "typestr": "|u1",
                    "data": np.random.randint(0, 255, (50, 50), dtype=np.uint8).tobytes(),
                }
                mock_open.return_value = mock_img

                results = pipeline.batch_analyze(temp_dir)

                assert len(results) == 2


class TestAdvancedDefectAnalyzer:
    """Tests for AdvancedDefectAnalyzer class"""

    def test_init(self):
        """Test analyzer initialization"""
        analyzer = AdvancedDefectAnalyzer()
        assert analyzer.confidence_threshold == 0.7
        assert analyzer.output_dir.exists()

    def test_init_custom_threshold(self):
        """Test analyzer with custom threshold"""
        analyzer = AdvancedDefectAnalyzer(confidence_threshold=0.5)
        assert analyzer.confidence_threshold == 0.5

    def test_ensemble_detect(self):
        """Test ensemble detection"""
        analyzer = AdvancedDefectAnalyzer()
        image = np.random.rand(100, 100)
        results = analyzer.ensemble_detect(image)

        assert "defects" in results
        assert "defects_count" in results
        assert "if_defects_count" in results
        assert "km_defects_count" in results
        assert "ensemble" in results
        assert results["ensemble"] is True

    def test_analyze_with_stats(self):
        """Test analysis with statistics"""
        analyzer = AdvancedDefectAnalyzer()
        image = np.random.rand(100, 100)
        results = analyzer.analyze_with_stats(image)

        assert "defects" in results
        # The method returns ensemble result with additional stats
        # Check for key fields that should be present
        assert "defects_count" in results
        assert "ensemble" in results

    def test_combine_detections(self):
        """Test combining detections from multiple models"""
        analyzer = AdvancedDefectAnalyzer()

        defects1 = [{"x": 50, "y": 50, "width": 10, "height": 10, "confidence": 0.8, "type": "pit"}]
        defects2 = [{"x": 52, "y": 51, "width": 12, "height": 11, "confidence": 0.7, "type": "pit"}]

        combined = analyzer._combine_detections(defects1, defects2)

        assert len(combined) > 0
        assert combined[0]["confidence"] < 0.8  # Should be averaged


class TestAnalyzeDefectsFunction:
    """Tests for the analyze_defects convenience function"""

    def test_analyze_defects_with_mock(self):
        """Test analyze_defects function with mocked image"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            with patch("PIL.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_img.__array_interface__ = {
                    "shape": (100, 100),
                    "typestr": "|u1",
                    "data": np.random.randint(0, 255, (100, 100), dtype=np.uint8).tobytes(),
                }
                mock_open.return_value = mock_img

                with patch("utils.ai.defect_analyzer.DefectAnalysisPipeline") as mock_pipeline:
                    mock_instance = MagicMock()
                    mock_instance.analyze_image.return_value = {
                        "analysis_id": "test_123",
                        "defects_count": 0,
                    }
                    mock_pipeline.return_value = mock_instance

                    result = analyze_defects(temp_path, output_dir="/tmp/test")

                    assert result["analysis_id"] == "test_123"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDefectCategories:
    """Tests for defect categories and classification"""

    def test_defect_categories_structure(self):
        """Test that defect categories are properly defined"""
        detector = DefectDetector()
        expected_categories = {
            0: "pit",
            1: "hillock",
            2: "scratch",
            3: "particle",
            4: "crack",
            5: "normal",
        }
        assert detector.defect_categories == expected_categories

    def test_defect_type_classification(self):
        """Test that defects are classified into valid types"""
        detector = DefectDetector()
        image = np.random.rand(100, 100)
        results = detector.detect_defects(image)

        valid_types = {"pit", "hillock", "scratch", "particle", "crack", "normal", "unknown"}
        for defect in results["defects"]:
            assert defect["type"] in valid_types

    def test_defect_confidence_range(self):
        """Test that confidence values are in valid range"""
        detector = DefectDetector()
        image = np.random.rand(100, 100)
        results = detector.detect_defects(image)

        for defect in results["defects"]:
            assert 0 <= defect["confidence"] <= 1
