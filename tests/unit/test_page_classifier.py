"""
Test-Driven Development: Page Classifier Tests
Following TDD principles - RED phase: Write failing tests first
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import tempfile

from src.classifiers.page_classifier import (
    PageClassifier,
    PageType,
    ClassificationResult,
    ClassificationConfidence,
    FeatureExtractor,
    ClassifierModel,
)
from src.processors.document_processor import Page


class TestPageClassifier:
    """Unit tests for PageClassifier following TDD principles"""

    @pytest.fixture
    def classifier(self):
        """Create a PageClassifier instance for testing"""
        return PageClassifier()

    @pytest.fixture
    def sample_financial_page(self):
        """Create a sample financial page"""
        return Page(
            page_number=1,
            content="""
            Account Statement
            Account Number: 1234567890
            Balance: $10,500.00
            Transaction Date    Description    Amount
            2024-01-01         Deposit        $1,000.00
            2024-01-02         Withdrawal     -$500.00
            """,
            document_path=Path("/tmp/financial.pdf"),
        )

    @pytest.fixture
    def sample_legal_page(self):
        """Create a sample legal page"""
        return Page(
            page_number=1,
            content="""
            CONTRACT AGREEMENT
            This Agreement is entered into between Party A and Party B.
            
            TERMS AND CONDITIONS:
            1. The parties agree to the following terms...
            2. This contract shall be governed by the laws of...
            3. Any disputes shall be resolved through arbitration...
            
            SIGNATURES:
            Party A: _____________
            Party B: _____________
            """,
            document_path=Path("/tmp/legal.pdf"),
        )

    @pytest.fixture
    def sample_technical_page(self):
        """Create a sample technical page"""
        return Page(
            page_number=1,
            content="""
            API Reference Documentation
            
            GET /api/users
            Returns a list of all users.
            
            Parameters:
            - limit (int): Maximum number of users to return
            - offset (int): Number of users to skip
            
            Response:
            {
                "users": [...],
                "total": 100
            }
            """,
            document_path=Path("/tmp/technical.pdf"),
        )

    @pytest.mark.unit
    def test_classifier_initialization(self, classifier):
        """Test that PageClassifier initializes correctly"""
        assert classifier is not None
        assert hasattr(classifier, "supported_page_types")
        assert PageType.FINANCIAL in classifier.supported_page_types
        assert PageType.LEGAL in classifier.supported_page_types
        assert PageType.TECHNICAL in classifier.supported_page_types

    @pytest.mark.unit
    def test_feature_extraction(self, classifier, sample_financial_page):
        """Test feature extraction from a page"""
        features = classifier.extract_features(sample_financial_page)
        
        assert isinstance(features, dict)
        assert "keyword_features" in features
        assert "structure_features" in features
        assert "statistical_features" in features
        assert "layout_features" in features

    @pytest.mark.unit
    def test_classify_financial_page(self, classifier, sample_financial_page):
        """Test classification of a financial page"""
        result = classifier.classify(sample_financial_page)
        
        assert isinstance(result, ClassificationResult)
        assert result.page_type == PageType.FINANCIAL
        assert result.confidence > 0.7
        assert len(result.probabilities) > 0

    @pytest.mark.unit
    def test_classify_legal_page(self, classifier, sample_legal_page):
        """Test classification of a legal page"""
        result = classifier.classify(sample_legal_page)
        
        assert result.page_type == PageType.LEGAL
        assert result.confidence > 0.7

    @pytest.mark.unit
    def test_classify_technical_page(self, classifier, sample_technical_page):
        """Test classification of a technical page"""
        result = classifier.classify(sample_technical_page)
        
        assert result.page_type == PageType.TECHNICAL
        assert result.confidence > 0.7

    @pytest.mark.unit
    def test_batch_classification(self, classifier, sample_financial_page, 
                                 sample_legal_page, sample_technical_page):
        """Test batch classification of multiple pages"""
        pages = [sample_financial_page, sample_legal_page, sample_technical_page]
        results = classifier.classify_batch(pages)
        
        assert len(results) == 3
        assert all(isinstance(r, ClassificationResult) for r in results)
        assert results[0].page_type == PageType.FINANCIAL
        assert results[1].page_type == PageType.LEGAL
        assert results[2].page_type == PageType.TECHNICAL

    @pytest.mark.unit
    def test_confidence_calibration(self, classifier):
        """Test that confidence scores are well-calibrated"""
        # Create a page with mixed content
        mixed_page = Page(
            page_number=1,
            content="Some generic text without clear category indicators.",
            document_path=Path("/tmp/mixed.pdf"),
        )
        
        result = classifier.classify(mixed_page)
        
        # For unclear content, confidence should be lower
        assert result.confidence < 0.5
        assert result.confidence_level == ClassificationConfidence.LOW

    @pytest.mark.unit
    def test_keyword_based_features(self, classifier):
        """Test keyword-based feature extraction"""
        page = Page(
            page_number=1,
            content="balance account transaction deposit withdrawal statement",
            document_path=Path("/tmp/test.pdf"),
        )
        
        features = classifier.extract_features(page)
        keyword_features = features["keyword_features"]
        
        # Should have high financial keyword score
        assert keyword_features["financial_score"] > 0.5
        assert keyword_features["legal_score"] < 0.3
        assert keyword_features["technical_score"] < 0.3

    @pytest.mark.unit
    def test_structural_features(self, classifier):
        """Test structural feature extraction"""
        # Page with table-like structure
        table_page = Page(
            page_number=1,
            content="""
            Date       | Amount    | Description
            -----------|-----------|--------------
            2024-01-01 | $100.00   | Payment
            2024-01-02 | $200.00   | Deposit
            """,
            document_path=Path("/tmp/table.pdf"),
        )
        
        features = classifier.extract_features(table_page)
        structure_features = features["structure_features"]
        
        assert structure_features["has_table_structure"] is True
        assert structure_features["line_uniformity"] > 0.5

    @pytest.mark.unit
    def test_train_classifier(self, classifier):
        """Test training the classifier with labeled data"""
        # Create training data
        training_pages = [
            (Page(1, "financial balance account", Path("/tmp/f1.pdf")), PageType.FINANCIAL),
            (Page(1, "contract agreement terms", Path("/tmp/l1.pdf")), PageType.LEGAL),
            (Page(1, "API endpoint parameters", Path("/tmp/t1.pdf")), PageType.TECHNICAL),
        ]
        
        # Train the classifier
        classifier.train(training_pages)
        
        # Verify the model was updated
        assert classifier.model is not None
        assert classifier.is_trained is True

    @pytest.mark.unit
    def test_save_and_load_model(self, classifier, tmp_path):
        """Test saving and loading a trained model"""
        # Train a simple model
        training_data = [
            (Page(1, "financial content", Path("/tmp/f.pdf")), PageType.FINANCIAL),
        ]
        classifier.train(training_data)
        
        # Save the model
        model_path = tmp_path / "model.pkl"
        classifier.save_model(model_path)
        assert model_path.exists()
        
        # Load the model in a new classifier
        new_classifier = PageClassifier()
        new_classifier.load_model(model_path)
        
        assert new_classifier.is_trained is True

    @pytest.mark.unit
    def test_ensemble_classification(self, classifier):
        """Test ensemble classification using multiple models"""
        page = Page(
            page_number=1,
            content="Account balance transaction",
            document_path=Path("/tmp/test.pdf"),
        )
        
        # Enable ensemble mode
        classifier.use_ensemble = True
        result = classifier.classify(page)
        
        # Should combine predictions from multiple models
        assert result.ensemble_used is True
        assert len(result.model_predictions) > 1

    @pytest.mark.unit
    def test_classification_with_metadata(self, classifier):
        """Test classification using page metadata"""
        page = Page(
            page_number=1,
            content="General content",
            document_path=Path("/tmp/test.pdf"),
            metadata={"document_type": "bank_statement"}
        )
        
        result = classifier.classify(page)
        
        # Metadata should influence classification
        assert result.page_type == PageType.FINANCIAL
        assert result.metadata_influence > 0

    @pytest.mark.unit
    def test_performance_metrics(self, classifier):
        """Test classification performance metrics"""
        test_data = [
            (Page(1, "financial balance", Path("/tmp/f.pdf")), PageType.FINANCIAL),
            (Page(1, "legal contract", Path("/tmp/l.pdf")), PageType.LEGAL),
            (Page(1, "API documentation", Path("/tmp/t.pdf")), PageType.TECHNICAL),
        ]
        
        metrics = classifier.evaluate(test_data)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert metrics["accuracy"] >= 0.0 and metrics["accuracy"] <= 1.0