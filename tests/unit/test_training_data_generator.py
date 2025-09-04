"""
Test-Driven Development: Training Data Generator Tests
Sprint 2: Classification Intelligence
"""

import pytest
import json
from pathlib import Path
from typing import List, Tuple
import tempfile

from src.training.data_generator import (
    TrainingDataGenerator,
    DocumentTemplate,
    DatasetConfig,
    GeneratedDataset,
    LabeledDocument,
)
from src.classifiers.page_classifier import PageType
from src.processors.document_processor import Page


class TestTrainingDataGenerator:
    """Unit tests for training data generation system"""

    @pytest.fixture
    def generator(self):
        """Create a TrainingDataGenerator instance"""
        return TrainingDataGenerator()

    @pytest.fixture
    def dataset_config(self):
        """Create a sample dataset configuration"""
        return DatasetConfig(
            num_samples_per_class=10,
            page_types=[PageType.FINANCIAL, PageType.LEGAL, PageType.TECHNICAL],
            train_test_split=0.8,
            random_seed=42,
        )

    @pytest.mark.unit
    def test_generator_initialization(self, generator):
        """Test that TrainingDataGenerator initializes correctly"""
        assert generator is not None
        assert hasattr(generator, "templates")
        assert hasattr(generator, "generate_dataset")
        assert len(generator.templates) > 0

    @pytest.mark.unit
    def test_financial_template_generation(self, generator):
        """Test generation of financial document templates"""
        template = generator.get_template(PageType.FINANCIAL)
        
        assert isinstance(template, DocumentTemplate)
        assert template.page_type == PageType.FINANCIAL
        
        # Generate sample document
        document = template.generate()
        assert "balance" in document.lower() or "account" in document.lower()
        assert any(char in document for char in ["$", "€", "£"])

    @pytest.mark.unit
    def test_legal_template_generation(self, generator):
        """Test generation of legal document templates"""
        template = generator.get_template(PageType.LEGAL)
        document = template.generate()
        
        # Check for legal terminology
        legal_terms = ["agreement", "party", "terms", "conditions", "shall", "hereby"]
        assert any(term in document.lower() for term in legal_terms)

    @pytest.mark.unit
    def test_technical_template_generation(self, generator):
        """Test generation of technical document templates"""
        template = generator.get_template(PageType.TECHNICAL)
        document = template.generate()
        
        # Check for technical terminology
        tech_terms = ["api", "function", "parameter", "response", "endpoint", "method"]
        assert any(term in document.lower() for term in tech_terms)

    @pytest.mark.unit
    def test_dataset_generation(self, generator, dataset_config):
        """Test complete dataset generation"""
        dataset = generator.generate_dataset(dataset_config)
        
        assert isinstance(dataset, GeneratedDataset)
        assert len(dataset.training_samples) > 0
        assert len(dataset.test_samples) > 0
        
        # Check train/test split
        total_samples = len(dataset.training_samples) + len(dataset.test_samples)
        train_ratio = len(dataset.training_samples) / total_samples
        assert abs(train_ratio - dataset_config.train_test_split) < 0.1

    @pytest.mark.unit
    def test_balanced_dataset_generation(self, generator, dataset_config):
        """Test that generated dataset is balanced across classes"""
        dataset = generator.generate_dataset(dataset_config)
        
        # Count samples per class in training set
        class_counts = {}
        for sample in dataset.training_samples:
            class_counts[sample.label] = class_counts.get(sample.label, 0) + 1
        
        # Check balance
        counts = list(class_counts.values())
        assert max(counts) - min(counts) <= 1  # Allow for rounding differences

    @pytest.mark.unit
    def test_document_variability(self, generator):
        """Test that generated documents have variability"""
        template = generator.get_template(PageType.FINANCIAL)
        
        # Generate multiple documents
        documents = [template.generate() for _ in range(10)]
        
        # Check they're not all identical
        unique_docs = set(documents)
        assert len(unique_docs) > 5  # At least 50% unique

    @pytest.mark.unit
    def test_save_dataset_to_json(self, generator, dataset_config, tmp_path):
        """Test saving dataset to JSON format"""
        dataset = generator.generate_dataset(dataset_config)
        
        output_file = tmp_path / "dataset.json"
        generator.save_dataset(dataset, output_file)
        
        assert output_file.exists()
        
        # Load and verify
        with open(output_file, "r") as f:
            loaded_data = json.load(f)
        
        assert "training_samples" in loaded_data
        assert "test_samples" in loaded_data
        assert "metadata" in loaded_data

    @pytest.mark.unit
    def test_load_dataset_from_json(self, generator, dataset_config, tmp_path):
        """Test loading dataset from JSON format"""
        # Generate and save dataset
        dataset = generator.generate_dataset(dataset_config)
        output_file = tmp_path / "dataset.json"
        generator.save_dataset(dataset, output_file)
        
        # Load dataset
        loaded_dataset = generator.load_dataset(output_file)
        
        assert len(loaded_dataset.training_samples) == len(dataset.training_samples)
        assert len(loaded_dataset.test_samples) == len(dataset.test_samples)

    @pytest.mark.unit
    def test_augmentation_strategies(self, generator):
        """Test data augmentation strategies"""
        original_text = "This is a financial statement showing account balance."
        
        augmented_samples = generator.augment_text(original_text)
        
        assert len(augmented_samples) > 1
        assert original_text in augmented_samples
        
        # Check augmentation techniques
        augmentations = [s for s in augmented_samples if s != original_text]
        assert any(s != original_text for s in augmentations)  # Some variation

    @pytest.mark.unit
    def test_noise_injection(self, generator):
        """Test noise injection for robustness"""
        clean_text = "This is a clean document text."
        
        noisy_text = generator.add_noise(clean_text, noise_level=0.1)
        
        # Should have some differences but still be similar
        assert noisy_text != clean_text
        assert len(noisy_text) > 0

    @pytest.mark.unit
    def test_template_customization(self, generator):
        """Test customization of document templates"""
        custom_template = DocumentTemplate(
            page_type=PageType.FINANCIAL,
            patterns=["Invoice #{number}\nAmount: ${amount}"],
            required_keywords=["invoice", "payment"],
        )
        
        generator.add_template(custom_template)
        document = custom_template.generate()
        
        assert "invoice" in document.lower()
        assert "$" in document

    @pytest.mark.unit
    def test_multilingual_support(self, generator):
        """Test generation of multilingual documents"""
        # Test if generator can handle different languages
        template = generator.get_template(PageType.LEGAL)
        template.language = "es"  # Spanish
        
        document = template.generate()
        # Basic check - would need more sophisticated validation
        assert len(document) > 0

    @pytest.mark.unit
    def test_synthetic_metadata_generation(self, generator):
        """Test generation of synthetic metadata for documents"""
        dataset = generator.generate_dataset(
            DatasetConfig(num_samples_per_class=5, page_types=[PageType.FINANCIAL])
        )
        
        for sample in dataset.training_samples:
            assert sample.metadata is not None
            assert "creation_date" in sample.metadata
            assert "document_id" in sample.metadata
            assert "confidence_score" in sample.metadata