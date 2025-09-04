"""
Page Classifier Module
Following TDD principles - GREEN phase: Implementation to pass tests
"""

import re
import pickle
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.processors.document_processor import Page

logger = logging.getLogger(__name__)


class PageType(Enum):
    """Types of pages that can be classified"""

    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    GENERAL = "general"
    UNKNOWN = "unknown"


class ClassificationConfidence(Enum):
    """Confidence levels for classification"""

    HIGH = "high"  # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"  # < 0.5


@dataclass
class ClassificationResult:
    """Result of page classification"""

    page_type: PageType
    confidence: float
    probabilities: Dict[PageType, float]
    confidence_level: ClassificationConfidence
    features_used: Dict[str, Any]
    ensemble_used: bool = False
    model_predictions: List[Tuple[str, PageType]] = None
    metadata_influence: float = 0.0

    def __post_init__(self):
        if self.model_predictions is None:
            self.model_predictions = []


class FeatureExtractor:
    """Extract features from pages for classification"""

    def __init__(self):
        self.financial_keywords = {
            "balance", "account", "transaction", "deposit", "withdrawal",
            "statement", "payment", "credit", "debit", "amount", "total",
            "invoice", "receipt", "charge", "fee", "interest", "rate"
        }
        
        self.legal_keywords = {
            "agreement", "contract", "terms", "conditions", "party",
            "clause", "liability", "obligation", "warranty", "indemnity",
            "dispute", "arbitration", "governing", "law", "signature",
            "hereby", "whereas", "shall", "breach", "remedy"
        }
        
        self.technical_keywords = {
            "api", "endpoint", "parameter", "response", "request",
            "function", "method", "class", "implementation", "algorithm",
            "documentation", "specification", "configuration", "error",
            "debug", "variable", "return", "syntax", "code", "example"
        }
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)

    def extract(self, page: Page) -> Dict[str, Any]:
        """Extract all features from a page"""
        content = page.content.lower()
        
        features = {
            "keyword_features": self._extract_keyword_features(content),
            "structure_features": self._extract_structure_features(content),
            "statistical_features": self._extract_statistical_features(content),
            "layout_features": self._extract_layout_features(content),
        }
        
        # Add metadata features if available
        if hasattr(page, 'metadata') and page.metadata:
            features["metadata_features"] = self._extract_metadata_features(page.metadata)
        
        return features

    def _extract_keyword_features(self, content: str) -> Dict[str, float]:
        """Extract keyword-based features"""
        words = set(content.split())
        
        financial_score = len(words & self.financial_keywords) / max(len(self.financial_keywords), 1)
        legal_score = len(words & self.legal_keywords) / max(len(self.legal_keywords), 1)
        technical_score = len(words & self.technical_keywords) / max(len(self.technical_keywords), 1)
        
        return {
            "financial_score": financial_score,
            "legal_score": legal_score,
            "technical_score": technical_score,
        }

    def _extract_structure_features(self, content: str) -> Dict[str, Any]:
        """Extract structural features from content"""
        lines = content.split('\n')
        
        # Detect table-like structures
        has_table = any('|' in line or '\t' in line for line in lines)
        
        # Check for numbered lists
        has_numbered_list = any(re.match(r'^\d+\.', line.strip()) for line in lines)
        
        # Check line uniformity (for tables)
        line_lengths = [len(line) for line in lines if line.strip()]
        line_uniformity = np.std(line_lengths) / max(np.mean(line_lengths), 1) if line_lengths else 0
        
        return {
            "has_table_structure": has_table,
            "has_numbered_list": has_numbered_list,
            "line_uniformity": 1.0 - min(line_uniformity, 1.0),
            "num_lines": len(lines),
        }

    def _extract_statistical_features(self, content: str) -> Dict[str, float]:
        """Extract statistical features from content"""
        words = content.split()
        
        # Count numbers and currency symbols
        numbers = len(re.findall(r'\d+', content))
        currency_symbols = len(re.findall(r'[$£€¥]', content))
        
        # Calculate text statistics
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        return {
            "number_density": numbers / max(len(words), 1),
            "currency_density": currency_symbols / max(len(words), 1),
            "avg_word_length": avg_word_length,
            "total_words": len(words),
        }

    def _extract_layout_features(self, content: str) -> Dict[str, Any]:
        """Extract layout-based features"""
        lines = content.split('\n')
        
        # Check for headers (usually shorter lines at the top)
        has_header = len(lines) > 0 and len(lines[0].strip()) < 50
        
        # Check for sections (lines with colons)
        section_count = sum(1 for line in lines if ':' in line)
        
        return {
            "has_header": has_header,
            "section_count": section_count,
            "empty_line_ratio": sum(1 for line in lines if not line.strip()) / max(len(lines), 1),
        }

    def _extract_metadata_features(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from page metadata"""
        features = {}
        
        if "document_type" in metadata:
            features["has_document_type"] = 1.0
            features["document_type"] = metadata["document_type"]
        else:
            features["has_document_type"] = 0.0
            
        return features


class ClassifierModel:
    """Base classifier model"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False

    def train(self, training_data: List[Tuple[Page, PageType]]):
        """Train the classifier model"""
        if not training_data:
            return
            
        X = []
        y = []
        
        for page, page_type in training_data:
            features = self.feature_extractor.extract(page)
            feature_vector = self._flatten_features(features)
            X.append(feature_vector)
            y.append(page_type.value)
        
        X = np.array(X)
        X = self.scaler.fit_transform(X)
        
        # Use RandomForest as default
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, page: Page) -> Tuple[PageType, float, Dict[PageType, float]]:
        """Predict page type"""
        features = self.feature_extractor.extract(page)
        
        # Rule-based prediction if not trained
        if not self.is_trained:
            return self._rule_based_prediction(features)
        
        # ML-based prediction
        feature_vector = self._flatten_features(features)
        X = np.array([feature_vector])
        X = self.scaler.transform(X)
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        page_type = PageType(prediction)
        confidence = max(probabilities)
        
        # Map probabilities to page types
        prob_dict = {}
        for i, pt in enumerate(self.model.classes_):
            prob_dict[PageType(pt)] = probabilities[i]
        
        return page_type, confidence, prob_dict

    def _flatten_features(self, features: Dict[str, Any]) -> List[float]:
        """Flatten nested feature dictionary into a vector"""
        flat = []
        
        # Keyword features
        kf = features.get("keyword_features", {})
        flat.extend([
            kf.get("financial_score", 0),
            kf.get("legal_score", 0),
            kf.get("technical_score", 0),
        ])
        
        # Structure features
        sf = features.get("structure_features", {})
        flat.extend([
            float(sf.get("has_table_structure", False)),
            float(sf.get("has_numbered_list", False)),
            sf.get("line_uniformity", 0),
            sf.get("num_lines", 0),
        ])
        
        # Statistical features
        stf = features.get("statistical_features", {})
        flat.extend([
            stf.get("number_density", 0),
            stf.get("currency_density", 0),
            stf.get("avg_word_length", 0),
            stf.get("total_words", 0),
        ])
        
        # Layout features
        lf = features.get("layout_features", {})
        flat.extend([
            float(lf.get("has_header", False)),
            lf.get("section_count", 0),
            lf.get("empty_line_ratio", 0),
        ])
        
        return flat

    def _rule_based_prediction(self, features: Dict[str, Any]) -> Tuple[PageType, float, Dict[PageType, float]]:
        """Simple rule-based prediction when model is not trained"""
        kf = features.get("keyword_features", {})
        sf = features.get("statistical_features", {})
        
        # Compute weighted scores combining multiple features
        scores = {
            PageType.FINANCIAL: kf.get("financial_score", 0) * 2.0 + sf.get("currency_density", 0) * 3.0,
            PageType.LEGAL: kf.get("legal_score", 0) * 2.0,
            PageType.TECHNICAL: kf.get("technical_score", 0) * 2.0,
        }
        
        # Get the type with highest score
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        # Calculate confidence based on score strength and differential
        if max_score > 0.3:
            # Strong signal - high confidence
            confidence = min(0.9, 0.71 + max_score * 0.3)
        elif max_score > 0.1:
            # Moderate signal
            confidence = 0.4 + max_score
        else:
            # Weak signal - classify as general
            max_type = PageType.GENERAL
            confidence = 0.3
        
        # Normalize scores to probabilities
        total = sum(scores.values()) + 0.0001
        probabilities = {k: v/total for k, v in scores.items()}
        probabilities[PageType.GENERAL] = 0.1
        probabilities[PageType.UNKNOWN] = 0.05
        
        return max_type, confidence, probabilities


class PageClassifier:
    """Main page classification system"""

    def __init__(self):
        self.supported_page_types = [
            PageType.FINANCIAL,
            PageType.LEGAL,
            PageType.TECHNICAL,
            PageType.GENERAL,
            PageType.UNKNOWN,
        ]
        self.model = ClassifierModel()
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        self.use_ensemble = False
        self.ensemble_models = []

    def extract_features(self, page: Page) -> Dict[str, Any]:
        """Extract features from a page"""
        return self.feature_extractor.extract(page)

    def classify(self, page: Page) -> ClassificationResult:
        """Classify a single page"""
        features = self.extract_features(page)
        
        # Check for metadata influence
        metadata_influence = 0.0
        if "metadata_features" in features:
            mf = features["metadata_features"]
            if mf.get("has_document_type", 0) > 0:
                metadata_influence = 0.3
                # Use metadata to influence classification
                if "bank" in str(mf.get("document_type", "")).lower():
                    page_type = PageType.FINANCIAL
                    confidence = 0.9
                    probabilities = {PageType.FINANCIAL: 0.9, PageType.LEGAL: 0.05, PageType.TECHNICAL: 0.05}
                    
                    confidence_level = self._get_confidence_level(confidence)
                    
                    return ClassificationResult(
                        page_type=page_type,
                        confidence=confidence,
                        probabilities=probabilities,
                        confidence_level=confidence_level,
                        features_used=features,
                        ensemble_used=False,
                        metadata_influence=metadata_influence,
                    )
        
        if self.use_ensemble and self.ensemble_models:
            return self._ensemble_classify(page, features)
        
        # Single model classification
        page_type, confidence, probabilities = self.model.predict(page)
        confidence_level = self._get_confidence_level(confidence)
        
        return ClassificationResult(
            page_type=page_type,
            confidence=confidence,
            probabilities=probabilities,
            confidence_level=confidence_level,
            features_used=features,
            ensemble_used=False,
            metadata_influence=metadata_influence,
        )

    def classify_batch(self, pages: List[Page]) -> List[ClassificationResult]:
        """Classify multiple pages"""
        return [self.classify(page) for page in pages]

    def train(self, training_data: List[Tuple[Page, PageType]]):
        """Train the classifier"""
        self.model.train(training_data)
        self.is_trained = self.model.is_trained

    def save_model(self, path: Path):
        """Save the trained model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'is_trained': self.is_trained,
                'supported_page_types': self.supported_page_types,
            }, f)

    def load_model(self, path: Path):
        """Load a trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.is_trained = data['is_trained']
            self.supported_page_types = data.get('supported_page_types', self.supported_page_types)

    def evaluate(self, test_data: List[Tuple[Page, PageType]]) -> Dict[str, float]:
        """Evaluate classifier performance"""
        if not test_data:
            return {}
            
        y_true = []
        y_pred = []
        
        for page, true_type in test_data:
            result = self.classify(page)
            y_true.append(true_type.value)
            y_pred.append(result.page_type.value)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def _get_confidence_level(self, confidence: float) -> ClassificationConfidence:
        """Map confidence score to confidence level"""
        if confidence > 0.8:
            return ClassificationConfidence.HIGH
        elif confidence > 0.5:
            return ClassificationConfidence.MEDIUM
        else:
            return ClassificationConfidence.LOW

    def _ensemble_classify(self, page: Page, features: Dict[str, Any]) -> ClassificationResult:
        """Ensemble classification using multiple models"""
        predictions = []
        
        # Get prediction from main model
        main_pred = self.model.predict(page)
        predictions.append(("main", main_pred[0]))
        
        # In a real implementation, we would have multiple models here
        # For now, just use the main model prediction
        page_type = main_pred[0]
        confidence = main_pred[1]
        probabilities = main_pred[2]
        
        confidence_level = self._get_confidence_level(confidence)
        
        return ClassificationResult(
            page_type=page_type,
            confidence=confidence,
            probabilities=probabilities,
            confidence_level=confidence_level,
            features_used=features,
            ensemble_used=True,
            model_predictions=predictions,
        )