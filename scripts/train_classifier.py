#!/usr/bin/env python
"""
ML Training Pipeline with Cross-Validation
Sprint 2: Classification Intelligence
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.training.data_generator import TrainingDataGenerator
from src.classifiers.page_classifier import PageType, PageClassifier, FeatureExtractor
from src.processors.document_processor import Page


class MLTrainingPipeline:
    """Advanced ML training pipeline with cross-validation"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
        self.models = {}
        self.best_model = None
        self.results = {}
        
    def load_dataset(self, dataset_path: Path) -> Tuple[List, List, List, List]:
        """Load dataset from JSON"""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        # Process training samples
        for sample in data['training_samples']:
            X_train.append(sample['text'])
            y_train.append(sample['label'])
        
        # Process test samples
        for sample in data['test_samples']:
            X_test.append(sample['text'])
            y_test.append(sample['label'])
        
        return X_train, y_train, X_test, y_test
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract features from text samples"""
        features = []
        
        for text in texts:
            # Create temporary page object
            page = Page(
                page_number=1,
                content=text,
                document_path=Path("/tmp/temp.txt")
            )
            
            # Extract structured features
            page_features = self.feature_extractor.extract(page)
            
            # Flatten features
            flat_features = self._flatten_features(page_features)
            features.append(flat_features)
        
        # Convert to numpy array
        structured_features = np.array(features)
        
        # Extract TF-IDF features
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        # Combine features
        combined_features = np.hstack([structured_features, tfidf_features])
        
        return combined_features
    
    def _flatten_features(self, features: Dict[str, Any]) -> List[float]:
        """Flatten feature dictionary"""
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
    
    def train_models(self, X_train: np.ndarray, y_train: List[str]):
        """Train multiple models with cross-validation"""
        print("\n🔬 Training Models with Cross-Validation")
        print("=" * 60)
        
        # Define models to train
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "NaiveBayes": MultinomialNB(),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        }
        
        # Stratified K-Fold for balanced folds
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Make features non-negative for NaiveBayes
        X_train_nonneg = X_train_scaled - X_train_scaled.min() + 1e-10
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use appropriate features for each model
            if name == "NaiveBayes":
                X_for_model = X_train_nonneg
            else:
                X_for_model = X_train_scaled
            
            # Cross-validation
            scores = cross_val_score(model, X_for_model, y_train, cv=cv, scoring='accuracy')
            
            # Fit on full training set
            model.fit(X_for_model, y_train)
            
            self.models[name] = model
            self.results[name] = {
                "cv_scores": scores,
                "cv_mean": scores.mean(),
                "cv_std": scores.std(),
            }
            
            print(f"  Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    def create_ensemble(self, X_train: np.ndarray, y_train: List[str]):
        """Create ensemble classifier"""
        print("\n🎯 Creating Ensemble Classifier")
        print("-" * 40)
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Select best models for ensemble
        best_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['cv_mean'],
            reverse=True
        )[:3]
        
        print("Selected models for ensemble:")
        for model_name, results in best_models:
            print(f"  - {model_name}: {results['cv_mean']:.3f}")
        
        # Create voting classifier
        estimators = [
            (name, self.models[name])
            for name, _ in best_models
        ]
        
        self.ensemble = VotingClassifier(estimators=estimators, voting='soft')
        self.ensemble.fit(X_train_scaled, y_train)
        
        # Cross-validation on ensemble
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ensemble_scores = cross_val_score(self.ensemble, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        print(f"\nEnsemble cross-validation accuracy: {ensemble_scores.mean():.3f} (+/- {ensemble_scores.std():.3f})")
        
        self.results['Ensemble'] = {
            "cv_scores": ensemble_scores,
            "cv_mean": ensemble_scores.mean(),
            "cv_std": ensemble_scores.std(),
        }
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: List[str]):
        """Evaluate all models on test set"""
        print("\n📈 Test Set Evaluation")
        print("=" * 60)
        
        X_test_scaled = self.scaler.transform(X_test)
        X_test_nonneg = X_test_scaled - X_test_scaled.min() + 1e-10
        
        test_results = {}
        
        for name, model in self.models.items():
            if name == "NaiveBayes":
                X_for_model = X_test_nonneg
            else:
                X_for_model = X_test_scaled
            
            y_pred = model.predict(X_for_model)
            accuracy = accuracy_score(y_test, y_pred)
            
            test_results[name] = accuracy
            print(f"{name}: {accuracy:.3f}")
        
        # Evaluate ensemble
        y_pred_ensemble = self.ensemble.predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        test_results['Ensemble'] = ensemble_accuracy
        print(f"Ensemble: {ensemble_accuracy:.3f}")
        
        # Detailed report for best model
        best_model_name = max(test_results, key=test_results.get)
        print(f"\n🏆 Best Model: {best_model_name} ({test_results[best_model_name]:.3f})")
        
        if best_model_name == 'Ensemble':
            y_pred_best = y_pred_ensemble
        else:
            best_model = self.models[best_model_name]
            if best_model_name == "NaiveBayes":
                y_pred_best = best_model.predict(X_test_nonneg)
            else:
                y_pred_best = best_model.predict(X_test_scaled)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_best))
        
        return test_results
    
    def save_best_model(self, output_path: Path):
        """Save the best model"""
        # Determine best model
        test_accuracies = {
            name: result.get('test_accuracy', result.get('cv_mean', 0))
            for name, result in self.results.items()
        }
        
        best_model_name = max(test_accuracies, key=test_accuracies.get)
        
        if best_model_name == 'Ensemble':
            best_model = self.ensemble
        else:
            best_model = self.models[best_model_name]
        
        # Save model and metadata
        model_data = {
            'model': best_model,
            'model_name': best_model_name,
            'scaler': self.scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_extractor': self.feature_extractor,
            'results': self.results,
            'training_date': datetime.now().isoformat(),
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Best model ({best_model_name}) saved to {output_path}")


def main():
    """Main training pipeline"""
    print("🚀 ML Training Pipeline for Page Classification")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    
    # Load dataset
    dataset_path = Path("data/training/classification_dataset.json")
    if not dataset_path.exists():
        print("❌ Dataset not found. Please run generate_training_data.py first.")
        return
    
    print(f"📁 Loading dataset from {dataset_path}")
    X_train_text, y_train, X_test_text, y_test = pipeline.load_dataset(dataset_path)
    print(f"  Training samples: {len(X_train_text)}")
    print(f"  Test samples: {len(X_test_text)}")
    
    # Extract features
    print("\n🔧 Extracting features...")
    X_train = pipeline.extract_features(X_train_text)
    X_test = pipeline.extract_features(X_test_text)
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    # Train models
    pipeline.train_models(X_train, y_train)
    
    # Create ensemble
    pipeline.create_ensemble(X_train, y_train)
    
    # Evaluate on test set
    test_results = pipeline.evaluate_on_test_set(X_test, y_test)
    
    # Update results with test accuracies
    for name, accuracy in test_results.items():
        if name in pipeline.results:
            pipeline.results[name]['test_accuracy'] = accuracy
    
    # Save best model
    output_path = Path("models/page_classifier_ml.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_best_model(output_path)
    
    print("\n✨ Training complete!")
    
    # Check if we meet the 85% accuracy target
    best_accuracy = max(test_results.values())
    if best_accuracy >= 0.85:
        print(f"✅ Target accuracy achieved: {best_accuracy:.1%} >= 85%")
    else:
        print(f"⚠️ Target accuracy not met: {best_accuracy:.1%} < 85%")
        print("   Consider: more training data, feature engineering, or hyperparameter tuning")


if __name__ == "__main__":
    main()