#!/usr/bin/env python
"""
Generate training data for page classification
Sprint 2: Classification Intelligence
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.data_generator import TrainingDataGenerator, DatasetConfig
from src.classifiers.page_classifier import PageType


def main():
    """Generate training dataset"""
    print("🚀 Generating Training Dataset for Page Classification")
    print("=" * 60)
    
    # Initialize generator
    generator = TrainingDataGenerator()
    
    # Configure dataset
    config = DatasetConfig(
        num_samples_per_class=100,  # 100 samples per class
        page_types=[PageType.FINANCIAL, PageType.LEGAL, PageType.TECHNICAL],
        train_test_split=0.8,
        random_seed=42
    )
    
    print(f"Configuration:")
    print(f"  - Samples per class: {config.num_samples_per_class}")
    print(f"  - Page types: {[pt.value for pt in config.page_types]}")
    print(f"  - Train/test split: {config.train_test_split:.0%}")
    print()
    
    # Generate dataset
    print("Generating samples...")
    dataset = generator.generate_dataset(config)
    
    print(f"✅ Generated {len(dataset.training_samples)} training samples")
    print(f"✅ Generated {len(dataset.test_samples)} test samples")
    
    # Save dataset
    output_path = Path("data/training/classification_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generator.save_dataset(dataset, output_path)
    print(f"\n💾 Dataset saved to {output_path}")
    
    # Display sample statistics
    print("\n📊 Dataset Statistics:")
    print("-" * 40)
    
    # Training set distribution
    train_dist = {}
    for sample in dataset.training_samples:
        train_dist[sample.label.value] = train_dist.get(sample.label.value, 0) + 1
    
    print("Training set distribution:")
    for label, count in train_dist.items():
        print(f"  - {label}: {count} samples")
    
    # Test set distribution
    test_dist = {}
    for sample in dataset.test_samples:
        test_dist[sample.label.value] = test_dist.get(sample.label.value, 0) + 1
    
    print("\nTest set distribution:")
    for label, count in test_dist.items():
        print(f"  - {label}: {count} samples")
    
    print("\n✨ Dataset generation complete!")


if __name__ == "__main__":
    main()