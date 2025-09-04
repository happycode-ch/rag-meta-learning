"""
Training Data Generator Module
Sprint 2: Classification Intelligence
Following TDD principles - GREEN phase: Implementation to pass tests
"""

import random
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
import re

from src.classifiers.page_classifier import PageType
from src.processors.document_processor import Page


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    num_samples_per_class: int = 100
    page_types: List[PageType] = field(default_factory=lambda: [
        PageType.FINANCIAL, PageType.LEGAL, PageType.TECHNICAL
    ])
    train_test_split: float = 0.8
    random_seed: int = 42
    augmentation_factor: int = 3
    noise_level: float = 0.05


@dataclass
class LabeledDocument:
    """A labeled document for training"""
    text: str
    label: PageType
    metadata: Dict[str, Any]
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class GeneratedDataset:
    """A complete generated dataset"""
    training_samples: List[LabeledDocument]
    test_samples: List[LabeledDocument]
    metadata: Dict[str, Any]
    config: DatasetConfig


class DocumentTemplate:
    """Template for generating documents of a specific type"""
    
    def __init__(self, page_type: PageType, patterns: List[str] = None, 
                 required_keywords: List[str] = None, language: str = "en"):
        self.page_type = page_type
        self.patterns = patterns or []
        self.required_keywords = required_keywords or []
        self.language = language
        self._setup_default_patterns()
    
    def _setup_default_patterns(self):
        """Setup default patterns based on page type"""
        if not self.patterns:
            if self.page_type == PageType.FINANCIAL:
                self.patterns = [
                    "Account Statement\nAccount Number: {account_number}\nBalance: ${balance:.2f}",
                    "Transaction History\nDate: {date}\nDescription: {description}\nAmount: ${amount:.2f}",
                    "Invoice #{invoice_number}\nBill To: {customer}\nTotal Due: ${total:.2f}",
                    "Financial Report - {period}\nRevenue: ${revenue:.2f}\nExpenses: ${expenses:.2f}",
                ]
                self.required_keywords = ["balance", "account", "payment", "transaction"]
                
            elif self.page_type == PageType.LEGAL:
                self.patterns = [
                    "AGREEMENT\nThis Agreement is entered into between {party1} and {party2}.",
                    "TERMS AND CONDITIONS\n1. {term1}\n2. {term2}\n3. {term3}",
                    "CONTRACT for {service}\nEffective Date: {date}\nParties hereby agree to the following:",
                    "LEGAL NOTICE\nPursuant to {statute}, {party} hereby provides notice that {action}.",
                ]
                self.required_keywords = ["agreement", "terms", "contract", "shall", "hereby"]
                
            elif self.page_type == PageType.TECHNICAL:
                self.patterns = [
                    "API Documentation\nEndpoint: {endpoint}\nMethod: {method}\nParameters: {params}",
                    "Function: {function_name}\nDescription: {description}\nReturns: {return_type}",
                    "Technical Specification\nVersion: {version}\nRequirements:\n- {req1}\n- {req2}",
                    "Error Code: {code}\nDescription: {error_desc}\nResolution: {resolution}",
                ]
                self.required_keywords = ["api", "function", "parameter", "response", "method"]
    
    def generate(self) -> str:
        """Generate a document from the template"""
        # Select random pattern
        pattern = random.choice(self.patterns)
        
        # Fill in placeholders - handle both {placeholder} and #{placeholder} formats
        placeholders = re.findall(r'[#$]?\{([^}:]+)(?::[^}]+)?\}', pattern)
        values = {}
        
        for placeholder in placeholders:
            if 'number' in placeholder or 'account' in placeholder or 'invoice' in placeholder:
                values[placeholder] = f"{random.randint(1000000, 9999999)}"
            elif 'balance' in placeholder or 'amount' in placeholder or 'total' in placeholder:
                values[placeholder] = random.uniform(100, 100000)
            elif 'revenue' in placeholder or 'expenses' in placeholder:
                values[placeholder] = random.uniform(10000, 1000000)
            elif 'date' in placeholder:
                date = datetime.now() - timedelta(days=random.randint(0, 365))
                values[placeholder] = date.strftime("%Y-%m-%d")
            elif 'party' in placeholder or 'customer' in placeholder:
                values[placeholder] = f"Company {chr(65 + random.randint(0, 25))}"
            elif 'period' in placeholder:
                values[placeholder] = f"Q{random.randint(1, 4)} {datetime.now().year}"
            elif 'endpoint' in placeholder:
                values[placeholder] = f"/api/v{random.randint(1, 3)}/{random.choice(['users', 'data', 'reports'])}"
            elif 'method' in placeholder:
                values[placeholder] = random.choice(["GET", "POST", "PUT", "DELETE"])
            elif 'function' in placeholder:
                values[placeholder] = f"{random.choice(['get', 'set', 'process'])}_{random.choice(['Data', 'User', 'Report'])}"
            elif 'version' in placeholder:
                values[placeholder] = f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 99)}"
            elif 'code' in placeholder:
                values[placeholder] = f"ERR_{random.randint(100, 999)}"
            else:
                # Generic placeholder
                values[placeholder] = f"{placeholder.replace('_', ' ').title()} {random.randint(1, 100)}"
        
        # Format the document - handle both regular and special format placeholders
        document = pattern
        for key, value in values.items():
            # Handle regular placeholders
            document = document.replace(f"{{{key}}}", str(value))
            # Handle format specifiers
            document = re.sub(f"\\{{\\s*{key}\\s*:[^}}]+\\}}", str(value) if not isinstance(value, float) else f"{value:.2f}", document)
            # Handle special formats like #{placeholder}
            document = document.replace(f"#{{{key}}}", f"#{value}")
            document = document.replace(f"${{{key}}}", f"${value}" if not isinstance(value, float) else f"${value:.2f}")
        
        # Add some required keywords if missing
        for keyword in self.required_keywords[:2]:
            if keyword not in document.lower():
                document += f"\n{keyword.title()}: Additional information"
        
        return document


class TrainingDataGenerator:
    """Main training data generation system"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
        self.random = random.Random()
    
    def _initialize_templates(self):
        """Initialize default templates for each page type"""
        for page_type in [PageType.FINANCIAL, PageType.LEGAL, PageType.TECHNICAL]:
            self.templates[page_type] = DocumentTemplate(page_type)
    
    def get_template(self, page_type: PageType) -> DocumentTemplate:
        """Get template for a specific page type"""
        if page_type not in self.templates:
            self.templates[page_type] = DocumentTemplate(page_type)
        return self.templates[page_type]
    
    def add_template(self, template: DocumentTemplate):
        """Add a custom template"""
        self.templates[template.page_type] = template
    
    def generate_dataset(self, config: DatasetConfig) -> GeneratedDataset:
        """Generate a complete dataset based on configuration"""
        self.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        all_samples = []
        
        # Generate samples for each page type
        for page_type in config.page_types:
            template = self.get_template(page_type)
            
            for _ in range(config.num_samples_per_class):
                # Generate base document
                text = template.generate()
                
                # Create metadata
                metadata = {
                    "creation_date": datetime.now().isoformat(),
                    "document_id": str(uuid.uuid4()),
                    "confidence_score": random.uniform(0.7, 1.0),
                    "page_type": page_type.value,
                    "language": template.language,
                }
                
                sample = LabeledDocument(
                    text=text,
                    label=page_type,
                    metadata=metadata
                )
                all_samples.append(sample)
        
        # Shuffle samples
        self.random.shuffle(all_samples)
        
        # Split into train/test
        split_idx = int(len(all_samples) * config.train_test_split)
        training_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
        
        # Create dataset metadata
        dataset_metadata = {
            "total_samples": len(all_samples),
            "training_samples": len(training_samples),
            "test_samples": len(test_samples),
            "page_types": [pt.value for pt in config.page_types],
            "generation_date": datetime.now().isoformat(),
            "config": asdict(config),
        }
        
        return GeneratedDataset(
            training_samples=training_samples,
            test_samples=test_samples,
            metadata=dataset_metadata,
            config=config
        )
    
    def augment_text(self, text: str, num_augmentations: int = 3) -> List[str]:
        """Apply data augmentation strategies to text"""
        augmented = [text]  # Include original
        
        augmentation_strategies = [
            self._synonym_replacement,
            self._random_insertion,
            self._random_swap,
            self._paraphrase,
        ]
        
        for _ in range(num_augmentations):
            strategy = self.random.choice(augmentation_strategies)
            augmented_text = strategy(text)
            if augmented_text != text:
                augmented.append(augmented_text)
        
        return augmented
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms"""
        words = text.split()
        if not words:
            return text
            
        # Simple synonym replacement (in production, use NLTK or spaCy)
        synonyms = {
            "balance": "amount",
            "account": "profile",
            "agreement": "contract",
            "api": "interface",
            "function": "method",
        }
        
        num_replacements = max(1, len(words) // 10)
        for _ in range(num_replacements):
            idx = self.random.randint(0, len(words) - 1)
            word_lower = words[idx].lower()
            if word_lower in synonyms:
                words[idx] = synonyms[word_lower].capitalize() if words[idx][0].isupper() else synonyms[word_lower]
        
        return " ".join(words)
    
    def _random_insertion(self, text: str) -> str:
        """Randomly insert words"""
        words = text.split()
        if not words:
            return text
            
        insertions = ["the", "a", "an", "this", "that", "these", "those"]
        idx = self.random.randint(0, len(words))
        words.insert(idx, self.random.choice(insertions))
        
        return " ".join(words)
    
    def _random_swap(self, text: str) -> str:
        """Randomly swap adjacent words"""
        words = text.split()
        if len(words) < 2:
            return text
            
        idx = self.random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return " ".join(words)
    
    def _paraphrase(self, text: str) -> str:
        """Simple paraphrasing"""
        # In production, use a paraphrasing model
        replacements = {
            "This is": "Here is",
            "shows": "displays",
            "contains": "includes",
            "provides": "offers",
        }
        
        result = text
        for old, new in replacements.items():
            if old in result:
                result = result.replace(old, new, 1)
                break
        
        return result
    
    def add_noise(self, text: str, noise_level: float = 0.05) -> str:
        """Add noise to text for robustness"""
        if noise_level <= 0:
            return text
            
        chars = list(text)
        num_changes = max(1, int(len(chars) * noise_level))
        
        for _ in range(num_changes):
            change_type = self.random.choice(['typo', 'case', 'punctuation'])
            
            if change_type == 'typo' and len(chars) > 0:
                # Random typo
                idx = self.random.randint(0, len(chars) - 1)
                if chars[idx].isalpha():
                    chars[idx] = chr(ord('a') + self.random.randint(0, 25))
                    
            elif change_type == 'case' and len(chars) > 0:
                # Random case change
                idx = self.random.randint(0, len(chars) - 1)
                if chars[idx].isalpha():
                    chars[idx] = chars[idx].swapcase()
                    
            elif change_type == 'punctuation':
                # Add/remove punctuation
                if self.random.choice([True, False]) and len(chars) > 0:
                    idx = self.random.randint(0, len(chars) - 1)
                    chars.insert(idx, self.random.choice([',', '.', '!', '?']))
        
        return "".join(chars)
    
    def save_dataset(self, dataset: GeneratedDataset, output_path: Path):
        """Save dataset to JSON format"""
        # Convert PageType enums in config to strings
        config_dict = asdict(dataset.config)
        if 'page_types' in config_dict:
            config_dict['page_types'] = [
                pt.value if isinstance(pt, PageType) else pt 
                for pt in config_dict['page_types']
            ]
        
        data = {
            "training_samples": [
                {
                    "text": sample.text,
                    "label": sample.label.value,
                    "metadata": sample.metadata,
                    "document_id": sample.document_id,
                }
                for sample in dataset.training_samples
            ],
            "test_samples": [
                {
                    "text": sample.text,
                    "label": sample.label.value,
                    "metadata": sample.metadata,
                    "document_id": sample.document_id,
                }
                for sample in dataset.test_samples
            ],
            "metadata": {**dataset.metadata, "config": config_dict},
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_dataset(self, input_path: Path) -> GeneratedDataset:
        """Load dataset from JSON format"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to LabeledDocument objects
        training_samples = [
            LabeledDocument(
                text=sample['text'],
                label=PageType(sample['label']),
                metadata=sample['metadata'],
                document_id=sample['document_id']
            )
            for sample in data['training_samples']
        ]
        
        test_samples = [
            LabeledDocument(
                text=sample['text'],
                label=PageType(sample['label']),
                metadata=sample['metadata'],
                document_id=sample['document_id']
            )
            for sample in data['test_samples']
        ]
        
        # Reconstruct config from metadata
        config_data = data['metadata'].get('config', {})
        config = DatasetConfig(**{
            k: v for k, v in config_data.items() 
            if k in DatasetConfig.__dataclass_fields__
        })
        
        return GeneratedDataset(
            training_samples=training_samples,
            test_samples=test_samples,
            metadata=data['metadata'],
            config=config
        )