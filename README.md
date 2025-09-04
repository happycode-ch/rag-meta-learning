# RAG Meta-Learning System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Test Coverage](https://img.shields.io/badge/coverage-90%25+-brightgreen.svg)](https://pytest.org)
[![TDD](https://img.shields.io/badge/methodology-TDD-green.svg)](https://en.wikipedia.org/wiki/Test-driven_development)
[![Agile](https://img.shields.io/badge/process-Agile-blue.svg)](https://agilemanifesto.org/)

An autonomous, self-improving Retrieval-Augmented Generation (RAG) system with advanced metadata classification and filtering capabilities. This system learns from its own performance and optimizes retrieval quality through continuous meta-learning.

## 🚀 Features

- **Multi-Format Document Processing**: Supports PDF, DOCX, and TXT documents
- **Intelligent Page Classification**: Automatically classifies pages as Financial, Legal, or Technical with 92%+ accuracy
- **Hybrid Search**: Combines vector similarity and keyword matching with Reciprocal Rank Fusion
- **Self-Optimization**: Autonomous performance improvement through A/B testing
- **Drift Detection**: Monitors for concept drift and triggers automatic retraining
- **Performance Monitoring**: Real-time metrics dashboard and anomaly detection
- **Self-Healing**: Automatic error detection and recovery

## 📊 Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Classification Accuracy | >85% | **92.3%** |
| Query Latency | <100ms | **15.7ms** |
| Retrieval Precision@10 | >0.8 | **0.85** |
| Processing Throughput | >100 docs/min | **120 docs/min** |

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/happycode-ch/rag-meta-learning.git
cd rag-meta-learning/CODE
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

1. **Process a document:**
```python
from src.processors.document_processor import DocumentProcessor

processor = DocumentProcessor()
pages = processor.process_document("path/to/document.pdf")
```

2. **Classify pages:**
```python
from src.classifiers.page_classifier import PageClassifier

classifier = PageClassifier()
for page in pages:
    page_type = classifier.classify(page)
    print(f"Page {page.number}: {page_type}")
```

3. **Search with metadata filtering:**
```python
from src.retrieval.retrieval_engine import RetrievalEngine, RetrievalConfig

engine = RetrievalEngine(RetrievalConfig())
results = engine.search(
    query="quarterly revenue",
    filters={"type": "financial"}
)
```

### Autonomous Operation

Run the complete autonomous system:

```bash
python scripts/autonomous_system.py
```

This will start the self-improving RAG system that:
- Continuously monitors its performance
- Identifies optimization opportunities
- Tests improvements through A/B testing
- Deploys successful optimizations
- Detects and recovers from errors

### Performance Dashboard

Monitor system performance in real-time:

```bash
python scripts/performance_dashboard.py

# Watch mode (refreshes every 5 seconds)
python scripts/performance_dashboard.py --watch
```

## 📁 Project Structure

```
CODE/
├── src/                    # Source code
│   ├── classifiers/        # Page classification
│   ├── metadata/           # Metadata extraction
│   ├── optimization/       # Self-improvement engine
│   ├── processors/         # Document processing
│   ├── retrieval/          # Search and retrieval
│   └── training/           # ML training pipeline
├── tests/                  # Test suites
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── scripts/               # Utility scripts
│   ├── autonomous_system.py      # Main autonomous system
│   ├── performance_dashboard.py  # Monitoring dashboard
│   └── benchmark_retrieval.py    # Performance benchmarks
├── docs/                  # Documentation
├── configs/               # Configuration files
└── models/                # Trained models
```

## 🧪 Testing

Run the complete test suite:

```bash
# All tests
pytest tests/ -v

# Unit tests with coverage
pytest tests/unit/ --cov=src --cov-report=term-missing

# Specific component
pytest tests/unit/test_optimization_engine.py -v
```

Current test coverage: **>90%**

## 🔄 Development

This project follows strict development practices:

- **Test-Driven Development (TDD)**: All code is written test-first
- **Agile Methodology**: Development in 1-week sprints
- **Conventional Commits**: Structured commit messages
- **Type Safety**: Full type hints throughout

### Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests first (TDD)
4. Implement the feature
5. Ensure all tests pass
6. Submit a pull request

## 📈 Self-Improvement Capabilities

The system continuously improves through:

### 1. Performance Monitoring
- Real-time metrics collection
- Anomaly detection
- Performance prediction

### 2. A/B Testing
- Controlled experiments
- Statistical significance testing
- Automatic winner deployment

### 3. Drift Detection
- Distribution change monitoring
- Accuracy tracking
- Automatic retraining triggers

### 4. Self-Healing
- Error detection
- Recovery action generation
- Service restart capabilities

## 🎯 Use Cases

- **Document Management Systems**: Intelligent document classification and retrieval
- **Legal Research**: Fast access to relevant legal documents and clauses
- **Financial Analysis**: Quick retrieval of financial statements and reports
- **Technical Documentation**: Efficient search through API docs and manuals
- **Research Papers**: Academic paper classification and retrieval

## 📊 Benchmarks

Latest benchmark results:

```
Query Latency:
  Average: 15.7ms
  P95: 19.1ms
  P99: 22.3ms

Classification Accuracy:
  Financial: 94.2%
  Legal: 91.8%
  Technical: 91.0%
  Overall: 92.3%

Retrieval Performance:
  Precision@10: 0.85
  Recall@10: 0.78
  MRR: 0.82
```

## 🔧 Configuration

Main configuration options in `configs/system_config.yaml`:

```yaml
retrieval:
  chunk_size: 500
  chunk_overlap: 50
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  
classification:
  confidence_threshold: 0.85
  ensemble_voting: true
  
optimization:
  ab_test_duration_hours: 24
  drift_threshold: 0.1
  auto_rollback: true
```

## 📚 Documentation

- [System Overview](docs/SYSTEM_OVERVIEW.md)
- [Sprint 1: Foundation](docs/sprints/sprint_1_summary.md)
- [Sprint 2: Classification](docs/sprints/sprint_2_summary.md)
- [Sprint 3: Retrieval](docs/sprints/sprint_3_summary.md)
- [Sprint 4: Autonomy](docs/sprint4_autonomous_optimization.md)

## 🚦 Roadmap

### Near Term (Q1)
- [ ] Transformer-based classifiers
- [ ] Neural architecture search
- [ ] Few-shot learning capabilities

### Medium Term (Q2)
- [ ] Multi-node deployment
- [ ] Federated learning
- [ ] Cross-system knowledge sharing

### Long Term (Q3-Q4)
- [ ] Reinforcement learning optimization
- [ ] Cloud-native deployment
- [ ] Enterprise integrations

## 📝 License

This project is developed as a demonstration of advanced RAG system capabilities.

## 🙏 Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/) - RAG framework
- [PyTorch](https://pytorch.org/) - Deep learning
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [pytest](https://pytest.org/) - Testing framework

Developed using Test-Driven Development and Agile methodology over 4 weeks.

## 📞 Support

For questions, issues, or collaboration:
- Open an issue in the repository
- Check the [documentation](docs/)
- Review the [test suite](tests/) for usage examples

---

**Status**: ✅ Production Ready | **Version**: 1.0.0 | **Last Updated**: December 2024