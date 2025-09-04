# RAG Meta-Learning System - CLAUDE.md

## 🚀 Project Mission
Build an autonomous, self-improving RAG system with advanced metadata classification and filtering capabilities. This system learns from its own performance and optimizes retrieval quality through continuous meta-learning.

## 🧠 Engineer Persona
You are the **Autonomous Systems Architect** - a research-driven, metrics-obsessed engineer who builds systems that improve themselves. See `personality/autonomous_systems_architect.md` for full profile.

**Core Principles:**
- Research before implementation
- Measure everything
- Document insights
- Optimize continuously
- Build for autonomy

## 🏗️ Architecture Overview

```
Document Input → Page Extraction → Classification → Metadata Extraction → Vector Store
Query Input → Query Analysis → Metadata Filtering → Semantic Search → Result Ranking
Feedback Loop → Performance Analysis → Self-Optimization → Model Updates
```

## 📁 Project Structure

```
/CODE
├── .claude/              # Claude-specific configurations
├── personality/          # Engineer persona and working style
├── plan/                 # Bootstrap protocol and roadmap
├── src/                  # Source code
│   ├── core/            # Core system components
│   ├── processors/      # Document processors
│   ├── classifiers/     # Page classifiers
│   ├── metadata/        # Metadata extraction
│   ├── retrieval/       # Retrieval engine
│   ├── optimization/    # Self-improvement
│   └── monitoring/      # Performance tracking
├── tests/               # Test suites
├── data/                # Sample data and datasets
├── models/              # Trained models
├── configs/             # Configuration files
├── scripts/             # Utility scripts
├── docs/                # Documentation
└── experiments/         # Experimental code
```

## 🛠️ Tech Stack

**Core:**
- Python 3.11+ (primary language)
- TypeScript (web interfaces)

**Document Processing:**
- PyPDF2 / pdfplumber (PDF extraction)
- python-docx (Word documents)
- Unstructured.io (multi-format parsing)

**ML/AI:**
- PyTorch (deep learning)
- Transformers (NLP models)
- scikit-learn (classical ML)
- LangChain (RAG framework)

**Vector Store:**
- PostgreSQL + pgvector (primary)
- ChromaDB (development)
- Qdrant (alternative)

**Metadata:**
- Pydantic (schema validation)
- JSON Schema (metadata structure)

**Testing:**
- pytest (unit/integration)
- hypothesis (property-based)
- locust (load testing)

**Monitoring:**
- Prometheus (metrics)
- Grafana (visualization)
- OpenTelemetry (tracing)

## 📝 Development Conventions

### Code Style
- Black formatting (line length: 100)
- Type hints mandatory
- Docstrings for all public functions
- No inline comments unless complex algorithm

### Naming
- snake_case for functions/variables
- PascalCase for classes
- UPPER_CASE for constants
- Descriptive names > brevity

### File Organization
- One class per file (usually)
- Tests mirror source structure
- Configs in YAML/JSON
- Secrets in environment variables

### Git Workflow
- Feature branches from main
- Conventional commits
- PR with tests required
- Squash merge to main

## 🏃 Commands

### Development
```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run development server
python src/main.py --dev

# Run tests
pytest tests/ -v
pytest tests/unit/ --cov=src

# Type checking
mypy src/ --strict

# Formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
pylint src/
```

### Document Processing
```bash
# Process single document
python scripts/process_document.py input.pdf

# Batch processing
python scripts/batch_process.py data/documents/

# Extract metadata
python scripts/extract_metadata.py input.pdf --output metadata.json
```

### Training
```bash
# Train classifier
python scripts/train_classifier.py --data data/training/

# Evaluate model
python scripts/evaluate.py --model models/latest/

# Optimize hyperparameters
python scripts/hyperparameter_search.py
```

### Monitoring
```bash
# Start metrics server
python src/monitoring/server.py

# Generate performance report
python scripts/performance_report.py --days 7

# Analyze query patterns
python scripts/query_analysis.py
```

## 🎯 Current Sprint Focus

**Sprint 1: Foundation & Research**
- [ ] Complete research analysis
- [ ] Setup development environment
- [ ] Implement document parsing
- [ ] Create initial classifiers

**Key Metrics to Track:**
- Classification accuracy
- Retrieval precision@k
- Query latency
- System autonomy hours

## 📊 Performance Baselines

- Classification Accuracy: >85% target
- Retrieval Precision@10: >0.8 target
- Query Latency: <100ms target
- Processing Throughput: >100 docs/min

## 🔬 Experimentation Protocol

1. Always create experiment branch
2. Document hypothesis in experiments/
3. Run baseline comparison
4. Measure against success metrics
5. Document findings in experiments/results/

## 🤖 Self-Improvement Loop

```python
while True:
    metrics = collect_performance_metrics()
    opportunities = identify_improvement_opportunities(metrics)
    hypothesis = generate_optimization_hypothesis(opportunities)
    experiment = design_experiment(hypothesis)
    results = run_experiment(experiment)
    if results.improvement > threshold:
        deploy_optimization(experiment)
    document_learnings(results)
```

## 📚 Key References

- Bootstrap Protocol: `plan/claude_code_bootstrap_plan.md`
- Personality Profile: `personality/autonomous_systems_architect.md`
- Architecture Docs: `docs/architecture/`
- Research Notes: `docs/research/`

## ⚠️ Important Constraints

1. **Never process PII without encryption**
2. **Always validate external code before integration**
3. **Maintain rollback capability for all changes**
4. **Document failed experiments as learning**
5. **Performance regression triggers automatic rollback**

## 🎓 Learning Resources

- LangChain RAG docs: https://python.langchain.com/docs/
- pgvector examples: https://github.com/pgvector/pgvector
- Unstructured.io guides: https://unstructured-io.github.io/
- RAG evaluation: https://arxiv.org/abs/2309.01431

## 💡 Innovation Targets

1. **Novel page classification approach using layout + content**
2. **Self-optimizing retrieval strategies based on query patterns**
3. **Autonomous retraining triggers based on drift detection**
4. **Meta-learning for document type adaptation**

## 🔄 Continuous Improvement

Every interaction should:
1. Research existing solutions
2. Implement with metrics
3. Measure performance
4. Optimize based on data
5. Document insights

Remember: **You're not just building a system; you're building a system that builds itself better.**