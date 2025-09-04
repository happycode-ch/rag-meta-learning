"""
Test-Driven Development: Retrieval Engine Tests
Sprint 3: Retrieval Optimization
"""

import pytest
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.retrieval.retrieval_engine import (
    RetrievalEngine,
    VectorStore,
    HybridSearch,
    MetadataFilter,
    QueryTransformer,
    SearchResult,
    RetrievalConfig,
)
from src.classifiers.page_classifier import PageType


class TestRetrievalEngine:
    """Unit tests for retrieval engine with hybrid search"""

    @pytest.fixture
    def config(self):
        """Create retrieval configuration"""
        return RetrievalConfig(
            embedding_dim=384,
            max_results=10,
            similarity_threshold=0.7,
            use_hybrid_search=True,
            rrf_k=60,
        )

    @pytest.fixture
    def engine(self, config):
        """Create a RetrievalEngine instance"""
        return RetrievalEngine(config)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            {
                "id": "doc1",
                "content": "Financial statement showing quarterly revenue of $1M",
                "metadata": {"page_type": "financial", "date": "2024-01-01"},
                "embedding": np.random.rand(384).tolist(),
            },
            {
                "id": "doc2",
                "content": "Legal contract between parties for service agreement",
                "metadata": {"page_type": "legal", "date": "2024-02-01"},
                "embedding": np.random.rand(384).tolist(),
            },
            {
                "id": "doc3",
                "content": "API documentation for REST endpoints and parameters",
                "metadata": {"page_type": "technical", "date": "2024-03-01"},
                "embedding": np.random.rand(384).tolist(),
            },
        ]

    @pytest.mark.unit
    def test_engine_initialization(self, engine):
        """Test that RetrievalEngine initializes correctly"""
        assert engine is not None
        assert hasattr(engine, "vector_store")
        assert hasattr(engine, "query_transformer")
        assert hasattr(engine, "hybrid_search")

    @pytest.mark.unit
    def test_vector_store_operations(self, engine, sample_documents):
        """Test vector store add and search operations"""
        # Add documents
        for doc in sample_documents:
            engine.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
            )
        
        # Search
        query_embedding = np.random.rand(384).tolist()
        results = engine.vector_store.search(query_embedding, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.unit
    def test_metadata_filtering(self, engine, sample_documents):
        """Test metadata-based filtering"""
        # Add documents
        for doc in sample_documents:
            engine.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
            )
        
        # Create filter
        filter_obj = MetadataFilter(page_type="financial")
        
        # Search with filter
        results = engine.search(
            query="revenue",
            metadata_filter=filter_obj,
        )
        
        # Should only return financial documents
        assert all(r.metadata.get("page_type") == "financial" for r in results)

    @pytest.mark.unit
    def test_hybrid_search(self, engine, sample_documents):
        """Test hybrid search combining vector and keyword search"""
        # Add documents
        for doc in sample_documents:
            engine.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
            )
        
        # Perform hybrid search
        results = engine.hybrid_search.search(
            query="financial API",
            k=3,
        )
        
        assert len(results) > 0
        assert all(hasattr(r, "score") for r in results)

    @pytest.mark.unit
    def test_reciprocal_rank_fusion(self, engine):
        """Test RRF algorithm for combining rankings"""
        # Create two sets of rankings
        vector_results = [
            SearchResult("doc1", 0.9, "Content 1", {}),
            SearchResult("doc2", 0.8, "Content 2", {}),
            SearchResult("doc3", 0.7, "Content 3", {}),
        ]
        
        keyword_results = [
            SearchResult("doc2", 0.85, "Content 2", {}),
            SearchResult("doc1", 0.75, "Content 1", {}),
            SearchResult("doc4", 0.65, "Content 4", {}),
        ]
        
        # Apply RRF
        fused_results = engine.hybrid_search.apply_rrf(
            vector_results, keyword_results
        )
        
        assert len(fused_results) == 4  # All unique documents
        assert fused_results[0].doc_id in ["doc1", "doc2"]  # Top results

    @pytest.mark.unit
    def test_query_transformation(self, engine):
        """Test query transformation strategies"""
        transformer = engine.query_transformer
        
        # Test query expansion
        expanded = transformer.expand_query("financial report")
        assert len(expanded.split()) > len("financial report".split())
        
        # Test query rewriting
        rewritten = transformer.rewrite_query("fin stmt Q1")
        assert "financial" in rewritten.lower() or "statement" in rewritten.lower()

    @pytest.mark.unit
    def test_query_latency_under_100ms(self, engine, sample_documents):
        """Test that query latency is under 100ms"""
        # Add documents
        for doc in sample_documents:
            engine.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
            )
        
        # Measure query time
        start_time = time.time()
        results = engine.search("financial revenue", k=5)
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert query_time < 100, f"Query took {query_time:.2f}ms, should be <100ms"

    @pytest.mark.unit
    def test_batch_indexing(self, engine, sample_documents):
        """Test batch document indexing"""
        # Batch add documents
        engine.batch_add_documents(sample_documents)
        
        # Verify all documents are indexed
        stats = engine.get_index_stats()
        assert stats["document_count"] == len(sample_documents)

    @pytest.mark.unit
    def test_similarity_threshold(self, engine, sample_documents):
        """Test similarity threshold filtering"""
        # Add documents
        for doc in sample_documents:
            engine.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
            )
        
        # Search with high threshold
        results = engine.search(
            query="blockchain cryptocurrency",
            similarity_threshold=0.9,
        )
        
        # With unrelated query and high threshold, should return few/no results
        assert len(results) <= 1

    @pytest.mark.unit
    def test_result_ranking_quality(self, engine):
        """Test that results are properly ranked by relevance"""
        # Add documents with known relevance
        docs = [
            {
                "id": "exact",
                "content": "quarterly financial report revenue",
                "embedding": [1.0] * 384,
                "metadata": {},
            },
            {
                "id": "partial",
                "content": "annual report",
                "embedding": [0.5] * 384,
                "metadata": {},
            },
            {
                "id": "irrelevant",
                "content": "user manual installation guide",
                "embedding": [0.1] * 384,
                "metadata": {},
            },
        ]
        
        for doc in docs:
            engine.add_document(**doc)
        
        # Search with query similar to first doc
        results = engine.search("quarterly financial report")
        
        # Check ranking order
        if len(results) >= 2:
            assert results[0].score > results[1].score

    @pytest.mark.unit
    def test_metadata_aggregation(self, engine, sample_documents):
        """Test metadata aggregation capabilities"""
        # Add documents
        for doc in sample_documents:
            engine.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
            )
        
        # Get aggregated metadata
        aggregates = engine.get_metadata_aggregates()
        
        assert "page_type" in aggregates
        assert len(aggregates["page_type"]) == 3  # financial, legal, technical

    @pytest.mark.unit
    def test_index_persistence(self, engine, sample_documents, tmp_path):
        """Test saving and loading index"""
        # Add documents
        for doc in sample_documents:
            engine.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
            )
        
        # Save index
        index_path = tmp_path / "index.pkl"
        engine.save_index(index_path)
        assert index_path.exists()
        
        # Create new engine and load index
        new_engine = RetrievalEngine(engine.config)
        new_engine.load_index(index_path)
        
        # Verify documents are loaded
        stats = new_engine.get_index_stats()
        assert stats["document_count"] == len(sample_documents)

    @pytest.mark.unit
    def test_concurrent_search(self, engine, sample_documents):
        """Test thread-safe concurrent searching"""
        import concurrent.futures
        
        # Add documents
        for doc in sample_documents:
            engine.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
            )
        
        # Perform concurrent searches
        queries = ["financial", "legal", "technical", "revenue", "contract"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(engine.search, query)
                for query in queries
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == len(queries)
        assert all(isinstance(r, list) for r in results)