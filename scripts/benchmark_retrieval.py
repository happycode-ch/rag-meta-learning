#!/usr/bin/env python
"""
Retrieval System Performance Benchmark
Sprint 3: Retrieval Optimization
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.retrieval_engine import (
    RetrievalEngine, RetrievalConfig, MetadataFilter
)
from src.training.data_generator import TrainingDataGenerator, DatasetConfig
from src.classifiers.page_classifier import PageType


def generate_test_corpus(num_documents: int = 1000) -> List[Dict]:
    """Generate test document corpus"""
    print(f"Generating {num_documents} test documents...")
    
    generator = TrainingDataGenerator()
    documents = []
    
    # Generate equal distribution across types
    docs_per_type = num_documents // 3
    
    for page_type in [PageType.FINANCIAL, PageType.LEGAL, PageType.TECHNICAL]:
        template = generator.get_template(page_type)
        
        for i in range(docs_per_type):
            content = template.generate()
            doc = {
                "id": f"{page_type.value}_{i}",
                "content": content,
                "metadata": {
                    "page_type": page_type.value,
                    "doc_index": i,
                    "timestamp": datetime.now().isoformat(),
                },
                # Random embedding for testing
                "embedding": np.random.rand(384).tolist(),
            }
            documents.append(doc)
    
    return documents


def benchmark_indexing(engine: RetrievalEngine, documents: List[Dict]) -> Dict[str, float]:
    """Benchmark document indexing performance"""
    print("\n📥 Benchmarking Indexing Performance...")
    
    results = {}
    
    # Single document indexing
    start_time = time.time()
    engine.add_document(
        doc_id=documents[0]["id"],
        content=documents[0]["content"],
        embedding=documents[0]["embedding"],
        metadata=documents[0]["metadata"],
    )
    single_time = time.time() - start_time
    results["single_doc_time_ms"] = single_time * 1000
    
    # Batch indexing
    start_time = time.time()
    engine.batch_add_documents(documents[1:])
    batch_time = time.time() - start_time
    results["batch_time_s"] = batch_time
    results["docs_per_second"] = (len(documents) - 1) / batch_time
    
    print(f"  Single document: {results['single_doc_time_ms']:.2f}ms")
    print(f"  Batch indexing: {results['docs_per_second']:.1f} docs/sec")
    
    return results


def benchmark_search_latency(engine: RetrievalEngine) -> Dict[str, float]:
    """Benchmark search latency"""
    print("\n⚡ Benchmarking Search Latency...")
    
    queries = [
        "financial revenue quarterly report",
        "legal contract agreement terms",
        "API documentation endpoint parameters",
        "balance sheet income statement",
        "service agreement liability clause",
        "REST API authentication token",
    ]
    
    results = {}
    latencies = []
    
    # Warm-up
    for _ in range(5):
        engine.search("warmup query")
    
    # Measure latencies
    for query in queries:
        start_time = time.time()
        search_results = engine.search(query, k=10)
        latency = (time.time() - start_time) * 1000  # ms
        latencies.append(latency)
    
    results["min_latency_ms"] = min(latencies)
    results["max_latency_ms"] = max(latencies)
    results["avg_latency_ms"] = np.mean(latencies)
    results["p50_latency_ms"] = np.percentile(latencies, 50)
    results["p95_latency_ms"] = np.percentile(latencies, 95)
    results["p99_latency_ms"] = np.percentile(latencies, 99)
    
    print(f"  Average: {results['avg_latency_ms']:.2f}ms")
    print(f"  P50: {results['p50_latency_ms']:.2f}ms")
    print(f"  P95: {results['p95_latency_ms']:.2f}ms")
    print(f"  P99: {results['p99_latency_ms']:.2f}ms")
    
    return results


def benchmark_metadata_filtering(engine: RetrievalEngine) -> Dict[str, float]:
    """Benchmark metadata filtering performance"""
    print("\n🔍 Benchmarking Metadata Filtering...")
    
    results = {}
    
    # No filter
    start_time = time.time()
    no_filter_results = engine.search("financial report", k=10)
    no_filter_time = (time.time() - start_time) * 1000
    results["no_filter_ms"] = no_filter_time
    results["no_filter_count"] = len(no_filter_results)
    
    # With filter
    filter_obj = MetadataFilter(page_type="financial")
    start_time = time.time()
    filtered_results = engine.search("financial report", k=10, metadata_filter=filter_obj)
    filter_time = (time.time() - start_time) * 1000
    results["with_filter_ms"] = filter_time
    results["with_filter_count"] = len(filtered_results)
    results["filter_overhead_ms"] = filter_time - no_filter_time
    
    print(f"  No filter: {results['no_filter_ms']:.2f}ms ({results['no_filter_count']} results)")
    print(f"  With filter: {results['with_filter_ms']:.2f}ms ({results['with_filter_count']} results)")
    print(f"  Filter overhead: {results['filter_overhead_ms']:.2f}ms")
    
    return results


def benchmark_hybrid_vs_vector(engine: RetrievalEngine) -> Dict[str, float]:
    """Compare hybrid search vs vector-only search"""
    print("\n🔄 Benchmarking Hybrid vs Vector Search...")
    
    results = {}
    query = "quarterly financial revenue statement"
    
    # Hybrid search (default)
    engine.config.use_hybrid_search = True
    start_time = time.time()
    hybrid_results = engine.search(query, k=10)
    hybrid_time = (time.time() - start_time) * 1000
    results["hybrid_ms"] = hybrid_time
    results["hybrid_results"] = len(hybrid_results)
    
    # Vector-only search
    engine.config.use_hybrid_search = False
    start_time = time.time()
    vector_results = engine.search(query, k=10)
    vector_time = (time.time() - start_time) * 1000
    results["vector_only_ms"] = vector_time
    results["vector_only_results"] = len(vector_results)
    
    # Reset to hybrid
    engine.config.use_hybrid_search = True
    
    results["hybrid_overhead_ms"] = hybrid_time - vector_time
    results["hybrid_overhead_percent"] = (hybrid_time / vector_time - 1) * 100
    
    print(f"  Hybrid search: {results['hybrid_ms']:.2f}ms")
    print(f"  Vector-only: {results['vector_only_ms']:.2f}ms")
    print(f"  Hybrid overhead: {results['hybrid_overhead_percent']:.1f}%")
    
    return results


def benchmark_scalability(max_docs: int = 5000) -> Dict[str, Any]:
    """Test scalability with increasing document counts"""
    print("\n📈 Benchmarking Scalability...")
    
    doc_counts = [100, 500, 1000, 2000, 5000]
    results = {"doc_counts": doc_counts, "latencies": [], "throughputs": []}
    
    for count in doc_counts:
        if count > max_docs:
            break
            
        print(f"\n  Testing with {count} documents...")
        
        # Create fresh engine
        config = RetrievalConfig()
        engine = RetrievalEngine(config)
        
        # Generate documents
        docs = generate_test_corpus(count)
        
        # Index documents
        start_time = time.time()
        engine.batch_add_documents(docs)
        index_time = time.time() - start_time
        throughput = count / index_time
        results["throughputs"].append(throughput)
        
        # Measure search latency
        queries = ["financial", "legal", "technical", "revenue", "contract"]
        latencies = []
        
        for query in queries:
            start_time = time.time()
            engine.search(query, k=10)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        results["latencies"].append(avg_latency)
        
        print(f"    Index throughput: {throughput:.1f} docs/sec")
        print(f"    Search latency: {avg_latency:.2f}ms")
    
    return results


def main():
    """Run complete retrieval benchmark suite"""
    print("=" * 60)
    print("RETRIEVAL SYSTEM PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Initialize retrieval engine
    config = RetrievalConfig(
        embedding_dim=384,
        max_results=10,
        similarity_threshold=0.7,
        use_hybrid_search=True,
        rrf_k=60,
    )
    engine = RetrievalEngine(config)
    
    # Generate test corpus
    documents = generate_test_corpus(1000)
    print(f"✅ Generated {len(documents)} test documents")
    
    # Run benchmarks
    benchmark_results = {}
    
    # 1. Indexing performance
    benchmark_results["indexing"] = benchmark_indexing(engine, documents)
    
    # 2. Search latency
    benchmark_results["latency"] = benchmark_search_latency(engine)
    
    # 3. Metadata filtering
    benchmark_results["filtering"] = benchmark_metadata_filtering(engine)
    
    # 4. Hybrid vs Vector
    benchmark_results["hybrid_comparison"] = benchmark_hybrid_vs_vector(engine)
    
    # 5. Scalability
    benchmark_results["scalability"] = benchmark_scalability(max_docs=2000)
    
    # Overall summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Check against targets
    target_latency = 100  # ms
    actual_latency = benchmark_results["latency"]["avg_latency_ms"]
    latency_status = "✅" if actual_latency < target_latency else "❌"
    
    print(f"\n📊 Key Metrics:")
    print(f"  Index throughput: {benchmark_results['indexing']['docs_per_second']:.1f} docs/sec")
    print(f"  Search latency: {actual_latency:.2f}ms {latency_status} (target: <{target_latency}ms)")
    print(f"  P99 latency: {benchmark_results['latency']['p99_latency_ms']:.2f}ms")
    print(f"  Filter overhead: {benchmark_results['filtering']['filter_overhead_ms']:.2f}ms")
    print(f"  Hybrid overhead: {benchmark_results['hybrid_comparison']['hybrid_overhead_percent']:.1f}%")
    
    # Performance rating
    if actual_latency < 50:
        rating = "⭐⭐⭐⭐⭐ Excellent"
    elif actual_latency < 100:
        rating = "⭐⭐⭐⭐ Good"
    elif actual_latency < 200:
        rating = "⭐⭐⭐ Acceptable"
    else:
        rating = "⭐⭐ Needs Improvement"
    
    print(f"\n🏆 Performance Rating: {rating}")
    
    # Save results
    benchmark_results["timestamp"] = datetime.now().isoformat()
    benchmark_results["summary"] = {
        "avg_latency_ms": actual_latency,
        "meets_target": actual_latency < target_latency,
        "index_throughput": benchmark_results["indexing"]["docs_per_second"],
        "rating": rating,
    }
    
    output_file = Path("retrieval_benchmarks.json")
    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_file}")
    
    return benchmark_results


if __name__ == "__main__":
    main()