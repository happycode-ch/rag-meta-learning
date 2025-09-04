#!/usr/bin/env python
"""
Performance Benchmark Script
Establishes baseline performance metrics for the RAG system
"""

import time
import tempfile
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

from src.processors.document_processor import DocumentProcessor
from src.classifiers.page_classifier import PageClassifier


def benchmark_document_processing(num_docs: int = 10) -> Dict[str, Any]:
    """Benchmark document processing performance"""
    processor = DocumentProcessor()
    results = {
        "total_documents": num_docs,
        "successful": 0,
        "failed": 0,
        "total_time": 0,
        "avg_time_per_doc": 0,
        "throughput_per_minute": 0,
    }
    
    # Create test documents
    docs = []
    for i in range(num_docs):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"Test document {i}\n" * 100)
            docs.append(Path(f.name))
    
    # Benchmark processing
    start_time = time.time()
    
    for doc in docs:
        try:
            result = processor.process_document(doc)
            if result.success:
                results["successful"] += 1
            else:
                results["failed"] += 1
        except Exception:
            results["failed"] += 1
    
    end_time = time.time()
    results["total_time"] = end_time - start_time
    results["avg_time_per_doc"] = results["total_time"] / num_docs
    results["throughput_per_minute"] = (num_docs / results["total_time"]) * 60
    
    # Clean up
    for doc in docs:
        doc.unlink(missing_ok=True)
    
    return results


def benchmark_classification(num_pages: int = 100) -> Dict[str, Any]:
    """Benchmark page classification performance"""
    classifier = PageClassifier()
    processor = DocumentProcessor()
    
    results = {
        "total_pages": num_pages,
        "classifications": 0,
        "total_time": 0,
        "avg_time_per_page": 0,
        "classifications_per_second": 0,
    }
    
    # Create test pages
    test_content = [
        "Financial statement with balance $1000 account transaction",
        "Legal contract agreement between parties with terms and conditions",
        "API documentation with endpoint parameters and response format",
        "General text without specific category markers",
    ]
    
    pages = []
    for i in range(num_pages):
        content = test_content[i % len(test_content)]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            doc_path = Path(f.name)
            
        # Extract page
        doc_pages = processor.extract_pages(doc_path)
        if doc_pages:
            pages.append(doc_pages[0])
        doc_path.unlink(missing_ok=True)
    
    # Benchmark classification
    start_time = time.time()
    
    for page in pages:
        try:
            result = classifier.classify(page)
            if result:
                results["classifications"] += 1
        except Exception:
            pass
    
    end_time = time.time()
    results["total_time"] = end_time - start_time
    results["avg_time_per_page"] = results["total_time"] / num_pages if num_pages > 0 else 0
    results["classifications_per_second"] = results["classifications"] / results["total_time"] if results["total_time"] > 0 else 0
    
    return results


def main():
    """Run all benchmarks and save results"""
    print("=" * 60)
    print("RAG SYSTEM PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    benchmarks = {}
    
    # Document Processing Benchmark
    print("\n📄 Document Processing Benchmark...")
    doc_results = benchmark_document_processing(num_docs=50)
    benchmarks["document_processing"] = doc_results
    
    print(f"  ✓ Processed {doc_results['successful']}/{doc_results['total_documents']} documents")
    print(f"  ⏱️  Average time per document: {doc_results['avg_time_per_doc']:.3f}s")
    print(f"  📊 Throughput: {doc_results['throughput_per_minute']:.1f} docs/min")
    
    # Classification Benchmark
    print("\n🏷️  Page Classification Benchmark...")
    class_results = benchmark_classification(num_pages=100)
    benchmarks["page_classification"] = class_results
    
    print(f"  ✓ Classified {class_results['classifications']}/{class_results['total_pages']} pages")
    print(f"  ⏱️  Average time per page: {class_results['avg_time_per_page']:.4f}s")
    print(f"  📊 Speed: {class_results['classifications_per_second']:.1f} classifications/sec")
    
    # Overall Performance Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE BASELINE SUMMARY")
    print("=" * 60)
    
    # Check against targets
    doc_target = 100  # docs/min
    doc_actual = doc_results['throughput_per_minute']
    doc_status = "✅" if doc_actual >= doc_target else "❌"
    
    print(f"\nDocument Processing:")
    print(f"  Target: {doc_target} docs/min")
    print(f"  Actual: {doc_actual:.1f} docs/min {doc_status}")
    
    class_target = 0.1  # 100ms per classification
    class_actual = class_results['avg_time_per_page']
    class_status = "✅" if class_actual <= class_target else "⚠️"
    
    print(f"\nPage Classification:")
    print(f"  Target: <{class_target*1000:.0f}ms per page")
    print(f"  Actual: {class_actual*1000:.1f}ms per page {class_status}")
    
    # Save results
    benchmarks["timestamp"] = datetime.now().isoformat()
    benchmarks["summary"] = {
        "document_processing_meets_target": doc_actual >= doc_target,
        "classification_meets_target": class_actual <= class_target,
    }
    
    output_file = Path("benchmarks.json")
    with open(output_file, "w") as f:
        json.dump(benchmarks, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    return benchmarks


if __name__ == "__main__":
    main()