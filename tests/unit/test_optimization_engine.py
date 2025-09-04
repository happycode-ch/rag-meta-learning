"""
Test-Driven Development: Self-Optimization Engine Tests
Sprint 4: Autonomous Optimization
"""

import pytest
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.optimization.optimization_engine import (
    OptimizationEngine,
    PerformanceMonitor,
    ABTestFramework,
    DriftDetector,
    FeedbackCollector,
    OptimizationStrategy,
    ExperimentResult,
    SystemMetrics,
)


class TestOptimizationEngine:
    """Unit tests for self-optimization engine"""

    @pytest.fixture
    def engine(self):
        """Create an OptimizationEngine instance"""
        return OptimizationEngine()

    @pytest.fixture
    def monitor(self):
        """Create a PerformanceMonitor instance"""
        return PerformanceMonitor()

    @pytest.fixture
    def sample_metrics(self):
        """Create sample system metrics"""
        return SystemMetrics(
            timestamp=datetime.now(),
            query_latency_ms=15.5,
            classification_accuracy=0.92,
            retrieval_precision=0.85,
            throughput_qps=100,
            error_rate=0.01,
            memory_usage_mb=512,
            cpu_usage_percent=45,
        )

    @pytest.mark.unit
    def test_engine_initialization(self, engine):
        """Test that OptimizationEngine initializes correctly"""
        assert engine is not None
        assert hasattr(engine, "performance_monitor")
        assert hasattr(engine, "ab_test_framework")
        assert hasattr(engine, "drift_detector")
        assert hasattr(engine, "feedback_collector")

    @pytest.mark.unit
    def test_performance_monitoring(self, monitor, sample_metrics):
        """Test performance metrics collection and storage"""
        # Record metrics
        monitor.record_metrics(sample_metrics)
        
        # Get recent metrics
        recent = monitor.get_recent_metrics(minutes=5)
        assert len(recent) == 1
        assert recent[0].query_latency_ms == 15.5

    @pytest.mark.unit
    def test_anomaly_detection(self, monitor):
        """Test anomaly detection in performance metrics"""
        # Add normal metrics
        for i in range(10):
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                query_latency_ms=10 + i * 0.5,
                classification_accuracy=0.95,
                retrieval_precision=0.90,
                throughput_qps=100,
            )
            monitor.record_metrics(metrics)
        
        # Add anomalous metric
        anomaly = SystemMetrics(
            timestamp=datetime.now(),
            query_latency_ms=100,  # 10x normal
            classification_accuracy=0.95,
            retrieval_precision=0.90,
            throughput_qps=100,
        )
        
        is_anomaly = monitor.detect_anomaly(anomaly)
        assert is_anomaly is True

    @pytest.mark.unit
    def test_ab_testing_framework(self, engine):
        """Test A/B testing for optimization strategies"""
        ab_test = engine.ab_test_framework
        
        # Define control and treatment strategies
        control = OptimizationStrategy(
            name="control",
            parameters={"chunk_size": 500, "overlap": 50}
        )
        
        treatment = OptimizationStrategy(
            name="treatment",
            parameters={"chunk_size": 300, "overlap": 100}
        )
        
        # Run A/B test
        result = ab_test.run_test(
            control=control,
            treatment=treatment,
            duration_hours=0.001,  # Short duration for testing
            metric="query_latency_ms"
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.winner in ["control", "treatment", "inconclusive"]
        assert result.confidence >= 0 and result.confidence <= 1

    @pytest.mark.unit
    def test_drift_detection(self):
        """Test concept drift detection"""
        detector = DriftDetector()
        
        # Simulate normal distribution
        for _ in range(100):
            detector.add_observation(
                predicted_class="financial",
                actual_class="financial",
                confidence=0.95
            )
        
        # Check no drift
        assert detector.detect_drift() is False
        
        # Simulate drift
        for _ in range(50):
            detector.add_observation(
                predicted_class="financial",
                actual_class="legal",  # Misclassification
                confidence=0.95
            )
        
        # Check drift detected
        assert detector.detect_drift() is True

    @pytest.mark.unit
    def test_feedback_collection(self):
        """Test user feedback collection and analysis"""
        collector = FeedbackCollector()
        
        # Add positive feedback
        collector.add_feedback(
            query="financial report",
            result_id="doc1",
            relevant=True,
            rank=1
        )
        
        # Add negative feedback
        collector.add_feedback(
            query="financial report",
            result_id="doc2",
            relevant=False,
            rank=2
        )
        
        # Analyze feedback
        analysis = collector.analyze_feedback()
        assert "precision" in analysis
        assert "user_satisfaction" in analysis
        assert analysis["total_feedback"] == 2

    @pytest.mark.unit
    def test_optimization_strategy_generation(self, engine):
        """Test generation of optimization strategies"""
        # Identify improvement opportunity
        opportunity = {
            "metric": "query_latency_ms",
            "current_value": 50,
            "target_value": 25,
            "severity": "high"
        }
        
        # Generate strategies
        strategies = engine.generate_optimization_strategies(opportunity)
        
        assert len(strategies) > 0
        assert all(isinstance(s, OptimizationStrategy) for s in strategies)
        assert all(hasattr(s, "expected_improvement") for s in strategies)

    @pytest.mark.unit
    def test_automatic_rollback(self, engine):
        """Test automatic rollback on performance regression"""
        # Deploy optimization
        strategy = OptimizationStrategy(
            name="test_optimization",
            parameters={"test_param": 100}
        )
        
        deployment_id = engine.deploy_optimization(strategy)
        assert deployment_id is not None
        
        # Simulate performance regression
        regression_metrics = SystemMetrics(
            timestamp=datetime.now(),
            query_latency_ms=200,  # Much worse
            classification_accuracy=0.50,  # Much worse
            error_rate=0.10,  # Much worse
        )
        
        # Check for rollback
        should_rollback = engine.check_rollback_conditions(
            deployment_id, 
            regression_metrics
        )
        assert should_rollback is True

    @pytest.mark.unit
    def test_learning_documentation(self, engine, tmp_path):
        """Test automatic documentation of learnings"""
        # Run experiment
        experiment = {
            "hypothesis": "Reducing chunk size improves retrieval precision",
            "strategy": OptimizationStrategy(
                name="small_chunks",
                parameters={"chunk_size": 100}
            ),
            "result": ExperimentResult(
                winner="treatment",
                improvement=0.15,
                confidence=0.95,
                sample_size=1000
            )
        }
        
        # Document learning
        doc_path = tmp_path / "learnings.json"
        engine.document_learning(experiment, doc_path)
        
        assert doc_path.exists()
        
        # Verify content
        with open(doc_path, "r") as f:
            learning = json.load(f)
        
        assert learning["hypothesis"] == experiment["hypothesis"]
        assert learning["outcome"]["improvement"] == 0.15

    @pytest.mark.unit
    def test_continuous_improvement_loop(self, engine):
        """Test the continuous improvement loop"""
        # Start improvement loop (non-blocking for test)
        loop_iterations = []
        
        def mock_iteration():
            loop_iterations.append(datetime.now())
            return len(loop_iterations) < 3  # Run 3 iterations
        
        engine.improvement_iteration = mock_iteration
        engine.run_improvement_loop(blocking=False)
        
        # Verify iterations occurred
        assert len(loop_iterations) >= 1

    @pytest.mark.unit
    def test_self_healing_capabilities(self, engine):
        """Test self-healing when errors are detected"""
        # Simulate error condition
        error_metrics = SystemMetrics(
            timestamp=datetime.now(),
            error_rate=0.15,  # High error rate
            query_latency_ms=20,
        )
        
        # Detect and heal
        healing_actions = engine.detect_and_heal(error_metrics)
        
        assert len(healing_actions) > 0
        assert any("restart" in action or "cache" in action 
                  for action in healing_actions)

    @pytest.mark.unit
    def test_performance_prediction(self, engine):
        """Test performance prediction capabilities"""
        # Add historical data
        for i in range(24):
            metrics = SystemMetrics(
                timestamp=datetime.now() - timedelta(hours=i),
                query_latency_ms=10 + i * 0.5,
                throughput_qps=100 - i * 2,
            )
            engine.performance_monitor.record_metrics(metrics)
        
        # Predict future performance
        prediction = engine.predict_performance(hours_ahead=1)
        
        assert "query_latency_ms" in prediction
        assert "throughput_qps" in prediction
        assert "confidence_interval" in prediction

    @pytest.mark.unit
    def test_optimization_scheduling(self, engine):
        """Test scheduling of optimization tasks"""
        # Schedule optimization
        task_id = engine.schedule_optimization(
            strategy=OptimizationStrategy(
                name="scheduled_opt",
                parameters={"test": True}
            ),
            run_at=datetime.now() + timedelta(seconds=1)
        )
        
        assert task_id is not None
        
        # Check pending tasks
        pending = engine.get_pending_optimizations()
        assert len(pending) == 1
        assert pending[0]["task_id"] == task_id

    @pytest.mark.unit
    def test_cost_benefit_analysis(self, engine):
        """Test cost-benefit analysis of optimizations"""
        strategy = OptimizationStrategy(
            name="expensive_optimization",
            parameters={"compute_cost": 100},
            expected_improvement=0.10,
            implementation_cost=500,
        )
        
        # Analyze cost-benefit
        analysis = engine.analyze_cost_benefit(strategy)
        
        assert "roi" in analysis  # Return on investment
        assert "payback_period" in analysis
        assert "recommendation" in analysis
        assert analysis["recommendation"] in ["proceed", "reject", "review"]