"""
Self-Optimization Engine for Autonomous RAG System
Sprint 4: Autonomous Optimization
Following TDD principles - GREEN phase: Implementation to pass tests
"""

import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np
from scipy import stats
import random

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    query_latency_ms: float = 0.0
    classification_accuracy: float = 0.0
    retrieval_precision: float = 0.0
    throughput_qps: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class OptimizationStrategy:
    """A strategy for optimization"""
    name: str
    parameters: Dict[str, Any]
    expected_improvement: float = 0.0
    implementation_cost: float = 0.0
    risk_level: str = "low"  # low, medium, high


@dataclass
class ExperimentResult:
    """Result of an A/B test experiment"""
    winner: str  # "control", "treatment", "inconclusive"
    improvement: float
    confidence: float
    sample_size: int
    p_value: float = 0.0
    effect_size: float = 0.0


class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.metrics_history = deque(maxlen=window_size)
        self.anomaly_threshold = 3.0  # Standard deviations
        self._lock = threading.Lock()
    
    def record_metrics(self, metrics: SystemMetrics):
        """Record system metrics"""
        with self._lock:
            self.metrics_history.append(metrics)
    
    def get_recent_metrics(self, minutes: int = 5) -> List[SystemMetrics]:
        """Get metrics from recent minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [m for m in self.metrics_history 
                   if m.timestamp > cutoff]
    
    def detect_anomaly(self, metrics: SystemMetrics) -> bool:
        """Detect if metrics are anomalous"""
        if len(self.metrics_history) < 10:
            return False
        
        # Get recent normal metrics
        recent = self.get_recent_metrics(minutes=30)
        if len(recent) < 5:
            return False
        
        # Calculate statistics for key metrics
        latencies = [m.query_latency_ms for m in recent]
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # Check if current latency is anomalous
        if metrics.query_latency_ms > mean_latency + self.anomaly_threshold * std_latency:
            return True
        
        # Check error rate
        if metrics.error_rate > 0.05:  # 5% error rate threshold
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of metrics"""
        if not self.metrics_history:
            return {}
        
        recent = list(self.metrics_history)
        
        return {
            "avg_latency": np.mean([m.query_latency_ms for m in recent]),
            "p95_latency": np.percentile([m.query_latency_ms for m in recent], 95),
            "avg_accuracy": np.mean([m.classification_accuracy for m in recent]),
            "avg_throughput": np.mean([m.throughput_qps for m in recent]),
            "total_metrics": len(recent),
        }


class ABTestFramework:
    """Framework for A/B testing optimizations"""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = []
        self._lock = threading.Lock()
    
    def run_test(self, control: OptimizationStrategy, 
                 treatment: OptimizationStrategy,
                 duration_hours: float = 1.0,
                 metric: str = "query_latency_ms") -> ExperimentResult:
        """Run an A/B test between control and treatment"""
        
        # Simulate test execution (in production, would actually deploy and measure)
        control_samples = self._simulate_strategy(control, duration_hours)
        treatment_samples = self._simulate_strategy(treatment, duration_hours)
        
        # Statistical test
        if len(control_samples) < 2 or len(treatment_samples) < 2:
            return ExperimentResult(
                winner="inconclusive",
                improvement=0.0,
                confidence=0.0,
                sample_size=0
            )
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(control_samples, treatment_samples)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(control_samples) + np.var(treatment_samples)) / 2)
        effect_size = (np.mean(treatment_samples) - np.mean(control_samples)) / pooled_std if pooled_std > 0 else 0
        
        # Determine winner
        improvement = (np.mean(control_samples) - np.mean(treatment_samples)) / np.mean(control_samples) if np.mean(control_samples) > 0 else 0
        
        if p_value < 0.05:  # Significant difference
            if improvement > 0:
                winner = "treatment"
            else:
                winner = "control"
                improvement = abs(improvement)
        else:
            winner = "inconclusive"
        
        confidence = 1 - p_value if p_value < 1 else 0
        
        result = ExperimentResult(
            winner=winner,
            improvement=improvement,
            confidence=confidence,
            sample_size=len(control_samples) + len(treatment_samples),
            p_value=p_value,
            effect_size=effect_size
        )
        
        with self._lock:
            self.test_results.append(result)
        
        return result
    
    def _simulate_strategy(self, strategy: OptimizationStrategy, 
                          duration_hours: float) -> List[float]:
        """Simulate metrics for a strategy (mock for testing)"""
        # In production, would actually deploy and measure
        num_samples = max(10, int(duration_hours * 100))
        
        # Simulate based on strategy parameters
        if "chunk_size" in strategy.parameters:
            # Smaller chunks generally mean higher latency but better precision
            chunk_size = strategy.parameters["chunk_size"]
            base_latency = 10 + (500 - chunk_size) * 0.02
        else:
            base_latency = 15
        
        # Add noise
        samples = [base_latency + np.random.normal(0, 2) for _ in range(num_samples)]
        return samples


class DriftDetector:
    """Detect concept drift in model predictions"""
    
    def __init__(self, window_size: int = 500, drift_threshold: float = 0.15):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.observations = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def add_observation(self, predicted_class: str, actual_class: str, 
                       confidence: float):
        """Add a prediction observation"""
        with self._lock:
            self.observations.append({
                "predicted": predicted_class,
                "actual": actual_class,
                "confidence": confidence,
                "timestamp": datetime.now(),
                "correct": predicted_class == actual_class
            })
    
    def detect_drift(self) -> bool:
        """Detect if drift has occurred"""
        if len(self.observations) < 50:
            return False
        
        with self._lock:
            recent = list(self.observations)
        
        # Split into old and new windows
        mid_point = len(recent) // 2
        old_window = recent[:mid_point]
        new_window = recent[mid_point:]
        
        # Calculate accuracy for each window
        old_accuracy = sum(1 for obs in old_window if obs["correct"]) / len(old_window)
        new_accuracy = sum(1 for obs in new_window if obs["correct"]) / len(new_window)
        
        # Check if accuracy has dropped significantly
        accuracy_drop = old_accuracy - new_accuracy
        
        return accuracy_drop > self.drift_threshold
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get drift detection statistics"""
        if not self.observations:
            return {}
        
        with self._lock:
            recent = list(self.observations)
        
        accuracy = sum(1 for obs in recent if obs["correct"]) / len(recent)
        avg_confidence = np.mean([obs["confidence"] for obs in recent])
        
        return {
            "current_accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "total_observations": len(recent),
            "drift_detected": self.detect_drift()
        }


class FeedbackCollector:
    """Collect and analyze user feedback"""
    
    def __init__(self):
        self.feedback_data = []
        self._lock = threading.Lock()
    
    def add_feedback(self, query: str, result_id: str, 
                     relevant: bool, rank: int):
        """Add user feedback"""
        with self._lock:
            self.feedback_data.append({
                "query": query,
                "result_id": result_id,
                "relevant": relevant,
                "rank": rank,
                "timestamp": datetime.now()
            })
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """Analyze collected feedback"""
        if not self.feedback_data:
            return {
                "total_feedback": 0,
                "precision": 0,
                "user_satisfaction": 0
            }
        
        with self._lock:
            feedback = list(self.feedback_data)
        
        total = len(feedback)
        relevant = sum(1 for f in feedback if f["relevant"])
        
        # Calculate precision
        precision = relevant / total if total > 0 else 0
        
        # Calculate user satisfaction (based on relevant results in top ranks)
        top_3_relevant = sum(1 for f in feedback 
                           if f["relevant"] and f["rank"] <= 3)
        satisfaction = top_3_relevant / total if total > 0 else 0
        
        return {
            "total_feedback": total,
            "precision": precision,
            "user_satisfaction": satisfaction,
            "avg_rank_of_relevant": np.mean([f["rank"] for f in feedback if f["relevant"]]) if relevant > 0 else 0
        }


class OptimizationEngine:
    """Main self-optimization engine"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.ab_test_framework = ABTestFramework()
        self.drift_detector = DriftDetector()
        self.feedback_collector = FeedbackCollector()
        
        self.active_optimizations = {}
        self.scheduled_tasks = []
        self.learnings = []
        self._running = False
        self._lock = threading.Lock()
        
        logger.info("OptimizationEngine initialized")
    
    def generate_optimization_strategies(self, 
                                        opportunity: Dict[str, Any]) -> List[OptimizationStrategy]:
        """Generate optimization strategies for an improvement opportunity"""
        strategies = []
        
        metric = opportunity.get("metric", "")
        current = opportunity.get("current_value", 0)
        target = opportunity.get("target_value", 0)
        
        if metric == "query_latency_ms":
            # Strategies to reduce latency
            strategies.extend([
                OptimizationStrategy(
                    name="increase_cache_size",
                    parameters={"cache_size_mb": 512},
                    expected_improvement=0.2
                ),
                OptimizationStrategy(
                    name="optimize_chunk_size",
                    parameters={"chunk_size": 300, "overlap": 50},
                    expected_improvement=0.15
                ),
                OptimizationStrategy(
                    name="enable_query_caching",
                    parameters={"query_cache": True, "ttl_seconds": 300},
                    expected_improvement=0.3
                ),
            ])
        
        elif metric == "classification_accuracy":
            # Strategies to improve accuracy
            strategies.extend([
                OptimizationStrategy(
                    name="retrain_classifier",
                    parameters={"training_samples": 1000},
                    expected_improvement=0.1
                ),
                OptimizationStrategy(
                    name="ensemble_models",
                    parameters={"num_models": 5},
                    expected_improvement=0.05
                ),
            ])
        
        return strategies
    
    def deploy_optimization(self, strategy: OptimizationStrategy) -> str:
        """Deploy an optimization strategy"""
        deployment_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock:
            self.active_optimizations[deployment_id] = {
                "strategy": strategy,
                "deployed_at": datetime.now(),
                "status": "active"
            }
        
        logger.info(f"Deployed optimization {deployment_id}: {strategy.name}")
        return deployment_id
    
    def check_rollback_conditions(self, deployment_id: str, 
                                 metrics: SystemMetrics) -> bool:
        """Check if optimization should be rolled back"""
        if deployment_id not in self.active_optimizations:
            return False
        
        # Check for severe regression
        if metrics.query_latency_ms > 100:  # Latency too high
            return True
        
        if metrics.classification_accuracy < 0.7:  # Accuracy too low
            return True
        
        if metrics.error_rate > 0.05:  # Too many errors
            return True
        
        return False
    
    def document_learning(self, experiment: Dict[str, Any], 
                         output_path: Path):
        """Document learnings from an experiment"""
        learning = {
            "timestamp": datetime.now().isoformat(),
            "hypothesis": experiment.get("hypothesis", ""),
            "strategy": {
                "name": experiment["strategy"].name,
                "parameters": experiment["strategy"].parameters
            },
            "outcome": {
                "improvement": experiment["result"].improvement,
                "confidence": experiment["result"].confidence,
                "winner": experiment["result"].winner
            },
            "insights": self._generate_insights(experiment)
        }
        
        self.learnings.append(learning)
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(learning, f, indent=2)
        
        logger.info(f"Documented learning: {learning['hypothesis']}")
    
    def _generate_insights(self, experiment: Dict[str, Any]) -> List[str]:
        """Generate insights from experiment results"""
        insights = []
        
        if experiment["result"].winner == "treatment":
            insights.append(f"Strategy '{experiment['strategy'].name}' improved performance by {experiment['result'].improvement:.1%}")
        
        if experiment["result"].confidence > 0.95:
            insights.append("High confidence in results - consider permanent deployment")
        
        if experiment["result"].effect_size > 0.8:
            insights.append("Large effect size indicates substantial practical significance")
        
        return insights
    
    def run_improvement_loop(self, blocking: bool = True):
        """Run continuous improvement loop"""
        self._running = True
        
        def loop():
            while self._running:
                if not self.improvement_iteration():
                    break
                time.sleep(60)  # Wait between iterations
        
        if blocking:
            loop()
        else:
            thread = threading.Thread(target=loop)
            thread.start()
    
    def improvement_iteration(self) -> bool:
        """Single iteration of improvement loop"""
        # This would be overridden in tests
        # In production, would:
        # 1. Collect metrics
        # 2. Identify opportunities
        # 3. Generate strategies
        # 4. Run experiments
        # 5. Deploy winners
        return True
    
    def detect_and_heal(self, metrics: SystemMetrics) -> List[str]:
        """Detect issues and apply self-healing"""
        healing_actions = []
        
        if metrics.error_rate > 0.10:
            healing_actions.append("restart_failed_services")
            healing_actions.append("clear_cache")
        
        if metrics.memory_usage_mb > 900:
            healing_actions.append("garbage_collection")
            healing_actions.append("reduce_cache_size")
        
        if metrics.query_latency_ms > 100:
            healing_actions.append("warm_cache")
            healing_actions.append("scale_resources")
        
        return healing_actions
    
    def predict_performance(self, hours_ahead: int = 1) -> Dict[str, Any]:
        """Predict future performance based on trends"""
        recent = self.performance_monitor.get_recent_metrics(minutes=60)
        
        if len(recent) < 5:
            return {
                "query_latency_ms": 15,
                "throughput_qps": 100,
                "confidence_interval": (10, 20)
            }
        
        # Simple linear extrapolation (in production, use proper time series model)
        latencies = [m.query_latency_ms for m in recent]
        throughputs = [m.throughput_qps for m in recent]
        
        # Calculate trends
        latency_trend = np.polyfit(range(len(latencies)), latencies, 1)[0]
        throughput_trend = np.polyfit(range(len(throughputs)), throughputs, 1)[0]
        
        # Predict
        predicted_latency = latencies[-1] + latency_trend * hours_ahead
        predicted_throughput = throughputs[-1] + throughput_trend * hours_ahead
        
        return {
            "query_latency_ms": predicted_latency,
            "throughput_qps": predicted_throughput,
            "confidence_interval": (predicted_latency * 0.8, predicted_latency * 1.2)
        }
    
    def schedule_optimization(self, strategy: OptimizationStrategy, 
                            run_at: datetime) -> str:
        """Schedule an optimization to run at a specific time"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock:
            self.scheduled_tasks.append({
                "task_id": task_id,
                "strategy": strategy,
                "run_at": run_at,
                "status": "pending"
            })
        
        return task_id
    
    def get_pending_optimizations(self) -> List[Dict[str, Any]]:
        """Get list of pending optimization tasks"""
        with self._lock:
            return [t for t in self.scheduled_tasks 
                   if t["status"] == "pending"]
    
    def analyze_cost_benefit(self, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Analyze cost-benefit of an optimization strategy"""
        expected_benefit = strategy.expected_improvement * 1000  # Value units
        implementation_cost = strategy.implementation_cost
        
        if implementation_cost > 0:
            roi = (expected_benefit - implementation_cost) / implementation_cost
            payback_period = implementation_cost / (expected_benefit / 12)  # Months
        else:
            roi = float('inf')
            payback_period = 0
        
        # Recommendation logic
        if roi > 2:
            recommendation = "proceed"
        elif roi > 0.5:
            recommendation = "review"
        else:
            recommendation = "reject"
        
        return {
            "roi": roi,
            "payback_period": payback_period,
            "expected_benefit": expected_benefit,
            "implementation_cost": implementation_cost,
            "recommendation": recommendation
        }