"""
Test-Driven Development: Autonomous System Tests
Sprint 4: Complete System Integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.autonomous_system import AutonomousRAGSystem
from src.optimization.optimization_engine import SystemMetrics


class TestAutonomousSystem:
    """Unit tests for autonomous RAG system"""

    @pytest.fixture
    def system(self):
        """Create an AutonomousRAGSystem instance"""
        return AutonomousRAGSystem()

    @pytest.mark.unit
    def test_system_initialization(self, system):
        """Test that AutonomousRAGSystem initializes correctly"""
        assert system is not None
        assert hasattr(system, 'optimization_engine')
        assert hasattr(system, 'retrieval_engine')
        assert hasattr(system, 'classifier')
        assert hasattr(system, 'data_generator')
        assert system.running is False
        assert system.start_time is None
        
    @pytest.mark.unit
    def test_metrics_collection(self, system):
        """Test system metrics collection"""
        metrics = system.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.query_latency_ms > 0
        assert 0 <= metrics.classification_accuracy <= 1
        assert 0 <= metrics.retrieval_precision <= 1
        assert metrics.error_rate >= 0
        
    @pytest.mark.unit
    def test_improvement_opportunities_identification(self, system):
        """Test identification of improvement opportunities"""
        # Create metrics exceeding targets
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            query_latency_ms=75,  # Above target of 50ms
            classification_accuracy=0.80,  # Below target of 0.85
            retrieval_precision=0.75,
            error_rate=0.02,  # Above target of 0.01
        )
        
        opportunities = system.identify_improvement_opportunities(metrics)
        
        assert len(opportunities) >= 3
        assert any(opp['metric'] == 'query_latency_ms' for opp in opportunities)
        assert any(opp['metric'] == 'classification_accuracy' for opp in opportunities)
        assert any(opp['metric'] == 'error_rate' for opp in opportunities)
        
    @pytest.mark.unit
    def test_self_optimization_execution(self, system):
        """Test self-optimization execution"""
        opportunity = {
            'metric': 'query_latency_ms',
            'current_value': 75,
            'target_value': 50,
            'severity': 'high'
        }
        
        # Mock the optimization engine methods
        with patch.object(system.optimization_engine, 'generate_optimization_strategies') as mock_gen:
            from src.optimization.optimization_engine import OptimizationStrategy
            
            mock_strategy = OptimizationStrategy(
                name="test_strategy",
                parameters={"test": True},
                expected_improvement=0.20
            )
            mock_gen.return_value = [mock_strategy]
            
            with patch.object(system.optimization_engine, 'deploy_optimization'):
                system.perform_self_optimization(opportunity)
                
                # Verify optimization was attempted
                mock_gen.assert_called_once()
                
    @pytest.mark.unit  
    def test_self_healing(self, system):
        """Test self-healing capabilities"""
        # Create error condition metrics
        error_metrics = SystemMetrics(
            timestamp=datetime.now(),
            query_latency_ms=20,
            error_rate=0.15  # High error rate
        )
        
        with patch.object(system.optimization_engine, 'detect_and_heal') as mock_heal:
            mock_heal.return_value = ["restart_service", "clear_cache"]
            
            system.perform_self_healing(error_metrics)
            
            mock_heal.assert_called_once_with(error_metrics)
            assert system.operation_stats['self_healing_actions'] > 0
            
    @pytest.mark.unit
    def test_workload_simulation(self, system):
        """Test workload simulation"""
        initial_queries = system.operation_stats['queries_processed']
        
        system.simulate_workload()
        
        assert system.operation_stats['queries_processed'] > initial_queries
        
    @pytest.mark.unit
    def test_final_report_generation(self, system, tmp_path):
        """Test final report generation"""
        # Set up system state
        system.start_time = datetime.now()
        system.operation_stats['queries_processed'] = 100
        system.operation_stats['optimizations_performed'] = 5
        
        # Change to tmp directory for report
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            system.generate_final_report()
            
            # Check report file was created
            report_path = tmp_path / "autonomous_operation_report.json"
            assert report_path.exists()
            
            # Verify report content
            with open(report_path, 'r') as f:
                report = json.load(f)
                
            assert 'operation_summary' in report
            assert 'statistics' in report
            assert report['statistics']['queries_processed'] == 100
            assert report['statistics']['optimizations_performed'] == 5
        finally:
            os.chdir(original_dir)
            
    @pytest.mark.unit
    def test_targets_configuration(self, system):
        """Test performance targets are properly configured"""
        assert system.targets['query_latency_ms'] == 50
        assert system.targets['classification_accuracy'] == 0.85
        assert system.targets['retrieval_precision'] == 0.80
        assert system.targets['error_rate'] == 0.01