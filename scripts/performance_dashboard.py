#!/usr/bin/env python
"""
Performance Dashboard for RAG Meta-Learning System
Sprint 4: Visualization of autonomous operation metrics
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse

sys.path.append(str(Path(__file__).parent.parent))


class PerformanceDashboard:
    """ASCII dashboard for monitoring system performance"""
    
    def __init__(self, report_path: str = "autonomous_operation_report.json"):
        self.report_path = Path(report_path)
        self.data = self.load_report()
        
    def load_report(self) -> Dict[str, Any]:
        """Load operation report"""
        if self.report_path.exists():
            with open(self.report_path, 'r') as f:
                return json.load(f)
        return {}
        
    def render_header(self):
        """Render dashboard header"""
        print("\n" + "="*80)
        print("📊 RAG META-LEARNING SYSTEM - PERFORMANCE DASHBOARD")
        print("="*80)
        
        if self.data:
            summary = self.data.get("operation_summary", {})
            print(f"Start Time: {summary.get('start_time', 'N/A')}")
            print(f"Duration: {summary.get('duration_hours', 0):.2f} hours")
            print(f"Status: {summary.get('status', 'unknown').upper()}")
            
    def render_metrics_bar(self, label: str, value: float, target: float, 
                          unit: str = "", width: int = 40):
        """Render a metric as an ASCII bar chart"""
        percentage = min(100, (value / target) * 100) if target > 0 else 0
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        
        status = "✅" if value <= target else "⚠️"
        print(f"\n{label:30} {status}")
        print(f"[{bar}] {value:.1f}{unit} / {target:.1f}{unit} target")
        print(f"Performance: {percentage:.0f}%")
        
    def render_performance_metrics(self):
        """Render performance metrics section"""
        print("\n" + "-"*80)
        print("⚡ PERFORMANCE METRICS")
        print("-"*80)
        
        if not self.data:
            print("No performance data available")
            return
            
        perf = self.data.get("performance", {})
        
        # Latency
        self.render_metrics_bar(
            "Query Latency",
            perf.get("avg_latency", 0),
            50,  # target
            "ms"
        )
        
        # Accuracy
        accuracy_pct = perf.get("avg_accuracy", 0) * 100
        self.render_metrics_bar(
            "Classification Accuracy",
            accuracy_pct,
            85,  # target percentage
            "%"
        )
        
        # Throughput
        print(f"\n{'Throughput':30} 📈")
        print(f"Average: {perf.get('avg_throughput', 0):.0f} QPS")
        
    def render_operations_summary(self):
        """Render operations summary"""
        print("\n" + "-"*80)
        print("🔧 OPERATIONS SUMMARY")
        print("-"*80)
        
        if not self.data:
            print("No operations data available")
            return
            
        stats = self.data.get("statistics", {})
        
        # Create simple table
        operations = [
            ("Queries Processed", stats.get("queries_processed", 0), "🔍"),
            ("Documents Indexed", stats.get("documents_indexed", 0), "📄"),
            ("Optimizations", stats.get("optimizations_performed", 0), "⚡"),
            ("Retraining Cycles", stats.get("retraining_cycles", 0), "🔄"),
            ("Self-Healing Actions", stats.get("self_healing_actions", 0), "🏥"),
        ]
        
        for name, value, icon in operations:
            bar_length = min(40, value * 2)  # Scale for visualization
            bar = "=" * bar_length
            print(f"{icon} {name:25} [{bar}] {value}")
            
    def render_drift_status(self):
        """Render drift detection status"""
        print("\n" + "-"*80)
        print("📈 DRIFT DETECTION STATUS")
        print("-"*80)
        
        if not self.data:
            print("No drift data available")
            return
            
        drift = self.data.get("drift_status", {})
        
        status = "🔴 DRIFT DETECTED" if drift.get("drift_detected") else "🟢 STABLE"
        print(f"Status: {status}")
        print(f"Current Accuracy: {drift.get('current_accuracy', 0):.1%}")
        print(f"Avg Confidence: {drift.get('avg_confidence', 0):.1%}")
        print(f"Total Observations: {drift.get('total_observations', 0)}")
        
    def render_improvement_trends(self):
        """Render improvement trends"""
        print("\n" + "-"*80)
        print("📊 IMPROVEMENT TRENDS")
        print("-"*80)
        
        if not self.data:
            print("No trend data available")
            return
            
        perf = self.data.get("performance", {})
        
        # Simple ASCII sparkline for metrics
        print("\nLatency Trend (lower is better):")
        avg_latency = perf.get("avg_latency", 0)
        p95_latency = perf.get("p95_latency", 0)
        
        # Create simple comparison
        if avg_latency < 50:
            trend = "📉 Excellent - Below target"
        elif avg_latency < 75:
            trend = "➡️ Good - Near target"
        else:
            trend = "📈 Needs improvement"
            
        print(f"  Average: {avg_latency:.1f}ms")
        print(f"  P95: {p95_latency:.1f}ms")
        print(f"  Trend: {trend}")
        
    def render_recommendations(self):
        """Render system recommendations"""
        print("\n" + "-"*80)
        print("💡 RECOMMENDATIONS")
        print("-"*80)
        
        if not self.data:
            print("No data for recommendations")
            return
            
        perf = self.data.get("performance", {})
        stats = self.data.get("statistics", {})
        
        recommendations = []
        
        # Check metrics and add recommendations
        if perf.get("avg_latency", 100) > 75:
            recommendations.append("• Consider optimizing query processing pipeline")
        
        if perf.get("avg_accuracy", 0) < 0.85:
            recommendations.append("• Retrain classifiers with more diverse data")
            
        if stats.get("self_healing_actions", 0) > 5:
            recommendations.append("• Investigate root cause of frequent healing actions")
            
        if stats.get("optimizations_performed", 0) == 0:
            recommendations.append("• Enable automatic optimization strategies")
            
        if not recommendations:
            print("✨ System performing optimally! No recommendations at this time.")
        else:
            for rec in recommendations:
                print(rec)
                
    def render_footer(self):
        """Render dashboard footer"""
        print("\n" + "="*80)
        print("Generated at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*80)
        
    def render(self):
        """Render complete dashboard"""
        self.render_header()
        self.render_performance_metrics()
        self.render_operations_summary()
        self.render_drift_status()
        self.render_improvement_trends()
        self.render_recommendations()
        self.render_footer()
        

def main():
    """Main entry point for dashboard"""
    parser = argparse.ArgumentParser(description="Performance Dashboard")
    parser.add_argument(
        "--report",
        default="autonomous_operation_report.json",
        help="Path to operation report JSON file"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - refresh every 5 seconds"
    )
    
    args = parser.parse_args()
    
    dashboard = PerformanceDashboard(args.report)
    
    if args.watch:
        import time
        import os
        
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            dashboard = PerformanceDashboard(args.report)
            dashboard.render()
            print("\n[Press Ctrl+C to exit watch mode]")
            time.sleep(5)
    else:
        dashboard.render()


if __name__ == "__main__":
    main()