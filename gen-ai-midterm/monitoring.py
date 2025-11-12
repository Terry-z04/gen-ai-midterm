#!/usr/bin/env python3
"""
Logging, Evaluation & Monitoring System
Group 5 | UChicago MS-ADS RAG System

Features:
- Query and response logging
- API usage and cost tracking
- Performance metrics
- Evaluation framework
- Alerting system
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryLogger:
    """Log all queries and responses"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.query_log_file = self.log_dir / "queries.jsonl"
        
    def log_query(
        self,
        query: str,
        method: str,
        model: str,
        retrieved_docs: List[Dict],
        scores: Dict,
        routing_decision: str,
        answer: str,
        tokens: Dict,
        latency: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Log a complete query-response cycle"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "method": method,
            "model": model,
            "retrieval": {
                "num_docs": len(retrieved_docs),
                "top_doc_ids": [doc.get('id', 'N/A') for doc in retrieved_docs[:3]],
                "top_scores": [doc.get('final_score', doc.get('distance', 0)) 
                              for doc in retrieved_docs[:3]],
                "routing": routing_decision,
                "confidence_scores": scores
            },
            "answer": answer,
            "tokens": tokens,
            "latency_seconds": latency,
            "success": success,
            "error": error
        }
        
        # Append to JSONL file
        with open(self.query_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"Logged query: {query[:50]}... | Latency: {latency:.2f}s")
        
        return log_entry
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent queries"""
        if not self.query_log_file.exists():
            return []
        
        queries = []
        with open(self.query_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                queries.append(json.loads(line))
        
        return queries[-limit:]
    
    def get_queries_by_date(self, date: str) -> List[Dict]:
        """Get queries for a specific date (YYYY-MM-DD)"""
        if not self.query_log_file.exists():
            return []
        
        queries = []
        with open(self.query_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry['timestamp'].startswith(date):
                    queries.append(entry)
        
        return queries


class CostTracker:
    """Track API usage and costs"""
    
    # Pricing (as of 2024)
    PRICING = {
        'gpt-4o-mini': {
            'input': 0.15 / 1_000_000,   # per token
            'output': 0.60 / 1_000_000
        },
        'gpt-4o': {
            'input': 2.50 / 1_000_000,
            'output': 10.00 / 1_000_000
        },
        'firecrawl': {
            'scrape': 0.002  # per page
        }
    }
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.cost_log_file = self.log_dir / "costs.jsonl"
        
    def log_openai_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate and log OpenAI API cost"""
        
        pricing = self.PRICING.get(model, self.PRICING['gpt-4o-mini'])
        
        prompt_cost = prompt_tokens * pricing['input']
        completion_cost = completion_tokens * pricing['output']
        total_cost = prompt_cost + completion_cost
        
        cost_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "openai",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": total_cost
        }
        
        with open(self.cost_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cost_entry) + '\n')
        
        return total_cost
    
    def log_firecrawl_cost(self, num_pages: int = 1) -> float:
        """Calculate and log Firecrawl API cost"""
        
        cost = num_pages * self.PRICING['firecrawl']['scrape']
        
        cost_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "firecrawl",
            "num_pages": num_pages,
            "cost_usd": cost
        }
        
        with open(self.cost_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cost_entry) + '\n')
        
        return cost
    
    def get_daily_costs(self, date: Optional[str] = None) -> Dict:
        """Get costs for a specific date (defaults to today)"""
        if date is None:
            date = datetime.utcnow().strftime('%Y-%m-%d')
        
        if not self.cost_log_file.exists():
            return {"total": 0, "openai": 0, "firecrawl": 0}
        
        costs = {"total": 0, "openai": 0, "firecrawl": 0}
        
        with open(self.cost_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry['timestamp'].startswith(date):
                    service = entry['service']
                    cost = entry['cost_usd']
                    costs[service] += cost
                    costs['total'] += cost
        
        return costs
    
    def get_monthly_costs(self, year_month: Optional[str] = None) -> Dict:
        """Get costs for a specific month (YYYY-MM)"""
        if year_month is None:
            year_month = datetime.utcnow().strftime('%Y-%m')
        
        if not self.cost_log_file.exists():
            return {"total": 0, "openai": 0, "firecrawl": 0}
        
        costs = {"total": 0, "openai": 0, "firecrawl": 0}
        
        with open(self.cost_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry['timestamp'].startswith(year_month):
                    service = entry['service']
                    cost = entry['cost_usd']
                    costs[service] += cost
                    costs['total'] += cost
        
        return costs


class MetricsTracker:
    """Track performance metrics"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.query_logger = QueryLogger(log_dir)
        
    def calculate_metrics(self, days: int = 7) -> Dict:
        """Calculate metrics for the last N days"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        if not self.query_logger.query_log_file.exists():
            return self._empty_metrics()
        
        queries = []
        with open(self.query_logger.query_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entry_date = datetime.fromisoformat(entry['timestamp'])
                if entry_date >= cutoff_date:
                    queries.append(entry)
        
        if not queries:
            return self._empty_metrics()
        
        # Calculate metrics
        total_queries = len(queries)
        successful_queries = sum(1 for q in queries if q['success'])
        failed_queries = total_queries - successful_queries
        
        latencies = [q['latency_seconds'] for q in queries]
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        
        # Token usage
        total_tokens = sum(q['tokens'].get('total', 0) for q in queries)
        avg_tokens = total_tokens / total_queries if total_queries > 0 else 0
        
        # Routing decisions
        routing_counts = defaultdict(int)
        for q in queries:
            routing = q['retrieval'].get('routing', 'unknown')
            routing_counts[routing] += 1
        
        # Confidence scores
        confidences = []
        for q in queries:
            conf = q['retrieval'].get('confidence_scores', {}).get('confidence')
            if conf is not None:
                confidences.append(conf)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "period_days": days,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "latency": {
                "avg_seconds": avg_latency,
                "p95_seconds": p95_latency,
                "min_seconds": min(latencies),
                "max_seconds": max(latencies)
            },
            "tokens": {
                "total": total_tokens,
                "avg_per_query": avg_tokens
            },
            "routing_distribution": dict(routing_counts),
            "avg_confidence": avg_confidence
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            "period_days": 0,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "success_rate": 0,
            "latency": {"avg_seconds": 0, "p95_seconds": 0, "min_seconds": 0, "max_seconds": 0},
            "tokens": {"total": 0, "avg_per_query": 0},
            "routing_distribution": {},
            "avg_confidence": 0
        }


class AlertSystem:
    """Simple alerting system"""
    
    def __init__(
        self,
        log_dir: str = "./logs",
        error_threshold: int = 5,
        latency_threshold: float = 10.0
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.alert_log_file = self.log_dir / "alerts.jsonl"
        self.error_threshold = error_threshold
        self.latency_threshold = latency_threshold
        
    def check_and_alert(self, metrics: Dict):
        """Check metrics and generate alerts"""
        
        alerts = []
        
        # Check error rate
        if metrics['failed_queries'] >= self.error_threshold:
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"High error rate: {metrics['failed_queries']} failures",
                "value": metrics['failed_queries']
            })
        
        # Check latency
        if metrics['latency']['avg_seconds'] > self.latency_threshold:
            alerts.append({
                "type": "high_latency",
                "severity": "warning",
                "message": f"High average latency: {metrics['latency']['avg_seconds']:.2f}s",
                "value": metrics['latency']['avg_seconds']
            })
        
        # Check if system is down (no queries in last hour)
        if metrics['total_queries'] == 0:
            alerts.append({
                "type": "no_queries",
                "severity": "critical",
                "message": "No queries received in monitoring period",
                "value": 0
            })
        
        # Log alerts
        for alert in alerts:
            alert['timestamp'] = datetime.utcnow().isoformat()
            with open(self.alert_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert) + '\n')
            logger.warning(f"ALERT: {alert['message']}")
        
        return alerts


class MonitoringSystem:
    """Main monitoring system"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.query_logger = QueryLogger(log_dir)
        self.cost_tracker = CostTracker(log_dir)
        self.metrics_tracker = MetricsTracker(log_dir)
        self.alert_system = AlertSystem(log_dir)
        
    def log_query_response(
        self,
        query: str,
        method: str,
        model: str,
        retrieved_docs: List[Dict],
        scores: Dict,
        routing: str,
        answer: str,
        tokens: Dict,
        latency: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Log a complete query-response cycle and track costs"""
        
        # Log query
        self.query_logger.log_query(
            query, method, model, retrieved_docs, scores,
            routing, answer, tokens, latency, success, error
        )
        
        # Track costs
        if tokens.get('prompt') and tokens.get('completion'):
            self.cost_tracker.log_openai_cost(
                model,
                tokens['prompt'],
                tokens['completion']
            )
        
        # Track Firecrawl usage if dynamic retrieval was used
        if routing == 'dynamic':
            self.cost_tracker.log_firecrawl_cost(num_pages=1)
    
    def get_dashboard_data(self, days: int = 7) -> Dict:
        """Get data for monitoring dashboard"""
        
        metrics = self.metrics_tracker.calculate_metrics(days)
        daily_costs = self.cost_tracker.get_daily_costs()
        monthly_costs = self.cost_tracker.get_monthly_costs()
        recent_queries = self.query_logger.get_recent_queries(limit=10)
        
        # Check for alerts
        alerts = self.alert_system.check_and_alert(metrics)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "costs": {
                "today": daily_costs,
                "this_month": monthly_costs
            },
            "recent_queries": recent_queries,
            "active_alerts": alerts
        }
    
    def print_dashboard(self, days: int = 7):
        """Print monitoring dashboard to console"""
        
        data = self.get_dashboard_data(days)
        
        print("\n" + "="*60)
        print("RAG SYSTEM MONITORING DASHBOARD")
        print("="*60)
        
        # Metrics
        print(f"\n📊 Metrics (Last {days} days):")
        metrics = data['metrics']
        print(f"  Total Queries: {metrics['total_queries']}")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Avg Latency: {metrics['latency']['avg_seconds']:.2f}s")
        print(f"  P95 Latency: {metrics['latency']['p95_seconds']:.2f}s")
        print(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
        print(f"  Avg Tokens/Query: {metrics['tokens']['avg_per_query']:.0f}")
        
        # Routing distribution
        print(f"\n🔀 Routing Distribution:")
        for route, count in metrics['routing_distribution'].items():
            pct = (count / metrics['total_queries'] * 100) if metrics['total_queries'] > 0 else 0
            print(f"  {route}: {count} ({pct:.1f}%)")
        
        # Costs
        print(f"\n💰 Costs:")
        print(f"  Today: ${data['costs']['today']['total']:.4f}")
        print(f"    - OpenAI: ${data['costs']['today']['openai']:.4f}")
        print(f"    - Firecrawl: ${data['costs']['today']['firecrawl']:.4f}")
        print(f"  This Month: ${data['costs']['this_month']['total']:.2f}")
        print(f"    - OpenAI: ${data['costs']['this_month']['openai']:.2f}")
        print(f"    - Firecrawl: ${data['costs']['this_month']['firecrawl']:.2f}")
        
        # Alerts
        if data['active_alerts']:
            print(f"\n⚠️  Active Alerts:")
            for alert in data['active_alerts']:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")
        else:
            print(f"\n✅ No active alerts")
        
        print("\n" + "="*60 + "\n")


# Global monitoring instance
monitoring = MonitoringSystem()


def main():
    """Test monitoring system"""
    
    # Example usage
    monitoring.print_dashboard(days=7)


if __name__ == "__main__":
    main()
