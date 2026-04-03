"""
Real-Time Monitoring Dashboard Backend
Generates real-time metrics and visualizations for dashboard
"""

from datetime import datetime
from typing import Dict, List
import json
from pathlib import Path
import numpy as np


class DashboardBackend:
    """Backend for real-time monitoring dashboard"""
    
    def __init__(self, data_path: str = "monitoring/dashboard"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        try:
            from core.monitoring_alerts import ProductionMonitor

            monitor = ProductionMonitor()
            alert_summary = monitor.get_alert_summary()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'good' if alert_summary['critical'] == 0 else 'warning' if alert_summary['warning'] == 0 else 'critical',
                'alerts': {
                    'critical': alert_summary['critical'],
                    'warning': alert_summary['warning'],
                    'info': alert_summary['info']
                },
                'uptime_hours': 24  # Placeholder
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_performance_timeline(self, model_name: str, hours: int = 24) -> List[Dict]:
        """Get model performance over time"""
        try:
            from core.drift_detection_advanced import DashboardMetrics
            
            metrics = DashboardMetrics()
            timeline = metrics.get_metrics_timeline(model_name, hours=hours)
            
            return [
                {
                    'timestamp': record['timestamp'],
                    'accuracy': record.get('accuracy', 0),
                    'f1_score': record.get('f1_score', 0),
                    'latency_ms': record.get('latency_ms', 0)
                }
                for record in timeline
            ]
        except Exception:
            return []
    
    def get_prediction_volume(self, model_name: str, granularity: str = 'hourly') -> Dict:
        """Get prediction volume metrics"""
        try:
            from core.model_versioning import PerformanceTracker
            
            tracker = PerformanceTracker()
            stats = tracker.get_performance_stats(model_name, f'{model_name}_v1')
            
            if stats:
                return {
                    'total_predictions': stats['total_predictions'],
                    'predictions_per_hour': stats['total_predictions'] // max(
                        (datetime.fromisoformat(stats['last_prediction']) - 
                         datetime.fromisoformat(stats['first_prediction'])).total_seconds() / 3600, 1
                    ),
                    'accuracy': stats['accuracy'],
                    'error_rate': stats['error_rate'],
                    'avg_confidence': stats['avg_confidence']
                }
        except Exception as e:
            return {'error': str(e)}
        
        return {}
    
    def get_top_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent top alerts"""
        try:
            from core.monitoring_alerts import ProductionMonitor
            
            monitor = ProductionMonitor()
            alerts = monitor.get_recent_alerts(hours=24)
            
            # Sort by severity
            severity_order = {'CRITICAL': 0, 'WARNING': 1, 'INFO': 2}
            sorted_alerts = sorted(
                alerts,
                key=lambda x: severity_order.get(x['severity'], 3)
            )
            
            return sorted_alerts[:limit]
        except Exception:
            return []
    
    def get_drift_analysis(self, model_name: str) -> Dict:
        """Get drift analysis summary"""
        try:
            from core.drift_detection_advanced import AdvancedDriftDetector

            detector = AdvancedDriftDetector()
            drift_trend = detector.get_drift_trend(model_name, hours=24)
            
            if drift_trend:
                latest_drift = drift_trend[-1]
                return {
                    'current_drift_score': latest_drift['drift_score'],
                    'drift_detected': latest_drift['drift_detected'],
                    'trend': 'increasing' if len(drift_trend) > 1 and drift_trend[-1]['drift_score'] > drift_trend[-2]['drift_score'] else 'stable',
                    'hours_monitored': len(drift_trend)
                }
        except Exception as e:
            return {'error': str(e)}
        
        return {}
    
    def get_model_comparison_stats(self) -> Dict:
        """Get comparison statistics across all models"""
        try:
            from utils.model_io import list_saved_models
            
            models = list_saved_models()
            
            model_stats = {}
            for model_name in models[:5]:  # Top 5 models
                model_stats[model_name] = {
                    "status": "active",
                    "last_updated": datetime.now().isoformat(),
                }
            
            return {
                'total_models': len(models),
                'active_models': len(model_stats),
                'models': model_stats
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_resource_utilization(self) -> Dict:
        """Get resource utilization metrics"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'timestamp': datetime.now().isoformat()
            }
        except Exception:
            return {'error': 'Could not retrieve system metrics'}
    
    def get_prediction_distribution(self, model_name: str) -> Dict:
        """Get distribution of predictions"""
        try:
            from core.model_versioning import PerformanceTracker
            
            tracker = PerformanceTracker()
            log_file = tracker.tracking_path / f"{model_name}_v1_predictions.jsonl"
            
            if log_file.exists():
                predictions = []
                with open(log_file, 'r') as f:
                    for line in f:
                        predictions.append(json.loads(line)['prediction'])
                
                unique, counts = np.unique(predictions, return_counts=True)
                
                return {
                    'class_distribution': dict(zip(unique.astype(int).tolist(), counts.tolist())),
                    'total_predictions': len(predictions)
                }
            return {}
        except Exception:
            return {}
    
    def generate_dashboard_json(self, model_name: str) -> Dict:
        """Generate complete dashboard JSON"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'model_name': model_name,
            'performance_timeline': self.get_model_performance_timeline(model_name),
            'prediction_volume': self.get_prediction_volume(model_name),
            'top_alerts': self.get_top_alerts(limit=5),
            'drift_analysis': self.get_drift_analysis(model_name),
            'model_comparison': self.get_model_comparison_stats(),
            'resource_utilization': self.get_resource_utilization(),
            'prediction_distribution': self.get_prediction_distribution(model_name)
        }


class AlertNotificationEngine:
    """Send alerts through multiple channels"""
    
    @staticmethod
    def send_email_alert(alert_message: str, recipients: List[str]) -> bool:
        """Send email alert (placeholder)"""
        print(f"[EMAIL ALERT] {alert_message}")
        return True
    
    @staticmethod
    def send_slack_alert(alert_message: str, webhook_url: str) -> bool:
        """Send Slack notification (placeholder)"""
        print(f"[SLACK ALERT] {alert_message}")
        return True
    
    @staticmethod
    def send_pagerduty_alert(alert_severity: str, alert_message: str) -> bool:
        """Send PagerDuty alert for critical issues (placeholder)"""
        if alert_severity == 'CRITICAL':
            print(f"[PAGERDUTY] Critical alert: {alert_message}")
        return True
    
    @staticmethod
    def send_alert(alert_type: str, message: str, severity: str, 
                  channels: List[str] = None) -> bool:
        """Send alert through configured channels"""
        channels = channels or ['log']
        
        for channel in channels:
            if channel == 'email':
                AlertNotificationEngine.send_email_alert(message, [])
            elif channel == 'slack':
                AlertNotificationEngine.send_slack_alert(message, '')
            elif channel == 'pagerduty':
                AlertNotificationEngine.send_pagerduty_alert(severity, message)
            elif channel == 'log':
                print(f"[{severity}] {message}")
        
        return True


class MetricsExporter:
    """Export metrics for external monitoring tools"""
    
    @staticmethod
    def export_prometheus_metrics(model_name: str) -> str:
        """Export metrics in Prometheus format"""
        dashboard = DashboardBackend()
        data = dashboard.generate_dashboard_json(model_name)
        
        metrics = []
        
        # System metrics
        system = data['system_status']
        metrics.append(f"metaai_system_health{{model=\"{model_name}\"}} {1 if system['overall_health'] == 'good' else 0}")
        
        # Performance metrics
        perf = data['prediction_volume']
        if 'accuracy' in perf:
            metrics.append(f"metaai_model_accuracy{{model=\"{model_name}\"}} {perf['accuracy']}")
            metrics.append(f"metaai_model_error_rate{{model=\"{model_name}\"}} {perf['error_rate']}")
        
        # Drift metrics
        drift = data['drift_analysis']
        if 'current_drift_score' in drift:
            metrics.append(f"metaai_data_drift_score{{model=\"{model_name}\"}} {drift['current_drift_score']}")
        
        # Resource metrics
        resources = data['resource_utilization']
        if 'cpu_usage_percent' in resources:
            metrics.append(f"metaai_cpu_usage_percent {resources['cpu_usage_percent']}")
            metrics.append(f"metaai_memory_usage_percent {resources['memory_usage_percent']}")
        
        return '\n'.join(metrics)
    
    @staticmethod
    def export_json_metrics(model_name: str) -> Dict:
        """Export metrics as JSON"""
        dashboard = DashboardBackend()
        return dashboard.generate_dashboard_json(model_name)
