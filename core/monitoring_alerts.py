"""
Automated Retraining & Production Monitoring
Triggers, alerts, and automated model retraining pipelines
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    CRITICAL = 3


class AlertType(Enum):
    """Types of alerts"""
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_ERROR_RATE = "high_error_rate"
    MODEL_STALENESS = "model_staleness"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PREDICTION_ANOMALY = "prediction_anomaly"


class MonitoringAlert:
    """Represents a monitoring alert"""
    
    def __init__(self, alert_type: AlertType, severity: AlertSeverity,
                 model_name: str, message: str, metadata: Dict = None):
        self.alert_type = alert_type
        self.severity = severity
        self.model_name = model_name
        self.message = message
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.resolved = False
    
    def to_dict(self) -> Dict:
        return {
            'type': self.alert_type.value,
            'severity': self.severity.name,
            'model': self.model_name,
            'message': self.message,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'resolved': self.resolved
        }


class ProductionMonitor:
    """Real-time production monitoring and alerting"""
    
    def __init__(self, monitoring_path: str = "monitoring/alerts"):
        self.monitoring_path = Path(monitoring_path)
        self.monitoring_path.mkdir(parents=True, exist_ok=True)
        
        self.alerts: List[MonitoringAlert] = []
        self.thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'drift_threshold': 0.3,
            'performance_drop': 0.1,  # 10% accuracy drop
            'prediction_latency': 1000,  # ms
            'model_age_days': 30
        }
        self.alert_callbacks: List[Callable] = []
    
    def add_alert_callback(self, callback: Callable):
        """Register callback for new alerts"""
        self.alert_callbacks.append(callback)
    
    def trigger_alert(self, alert: MonitoringAlert):
        """Trigger new alert"""
        self.alerts.append(alert)
        
        # Log alert
        alert_file = self.monitoring_path / "alerts.jsonl"
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert.to_dict()) + '\n')
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def check_error_rate(self, model_name: str, recent_errors: int, 
                        recent_total: int) -> Optional[MonitoringAlert]:
        """Check prediction error rate"""
        if recent_total == 0:
            return None
        
        error_rate = recent_errors / recent_total
        
        if error_rate > self.thresholds['error_rate']:
            alert = MonitoringAlert(
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=AlertSeverity.CRITICAL,
                model_name=model_name,
                message=f"Error rate {error_rate:.2%} exceeds threshold {self.thresholds['error_rate']:.2%}",
                metadata={'error_rate': error_rate, 'recent_errors': recent_errors}
            )
            self.trigger_alert(alert)
            return alert
        
        return None
    
    def check_data_drift(self, model_name: str, drift_score: float) -> Optional[MonitoringAlert]:
        """Check for data drift"""
        if drift_score > self.thresholds['drift_threshold']:
            severity = AlertSeverity.CRITICAL if drift_score > 0.5 else AlertSeverity.WARNING
            
            alert = MonitoringAlert(
                alert_type=AlertType.DATA_DRIFT,
                severity=severity,
                model_name=model_name,
                message=f"Data drift detected (score: {drift_score:.3f})",
                metadata={'drift_score': drift_score}
            )
            self.trigger_alert(alert)
            return alert
        
        return None
    
    def check_performance_degradation(self, model_name: str, 
                                     current_accuracy: float,
                                     baseline_accuracy: float) -> Optional[MonitoringAlert]:
        """Check for model performance degradation"""
        degradation = baseline_accuracy - current_accuracy
        
        if degradation > self.thresholds['performance_drop']:
            alert = MonitoringAlert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.WARNING,
                model_name=model_name,
                message=f"Accuracy dropped from {baseline_accuracy:.2%} to {current_accuracy:.2%}",
                metadata={'degradation': degradation, 'baseline': baseline_accuracy, 'current': current_accuracy}
            )
            self.trigger_alert(alert)
            return alert
        
        return None
    
    def check_model_staleness(self, model_name: str, 
                             last_training_date: datetime) -> Optional[MonitoringAlert]:
        """Check if model is too old"""
        model_age = datetime.now() - last_training_date
        
        if model_age.days > self.thresholds['model_age_days']:
            alert = MonitoringAlert(
                alert_type=AlertType.MODEL_STALENESS,
                severity=AlertSeverity.INFO,
                model_name=model_name,
                message=f"Model is {model_age.days} days old (threshold: {self.thresholds['model_age_days']} days)",
                metadata={'model_age_days': model_age.days}
            )
            self.trigger_alert(alert)
            return alert
        
        return None
    
    def get_recent_alerts(self, hours: int = 24, 
                         severity: AlertSeverity = None) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert.timestamp)
            if alert_time > cutoff:
                if severity is None or alert.severity == severity:
                    recent_alerts.append(alert.to_dict())
        
        return recent_alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        summary = {
            'total_alerts': len(self.alerts),
            'critical': sum(1 for a in self.alerts if a.severity == AlertSeverity.CRITICAL),
            'warning': sum(1 for a in self.alerts if a.severity == AlertSeverity.WARNING),
            'info': sum(1 for a in self.alerts if a.severity == AlertSeverity.INFO),
            'resolved': sum(1 for a in self.alerts if a.resolved),
            'by_type': {}
        }
        
        for alert_type in AlertType:
            count = sum(1 for a in self.alerts if a.alert_type == alert_type)
            if count > 0:
                summary['by_type'][alert_type.value] = count
        
        return summary


class AutoRetrainingEngine:
    """Automatic retraining trigger system"""
    
    def __init__(self, model_registry_path: str = "models/registry"):
        self.model_registry_path = Path(model_registry_path)
        self.retraining_jobs: Dict = {}
        self.retrain_conditions = {
            'error_rate_threshold': 0.1,
            'drift_threshold': 0.3,
            'performance_drop_threshold': 0.15,
            'staleness_days': 30,
            'sample_size_trigger': 1000
        }
    
    def evaluate_retrain_need(self, model_name: str, metrics: Dict) -> Tuple[bool, str]:
        """Evaluate if model needs retraining"""
        reasons = []
        
        # Check error rate
        if metrics.get('error_rate', 0) > self.retrain_conditions['error_rate_threshold']:
            reasons.append(f"High error rate: {metrics['error_rate']:.2%}")
        
        # Check drift
        if metrics.get('drift_score', 0) > self.retrain_conditions['drift_threshold']:
            reasons.append(f"Data drift detected: {metrics['drift_score']:.3f}")
        
        # Check performance
        baseline = metrics.get('baseline_accuracy', 0.9)
        current = metrics.get('current_accuracy', 0.9)
        if (baseline - current) > self.retrain_conditions['performance_drop_threshold']:
            reasons.append(f"Performance drop: {baseline:.2%} → {current:.2%}")
        
        # Check staleness
        if metrics.get('age_days', 0) > self.retrain_conditions['staleness_days']:
            reasons.append(f"Model age: {metrics['age_days']} days")
        
        return len(reasons) > 0, "; ".join(reasons)
    
    def schedule_retraining(self, model_name: str, priority: str = 'normal',
                          reason: str = None) -> str:
        """Schedule model for retraining"""
        job_id = f"{model_name}_{datetime.now().timestamp()}"
        
        self.retraining_jobs[job_id] = {
            'model_name': model_name,
            'scheduled_at': datetime.now().isoformat(),
            'priority': priority,
            'reason': reason,
            'status': 'scheduled'
        }
        
        return job_id
    
    def get_retraining_queue(self) -> List[Dict]:
        """Get pending retraining jobs"""
        return [j for j in self.retraining_jobs.values() if j['status'] == 'scheduled']
    
    def mark_retrain_complete(self, job_id: str, new_metrics: Dict):
        """Mark retraining as complete"""
        if job_id in self.retraining_jobs:
            self.retraining_jobs[job_id]['status'] = 'completed'
            self.retraining_jobs[job_id]['completed_at'] = datetime.now().isoformat()
            self.retraining_jobs[job_id]['new_metrics'] = new_metrics


class PredictionAnomalyDetector:
    """Detect anomalous predictions"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.prediction_history: Dict[str, List] = {}
        self.confidence_baseline: Dict[str, float] = {}
    
    def update_prediction(self, model_name: str, prediction: int, 
                         confidence: float, features_hash: str = None):
        """Record prediction"""
        if model_name not in self.prediction_history:
            self.prediction_history[model_name] = []
        
        self.prediction_history[model_name].append({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'features_hash': features_hash
        })
        
        # Keep only recent predictions
        if len(self.prediction_history[model_name]) > self.window_size:
            self.prediction_history[model_name] = \
                self.prediction_history[model_name][-self.window_size:]
    
    def detect_anomalies(self, model_name: str) -> List[Dict]:
        """Detect anomalous predictions"""
        if model_name not in self.prediction_history:
            return []
        
        history = self.prediction_history[model_name]
        if len(history) < 10:
            return []
        
        confidences = np.array([p['confidence'] for p in history])
        
        # Calculate statistics
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        # Flag predictions with low confidence (> 2 std below mean)
        anomalies = []
        threshold = mean_conf - (2 * std_conf)
        
        for pred in history[-10:]:  # Check recent predictions
            if pred['confidence'] < threshold:
                anomalies.append({
                    'timestamp': pred['timestamp'],
                    'confidence': pred['confidence'],
                    'anomaly_score': abs(mean_conf - pred['confidence']) / (std_conf + 1e-10)
                })
        
        return anomalies
    
    def detect_class_imbalance_shift(self, model_name: str) -> Optional[Dict]:
        """Detect shift in class distribution of predictions"""
        if model_name not in self.prediction_history:
            return None
        
        history = self.prediction_history[model_name]
        if len(history) < 50:
            return None
        
        # Check prediction class distribution
        predictions = np.array([p['prediction'] for p in history])
        unique, counts = np.unique(predictions, return_counts=True)
        
        class_dist = dict(zip(unique, counts / len(predictions)))
        
        # Flag if any class < 5% (unusual for balanced data)
        imbalanced = any(p < 0.05 for p in class_dist.values())
        
        if imbalanced:
            return {
                'class_distribution': class_dist,
                'imbalanced': True,
                'detected_at': datetime.now().isoformat()
            }
        
        return None


# Helper function
def trigger_alert_email(alert: MonitoringAlert):
    """Callback to send alert email (placeholder)"""
    print(f"[ALERT EMAIL] {alert.severity.name}: {alert.message}")
