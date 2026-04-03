"""
Model Versioning & Registry System
Tracks multiple versions of models with performance history and rollback capability
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import hashlib


class ModelVersion:
    """Represents a single model version"""
    
    def __init__(self, model_name: str, version: int, algorithm: str, 
                 metrics: Dict, hyperparams: Dict, training_data_hash: str):
        self.model_name = model_name
        self.version = version
        self.algorithm = algorithm
        self.metrics = metrics
        self.hyperparams = hyperparams
        self.training_data_hash = training_data_hash
        self.created_at = datetime.now().isoformat()
        self.status = "active"  # active, inactive, deprecated
        self.promotion_history: List[str] = []
        self.performance_degradation = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'version': self.version,
            'algorithm': self.algorithm,
            'metrics': self.metrics,
            'hyperparams': self.hyperparams,
            'training_data_hash': self.training_data_hash,
            'created_at': self.created_at,
            'status': self.status,
            'promotion_history': self.promotion_history,
            'performance_degradation': self.performance_degradation
        }


class ModelRegistry:
    """Central registry for all model versions"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "model_registry.json"
        self.versions_path = self.registry_path / "versions"
        self.versions_path.mkdir(exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_version(self, model_version: ModelVersion, model_path: str) -> str:
        """Register new model version"""
        model_name = model_version.model_name
        version = model_version.version
        
        # Create version in registry
        if model_name not in self.registry:
            self.registry[model_name] = {
                'versions': {},
                'latest': None,
                'production': None
            }
        
        version_key = f"v{version}"
        version_dict = model_version.to_dict()
        version_dict['model_path'] = model_path
        
        self.registry[model_name]['versions'][version_key] = version_dict
        self.registry[model_name]['latest'] = version_key
        
        # Copy model to versioned location
        versioned_path = self.versions_path / f"{model_name}_{version_key}.pkl"
        if os.path.exists(model_path):
            joblib.dump(joblib.load(model_path), versioned_path)
        
        self._save_registry()
        
        return version_key
    
    def get_version(self, model_name: str, version: str) -> Optional[Dict]:
        """Get specific model version metadata"""
        if model_name in self.registry:
            return self.registry[model_name]['versions'].get(version)
        return None
    
    def list_versions(self, model_name: str) -> List[Dict]:
        """List all versions of a model"""
        if model_name in self.registry:
            versions = []
            for v_key, v_data in self.registry[model_name]['versions'].items():
                v_data['version_key'] = v_key
                versions.append(v_data)
            return sorted(versions, key=lambda x: x['version'])
        return []
    
    def promote_to_production(self, model_name: str, version: str):
        """Promote version to production"""
        if model_name in self.registry:
            self.registry[model_name]['production'] = version
            # Mark in version history
            if version in self.registry[model_name]['versions']:
                self.registry[model_name]['versions'][version]['promotion_history'].append(
                    f"Promoted to production at {datetime.now().isoformat()}"
                )
            self._save_registry()
    
    def rollback_to_version(self, model_name: str, version: str) -> bool:
        """Rollback to previous version"""
        version_data = self.get_version(model_name, version)
        if version_data:
            self.promote_to_production(model_name, version)
            version_data['promotion_history'].append(
                f"Rolled back to at {datetime.now().isoformat()}"
            )
            self._save_registry()
            return True
        return False
    
    def load_model_version(self, model_name: str, version: str = None):
        """Load model from registry"""
        if version is None:
            version = self.registry[model_name]['production']
        
        versioned_path = self.versions_path / f"{model_name}_{version}.pkl"
        if versioned_path.exists():
            return joblib.load(versioned_path)
        return None
    
    def get_model_comparison(self, model_name: str) -> Dict:
        """Get comparison of all versions"""
        versions = self.list_versions(model_name)
        comparison = {
            'model_name': model_name,
            'versions': []
        }
        
        for v in versions:
            comparison['versions'].append({
                'version': v.get('version'),
                'algorithm': v.get('algorithm'),
                'f1_score': v.get('metrics', {}).get('f1_score'),
                'accuracy': v.get('metrics', {}).get('accuracy'),
                'created_at': v.get('created_at'),
                'status': v.get('status'),
                'production': self.registry[model_name]['production'] == v.get('version_key')
            })
        
        return comparison


class ABTestingFramework:
    """A/B testing framework for comparing model versions"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry = ModelRegistry(registry_path)
        self.ab_tests: Dict = {}
        self.test_results_path = Path(registry_path) / "ab_tests"
        self.test_results_path.mkdir(exist_ok=True)
    
    def create_ab_test(self, test_name: str, model_name: str, 
                      version_a: str, version_b: str, 
                      traffic_split: Tuple[float, float] = (0.5, 0.5)) -> str:
        """Create new A/B test"""
        test_id = hashlib.md5(
            f"{test_name}{datetime.now().isoformat()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:8]
        
        self.ab_tests[test_id] = {
            'test_name': test_name,
            'model_name': model_name,
            'version_a': version_a,
            'version_b': version_b,
            'traffic_split': traffic_split,
            'created_at': datetime.now().isoformat(),
            'status': 'running',
            'results': {
                'version_a': {'predictions': 0, 'correct': 0, 'avg_confidence': 0},
                'version_b': {'predictions': 0, 'correct': 0, 'avg_confidence': 0}
            }
        }
        
        return test_id
    
    def route_prediction(self, test_id: str, features: List) -> Tuple[str, str]:
        """Route prediction to A or B version based on traffic split"""
        if test_id not in self.ab_tests:
            return None, None
        
        import random
        test = self.ab_tests[test_id]
        split_a, split_b = test['traffic_split']
        
        rand = random.random()  # nosec B311
        if rand < split_a:
            version = test['version_a']
            version_label = 'version_a'
        else:
            version = test['version_b']
            version_label = 'version_b'
        
        model = self.registry.load_model_version(test['model_name'], version)
        return model, version_label
    
    def record_prediction(self, test_id: str, version_label: str, 
                         prediction: int, actual: int, confidence: float):
        """Record prediction result for analysis"""
        if test_id in self.ab_tests:
            result = self.ab_tests[test_id]['results'][version_label]
            result['predictions'] += 1
            if prediction == actual:
                result['correct'] += 1
            result['avg_confidence'] = (
                (result['avg_confidence'] * (result['predictions'] - 1) + confidence) / 
                result['predictions']
            )
    
    def get_test_results(self, test_id: str) -> Dict:
        """Get A/B test results and statistical significance"""
        if test_id not in self.ab_tests:
            return None
        
        test = self.ab_tests[test_id]
        results_a = test['results']['version_a']
        results_b = test['results']['version_b']
        
        # Calculate metrics
        def calc_metrics(result):
            if result['predictions'] == 0:
                return {'accuracy': 0, 'confidence': 0}
            return {
                'accuracy': result['correct'] / result['predictions'],
                'confidence': result['avg_confidence']
            }
        
        metrics_a = calc_metrics(results_a)
        metrics_b = calc_metrics(results_b)
        
        # Chi-square test for statistical significance
        from scipy.stats import chi2_contingency
        
        contingency = [
            [results_a['correct'], results_a['predictions'] - results_a['correct']],
            [results_b['correct'], results_b['predictions'] - results_b['correct']]
        ]
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        return {
            'test_name': test['test_name'],
            'status': test['status'],
            'version_a': {
                'version': test['version_a'],
                **metrics_a,
                'samples': results_a['predictions']
            },
            'version_b': {
                'version': test['version_b'],
                **metrics_b,
                'samples': results_b['predictions']
            },
            'statistical_significance': {
                'p_value': p_value,
                'significant': p_value < 0.05,
                'winner': test['version_a'] if metrics_a['accuracy'] > metrics_b['accuracy'] 
                         else test['version_b']
            }
        }
    
    def finish_ab_test(self, test_id: str, winner: str = None):
        """Finish A/B test and promote winner"""
        if test_id not in self.ab_tests:
            return False
        
        test = self.ab_tests[test_id]

        if winner:
            self.registry.promote_to_production(test['model_name'], winner)
            test['status'] = 'finished'
            test['winner'] = winner
            return True
        
        return False


class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry = ModelRegistry(registry_path)
        self.tracking_path = Path(registry_path) / "performance_tracking"
        self.tracking_path.mkdir(exist_ok=True)
    
    def log_prediction(self, model_name: str, version: str, 
                      prediction: int, actual: int, confidence: float,
                      features_hash: str = None):
        """Log prediction for performance tracking"""
        log_file = self.tracking_path / f"{model_name}_{version}_predictions.jsonl"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': int(prediction),
            'actual': int(actual),
            'confidence': float(confidence),
            'correct': int(prediction == actual),
            'features_hash': features_hash
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def calculate_drift(self, model_name: str, version: str, 
                       window_size: int = 100) -> float:
        """Calculate performance drift for model"""
        log_file = self.tracking_path / f"{model_name}_{version}_predictions.jsonl"
        
        if not log_file.exists():
            return 0.0
        
        lines = []
        with open(log_file, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        
        if len(lines) < window_size:
            return 0.0
        
        # Compare recent performance with training performance
        recent = lines[-window_size:]
        recent_accuracy = sum(1 for p in recent if p['correct']) / len(recent)
        
        # Get training accuracy from registry
        version_data = self.registry.get_version(model_name, f"v{version.split('_')[0]}")
        if version_data:
            training_accuracy = version_data['metrics'].get('accuracy', 0.9)
        else:
            training_accuracy = 0.9
        
        drift = training_accuracy - recent_accuracy
        
        # Store drift for alerts
        self.registry.registry[model_name]['versions'][f'v{version.split("_")[0]}']['performance_degradation'] = drift
        
        return max(0, drift)
    
    def get_performance_stats(self, model_name: str, version: str) -> Dict:
        """Get performance statistics"""
        log_file = self.tracking_path / f"{model_name}_{version}_predictions.jsonl"
        
        if not log_file.exists():
            return None
        
        lines = []
        with open(log_file, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        
        if not lines:
            return None
        
        correct = sum(1 for p in lines if p['correct'])
        accuracy = correct / len(lines)
        avg_confidence = sum(p['confidence'] for p in lines) / len(lines)
        
        return {
            'total_predictions': len(lines),
            'accuracy': accuracy,
            'error_rate': 1 - accuracy,
            'avg_confidence': avg_confidence,
            'first_prediction': lines[0]['timestamp'],
            'last_prediction': lines[-1]['timestamp']
        }
