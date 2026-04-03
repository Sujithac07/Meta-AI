"""
Advanced Drift Detection & Real-time Dashboard
Statistical monitoring, visualizations, and anomaly detection
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class AdvancedDriftDetector:
    """Advanced statistical drift detection"""
    
    def __init__(self):
        self.baseline_stats = {}
        self.drift_history = {}
    
    def set_baseline(self, data: pd.DataFrame, model_name: str):
        """Set baseline statistics"""
        self.baseline_stats[model_name] = {
            'mean': data.mean().to_dict(),
            'std': data.std().to_dict(),
            'min': data.min().to_dict(),
            'max': data.max().to_dict(),
            'median': data.median().to_dict(),
            'q25': data.quantile(0.25).to_dict(),
            'q75': data.quantile(0.75).to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        self.drift_history[model_name] = []
    
    def kolmogorov_smirnov_test(self, baseline_data: np.ndarray, 
                               current_data: np.ndarray) -> Tuple[float, float]:
        """KS test for distributional drift"""
        statistic, p_value = stats.ks_2samp(baseline_data, current_data)
        return statistic, p_value
    
    def wasserstein_distance(self, baseline_data: np.ndarray,
                            current_data: np.ndarray) -> float:
        """Wasserstein distance between distributions"""
        return stats.wasserstein_distance(baseline_data, current_data)
    
    def hellinger_distance(self, baseline_data: np.ndarray,
                          current_data: np.ndarray) -> float:
        """Hellinger distance between distributions"""
        # Binned version for continuous data
        bins = min(len(baseline_data), len(current_data)) // 10
        if bins < 5:
            bins = 5
        
        hist_baseline, bin_edges = np.histogram(baseline_data, bins=bins)
        hist_current, _ = np.histogram(current_data, bins=bin_edges)
        
        # Normalize
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_current = hist_current / hist_current.sum()
        
        # Hellinger distance
        distance = np.sqrt(np.sum((np.sqrt(hist_baseline) - np.sqrt(hist_current)) ** 2)) / np.sqrt(2)
        
        return distance
    
    def chi_square_test(self, baseline_counts: Dict, current_counts: Dict) -> Tuple[float, float]:
        """Chi-square test for categorical drift"""
        all_categories = set(baseline_counts.keys()) | set(current_counts.keys())
        
        observed = [current_counts.get(cat, 0) for cat in all_categories]
        expected = [baseline_counts.get(cat, 0) for cat in all_categories]
        
        if sum(expected) == 0:
            return 0, 1
        
        # Normalize expected to match observed total
        expected = np.array(expected) * sum(observed) / sum(expected)
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        return chi2, p_value
    
    def psi_divergence(self, baseline_pct: np.ndarray,
                       current_pct: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index (PSI)"""
        # Avoid log(0)
        baseline_pct = np.where(baseline_pct == 0, 1e-10, baseline_pct)
        current_pct = np.where(current_pct == 0, 1e-10, current_pct)
        
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return psi
    
    def detect_drift(self, model_name: str, current_data: pd.DataFrame,
                    test_methods: List[str] = None) -> Dict:
        """Detect drift using multiple statistical tests"""
        
        if model_name not in self.baseline_stats:
            return {'error': 'No baseline set'}
        
        if test_methods is None:
            test_methods = ['ks', 'wasserstein', 'hellinger', 'psi']
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'tests': {}
        }
        
        baseline_stats = self.baseline_stats[model_name]
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        # KS Test
        if 'ks' in test_methods:
            ks_scores = []
            for col in numeric_cols:
                if col in baseline_stats['mean']:
                    # Approximate baseline data from statistics
                    ks_stat, p_val = self.kolmogorov_smirnov_test(
                        np.random.normal(baseline_stats['mean'][col], 
                                       baseline_stats['std'][col], 1000),
                        current_data[col].values
                    )
                    ks_scores.append(ks_stat)
            
            drift_results['tests']['ks'] = {
                'mean_statistic': np.mean(ks_scores) if ks_scores else 0,
                'max_statistic': np.max(ks_scores) if ks_scores else 0,
                'drift_detected': np.mean(ks_scores) > 0.15 if ks_scores else False
            }
        
        # PSI
        if 'psi' in test_methods:
            psi_scores = []
            for col in numeric_cols:
                if col in baseline_stats['mean']:
                    baseline_bins = np.histogram(
                        np.random.normal(baseline_stats['mean'][col],
                                       baseline_stats['std'][col], 10000),
                        bins=10
                    )[0]
                    current_bins = np.histogram(current_data[col].values, bins=10)[0]
                    
                    baseline_pct = baseline_bins / baseline_bins.sum()
                    current_pct = current_bins / current_bins.sum()
                    
                    psi = self.psi_divergence(baseline_pct, current_pct)
                    psi_scores.append(psi)
            
            drift_results['tests']['psi'] = {
                'mean_psi': np.mean(psi_scores) if psi_scores else 0,
                'max_psi': np.max(psi_scores) if psi_scores else 0,
                'drift_detected': np.mean(psi_scores) > 0.1 if psi_scores else False
            }
        
        # Overall drift score (0-1)
        test_scores = []
        if 'ks' in drift_results['tests']:
            test_scores.append(min(drift_results['tests']['ks']['mean_statistic'], 1))
        if 'psi' in drift_results['tests']:
            test_scores.append(min(drift_results['tests']['psi']['mean_psi'] / 10, 1))
        
        drift_results['overall_drift_score'] = np.mean(test_scores) if test_scores else 0
        drift_results['drift_detected'] = drift_results['overall_drift_score'] > 0.3
        
        # Store in history
        if model_name in self.drift_history:
            self.drift_history[model_name].append(drift_results)
        
        return drift_results
    
    def get_drift_trend(self, model_name: str, hours: int = 24) -> List[Dict]:
        """Get drift trend over time"""
        if model_name not in self.drift_history:
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        trend = []
        
        for record in self.drift_history[model_name]:
            record_time = datetime.fromisoformat(record['timestamp'])
            if record_time > cutoff:
                trend.append({
                    'timestamp': record['timestamp'],
                    'drift_score': record['overall_drift_score'],
                    'drift_detected': record['drift_detected']
                })
        
        return trend


class FeatureDriftAnalyzer:
    """Analyze drift at individual feature level"""
    
    def __init__(self):
        self.feature_baseline = {}
        self.feature_drift_history = {}
    
    def set_baseline_features(self, X: pd.DataFrame, model_name: str):
        """Set baseline feature statistics"""
        self.feature_baseline[model_name] = {}
        
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:
                self.feature_baseline[model_name][col] = {
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std()),
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'median': float(X[col].median())
                }
        
        self.feature_drift_history[model_name] = {}
    
    def analyze_feature_drift(self, X: pd.DataFrame, model_name: str) -> Dict:
        """Analyze drift for each feature"""
        if model_name not in self.feature_baseline:
            return {}
        
        feature_drift = {
            'timestamp': datetime.now().isoformat(),
            'features': {}
        }
        
        baseline = self.feature_baseline[model_name]
        
        for col in X.columns:
            if col in baseline:
                current_mean = X[col].mean()
                current_std = X[col].std()
                
                baseline_mean = baseline[col]['mean']
                baseline_std = baseline[col]['std']
                
                # Z-score shift
                mean_shift = abs(current_mean - baseline_mean) / (baseline_std + 1e-10)
                
                # Coefficient of variation
                baseline_cv = baseline_std / (abs(baseline_mean) + 1e-10)
                current_cv = current_std / (abs(current_mean) + 1e-10)
                cv_shift = abs(current_cv - baseline_cv) / (baseline_cv + 1e-10)
                
                drift_score = (mean_shift + cv_shift) / 2
                
                feature_drift['features'][col] = {
                    'drift_score': float(drift_score),
                    'mean_shift': float(mean_shift),
                    'cv_shift': float(cv_shift),
                    'current_mean': float(current_mean),
                    'baseline_mean': float(baseline_mean),
                    'drift_detected': drift_score > 2  # > 2 std
                }
        
        # Store history
        if model_name not in self.feature_drift_history:
            self.feature_drift_history[model_name] = {}
        
        self.feature_drift_history[model_name][feature_drift['timestamp']] = feature_drift
        
        return feature_drift
    
    def get_top_drifting_features(self, model_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get features with most drift"""
        if model_name not in self.feature_drift_history:
            return []
        
        latest_analysis = list(self.feature_drift_history[model_name].values())[-1]
        
        features_by_drift = sorted(
            latest_analysis['features'].items(),
            key=lambda x: x[1]['drift_score'],
            reverse=True
        )
        
        return [(f, data['drift_score']) for f, data in features_by_drift[:top_k]]


class DashboardMetrics:
    """Real-time dashboard metrics"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def record_metrics(self, model_name: str, metrics: Dict):
        """Record model metrics"""
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = []
        
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history[model_name].append(metrics)
        
        # Keep only last 1000 records
        if len(self.metrics_history[model_name]) > 1000:
            self.metrics_history[model_name] = self.metrics_history[model_name][-1000:]
    
    def get_dashboard_summary(self, model_name: str) -> Dict:
        """Get dashboard summary"""
        if model_name not in self.metrics_history or not self.metrics_history[model_name]:
            return {}
        
        history = self.metrics_history[model_name]
        latest = history[-1]
        
        # Calculate trends
        if len(history) > 1:
            prev = history[-2] if len(history) > 1 else history[0]
            accuracy_trend = latest.get('accuracy', 0) - prev.get('accuracy', 0)
        else:
            accuracy_trend = 0
        
        return {
            'current_metrics': latest,
            'accuracy_trend': accuracy_trend,
            'history_length': len(history),
            'last_update': datetime.fromisoformat(latest['timestamp']),
            'health_status': 'good' if latest.get('accuracy', 0) > 0.8 else 'warning' if latest.get('accuracy', 0) > 0.7 else 'critical'
        }
    
    def get_metrics_timeline(self, model_name: str, hours: int = 24) -> List[Dict]:
        """Get metrics over time"""
        if model_name not in self.metrics_history:
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        timeline = []
        
        for record in self.metrics_history[model_name]:
            record_time = datetime.fromisoformat(record['timestamp'])
            if record_time > cutoff:
                timeline.append(record)
        
        return timeline
