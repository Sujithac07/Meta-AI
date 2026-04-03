"""
Integration layer for advanced features with Gradio UI
Connects core ML components to the interface
"""

import pandas as pd
import numpy as np
from typing import Dict


class AdvancedFeaturesIntegration:
    """Integration point for advanced features"""
    
    @staticmethod
    def run_feature_engineering(X: pd.DataFrame, y: np.ndarray,
                               apply_poly: bool = True,
                               apply_stats: bool = True,
                               apply_domain: bool = True) -> Dict:
        """Run feature engineering pipeline"""
        try:
            from core.advanced_features import AdvancedFeatureEngineer
            
            engineer = AdvancedFeatureEngineer()
            
            X_engineered = X.copy()
            
            if apply_poly:
                X_engineered = engineer.generate_interaction_features(X_engineered)
            
            if apply_stats:
                X_engineered = engineer.generate_statistical_features(X_engineered)
            
            if apply_domain:
                X_engineered = engineer.generate_domain_features(X_engineered)
            
            X_engineered, selected = engineer.select_best_features(X_engineered, y)
            
            _, report = engineer.execute_pipeline(X, y, apply_all=True)
            
            return {
                'status': 'success',
                'original_features': report['original_features'],
                'engineered_features': report['engineered_features'],
                'feature_types': list(report['feature_types_created']),
                'steps_applied': report['engineering_steps'],
                'selected_count': len(report['selected_features']),
                'improvement': f"+{report['engineered_features'] - report['original_features']} features"
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def build_voting_ensemble(models_dict: Dict, voting: str = 'soft') -> Dict:
        """Build voting ensemble from trained models"""
        try:
            from core.ensemble_compression import VotingEnsemble
            
            if not models_dict:
                return {'status': 'error', 'message': 'No models available'}
            
            ensemble = VotingEnsemble()
            
            for name, model in models_dict.items():
                ensemble.add_model(name, model)
            
            ensemble.build_ensemble(voting=voting)
            
            return {
                'status': 'success',
                'ensemble_type': 'Voting',
                'voting_method': voting,
                'models_in_ensemble': len(models_dict),
                'model_names': list(models_dict.keys()),
                'weights': ensemble.get_model_weights()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def build_stacking_ensemble(models_dict: Dict) -> Dict:
        """Build stacking ensemble"""
        try:
            from core.ensemble_compression import StackingEnsemble
            
            base_models = [(name, model) for name, model in models_dict.items()]
            
            if not base_models:
                return {'status': 'error', 'message': 'No models for base learners'}
            
            ensemble = StackingEnsemble(base_models=base_models)
            ensemble.build_ensemble()
            
            return {
                'status': 'success',
                'ensemble_type': 'Stacking',
                'base_learners': len(base_models),
                'meta_learner': 'LogisticRegression',
                'base_model_names': list(models_dict.keys())
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def compress_model(model, method: str = 'quantization') -> Dict:
        """Compress model using specified method"""
        try:
            from core.ensemble_compression import ModelCompression
            
            if method == 'quantization':
                compressed = ModelCompression.quantize_model(model, bit_depth=8)
                return {
                    'status': 'success',
                    'method': 'Quantization',
                    'original_size_bytes': compressed['original_size'],
                    'bit_depth': compressed['bit_depth'],
                    'compression_ratio': f"{compressed['compression_ratio']:.1f}%",
                    'compression_info': compressed['quantization_info']
                }
            
            elif method == 'pruning':
                pruned = ModelCompression.prune_model(model, threshold=0.01)
                return {
                    'status': 'success',
                    'method': 'Pruning',
                    'original_params': pruned['original_params'],
                    'pruned_params': pruned['pruned_params'],
                    'pruned_percentage': f"{pruned['pruned_percentage']:.2f}%"
                }
            
            elif method == 'knowledge_distillation':
                size_info = ModelCompression.estimate_model_size(model)
                return {
                    'status': 'success',
                    'method': 'Knowledge Distillation',
                    'model_size_mb': f"{size_info['size_mb']:.2f}",
                    'can_fit_edge': size_info['can_fit_edge'],
                    'parameters_estimate': size_info['parameters_estimate']
                }
            
            else:
                return {'status': 'error', 'message': f'Unknown compression method: {method}'}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def analyze_drift(model_name: str, current_data: pd.DataFrame) -> Dict:
        """Analyze data drift"""
        try:
            from core.drift_detection_advanced import AdvancedDriftDetector, FeatureDriftAnalyzer
            
            drift_detector = AdvancedDriftDetector()
            feature_analyzer = FeatureDriftAnalyzer()
            
            # Need baseline - create synthetic baseline from current data stats
            if not hasattr(drift_detector, 'baseline_stats') or model_name not in drift_detector.baseline_stats:
                drift_detector.set_baseline(current_data, model_name)
                feature_analyzer.set_baseline_features(current_data, model_name)
            
            drift_results = drift_detector.detect_drift(model_name, current_data)
            feature_drift = feature_analyzer.analyze_feature_drift(current_data, model_name)
            
            top_drifting = feature_analyzer.get_top_drifting_features(model_name, top_k=5)
            
            return {
                'status': 'success',
                'overall_drift_score': drift_results.get('overall_drift_score', 0),
                'drift_detected': drift_results.get('drift_detected', False),
                'feature_count': len(feature_drift.get('features', {})),
                'top_drifting_features': top_drifting,
                'drift_tests': drift_results.get('tests', {})
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def get_monitoring_status(model_name: str = None) -> Dict:
        """Get current monitoring status"""
        try:
            from core.monitoring_alerts import ProductionMonitor, AutoRetrainingEngine
            
            monitor = ProductionMonitor()
            retrain_engine = AutoRetrainingEngine()
            
            alerts_summary = monitor.get_alert_summary()
            recent_alerts = monitor.get_recent_alerts(hours=24)
            
            queue = retrain_engine.get_retraining_queue()
            
            return {
                'status': 'success',
                'alerts': {
                    'total': alerts_summary['total_alerts'],
                    'critical': alerts_summary['critical'],
                    'warning': alerts_summary['warning'],
                    'info': alerts_summary['info']
                },
                'recent_alerts_24h': len(recent_alerts),
                'retraining_jobs_pending': len(queue),
                'system_health': 'good' if alerts_summary['critical'] == 0 else 'warning' if alerts_summary['warning'] == 0 else 'critical'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def create_model_version(model, model_name: str, algorithm: str, 
                            metrics: Dict, hyperparams: Dict,
                            training_data_hash: str) -> Dict:
        """Register model version"""
        try:
            from core.model_versioning import ModelRegistry, ModelVersion
            
            registry = ModelRegistry()
            
            # Get next version number
            versions = registry.list_versions(model_name)
            next_version = len(versions) + 1
            
            model_version = ModelVersion(
                model_name=model_name,
                version=next_version,
                algorithm=algorithm,
                metrics=metrics,
                hyperparams=hyperparams,
                training_data_hash=training_data_hash
            )
            
            # Save model
            import tempfile
            import joblib
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                model_path = f.name
                joblib.dump(model, f)
            
            version_key = registry.register_version(model_version, model_path)
            
            return {
                'status': 'success',
                'version_key': version_key,
                'version_number': next_version,
                'model_name': model_name,
                'message': f'Model version {version_key} registered successfully'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
