"""
Advanced Ensemble Methods & Model Compression
Voting, stacking, boosting ensemble techniques + quantization
"""

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import List, Dict
import joblib


class VotingEnsemble:
    """Voting ensemble combining multiple models"""
    
    def __init__(self, models: Dict = None):
        self.models = models or {}
        self.ensemble = None
        self.voting_method = 'soft'  # soft (probabilities) or hard (majority)
    
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add model to ensemble"""
        self.models[name] = {'model': model, 'weight': weight}
    
    def build_ensemble(self, voting: str = 'soft') -> VotingClassifier:
        """Build voting ensemble"""
        self.voting_method = voting
        
        estimators = [(name, data['model']) for name, data in self.models.items()]
        weights = [data['weight'] for data in self.models.values()]
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
        return self.ensemble
    
    def fit(self, X, y):
        """Fit ensemble"""
        if self.ensemble is None:
            self.build_ensemble()
        self.ensemble.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.ensemble.predict_proba(X)
    
    def get_ensemble_score(self, X, y) -> float:
        """Get ensemble accuracy"""
        return self.ensemble.score(X, y)
    
    def get_model_weights(self) -> Dict:
        """Get contribution of each model"""
        return {name: data['weight'] for name, data in self.models.items()}


class StackingEnsemble:
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, base_models: List = None, meta_model=None):
        self.base_models = base_models or [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000))
        ]
        self.meta_model = meta_model or LogisticRegression()
        self.ensemble = None
    
    def build_ensemble(self) -> StackingClassifier:
        """Build stacking ensemble"""
        self.ensemble = StackingClassifier(
            estimators=self.base_models,
            final_estimator=self.meta_model,
            cv=5
        )
        return self.ensemble
    
    def fit(self, X, y):
        """Fit stacking ensemble"""
        if self.ensemble is None:
            self.build_ensemble()
        self.ensemble.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.ensemble.predict_proba(X)
    
    def get_base_model_predictions(self, X) -> Dict:
        """Get predictions from each base model"""
        predictions = {}
        for name, model in self.base_models:
            predictions[name] = model.predict(X)
        return predictions


class AdaptiveEnsemble:
    """Adaptive ensemble that learns optimal model for different data regions"""
    
    def __init__(self, models: Dict):
        self.models = models
        self.region_assignment = None
        self.model_performance_by_region = {}
    
    def fit_adaptive(self, X, y, n_regions: int = 3):
        """Fit adaptive ensemble by clustering data into regions"""
        from sklearn.cluster import KMeans
        
        # Cluster data into regions
        clustering = KMeans(n_clusters=n_regions, random_state=42)
        self.region_assignment = clustering.fit_predict(X)
        
        # Train models if not already trained
        for name, model in self.models.items():
            model.fit(X, y)
        
        # Evaluate each model in each region
        for region in range(n_regions):
            mask = self.region_assignment == region
            X_region = X[mask]
            y_region = y[mask]
            
            self.model_performance_by_region[region] = {}
            
            for name, model in self.models.items():
                score = model.score(X_region, y_region)
                self.model_performance_by_region[region][name] = score
    
    def predict(self, X) -> np.ndarray:
        """Make predictions using adaptive ensemble"""
        from sklearn.cluster import KMeans
        
        # Assign test data to regions
        clustering = KMeans(n_clusters=len(self.model_performance_by_region), random_state=42)
        regions = clustering.fit_predict(X)
        
        predictions = np.zeros(len(X))
        
        for i, (sample, region) in enumerate(zip(X, regions)):
            # Find best model for this region
            if region in self.model_performance_by_region:
                best_model = max(
                    self.model_performance_by_region[region],
                    key=self.model_performance_by_region[region].get
                )
                predictions[i] = self.models[best_model].predict([sample])[0]
            else:
                # Fallback to voting
                votes = [model.predict([sample])[0] for model in self.models.values()]
                predictions[i] = max(set(votes), key=votes.count)
        
        return predictions


class ModelCompression:
    """Model compression and quantization techniques"""
    
    @staticmethod
    def quantize_model(model, bit_depth: int = 8) -> Dict:
        """Quantize model weights to lower precision"""
        quantized_model = {
            'type': type(model).__name__,
            'original_size': len(joblib.dumps(model)),
            'bit_depth': bit_depth,
            'quantization_info': {}
        }
        
        # Extract model parameters
        if hasattr(model, 'coef_'):  # Linear model
            quantized_model['coef_'] = ModelCompression._quantize_array(
                model.coef_, bit_depth
            )
            quantized_model['intercept_'] = ModelCompression._quantize_array(
                model.intercept_, bit_depth
            )
        
        if hasattr(model, 'tree_'):  # Tree-based model
            quantized_model['feature_importances_'] = ModelCompression._quantize_array(
                model.feature_importances_, bit_depth
            )
        
        # Calculate compression ratio
        compressed_size = len(str(quantized_model))
        quantized_model['compression_ratio'] = (
            1 - (compressed_size / quantized_model['original_size'])
        ) * 100
        
        return quantized_model
    
    @staticmethod
    def _quantize_array(array: np.ndarray, bit_depth: int) -> List:
        """Quantize numpy array to lower bit depth"""
        if bit_depth == 8:
            # Quantize to 8-bit integers
            min_val = array.min()
            max_val = array.max()
            
            if max_val - min_val == 0:
                quantized = np.zeros_like(array, dtype=np.uint8)
            else:
                normalized = (array - min_val) / (max_val - min_val)
                quantized = (normalized * 255).astype(np.uint8)
            
            return {
                'data': quantized.tolist(),
                'min': float(min_val),
                'max': float(max_val),
                'dtype': 'uint8'
            }
        
        return array.tolist()
    
    @staticmethod
    def prune_model(model, threshold: float = 0.01):
        """Prune small weights from model"""
        pruned_info = {
            'original_params': 0,
            'pruned_params': 0,
            'pruned_percentage': 0
        }
        
        if hasattr(model, 'coef_'):
            original = model.coef_.size
            mask = np.abs(model.coef_) > threshold
            model.coef_ = model.coef_ * mask
            pruned_info['pruned_params'] = original - mask.sum()
            pruned_info['original_params'] = original
            pruned_info['pruned_percentage'] = (pruned_info['pruned_params'] / original) * 100
        
        return pruned_info
    
    @staticmethod
    def estimate_model_size(model) -> Dict:
        """Estimate model memory footprint"""
        serialized = joblib.dumps(model)
        
        size_bytes = len(serialized)
        
        return {
            'size_bytes': size_bytes,
            'size_kb': size_bytes / 1024,
            'size_mb': size_bytes / (1024 * 1024),
            'can_fit_edge': size_bytes < 50 * 1024 * 1024,  # 50MB threshold
            'parameters_estimate': size_bytes / 8  # Rough estimate
        }


class KnowledgeDistillation:
    """Knowledge distillation: train small model to mimic large model"""
    
    def __init__(self, teacher_model, temperature: float = 4.0):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.student_model = None
    
    def generate_soft_targets(self, X, temperature: float = None) -> np.ndarray:
        """Generate soft targets from teacher model"""
        if temperature is None:
            temperature = self.temperature
        
        logits = self.teacher_model.predict_proba(X)
        # Apply temperature scaling
        soft_targets = np.power(logits, 1 / temperature)
        soft_targets = soft_targets / soft_targets.sum(axis=1, keepdims=True)
        
        return soft_targets
    
    def train_student(self, student_model, X_train, y_train, X_val, y_val,
                     distillation_weight: float = 0.5):
        """Train student model with knowledge distillation"""
        
        # Train student model (hard labels; soft-target distillation can extend this path)
        student_model.fit(X_train, y_train)
        
        # Evaluate
        student_score = student_model.score(X_val, y_val)
        teacher_score = self.teacher_model.score(X_val, y_val)
        
        return {
            'student_score': student_score,
            'teacher_score': teacher_score,
            'performance_gap': teacher_score - student_score,
            'model_size_reduction': self._calculate_size_reduction(student_model)
        }
    
    def _calculate_size_reduction(self, student_model) -> float:
        """Calculate size reduction vs teacher"""
        teacher_size = len(joblib.dumps(self.teacher_model))
        student_size = len(joblib.dumps(student_model))
        return (1 - student_size / teacher_size) * 100
