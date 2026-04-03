"""
Smart Pipeline Generator - Autonomous ML Pipeline Design
Analyzes dataset and automatically recommends optimal architecture
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class DatasetAnalyzer:
    """Analyze dataset characteristics"""
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.metrics = {}
    
    def analyze(self) -> Dict:
        """Complete dataset analysis"""
        return {
            'size': self._analyze_size(),
            'types': self._analyze_types(),
            'target': self._analyze_target(),
            'quality': self._analyze_quality(),
            'complexity': self._analyze_complexity()
        }
    
    def _analyze_size(self) -> Dict:
        """Analyze dataset size"""
        rows, cols = self.df.shape
        size_mb = self.df.memory_usage(deep=True).sum() / (1024**2)
        
        if rows < 1000:
            category = "tiny"
        elif rows < 10000:
            category = "small"
        elif rows < 100000:
            category = "medium"
        elif rows < 1000000:
            category = "large"
        else:
            category = "huge"
        
        return {
            'rows': rows,
            'cols': cols,
            'category': category,
            'memory_mb': round(size_mb, 2)
        }
    
    def _analyze_types(self) -> Dict:
        """Analyze feature types"""
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = self.df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric_count': len(numeric),
            'categorical_count': len(categorical),
            'datetime_count': len(datetime_cols),
            'has_text': any(self.df[col].astype(str).str.len().max() > 100 
                          for col in categorical if col != self.target_col)
        }
    
    def _analyze_target(self) -> Dict:
        """Analyze target variable"""
        if self.target_col not in self.df.columns:
            return {'error': 'Target column not found'}
        
        target = self.df[self.target_col]
        
        if target.dtype in ['object', 'category']:
            # Classification
            unique = target.nunique()
            counts = target.value_counts()
            imbalance_ratio = counts.max() / counts.min()
            
            return {
                'type': 'classification',
                'classes': unique,
                'class_names': counts.index.tolist(),
                'imbalance_ratio': imbalance_ratio,
                'is_binary': unique == 2,
                'is_multiclass': unique > 2,
                'imbalance_level': self._classify_imbalance(imbalance_ratio)
            }
        else:
            # Regression
            return {
                'type': 'regression',
                'min': float(target.min()),
                'max': float(target.max()),
                'mean': float(target.mean()),
                'std': float(target.std())
            }
    
    def _analyze_quality(self) -> Dict:
        """Analyze data quality"""
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        duplicates = self.df.duplicated().sum()
        duplicate_pct = (duplicates / len(self.df)) * 100
        
        quality_score = 100 - (missing_pct * 0.5) - (duplicate_pct * 0.5)
        quality_score = max(0, min(100, quality_score))
        
        return {
            'missing_percent': round(missing_pct, 2),
            'duplicate_percent': round(duplicate_pct, 2),
            'quality_score': round(quality_score, 1),
            'quality_level': self._classify_quality(quality_score)
        }
    
    def _analyze_complexity(self) -> Dict:
        """Analyze dataset complexity"""
        rows, cols = self.df.shape
        
        # Dimensionality
        if cols < 10:
            dimensionality = "low"
        elif cols < 50:
            dimensionality = "medium"
        else:
            dimensionality = "high"
        
        # Feature density (non-null ratio)
        density = 1.0 - (self.df.isnull().sum().sum() / (rows * cols))
        
        return {
            'dimensionality': dimensionality,
            'feature_density': round(density, 2),
            'num_features': cols,
            'feature_ratio': round(cols / rows, 4)
        }
    
    @staticmethod
    def _classify_imbalance(ratio: float) -> str:
        if ratio < 1.5:
            return "balanced"
        elif ratio < 3:
            return "slightly_imbalanced"
        elif ratio < 10:
            return "imbalanced"
        else:
            return "severely_imbalanced"
    
    @staticmethod
    def _classify_quality(score: float) -> str:
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        else:
            return "poor"


class ArchitectureRecommender:
    """Recommend optimal ML architecture based on dataset"""
    
    def __init__(self, analysis: Dict):
        self.analysis = analysis
        self.recommendation = {}
    
    def recommend(self) -> Dict:
        """Generate architecture recommendation"""
        
        size = self.analysis['size']['category']
        types = self.analysis['types']
        target = self.analysis['target']
        quality = self.analysis['quality']['quality_level']
        complexity = self.analysis['complexity']['dimensionality']
        
        # Determine problem type
        problem_type = target.get('type', 'classification')
        
        if problem_type == 'classification':
            if target['is_binary']:
                return self._recommend_binary_classification(size, types, quality, complexity)
            else:
                return self._recommend_multiclass_classification(size, types, quality, complexity)
        else:
            return self._recommend_regression(size, types, quality, complexity)
    
    def _recommend_binary_classification(self, size, types, quality, complexity) -> Dict:
        """Recommend for binary classification"""
        
        if size == 'tiny' or (size == 'small' and quality == 'poor'):
            # Small data needs regularization
            algorithm = 'LogisticRegression'
            reason = "Small dataset with poor quality. Logistic Regression prevents overfitting."
            preprocessing = ['handle_missing', 'scale_features', 'remove_outliers']
            
        elif size in ['small', 'medium'] and complexity == 'low':
            algorithm = 'RandomForest'
            reason = "Small-medium dataset with low complexity. Random Forest is robust and interpretable."
            preprocessing = ['handle_missing', 'encode_categorical', 'scale_features']
            
        elif size in ['medium', 'large'] and types['categorical_count'] > 5:
            algorithm = 'LightGBM'
            reason = "Many categorical features. LightGBM handles them efficiently."
            preprocessing = ['handle_missing', 'encode_categorical', 'feature_selection']
            
        elif size == 'large' and complexity == 'high':
            algorithm = 'XGBoost'
            reason = "Large complex dataset. XGBoost is state-of-the-art for tabular data."
            preprocessing = ['handle_missing', 'feature_engineering', 'scale_features']
            
        elif types['has_text']:
            algorithm = 'NaiveBayes'
            reason = "Text features detected. Naive Bayes works well with text data."
            preprocessing = ['handle_missing', 'vectorize_text', 'encode_categorical']
            
        else:
            algorithm = 'RandomForest'
            reason = "Balanced choice for binary classification."
            preprocessing = ['handle_missing', 'encode_categorical', 'scale_features']
        
        return {
            'primary_algorithm': algorithm,
            'secondary_algorithms': self._get_secondary_algorithms(algorithm),
            'reason': reason,
            'preprocessing_steps': preprocessing,
            'hyperparameter_ranges': self._get_hyperparameters(algorithm),
            'expected_metrics': {
                'accuracy': 0.80 if quality == 'good' else 0.75,
                'auc': 0.85 if quality == 'good' else 0.80,
                'f1': 0.78 if quality == 'good' else 0.73
            },
            'training_time_minutes': self._estimate_training_time(size),
            'confidence': 0.95 if quality == 'good' else 0.80
        }
    
    def _recommend_multiclass_classification(self, size, types, quality, complexity) -> Dict:
        """Recommend for multiclass classification"""
        
        if size in ['small', 'medium']:
            algorithm = 'RandomForest'
            reason = "Small-medium dataset. Random Forest handles multiclass well."
            
        elif size in ['medium', 'large'] and complexity == 'high':
            algorithm = 'XGBoost'
            reason = "Large multiclass problem. XGBoost excels at this."
            
        else:
            algorithm = 'LightGBM'
            reason = "Efficient for large multiclass datasets."
        
        return {
            'primary_algorithm': algorithm,
            'secondary_algorithms': self._get_secondary_algorithms(algorithm),
            'reason': reason,
            'preprocessing_steps': ['handle_missing', 'encode_categorical', 'feature_selection'],
            'hyperparameter_ranges': self._get_hyperparameters(algorithm),
            'expected_metrics': {
                'accuracy': 0.75 if quality == 'good' else 0.70
            },
            'training_time_minutes': self._estimate_training_time(size),
            'confidence': 0.90
        }
    
    def _recommend_regression(self, size, types, quality, complexity) -> Dict:
        """Recommend for regression"""
        
        if size in ['small', 'medium']:
            algorithm = 'RandomForest'
            reason = "Small-medium dataset. Random Forest is robust for regression."
            
        elif size in ['medium', 'large']:
            algorithm = 'GradientBoosting'
            reason = "Large dataset. Gradient Boosting provides excellent generalization."
            
        else:
            algorithm = 'LightGBM'
            reason = "Very large dataset. LightGBM is fastest for regression."
        
        return {
            'primary_algorithm': algorithm,
            'secondary_algorithms': self._get_secondary_algorithms(algorithm),
            'reason': reason,
            'preprocessing_steps': ['handle_missing', 'scale_features', 'handle_outliers'],
            'hyperparameter_ranges': self._get_hyperparameters(algorithm),
            'expected_metrics': {
                'r2': 0.85 if quality == 'good' else 0.75,
                'rmse': 0.15 if quality == 'good' else 0.25
            },
            'training_time_minutes': self._estimate_training_time(size),
            'confidence': 0.85
        }
    
    @staticmethod
    def _get_secondary_algorithms(primary: str) -> List[str]:
        """Get algorithms to try if primary doesn't work"""
        
        alternatives = {
            'LogisticRegression': ['RandomForest', 'SVC'],
            'RandomForest': ['XGBoost', 'LightGBM', 'GradientBoosting'],
            'XGBoost': ['LightGBM', 'GradientBoosting'],
            'LightGBM': ['XGBoost', 'GradientBoosting'],
            'GradientBoosting': ['XGBoost', 'RandomForest'],
            'NaiveBayes': ['LogisticRegression', 'RandomForest'],
            'SVC': ['RandomForest', 'LogisticRegression']
        }
        
        return alternatives.get(primary, ['RandomForest', 'XGBoost'])
    
    @staticmethod
    def _get_hyperparameters(algorithm: str) -> Dict:
        """Get recommended hyperparameter ranges"""
        
        params = {
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'max_iter': [100, 500, 1000]
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
        }
        
        return params.get(algorithm, {})
    
    @staticmethod
    def _estimate_training_time(size: str) -> float:
        """Estimate training time in minutes"""
        
        estimates = {
            'tiny': 0.1,
            'small': 0.5,
            'medium': 2,
            'large': 10,
            'huge': 60
        }
        
        return estimates.get(size, 5)


class SmartPipelineGenerator:
    """Main orchestrator for smart pipeline generation"""
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.analysis = None
        self.recommendation = None
    
    def generate(self) -> Dict:
        """Generate complete pipeline recommendation"""
        
        # Step 1: Analyze dataset
        analyzer = DatasetAnalyzer(self.df, self.target_col)
        self.analysis = analyzer.analyze()
        
        # Step 2: Recommend architecture
        recommender = ArchitectureRecommender(self.analysis)
        self.recommendation = recommender.recommend()
        
        # Step 3: Generate pipeline spec
        pipeline_spec = self._build_pipeline_spec()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis': self.analysis,
            'recommendation': self.recommendation,
            'pipeline': pipeline_spec,
            'next_steps': [
                'Review recommendation',
                'Click "Generate Pipeline" to train',
                'View results and metrics',
                'Save pipeline version'
            ]
        }
    
    def _build_pipeline_spec(self) -> Dict:
        """Build executable pipeline specification"""
        
        return {
            'name': f"AutoML_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'target': self.target_col,
            'problem_type': self.analysis['target'].get('type', 'classification'),
            'algorithm': self.recommendation['primary_algorithm'],
            'preprocessing': self.recommendation['preprocessing_steps'],
            'hyperparameters': self.recommendation['hyperparameter_ranges'],
            'features': {
                'numeric': self.analysis['types']['numeric_count'],
                'categorical': self.analysis['types']['categorical_count']
            },
            'expected_performance': self.recommendation['expected_metrics'],
            'estimated_training_time': self.recommendation['training_time_minutes']
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        
        rec = self.recommendation
        return f"""
🤖 SMART PIPELINE RECOMMENDATION
{'='*60}

Algorithm: {rec['primary_algorithm']}
Confidence: {rec['confidence']*100:.0f}%

Why: {rec['reason']}

Preprocessing Steps:
{chr(10).join('  • ' + step for step in rec['preprocessing_steps'])}

Expected Performance:
{chr(10).join(f"  • {k}: {v}" for k, v in rec['expected_metrics'].items())}

Estimated Training Time: {rec['training_time_minutes']} minutes

Alternative Algorithms to Try:
{chr(10).join('  • ' + algo for algo in rec['secondary_algorithms'])}
{'='*60}
        """
