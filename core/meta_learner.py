"""
Meta-Learner: Learns from past experiments to recommend best models
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional


class MetaLearner:
    """
    Meta-learning system that learns from past experiments to recommend
    the best model for new datasets based on dataset characteristics.
    """
    
    def __init__(self, memory_file: str = "meta_learner_memory.json"):
        """
        Initialize MetaLearner
        
        Args:
            memory_file: Path to JSON file storing past experiments
        """
        self.memory_file = memory_file
        self._ensure_memory_file()
    
    def _ensure_memory_file(self):
        """Create memory file if it doesn't exist"""
        try:
            if not os.path.exists(self.memory_file):
                with open(self.memory_file, 'w') as f:
                    json.dump({"experiments": []}, f)
        except Exception as e:
            print(f"Warning: Could not create memory file: {e}")
    
    def extract_meta_features(self, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Extract meta-features from a dataset
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Dictionary of meta-features
        """
        try:
            meta_features = {}
            
            # Basic shape features
            meta_features['n_rows'] = len(df)
            meta_features['n_cols'] = len(df.columns)
            
            # Separate features from target
            if target_col in df.columns:
                X = df.drop(columns=[target_col])
                y = df[target_col]
            else:
                X = df
                y = None
            
            # Numeric vs categorical features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            
            meta_features['n_numeric'] = len(numeric_cols)
            meta_features['n_categorical'] = len(categorical_cols)
            
            # Class imbalance ratio (for classification)
            if y is not None:
                try:
                    class_counts = y.value_counts()
                    if len(class_counts) > 0:
                        max_class = class_counts.max()
                        min_class = class_counts.min()
                        meta_features['class_imbalance'] = float(max_class / min_class) if min_class > 0 else 1.0
                    else:
                        meta_features['class_imbalance'] = 1.0
                except Exception:
                    meta_features['class_imbalance'] = 1.0
            else:
                meta_features['class_imbalance'] = 1.0
            
            # Mean correlation between numeric features
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = X[numeric_cols].corr().abs()
                    # Get upper triangle (excluding diagonal)
                    upper_tri = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    mean_corr = upper_tri.stack().mean()
                    meta_features['mean_correlation'] = float(mean_corr) if not np.isnan(mean_corr) else 0.0
                except Exception:
                    meta_features['mean_correlation'] = 0.0
            else:
                meta_features['mean_correlation'] = 0.0
            
            # Percentage of missing values
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            meta_features['missing_percentage'] = float(missing_cells / total_cells * 100) if total_cells > 0 else 0.0
            
            # Mean skewness of numeric features
            if len(numeric_cols) > 0:
                try:
                    skewness_values = X[numeric_cols].skew()
                    mean_skewness = skewness_values.abs().mean()
                    meta_features['skewness_mean'] = float(mean_skewness) if not np.isnan(mean_skewness) else 0.0
                except Exception:
                    meta_features['skewness_mean'] = 0.0
            else:
                meta_features['skewness_mean'] = 0.0
            
            return meta_features
            
        except Exception as e:
            print(f"Error extracting meta-features: {e}")
            # Return default meta-features
            return {
                'n_rows': 0,
                'n_cols': 0,
                'n_numeric': 0,
                'n_categorical': 0,
                'class_imbalance': 1.0,
                'mean_correlation': 0.0,
                'missing_percentage': 0.0,
                'skewness_mean': 0.0
            }
    
    def log_experiment(self, meta_features: Dict[str, float], model_name: str, 
                      f1_score: float, accuracy: float) -> bool:
        """
        Log an experiment to memory
        
        Args:
            meta_features: Dictionary of dataset meta-features
            model_name: Name of the model used
            f1_score: F1 score achieved
            accuracy: Accuracy achieved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing memory
            with open(self.memory_file, 'r') as f:
                memory = json.load(f)
            
            # Create experiment record
            experiment = {
                'meta_features': meta_features,
                'model_name': model_name,
                'f1_score': float(f1_score),
                'accuracy': float(accuracy)
            }
            
            # Add to memory
            memory['experiments'].append(experiment)
            
            # Save back to file
            with open(self.memory_file, 'w') as f:
                json.dump(memory, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error logging experiment: {e}")
            return False
    
    def _euclidean_distance(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """
        Calculate euclidean distance between two meta-feature vectors
        
        Args:
            features1: First meta-features dict
            features2: Second meta-features dict
            
        Returns:
            Euclidean distance
        """
        try:
            # Get common keys
            keys = set(features1.keys()) & set(features2.keys())
            
            if not keys:
                return float('inf')
            
            # Calculate euclidean distance
            squared_diffs = [(features1[k] - features2[k]) ** 2 for k in keys]
            distance = np.sqrt(sum(squared_diffs))
            
            return float(distance)
            
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return float('inf')
    
    def predict_best_model(self, meta_features: Dict[str, float]) -> Optional[str]:
        """
        Predict best model based on most similar past dataset
        
        Args:
            meta_features: Meta-features of current dataset
            
        Returns:
            Name of recommended model, or None if no memory
        """
        try:
            # Load memory
            with open(self.memory_file, 'r') as f:
                memory = json.load(f)
            
            experiments = memory.get('experiments', [])
            
            if len(experiments) < 3:
                return None  # Not enough data
            
            # Find most similar dataset
            min_distance = float('inf')
            best_model = None
            
            for exp in experiments:
                exp_features = exp.get('meta_features', {})
                distance = self._euclidean_distance(meta_features, exp_features)
                
                if distance < min_distance:
                    min_distance = distance
                    best_model = exp.get('model_name')
            
            return best_model
            
        except Exception as e:
            print(f"Error predicting best model: {e}")
            return None
    
    def get_recommendation(self, df: pd.DataFrame, target_col: str) -> Tuple[str, float]:
        """
        Get model recommendation for a dataset
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (recommended_model_name, confidence_score)
        """
        try:
            # Extract meta-features
            meta_features = self.extract_meta_features(df, target_col)
            
            # Load memory to check number of experiments
            with open(self.memory_file, 'r') as f:
                memory = json.load(f)
            
            experiments = memory.get('experiments', [])
            n_experiments = len(experiments)
            
            # If less than 3 experiments, return default
            if n_experiments < 3:
                return "RandomForest", 0.3  # Low confidence
            
            # Predict best model
            best_model = self.predict_best_model(meta_features)
            
            if best_model is None:
                return "RandomForest", 0.3
            
            # Calculate confidence based on number of experiments and similarity
            # More experiments = higher confidence (up to a point)
            base_confidence = min(0.5 + (n_experiments / 20), 0.9)
            
            return best_model, base_confidence
            
        except Exception as e:
            print(f"Error getting recommendation: {e}")
            return "RandomForest", 0.3
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the meta-learner memory
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            with open(self.memory_file, 'r') as f:
                memory = json.load(f)
            
            experiments = memory.get('experiments', [])
            
            if not experiments:
                return {
                    'total_experiments': 0,
                    'models_used': [],
                    'best_performing_model': None
                }
            
            # Count models
            model_counts = {}
            model_scores = {}
            
            for exp in experiments:
                model = exp.get('model_name', 'Unknown')
                f1 = exp.get('f1_score', 0.0)
                
                model_counts[model] = model_counts.get(model, 0) + 1
                
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(f1)
            
            # Find best performing model
            best_model = None
            best_avg_f1 = 0.0
            
            for model, scores in model_scores.items():
                avg_f1 = np.mean(scores)
                if avg_f1 > best_avg_f1:
                    best_avg_f1 = avg_f1
                    best_model = model
            
            return {
                'total_experiments': len(experiments),
                'models_used': list(model_counts.keys()),
                'model_counts': model_counts,
                'best_performing_model': best_model,
                'best_avg_f1_score': best_avg_f1
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'total_experiments': 0,
                'models_used': [],
                'best_performing_model': None
            }


def extract_meta_features(df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    """
    Standalone function to extract meta-features from a dataset
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
        Returns:
        Dictionary of meta-features
    """
    try:
        learner = MetaLearner()
        return learner.extract_meta_features(df, target_col)
    except Exception as e:
        print(f"Error in extract_meta_features: {e}")
        return {
            'n_rows': 0,
            'n_cols': 0,
            'n_numeric': 0,
            'n_categorical': 0,
            'class_imbalance': 1.0,
            'mean_correlation': 0.0,
            'missing_percentage': 0.0,
            'skewness_mean': 0.0
        }


if __name__ == "__main__":
    # Example usage
    print("Meta-Learner Example")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Initialize meta-learner
    ml = MetaLearner("test_memory.json")
    
    # Extract meta-features
    meta_features = ml.extract_meta_features(sample_data, 'target')
    print("\nMeta-features:")
    for key, value in meta_features.items():
        print(f"  {key}: {value:.4f}")
    
    # Log some experiments
    print("\nLogging experiments...")
    ml.log_experiment(meta_features, "RandomForest", 0.85, 0.87)
    ml.log_experiment(meta_features, "XGBoost", 0.88, 0.90)
    ml.log_experiment(meta_features, "LogisticRegression", 0.82, 0.84)
    
    # Get recommendation
    print("\nGetting recommendation...")
    model, confidence = ml.get_recommendation(sample_data, 'target')
    print(f"Recommended model: {model} (confidence: {confidence:.2f})")
    
    # Get stats
    print("\nMemory statistics:")
    stats = ml.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clean up test file
    if os.path.exists("test_memory.json"):
        os.remove("test_memory.json")
    
    print("\nDone!")