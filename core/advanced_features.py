"""
Advanced Feature Engineering Pipeline
Automatic feature creation, selection, and engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """Automatic feature engineering"""
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.feature_engineering_steps = []
        self.created_features = {}
    
    def generate_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial and interaction features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return X
        
        X_interactions = X.copy()
        
        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(X[numeric_cols])
        
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Add polynomial features
        for i, fname in enumerate(poly_feature_names):
            if fname not in X_interactions.columns:
                X_interactions[f'poly_{fname}'] = X_poly[:, i]
                self.created_features[f'poly_{fname}'] = 'polynomial'
        
        self.feature_engineering_steps.append('interaction_features')
        
        return X_interactions
    
    def generate_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical aggregate features"""
        X_stat = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return X_stat
        
        # Row-wise statistics
        X_stat['row_mean'] = X[numeric_cols].mean(axis=1)
        X_stat['row_std'] = X[numeric_cols].std(axis=1)
        X_stat['row_min'] = X[numeric_cols].min(axis=1)
        X_stat['row_max'] = X[numeric_cols].max(axis=1)
        X_stat['row_skew'] = X[numeric_cols].skew(axis=1)
        
        self.created_features['row_mean'] = 'statistical'
        self.created_features['row_std'] = 'statistical'
        self.created_features['row_min'] = 'statistical'
        self.created_features['row_max'] = 'statistical'
        self.created_features['row_skew'] = 'statistical'
        
        # Pairwise interactions (top numeric features)
        top_features = X[numeric_cols].var().nlargest(3).index.tolist()
        for i, f1 in enumerate(top_features):
            for f2 in top_features[i+1:]:
                X_stat[f'{f1}_x_{f2}'] = X[f1] * X[f2]
                X_stat[f'{f1}_div_{f2}'] = np.where(X[f2] != 0, X[f1] / X[f2], 0)
                self.created_features[f'{f1}_x_{f2}'] = 'interaction'
                self.created_features[f'{f1}_div_{f2}'] = 'ratio'
        
        self.feature_engineering_steps.append('statistical_features')
        
        return X_stat
    
    def generate_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-specific features"""
        X_domain = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            # Binning
            X_domain[f'{col}_binned'] = pd.qcut(X[col], q=5, labels=False, duplicates='drop')
            self.created_features[f'{col}_binned'] = 'binning'
            
            # Log transformation (for positive values)
            if (X[col] > 0).all():
                X_domain[f'{col}_log'] = np.log1p(X[col])
                self.created_features[f'{col}_log'] = 'log_transform'
            
            # Square root (for non-negative values)
            if (X[col] >= 0).all():
                X_domain[f'{col}_sqrt'] = np.sqrt(X[col])
                self.created_features[f'{col}_sqrt'] = 'sqrt_transform'
        
        self.feature_engineering_steps.append('domain_features')
        
        return X_domain
    
    def select_best_features(self, X: pd.DataFrame, y: np.ndarray, 
                            k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Select k best features using multiple methods"""
        
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.shape[1] <= k:
            return X, numeric_X.columns.tolist()
        
        # Method 1: ANOVA F-test
        selector_f = SelectKBest(f_classif, k=min(k, numeric_X.shape[1]))
        selector_f.fit(numeric_X, y)
        selected_f = numeric_X.columns[selector_f.get_support()].tolist()
        
        # Method 2: Mutual Information
        selector_mi = SelectKBest(mutual_info_classif, k=min(k, numeric_X.shape[1]))
        selector_mi.fit(numeric_X, y)
        selected_mi = numeric_X.columns[selector_mi.get_support()].tolist()
        
        # Combine selections (features selected by either method)
        selected_features = list(set(selected_f + selected_mi))[:k]
        
        self.feature_engineering_steps.append(f'feature_selection_{k}')
        
        return X[selected_features + [c for c in X.columns if c not in numeric_X.columns]], selected_features
    
    def apply_dimensionality_reduction(self, X: pd.DataFrame, 
                                      n_components: int = 10) -> Tuple[np.ndarray, List[str]]:
        """Apply PCA dimensionality reduction"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) <= n_components:
            return X[numeric_cols].values, numeric_cols
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X[numeric_cols])
        
        component_names = [f'pca_component_{i}' for i in range(n_components)]
        
        self.feature_engineering_steps.append(f'pca_{n_components}')
        
        return X_pca, component_names
    
    def execute_pipeline(self, X: pd.DataFrame, y: np.ndarray,
                        apply_all: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Execute complete feature engineering pipeline"""
        
        X_engineered = X.copy()
        
        # Step 1: Generate features
        if apply_all:
            X_engineered = self.generate_interaction_features(X_engineered)
            X_engineered = self.generate_statistical_features(X_engineered)
            X_engineered = self.generate_domain_features(X_engineered)
        
        # Step 2: Select best features
        X_engineered, selected = self.select_best_features(
            X_engineered, y, k=min(20, X_engineered.shape[1])
        )
        
        report = {
            'original_features': X.shape[1],
            'engineered_features': X_engineered.shape[1],
            'feature_types_created': set(self.created_features.values()),
            'engineering_steps': self.feature_engineering_steps,
            'selected_features': selected,
            'created_features': self.created_features
        }
        
        return X_engineered, report


class EnsembleFeatureSelection:
    """Feature selection using ensemble of methods"""
    
    def __init__(self):
        self.feature_scores = {}
    
    def select_features_ensemble(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        """Select features using ensemble voting"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Method 1: Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_numeric, y)
        rf_importance = rf.feature_importances_
        
        # Method 2: Permutation importance
        perm_importance = permutation_importance(rf, X_numeric, y, n_repeats=10)
        perm_scores = perm_importance.importances_mean
        
        # Method 3: F-test scores
        from sklearn.feature_selection import f_classif
        f_scores, _ = f_classif(X_numeric, y)
        
        # Normalize scores to 0-1
        rf_importance = rf_importance / (rf_importance.sum() + 1e-10)
        perm_scores = perm_scores / (perm_scores.sum() + 1e-10)
        f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-10)
        
        # Ensemble voting
        ensemble_scores = (rf_importance + perm_scores + f_scores) / 3
        
        self.feature_scores = dict(zip(numeric_cols, ensemble_scores))
        
        # Select top 70% of features
        threshold = np.percentile(ensemble_scores, 30)
        selected = [col for col, score in self.feature_scores.items() if score >= threshold]
        
        return selected
    
    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get ranked feature importance"""
        return sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
