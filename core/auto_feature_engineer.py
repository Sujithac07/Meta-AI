"""
Autonomous Feature Discovery Engine - OPTIMIZED
Auto-generates superior features using math and information theory
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from itertools import combinations
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class AutoFeatureEngineer:
    """
    Autonomous feature engineering that 'invents' better features:
    - Interaction Discovery (Ratios, Products)
    - Polynomial Expansion
    - Information Gain Filtering
    
    OPTIMIZED: Limits pairs, uses fewer estimators, samples large datasets
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.original_features = []
        self.engineered_features = []
        self.dropped_features = []
        self.feature_importance = {}
        self.mutual_info_scores = {}
        
    def auto_engineer(self, df: pd.DataFrame, target_col: str,
                     task_type: str = 'classification',
                     keep_top_interactions: int = 10,  # Reduced from 20
                     polynomial_degree: int = 2,
                     mi_threshold: float = 0.01) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete autonomous feature engineering pipeline.
        OPTIMIZED for speed.
        """
        if target_col not in df.columns:
            return df, {"error": f"Target column '{target_col}' not found"}
        
        df_result = df.copy()
        self.original_features = [c for c in df.columns if c != target_col]
        
        # Get numeric columns (excluding target)
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        
        if len(numeric_cols) < 2:
            return df_result, {"error": "Need at least 2 numeric columns for feature engineering"}
        
        # OPTIMIZATION: Limit columns for interaction (top 10 by variance)
        if len(numeric_cols) > 10:
            variances = df_result[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(10).index.tolist()
        
        # Get target
        y = df_result[target_col].values
        
        # Step 1: Interaction Discovery (FAST)
        df_result, interaction_report = self.interaction_discovery(
            df_result, numeric_cols, target_col, y, keep_top_interactions
        )
        
        # Step 2: Get feature importance (FAST - fewer trees)
        importance_report = self._compute_feature_importance(df_result, target_col, task_type)
        
        # Step 3: Polynomial Expansion on top features
        df_result, polynomial_report = self.polynomial_expansion(
            df_result, target_col, top_n=3, degree=polynomial_degree
        )
        
        # Step 4: Information Gain Filter (FAST - sampling)
        df_result, filter_report = self.information_gain_filter(
            df_result, target_col, task_type, mi_threshold
        )
        
        # Compile report
        new_features = [c for c in df_result.columns if c not in self.original_features and c != target_col]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "original_feature_count": len(self.original_features),
            "final_feature_count": len([c for c in df_result.columns if c != target_col]),
            "new_features_created": len(new_features),
            "features_dropped": len(self.dropped_features),
            "new_features": {
                "ratio_features": [f for f in new_features if '_div_' in f],
                "product_features": [f for f in new_features if '_x_' in f],
                "polynomial_features": [f for f in new_features if f.startswith('poly_')]
            },
            "interaction_discovery": interaction_report,
            "feature_importance": importance_report,
            "polynomial_expansion": polynomial_report,
            "information_filter": filter_report,
            "engineered_features": new_features[:20],
            "dropped_features": self.dropped_features[:20]
        }
        
        return df_result, report
    
    def interaction_discovery(self, df: pd.DataFrame, numeric_cols: List[str],
                             target_col: str, y: np.ndarray,
                             keep_top: int = 10) -> Tuple[pd.DataFrame, Dict]:
        """
        Create Ratio and Product features - OPTIMIZED.
        Only processes limited column pairs.
        """
        df_result = df.copy()
        
        # OPTIMIZATION: Limit to max 8 columns (28 pairs max)
        if len(numeric_cols) > 8:
            numeric_cols = numeric_cols[:8]
        
        # Calculate baseline correlations
        baseline_corrs = {}
        for col in numeric_cols:
            if df_result[col].std() > 0:
                corr = abs(np.corrcoef(df_result[col].fillna(0), y)[0, 1])
                baseline_corrs[col] = corr if not np.isnan(corr) else 0
        
        max_baseline = max(baseline_corrs.values()) if baseline_corrs else 0
        
        # Generate interaction features
        interaction_candidates = []
        
        for col_a, col_b in combinations(numeric_cols, 2):
            a = df_result[col_a].fillna(0).values
            b = df_result[col_b].fillna(0).values
            
            # Product feature only
            product = a * b
            if np.std(product) > 0:
                corr = abs(np.corrcoef(product, y)[0, 1])
                if not np.isnan(corr) and corr > max_baseline * 0.3:
                    interaction_candidates.append({
                        'name': f'{col_a}_x_{col_b}',
                        'type': 'product',
                        'values': product,
                        'correlation': corr,
                        'improvement': corr - max(baseline_corrs.get(col_a, 0), baseline_corrs.get(col_b, 0))
                    })
            
            # Single ratio (A/B only, skip B/A)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(b != 0, a / b, 0)
                ratio = np.clip(ratio, -1e6, 1e6)
                if np.std(ratio) > 0 and not np.any(np.isinf(ratio)):
                    corr = abs(np.corrcoef(ratio, y)[0, 1])
                    if not np.isnan(corr) and corr > max_baseline * 0.3:
                        interaction_candidates.append({
                            'name': f'{col_a}_div_{col_b}',
                            'type': 'ratio',
                            'values': ratio,
                            'correlation': corr,
                            'improvement': corr - max(baseline_corrs.get(col_a, 0), baseline_corrs.get(col_b, 0))
                        })
        
        # Sort by correlation and keep top
        interaction_candidates.sort(key=lambda x: x['correlation'], reverse=True)
        selected = interaction_candidates[:keep_top]
        
        # Add selected features to dataframe
        for feat in selected:
            df_result[feat['name']] = feat['values']
            self.engineered_features.append(feat['name'])
        
        report = {
            "total_candidates_generated": len(interaction_candidates),
            "features_kept": len(selected),
            "baseline_max_correlation": round(max_baseline, 4),
            "top_interactions": [
                {
                    "name": f['name'],
                    "type": f['type'],
                    "correlation": round(f['correlation'], 4),
                    "improvement": round(f['improvement'], 4)
                }
                for f in selected[:5]
            ]
        }
        
        return df_result, report
    
    def _compute_feature_importance(self, df: pd.DataFrame, target_col: str,
                                   task_type: str) -> Dict:
        """Compute feature importance - OPTIMIZED with fewer trees."""
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # OPTIMIZATION: Sample if too large
        if len(X) > 5000:
            sample_idx = np.random.choice(len(X), 5000, replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]
        
        try:
            if task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=20,  # Reduced from 50
                    max_depth=10,     # Limit depth
                    random_state=self.random_state, 
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=20,
                    max_depth=10,
                    random_state=self.random_state, 
                    n_jobs=-1
                )
            
            model.fit(X, y)
            importances = dict(zip(feature_cols, model.feature_importances_))
            self.feature_importance = importances
            
            sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "method": "RandomForest (fast)",
                "top_10_features": [
                    {"feature": f, "importance": round(imp, 4)}
                    for f, imp in sorted_importance[:10]
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def polynomial_expansion(self, df: pd.DataFrame, target_col: str,
                            top_n: int = 3, degree: int = 2) -> Tuple[pd.DataFrame, Dict]:
        """Apply polynomial transformations to top N features."""
        df_result = df.copy()
        
        if not self.feature_importance:
            return df_result, {"status": "skipped", "reason": "No feature importance computed"}
        
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [f for f, _ in sorted_features[:top_n] if f in df_result.columns]
        
        if len(top_features) < 2:
            return df_result, {"status": "skipped", "reason": "Not enough features"}
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        
        X_top = df_result[top_features].fillna(0).values
        X_poly = poly.fit_transform(X_top)
        
        poly_feature_names = poly.get_feature_names_out(top_features)
        
        new_poly_features = []
        for i, name in enumerate(poly_feature_names):
            if name not in top_features:
                clean_name = f"poly_{name.replace(' ', '_')}"
                df_result[clean_name] = X_poly[:, i]
                new_poly_features.append(clean_name)
                self.engineered_features.append(clean_name)
        
        return df_result, {
            "status": "success",
            "degree": degree,
            "source_features": top_features,
            "new_features_created": len(new_poly_features),
            "polynomial_features": new_poly_features[:10]
        }
    
    def information_gain_filter(self, df: pd.DataFrame, target_col: str,
                               task_type: str, threshold: float = 0.01) -> Tuple[pd.DataFrame, Dict]:
        """Filter features by Mutual Information - OPTIMIZED."""
        df_result = df.copy()
        feature_cols = [c for c in df_result.columns if c != target_col]
        
        X = df_result[feature_cols].fillna(0)
        y = df_result[target_col]
        
        # OPTIMIZATION: Sample if large
        if len(X) > 3000:
            sample_idx = np.random.choice(len(X), 3000, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y
        
        try:
            if task_type == 'classification':
                mi_scores = mutual_info_classif(
                    X_sample, y_sample, 
                    random_state=self.random_state, 
                    n_neighbors=3  # Reduced from 5
                )
            else:
                mi_scores = mutual_info_regression(
                    X_sample, y_sample,
                    random_state=self.random_state,
                    n_neighbors=3
                )
            
            self.mutual_info_scores = dict(zip(feature_cols, mi_scores))
            
            features_to_drop = [
                col for col, score in self.mutual_info_scores.items()
                if score < threshold
            ]
            
            # Don't drop too many
            max_drop = len(feature_cols) // 3
            features_to_drop = features_to_drop[:max_drop]
            
            df_result = df_result.drop(columns=features_to_drop)
            self.dropped_features.extend(features_to_drop)
            
            sorted_mi = sorted(self.mutual_info_scores.items(), key=lambda x: x[1], reverse=True)
            
            return df_result, {
                "status": "success",
                "method": "Mutual Information (fast)",
                "threshold": threshold,
                "features_analyzed": len(feature_cols),
                "features_dropped": len(features_to_drop),
                "features_kept": len(feature_cols) - len(features_to_drop),
                "dropped_features": features_to_drop[:5],
                "top_mi_scores": [
                    {"feature": f, "mi_score": round(s, 4)}
                    for f, s in sorted_mi[:5]
                ]
            }
            
        except Exception as e:
            return df_result, {"status": "failed", "error": str(e)}


def auto_feature_engineer(df: pd.DataFrame, target_col: str,
                         task_type: str = 'classification') -> Tuple[pd.DataFrame, Dict]:
    """Convenience function for autonomous feature engineering."""
    engineer = AutoFeatureEngineer()
    return engineer.auto_engineer(df, target_col, task_type)


def format_feature_report(report: Dict[str, Any]) -> str:
    """Format feature engineering report for display."""
    if 'error' in report:
        return f"Error: {report['error']}"
    
    lines = []
    lines.append("=" * 50)
    lines.append("FEATURE ENGINEERING REPORT")
    lines.append("=" * 50)
    
    lines.append(f"\nOriginal Features: {report.get('original_feature_count', 0)}")
    lines.append(f"Final Features: {report.get('final_feature_count', 0)}")
    lines.append(f"New Features: {report.get('new_features_created', 0)}")
    lines.append(f"Dropped: {report.get('features_dropped', 0)}")
    
    # Interactions
    inter = report.get('interaction_discovery', {})
    lines.append("\n--- Interactions ---")
    lines.append(f"Generated: {inter.get('total_candidates_generated', 0)}")
    lines.append(f"Kept: {inter.get('features_kept', 0)}")
    
    top_inter = inter.get('top_interactions', [])
    if top_inter:
        lines.append("Top:")
        for f in top_inter[:3]:
            lines.append(f"  {f['name']}: r={f['correlation']:.3f}")
    
    # Polynomial
    poly = report.get('polynomial_expansion', {})
    lines.append("\n--- Polynomial ---")
    lines.append(f"Status: {poly.get('status', 'N/A')}")
    if poly.get('new_features_created'):
        lines.append(f"Created: {poly.get('new_features_created', 0)}")
    
    # MI Filter
    mi = report.get('information_filter', {})
    lines.append("\n--- MI Filter ---")
    lines.append(f"Dropped: {mi.get('features_dropped', 0)}")
    
    lines.append("\n" + "=" * 50)
    
    return '\n'.join(lines)
