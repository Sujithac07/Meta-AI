"""
Forensic Data Reconstruction Engine
Advanced imputation, anomaly detection, and stability analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class ForensicCleaner:
    """
    Advanced data cleaning with:
    - Bayesian Iterative Imputation (predictive filling)
    - IsolationForest anomaly labeling (not deletion)
    - Distribution stability checks
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.raw_stats = {}
        self.cleaned_stats = {}
        self.stability_report = {}
        self.anomaly_model = None
        self.imputer = None
        self.scaler = StandardScaler()
        
    def full_reconstruction(self, df: pd.DataFrame, 
                           exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete forensic reconstruction pipeline.
        Returns cleaned DataFrame and comprehensive report.
        """
        exclude_cols = exclude_cols or []
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if not numeric_cols:
            return df, {"error": "No numeric columns to process"}
        
        # Store raw statistics
        self.raw_stats = self._compute_statistics(df[numeric_cols])
        
        # Step 1: Bayesian Iterative Imputation
        df_imputed, imputation_report = self.bayesian_imputation(df, numeric_cols)
        
        # Step 2: Anomaly Labeling
        df_anomaly, anomaly_report = self.anomaly_labeling(df_imputed, numeric_cols)
        
        # Step 3: Stability Check
        self.cleaned_stats = self._compute_statistics(df_anomaly[numeric_cols])
        stability_report = self.stability_check()
        
        # Compile full report
        report = {
            "timestamp": datetime.now().isoformat(),
            "columns_processed": numeric_cols,
            "imputation": imputation_report,
            "anomaly_detection": anomaly_report,
            "stability": stability_report,
            "summary": self._generate_summary(imputation_report, anomaly_report, stability_report)
        }
        
        return df_anomaly, report
    
    def bayesian_imputation(self, df: pd.DataFrame, 
                           numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Bayesian Iterative Imputation using sklearn's IterativeImputer.
        Treats each missing feature as a function of all other features.
        """
        df_result = df.copy()
        
        # Count missing before
        missing_before = df_result[numeric_cols].isna().sum().to_dict()
        total_missing = sum(missing_before.values())
        
        if total_missing == 0:
            return df_result, {
                "method": "Bayesian Iterative Imputation",
                "status": "skipped",
                "reason": "No missing values found",
                "values_imputed": 0
            }
        
        # Initialize Bayesian Ridge based Iterative Imputer
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=100,
            tol=1e-3,
            random_state=self.random_state,
            initial_strategy='median',
            imputation_order='ascending',
            skip_complete=True
        )
        
        # Fit and transform
        try:
            imputed_values = self.imputer.fit_transform(df_result[numeric_cols])
            df_result[numeric_cols] = imputed_values
            
            # Count missing after (should be 0)
            missing_after = df_result[numeric_cols].isna().sum().sum()
            
            report = {
                "method": "Bayesian Iterative Imputation",
                "status": "success",
                "estimator": "BayesianRidge",
                "max_iterations": 100,
                "missing_before": missing_before,
                "total_imputed": int(total_missing),
                "missing_after": int(missing_after),
                "columns_affected": [col for col, count in missing_before.items() if count > 0]
            }
            
        except Exception as e:
            report = {
                "method": "Bayesian Iterative Imputation",
                "status": "failed",
                "error": str(e)
            }
        
        return df_result, report
    
    def anomaly_labeling(self, df: pd.DataFrame, 
                        numeric_cols: List[str],
                        contamination: float = 0.1) -> Tuple[pd.DataFrame, Dict]:
        """
        Use IsolationForest to identify outliers.
        Adds anomaly_score column instead of deleting outliers.
        """
        df_result = df.copy()
        
        if not numeric_cols:
            return df_result, {"status": "skipped", "reason": "No numeric columns"}
        
        # Prepare data for IsolationForest
        X = df_result[numeric_cols].copy()
        
        # Handle any remaining NaNs (shouldn't be any after imputation)
        X = X.fillna(X.median())
        
        # Scale for better anomaly detection
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit IsolationForest
        self.anomaly_model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        try:
            # Get predictions (-1 for anomaly, 1 for normal)
            predictions = self.anomaly_model.fit_predict(X_scaled)
            
            # Get anomaly scores (lower = more anomalous)
            raw_scores = self.anomaly_model.decision_function(X_scaled)
            
            # Normalize scores to 0-1 range (1 = most anomalous)
            min_score, max_score = raw_scores.min(), raw_scores.max()
            if max_score > min_score:
                normalized_scores = 1 - (raw_scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros(len(raw_scores))
            
            # Add columns to dataframe
            df_result['anomaly_label'] = np.where(predictions == -1, 'ANOMALY', 'NORMAL')
            df_result['anomaly_score'] = normalized_scores
            
            # Statistics
            anomaly_count = (predictions == -1).sum()
            normal_count = (predictions == 1).sum()
            
            # Get top anomalies
            top_idx = np.argsort(normalized_scores)[-10:][::-1]
            
            report = {
                "method": "IsolationForest",
                "status": "success",
                "contamination_rate": contamination,
                "total_samples": len(df_result),
                "anomalies_detected": int(anomaly_count),
                "anomaly_percentage": round(anomaly_count / len(df_result) * 100, 2),
                "normal_samples": int(normal_count),
                "score_range": {
                    "min": float(normalized_scores.min()),
                    "max": float(normalized_scores.max()),
                    "mean": float(normalized_scores.mean())
                },
                "columns_added": ["anomaly_label", "anomaly_score"],
                "note": "Outliers labeled, NOT deleted. Model will learn their importance.",
                "top_anomaly_row_indices": top_idx.tolist(),
            }
            
        except Exception as e:
            report = {
                "method": "IsolationForest",
                "status": "failed",
                "error": str(e)
            }
        
        return df_result, report
    
    def stability_check(self, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Compare cleaned data distribution to raw data.
        Flag if cleaning shifted mean by more than threshold (default 5%).
        """
        if not self.raw_stats or not self.cleaned_stats:
            return {"status": "skipped", "reason": "Missing statistics"}
        
        stability_flags = []
        column_stability = {}
        
        for col in self.raw_stats.keys():
            if col not in self.cleaned_stats:
                continue
                
            raw = self.raw_stats[col]
            cleaned = self.cleaned_stats[col]
            
            # Calculate mean shift
            raw_mean = raw.get('mean', 0)
            cleaned_mean = cleaned.get('mean', 0)
            
            if raw_mean != 0:
                mean_shift = abs(cleaned_mean - raw_mean) / abs(raw_mean)
            else:
                mean_shift = abs(cleaned_mean - raw_mean)
            
            # Calculate std shift
            raw_std = raw.get('std', 0)
            cleaned_std = cleaned.get('std', 0)
            
            if raw_std != 0:
                std_shift = abs(cleaned_std - raw_std) / abs(raw_std)
            else:
                std_shift = abs(cleaned_std - raw_std)
            
            is_stable = mean_shift <= threshold
            
            column_stability[col] = {
                "raw_mean": round(raw_mean, 4),
                "cleaned_mean": round(cleaned_mean, 4),
                "mean_shift_pct": round(mean_shift * 100, 2),
                "raw_std": round(raw_std, 4),
                "cleaned_std": round(cleaned_std, 4),
                "std_shift_pct": round(std_shift * 100, 2),
                "is_stable": is_stable
            }
            
            if not is_stable:
                stability_flags.append({
                    "column": col,
                    "mean_shift": round(mean_shift * 100, 2),
                    "threshold": threshold * 100,
                    "message": f"Mean shifted by {mean_shift*100:.1f}% (threshold: {threshold*100}%)"
                })
        
        # Overall stability
        stable_columns = sum(1 for v in column_stability.values() if v['is_stable'])
        total_columns = len(column_stability)
        overall_stable = len(stability_flags) == 0
        
        return {
            "status": "completed",
            "threshold_pct": threshold * 100,
            "overall_stable": overall_stable,
            "stable_columns": stable_columns,
            "unstable_columns": total_columns - stable_columns,
            "total_columns": total_columns,
            "stability_score": round(stable_columns / total_columns * 100, 1) if total_columns > 0 else 100,
            "flags": stability_flags,
            "column_details": column_stability
        }
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Compute statistics for each numeric column."""
        stats = {}
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                stats[col] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()) if len(series) > 1 else 0,
                    'median': float(series.median()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'count': int(len(series))
                }
        return stats
    
    def _generate_summary(self, imputation: Dict, anomaly: Dict, stability: Dict) -> Dict:
        """Generate high-level summary of reconstruction."""
        return {
            "imputation_status": imputation.get('status', 'unknown'),
            "values_imputed": imputation.get('total_imputed', 0),
            "anomalies_labeled": anomaly.get('anomalies_detected', 0),
            "distribution_stable": stability.get('overall_stable', True),
            "stability_score": stability.get('stability_score', 100),
            "overall_health": "GOOD" if stability.get('overall_stable', True) else "REVIEW_NEEDED"
        }


def forensic_clean(df: pd.DataFrame, exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for forensic data reconstruction.
    Returns cleaned DataFrame and comprehensive report.
    """
    cleaner = ForensicCleaner()
    return cleaner.full_reconstruction(df, exclude_cols)


def format_forensic_report(report: Dict[str, Any]) -> str:
    """Format forensic cleaning report as readable text for Gradio display."""
    if 'error' in report:
        return f"Error: {report['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("FORENSIC DATA RECONSTRUCTION REPORT")
    lines.append("=" * 60)
    
    # Summary
    summary = report.get('summary', {})
    lines.append(f"\nOverall Health: {summary.get('overall_health', 'UNKNOWN')}")
    lines.append(f"Stability Score: {summary.get('stability_score', 0)}/100")
    
    # Imputation Section
    imp = report.get('imputation', {})
    lines.append("\n" + "-" * 40)
    lines.append("BAYESIAN ITERATIVE IMPUTATION")
    lines.append("-" * 40)
    lines.append(f"Status: {imp.get('status', 'unknown').upper()}")
    if imp.get('status') == 'success':
        lines.append(f"Estimator: {imp.get('estimator', 'BayesianRidge')}")
        lines.append(f"Values Imputed: {imp.get('total_imputed', 0):,}")
        affected = imp.get('columns_affected', [])
        if affected:
            lines.append(f"Columns Affected: {', '.join(affected[:5])}")
            if len(affected) > 5:
                lines.append(f"  ... and {len(affected) - 5} more")
    elif imp.get('status') == 'skipped':
        lines.append(f"Reason: {imp.get('reason', 'N/A')}")
    
    # Anomaly Section
    anom = report.get('anomaly_detection', {})
    lines.append("\n" + "-" * 40)
    lines.append("ISOLATION FOREST ANOMALY LABELING")
    lines.append("-" * 40)
    lines.append(f"Status: {anom.get('status', 'unknown').upper()}")
    if anom.get('status') == 'success':
        lines.append(f"Anomalies Detected: {anom.get('anomalies_detected', 0):,} ({anom.get('anomaly_percentage', 0)}%)")
        lines.append(f"Normal Samples: {anom.get('normal_samples', 0):,}")
        lines.append(f"Columns Added: {', '.join(anom.get('columns_added', []))}")
        lines.append(f"Note: {anom.get('note', '')}")
    
    # Stability Section
    stab = report.get('stability', {})
    lines.append("\n" + "-" * 40)
    lines.append("DISTRIBUTION STABILITY CHECK")
    lines.append("-" * 40)
    lines.append(f"Threshold: {stab.get('threshold_pct', 5)}% mean shift")
    lines.append(f"Stable Columns: {stab.get('stable_columns', 0)}/{stab.get('total_columns', 0)}")
    lines.append(f"Overall Stable: {'YES' if stab.get('overall_stable', True) else 'NO - REVIEW NEEDED'}")
    
    flags = stab.get('flags', [])
    if flags:
        lines.append("\nSTABILITY WARNINGS:")
        for flag in flags[:5]:
            lines.append(f"  ! {flag['column']}: {flag['message']}")
    
    lines.append("\n" + "=" * 60)
    
    return '\n'.join(lines)
