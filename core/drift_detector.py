"""
Drift Detector: Detects data drift between reference and production data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy.stats import ks_2samp

# Try to import evidently
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Warning: evidently library not available. Using statistical fallback for drift detection.")


class DriftDetector:
    """
    Detects data drift between reference and current datasets.
    Uses evidently library if available, falls back to KS test.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize DriftDetector
        
        Args:
            significance_level: P-value threshold for KS test (default: 0.05)
        """
        self.significance_level = significance_level
        self.evidently_available = EVIDENTLY_AVAILABLE
    
    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, 
                    target_col: Optional[str] = None) -> Dict:
        """
        Detect data drift between reference and current datasets
        
        Args:
            reference_df: Reference (training) dataset
            current_df: Current (production) dataset
            target_col: Name of target column (optional)
            
        Returns:
            Dictionary with drift detection results:
            - drift_detected: bool
            - drift_score: float (0-1)
            - drifted_features: list of feature names
            - report: string summary
        """
        try:
            if self.evidently_available:
                return self._detect_drift_evidently(reference_df, current_df, target_col)
            else:
                return self._detect_drift_ks_test(reference_df, current_df, target_col)
        except Exception as e:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drifted_features': [],
                'report': f"Error detecting drift: {str(e)}"
            }
    
    def _detect_drift_evidently(self, reference_df: pd.DataFrame, 
                               current_df: pd.DataFrame,
                               target_col: Optional[str] = None) -> Dict:
        """
        Detect drift using evidently library
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            target_col: Target column name
            
        Returns:
            Dictionary with drift results
        """
        try:
            # Create evidently report
            report = Report(metrics=[
                DataDriftPreset()
            ])
            
            # Run report
            report.run(reference_data=reference_df, current_data=current_df)
            
            # Extract results
            report_dict = report.as_dict()
            
            # Parse metrics
            drift_metrics = report_dict.get('metrics', [{}])[0]
            
            dataset_drift = drift_metrics.get('result', {}).get('dataset_drift', False)
            drift_share = drift_metrics.get('result', {}).get('share_of_drifted_columns', 0.0)
            
            # Get drifted features
            drifted_features = []
            drift_by_columns = drift_metrics.get('result', {}).get('drift_by_columns', {})
            for col, info in drift_by_columns.items():
                if isinstance(info, dict) and info.get('drift_detected', False):
                    drifted_features.append(col)
            
            # Create summary report
            report_text = f"""
Data Drift Analysis (Evidently)
================================
Dataset Drift Detected: {dataset_drift}
Drift Score: {drift_share:.2%}
Number of Drifted Features: {len(drifted_features)}
Drifted Features: {', '.join(drifted_features) if drifted_features else 'None'}
            """.strip()
            
            return {
                'drift_detected': dataset_drift,
                'drift_score': float(drift_share),
                'drifted_features': drifted_features,
                'report': report_text
            }
            
        except Exception as e:
            print(f"Error in evidently drift detection: {e}")
            # Fallback to KS test
            return self._detect_drift_ks_test(reference_df, current_df, target_col)
    
    def _detect_drift_ks_test(self, reference_df: pd.DataFrame, 
                             current_df: pd.DataFrame,
                             target_col: Optional[str] = None) -> Dict:
        """
        Detect drift using Kolmogorov-Smirnov test (fallback method)
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            target_col: Target column name
            
        Returns:
            Dictionary with drift results
        """
        try:
            # Get numeric columns (excluding target)
            numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_col and target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            if not numeric_cols:
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'drifted_features': [],
                    'report': 'No numeric features to test for drift'
                }
            
            # Perform KS test on each numeric column
            drifted_features = []
            p_values = {}
            
            for col in numeric_cols:
                try:
                    # Get data for both datasets
                    ref_data = reference_df[col].dropna()
                    curr_data = current_df[col].dropna()
                    
                    if len(ref_data) > 0 and len(curr_data) > 0:
                        # Perform KS test
                        statistic, p_value = ks_2samp(ref_data, curr_data)
                        p_values[col] = p_value
                        
                        # Check if drift detected
                        if p_value < self.significance_level:
                            drifted_features.append(col)
                except Exception as e:
                    print(f"Error testing column {col}: {e}")
                    continue
            
            # Calculate drift score (proportion of drifted features)
            drift_score = len(drifted_features) / len(numeric_cols) if numeric_cols else 0.0
            drift_detected = drift_score > 0.0
            
            # Create summary report
            report_lines = [
                "Data Drift Analysis (KS Test)",
                "=" * 40,
                f"Drift Detected: {drift_detected}",
                f"Drift Score: {drift_score:.2%}",
                f"Drifted Features: {len(drifted_features)}/{len(numeric_cols)}",
                "",
                "Feature P-values:",
            ]
            
            for col, p_val in sorted(p_values.items(), key=lambda x: x[1]):
                drift_status = "DRIFT" if p_val < self.significance_level else "OK"
                report_lines.append(f"  {col}: {p_val:.4f} [{drift_status}]")
            
            report_text = "\n".join(report_lines)
            
            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'drifted_features': drifted_features,
                'report': report_text
            }
            
        except Exception as e:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drifted_features': [],
                'report': f"Error in KS test drift detection: {str(e)}"
            }
    
    def should_retrain(self, drift_score: float, threshold: float = 0.3) -> bool:
        """
        Determine if model should be retrained based on drift score
        
        Args:
            drift_score: Drift score (0-1)
            threshold: Threshold for retraining (default: 0.3)
            
        Returns:
            True if should retrain, False otherwise
        """
        try:
            return float(drift_score) > float(threshold)
        except Exception as e:
            print(f"Error in should_retrain: {e}")
            return False
    
    def get_drift_report_html(self, reference_df: pd.DataFrame, 
                             current_df: pd.DataFrame) -> str:
        """
        Generate HTML drift report
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            
        Returns:
            HTML string with drift report
        """
        try:
            if self.evidently_available:
                return self._get_evidently_html(reference_df, current_df)
            else:
                return self._get_ks_test_html(reference_df, current_df)
        except Exception as e:
            return f"<div class='error'>Error generating drift report: {str(e)}</div>"
    
    def _get_evidently_html(self, reference_df: pd.DataFrame, 
                           current_df: pd.DataFrame) -> str:
        """
        Generate HTML report using evidently
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            
        Returns:
            HTML string
        """
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)
            
            # Get HTML report
            html = report.get_html()
            return html
            
        except Exception as e:
            print(f"Error generating evidently HTML: {e}")
            return self._get_ks_test_html(reference_df, current_df)
    
    def _get_ks_test_html(self, reference_df: pd.DataFrame, 
                         current_df: pd.DataFrame) -> str:
        """
        Generate simple HTML table with KS test results
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            
        Returns:
            HTML string
        """
        try:
            # Get numeric columns
            numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                return "<div>No numeric features to analyze</div>"
            
            # Perform KS tests
            results = []
            for col in numeric_cols:
                try:
                    ref_data = reference_df[col].dropna()
                    curr_data = current_df[col].dropna()
                    
                    if len(ref_data) > 0 and len(curr_data) > 0:
                        statistic, p_value = ks_2samp(ref_data, curr_data)
                        
                        drift_status = "DRIFT DETECTED" if p_value < self.significance_level else "OK"
                        status_color = "#ff4444" if p_value < self.significance_level else "#44ff44"
                        
                        results.append({
                            'feature': col,
                            'ks_statistic': statistic,
                            'p_value': p_value,
                            'status': drift_status,
                            'color': status_color
                        })
                except Exception as e:
                    print(f"Error testing {col}: {e}")
                    continue
            
            # Generate HTML
            drifted_count = sum(1 for r in results if r["p_value"] < self.significance_level)
            html = """
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .drift {{ background-color: #ffcccc; }}
                    .ok {{ background-color: #ccffcc; }}
                </style>
            </head>
            <body>
                <h2>Data Drift Analysis - KS Test Results</h2>
                <p><strong>Significance Level:</strong> {significance}</p>
                <p><strong>Total Features Analyzed:</strong> {total}</p>
                <p><strong>Features with Drift:</strong> {drifted}</p>
                
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>KS Statistic</th>
                        <th>P-Value</th>
                        <th>Status</th>
                    </tr>
            """.format(
                significance=self.significance_level,
                total=len(results),
                drifted=drifted_count,
            )
            
            # Add rows
            for result in sorted(results, key=lambda x: x['p_value']):
                row_class = 'drift' if result['p_value'] < self.significance_level else 'ok'
                html += f"""
                    <tr class="{row_class}">
                        <td>{result['feature']}</td>
                        <td>{result['ks_statistic']:.4f}</td>
                        <td>{result['p_value']:.4f}</td>
                        <td style="color: {result['color']}; font-weight: bold;">{result['status']}</td>
                    </tr>
                """
            
            html += """
                </table>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            return f"<div class='error'>Error generating KS test HTML: {str(e)}</div>"
    
    def get_feature_statistics(self, reference_df: pd.DataFrame, 
                              current_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare feature statistics between reference and current data
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            
        Returns:
            DataFrame with statistical comparison
        """
        try:
            numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
            
            stats_data = []
            for col in numeric_cols:
                try:
                    ref_data = reference_df[col].dropna()
                    curr_data = current_df[col].dropna()
                    
                    stats_data.append({
                        'feature': col,
                        'ref_mean': ref_data.mean(),
                        'curr_mean': curr_data.mean(),
                        'ref_std': ref_data.std(),
                        'curr_std': curr_data.std(),
                        'ref_min': ref_data.min(),
                        'curr_min': curr_data.min(),
                        'ref_max': ref_data.max(),
                        'curr_max': curr_data.max(),
                        'mean_diff': abs(curr_data.mean() - ref_data.mean()),
                        'std_diff': abs(curr_data.std() - ref_data.std())
                    })
                except Exception as e:
                    print(f"Error getting stats for {col}: {e}")
                    continue
            
            return pd.DataFrame(stats_data)
            
        except Exception as e:
            print(f"Error in get_feature_statistics: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    print("Drift Detector Example")
    print("=" * 50)
    print(f"Evidently Available: {EVIDENTLY_AVAILABLE}")
    print()
    
    # Create sample datasets
    np.random.seed(42)
    
    # Reference data (training)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.uniform(0, 10, 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Current data (with drift in feature1)
    current_data = pd.DataFrame({
        'feature1': np.random.normal(2, 1, 1000),  # Mean shifted from 0 to 2
        'feature2': np.random.normal(5, 2, 1000),  # No drift
        'feature3': np.random.uniform(0, 10, 1000),  # No drift
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Initialize detector
    detector = DriftDetector()
    
    # Detect drift
    print("Detecting drift...")
    drift_result = detector.detect_drift(reference_data, current_data, 'target')
    
    print("\nDrift Detection Results:")
    print(f"  Drift Detected: {drift_result['drift_detected']}")
    print(f"  Drift Score: {drift_result['drift_score']:.2%}")
    print(f"  Drifted Features: {drift_result['drifted_features']}")
    print(f"\n{drift_result['report']}")
    
    # Check if should retrain
    should_retrain = detector.should_retrain(drift_result['drift_score'])
    print(f"\nShould Retrain: {should_retrain}")
    
    # Get feature statistics
    print("\nFeature Statistics:")
    stats = detector.get_feature_statistics(reference_data, current_data)
    print(stats[['feature', 'ref_mean', 'curr_mean', 'mean_diff']].to_string(index=False))
    
    print("\nDone!")