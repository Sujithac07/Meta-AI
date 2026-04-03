"""
Black-Box Breaker - Explainable AI Engine
Complete model transparency with SHAP, local explanations, and what-if analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import warnings

warnings.filterwarnings('ignore')

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class BlackBoxBreaker:
    """
    Explainable AI Engine for complete model transparency.
    - Global SHAP insights
    - Local waterfall explanations
    - What-If sensitivity analysis
    """
    
    def __init__(self, model: Any, X: pd.DataFrame, feature_names: List[str] = None):
        """
        Initialize with trained model and data.
        
        Args:
            model: Trained sklearn-compatible model
            X: Feature DataFrame used for training
            feature_names: Optional list of feature names
        """
        self.model = model
        self.X = X.copy()
        self.feature_names = feature_names or list(X.columns)
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        
    def initialize_explainer(self, max_samples: int = 100) -> Dict:
        """Initialize SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not installed. Run: pip install shap"}
        
        try:
            # Sample data for background if dataset is large
            if len(self.X) > max_samples:
                background = shap.sample(self.X, max_samples)
            else:
                background = self.X
            
            # Try TreeExplainer first (faster for tree models)
            model_type = type(self.model).__name__
            
            if hasattr(self.model, 'estimators_') or 'Forest' in model_type or 'XGB' in model_type or 'LGBM' in model_type or 'CatBoost' in model_type or 'Gradient' in model_type:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    explainer_type = "TreeExplainer"
                except Exception:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                        background
                    )
                    explainer_type = "KernelExplainer"
            else:
                # Use KernelExplainer for other models
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    background
                )
                explainer_type = "KernelExplainer"
            
            return {
                "status": "success",
                "explainer_type": explainer_type,
                "model_type": model_type,
                "background_samples": len(background)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def compute_global_shap(self, max_samples: int = 500) -> Tuple[np.ndarray, Dict]:
        """Compute SHAP values for global analysis."""
        if self.explainer is None:
            init_result = self.initialize_explainer()
            if 'error' in init_result:
                return None, init_result
        
        try:
            # Sample if needed
            if len(self.X) > max_samples:
                X_sample = self.X.sample(n=max_samples, random_state=42)
            else:
                X_sample = self.X
            
            # Compute SHAP values
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-class (take positive class for binary)
            if isinstance(self.shap_values, list):
                if len(self.shap_values) == 2:
                    self.shap_values = self.shap_values[1]  # Positive class
                else:
                    self.shap_values = self.shap_values[0]  # First class
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                ev = self.explainer.expected_value
                self.base_value = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev
            else:
                self.base_value = 0.5
            
            # Calculate feature importance from SHAP
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            feature_importance = dict(zip(self.feature_names, mean_abs_shap))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            report = {
                "status": "success",
                "samples_analyzed": len(X_sample),
                "features": len(self.feature_names),
                "top_10_features": [
                    {"feature": f, "importance": round(float(imp), 4)}
                    for f, imp in sorted_importance[:10]
                ]
            }
            
            return self.shap_values, report
            
        except Exception as e:
            return None, {"error": str(e)}
    
    def generate_summary_plot(self, max_display: int = 10) -> Tuple[str, Dict]:
        """
        Generate SHAP Summary Plot (beeswarm) for global insights.
        Returns base64 encoded image.
        """
        if self.shap_values is None:
            shap_vals, result = self.compute_global_shap()
            if shap_vals is None:
                return None, result
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Use sample of X that matches shap_values size
            X_plot = self.X.head(len(self.shap_values))
            
            shap.summary_plot(
                self.shap_values,
                X_plot,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            
            plt.title("SHAP Global Feature Impact", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_base64, {"status": "success", "plot_type": "summary_beeswarm"}
            
        except Exception as e:
            plt.close()
            return None, {"error": str(e)}
    
    def generate_bar_plot(self, max_display: int = 10) -> Tuple[str, Dict]:
        """Generate SHAP bar plot showing mean absolute impact."""
        if self.shap_values is None:
            shap_vals, result = self.compute_global_shap()
            if shap_vals is None:
                return None, result
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
            
            features = [self.feature_names[i] for i in sorted_idx]
            values = mean_abs_shap[sorted_idx]
            
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(features)))
            
            plt.barh(range(len(features)), values[::-1], color=colors[::-1])
            plt.yticks(range(len(features)), features[::-1])
            plt.xlabel('Mean |SHAP Value|', fontsize=12)
            plt.title('Top Feature Importances (SHAP)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_base64, {"status": "success", "plot_type": "bar"}
            
        except Exception as e:
            plt.close()
            return None, {"error": str(e)}
    
    def explain_single_prediction(self, row: Union[pd.Series, pd.DataFrame, np.ndarray],
                                  row_index: int = 0) -> Tuple[str, Dict]:
        """
        Generate Waterfall Plot explaining a single prediction.
        
        Args:
            row: Single row of features
            row_index: Index for display purposes
        
        Returns:
            Base64 encoded waterfall plot and explanation dict
        """
        if self.explainer is None:
            init_result = self.initialize_explainer()
            if 'error' in init_result:
                return None, init_result
        
        try:
            # Ensure row is 2D
            if isinstance(row, pd.Series):
                row_df = row.to_frame().T
            elif isinstance(row, pd.DataFrame):
                row_df = row.head(1)
            else:
                row_df = pd.DataFrame([row], columns=self.feature_names)
            
            # Get SHAP values for this instance
            shap_vals = self.explainer.shap_values(row_df)
            
            # Handle multi-class
            if isinstance(shap_vals, list):
                if len(shap_vals) == 2:
                    shap_vals = shap_vals[1]
                else:
                    shap_vals = shap_vals[0]
            
            shap_vals = shap_vals.flatten()
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                pred_proba = self.model.predict_proba(row_df)[0]
                prediction = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
            else:
                prediction = self.model.predict(row_df)[0]
            
            # Create waterfall plot
            plt.figure(figsize=(10, 8))
            
            # Sort by absolute value
            sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:15]  # Top 15
            
            features = [self.feature_names[i] for i in sorted_idx]
            values = shap_vals[sorted_idx]
            feature_values = [row_df.iloc[0, i] for i in sorted_idx]
            
            # Create waterfall-style bar chart
            colors = ['#ff6b6b' if v < 0 else '#4ecdc4' for v in values]
            
            y_pos = range(len(features))
            plt.barh(y_pos, values, color=colors)
            
            # Add feature values as labels
            labels = [f"{f} = {v:.2f}" if isinstance(v, (int, float)) else f"{f} = {v}" 
                     for f, v in zip(features, feature_values)]
            plt.yticks(y_pos, labels)
            
            plt.axvline(x=0, color='black', linewidth=0.5)
            plt.xlabel('SHAP Value (impact on prediction)', fontsize=12)
            plt.title(f'Local Explanation - Prediction: {prediction:.4f}', fontsize=14, fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4ecdc4', label='Increases prediction'),
                Patch(facecolor='#ff6b6b', label='Decreases prediction')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Build explanation dict
            explanation = {
                "status": "success",
                "prediction": float(prediction),
                "base_value": float(self.base_value) if self.base_value is not None else 0.5,
                "top_contributors": [
                    {
                        "feature": features[i],
                        "value": float(feature_values[i]) if isinstance(feature_values[i], (int, float, np.number)) else str(feature_values[i]),
                        "shap_impact": float(values[i]),
                        "direction": "increases" if values[i] > 0 else "decreases"
                    }
                    for i in range(min(5, len(features)))
                ]
            }
            
            return img_base64, explanation
            
        except Exception as e:
            plt.close()
            return None, {"error": str(e)}
    
    def what_if_analysis(self, row: Union[pd.Series, pd.DataFrame],
                        feature_to_vary: str,
                        value_range: Tuple[float, float] = None,
                        n_points: int = 20) -> Tuple[str, Dict]:
        """
        What-If sensitivity analysis: how does prediction change when varying a feature?
        
        Args:
            row: Base row to analyze
            feature_to_vary: Feature name to vary
            value_range: (min, max) range for the feature
            n_points: Number of points to evaluate
        
        Returns:
            Base64 plot and analysis dict
        """
        if feature_to_vary not in self.feature_names:
            return None, {"error": f"Feature '{feature_to_vary}' not found"}
        
        try:
            # Prepare base row
            if isinstance(row, pd.Series):
                base_row = row.to_frame().T.copy()
            else:
                base_row = row.head(1).copy()
            
            # Determine value range
            if value_range is None:
                col_data = self.X[feature_to_vary]
                value_range = (col_data.min(), col_data.max())
            
            # Generate test values
            test_values = np.linspace(value_range[0], value_range[1], n_points)
            
            # Get predictions for each value
            predictions = []
            for val in test_values:
                test_row = base_row.copy()
                test_row[feature_to_vary] = val
                
                if hasattr(self.model, 'predict_proba'):
                    pred = self.model.predict_proba(test_row)[0]
                    pred = pred[1] if len(pred) > 1 else pred[0]
                else:
                    pred = self.model.predict(test_row)[0]
                
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Create sensitivity plot
            plt.figure(figsize=(10, 6))
            
            # Main line
            plt.plot(test_values, predictions, 'b-', linewidth=2, label='Prediction')
            plt.fill_between(test_values, predictions, alpha=0.3)
            
            # Mark current value
            current_val = float(base_row[feature_to_vary].iloc[0])
            current_pred = predictions[np.argmin(np.abs(test_values - current_val))]
            plt.axvline(x=current_val, color='red', linestyle='--', label=f'Current: {current_val:.2f}')
            plt.scatter([current_val], [current_pred], color='red', s=100, zorder=5)
            
            plt.xlabel(feature_to_vary, fontsize=12)
            plt.ylabel('Prediction', fontsize=12)
            plt.title(f'What-If Analysis: Impact of {feature_to_vary}', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Calculate sensitivity metrics
            pred_range = predictions.max() - predictions.min()
            sensitivity = pred_range / (value_range[1] - value_range[0]) if value_range[1] != value_range[0] else 0
            
            # Find optimal value (max prediction)
            optimal_idx = np.argmax(predictions)
            optimal_value = test_values[optimal_idx]
            optimal_prediction = predictions[optimal_idx]
            
            analysis = {
                "status": "success",
                "feature": feature_to_vary,
                "value_range": {"min": float(value_range[0]), "max": float(value_range[1])},
                "current_value": float(current_val),
                "current_prediction": float(current_pred),
                "prediction_range": {"min": float(predictions.min()), "max": float(predictions.max())},
                "sensitivity": round(float(sensitivity), 4),
                "optimal": {
                    "value": float(optimal_value),
                    "prediction": float(optimal_prediction),
                    "improvement": float(optimal_prediction - current_pred)
                }
            }
            
            return img_base64, analysis
            
        except Exception as e:
            plt.close()
            return None, {"error": str(e)}


def create_explainer(model: Any, X: pd.DataFrame) -> BlackBoxBreaker:
    """Convenience function to create explainer."""
    return BlackBoxBreaker(model, X)


def format_xai_report(report: Dict[str, Any]) -> str:
    """Format XAI report for display."""
    if 'error' in report:
        return f"Error: {report['error']}"
    
    lines = []
    lines.append("=" * 50)
    lines.append("EXPLAINABLE AI ANALYSIS")
    lines.append("=" * 50)
    
    if 'top_10_features' in report:
        lines.append("\nTop 10 Feature Impacts (SHAP):")
        for i, feat in enumerate(report['top_10_features'], 1):
            lines.append(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
    
    if 'prediction' in report:
        lines.append(f"\nPrediction: {report['prediction']:.4f}")
        if 'top_contributors' in report:
            lines.append("\nTop Contributing Factors:")
            for contrib in report['top_contributors']:
                direction = "↑" if contrib['direction'] == 'increases' else "↓"
                lines.append(f"  {direction} {contrib['feature']} = {contrib['value']}: {contrib['shap_impact']:.4f}")
    
    if 'sensitivity' in report:
        lines.append(f"\nSensitivity Analysis for: {report['feature']}")
        lines.append(f"  Current Value: {report['current_value']:.2f}")
        lines.append(f"  Current Prediction: {report['current_prediction']:.4f}")
        lines.append(f"  Sensitivity: {report['sensitivity']:.4f}")
        opt = report.get('optimal', {})
        if opt:
            lines.append(f"  Optimal Value: {opt['value']:.2f} -> Prediction: {opt['prediction']:.4f}")
    
    return '\n'.join(lines)
