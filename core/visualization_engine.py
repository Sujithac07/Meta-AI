"""
Advanced Visualization Engine
Integrates: NumPy, Pandas, Matplotlib, Seaborn, Plotly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')


class VisualizationEngine:
    """
    Comprehensive visualization system for ML pipelines
    """
    
    def __init__(self, theme: str = 'plotly_dark'):
        """
        Initialize visualization engine
        
        Args:
            theme: Plotly theme ('plotly_dark', 'plotly_white', 'seaborn', etc.)
        """
        self.theme = theme
        self.color_palette = sns.color_palette("husl", 10)
        
    # ==========================================
    # 3D VISUALIZATIONS (PLOTLY)
    # ==========================================
    
    def create_3d_feature_space(
        self, 
        df: pd.DataFrame, 
        method: str = 'pca',
        title: str = "3D Feature Space Visualization"
    ) -> go.Figure:
        # Separate features
        X = df.select_dtypes(include=[np.number])
        if X.empty:
            return go.Figure()
        
        # Simple PCA
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X.fillna(0))
        
        fig = px.scatter_3d(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            z=X_reduced[:, 2],
            title=title,
            template=self.theme,
            opacity=0.7
        )
        return fig

    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create an interactive heatmap of feature correlations"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return go.Figure()
            
        corr = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            template=self.theme,
            height=600
        )
        return fig
    
    # ==========================================
    # STATISTICAL ANALYSIS (SEABORN + MATPLOTLIB)
    # ==========================================
    
    def plot_statistical_analysis(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        max_features: int = 10
    ) -> Dict[str, plt.Figure]:
        """
        Create comprehensive statistical analysis plots
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            max_features: Maximum number of features to plot
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        # 1. Correlation Heatmap
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax1
        )
        ax1.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        figures['correlation_heatmap'] = fig1
        
        # 2. Distribution Plots
        numeric_features = [col for col in numeric_cols if col != target_col][:max_features]
        n_features = len(numeric_features)
        
        if n_features > 0:
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            axes2 = np.array(axes2).flatten() if n_features > 1 else [axes2]
            
            for idx, feature in enumerate(numeric_features):
                sns.histplot(
                    data=df,
                    x=feature,
                    hue=target_col,
                    kde=True,
                    ax=axes2[idx],
                    palette='Set2'
                )
                axes2[idx].set_title(f'Distribution: {feature}', fontweight='bold')
            
            # Hide unused subplots
            for idx in range(n_features, len(axes2)):
                axes2[idx].axis('off')
            
            plt.tight_layout()
            figures['distributions'] = fig2
        
        # 3. Pairplot (if not too many features)
        if len(numeric_features) <= 5 and len(df) <= 1000:
            pairplot_cols = numeric_features[:5] + [target_col]
            fig3 = sns.pairplot(
                df[pairplot_cols],
                hue=target_col,
                palette='husl',
                diag_kind='kde',
                plot_kws={'alpha': 0.6}
            )
            fig3.fig.suptitle('Feature Pairplot', y=1.02, fontsize=16, fontweight='bold')
            figures['pairplot'] = fig3.fig
        
        return figures
    
    # ==========================================
    # MODEL PERFORMANCE (PLOTLY)
    # ==========================================
    
    def visualize_model_performance(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Model Performance Comparison"
    ) -> go.Figure:
        """
        Create interactive model performance comparison
        
        Args:
            metrics_dict: Dictionary of {model_name: {metric: value}}
            title: Plot title
            
        Returns:
            Plotly Figure
        """
        # Convert to DataFrame
        df_metrics = pd.DataFrame(metrics_dict).T
        
        # Create subplots for each metric
        metrics = df_metrics.columns.tolist()
        n_metrics = len(metrics)
        
        fig = make_subplots(
            rows=1,
            cols=n_metrics,
            subplot_titles=metrics,
            specs=[[{"type": "bar"}] * n_metrics]
        )
        
        colors = px.colors.qualitative.Plotly
        
        for idx, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    x=df_metrics.index,
                    y=df_metrics[metric],
                    name=metric,
                    marker_color=colors[idx % len(colors)],
                    text=df_metrics[metric].round(3),
                    textposition='auto',
                    showlegend=False
                ),
                row=1,
                col=idx + 1
            )
        
        fig.update_layout(
            title_text=title,
            template=self.theme,
            height=500,
            showlegend=False
        )
        
        return fig
    
    # ==========================================
    # INTERACTIVE DASHBOARDS (PLOTLY)
    # ==========================================
    
    def create_interactive_dashboard(
        self,
        experiment_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Create comprehensive interactive dashboard
        
        Args:
            experiment_data: Dictionary containing:
                - metrics: Model metrics
                - feature_importance: Feature importance scores
                - confusion_matrix: Confusion matrix (optional)
                
        Returns:
            Plotly Figure with subplots
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Model Accuracy Comparison',
                'Feature Importance',
                'Training Progress',
                'Metric Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "box"}]
            ]
        )
        
        # 1. Model Accuracy Comparison
        if 'metrics' in experiment_data:
            metrics_df = pd.DataFrame(experiment_data['metrics']).T
            if 'accuracy' in metrics_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=metrics_df.index,
                        y=metrics_df['accuracy'],
                        name='Accuracy',
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
        
        # 2. Feature Importance
        if 'feature_importance' in experiment_data:
            feat_imp = experiment_data['feature_importance']
            fig.add_trace(
                go.Bar(
                    x=list(feat_imp.values()),
                    y=list(feat_imp.keys()),
                    orientation='h',
                    name='Importance',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
        
        # 3. Training Progress (if available)
        if 'training_history' in experiment_data:
            history = experiment_data['training_history']
            epochs = list(range(1, len(history['loss']) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['loss'],
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        # 4. Metric Distribution
        if 'metrics' in experiment_data:
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if metric in metrics_df.columns:
                    fig.add_trace(
                        go.Box(
                            y=metrics_df[metric],
                            name=metric.capitalize()
                        ),
                        row=2, col=2
                    )
        
        fig.update_layout(
            title_text="ML Experiment Dashboard",
            template=self.theme,
            height=800,
            showlegend=True
        )
        
        return fig
    
    # ==========================================
    # ANIMATED VISUALIZATIONS (PLOTLY)
    # ==========================================
    
    def animate_training_progress(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Progress Animation"
    ) -> go.Figure:
        """
        Create animated visualization of training progress
        
        Args:
            history: Training history with 'loss', 'accuracy', etc.
            title: Plot title
            
        Returns:
            Animated Plotly Figure
        """
        epochs = list(range(1, len(history['loss']) + 1))
        
        # Create frames for animation
        frames = []
        for i in range(1, len(epochs) + 1):
            frame_data = []
            
            # Loss trace
            frame_data.append(
                go.Scatter(
                    x=epochs[:i],
                    y=history['loss'][:i],
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='red', width=3)
                )
            )
            
            # Accuracy trace (if available)
            if 'accuracy' in history:
                frame_data.append(
                    go.Scatter(
                        x=epochs[:i],
                        y=history['accuracy'][:i],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='green', width=3),
                        yaxis='y2'
                    )
                )
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
        
        # Initial figure
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title='Epoch', range=[0, len(epochs) + 1]),
            yaxis=dict(title='Loss', side='left'),
            yaxis2=dict(title='Accuracy', overlaying='y', side='right'),
            template=self.theme,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 200, 'redraw': True},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': str(k),
                        'method': 'animate'
                    }
                    for k, f in enumerate(frames)
                ],
                'active': 0,
                'y': 0,
                'len': 0.9,
                'x': 0.1
            }]
        )
        
        return fig
    
    # ==========================================
    # CONFUSION MATRIX (SEABORN)
    # ==========================================
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix"
    ) -> plt.Figure:
        """
        Create beautiful confusion matrix visualization
        
        Args:
            cm: Confusion matrix array
            class_names: Optional class names
            title: Plot title
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=cm,
            fmt='d',
            cmap='Blues',
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto',
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def save_plotly_figure(fig: go.Figure, filepath: str, format: str = 'html'):
    """Save Plotly figure to file"""
    if format == 'html':
        fig.write_html(filepath)
    elif format == 'png':
        fig.write_image(filepath)
    elif format == 'json':
        fig.write_json(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_matplotlib_figure(fig: plt.Figure, filepath: str, dpi: int = 300):
    """Save Matplotlib figure to file"""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
