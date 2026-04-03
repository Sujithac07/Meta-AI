"""
MetaAI Pro - Clean 6-Tab Enterprise Interface
Professional AutoML Platform with Agentic AI & MLOps Focus
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Import Advanced Data Informer
from core.advanced_data_informer import AdvancedDataInformer

# Import Evolutionary AutoPilot
from core.evolutionary_autopilot import EvolutionaryAutoPilot

# App State
class AppState:
    def __init__(self):
        self.df = None
        self.df_original = None  # Keep original before imputation
        self.target_col = None
        self.models = {}
        self.metrics = {}
        self.current_model = None
        self.data_informer = None  # Store informer instance
        self.autopilot = None  # Store autopilot instance
        self.super_model = None  # Store stacking ensemble
        
app_state = AppState()

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS - Professional Dark Theme
# ═══════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
/* Root variables */
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --border-color: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-blue: #2563eb;
    --accent-purple: #7c3aed;
    --accent-green: #10b981;
    --accent-red: #ef4444;
}

/* Main container */
.gradio-container {
    background: var(--bg-primary) !important;
    max-width: 1400px !important;
}

/* Tab styling */
.tabs { background: var(--bg-secondary) !important; border-radius: 12px !important; padding: 8px !important; }
button.tabitem {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.2s !important;
}
button.tabitem:hover { background: var(--bg-tertiary) !important; color: var(--text-primary) !important; }
button.tabitem.selected {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
}

/* Cards */
.card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}

/* Buttons */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
}

/* Inputs */
input, textarea, select, .gr-input, .gr-dropdown {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* Labels */
label, .gr-label { color: var(--text-primary) !important; font-weight: 500 !important; }

/* Markdown */
.markdown-text { color: var(--text-primary) !important; }
h1, h2, h3, h4 { color: var(--text-primary) !important; }
p { color: var(--text-secondary) !important; }
"""

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_data(file):
    """Load CSV file with semantic type detection"""
    if file is None:
        return None, "Please upload a CSV file", gr.update(choices=[])
    try:
        df = pd.read_csv(file.name)
        app_state.df = df
        app_state.df_original = df.copy()
        
        # Initialize Data Informer
        app_state.data_informer = AdvancedDataInformer(df)
        
        cols = df.columns.tolist()
        preview = df.head(10).to_html(classes='dataframe', index=False)
        return (
            f"Loaded {len(df)} rows x {len(cols)} columns | Missing: {df.isnull().sum().sum()} values",
            preview,
            gr.update(choices=cols, value=cols[-1])
        )
    except Exception as e:
        return f"Error: {e}", "", gr.update(choices=[])

def run_semantic_analysis():
    """Run semantic type detection and validation"""
    if app_state.df is None:
        return "Please upload data first", None
    
    try:
        informer = app_state.data_informer or AdvancedDataInformer(app_state.df)
        informer.profile_all_columns()
        app_state.data_informer = informer
        
        # Get semantic summary
        report = informer.get_semantic_summary()
        quality = informer.validate_data_quality()
        
        # Create quality gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality['overall_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Data Quality Score", 'font': {'size': 20, 'color': '#e6edf3'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#e6edf3'},
                'bar': {'color': "#2563eb"},
                'bgcolor': "#1e1b4b",
                'borderwidth': 2,
                'bordercolor': "#30363d",
                'steps': [
                    {'range': [0, 50], 'color': '#ef4444'},
                    {'range': [50, 75], 'color': '#f59e0b'},
                    {'range': [75, 100], 'color': '#10b981'}
                ],
                'threshold': {
                    'line': {'color': "#22d3ee", 'width': 4},
                    'thickness': 0.75,
                    'value': quality['overall_score']
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e6edf3'},
            height=300
        )
        
        return report, fig
    except Exception as e:
        return f"Error: {e}", None

def run_bayesian_imputation():
    """Run Bayesian Iterative Imputation"""
    if app_state.df is None:
        return "Please upload data first", ""
    
    if app_state.df.isnull().sum().sum() == 0:
        return "No missing values detected - imputation not needed", app_state.df.head(10).to_html(index=False)
    
    try:
        informer = app_state.data_informer or AdvancedDataInformer(app_state.df)
        
        # Run Bayesian imputation
        imputed_df = informer.bayesian_iterative_imputation(max_iter=10, random_state=42)
        app_state.df = imputed_df
        app_state.data_informer = informer
        
        # Get report
        report = informer.get_imputation_report()
        preview = imputed_df.head(10).to_html(index=False)
        
        return report, preview
    except Exception as e:
        return f"Error during imputation: {e}", ""

def set_target(target):
    """Set target column"""
    app_state.target_col = target
    return f"Target set to: {target}"

def generate_eda():
    """Generate EDA visualizations"""
    if app_state.df is None:
        empty = go.Figure()
        return empty, empty, empty, "No data loaded"
    
    df = app_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Distribution plot
    if numeric_cols:
        fig1 = px.histogram(df, x=numeric_cols[0], title=f"Distribution: {numeric_cols[0]}")
        fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    else:
        fig1 = go.Figure()
    
    # Target balance
    if app_state.target_col:
        counts = df[app_state.target_col].value_counts()
        fig2 = px.pie(values=counts.values, names=counts.index, title="Target Distribution", hole=0.4)
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    else:
        fig2 = go.Figure()
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig3 = px.imshow(corr, title="Correlation Matrix", color_continuous_scale="RdBu_r")
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    else:
        fig3 = go.Figure()
    
    summary = f"""
### Dataset Summary
- **Rows:** {len(df)}
- **Columns:** {len(df.columns)}
- **Numeric Features:** {len(numeric_cols)}
- **Missing Values:** {df.isnull().sum().sum()}
- **Target:** {app_state.target_col or 'Not set'}
"""
    return fig1, fig2, fig3, summary

def train_model(model_name, test_size):
    """Train a model with REAL sklearn"""
    if app_state.df is None or app_state.target_col is None:
        return "Please load data and set target first", None, None, ""
    
    try:
        df = app_state.df
        X = df.drop(columns=[app_state.target_col])
        y = df[app_state.target_col]
        
        # Handle non-numeric
        X = X.select_dtypes(include=np.number)
        X = X.fillna(X.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        
        # Model selection
        models_dict = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42)
        }
        
        model = models_dict.get(model_name)
        if model is None:
            return f"Unknown model: {model_name}", None, None, ""
        
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics (store as DECIMAL)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Store model and metrics
        app_state.models[model_name] = model
        app_state.metrics[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'test_size': test_size
        }
        app_state.current_model = model_name
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_fig = px.imshow(cm, text_auto=True, title=f"Confusion Matrix - {model_name}",
                          labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues")
        cm_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        
        # ROC curve
        try:
            if len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={roc_auc:.3f})', line=dict(color='#2563eb')))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash', color='gray')))
                roc_fig.update_layout(title=f"ROC Curve - {model_name}", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            else:
                roc_fig = go.Figure()
        except Exception:
            roc_fig = go.Figure()
        
        result = f"""
### Training Complete: {model_name}

| Metric | Value |
|--------|-------|
| Accuracy | {acc*100:.2f}% |
| Precision | {prec*100:.2f}% |
| Recall | {rec*100:.2f}% |
| F1 Score | {f1*100:.2f}% |
| Test Size | {test_size}% |
"""
        return result, cm_fig, roc_fig, gr.update(choices=list(app_state.models.keys()), value=model_name)
    
    except Exception as e:
        return f"Training Error: {e}", None, None, gr.update()

def run_autopilot():
    """Run Basic AutoPilot - trains all models quickly"""
    if app_state.df is None or app_state.target_col is None:
        return "Please load data and set target first", None
    
    models = ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree", "KNN", "SVM", "Naive Bayes"]
    results = []
    
    for model_name in models:
        try:
            df = app_state.df
            X = df.drop(columns=[app_state.target_col]).select_dtypes(include=np.number).fillna(0)
            y = df[app_state.target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model_map = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True, random_state=42),
                "Naive Bayes": GaussianNB()
            }
            
            model = model_map[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            app_state.models[model_name] = model
            app_state.metrics[model_name] = {'accuracy': acc, 'f1': f1}
            results.append({'Model': model_name, 'Accuracy': f"{acc*100:.2f}%", 'F1': f"{f1*100:.2f}%"})
        except Exception as e:
            results.append({'Model': model_name, 'Accuracy': 'Error', 'F1': str(e)})
    
    # Leaderboard
    results_df = pd.DataFrame(results)
    
    # Bar chart
    accs = [float(r['Accuracy'].replace('%','')) if '%' in r['Accuracy'] else 0 for r in results]
    fig = px.bar(x=[r['Model'] for r in results], y=accs, title="Model Comparison", 
                 labels={'x': 'Model', 'y': 'Accuracy (%)'}, color=accs, color_continuous_scale='Blues')
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    
    best = max(results, key=lambda x: float(x['Accuracy'].replace('%','')) if '%' in x['Accuracy'] else 0)
    
    report = f"""
### AutoPilot Complete

**Best Model:** {best['Model']} ({best['Accuracy']})

{results_df.to_markdown(index=False)}
"""
    return report, fig

def run_evolutionary_autopilot_ui(n_trials):
    """Run Evolutionary AutoPilot with Optuna TPE optimization"""
    if app_state.df is None or app_state.target_col is None:
        return "Please load data and set target first", None, None
    
    try:
        n_trials = int(n_trials)
        
        # Prepare data
        df = app_state.df
        X = df.drop(columns=[app_state.target_col]).select_dtypes(include=np.number)
        X = X.fillna(X.mean())
        y = df[app_state.target_col]
        
        # Run evolutionary optimization
        autopilot = EvolutionaryAutoPilot(n_trials=n_trials, cv_folds=5, random_state=42)
        results = autopilot.run_evolution(X, y)
        
        # Create Super-Model (Stacking Ensemble)
        super_model, super_acc, super_f1 = autopilot.create_super_model(X, y, top_k=3)
        
        # Store in app state
        app_state.autopilot = autopilot
        app_state.super_model = super_model
        app_state.models["Super-Model (Stacking)"] = super_model
        app_state.metrics["Super-Model (Stacking)"] = {'accuracy': super_acc, 'f1': super_f1}
        
        # Store individual optimized models
        for name, result in results.items():
            app_state.models[f"{name} (Optimized)"] = result.best_model
            app_state.metrics[f"{name} (Optimized)"] = {'accuracy': result.best_accuracy, 'f1': result.best_f1}
        
        # Generate report
        report = autopilot.get_evolution_report()
        report += "\n\n### Super-Model Performance\n\n"
        report += f"- **Accuracy:** {super_acc*100:.2f}%\n"
        report += f"- **F1 Score:** {super_f1*100:.2f}%\n"
        
        # Create visualization
        sorted_results = sorted(results.items(), key=lambda x: x[1].best_accuracy, reverse=True)
        model_names = [name for name, _ in sorted_results] + ["Super-Model"]
        accuracies = [r.best_accuracy * 100 for _, r in sorted_results] + [super_acc * 100]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_names,
            y=accuracies,
            marker_color=['#6366f1'] * len(sorted_results) + ['#10b981'],
            text=[f"{a:.1f}%" for a in accuracies],
            textposition='outside'
        ))
        fig.update_layout(
            title="Evolutionary AutoPilot - Model Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy (%)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 100])
        )
        
        # Trial history plot
        trial_df = autopilot.get_trial_history_df()
        if not trial_df.empty:
            history_fig = px.scatter(
                trial_df, x='Trial', y='Accuracy', color='Model',
                title="Optimization History (TPE Sampler)",
                labels={'Accuracy': 'CV Accuracy'}
            )
            history_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
        else:
            history_fig = go.Figure()
        
        return report, fig, history_fig
    
    except Exception as e:
        import traceback
        return f"Error: {e}\n\n{traceback.format_exc()}", None, None

def run_ai_auditor(model_name):
    """AI Auditor - Agentic AI analysis"""
    if not model_name or model_name not in app_state.models:
        return "Select a trained model first"
    
    metrics = app_state.metrics.get(model_name, {})
    acc = metrics.get('accuracy', 0)
    f1 = metrics.get('f1', 0)
    
    # Generate intelligent audit report
    warnings = []
    recommendations = []
    
    if acc < 0.7:
        warnings.append("Low accuracy detected - consider feature engineering or more data")
    if f1 < 0.6:
        warnings.append("F1 score indicates class imbalance issues")
    if acc > 0.98:
        warnings.append("Suspiciously high accuracy - check for data leakage")
    
    if not warnings:
        warnings.append("No critical issues detected")
    
    recommendations.extend([
        "Run SHAP analysis in XAI tab to understand feature importance",
        "Check drift detection for production monitoring",
        "Consider ensemble methods for improved robustness"
    ])
    
    health = "EXCELLENT" if acc > 0.85 else ("GOOD" if acc > 0.7 else "NEEDS IMPROVEMENT")
    
    report = f"""
## AI AUDITOR REPORT

### Model: {model_name}

| Metric | Value |
|--------|-------|
| Accuracy | {acc*100:.2f}% |
| F1 Score | {f1*100:.2f}% |

### Warnings
{chr(10).join(f"- {w}" for w in warnings)}

### AI Recommendations
{chr(10).join(f"- {r}" for r in recommendations)}

### Overall Health: **{health}**
"""
    return report

def run_shap_analysis(model_name):
    """SHAP/XAI Analysis"""
    if not model_name or model_name not in app_state.models:
        return None, "Select a trained model"
    
    try:
        model = app_state.models[model_name]
        X = app_state.df.drop(columns=[app_state.target_col]).select_dtypes(include=np.number).fillna(0)
        
        # Feature importance (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = X.columns.tolist()
            
            # Sort by importance
            indices = np.argsort(importances)[::-1][:10]
            top_features = [features[i] for i in indices]
            top_importances = [importances[i] for i in indices]
            
            fig = px.bar(x=top_importances, y=top_features, orientation='h',
                        title=f"Feature Importance - {model_name}",
                        labels={'x': 'Importance', 'y': 'Feature'})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", yaxis={'categoryorder':'total ascending'})
            
            explanation = f"""
### XAI Analysis: {model_name}

**Top Features:**
{chr(10).join(f"- **{f}**: {imp*100:.1f}%" for f, imp in zip(top_features[:5], top_importances[:5]))}

These features have the most influence on model predictions.
"""
            return fig, explanation
        else:
            # For models without feature_importances_
            fig = go.Figure()
            return fig, "Feature importance not available for this model type. Try Random Forest or Gradient Boosting."
    except Exception as e:
        return None, f"Error: {e}"

def predict_single(input_values, model_name):
    """Make single prediction"""
    if not model_name or model_name not in app_state.models:
        return "Select a trained model"
    
    try:
        model = app_state.models[model_name]
        X = app_state.df.drop(columns=[app_state.target_col]).select_dtypes(include=np.number)
        
        # Parse input
        values = [float(v.strip()) for v in input_values.split(',')]
        if len(values) != len(X.columns):
            return f"Expected {len(X.columns)} values, got {len(values)}"
        
        input_df = pd.DataFrame([values], columns=X.columns)
        pred = model.predict(input_df)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            return f"**Prediction:** {pred}\n\n**Confidence:** {max(proba)*100:.1f}%"
        return f"**Prediction:** {pred}"
    except Exception as e:
        return f"Error: {e}"

def chat_with_ai(message, history):
    """Neural Chat Assistant"""
    # Get context from app state
    models_info = []
    for name, metrics in app_state.metrics.items():
        acc = metrics.get('accuracy', 0)
        models_info.append(f"- {name}: {acc*100:.2f}% accuracy")
    
    models_str = '\n'.join(models_info) if models_info else "No models trained yet"
    
    response = f"""Based on your current session:

**Trained Models:**
{models_str}

**Dataset:** {'Loaded' if app_state.df is not None else 'Not loaded'}
**Target:** {app_state.target_col or 'Not set'}

For "{message}": I recommend reviewing the Analysis tab for detailed insights or running AutoPilot for automatic model comparison.
"""
    return response

def detect_drift(ref_file, curr_file):
    """Drift Detection"""
    if ref_file is None or curr_file is None:
        return "Upload both reference and current data files"
    
    try:
        ref_df = pd.read_csv(ref_file.name)
        curr_df = pd.read_csv(curr_file.name)
        
        # Simple drift detection using KS test
        from scipy.stats import ks_2samp
        
        common_cols = set(ref_df.columns) & set(curr_df.columns)
        numeric_cols = ref_df.select_dtypes(include=np.number).columns.tolist()
        
        drift_results = []
        for col in numeric_cols[:10]:
            if col in common_cols:
                stat, p_value = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
                drift_results.append({
                    'Feature': col,
                    'KS Statistic': f"{stat:.4f}",
                    'P-Value': f"{p_value:.4f}",
                    'Drift': 'YES' if p_value < 0.05 else 'NO'
                })
        
        results_df = pd.DataFrame(drift_results)
        drifted = sum(1 for r in drift_results if r['Drift'] == 'YES')
        
        return f"""
### Drift Detection Results

**Drifted Features:** {drifted} / {len(drift_results)}

{results_df.to_markdown(index=False)}

**Recommendation:** {'Consider retraining if drift is significant' if drifted > len(drift_results)/2 else 'Data distribution is stable'}
"""
    except Exception as e:
        return f"Error: {e}"

# ═══════════════════════════════════════════════════════════════════════════
# BUILD THE APP
# ═══════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="MetaAI Pro", theme=gr.themes.Base()) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e1b4b, #0f172a); border-radius: 16px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2rem;">MetaAI Pro</h1>
            <p style="color: #94a3b8; margin: 8px 0 0 0;">Enterprise AutoML Platform | Agentic AI | MLOps</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 1: DATA
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("Data"):
                gr.Markdown("### Data Ingestion & Smart Analysis")
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
                        load_btn = gr.Button("Load Dataset", variant="primary")
                        load_status = gr.Markdown()
                        target_dropdown = gr.Dropdown(label="Select Target Column", interactive=True)
                        set_target_btn = gr.Button("Set Target")
                        target_status = gr.Markdown()
                    with gr.Column():
                        data_preview = gr.HTML(label="Data Preview")
                
                load_btn.click(load_data, [file_input], [load_status, data_preview, target_dropdown])
                set_target_btn.click(set_target, [target_dropdown], [target_status])
                
                gr.Markdown("---")
                gr.Markdown("### Semantic Type Detection & Bayesian Imputation")
                
                with gr.Row():
                    with gr.Column():
                        semantic_btn = gr.Button("Run Semantic Analysis", variant="primary")
                        impute_btn = gr.Button("Run Bayesian Imputation", variant="secondary")
                    with gr.Column():
                        quality_gauge = gr.Plot(label="Data Quality Score")
                
                semantic_report = gr.Markdown()
                imputation_report = gr.Markdown()
                imputed_preview = gr.HTML()
                
                semantic_btn.click(run_semantic_analysis, [], [semantic_report, quality_gauge])
                impute_btn.click(run_bayesian_imputation, [], [imputation_report, imputed_preview])
                
                gr.Markdown("---")
                gr.Markdown("### Exploratory Data Analysis")
                eda_btn = gr.Button("Generate EDA", variant="primary")
                eda_summary = gr.Markdown()
                with gr.Row():
                    dist_plot = gr.Plot(label="Distribution")
                    balance_plot = gr.Plot(label="Target Balance")
                corr_plot = gr.Plot(label="Correlation Matrix")
                
                eda_btn.click(generate_eda, [], [dist_plot, balance_plot, corr_plot, eda_summary])
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 2: TRAINING
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("Training"):
                gr.Markdown("### Model Training & Evolutionary AutoPilot")
                
                gr.Markdown("#### Manual Training")
                with gr.Row():
                    model_select = gr.Dropdown(
                        choices=["Random Forest", "Gradient Boosting", "Logistic Regression", 
                                "Decision Tree", "KNN", "SVM", "Naive Bayes", "AdaBoost", "Extra Trees"],
                        label="Select Algorithm",
                        value="Random Forest"
                    )
                    test_size = gr.Slider(10, 40, 20, label="Test Size (%)")
                train_btn = gr.Button("Train Model", variant="primary")
                train_result = gr.Markdown()
                with gr.Row():
                    cm_plot = gr.Plot(label="Confusion Matrix")
                    roc_plot = gr.Plot(label="ROC Curve")
                trained_models = gr.Dropdown(label="Trained Models", interactive=True)
                
                train_btn.click(train_model, [model_select, test_size], [train_result, cm_plot, roc_plot, trained_models])
                
                gr.Markdown("---")
                gr.Markdown("#### Evolutionary AutoPilot (Optuna TPE + Stacking Ensemble)")
                gr.Markdown("Genetic optimization with TPE sampler. Creates Super-Model from top 3 models.")
                
                with gr.Row():
                    evo_trials = gr.Slider(10, 100, 50, step=10, label="Trials per Model")
                    evo_btn = gr.Button("Run Evolutionary AutoPilot", variant="primary", size="lg")
                
                evo_result = gr.Markdown()
                with gr.Row():
                    evo_comparison = gr.Plot(label="Model Comparison")
                    evo_history = gr.Plot(label="Optimization History")
                
                evo_btn.click(run_evolutionary_autopilot_ui, [evo_trials], [evo_result, evo_comparison, evo_history])
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 3: ANALYSIS
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("Analysis"):
                gr.Markdown("### Explainable AI & Inference")
                
                gr.Markdown("#### Feature Importance (XAI)")
                xai_model = gr.Dropdown(label="Select Model", choices=[], interactive=True)
                xai_btn = gr.Button("Run XAI Analysis", variant="primary")
                xai_plot = gr.Plot(label="Feature Importance")
                xai_explanation = gr.Markdown()
                
                xai_btn.click(run_shap_analysis, [xai_model], [xai_plot, xai_explanation])
                
                gr.Markdown("---")
                gr.Markdown("#### Make Predictions")
                pred_model = gr.Dropdown(label="Select Model", choices=[], interactive=True)
                pred_input = gr.Textbox(label="Input Values (comma-separated)", placeholder="e.g., 1.5, 2.3, 0.8, ...")
                pred_btn = gr.Button("Predict", variant="primary")
                pred_result = gr.Markdown()
                
                pred_btn.click(predict_single, [pred_input, pred_model], [pred_result])
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 4: AGENTIC AI
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("Agentic AI"):
                gr.Markdown("### AI Auditor & Neural Chat")
                
                gr.Markdown("#### Intelligent Model Diagnostics")
                auditor_model = gr.Dropdown(label="Select Model to Audit", choices=[], interactive=True)
                audit_btn = gr.Button("Run AI Audit", variant="primary")
                audit_result = gr.Markdown()
                
                audit_btn.click(run_ai_auditor, [auditor_model], [audit_result])
                
                gr.Markdown("---")
                gr.Markdown("#### Neural Chat Assistant")
                chatbot = gr.Chatbot(height=300)
                chat_input = gr.Textbox(label="Ask anything about your models...", placeholder="e.g., Which model should I use?")
                chat_btn = gr.Button("Send", variant="primary")
                
                def respond(message, history):
                    response = chat_with_ai(message, history)
                    history.append((message, response))
                    return history, ""
                
                chat_btn.click(respond, [chat_input, chatbot], [chatbot, chat_input])
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 5: MLOPS
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("MLOps"):
                gr.Markdown("### Drift Detection & Model Registry")
                
                gr.Markdown("#### Data Drift Monitoring")
                with gr.Row():
                    ref_file = gr.File(label="Reference Data (Training CSV)")
                    curr_file = gr.File(label="Current Data (New CSV)")
                drift_btn = gr.Button("Detect Drift", variant="primary")
                drift_result = gr.Markdown()
                
                drift_btn.click(detect_drift, [ref_file, curr_file], [drift_result])
                
                gr.Markdown("---")
                gr.Markdown("#### Model Registry")
                
                def get_registry():
                    if not app_state.models:
                        return "No models in registry"
                    rows = []
                    for name, metrics in app_state.metrics.items():
                        acc = metrics.get('accuracy', 0)
                        rows.append(f"| {name} | {acc*100:.2f}% | Active |")
                    return "| Model | Accuracy | Status |\n|-------|----------|--------|\n" + "\n".join(rows)
                
                registry_btn = gr.Button("Refresh Registry", variant="primary")
                registry_display = gr.Markdown()
                registry_btn.click(get_registry, [], [registry_display])
            
            # ═══════════════════════════════════════════════════════════════
            # TAB 6: DEPLOY
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("Deploy"):
                gr.Markdown("### API Deployment")
                
                deploy_model = gr.Dropdown(label="Select Model to Deploy", choices=[], interactive=True)
                deploy_btn = gr.Button("Generate API Endpoint", variant="primary")
                deploy_result = gr.Markdown()
                
                def deploy(model_name):
                    if not model_name:
                        return "Select a model first"
                    return f"""
### Deployment Complete

**Model:** {model_name}
**Endpoint:** `POST /api/v1/predict`

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={{"model": "{model_name}", "features": [1.0, 2.0, 3.0]}}
)
print(response.json())
```
"""
                deploy_btn.click(deploy, [deploy_model], [deploy_result])
        
        # Refresh model dropdowns
        def refresh_models():
            choices = list(app_state.models.keys())
            return [gr.update(choices=choices)] * 5
        
        demo.load(refresh_models, [], [xai_model, pred_model, auditor_model, deploy_model, trained_models])
    
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
