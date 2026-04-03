"""
MetaAI Pro - Step-by-Step Wizard UI
Enterprise AutoML Platform with Guided Pipeline
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import json
from typing import Dict

# Import core modules
from core.smart_ingestion import SmartIngestionEngine, format_ingestion_report
from core.forensic_cleaner import ForensicCleaner, format_forensic_report
from core.auto_feature_engineer import AutoFeatureEngineer, format_feature_report
from core.elite_trainer import EliteTrainer, format_tournament_report, OPTUNA_AVAILABLE
from core.black_box_breaker import BlackBoxBreaker, format_xai_report
from core.deployment_guard import DeploymentGuard, format_drift_report


# Global state
class AppState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.df_raw = None
        self.df_cleaned = None
        self.df_engineered = None
        self.target_column = None
        self.task_type = "classification"
        self.ingestion_report = None
        self.cleaning_report = None
        self.feature_report = None
        self.training_report = None
        self.model = None
        self.explainer = None
        self.guard = None
        self.current_step = 0
        self.raw_stats = {}
        self.cleaned_stats = {}

state = AppState()


def create_step_indicator(current: int, total: int = 6) -> str:
    """Create visual step indicator."""
    steps = ["Data Quality", "Cleaning", "Features", "Training", "Explainability", "Deployment"]
    lines = ["PIPELINE PROGRESS", "=" * 50]
    
    for i, step in enumerate(steps):
        if i < current:
            status = "[DONE]"
        elif i == current:
            status = "[CURRENT]"
        else:
            status = "[PENDING]"
        lines.append(f"Step {i+1}: {step} {status}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


# ==================== STEP 1: DATA QUALITY ====================

def step1_load_data(file, target_col):
    """Step 1: Load data and show quality report."""
    if file is None:
        return "Please upload a CSV file", None, create_step_indicator(0), None
    
    try:
        state.reset()
        state.df_raw = pd.read_csv(file.name)
        state.target_column = target_col
        
        # Detect task type
        if target_col and target_col in state.df_raw.columns:
            unique_vals = state.df_raw[target_col].nunique()
            state.task_type = "classification" if unique_vals <= 20 else "regression"
        
        # Run smart ingestion
        engine = SmartIngestionEngine()
        state.ingestion_report = engine.smart_ingest(state.df_raw)
        
        # Store raw stats for comparison
        numeric_cols = state.df_raw.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            state.raw_stats[col] = {
                'mean': state.df_raw[col].mean(),
                'std': state.df_raw[col].std(),
                'values': state.df_raw[col].dropna().values
            }
        
        # Format report
        report = format_ingestion_report(state.ingestion_report)
        
        # Create quality visualization
        quality_plot = create_quality_plot(state.ingestion_report)
        
        state.current_step = 1
        
        return report, quality_plot, create_step_indicator(1), gr.update(interactive=True)
        
    except Exception as e:
        return f"Error: {str(e)}", None, create_step_indicator(0), None


def create_quality_plot(report: Dict) -> str:
    """Create data quality visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Quality Score Gauge
    ax1 = axes[0]
    score = report.get('quality_report', {}).get('overall_score', 0)
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
    color = colors[0] if score < 50 else colors[1] if score < 80 else colors[2]
    ax1.pie([score, 100-score], colors=[color, '#e0e0e0'], startangle=90,
            wedgeprops=dict(width=0.3))
    ax1.text(0, 0, f'{score:.0f}', ha='center', va='center', fontsize=24, fontweight='bold')
    ax1.set_title('Quality Score', fontsize=12, fontweight='bold')
    
    # Column Categories
    ax2 = axes[1]
    col_analysis = report.get('column_analysis', {})
    categories = {}
    for col, info in col_analysis.items():
        cat = info.get('category', 'UNKNOWN')
        categories[cat] = categories.get(cat, 0) + 1
    
    if categories:
        ax2.barh(list(categories.keys()), list(categories.values()), color='#4ecdc4')
        ax2.set_xlabel('Count')
        ax2.set_title('Column Categories', fontsize=12, fontweight='bold')
    
    # Missing Data
    ax3 = axes[2]
    missing_pct = report.get('quality_report', {}).get('missing_percentage', 0)
    complete_pct = 100 - missing_pct
    ax3.pie([complete_pct, missing_pct], labels=['Complete', 'Missing'],
            colors=['#4ecdc4', '#ff6b6b'], autopct='%1.1f%%', startangle=90)
    ax3.set_title('Data Completeness', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ==================== STEP 2: CLEANING ====================

def step2_clean_data():
    """Step 2: Clean data and show before/after comparison."""
    if state.df_raw is None:
        return "Please complete Step 1 first", None, create_step_indicator(0), None
    
    try:
        # Run forensic cleaner
        cleaner = ForensicCleaner()
        exclude = [state.target_column] if state.target_column else []
        state.df_cleaned, state.cleaning_report = cleaner.full_reconstruction(
            state.df_raw.copy(), exclude
        )
        
        # Store cleaned stats
        numeric_cols = state.df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['anomaly_label', 'anomaly_score']:
                state.cleaned_stats[col] = {
                    'mean': state.df_cleaned[col].mean(),
                    'std': state.df_cleaned[col].std(),
                    'values': state.df_cleaned[col].dropna().values
                }
        
        # Format report
        report = format_forensic_report(state.cleaning_report)
        
        # Create before/after plot
        comparison_plot = create_before_after_plot()
        
        state.current_step = 2
        
        return report, comparison_plot, create_step_indicator(2), gr.update(interactive=True)
        
    except Exception as e:
        return f"Error: {str(e)}", None, create_step_indicator(1), None


def create_before_after_plot() -> str:
    """Create before vs after distribution comparison."""
    # Get common columns
    common_cols = [c for c in state.raw_stats.keys() 
                   if c in state.cleaned_stats and c != state.target_column][:6]
    
    if not common_cols:
        return None
    
    n_cols = len(common_cols)
    fig, axes = plt.subplots(2, min(n_cols, 3), figsize=(14, 8))
    if n_cols < 3:
        axes = axes.reshape(2, -1)
    
    for i, col in enumerate(common_cols[:3]):
        # Before
        ax_before = axes[0, i] if n_cols > 1 else axes[0]
        raw_vals = state.raw_stats[col]['values']
        ax_before.hist(raw_vals, bins=30, color='#ff6b6b', alpha=0.7, edgecolor='black')
        ax_before.set_title(f'{col}\nBEFORE', fontsize=10)
        ax_before.axvline(np.mean(raw_vals), color='red', linestyle='--', label=f'Mean: {np.mean(raw_vals):.2f}')
        ax_before.legend(fontsize=8)
        
        # After
        ax_after = axes[1, i] if n_cols > 1 else axes[1]
        clean_vals = state.cleaned_stats[col]['values']
        ax_after.hist(clean_vals, bins=30, color='#4ecdc4', alpha=0.7, edgecolor='black')
        ax_after.set_title(f'{col}\nAFTER', fontsize=10)
        ax_after.axvline(np.mean(clean_vals), color='green', linestyle='--', label=f'Mean: {np.mean(clean_vals):.2f}')
        ax_after.legend(fontsize=8)
    
    plt.suptitle('Before vs After Cleaning - Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ==================== STEP 3: FEATURE ENGINEERING ====================

def step3_engineer_features():
    """Step 3: Auto feature engineering."""
    if state.df_cleaned is None:
        return "Please complete Step 2 first", None, create_step_indicator(1), None
    
    try:
        # Prepare data (remove anomaly columns)
        df = state.df_cleaned.copy()
        df = df.drop(columns=['anomaly_label', 'anomaly_score'], errors='ignore')
        
        # Run feature engineering
        engineer = AutoFeatureEngineer()
        state.df_engineered, state.feature_report = engineer.auto_engineer(
            df, state.target_column, state.task_type
        )
        
        # Format report
        report = format_feature_report(state.feature_report)
        
        # Create feature importance plot
        feature_plot = create_feature_plot(state.feature_report)
        
        state.current_step = 3
        
        return report, feature_plot, create_step_indicator(3), gr.update(interactive=True)
        
    except Exception as e:
        return f"Error: {str(e)}", None, create_step_indicator(2), None


def create_feature_plot(report: Dict) -> str:
    """Create feature engineering visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feature count comparison
    ax1 = axes[0]
    orig = report.get('original_features', 0)
    final = report.get('final_features', 0)
    new = report.get('new_features_created', 0)
    dropped = report.get('features_dropped', 0)
    
    categories = ['Original', 'New Created', 'Dropped', 'Final']
    values = [orig, new, dropped, final]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    ax1.bar(categories, values, color=colors)
    ax1.set_ylabel('Count')
    ax1.set_title('Feature Engineering Summary', fontsize=12, fontweight='bold')
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # Top interactions
    ax2 = axes[1]
    interactions = report.get('interaction_discovery', {}).get('top_interactions', [])
    if interactions:
        names = [i['name'][:20] for i in interactions[:8]]
        corrs = [i['correlation'] for i in interactions[:8]]
        colors = ['#4ecdc4' if i['type'] == 'product' else '#ff6b6b' for i in interactions[:8]]
        
        ax2.barh(names[::-1], corrs[::-1], color=colors[::-1])
        ax2.set_xlabel('Correlation with Target')
        ax2.set_title('Top Engineered Features', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No interactions created', ha='center', va='center')
        ax2.set_title('Top Engineered Features', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ==================== STEP 4: TRAINING ====================

def step4_train_models(n_trials):
    """Step 4: Elite training tournament."""
    if state.df_engineered is None:
        return "Please complete Step 3 first", None, create_step_indicator(2), None
    
    if not OPTUNA_AVAILABLE:
        return "Optuna not installed. Run: pip install optuna", None, create_step_indicator(3), None
    
    try:
        # Prepare data
        X = state.df_engineered.drop(columns=[state.target_column])
        y = state.df_engineered[state.target_column]
        
        # Run elite tournament
        trainer = EliteTrainer(n_trials=int(n_trials))
        state.model, state.training_report = trainer.run_tournament(
            X, y, state.task_type
        )
        
        # Store for XAI
        state.X_train = X
        state.y_train = y
        
        # Format report
        report = format_tournament_report(state.training_report)
        
        # Create optimization plot
        training_plot = create_training_plot(state.training_report)
        
        state.current_step = 4
        
        return report, training_plot, create_step_indicator(4), gr.update(interactive=True)
        
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", None, create_step_indicator(3), None


def create_training_plot(report: Dict) -> str:
    """Create training tournament visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Model Rankings
    ax1 = axes[0]
    rankings = report.get('rankings', [])
    if rankings:
        models = [r['model'] for r in rankings]
        scores = [r['score'] for r in rankings]
        colors = ['#ffd700', '#c0c0c0', '#cd7f32'] + ['#3498db'] * (len(rankings) - 3)
        
        bars = ax1.barh(models[::-1], scores[::-1], color=colors[:len(rankings)][::-1])
        ax1.set_xlabel('Score')
        ax1.set_title('Tournament Rankings', fontsize=12, fontweight='bold')
        ax1.set_xlim(min(scores) * 0.95, max(scores) * 1.02)
        
        for i, (bar, score) in enumerate(zip(bars, scores[::-1])):
            ax1.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontweight='bold')
    
    # Super Model Comparison
    ax2 = axes[1]
    super_m = report.get('super_model', {})
    if super_m.get('status') == 'success':
        best_single = super_m.get('best_single_score', 0)
        super_score = super_m.get('super_model_score', 0)
        
        bars = ax2.bar(['Best Single Model', 'Super-Model (Stacked)'], 
                      [best_single, super_score],
                      color=['#3498db', '#e74c3c'])
        ax2.set_ylabel('Score')
        ax2.set_title('Super-Model vs Best Single', fontsize=12, fontweight='bold')
        ax2.set_ylim(min(best_single, super_score) * 0.95, max(best_single, super_score) * 1.05)
        
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement annotation
        improvement = super_m.get('improvement_pct', 0)
        if improvement > 0:
            ax2.annotate(f'+{improvement:.2f}%', xy=(1, super_score),
                        xytext=(1.3, super_score), fontsize=12, color='green',
                        fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Super-Model not created', ha='center', va='center')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ==================== STEP 5: EXPLAINABILITY ====================

def step5_explain_model():
    """Step 5: SHAP explainability analysis."""
    if state.model is None:
        return "Please complete Step 4 first", None, None, create_step_indicator(3), None
    
    try:
        # Create explainer
        state.explainer = BlackBoxBreaker(state.model, state.X_train)
        
        # Compute SHAP values
        shap_vals, global_report = state.explainer.compute_global_shap()
        
        # Generate plots
        summary_b64, _ = state.explainer.generate_summary_plot()
        bar_b64, _ = state.explainer.generate_bar_plot()
        
        # Format report
        report = format_xai_report(global_report)
        
        # Convert base64 to image
        summary_img = None
        bar_img = None
        
        if summary_b64:
            summary_img = io.BytesIO(base64.b64decode(summary_b64))
        if bar_b64:
            bar_img = io.BytesIO(base64.b64decode(bar_b64))
        
        state.current_step = 5
        
        return report, summary_img, bar_img, create_step_indicator(5), gr.update(interactive=True)
        
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", None, None, create_step_indicator(4), None


# ==================== STEP 6: DEPLOYMENT ====================

def step6_deploy_model():
    """Step 6: Deployment setup."""
    if state.model is None:
        return "Please complete Step 4 first", None, create_step_indicator(4), None, None
    
    try:
        # Create deployment guard
        state.guard = DeploymentGuard()
        
        # Get accuracy
        super_report = state.training_report.get('super_model', {})
        accuracy = super_report.get('super_model_score', 0)
        
        # Save model
        save_result = state.guard.save_model(
            model=state.model,
            model_name="MetaAI_SuperModel",
            accuracy=accuracy,
            training_data=state.X_train,
            extra_metadata={
                "target_column": state.target_column,
                "task_type": state.task_type,
                "features": list(state.X_train.columns)
            }
        )
        
        # Generate FastAPI
        api_result = state.guard.generate_fastapi_app(
            model_file=save_result.get('model_file', ''),
            feature_columns=list(state.X_train.columns)
        )
        
        # Create deployment summary
        report_lines = [
            "=" * 50,
            "DEPLOYMENT COMPLETE",
            "=" * 50,
            "",
            "MODEL SAVED:",
            f"  File: {save_result.get('model_file', 'N/A')}",
            f"  Size: {save_result.get('model_size_kb', 0)} KB",
            f"  Accuracy: {save_result.get('accuracy', 0):.4f}",
            f"  Fingerprint: {save_result.get('fingerprint', 'N/A')}",
            "",
            "API GENERATED:",
            f"  File: {api_result.get('output_path', 'N/A')}",
            f"  Features: {api_result.get('features_count', 0)}",
            "",
            "ENDPOINTS:",
        ]
        
        for ep in api_result.get('endpoints', []):
            report_lines.append(f"  {ep['method']} {ep['path']} - {ep['description']}")
        
        report_lines.extend([
            "",
            "RUN API:",
            f"  {api_result.get('run_command', 'uvicorn app:app --reload')}",
            "",
            "=" * 50
        ])
        
        report = "\n".join(report_lines)
        
        # Create health monitor plot
        health_plot = create_health_plot()
        
        state.current_step = 6
        
        # Create downloadable files info
        files_info = {
            "model_file": save_result.get('model_file'),
            "api_file": api_result.get('output_path'),
            "metadata_file": save_result.get('metadata_file')
        }
        
        return report, health_plot, create_step_indicator(6), json.dumps(files_info, indent=2), gr.update(visible=True)
        
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", None, create_step_indicator(5), None, None


def create_health_plot() -> str:
    """Create deployment health visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Model Health
    ax1 = axes[0]
    accuracy = state.training_report.get('super_model', {}).get('super_model_score', 0)
    health_score = min(accuracy * 100, 100)
    color = '#2ecc71' if health_score > 80 else '#f39c12' if health_score > 60 else '#e74c3c'
    ax1.pie([health_score, 100-health_score], colors=[color, '#ecf0f1'],
            startangle=90, wedgeprops=dict(width=0.3))
    ax1.text(0, 0, f'{health_score:.0f}%', ha='center', va='center', fontsize=20, fontweight='bold')
    ax1.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    
    # Data Coverage
    ax2 = axes[1]
    n_features = len(state.X_train.columns)
    n_samples = len(state.X_train)
    ax2.bar(['Features', 'Samples (K)'], [n_features, n_samples/1000], color=['#3498db', '#9b59b6'])
    ax2.set_title('Data Coverage', fontsize=12, fontweight='bold')
    
    # Pipeline Status
    ax3 = axes[2]
    stages = ['Ingest', 'Clean', 'Features', 'Train', 'XAI', 'Deploy']
    status = [1 if state.current_step >= i+1 else 0 for i in range(6)]
    colors = ['#2ecc71' if s else '#ecf0f1' for s in status]
    ax3.barh(stages, [1]*6, color=colors)
    ax3.set_xlim(0, 1)
    ax3.set_title('Pipeline Status', fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def check_drift(new_file):
    """Check for data drift on new data."""
    if state.guard is None or new_file is None:
        return "Please complete Step 6 first and upload a file"
    
    try:
        new_df = pd.read_csv(new_file.name)
        drift_report = state.guard.detect_drift(new_df)
        return format_drift_report(drift_report)
    except Exception as e:
        return f"Error: {str(e)}"


# ==================== BUILD UI ====================

def build_wizard_app():
    """Build the step-by-step wizard UI."""
    
    with gr.Blocks(
        title="MetaAI Pro - AutoML Wizard",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
        css="""
        .step-header { font-size: 1.2em; font-weight: bold; padding: 10px; background: #f0f7ff; border-radius: 8px; margin-bottom: 15px; }
        .complete { background: #d4edda !important; }
        .pending { background: #fff3cd !important; }
        """
    ) as app:
        
        # Header
        gr.Markdown("""
        # MetaAI Pro - Enterprise AutoML Platform
        ### Step-by-Step ML Pipeline Wizard
        Complete each step in order to build your production-ready model.
        """)
        
        # Progress indicator
        progress_display = gr.Textbox(
            value=create_step_indicator(0),
            label="Pipeline Progress",
            lines=10,
            interactive=False
        )
        
        # ==================== STEP 1 ====================
        with gr.Accordion("Step 1: Data Quality Analysis", open=True):
            gr.Markdown("**Upload your data and analyze quality metrics**")
            
            with gr.Row():
                file_input = gr.File(label="Upload CSV", file_types=[".csv"])
                target_input = gr.Textbox(label="Target Column Name", placeholder="e.g., target, label, class")
            
            step1_btn = gr.Button("Analyze Data Quality", variant="primary", size="lg")
            
            with gr.Row():
                step1_report = gr.Textbox(label="Quality Report", lines=15)
                step1_plot = gr.Image(label="Quality Visualization")
        
        # ==================== STEP 2 ====================
        with gr.Accordion("Step 2: Data Cleaning & Reconstruction", open=False):
            gr.Markdown("**Bayesian imputation and anomaly detection**")
            
            step2_btn = gr.Button("Run Forensic Cleaning", variant="primary", size="lg", interactive=False)
            
            with gr.Row():
                step2_report = gr.Textbox(label="Cleaning Report", lines=15)
                step2_plot = gr.Image(label="Before vs After Comparison")
        
        # ==================== STEP 3 ====================
        with gr.Accordion("Step 3: Autonomous Feature Engineering", open=False):
            gr.Markdown("**Auto-discover interaction features and apply MI filtering**")
            
            step3_btn = gr.Button("Engineer Features", variant="primary", size="lg", interactive=False)
            
            with gr.Row():
                step3_report = gr.Textbox(label="Feature Report", lines=15)
                step3_plot = gr.Image(label="Feature Engineering Results")
        
        # ==================== STEP 4 ====================
        with gr.Accordion("Step 4: Elite Training Tournament", open=False):
            gr.Markdown("**Optuna-powered hyperparameter optimization with model stacking**")
            
            with gr.Row():
                n_trials_input = gr.Slider(10, 50, value=30, step=5, label="Trials per Model")
                step4_btn = gr.Button("Run Tournament", variant="primary", size="lg", interactive=False)
            
            with gr.Row():
                step4_report = gr.Textbox(label="Tournament Results", lines=15)
                step4_plot = gr.Image(label="Training Visualization")
        
        # ==================== STEP 5 ====================
        with gr.Accordion("Step 5: Explainable AI Analysis", open=False):
            gr.Markdown("**SHAP-based model explainability**")
            
            step5_btn = gr.Button("Generate Explanations", variant="primary", size="lg", interactive=False)
            
            step5_report = gr.Textbox(label="XAI Report", lines=10)
            
            with gr.Row():
                step5_summary = gr.Image(label="SHAP Summary Plot")
                step5_bar = gr.Image(label="Feature Importance")
        
        # ==================== STEP 6 ====================
        with gr.Accordion("Step 6: Deployment & Monitoring", open=False):
            gr.Markdown("**Save model, generate API, and monitor drift**")
            
            step6_btn = gr.Button("Deploy Model", variant="primary", size="lg", interactive=False)
            
            with gr.Row():
                step6_report = gr.Textbox(label="Deployment Report", lines=15)
                step6_plot = gr.Image(label="Health Monitor")
            
            step6_files = gr.Textbox(label="Generated Files", lines=5, visible=False)
            
            gr.Markdown("---")
            gr.Markdown("### Live Drift Monitor")
            
            with gr.Row():
                drift_file = gr.File(label="Upload New Data for Drift Check", file_types=[".csv"])
                drift_btn = gr.Button("Check Drift", variant="secondary")
            
            drift_result = gr.Textbox(label="Drift Analysis", lines=10)
        
        # ==================== EVENT HANDLERS ====================
        
        step1_btn.click(
            fn=step1_load_data,
            inputs=[file_input, target_input],
            outputs=[step1_report, step1_plot, progress_display, step2_btn]
        )
        
        step2_btn.click(
            fn=step2_clean_data,
            inputs=[],
            outputs=[step2_report, step2_plot, progress_display, step3_btn]
        )
        
        step3_btn.click(
            fn=step3_engineer_features,
            inputs=[],
            outputs=[step3_report, step3_plot, progress_display, step4_btn]
        )
        
        step4_btn.click(
            fn=step4_train_models,
            inputs=[n_trials_input],
            outputs=[step4_report, step4_plot, progress_display, step5_btn]
        )
        
        step5_btn.click(
            fn=step5_explain_model,
            inputs=[],
            outputs=[step5_report, step5_summary, step5_bar, progress_display, step6_btn]
        )
        
        step6_btn.click(
            fn=step6_deploy_model,
            inputs=[],
            outputs=[step6_report, step6_plot, progress_display, step6_files, step6_files]
        )
        
        drift_btn.click(
            fn=check_drift,
            inputs=[drift_file],
            outputs=[drift_result]
        )
    
    return app


if __name__ == "__main__":
    app = build_wizard_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
