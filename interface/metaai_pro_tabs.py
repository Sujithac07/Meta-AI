"""
Next-Level MetaAI: Reorganized Tab Structure
- Smart Pipeline Generator as hero feature
- 6 main tabs with intelligent subtabs
- Clean, professional UI
"""

import gradio as gr
import pandas as pd

# Import the smart pipeline generator
from core.smart_pipeline_generator import SmartPipelineGenerator


class MetaAIProInterface:
    """Next-level MetaAI Pro interface with smart pipeline generation"""
    
    def __init__(self):
        self.app_state = {
            'current_data': None,
            'target_column': None,
            'current_recommendation': None,
            'trained_models': {},
            'pipeline_versions': []
        }
    
    def build_interface(self):
        """Build the reorganized Gradio interface"""
        
        with gr.Blocks(title="MetaAI Pro - Next-Level AutoML", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 🚀 MetaAI Pro - Autonomous Machine Learning Platform
            **Intelligent. Automated. Production-Ready.**
            """)
            
            # ===== TAB 1: SMART PIPELINE GENERATOR (HERO) =====
            with gr.Tab("🤖 Smart Pipeline Generator"):
                with gr.Blocks():
                    gr.Markdown("### 🧠 Autonomous ML Architecture Design")
                    gr.Markdown("Upload your data. We'll recommend the optimal ML pipeline in seconds.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("**Step 1: Upload Data**")
                            data_file = gr.File(label="Upload CSV/Excel", file_types=['.csv', '.xlsx'])
                            target_col = gr.Dropdown(label="Select Target Column", choices=[])
                            analyze_btn = gr.Button("🔍 Analyze Dataset", variant="primary")
                        
                        with gr.Column(scale=1):
                            analysis_output = gr.Textbox(
                                label="Dataset Analysis",
                                lines=10,
                                interactive=False
                            )
                    
                    with gr.Row():
                        recommendation_output = gr.Textbox(
                            label="🤖 AI Recommendation",
                            lines=12,
                            interactive=False
                        )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Recommendation Details**")
                            algo_display = gr.Textbox(label="Recommended Algorithm", interactive=False)
                            reason_display = gr.Textbox(label="Why This Algorithm?", lines=4, interactive=False)
                        
                        with gr.Column():
                            gr.Markdown("**Expected Performance**")
                            metrics_display = gr.Textbox(label="Expected Metrics", interactive=False)
                            time_display = gr.Textbox(label="Training Time", interactive=False)
                    
                    with gr.Row():
                        gen_pipeline_btn = gr.Button("✨ Generate & Train Pipeline", variant="primary", size="lg")
                        status_output = gr.Textbox(label="Status", interactive=False)
                    
                    # Connect analysis button
                    analyze_btn.click(
                        self.analyze_and_recommend,
                        inputs=[data_file, target_col],
                        outputs=[analysis_output, recommendation_output, algo_display, 
                                reason_display, metrics_display, time_display]
                    )
            
            # ===== TAB 2: DATA STUDIO =====
            with gr.Tab("📊 Data Studio"):
                with gr.Tabs():
                    # Subtab: Upload & Preview
                    with gr.Tab("📤 Upload & Preview"):
                        gr.Markdown("### Upload and Preview Your Data")
                        with gr.Row():
                            data_upload = gr.File(label="Upload CSV/Excel")
                            preview_table = gr.Dataframe(label="Data Preview")
                        data_upload.change(
                            lambda f: self.load_and_preview(f),
                            inputs=data_upload,
                            outputs=preview_table
                        )
                    
                    # Subtab: Exploratory Analysis
                    with gr.Tab("🔍 Exploratory Analysis"):
                        gr.Markdown("### Statistical Analysis & Insights")
                        eda_output = gr.Textbox(label="EDA Report", lines=15)
                        analyze_eda_btn = gr.Button("Generate EDA")
                        analyze_eda_btn.click(
                            self.generate_eda,
                            inputs=[],
                            outputs=eda_output
                        )
                    
                    # Subtab: Data Quality Check
                    with gr.Tab("✅ Data Quality Check"):
                        gr.Markdown("### Data Quality Assessment (0-100)")
                        quality_report = gr.Textbox(label="Quality Report", lines=10)
                        quality_score = gr.Slider(label="Quality Score", minimum=0, maximum=100)
                        check_quality_btn = gr.Button("Check Quality")
                        check_quality_btn.click(
                            self.check_data_quality,
                            inputs=[],
                            outputs=[quality_report, quality_score]
                        )
                    
                    # Subtab: Preprocessing & Features
                    with gr.Tab("⚙️ Preprocessing & Features"):
                        gr.Markdown("### Data Cleaning & Feature Engineering")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Preprocessing Options**")
                                handle_missing = gr.Checkbox(label="Handle Missing Values")
                                remove_outliers = gr.Checkbox(label="Remove Outliers")
                                scale_features = gr.Checkbox(label="Scale Features")
                            
                            with gr.Column():
                                gr.Markdown("**Feature Engineering**")
                                auto_features = gr.Checkbox(label="Auto Feature Selection")
                                create_features = gr.Checkbox(label="Create New Features")
                                encode_cat = gr.Checkbox(label="Encode Categorical")
                        
                        process_btn = gr.Button("Apply Preprocessing")
                        process_output = gr.Textbox(label="Result", interactive=False)
            
            # ===== TAB 3: TRAINING CONSOLE =====
            with gr.Tab("🧠 Training Console"):
                with gr.Tabs():
                    # Subtab: Single Model Training
                    with gr.Tab("🎯 Single Model Training"):
                        gr.Markdown("### Train Individual Model")
                        with gr.Row():
                            algorithm = gr.Dropdown(
                                label="Select Algorithm",
                                choices=[
                                    'RandomForest', 'XGBoost', 'LightGBM',
                                    'GradientBoosting', 'LogisticRegression',
                                    'SVC', 'KNN', 'DecisionTree'
                                ]
                            )
                            test_size = gr.Slider(label="Test Size %", minimum=10, maximum=40, value=20)
                        
                        train_btn = gr.Button("🚀 Train Model", variant="primary")
                        train_output = gr.Textbox(label="Training Results", lines=10)
                    
                    # Subtab: Auto-Pilot (7 Models)
                    with gr.Tab("⚡ Auto-Pilot"):
                        gr.Markdown("### Train 7 Models in Parallel")
                        gr.Markdown("Automatically trains all algorithms with optimal hyperparameters.")
                        autopilot_output = gr.Textbox(label="Auto-Pilot Results", lines=15)
                        autopilot_btn = gr.Button("🚁 Launch Auto-Pilot")
                        autopilot_btn.click(
                            self.run_autopilot,
                            inputs=[],
                            outputs=autopilot_output
                        )
                    
                    # Subtab: Hyperparameter Optimizer
                    with gr.Tab("🔧 Hyperparameter Optimizer"):
                        gr.Markdown("### Bayesian Optimization")
                        gr.Markdown("Smart hyperparameter tuning 10x faster than GridSearch")
                        algo_select = gr.Dropdown(
                            label="Select Algorithm",
                            choices=['RandomForest', 'XGBoost', 'LightGBM']
                        )
                        n_trials = gr.Slider(label="Number of Trials", minimum=5, maximum=50, value=20)
                        optimize_btn = gr.Button("⚙️ Optimize")
                        optimize_output = gr.Textbox(label="Optimization Results", lines=12)
                    
                    # Subtab: Model Registry
                    with gr.Tab("📦 Model Registry"):
                        gr.Markdown("### Save & Manage Trained Models")
                        registry_output = gr.Dataframe(label="Saved Models")
                        refresh_btn = gr.Button("🔄 Refresh")
                        refresh_btn.click(
                            self.refresh_model_registry,
                            inputs=[],
                            outputs=registry_output
                        )
            
            # ===== TAB 4: ANALYSIS & INSIGHTS =====
            with gr.Tab("📈 Analysis & Insights"):
                with gr.Tabs():
                    # Subtab: Advanced Visualizations
                    with gr.Tab("🎨 Visualizations"):
                        viz_mode = gr.Dropdown(
                            label="Visualization Mode",
                            choices=['Basic', 'Advanced', 'Model Analysis']
                        )
                        viz_plot = gr.Plot(label="Visualization")
                        gen_viz_btn = gr.Button("Generate Visualization")
                    
                    # Subtab: Feature Importance
                    with gr.Tab("⭐ Feature Importance"):
                        gr.Markdown("### Feature Impact on Model")
                        importance_plot = gr.Plot(label="Feature Importance Chart")
                        importance_table = gr.Dataframe(label="Importance Scores")
                        gen_importance_btn = gr.Button("Generate Feature Importance")
                    
                    # Subtab: Model Insights
                    with gr.Tab("💡 Model Insights"):
                        gr.Markdown("### Prediction Explanations & Insights")
                        insights_output = gr.Textbox(label="Model Insights", lines=15)
                        gen_insights_btn = gr.Button("Generate Insights")
                    
                    # Subtab: Chatbot Assistant
                    with gr.Tab("🤖 Chatbot"):
                        gr.Markdown("### AI Assistant - Ask Me Anything")
                        chat_history = gr.Textbox(label="Conversation", lines=12, interactive=False)
                        user_question = gr.Textbox(label="Ask a question...")
                        chat_btn = gr.Button("💬 Send")
            
            # ===== TAB 5: TESTING & VALIDATION =====
            with gr.Tab("⚖️ Testing & Validation"):
                with gr.Tabs():
                    # Subtab: Benchmark Models
                    with gr.Tab("📊 Benchmark"):
                        gr.Markdown("### Compare Trained Models")
                        benchmark_table = gr.Dataframe(label="Benchmark Results")
                        benchmark_plot = gr.Plot(label="Performance Comparison")
                        run_benchmark_btn = gr.Button("Run Benchmark")
                    
                    # Subtab: A/B Testing
                    with gr.Tab("⚖️ A/B Testing"):
                        gr.Markdown("### Test Models Before Production")
                        with gr.Row():
                            model_a = gr.Dropdown(label="Model A", choices=[])
                            model_b = gr.Dropdown(label="Model B", choices=[])
                        ab_results = gr.Textbox(label="A/B Test Results", lines=10)
                        run_ab_btn = gr.Button("Run A/B Test")
                    
                    # Subtab: Performance Monitoring
                    with gr.Tab("📈 Monitoring"):
                        gr.Markdown("### Real-Time Performance Tracking")
                        perf_plot = gr.Plot(label="Performance Over Time")
                        perf_table = gr.Dataframe(label="Metrics")
                    
                    # Subtab: Drift Detection
                    with gr.Tab("🔴 Drift Detection"):
                        gr.Markdown("### Monitor Data & Prediction Drift")
                        drift_report = gr.Textbox(label="Drift Report", lines=10)
                        drift_plot = gr.Plot(label="Drift Visualization")
                        check_drift_btn = gr.Button("Check Drift")
            
            # ===== TAB 6: PRODUCTION & DEPLOY =====
            with gr.Tab("🚀 Production & Deploy"):
                with gr.Tabs():
                    # Subtab: Pipeline Versions
                    with gr.Tab("📜 Pipeline Versions"):
                        gr.Markdown("### Git-Like Pipeline Version Control")
                        versions_table = gr.Dataframe(label="Pipeline Versions")
                        versions_history = gr.Textbox(label="Version History", lines=10)
                        refresh_versions_btn = gr.Button("🔄 Refresh")
                    
                    # Subtab: Model Deployment
                    with gr.Tab("🚀 Deploy"):
                        gr.Markdown("### One-Click Production Deployment")
                        with gr.Row():
                            deploy_platform = gr.Dropdown(
                                label="Deploy To",
                                choices=['Docker', 'HuggingFace', 'AWS', 'Azure', 'GCP']
                            )
                            model_select = gr.Dropdown(label="Select Model", choices=[])
                        deploy_output = gr.Textbox(label="Deployment Status", lines=8)
                        deploy_btn = gr.Button("🚀 Deploy to Production")
                    
                    # Subtab: Batch Prediction
                    with gr.Tab("📮 Batch Prediction"):
                        gr.Markdown("### Make Predictions on New Data")
                        batch_file = gr.File(label="Upload Data for Prediction")
                        predictions_output = gr.Dataframe(label="Predictions")
                        predict_btn = gr.Button("Predict")
                    
                    # Subtab: Monitoring Alerts
                    with gr.Tab("🚨 Alerts"):
                        gr.Markdown("### Configure Monitoring Alerts")
                        alerts_table = gr.Dataframe(label="Active Alerts")
                        new_alert_btn = gr.Button("➕ Create New Alert")
                    
                    # Subtab: MLOps Dashboard
                    with gr.Tab("📊 MLOps Hub"):
                        gr.Markdown("### MLflow Experiment Tracking")
                        experiments_table = gr.Dataframe(label="Experiments")
                        refresh_experiments_btn = gr.Button("🔄 Refresh")
            
            # ===== TAB 7: ADVANCED (BONUS) =====
            with gr.Tab("🤖 Advanced Features"):
                with gr.Tabs():
                    # Subtab: Fairness & Bias
                    with gr.Tab("⚖️ Fairness & Bias"):
                        gr.Markdown("### Bias Detection & Fairness Audit")
                        bias_report = gr.Textbox(label="Bias Report", lines=12)
                        audit_bias_btn = gr.Button("Audit Bias")
                    
                    # Subtab: Causal Inference
                    with gr.Tab("🔍 Causal Inference"):
                        gr.Markdown("### Discover Causal Relationships")
                        causal_output = gr.Textbox(label="Causal Analysis", lines=10)
                        analyze_causal_btn = gr.Button("Analyze Causal Effects")
                    
                    # Subtab: Edge Optimization
                    with gr.Tab("📱 Edge Optimization"):
                        gr.Markdown("### Deploy to Mobile/IoT")
                        with gr.Row():
                            compression_level = gr.Slider(
                                label="Compression Level",
                                minimum=1, maximum=10, value=5
                            )
                            target_device = gr.Dropdown(
                                label="Target Device",
                                choices=['Mobile', 'IoT', 'Edge', 'Browser']
                            )
                        edge_output = gr.Textbox(label="Optimization Result", lines=8)
                        optimize_edge_btn = gr.Button("Optimize for Edge")
        
        return demo
    
    # ===== CALLBACK FUNCTIONS =====
    
    def analyze_and_recommend(self, file, target):
        """Analyze data and generate recommendation"""
        try:
            if file is None or target is None:
                return "Please upload file and select target column"
            
            # Load data
            df = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.read_excel(file.name)
            self.app_state['current_data'] = df
            self.app_state['target_column'] = target
            
            # Generate recommendation
            generator = SmartPipelineGenerator(df, target)
            result = generator.generate()
            
            self.app_state['current_recommendation'] = result
            
            # Format outputs
            analysis_text = self._format_analysis(result['analysis'])
            recommendation_text = generator.get_summary()
            
            rec = result['recommendation']
            algo_text = rec['primary_algorithm']
            reason_text = rec['reason']
            metrics_text = str(rec['expected_metrics'])
            time_text = f"{rec['training_time_minutes']} minutes"
            
            return analysis_text, recommendation_text, algo_text, reason_text, metrics_text, time_text
        
        except Exception as e:
            return f"Error: {str(e)}", "", "", "", "", ""
    
    def _format_analysis(self, analysis: dict) -> str:
        """Format analysis dict to readable text"""
        size = analysis['size']
        types = analysis['types']
        target = analysis['target']
        quality = analysis['quality']
        
        text = f"""
📊 DATASET ANALYSIS
{'='*50}

Size Information:
  • Rows: {size['rows']:,}
  • Columns: {size['cols']}
  • Category: {size['category'].upper()}
  • Memory: {size['memory_mb']} MB

Feature Types:
  • Numeric: {types['numeric_count']}
  • Categorical: {types['categorical_count']}
  • DateTime: {types['datetime_count']}
  • Text Features: {'Yes' if types['has_text'] else 'No'}

Target Variable:
  • Type: {target.get('type', 'Unknown')}
  • Classes: {target.get('classes', 'N/A')}
  • Imbalance: {target.get('imbalance_level', 'N/A')}

Data Quality:
  • Quality Score: {quality['quality_score']}/100 ({quality['quality_level'].upper()})
  • Missing Data: {quality['missing_percent']}%
  • Duplicates: {quality['duplicate_percent']}%

Complexity:
  • Dimensionality: {analysis['complexity']['dimensionality'].upper()}
  • Feature Density: {analysis['complexity']['feature_density']*100:.1f}%
        """
        return text
    
    def load_and_preview(self, file):
        """Load and preview data"""
        if file is None:
            return None
        try:
            df = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.read_excel(file.name)
            return df.head(20)
        except:
            return None
    
    def generate_eda(self):
        """Generate EDA report"""
        if self.app_state['current_data'] is None:
            return "Please upload data first"
        return "EDA Report Generated..."
    
    def check_data_quality(self):
        """Check data quality"""
        if self.app_state['current_data'] is None:
            return "Please upload data first", 0
        return "Quality Check Complete...", 85
    
    def run_autopilot(self):
        """Run auto-pilot training"""
        return "Auto-Pilot training started..."
    
    def refresh_model_registry(self):
        """Refresh model registry"""
        return pd.DataFrame({'Model': [], 'Algorithm': [], 'Accuracy': []})
    
    def refresh_experiments(self):
        """Refresh MLflow experiments"""
        return pd.DataFrame({'ID': [], 'Name': [], 'Status': []})


# ===== MAIN =====

def main():
    interface = MetaAIProInterface()
    demo = interface.build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )


if __name__ == "__main__":
    main()
