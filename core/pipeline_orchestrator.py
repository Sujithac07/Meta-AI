"""
Pipeline Orchestrator - Unified ML Pipeline Manager
Manages flow between all modules with state dictionary and conditional execution
"""

import pandas as pd
from typing import Dict, List, Any, Callable
from datetime import datetime
from enum import Enum
import traceback

# Import all modules
from core.smart_ingestion import SmartIngestionEngine
from core.forensic_cleaner import ForensicCleaner
from core.auto_feature_engineer import AutoFeatureEngineer
from core.elite_trainer import EliteTrainer
from core.black_box_breaker import BlackBoxBreaker
from core.deployment_guard import DeploymentGuard


class PipelineStage(Enum):
    """Pipeline stages in execution order."""
    INIT = "init"
    INGESTION = "smart_ingestion"
    CLEANING = "forensic_cleaning"
    REVIEW_REQUIRED = "manual_review"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "elite_training"
    EXPLANATION = "xai_analysis"
    DEPLOYMENT = "deployment_guard"
    COMPLETE = "complete"
    ERROR = "error"


class PipelineOrchestrator:
    """
    Unified orchestrator managing flow between all ML pipeline modules.
    Uses state dictionary for data passing with conditional execution.
    """
    
    def __init__(self, anomaly_threshold: float = 0.2):
        """
        Initialize orchestrator.
        
        Args:
            anomaly_threshold: If anomaly ratio exceeds this, pause for review
        """
        self.anomaly_threshold = anomaly_threshold
        self.state = self._init_state()
        self.callbacks = {}
        
    def _init_state(self) -> Dict[str, Any]:
        """Initialize empty state dictionary."""
        return {
            # Pipeline status
            "stage": PipelineStage.INIT.value,
            "started_at": None,
            "completed_at": None,
            "error": None,
            
            # Data containers
            "raw_data": None,
            "target_column": None,
            "task_type": "classification",
            
            # Stage outputs
            "ingestion": {
                "data": None,
                "report": None
            },
            "cleaning": {
                "data": None,
                "report": None,
                "requires_review": False,
                "review_approved": False
            },
            "features": {
                "data": None,
                "report": None
            },
            "training": {
                "model": None,
                "report": None,
                "super_model": None
            },
            "xai": {
                "explainer": None,
                "global_report": None,
                "plots": {}
            },
            "deployment": {
                "guard": None,
                "drift_report": None,
                "api_generated": False,
                "model_saved": False
            },
            
            # Execution log
            "log": []
        }
    
    def reset(self):
        """Reset orchestrator state."""
        self.state = self._init_state()
        self._log("Pipeline reset")
    
    def _log(self, message: str, level: str = "INFO"):
        """Add entry to execution log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.state["log"].append(entry)
        print(f"[{level}] {message}")
    
    def set_callback(self, event: str, callback: Callable):
        """Set callback for pipeline events."""
        self.callbacks[event] = callback
    
    def _trigger_callback(self, event: str, data: Any = None):
        """Trigger registered callback."""
        if event in self.callbacks:
            try:
                self.callbacks[event](data)
            except Exception as e:
                self._log(f"Callback error for {event}: {e}", "WARNING")
    
    # ==================== MAIN ORCHESTRATION ====================
    
    def run_full_pipeline(self, df: pd.DataFrame, 
                         target_column: str,
                         task_type: str = "classification",
                         auto_approve_review: bool = False) -> Dict[str, Any]:
        """
        Run complete ML pipeline from data to deployment.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            task_type: 'classification' or 'regression'
            auto_approve_review: Skip manual review pause if True
        
        Returns:
            Final state dictionary
        """
        self.reset()
        self.state["started_at"] = datetime.now().isoformat()
        self.state["raw_data"] = df
        self.state["target_column"] = target_column
        self.state["task_type"] = task_type
        
        self._log(f"Starting pipeline: {len(df)} rows, target='{target_column}', task={task_type}")
        
        try:
            # Stage 1: Smart Ingestion
            self._run_ingestion()
            
            # Stage 2: Forensic Cleaning
            self._run_cleaning()
            
            # Check if review required
            if self.state["cleaning"]["requires_review"] and not auto_approve_review:
                self.state["stage"] = PipelineStage.REVIEW_REQUIRED.value
                self._log("Pipeline PAUSED - Manual review required due to high anomaly rate", "WARNING")
                self._trigger_callback("review_required", self.state["cleaning"]["report"])
                return self.state
            
            # Stage 3: Feature Engineering
            self._run_feature_engineering()
            
            # Stage 4: Elite Training
            self._run_training()
            
            # Stage 5: XAI Analysis
            self._run_xai()
            
            # Stage 6: Deployment Guard
            self._run_deployment()
            
            # Complete
            self.state["stage"] = PipelineStage.COMPLETE.value
            self.state["completed_at"] = datetime.now().isoformat()
            self._log("Pipeline completed successfully!")
            self._trigger_callback("complete", self.state)
            
        except Exception as e:
            self.state["stage"] = PipelineStage.ERROR.value
            self.state["error"] = str(e)
            self._log(f"Pipeline error: {e}", "ERROR")
            self._log(traceback.format_exc(), "ERROR")
            self._trigger_callback("error", e)
        
        return self.state
    
    def resume_after_review(self, approved: bool = True) -> Dict[str, Any]:
        """
        Resume pipeline after manual review.
        
        Args:
            approved: Whether to proceed with current data
        
        Returns:
            Updated state dictionary
        """
        if self.state["stage"] != PipelineStage.REVIEW_REQUIRED.value:
            self._log("Cannot resume - not in review state", "WARNING")
            return self.state
        
        self.state["cleaning"]["review_approved"] = approved
        
        if not approved:
            self._log("Review rejected - pipeline stopped", "WARNING")
            self.state["stage"] = PipelineStage.ERROR.value
            self.state["error"] = "Manual review rejected"
            return self.state
        
        self._log("Review approved - resuming pipeline")
        
        try:
            # Continue with remaining stages
            self._run_feature_engineering()
            self._run_training()
            self._run_xai()
            self._run_deployment()
            
            self.state["stage"] = PipelineStage.COMPLETE.value
            self.state["completed_at"] = datetime.now().isoformat()
            self._log("Pipeline completed successfully!")
            
        except Exception as e:
            self.state["stage"] = PipelineStage.ERROR.value
            self.state["error"] = str(e)
            self._log(f"Pipeline error: {e}", "ERROR")
        
        return self.state
    
    # ==================== STAGE RUNNERS ====================
    
    def _run_ingestion(self):
        """Run smart ingestion stage."""
        self.state["stage"] = PipelineStage.INGESTION.value
        self._log("Stage 1/6: Smart Ingestion")
        
        df = self.state["raw_data"]
        
        engine = SmartIngestionEngine()
        report = engine.smart_ingest(df)
        
        self.state["ingestion"]["data"] = df
        self.state["ingestion"]["report"] = report
        
        self._log(f"  Domain: {report.get('detected_domain', {}).get('primary_domain', 'UNKNOWN')}")
        self._log(f"  Quality Score: {report.get('quality_report', {}).get('overall_score', 0)}/100")
    
    def _run_cleaning(self):
        """Run forensic cleaning stage."""
        self.state["stage"] = PipelineStage.CLEANING.value
        self._log("Stage 2/6: Forensic Cleaning")
        
        df = self.state["ingestion"]["data"].copy()
        target_col = self.state["target_column"]
        
        # Exclude target from cleaning
        exclude_cols = [target_col] if target_col in df.columns else []
        
        cleaner = ForensicCleaner()
        df_cleaned, report = cleaner.full_reconstruction(df, exclude_cols)
        
        self.state["cleaning"]["data"] = df_cleaned
        self.state["cleaning"]["report"] = report
        
        # Check anomaly threshold
        anomaly_report = report.get("anomaly_detection", {})
        anomaly_pct = anomaly_report.get("anomaly_percentage", 0) / 100
        
        if anomaly_pct > self.anomaly_threshold:
            self.state["cleaning"]["requires_review"] = True
            self._log(f"  HIGH ANOMALY RATE: {anomaly_pct*100:.1f}% (threshold: {self.anomaly_threshold*100}%)", "WARNING")
        else:
            self._log(f"  Anomalies: {anomaly_pct*100:.1f}%")
        
        self._log(f"  Imputed: {report.get('imputation', {}).get('total_imputed', 0)} values")
        self._log(f"  Stability: {report.get('stability', {}).get('stability_score', 100)}/100")
    
    def _run_feature_engineering(self):
        """Run auto feature engineering stage."""
        self.state["stage"] = PipelineStage.FEATURE_ENGINEERING.value
        self._log("Stage 3/6: Auto Feature Engineering")
        
        df = self.state["cleaning"]["data"].copy()
        target_col = self.state["target_column"]
        task_type = self.state["task_type"]
        
        # Remove anomaly columns for feature engineering
        if "anomaly_label" in df.columns:
            df = df.drop(columns=["anomaly_label", "anomaly_score"], errors='ignore')
        
        engineer = AutoFeatureEngineer()
        df_engineered, report = engineer.auto_engineer(df, target_col, task_type)
        
        self.state["features"]["data"] = df_engineered
        self.state["features"]["report"] = report
        
        self._log(f"  Original features: {report.get('original_features', 0)}")
        self._log(f"  Final features: {report.get('final_features', 0)}")
        self._log(f"  New features created: {report.get('new_features_created', 0)}")
    
    def _run_training(self):
        """Run elite trainer tournament."""
        self.state["stage"] = PipelineStage.TRAINING.value
        self._log("Stage 4/6: Elite Training Tournament")
        
        df = self.state["features"]["data"].copy()
        target_col = self.state["target_column"]
        task_type = self.state["task_type"]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        trainer = EliteTrainer(n_trials=30)
        super_model, report = trainer.run_tournament(X, y, task_type)
        
        self.state["training"]["model"] = super_model
        self.state["training"]["super_model"] = super_model
        self.state["training"]["report"] = report
        self.state["training"]["best_models"] = trainer.best_models
        self.state["training"]["X"] = X
        self.state["training"]["y"] = y
        
        # Log rankings
        rankings = report.get("rankings", [])
        if rankings:
            self._log(f"  Winner: {rankings[0]['model']} ({rankings[0]['score']:.4f})")
        
        super_report = report.get("super_model", {})
        if super_report.get("status") == "success":
            self._log(f"  Super-Model: {super_report.get('super_model_score', 0):.4f}")
    
    def _run_xai(self):
        """Run XAI analysis."""
        self.state["stage"] = PipelineStage.EXPLANATION.value
        self._log("Stage 5/6: Explainable AI Analysis")
        
        model = self.state["training"]["super_model"]
        X = self.state["training"]["X"]
        
        if model is None:
            self._log("  Skipped - no model available", "WARNING")
            return
        
        explainer = BlackBoxBreaker(model, X)
        shap_values, report = explainer.compute_global_shap()
        
        self.state["xai"]["explainer"] = explainer
        self.state["xai"]["global_report"] = report
        
        if report.get("status") == "success":
            # Generate plots
            summary_plot, _ = explainer.generate_summary_plot()
            bar_plot, _ = explainer.generate_bar_plot()
            
            self.state["xai"]["plots"]["summary"] = summary_plot
            self.state["xai"]["plots"]["bar"] = bar_plot
            
            top_feats = report.get("top_10_features", [])
            if top_feats:
                self._log(f"  Top feature: {top_feats[0]['feature']} ({top_feats[0]['importance']:.4f})")
        else:
            self._log(f"  SHAP error: {report.get('error', 'unknown')}", "WARNING")
    
    def _run_deployment(self):
        """Run deployment guard setup."""
        self.state["stage"] = PipelineStage.DEPLOYMENT.value
        self._log("Stage 6/6: Deployment Guard Setup")
        
        model = self.state["training"]["super_model"]
        X = self.state["training"]["X"]
        report = self.state["training"]["report"]
        
        if model is None:
            self._log("  Skipped - no model available", "WARNING")
            return
        
        guard = DeploymentGuard()
        
        # Set reference data for drift detection
        guard.set_reference_data(X)
        
        # Get accuracy from super model
        super_report = report.get("super_model", {})
        accuracy = super_report.get("super_model_score", 0)
        
        # Save model
        save_result = guard.save_model(
            model=model,
            model_name="super_model",
            accuracy=accuracy,
            training_data=X,
            extra_metadata={
                "pipeline_version": "1.0",
                "task_type": self.state["task_type"],
                "target_column": self.state["target_column"]
            }
        )
        
        self.state["deployment"]["guard"] = guard
        self.state["deployment"]["model_saved"] = save_result.get("status") == "success"
        self.state["deployment"]["save_result"] = save_result
        
        # Generate FastAPI
        if save_result.get("status") == "success":
            api_result = guard.generate_fastapi_app(
                model_file=save_result["model_file"],
                feature_columns=list(X.columns)
            )
            self.state["deployment"]["api_generated"] = api_result.get("status") == "success"
            self.state["deployment"]["api_result"] = api_result
            
            self._log(f"  Model saved: {save_result['model_file']}")
            self._log(f"  API generated: {api_result.get('output_path', 'N/A')}")
    
    # ==================== UTILITY METHODS ====================
    
    def get_stage_data(self, stage: str) -> Dict:
        """Get data for a specific stage."""
        return self.state.get(stage, {})
    
    def get_current_stage(self) -> str:
        """Get current pipeline stage."""
        return self.state["stage"]
    
    def get_execution_log(self) -> List[Dict]:
        """Get execution log."""
        return self.state["log"]
    
    def get_final_model(self):
        """Get the final trained model."""
        return self.state["training"]["super_model"]
    
    def get_summary(self) -> Dict:
        """Get pipeline execution summary."""
        return {
            "stage": self.state["stage"],
            "started_at": self.state["started_at"],
            "completed_at": self.state["completed_at"],
            "error": self.state["error"],
            "ingestion_quality": self.state["ingestion"].get("report", {}).get("quality_report", {}).get("overall_score"),
            "anomaly_rate": self.state["cleaning"].get("report", {}).get("anomaly_detection", {}).get("anomaly_percentage"),
            "features_created": self.state["features"].get("report", {}).get("new_features_created", 0),
            "best_model_score": self.state["training"].get("report", {}).get("super_model", {}).get("super_model_score"),
            "model_saved": self.state["deployment"].get("model_saved", False),
            "api_generated": self.state["deployment"].get("api_generated", False)
        }


def format_pipeline_summary(state: Dict) -> str:
    """Format pipeline state as readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("PIPELINE ORCHESTRATOR SUMMARY")
    lines.append("=" * 60)
    
    lines.append(f"\nStage: {state.get('stage', 'unknown').upper()}")
    lines.append(f"Started: {state.get('started_at', 'N/A')}")
    lines.append(f"Completed: {state.get('completed_at', 'N/A')}")
    
    if state.get('error'):
        lines.append(f"\nERROR: {state['error']}")
    
    # Ingestion
    ing = state.get("ingestion", {}).get("report", {})
    if ing:
        lines.append("\n1. INGESTION")
        lines.append(f"   Quality: {ing.get('quality_report', {}).get('overall_score', 0)}/100")
        lines.append(f"   Domain: {ing.get('detected_domain', {}).get('primary_domain', 'UNKNOWN')}")
    
    # Cleaning
    clean = state.get("cleaning", {}).get("report", {})
    if clean:
        lines.append("\n2. CLEANING")
        lines.append(f"   Imputed: {clean.get('imputation', {}).get('total_imputed', 0)} values")
        lines.append(f"   Anomalies: {clean.get('anomaly_detection', {}).get('anomaly_percentage', 0)}%")
        if state.get("cleaning", {}).get("requires_review"):
            lines.append("   STATUS: REVIEW REQUIRED")
    
    # Features
    feat = state.get("features", {}).get("report", {})
    if feat:
        lines.append("\n3. FEATURES")
        lines.append(f"   Original: {feat.get('original_features', 0)}")
        lines.append(f"   Final: {feat.get('final_features', 0)}")
        lines.append(f"   New: {feat.get('new_features_created', 0)}")
    
    # Training
    train = state.get("training", {}).get("report", {})
    if train:
        lines.append("\n4. TRAINING")
        rankings = train.get("rankings", [])
        if rankings:
            lines.append(f"   Winner: {rankings[0]['model']} ({rankings[0]['score']:.4f})")
        super_m = train.get("super_model", {})
        if super_m.get("status") == "success":
            lines.append(f"   Super-Model: {super_m.get('super_model_score', 0):.4f}")
    
    # XAI
    xai = state.get("xai", {}).get("global_report", {})
    if xai and xai.get("status") == "success":
        lines.append("\n5. EXPLAINABILITY")
        top = xai.get("top_10_features", [])
        if top:
            lines.append(f"   Top Feature: {top[0]['feature']}")
    
    # Deployment
    deploy = state.get("deployment", {})
    if deploy:
        lines.append("\n6. DEPLOYMENT")
        lines.append(f"   Model Saved: {'Yes' if deploy.get('model_saved') else 'No'}")
        lines.append(f"   API Generated: {'Yes' if deploy.get('api_generated') else 'No'}")
    
    lines.append("\n" + "=" * 60)
    
    return '\n'.join(lines)
