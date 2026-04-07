"""
Elite Trainer Engine - FAST Multi-Agent Competition
Optimized for speed while maintaining quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class EliteTrainer:
    """
    FAST Multi-Agent Model Competition Engine.
    Optimized: 10 trials, 3-fold CV, sampling, aggressive pruning.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_trials: int = 10,
        n_jobs: int = -1,
        max_competitors: int = 8,
    ):
        self.random_state = random_state
        self.n_trials = min(n_trials, 15)  # Cap at 15 trials max
        self.n_jobs = n_jobs
        self.max_competitors = max(3, min(max_competitors, 10))
        self.best_models = {}
        self.tournament_results = {}
        self.super_model = None
        self.study_results = {}
        
    def run_tournament(self, X: pd.DataFrame, y: pd.Series,
                      task_type: str = 'classification',
                      cv_folds: int = 3) -> Tuple[Any, Dict]:
        """Run FAST tournament competition."""
        if not OPTUNA_AVAILABLE:
            return None, {"error": "Optuna not installed. Run: pip install optuna"}
        
        X_clean = X.fillna(0)
        
        # OPTIMIZATION: Sample if dataset is large
        if len(X_clean) > 5000:
            sample_idx = np.random.choice(len(X_clean), 5000, replace=False)
            X_sample = X_clean.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X_clean
            y_sample = y
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Train several algorithms (default up to 8), then pick champion by CV score
        competitors = self._get_competitors(task_type)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "n_trials_per_model": self.n_trials,
            "cv_folds": cv_folds,
            "data_sampled": len(X_sample) < len(X_clean),
            "competitors": list(competitors.keys()),
            "tournament_results": {},
            "best_models": {},
            "super_model": {},
            "rankings": []
        }
        
        # Run tournament for each competitor
        for name, config in competitors.items():
            print(f"  [MODEL] {name}...", end=" ", flush=True)
            study_result = self._run_optuna_study(
                name, config, X_sample, y_sample, task_type, cv_folds
            )
            print(f"✓ ({study_result['best_score']:.3f})")
            
            self.study_results[name] = study_result
            report["tournament_results"][name] = {
                "best_score": round(study_result['best_score'], 4),
                "best_params": study_result['best_params'],
                "trials_completed": study_result['trials_completed'],
                "trials_pruned": study_result['trials_pruned'],
                "optimization_time": round(study_result['time_seconds'], 2)
            }
            
            if study_result['best_model'] is not None:
                self.best_models[name] = {
                    'model': study_result['best_model'],
                    'score': study_result['best_score'],
                    'params': study_result['best_params']
                }
        
        # Rank competitors
        rankings = sorted(
            self.best_models.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        report["rankings"] = [
            {"rank": i+1, "model": name, "score": round(info['score'], 4)}
            for i, (name, info) in enumerate(rankings)
        ]
        
        # Champion = best single model by CV (what we deploy & explain with SHAP)
        champion_name = rankings[0][0] if rankings else None
        champion_model = self.best_models[champion_name]["model"] if champion_name else None
        report["champion"] = {
            "model": champion_name,
            "score": round(rankings[0][1]["score"], 4) if rankings else None,
            "note": "Primary model: best CV score among all trained algorithms (not the stacking meta-learner).",
        }
        
        # Optional stacking blend (for metrics only) — top-K bases + non-linear meta
        if len(self.best_models) >= 2:
            print("  [STACK] Evaluating stacking blend (optional)...", end=" ", flush=True)
            self.super_model, stacking_report = self._create_super_model(
                X_clean, y, task_type, cv_folds=2
            )
            print("✓")
            report["super_model"] = stacking_report
        else:
            report["super_model"] = {"status": "skipped", "reason": "need at least 2 trained models"}
        
        if not rankings:
            report["error"] = "No models finished training successfully."
            return None, report
        
        # Return the *single best* estimator so SHAP/UI match tournament winner
        return champion_model, report
    
    def _get_competitors(self, task_type: str) -> Dict:
        """Up to ``max_competitors`` algorithms (trees, boosting, linear, etc.)."""
        competitors = {}
        cap = self.max_competitors
        clf = task_type == "classification"

        def add(name: str, cls, space_fn):
            if len(competitors) >= cap:
                return
            competitors[name] = {"class": cls, "search_space": space_fn}

        if clf:
            if LIGHTGBM_AVAILABLE:
                add("LightGBM", LGBMClassifier, self._lgbm_search_space)
            if XGBOOST_AVAILABLE:
                add("XGBoost", XGBClassifier, self._xgb_search_space)
            add("RandomForest", RandomForestClassifier, self._rf_search_space)
            add("HistGradientBoosting", HistGradientBoostingClassifier, self._hgb_search_space)
            add("GradientBoosting", GradientBoostingClassifier, self._gb_search_space)
            add("ExtraTrees", ExtraTreesClassifier, self._et_search_space)
            add("LogisticRegression", LogisticRegression, self._lr_search_space)
            if CATBOOST_AVAILABLE:
                add("CatBoost", CatBoostClassifier, self._catboost_search_space)
        else:
            if LIGHTGBM_AVAILABLE:
                add("LightGBM", LGBMRegressor, self._lgbm_search_space)
            if XGBOOST_AVAILABLE:
                add("XGBoost", XGBRegressor, self._xgb_search_space)
            add("RandomForest", RandomForestRegressor, self._rf_search_space)
            add("HistGradientBoosting", HistGradientBoostingRegressor, self._hgb_search_space_reg)
            add("GradientBoosting", GradientBoostingRegressor, self._gb_search_space_reg)
            add("ExtraTrees", ExtraTreesRegressor, self._et_search_space_reg)
            add("Ridge", Ridge, self._ridge_search_space)
            if CATBOOST_AVAILABLE:
                add("CatBoost", CatBoostRegressor, self._catboost_search_space)

        # sklearn fallbacks if deps missing
        if len(competitors) < 3 and clf:
            add("RandomForest", RandomForestClassifier, self._rf_search_space)
            add("LogisticRegression", LogisticRegression, self._lr_search_space)
        elif len(competitors) < 3:
            add("RandomForest", RandomForestRegressor, self._rf_search_space)
            add("Ridge", Ridge, self._ridge_search_space)

        return competitors
    
    def _run_optuna_study(self, name: str, config: Dict,
                         X: pd.DataFrame, y: pd.Series,
                         task_type: str, cv_folds: int) -> Dict:
        """Run FAST Optuna optimization."""
        import time
        start_time = time.time()
        
        # Aggressive pruning
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=1)
        sampler = TPESampler(seed=self.random_state)
        
        study = optuna.create_study(
            direction='maximize',
            pruner=pruner,
            sampler=sampler
        )
        
        def objective(trial):
            params = config['search_space'](trial)
            
            # Add common params
            if 'random_state' not in params:
                params['random_state'] = self.random_state
            if name in ['XGBoost', 'LightGBM']:
                params['n_jobs'] = self.n_jobs
                params['verbosity'] = 0
            if name == 'CatBoost':
                params['verbose'] = False
            if name == 'RandomForest':
                params['n_jobs'] = self.n_jobs
            if name in ('HistGradientBoosting', 'ExtraTrees', 'GradientBoosting'):
                params['random_state'] = self.random_state
            if name == 'ExtraTrees':
                params['n_jobs'] = self.n_jobs
            if name == 'LogisticRegression':
                params['solver'] = params.get('solver', 'lbfgs')
                params['max_iter'] = params.get('max_iter', 2000)
            
            model = config['class'](**params)
            
            # FAST: 3-fold CV
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=1)
            
            return np.mean(scores)
        
        # Run with timeout
        study.optimize(
            objective, 
            n_trials=self.n_trials, 
            show_progress_bar=False,
            timeout=60  # 60 second timeout per model
        )
        
        elapsed = time.time() - start_time
        
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        # Train best model on full data
        best_params = study.best_params.copy()
        if 'random_state' not in best_params:
            best_params['random_state'] = self.random_state
        if name in ['XGBoost', 'LightGBM']:
            best_params['n_jobs'] = self.n_jobs
            best_params['verbosity'] = 0
        if name == 'CatBoost':
            best_params['verbose'] = False
        if name == 'RandomForest':
            best_params['n_jobs'] = self.n_jobs
        if name in ('HistGradientBoosting', 'ExtraTrees', 'GradientBoosting'):
            best_params['random_state'] = self.random_state
        if name == 'ExtraTrees':
            best_params['n_jobs'] = self.n_jobs
        if name == 'LogisticRegression':
            best_params['solver'] = best_params.get('solver', 'lbfgs')
            best_params['max_iter'] = best_params.get('max_iter', 2000)
        
        best_model = config['class'](**best_params)
        best_model.fit(X, y)
        
        return {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'best_model': best_model,
            'trials_completed': completed,
            'trials_pruned': pruned,
            'time_seconds': elapsed
        }
    
    def _create_super_model(self, X: pd.DataFrame, y: pd.Series,
                           task_type: str, cv_folds: int = 2) -> Tuple[Any, Dict]:
        """Stacking over top-3 single models; shallow GB meta-learner (not plain LR)."""
        sorted_models = sorted(
            self.best_models.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        top_models = sorted_models[: min(3, len(sorted_models))]
        
        estimators = [
            (name, info['model'])
            for name, info in top_models
        ]
        
        if task_type == 'classification':
            meta_learner = GradientBoostingClassifier(
                n_estimators=40,
                max_depth=2,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            super_model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv_folds,
                stack_method='predict_proba',
                n_jobs=self.n_jobs,
                passthrough=False
            )
        else:
            meta_learner = GradientBoostingRegressor(
                n_estimators=40,
                max_depth=2,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            super_model = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv_folds,
                n_jobs=self.n_jobs,
                passthrough=False
            )
        
        super_model.fit(X, y)
        
        # Quick evaluation
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(super_model, X, y, cv=cv, scoring='accuracy')
        else:
            cv = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(super_model, X, y, cv=cv, scoring='r2')
        
        super_score = scores.mean()
        best_single = max(info['score'] for info in self.best_models.values())
        
        return super_model, {
            "status": "success",
            "base_models": [name for name, _ in top_models],
            "meta_learner": "GradientBoosting (shallow)",
            "super_model_score": round(super_score, 4),
            "best_single_score": round(best_single, 4),
            "improvement": round(super_score - best_single, 4),
            "outperforms_singles": super_score > best_single,
            "note": "For reference only; the deployed model is the single best from rankings.",
        }
    
    # OPTIMIZED search spaces - narrower ranges, fewer params
    def _xgb_search_space(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        }
    
    def _lgbm_search_space(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        }
    
    def _catboost_search_space(self, trial):
        return {
            'iterations': trial.suggest_int('iterations', 50, 150),
            'depth': trial.suggest_int('depth', 4, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
        }
    
    def _rf_search_space(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        }
    
    def _lr_search_space(self, trial):
        return {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'solver': 'lbfgs',
            'max_iter': 2000,
        }
    
    def _ridge_search_space(self, trial):
        return {'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)}
    
    def _hgb_search_space(self, trial):
        return {
            'max_iter': trial.suggest_int('max_iter', 80, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 40),
        }
    
    def _hgb_search_space_reg(self, trial):
        return self._hgb_search_space(trial)
    
    def _gb_search_space(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.25),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        }
    
    def _gb_search_space_reg(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.25),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'loss': 'squared_error',
        }
    
    def _et_search_space(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 12),
        }
    
    def _et_search_space_reg(self, trial):
        return self._et_search_space(trial)


def format_tournament_report(report: Dict) -> str:
    """Format tournament report as Markdown string."""
    if not report or "error" in report:
        return f"## ⚠️ Tournament Error\n\n{report.get('error', 'Unknown error')}"
    
    lines = [
        "## 🏆 Model Tournament Results",
        "",
        f"**Task Type:** {report.get('task_type', 'N/A')}",
        f"**Trials per Model:** {report.get('n_trials_per_model', 'N/A')}",
        f"**CV Folds:** {report.get('cv_folds', 'N/A')}",
        "",
        "### 📊 Rankings",
        ""
    ]
    
    # Add rankings
    for rank_info in report.get('rankings', []):
        medal = "🥇" if rank_info['rank'] == 1 else "🥈" if rank_info['rank'] == 2 else "🥉" if rank_info['rank'] == 3 else "  "
        lines.append(f"{medal} **#{rank_info['rank']} {rank_info['model']}** - Score: {rank_info['score']:.4f}")
    
    champ = report.get("champion") or {}
    if champ.get("model"):
        lines.append("")
        lines.append("### 🎯 Champion (used for export & SHAP)")
        lines.append("")
        lines.append(
            f"- **Model:** `{champ.get('model')}` — **CV score:** {champ.get('score')} "
            f"(best among all algorithms above)."
        )
        if champ.get("note"):
            lines.append(f"- {champ['note']}")
    
    lines.append("")
    lines.append("### ⚡ Training Details")
    lines.append("")
    
    # Add tournament results
    for model_name, result in report.get('tournament_results', {}).items():
        lines.append(f"**{model_name}:**")
        lines.append(f"  - Best Score: {result.get('best_score', 'N/A')}")
        lines.append(f"  - Trials: {result.get('trials_completed', 0)} completed, {result.get('trials_pruned', 0)} pruned")
        lines.append(f"  - Time: {result.get('optimization_time', 0):.1f}s")
        lines.append("")
    
    # Add super model info
    super_model = report.get('super_model', {})
    if super_model.get('status') == 'success':
        lines.append("### 🚀 Super-Model (Stacking Ensemble)")
        lines.append("")
        lines.append(f"- **Base Models:** {', '.join(super_model.get('base_models', []))}")
        lines.append(f"- **Meta Learner:** {super_model.get('meta_learner', 'N/A')}")
        lines.append(f"- **Super Model Score:** {super_model.get('super_model_score', 'N/A')}")
        lines.append(f"- **Best Single Score:** {super_model.get('best_single_score', 'N/A')}")
        improvement = super_model.get('improvement', 0)
        if improvement > 0:
            lines.append(f"- **Improvement:** +{improvement:.4f} ✅")
        else:
            lines.append(f"- **Improvement:** {improvement:.4f}")
    
    return "\n".join(lines)
