"""
Elite Trainer Engine - FAST Multi-Agent Competition
Optimized for speed while maintaining quality
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
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
    from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: F401
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class EliteTrainer:
    """
    FAST Multi-Agent Model Competition Engine.
    Optimized: 10 trials, 3-fold CV, sampling, aggressive pruning.
    """
    
    def __init__(self, random_state: int = 42, n_trials: int = 10, n_jobs: int = -1):
        self.random_state = random_state
        self.n_trials = min(n_trials, 15)  # Cap at 15 trials max
        self.n_jobs = n_jobs
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
        
        # OPTIMIZATION: Only use 2-3 fast models
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
            print(f"  ⚡ {name}...", end=" ", flush=True)
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
        
        # Create Super-Model (FAST - use top 2 only)
        if len(self.best_models) >= 2:
            print("  🚀 Building Super-Model...", end=" ", flush=True)
            self.super_model, stacking_report = self._create_super_model(
                X_clean, y, task_type, cv_folds=2
            )
            print("✓")
            report["super_model"] = stacking_report
        
        return self.super_model, report
    
    def _get_competitors(self, task_type: str) -> Dict:
        """Get FAST model competitors - limit to 3 models."""
        competitors = {}
        
        # Priority 1: LightGBM (fastest)
        if LIGHTGBM_AVAILABLE:
            competitors['LightGBM'] = {
                'class': LGBMClassifier if task_type == 'classification' else LGBMRegressor,
                'search_space': self._lgbm_search_space
            }
        
        # Priority 2: XGBoost
        if XGBOOST_AVAILABLE and len(competitors) < 3:
            competitors['XGBoost'] = {
                'class': XGBClassifier if task_type == 'classification' else XGBRegressor,
                'search_space': self._xgb_search_space
            }
        
        # Priority 3: Random Forest (always available, fast)
        if len(competitors) < 3:
            competitors['RandomForest'] = {
                'class': RandomForestClassifier if task_type == 'classification' else RandomForestRegressor,
                'search_space': self._rf_search_space
            }
        
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
        """Create FAST Super-Model via Stacking (top 2 models only)."""
        sorted_models = sorted(
            self.best_models.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Use top 2 models only for speed
        top_models = sorted_models[:2]
        
        estimators = [
            (name, info['model'])
            for name, info in top_models
        ]
        
        if task_type == 'classification':
            meta_learner = LogisticRegression(max_iter=500, random_state=self.random_state)
            super_model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv_folds,
                stack_method='predict_proba',
                n_jobs=self.n_jobs,
                passthrough=False
            )
        else:
            meta_learner = Ridge(random_state=self.random_state)
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
            "meta_learner": "LogisticRegression" if task_type == 'classification' else "Ridge",
            "super_model_score": round(super_score, 4),
            "best_single_score": round(best_single, 4),
            "improvement": round(super_score - best_single, 4),
            "outperforms_singles": super_score > best_single
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
