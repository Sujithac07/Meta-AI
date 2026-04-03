"""
Evolutionary AutoPilot - Genetic Optimization with Optuna
Enterprise-grade hyperparameter tuning with Stacking Ensemble
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Advanced boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


@dataclass
class TrialResult:
    """Result from a single Optuna trial"""
    model_name: str
    trial_number: int
    params: Dict[str, Any]
    accuracy: float
    f1_score: float
    training_time: float


@dataclass
class ModelResult:
    """Final result for a model after optimization"""
    model_name: str
    best_params: Dict[str, Any]
    best_accuracy: float
    best_f1: float
    best_model: Any
    n_trials: int
    optimization_time: float


class EvolutionaryAutoPilot:
    """
    Evolutionary AutoPilot with Genetic Optimization
    
    Uses Optuna's TPESampler for Bayesian hyperparameter optimization
    Creates a Super-Model using StackingClassifier ensemble
    """
    
    def __init__(self, n_trials: int = 50, cv_folds: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results: Dict[str, ModelResult] = {}
        self.trial_history: List[TrialResult] = []
        self.super_model = None
        self.best_models: List[Tuple[str, Any, float]] = []
        
    def _get_model_search_space(self, model_name: str, trial: optuna.Trial) -> Any:
        """Define hyperparameter search space for each model"""
        
        if model_name == "XGBoost" and HAS_XGBOOST:
            return xgb.XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        
        elif model_name == "LightGBM" and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                random_state=self.random_state,
                verbosity=-1
            )
        
        elif model_name == "CatBoost" and HAS_CATBOOST:
            return CatBoostClassifier(
                iterations=trial.suggest_int('iterations', 50, 300),
                depth=trial.suggest_int('depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
                random_seed=self.random_state,
                verbose=False
            )
        
        elif model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                random_state=self.random_state
            )
        
        elif model_name == "Extra Trees":
            return ExtraTreesClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif model_name == "AdaBoost":
            return AdaBoostClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
                random_state=self.random_state
            )
        
        elif model_name == "SVM":
            return SVC(
                C=trial.suggest_float('C', 1e-3, 100, log=True),
                kernel=trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
                probability=True,
                random_state=self.random_state
            )
        
        elif model_name == "KNN":
            return KNeighborsClassifier(
                n_neighbors=trial.suggest_int('n_neighbors', 3, 30),
                weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
                metric=trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
                n_jobs=-1
            )
        
        else:
            # Default: Logistic Regression
            return LogisticRegression(
                C=trial.suggest_float('C', 1e-4, 100, log=True),
                solver=trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
                max_iter=1000,
                random_state=self.random_state
            )
    
    def _optimize_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                        progress_callback=None) -> ModelResult:
        """Optimize a single model using Optuna TPESampler"""
        
        start_time = time.time()
        
        def objective(trial):
            try:
                model = self._get_model_search_space(model_name, trial)
                
                # Cross-validation
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                
                accuracy = scores.mean()
                
                # Store trial result
                self.trial_history.append(TrialResult(
                    model_name=model_name,
                    trial_number=trial.number,
                    params=trial.params.copy(),
                    accuracy=accuracy,
                    f1_score=0,  # Will compute for best model
                    training_time=time.time() - start_time
                ))
                
                if progress_callback:
                    progress_callback(model_name, trial.number, self.n_trials, accuracy)
                
                return accuracy
            
            except Exception:
                return 0.0
        
        # Create Optuna study with TPESampler
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        # Train best model on full data
        best_trial = study.best_trial
        
        # Recreate best model with best params
        class BestTrialWrapper:
            def __init__(self, params):
                self.params = params
            def suggest_int(self, name, *args, **kwargs):
                return self.params.get(name, args[0])
            def suggest_float(self, name, *args, **kwargs):
                return self.params.get(name, args[0])
            def suggest_categorical(self, name, choices, *args, **kwargs):
                return self.params.get(name, choices[0])
        
        wrapper = BestTrialWrapper(best_trial.params)
        best_model = self._get_model_search_space(model_name, wrapper)
        
        # Fit on training data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        best_model.fit(X_train, y_train)
        
        y_pred = best_model.predict(X_val)
        best_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        optimization_time = time.time() - start_time
        
        result = ModelResult(
            model_name=model_name,
            best_params=best_trial.params,
            best_accuracy=best_trial.value,
            best_f1=best_f1,
            best_model=best_model,
            n_trials=self.n_trials,
            optimization_time=optimization_time
        )
        
        self.results[model_name] = result
        return result
    
    def run_evolution(self, X: pd.DataFrame, y: pd.Series, 
                      models: List[str] = None,
                      progress_callback=None) -> Dict[str, ModelResult]:
        """
        Run evolutionary optimization on all models
        
        Args:
            X: Feature DataFrame
            y: Target Series
            models: List of model names to optimize (default: all available)
            progress_callback: Function(model_name, trial, total_trials, accuracy)
        
        Returns:
            Dictionary of model results
        """
        
        # Convert to numpy
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Default models
        if models is None:
            models = ["Random Forest", "Gradient Boosting", "Extra Trees", "AdaBoost"]
            
            # Add advanced models if available
            if HAS_XGBOOST:
                models.append("XGBoost")
            if HAS_LIGHTGBM:
                models.append("LightGBM")
            if HAS_CATBOOST:
                models.append("CatBoost")
        
        # Optimize each model
        for model_name in models:
            try:
                self._optimize_model(model_name, X_np, y_np, progress_callback)
            except Exception as e:
                print(f"Error optimizing {model_name}: {e}")
        
        return self.results
    
    def create_super_model(self, X: pd.DataFrame, y: pd.Series, top_k: int = 3) -> Tuple[Any, float]:
        """
        Create a Super-Model by stacking the top-k best models
        
        Uses StackingClassifier with LogisticRegression meta-learner
        
        Args:
            X: Feature DataFrame
            y: Target Series
            top_k: Number of top models to stack
            
        Returns:
            Tuple of (StackingClassifier, accuracy)
        """
        
        if not self.results:
            raise ValueError("No models trained. Run run_evolution() first.")
        
        # Sort models by accuracy
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1].best_accuracy,
            reverse=True
        )[:top_k]
        
        # Store best models
        self.best_models = [(name, result.best_model, result.best_accuracy) 
                           for name, result in sorted_models]
        
        # Create estimators for stacking
        estimators = [(name, result.best_model) for name, result in sorted_models]
        
        # Create StackingClassifier with LogisticRegression meta-learner
        self.super_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=self.cv_folds,
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=False
        )
        
        # Train super model
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=self.random_state
        )
        
        self.super_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.super_model.predict(X_test)
        super_accuracy = accuracy_score(y_test, y_pred)
        super_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return self.super_model, super_accuracy, super_f1
    
    def get_evolution_report(self) -> str:
        """Generate markdown report of evolution results"""
        
        if not self.results:
            return "No evolution results available. Run run_evolution() first."
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].best_accuracy,
            reverse=True
        )
        
        lines = [
            "## Evolutionary AutoPilot Report",
            "",
            "**Optimization Method:** TPE (Tree-structured Parzen Estimator)",
            f"**Trials per Model:** {self.n_trials}",
            f"**Cross-Validation Folds:** {self.cv_folds}",
            "",
            "### Model Leaderboard",
            "",
            "| Rank | Model | Accuracy | F1 Score | Optimization Time |",
            "|------|-------|----------|----------|-------------------|"
        ]
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
            lines.append(
                f"| {medal} {rank} | {name} | {result.best_accuracy*100:.2f}% | "
                f"{result.best_f1*100:.2f}% | {result.optimization_time:.1f}s |"
            )
        
        # Best model details
        best_name, best_result = sorted_results[0]
        lines.extend([
            "",
            f"### Best Model: {best_name}",
            "",
            "**Optimized Hyperparameters:**",
            "```"
        ])
        
        for param, value in best_result.best_params.items():
            lines.append(f"  {param}: {value}")
        
        lines.append("```")
        
        # Super model info
        if self.super_model is not None:
            lines.extend([
                "",
                "### Super-Model (Stacking Ensemble)",
                "",
                "**Base Models:**"
            ])
            for name, model, acc in self.best_models:
                lines.append(f"- {name} ({acc*100:.2f}%)")
            
            lines.extend([
                "",
                "**Meta-Learner:** Logistic Regression",
                "",
                "The Super-Model combines predictions from the top models using",
                "a Logistic Regression meta-learner for optimal ensemble performance."
            ])
        
        return "\n".join(lines)
    
    def get_trial_history_df(self) -> pd.DataFrame:
        """Get trial history as DataFrame for visualization"""
        
        if not self.trial_history:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'Model': t.model_name,
                'Trial': t.trial_number,
                'Accuracy': t.accuracy,
                'Time': t.training_time
            }
            for t in self.trial_history
        ])


def run_evolutionary_autopilot(df: pd.DataFrame, target_col: str, 
                                n_trials: int = 50,
                                progress_callback=None) -> Tuple[str, Any, Dict]:
    """
    Convenience function to run full evolutionary autopilot
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        n_trials: Number of Optuna trials per model
        progress_callback: Progress update function
        
    Returns:
        Tuple of (report_markdown, super_model, all_results)
    """
    
    # Prepare data
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    X = X.fillna(X.mean())
    y = df[target_col]
    
    # Run evolution
    autopilot = EvolutionaryAutoPilot(n_trials=n_trials)
    results = autopilot.run_evolution(X, y, progress_callback=progress_callback)
    
    # Create super model
    super_model, super_acc, super_f1 = autopilot.create_super_model(X, y, top_k=3)
    
    # Generate report
    report = autopilot.get_evolution_report()
    report += "\n\n### Super-Model Performance\n\n"
    report += f"- **Accuracy:** {super_acc*100:.2f}%\n"
    report += f"- **F1 Score:** {super_f1*100:.2f}%\n"
    
    return report, super_model, results, autopilot
