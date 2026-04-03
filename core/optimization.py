
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def optimize_hyperparameters(model_name, X, y, n_trials=10):
    """
    Runs Optuna optimization for the given model name.
    """
    
    def objective(trial):
        if model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
            }
            model = RandomForestClassifier(**params, random_state=42)
            
        elif model_name == "GradientBoosting":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 8)
            }
            model = GradientBoostingClassifier(**params, random_state=42)
            
        elif model_name == "LogisticRegression":
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e1, log=True)
            }
            model = LogisticRegression(**params, max_iter=1000, random_state=42)
            
        elif model_name == "SVC":
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf'])
            }
            model = SVC(**params, probability=True, random_state=42)
            
        elif model_name == "KNN":
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
            }
            model = KNeighborsClassifier(**params)
            
        elif model_name == "AdaBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0)
            }
            model = AdaBoostClassifier(**params, random_state=42)
            
        elif model_name == "ExtraTrees":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20)
            }
            model = ExtraTreesClassifier(**params, random_state=42)
            
        elif model_name == "HistGradientBoosting":
            params = {
                'max_iter': trial.suggest_int('max_iter', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 12)
            }
            model = HistGradientBoostingClassifier(**params, random_state=42)
            
        else:
            return 0.0

        # robust CV
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(model, X, y, cv=cv, scoring='f1_macro').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

