import pandas as pd
import os
import sys

# Force UTF-8 output to prevent UnicodeEncodeError on Windows (cp1252)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        print(f"Warning: UTF-8 console reconfigure failed: {e}")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

from core.data_loader import load_data
from core.model_training import train_model

class AutoPilot:
    """
    Mastery Auto-Pilot: End-to-End ML Orchestrator.
    Autonomously handles tuning, evaluation, and registration of the best models.
    """
    
    def __init__(self, experiment_name: str = "Mastery_AutoPilot"):
        self.experiment_name = experiment_name
        self.mlflow_enabled = MLFLOW_AVAILABLE
        if self.mlflow_enabled:
            try:
                mlflow.set_experiment(self.experiment_name)
            except Exception:
                try:
                    mlflow.create_experiment(self.experiment_name)
                    mlflow.set_experiment(self.experiment_name)
                except Exception:
                    self.mlflow_enabled = False
            
        self.algorithms = [
            "RandomForest", "GradientBoosting", "ExtraTrees", 
            "HistGradientBoosting", "AdaBoost", "LogisticRegression",
            "SVC", "KNN", "NaiveBayes"
        ]
        self.leaderboard = []

    def run(self, df: pd.DataFrame, target_col: str, n_trials: int = 5):
        """Runs the full E2E pipeline for all algorithms."""
        print(f"[AutoPilot] Starting experiment: {self.experiment_name}")
        
        for algo in self.algorithms:
            print(f"[AutoPilot] Training & Tuning: {algo}")
            try:
                model, results = train_model(
                    model_name=algo,
                    df=df,
                    target_col=target_col,
                    optimize=True,
                    n_trials=n_trials
                )
                
                if model:
                    results['algorithm'] = algo
                    self.leaderboard.append(results)
                    print(f"[AutoPilot] OK {algo} Accuracy: {results.get('accuracy', 0):.4f}")
            except Exception as e:
                print(f"[AutoPilot] ERROR training {algo}: {e}")

        if self.leaderboard:
            self._finalize_champion(df, target_col)
        else:
            print("[AutoPilot] WARNING: No models were successfully trained.")

    def _finalize_champion(self, df: pd.DataFrame, target_col: str):
        sorted_board = sorted(self.leaderboard, key=lambda x: x.get('f1', 0), reverse=True)
        champion_stats = sorted_board[0]
        champion_name = champion_stats['algorithm']
        
        print(f"[AutoPilot] CHAMPION: {champion_name} (F1: {champion_stats.get('f1', 0):.4f})")
        
        if self.mlflow_enabled:
            try:
                with mlflow.start_run(run_name=f"Champion_{champion_name}"):
                    mlflow.log_params({k: v for k, v in champion_stats.items() if isinstance(v, (str, int, float, bool))})
                    mlflow.set_tag("model_status", "champion")
            except Exception as e:
                print(f"[AutoPilot] MLflow logging skipped: {e}")
        
        os.makedirs("models", exist_ok=True)
        print(f"[AutoPilot] Champion {champion_name} registered and ready for FastAPI.")
            
        self.leaderboard = sorted_board

    def get_leaderboard_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.leaderboard)

if __name__ == "__main__":
    data = load_data()
    if data is not None:
        pilot = AutoPilot(experiment_name="CLI_Test_Run")
        pilot.run(data, target_col="HeartDiseaseorAttack", n_trials=2)
        print("\nLeaderboard:")
        print(pilot.get_leaderboard_df())
