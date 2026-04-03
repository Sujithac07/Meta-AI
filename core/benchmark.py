"""
BenchmarkRunner: Compare MetaAI against baseline AutoML approaches
"""

import json
import time
import sys
import os
import pandas as pd
from typing import Tuple, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BenchmarkRunner:
    """
    Benchmark runner for comparing multiple ML models
    """
    
    def __init__(self):
        """Initialize benchmark runner"""
        self.results = []
        self.models_to_test = [
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
            "LightGBM",
            "LogisticRegression",
            "ExtraTrees",
            "HistGradientBoosting"
        ]
    
    def run_benchmark(self, df: pd.DataFrame, target_col: str, n_trials: int = 5) -> None:
        """
        Run benchmark on all models
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            n_trials: Number of Optuna trials (not used in benchmark, just for compatibility)
        """
        try:
            from core.model_training import train_model
            
            print(f"\n{'='*70}")
            print(f"  BENCHMARK: Testing {len(self.models_to_test)} Models")
            print(f"{'='*70}\n")
            
            self.results = []
            
            # Fast benchmark defaults to avoid long-running UI hangs
            fast_kwargs_map = {
                "RandomForest": {"n_estimators": 60},
                "GradientBoosting": {"n_estimators": 80},
                "XGBoost": {"n_estimators": 80},
                "LightGBM": {"n_estimators": 80},
                "ExtraTrees": {"n_estimators": 80},
            }
            timeout_seconds = 120

            for model_name in self.models_to_test:
                try:
                    print(f"[{len(self.results)+1}/{len(self.models_to_test)}] Training {model_name}...", end=" ")
                    
                    # Measure training time
                    start_time = time.time()
                    model, metrics = train_model(
                        model_name,
                        df,
                        target_col,
                        optimize=False,
                        n_trials=int(n_trials),
                        **fast_kwargs_map.get(model_name, {})
                    )
                    training_time = time.time() - start_time

                    if training_time > timeout_seconds:
                        print(f"⏱ TIMEOUT ({training_time:.2f}s)")
                        continue
                    
                    if model is None or not metrics:
                        print("❌ FAILED")
                        continue
                    
                    # Record results
                    result = {
                        'model_name': model_name,
                        'accuracy': metrics.get('accuracy', 0.0),
                        'f1': metrics.get('f1', 0.0),
                        'precision': metrics.get('precision', 0.0),
                        'recall': metrics.get('recall', 0.0),
                        'roc_auc': metrics.get('roc_auc', 0.0),
                        'training_time_seconds': round(training_time, 3)
                    }
                    
                    self.results.append(result)
                    print(f"✓ F1: {result['f1']:.4f} ({training_time:.2f}s)")
                    
                except Exception as e:
                    print(f"❌ ERROR: {str(e)}")
                    continue
            
            # Save results
            self._save_results()
            
            print(f"\n{'='*70}")
            print(f"  Benchmark Complete! {len(self.results)}/{len(self.models_to_test)} models trained")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"Benchmark error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_results_df(self) -> pd.DataFrame:
        """
        Get results as a DataFrame sorted by F1 score descending
        
        Returns:
            DataFrame with benchmark results
        """
        try:
            if not self.results:
                return pd.DataFrame()
            
            df = pd.DataFrame(self.results)
            df = df.sort_values('f1', ascending=False).reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"Error getting results DataFrame: {e}")
            return pd.DataFrame()
    
    def get_champion(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Get the champion model (best F1 score)
        
        Returns:
            Tuple of (model_name, f1_score) or (None, None) if no results
        """
        try:
            if not self.results:
                return None, None
            
            df = self.get_results_df()
            if df.empty:
                return None, None
            
            champion_row = df.iloc[0]
            return champion_row['model_name'], champion_row['f1']
            
        except Exception as e:
            print(f"Error getting champion: {e}")
            return None, None
    
    def generate_report(self) -> str:
        """
        Generate a markdown report with full comparison
        
        Returns:
            Markdown formatted report string
        """
        try:
            if not self.results:
                return "⚠️ No benchmark results available. Run benchmark first."
            
            df = self.get_results_df()
            champion_name, champion_f1 = self.get_champion()
            
            # Build markdown report
            report = f"""
# 🏆 BENCHMARK RESULTS

## Champion Model

**🥇 {champion_name}** achieved the highest F1 score: **{champion_f1:.4f}**

---

## 📊 Full Comparison Table

| Rank | Model | F1 Score | Accuracy | Precision | Recall | ROC-AUC | Time (s) |
|------|-------|----------|----------|-----------|--------|---------|----------|
"""
            
            for idx, row in df.iterrows():
                rank = idx + 1
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
                
                report += f"| {medal} | {row['model_name']} | "
                report += f"{row['f1']:.4f} | {row['accuracy']:.4f} | "
                report += f"{row['precision']:.4f} | {row['recall']:.4f} | "
                report += f"{row['roc_auc']:.4f} | {row['training_time_seconds']:.2f} |\n"
            
            # Add statistics
            report += "\n---\n\n## 📈 Statistics\n\n"
            report += f"- **Models Tested:** {len(self.results)}\n"
            report += f"- **Best F1 Score:** {df['f1'].max():.4f}\n"
            report += f"- **Average F1 Score:** {df['f1'].mean():.4f}\n"
            report += f"- **Worst F1 Score:** {df['f1'].min():.4f}\n"
            report += f"- **Total Training Time:** {df['training_time_seconds'].sum():.2f}s\n"
            report += f"- **Average Training Time:** {df['training_time_seconds'].mean():.2f}s\n"
            
            # Add recommendations
            report += "\n---\n\n## 💡 Recommendations\n\n"
            
            if champion_f1 and champion_f1 > 0.85:
                report += f"✅ **Excellent Performance:** {champion_name} achieved {champion_f1:.1%} F1 score.\n"
            elif champion_f1 and champion_f1 > 0.70:
                report += f"⚠️ **Good Performance:** {champion_name} achieved {champion_f1:.1%} F1 score. Consider hyperparameter tuning for improvement.\n"
            else:
                report += f"❌ **Poor Performance:** Best F1 score is only {champion_f1:.1%}. Consider feature engineering or data quality improvements.\n"
            
            # Find fastest model
            fastest = df.loc[df['training_time_seconds'].idxmin()]
            report += f"\n⚡ **Fastest Model:** {fastest['model_name']} ({fastest['training_time_seconds']:.2f}s)\n"
            
            # Find best balanced model
            df['balance_score'] = (df['precision'] + df['recall']) / 2
            best_balanced = df.loc[df['balance_score'].idxmax()]
            report += f"\n⚖️ **Best Balanced (Precision/Recall):** {best_balanced['model_name']}\n"
            
            return report.strip()
            
        except Exception as e:
            return f"❌ Error generating report: {str(e)}"
    
    def _save_results(self) -> None:
        """Save results to JSON file"""
        try:
            output = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_models': len(self.results),
                'results': self.results
            }
            
            with open('benchmark_results.json', 'w') as f:
                json.dump(output, f, indent=2)
            
            print("\n✓ Results saved to: benchmark_results.json")
            
        except Exception as e:
            print(f"Warning: Could not save results to JSON: {e}")


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create sample data
    print("Creating sample dataset...")
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'feature4': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Run benchmark
    runner = BenchmarkRunner()
    runner.run_benchmark(sample_data, 'target')
    
    # Get results
    report = runner.generate_report()
    print("\n" + report)
    
    # Get champion (with None check)
    champion, f1 = runner.get_champion()
    if champion and f1:
        print(f"\n🏆 Champion: {champion} with F1={f1:.4f}")
    else:
        print("\n⚠️ No champion determined (benchmark may have failed)")
