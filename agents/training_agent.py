from core.model_training import train_model
from crewai import Agent as CrewAgent, Task, Crew
import optuna
# LangChain imports removed due to API changes

def train_single_model_standalone(model_name, df, target_col, hyperparameter_tuner):
    """Standalone function for parallel training that can be pickled"""
    try:
        # Use Optuna for hyperparameter optimization if available
        if hyperparameter_tuner and model_name in ["XGBoost", "LightGBM"]:
            best_params = optimize_hyperparameters_optuna_standalone(model_name, df, target_col, hyperparameter_tuner)
            # Apply best parameters to model training
            model, metrics = train_model(model_name, df, target_col, **best_params)
        else:
            model, metrics = train_model(model_name, df, target_col)

        return model, metrics
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0}

def optimize_hyperparameters_optuna_standalone(model_name, df, target_col, hyperparameter_tuner):
    """Standalone function for hyperparameter optimization"""
    try:
        def objective(trial):
            if model_name == "XGBoost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                }
            elif model_name == "LightGBM":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                }
            else:
                return 0

            try:
                _, metrics = train_model(model_name, df, target_col, **params)
                return metrics.get("accuracy", 0)
            except Exception:
                return 0

        if hyperparameter_tuner:
            hyperparameter_tuner.optimize(objective, n_trials=10)
            return hyperparameter_tuner.best_params
        return {}
    except Exception:
        return {}

class TrainingAgent:
    def __init__(self):
        # Initialize advanced ML tools
        self.hyperparameter_tuner = None
        self.autogen_agents = None
        self.crew_agents = None

        self._setup_advanced_tools()

    def _setup_advanced_tools(self):
        """Setup advanced ML and AI tools"""
        # Setup Optuna for hyperparameter tuning
        try:
            self.hyperparameter_tuner = optuna.create_study(direction='maximize')
        except Exception:
            self.hyperparameter_tuner = None

        # LangChain agent setup removed due to API compatibility issues
        self.langchain_agent = None



        # Setup CrewAI for collaborative model development
        try:
            model_trainer = CrewAgent(
                role='Model Trainer',
                goal='Train and optimize ML models',
                backstory='Expert in training various ML algorithms with best practices',
                verbose=True
            )

            hyperparameter_optimizer = CrewAgent(
                role='Hyperparameter Optimizer',
                goal='Optimize model hyperparameters for best performance',
                backstory='Specialist in hyperparameter tuning and optimization techniques',
                verbose=True
            )

            performance_evaluator = CrewAgent(
                role='Performance Evaluator',
                goal='Evaluate and validate model performance',
                backstory='Expert in model evaluation metrics and validation techniques',
                verbose=True
            )

            self.crew_agents = [model_trainer, hyperparameter_optimizer, performance_evaluator]
        except Exception:
            self.crew_agents = None



    def _optimize_hyperparameters_optuna(self, model_name, df, target_col):
        """Use Optuna for hyperparameter optimization"""
        def objective(trial):
            if model_name == "XGBoost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                }
            elif model_name == "LightGBM":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                }
            else:
                return 0

            try:
                _, metrics = train_model(model_name, df, target_col, **params)
                return metrics.get("accuracy", 0)
            except Exception:
                return 0

        if self.hyperparameter_tuner:
            self.hyperparameter_tuner.optimize(objective, n_trials=10)
            return self.hyperparameter_tuner.best_params
        return {}

    def _compare_model_performance(self, results_dict):
        """Compare model performances using advanced metrics"""
        comparison = {}
        for model_name, result in results_dict.items():
            metrics = result.get('metrics', {})
            # Calculate composite score
            composite_score = (
                metrics.get('accuracy', 0) * 0.3 +
                metrics.get('f1', 0) * 0.3 +
                metrics.get('roc_auc', 0) * 0.4
            )
            comparison[model_name] = {
                'composite_score': composite_score,
                'metrics': metrics
            }

        return comparison

    def _optimize_hyperparameters(self, model_name, df, target_col):
        """Advanced hyperparameter optimization"""
        return self._optimize_hyperparameters_optuna(model_name, df, target_col)

    def _crew_collaborative_training(self, df, target_col, strategy):
        """Use CrewAI for collaborative model training"""
        if not self.crew_agents:
            return None

        try:
            models_to_try = strategy["models_to_try"]

            # Create tasks for each agent
            training_task = Task(
                expected_output="Training results and metrics",
                description=f"Train models {models_to_try} on dataset with {len(df)} samples",
                agent=self.crew_agents[0]
            )

            optimization_task = Task(
                expected_output="Optimized hyperparameters and improved metrics",
                description="Optimize hyperparameters for best performing models",
                agent=self.crew_agents[1]
            )

            evaluation_task = Task(
                expected_output="Performance evaluation and recommendations",
                description="Evaluate model performance and provide recommendations",
                agent=self.crew_agents[2]
            )

            # Create and run crew
            crew = Crew(
                agents=self.crew_agents,
                tasks=[training_task, optimization_task, evaluation_task],
                verbose=True,
                tracing=True
            )
            return crew.kickoff()
        except Exception as e:
            print(f"CrewAI collaboration failed: {e}")
            return None

    def run(self, df, target_col, strategy):
        models_to_try = strategy["models_to_try"]
        primary_metric = strategy["primary_metric"]

        # Train models sequentially to avoid pickling issues
        parallel_results = []
        for model_name in models_to_try:
            result = train_single_model_standalone(model_name, df, target_col, self.hyperparameter_tuner)
            parallel_results.append(result)

        results = {}
        best_model = None
        best_score = -float('inf')
        best_metrics = None
        best_model_name = None

        for model_name, (model, metrics) in zip(models_to_try, parallel_results):
            if model is not None:
                results[model_name] = {
                    "model": model,
                    "metrics": metrics
                }
                score = metrics.get(primary_metric, 0)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_metrics = metrics
                    best_model_name = model_name

        # Add CrewAI collaborative insights
        crew_insights = self._crew_collaborative_training(df, target_col, strategy)
        if crew_insights:
            results["crew_collaboration"] = str(crew_insights)

        # Add advanced tool usage tracking
        results["advanced_tools_used"] = {
            "crewai": self.crew_agents is not None
        }

        return {
            "all_results": results,
            "best_model": best_model,
            "best_metrics": best_metrics,
            "best_model_name": best_model_name
        }
