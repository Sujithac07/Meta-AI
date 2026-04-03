import pandas as pd
import numpy as np
import os
import re

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: Torch/PyTorch not available or failed to initialize: {e}")
    TORCH_AVAILABLE = False

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


try:
    from core.deep_learning import TabularDeepLearningSpecs
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

from core.optimization import optimize_hyperparameters

# --- MLOps Imports ---
try:
    import mlflow
    import mlflow.sklearn
    try:
        import mlflow.pytorch
        import mlflow.tensorflow
    except (ImportError, OSError):
        mlflow.pytorch = None  # Handle case where specific framework loggers fail
        mlflow.tensorflow = None
    MLFLOW_AVAILABLE = True
except (ImportError, OSError) as e:
    MLFLOW_AVAILABLE = False
    print(f"Warning: MLflow not available or failed to initialize: {e}")



def _build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor

def _safe_roc_auc(y_true, y_prob, num_classes):
    if len(set(y_true)) <= 1:
        return 0.5
    try:
        if num_classes == 2:
            return roc_auc_score(y_true, y_prob)
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        return 0.5


def _is_series_exact_match(a: pd.Series, b: pd.Series) -> bool:
    """Return True when two series represent identical values."""
    if len(a) != len(b):
        return False

    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    mask = a_num.notna() & b_num.notna()
    if mask.sum() > 0:
        if (a_num[mask] - b_num[mask]).abs().max() <= 1e-12 and (a_num.isna() == b_num.isna()).all():
            return True

    a_txt = a.fillna("__NA__").astype(str).str.strip().str.lower()
    b_txt = b.fillna("__NA__").astype(str).str.strip().str.lower()
    return bool((a_txt == b_txt).all())


def _drop_target_leakage_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:
    """Drop obvious target-leakage columns before train/test split."""
    if target_col not in df.columns:
        return df, {}

    y = df[target_col]
    y_num = pd.to_numeric(y, errors="coerce")
    target_norm = re.sub(r"[^a-z0-9]+", "", str(target_col).lower())
    leaked = {}

    for col in df.columns:
        if col == target_col:
            continue
        x = df[col]
        col_norm = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        if target_norm and col_norm == target_norm:
            leaked[col] = "same semantic name as target"
            continue

        if _is_series_exact_match(x, y):
            leaked[col] = "identical to target values"
            continue

        x_num = pd.to_numeric(x, errors="coerce")
        valid = x_num.notna() & y_num.notna()
        if valid.sum() >= 50 and x_num[valid].nunique() > 1 and y_num[valid].nunique() > 1:
            corr = np.corrcoef(x_num[valid], y_num[valid])[0, 1]
            if np.isfinite(corr) and abs(corr) >= 0.9999:
                leaked[col] = f"near-perfect correlation with target (corr={corr:.4f})"

    if not leaked:
        return df, {}

    filtered_df = df.drop(columns=list(leaked.keys()), errors="ignore")
    return filtered_df, leaked

def train_model(model_name, df, target_col, metric=None, random_state=42, **kwargs):
    df_filtered, leakage_map = _drop_target_leakage_features(df, target_col)
    X = df_filtered.drop(columns=[target_col])
    y = df_filtered[target_col]

    if leakage_map:
        print(f"[LeakageGuard] Removed {len(leakage_map)} suspicious feature(s): {list(leakage_map.keys())[:10]}")
    if X.shape[1] == 0:
        print("[LeakageGuard] No usable features left after leakage filtering.")
        return None, {}

    # Adjust test size for small datasets
    test_size = 0.2 if len(X) > 10 else 0.5

    # Robust Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    num_classes = len(classes)

    if num_classes < 2:
        print("Only one class found in target. Skipping training.")
        return None, {}

    # Use Stratified split if possible
    # We need at least 2 members in each class to stratify
    class_counts = pd.Series(y_encoded).value_counts()
    min_class_count = class_counts.min() if not class_counts.empty else 0
    stratify_param = y_encoded if num_classes > 1 and len(y_encoded) > 10 and min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    # Build Preprocessor
    preprocessor = _build_preprocessor(X_train)
    
    # --- Optimization Layer ---
    run_optimization = kwargs.pop('optimize', False)
    n_trials = kwargs.pop('n_trials', 10)
    
    if run_optimization:
        print(f"⚡ Optimizing hyperparameters for {model_name} (AutoML)... n_trials={n_trials}")
        # Need to preprocess for optimization
        X_train_opt = preprocessor.fit_transform(X_train)
        best_params = optimize_hyperparameters(model_name, X_train_opt, y_train, n_trials=n_trials)
        print(f"   Best Params: {best_params}")
        kwargs.update(best_params)
    # --------------------------


    if model_name == "LogisticRegression":
        estimator = LogisticRegression(max_iter=1000, **kwargs)
    elif model_name == "RandomForest":
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
    elif model_name == "GradientBoosting":
        estimator = GradientBoostingClassifier(n_estimators=100, random_state=42, **kwargs)
    
    # --- New Mastery Algorithms ---
    elif model_name == "SVC":
        estimator = SVC(probability=True, random_state=random_state, **kwargs)
    
    elif model_name == "KNN":
        n_neighbors = min(5, len(X_train) - 1) if len(X_train) > 1 else 1
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    
    elif model_name == "AdaBoost":
        estimator = AdaBoostClassifier(random_state=random_state, **kwargs)
    
    elif model_name == "NaiveBayes":
        estimator = GaussianNB(**kwargs)
    
    elif model_name == "ExtraTrees":
        estimator = ExtraTreesClassifier(random_state=random_state, **kwargs)
        
    elif model_name == "HistGradientBoosting":
        estimator = HistGradientBoostingClassifier(random_state=random_state, **kwargs)
    # ------------------------------
    elif model_name == "DecisionTree":
        estimator = DecisionTreeClassifier(random_state=42, **kwargs)
    elif model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost not available")
        objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
        estimator = xgb.XGBClassifier(
            objective=objective,
            n_estimators=100,
            random_state=42,
            **kwargs
        )
    elif model_name == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM not available")
        estimator = lgb.LGBMClassifier(n_estimators=100, random_state=42, **kwargs)
        
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch/Torch not available due to initialization error.")
        # Define device
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
        
        estimator = TabularDeepLearningSpecs(
            n_d=16, n_a=16, n_steps=5,
            optimizer_params=dict(lr=2e-2),
            max_epochs=50,
            device_name=device,
            **kwargs
        )
        
    elif model_name == "PyTorchMLP":
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch/Torch not available due to initialization error.")
        # ... (Legacy MLP code kept as fallback or alternative)

        # Research-Grade Residual Network for Tabular Data
        class ResNetBlock(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.linear1 = nn.Linear(features, features)
                self.bn1 = nn.BatchNorm1d(features)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.linear2 = nn.Linear(features, features)
                self.bn2 = nn.BatchNorm1d(features)

            def forward(self, x):
                identity = x
                out = self.linear1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.linear2(out)
                out = self.bn2(out)
                out += identity
                out = self.relu(out)
                return out

        class AdvancedMLP(nn.Module):
            def __init__(self, input_size, num_classes):
                super(AdvancedMLP, self).__init__()
                # Embedding / Projection Layer
                self.project = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU()
                )
                
                # Residual Blocks
                self.res1 = ResNetBlock(128)
                self.res2 = ResNetBlock(128)
                
                # Output Head
                output_size = 1 if num_classes == 2 else num_classes
                self.output = nn.Linear(128, output_size)
                self.num_classes = num_classes
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.project(x)
                x = self.res1(x)
                x = self.res2(x)
                x = self.output(x)
                if self.num_classes == 2:
                    x = self.sigmoid(x)
                return x

        # Prepare data
        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()
        X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_p, dtype=torch.float32)
        
        # Targets: Float for BCE, Long for CrossEntropy
        if num_classes == 2:
            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            criterion = nn.BCELoss()
        else:
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            y_test_t = torch.tensor(y_test, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()

        model = AdvancedMLP(X_train_t.shape[1], num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"Training PyTorch AdvancedMLP ({'Binary' if num_classes==2 else 'Multi-class'})...")
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_t)
                val_loss = criterion(val_outputs, y_test_t)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                # print(f"Early stopping at epoch {epoch}")
                break

        # Interface wrappers
        model.eval()
        model._preprocessor = preprocessor
        model._feature_names = feature_names
        if num_classes == 2:
            model.predict = lambda X: (model(torch.tensor(preprocessor.transform(X), dtype=torch.float32)).detach().numpy() > 0.5).astype(int).flatten()
            model.predict_proba = lambda X: torch.cat(
                [1 - model(torch.tensor(preprocessor.transform(X), dtype=torch.float32)),
                 model(torch.tensor(preprocessor.transform(X), dtype=torch.float32))],
                dim=1
            ).detach().numpy()
        else:
            model.predict = lambda X: torch.argmax(model(torch.tensor(preprocessor.transform(X), dtype=torch.float32)), dim=1).detach().numpy()
            model.predict_proba = lambda X: torch.softmax(model(torch.tensor(preprocessor.transform(X), dtype=torch.float32)), dim=1).detach().numpy()

    elif model_name == "TensorFlowNN":
        if TENSORFLOW_AVAILABLE:
            # Advanced Keras Model
            output_units = 1 if num_classes == 2 else num_classes
            final_activation = 'sigmoid' if num_classes == 2 else 'softmax'
            loss_fn = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
            
            model = keras.Sequential([
                keras.layers.Dense(128, input_shape=(X_train.shape[1],)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(output_units, activation=final_activation)
            ])
            
            model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
            
            print(f"Training TensorFlow Advanced NN ({'Binary' if num_classes==2 else 'Multi-class'})...")
            X_train_p = preprocessor.fit_transform(X_train)
            X_test_p = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out()
            model.fit(
                X_train_p, y_train, 
                validation_data=(X_test_p, y_test),
                epochs=100, 
                batch_size=32, 
                callbacks=[early_stopping],
                verbose=0
            )
            model._preprocessor = preprocessor
            model._feature_names = feature_names

            keras_model = model
            model.predict = lambda X: (keras_model.predict(preprocessor.transform(X)) > 0.5).astype(int).flatten() if num_classes == 2 else np.argmax(keras_model.predict(preprocessor.transform(X)), axis=1)
            model.predict_proba = lambda X: keras_model.predict(preprocessor.transform(X))
        else:
            raise ValueError("TensorFlow not available")
    else:
        raise ValueError(f"Unknown model: {model_name}")


    # Only call fit for sklearn-like models
    if model_name not in ["PyTorchMLP", "TensorFlowNN"]:
        try:
            model = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
            model.fit(X_train, y_train)
            try:
                model._feature_names = model.named_steps["preprocess"].get_feature_names_out()
                model._estimator = model.named_steps["model"]
            except Exception:
                model._feature_names = None
                model._estimator = model.named_steps.get("model")
        except Exception as e:
            print(f"Error fitting {model_name}: {e}")
            # Return dummy results if training fails
            return None, {}

    try:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                if y_prob.shape[1] > 1:
                    y_prob = y_prob if num_classes > 2 else y_prob[:, 1]
                else:
                    y_prob = y_prob.flatten()
            except Exception:
                y_prob = y_pred # Fallback
        else:
            y_prob = y_pred # Fallback

        scores_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # Calculate AUC safely
        scores_dict["roc_auc"] = _safe_roc_auc(y_test, y_prob, num_classes)

        results = scores_dict

        # --- Professional MLOps Integration (MLflow) ---
        if MLFLOW_AVAILABLE:
            try:
                # Use a specific experiment name
                experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Meta_AI_Core_Pipeline")
                mlflow.set_experiment(experiment_name)

                # Determine if we are within an existing context (e.g., from main.py)
                # If not, we start a new run
                active_run = mlflow.active_run()
                context_manager = mlflow.start_run(nested=True, run_name=f"Node_{model_name}") if active_run else mlflow.start_run(run_name=f"Train_{model_name}")
                
                with context_manager:
                    # 1. Log Basic Info
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("num_features", X_train.shape[1])
                    mlflow.log_param("rows_train", len(X_train))
                    
                    # 2. Log Hyperparameters (if any passed via kwargs or found by optuna)
                    if kwargs:
                        mlflow.log_params({f"hp_{k}": v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})

                    # 3. Log Performance Metrics
                    mlflow.log_metrics(results)

                    # 4. Log Model with Signature (Meta-data)
                    from mlflow.models import infer_signature
                    signature = infer_signature(X_test[:5], y_pred[:5])
                    
                    if model_name == "PyTorchMLP" and TORCH_AVAILABLE:
                        mlflow.pytorch.log_model(model, "pytorch_model", signature=signature)
                    elif model_name == "TensorFlowNN" and TENSORFLOW_AVAILABLE:
                        mlflow.tensorflow.log_model(model, "tf_model", signature=signature)
                    else:
                        mlflow.sklearn.log_model(model, "sklearn_model", signature=signature)
                    
                    # 5. Add Tags for Searchability
                    mlflow.set_tag("stage", "development")
                    mlflow.set_tag("user", os.getenv("USERNAME", "antigravity"))
                    
                    # 6. Log to MetaLearner for future recommendations
                    if results and model is not None:
                        try:
                            from core.meta_learner import MetaLearner, extract_meta_features
                            
                            # Extract meta-features from dataset
                            meta_features = extract_meta_features(df, target_col)
                            
                            # Create MetaLearner instance and log experiment
                            meta_learner = MetaLearner()
                            meta_learner.log_experiment(
                                meta_features=meta_features,
                                model_name=model_name,
                                f1_score=results.get('f1', 0),
                                accuracy=results.get('accuracy', 0)
                            )
                            print("   ✓ Logged to MetaLearner memory for future recommendations")
                        except Exception as meta_error:
                            print(f"Warning: MetaLearner logging failed (non-fatal): {meta_error}")

            except Exception as e:
                print(f"Warning: MLflow tracking encountered a non-fatal error: {e}")
        # ---------------------------------------------


    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, {}

    return model, results
