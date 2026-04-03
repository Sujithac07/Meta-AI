"""Hardening tests for production export packaging. Run: pytest tests/test_export_hardening.py -q"""

import os
import tempfile
import zipfile
import py_compile

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core.production_export import ProductionExporter


def _train_small_model(use_scaler: bool):
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 35, 45],
            "bmi": [21.5, 24.0, 29.1, 30.0, 26.2, 27.4],
            "income": [30, 45, 70, 85, 55, 60],
        }
    )
    y = np.array([0, 0, 1, 1, 0, 1])

    if use_scaler:
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=200, random_state=42).fit(scaler.transform(X), y)
        preprocessors = {"scaler": scaler}
    else:
        model = DecisionTreeClassifier(random_state=42).fit(X, y)
        preprocessors = None

    return model, X.columns.tolist(), preprocessors


def _assert_zip_contains(zip_path: str, required_files: list[str]):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    missing = [f for f in required_files if f not in names]
    assert not missing, f"Missing expected files: {missing}"


def _assert_python_files_compile(zip_path: str):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        for rel_path in ["api.py", "schema.py", "main.py"]:
            full_path = os.path.join(tmp, rel_path)
            assert os.path.exists(full_path), f"{rel_path} was not exported"
            py_compile.compile(full_path, doraise=True)


def test_export_without_preprocessors():
    with tempfile.TemporaryDirectory() as export_dir:
        exporter = ProductionExporter(export_dir=export_dir)
        model, features, preprocessors = _train_small_model(use_scaler=False)

        zip_path = exporter.export_production_package(
            model=model,
            feature_columns=features,
            target_column="target",
            task_type="classification",
            model_name="hardening_tree",
            preprocessors=preprocessors,
        )

        _assert_zip_contains(
            zip_path,
            [
                "api.py",
                "model/model.joblib",
                "model/metadata.json",
                "requirements.txt",
                "Dockerfile",
                "README.md",
            ],
        )
        _assert_python_files_compile(zip_path)


def test_export_with_scaler_preprocessor():
    with tempfile.TemporaryDirectory() as export_dir:
        exporter = ProductionExporter(export_dir=export_dir)
        model, features, preprocessors = _train_small_model(use_scaler=True)

        zip_path = exporter.export_production_package(
            model=model,
            feature_columns=features,
            target_column="target",
            task_type="classification",
            model_name="hardening_scaled",
            preprocessors=preprocessors,
        )

        _assert_zip_contains(
            zip_path,
            [
                "api.py",
                "model/model.joblib",
                "model/scaler.joblib",
                "requirements.txt",
                "Dockerfile",
                "README.md",
            ],
        )
        _assert_python_files_compile(zip_path)
