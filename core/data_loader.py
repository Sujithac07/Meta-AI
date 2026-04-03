import pandas as pd
import os
from sklearn.datasets import make_classification

def load_data(file_path="data/raw/your_dataset.csv", target_col="target"):
    """
    Robust data loader that attempts to load a CSV.
    If not found, generates synthetic classification data.
    Returns:
        df: pandas DataFrame
        target_col: str name of target column
    """
    if os.path.exists(file_path):
        try:
            print(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path)
            # Basic validation
            if target_col not in df.columns:
                print(f"Warning: Target column '{target_col}' not found. Using last column.")
                target_col = df.columns[-1]
            return df, target_col
        except Exception as e:
            print(f"Error reading CSV: {e}")
            print("Falling back to synthetic data...")
    else:
        print(f"File not found: {file_path}")
        print("Generating high-quality synthetic data for 'Advanced AI' simulation...")

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        flip_y=0.05,  # Add some noise
        class_sep=1.5
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    print(f"Generated synthetic dataset: {df.shape}")
    return df, "target"
