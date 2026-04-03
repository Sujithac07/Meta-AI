import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

from core.data_profiling import profile_dataset


class DataAgent:
    """
    Intelligent Data Agent responsible for profiling and cleaning.
    """
    def __init__(self):
        pass

    def analyze_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Profiles the data and returns a cleaned version + summary metadata.
        """
        summary = {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "types": df.dtypes.apply(lambda x: str(x)).to_dict(),
        }

        cleaned_df = df.copy()

        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

        cat_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if cleaned_df[col].isnull().any():
                mode = cleaned_df[col].mode(dropna=True)
                if not mode.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode.iloc[0])

        summary["cleaning_actions"] = ["Median imputation for numeric", "Mode imputation for categorical"]

        return cleaned_df, summary

    def run(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Legacy entry point retained for the existing pipeline/tests."""
        cleaned_df, summary = self.analyze_and_clean(df)
        profile = profile_dataset(cleaned_df, target_col)
        profile["cleaning_summary"] = summary
        return {
            "profile": profile,
            "cleaned_df": cleaned_df,
            "comment": "Dataset successfully analyzed by DataAgent.",
        }
