
def profile_dataset(df, target_col):
    profile = {}

    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]
    profile["duplicate_rows"] = int(df.duplicated().sum())

    profile["numeric_features"] = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    profile["categorical_features"] = df.select_dtypes(
        exclude=["int64", "float64"]
    ).columns.tolist()

    profile["missing_values"] = df.isnull().sum().to_dict()

    target_distribution = df[target_col].value_counts(normalize=True).to_dict()
    profile["target_distribution"] = target_distribution

    # Simple skewness summary
    try:
        skewness = df[profile["numeric_features"]].skew().to_dict()
        profile["numeric_skewness"] = skewness
    except Exception:
        profile["numeric_skewness"] = {}

    # Outlier counts (IQR method)
    outliers = {}
    try:
        for col in profile["numeric_features"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            outliers[col] = int(((df[col] < low) | (df[col] > high)).sum())
    except Exception:
        outliers = {}
    profile["outlier_counts"] = outliers

    return profile
