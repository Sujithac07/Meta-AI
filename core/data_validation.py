def validate_dataset(df, target_col):
    """
    Lightweight dataset validation for quality and leakage signals.
    Returns a dict with checks and warnings.
    """
    checks = {}
    warnings = []

    # Target presence
    checks["target_present"] = target_col in df.columns
    if not checks["target_present"]:
        warnings.append(f"Target column '{target_col}' not found.")
        return {"checks": checks, "warnings": warnings}

    # Basic stats
    checks["rows"] = df.shape[0]
    checks["columns"] = df.shape[1]
    checks["duplicate_rows"] = int(df.duplicated().sum())
    if checks["duplicate_rows"] > 0:
        warnings.append("Duplicate rows detected.")

    # Missing values
    missing = df.isnull().sum().to_dict()
    checks["missing_values"] = missing
    if any(v > 0 for v in missing.values()):
        warnings.append("Missing values detected.")

    # Leakage heuristic: columns identical to target
    leakage_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        try:
            if df[col].equals(df[target_col]):
                leakage_cols.append(col)
        except Exception:
            warnings.append(f"Could not compare column '{col}' to target for leakage checks.")
    checks["leakage_candidates"] = leakage_cols
    if leakage_cols:
        warnings.append(f"Potential leakage columns: {leakage_cols}")

    # Target distribution
    try:
        dist = df[target_col].value_counts(normalize=True).to_dict()
        checks["target_distribution"] = dist
    except Exception:
        checks["target_distribution"] = {}

    return {"checks": checks, "warnings": warnings}
