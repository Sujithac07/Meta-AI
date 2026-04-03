import numpy as np

def stability_test(train_fn, model_name, df, target_col, metric, runs=3):
    scores = []

    for seed in range(runs):
        model, results = train_fn(
            model_name,
            df,
            target_col,
            metric,
            random_state=seed
        )

        if results and "f1" in results:
            scores.append(results["f1"])

    mean_score = np.mean(scores) if scores else 0.0
    std_score = np.std(scores) if scores else 0.0

    stability = "high" if std_score < 0.02 else "low"

    return {
        "scores": scores,
        "mean": mean_score,
        "std": std_score,
        "stability": stability
    }
