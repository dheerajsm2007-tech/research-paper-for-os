"""Fit and evaluate every predictor in src/predictors.py on a chronological
80/20 split of data/processed/features.csv, and write
results/tables/prediction_metrics.csv.

Stage 3 spec (CLAUDE_CODE_PROMPTS.md): split chronologically (never randomly
-- CLAUDE.md rule 3), assert max(train.arrival_time) < min(test.arrival_time)
and raise if that invariant is violated, report MAE / RMSE / under-prediction
rate separately (never combined into one score), and print GBM feature
importances so an implausibly good MAE can be checked for leakage before
being trusted.
"""

import os

import numpy as np
import pandas as pd

from predictors import (
    FEATURE_COLUMNS,
    ExponentialAveraging,
    GBMPredictor,
    Last2,
    LinearRegressionPredictor,
    MedianBaseline,
    Oracle,
)

FEATURES_PATH = os.path.join("data", "processed", "features.csv")
OUTPUT_PATH = os.path.join("results", "tables", "prediction_metrics.csv")
PREDICTIONS_PATH = os.path.join("results", "tables", "test_predictions.csv")
TRAIN_FRACTION = 0.8
SEED = 42
LEAK_SUSPICION_RATIO = 10  # GBM MAE this many times better than Linear MAE is treated as a leak signal


def chronological_split(df: pd.DataFrame, train_frac: float = TRAIN_FRACTION):
    """Split df (already causally ordered by arrival_time) at train_frac.

    The naive index at round(n * train_frac) can land in the middle of a
    cluster of tasks sharing the exact same arrival_time -- this trace has
    bursty simultaneous submissions (see Stage 1/2 notes), so that is a real
    possibility, not a hypothetical. Rather than silently split a tied
    cluster (which would make "train ends before test starts" ambiguous)
    the boundary is pushed forward to the next distinct arrival_time. The
    spec's own hard invariant is still checked and raised on afterwards, as
    a safety net in case this adjustment logic itself has a bug.
    """
    n = len(df)
    split_idx = int(n * train_frac)
    original_split_idx = split_idx
    while split_idx < n and df["arrival_time"].iloc[split_idx] == df["arrival_time"].iloc[split_idx - 1]:
        split_idx += 1
    if split_idx != original_split_idx:
        print(
            f"Adjusted chronological split index from {original_split_idx} to {split_idx} "
            f"to avoid cutting a tied arrival_time cluster."
        )
    if split_idx >= n:
        raise ValueError(
            "Chronological split pushed past the end of the dataset -- the trailing "
            "arrival_time cluster is too large for an 80/20 split."
        )

    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)

    if not (train["arrival_time"].max() < test["arrival_time"].min()):
        raise AssertionError(
            "Chronological split invariant violated: "
            f"max(train.arrival_time)={train['arrival_time'].max()} is not < "
            f"min(test.arrival_time)={test['arrival_time'].min()}"
        )
    return train, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.isnan(y_pred).any():
        raise ValueError(
            f"{int(np.isnan(y_pred).sum())} NaN predictions on the test set -- "
            "investigate before trusting these metrics (see predictors.py's fallback logic)."
        )
    errors = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "under_prediction_rate": float(np.mean(y_pred < y_true)),
    }


def main():
    features = pd.read_csv(FEATURES_PATH).sort_values("arrival_time", kind="mergesort").reset_index(drop=True)
    train, test = chronological_split(features)

    print(
        f"Train: {len(train):,} rows, arrival_time [{train['arrival_time'].min()}, "
        f"{train['arrival_time'].max()}]"
    )
    print(
        f"Test:  {len(test):,} rows, arrival_time [{test['arrival_time'].min()}, "
        f"{test['arrival_time'].max()}]"
    )
    print()

    y_train = train["runtime"]
    y_test = test["runtime"].to_numpy(dtype=float)

    # LinearRegression/GBM cannot fit on NaN features. Stage 2 established
    # that the only NaN feature rows are the earliest arrivals in the whole
    # trace (before anything anywhere had completed) -- verify that belief
    # explicitly rather than assuming it still holds.
    train_nan_mask = train[FEATURE_COLUMNS].isna().any(axis=1)
    test_nan_mask = test[FEATURE_COLUMNS].isna().any(axis=1)
    if test_nan_mask.any():
        raise ValueError(
            f"{int(test_nan_mask.sum())} test rows have NaN features -- unexpected this late "
            "in the trace; investigate before evaluating (see src/features.py)."
        )
    if train_nan_mask.any():
        print(
            f"Dropping {int(train_nan_mask.sum()):,} training rows with NaN features "
            "(no completed-task history existed anywhere yet for these) before fitting "
            "LinearRegression/GBM."
        )
    train_reg = train.loc[~train_nan_mask]
    y_train_reg = y_train.loc[~train_nan_mask]
    print()

    predictors = {
        "Median": MedianBaseline(),
        "ExpAvg": ExponentialAveraging(alpha=0.5),
        "Last2": Last2(),
        "Linear": LinearRegressionPredictor(),
        "GBM": GBMPredictor(seed=SEED),
        "Oracle": Oracle(),
    }

    rows = []
    predictions = pd.DataFrame(
        {"task_id": test["task_id"], "arrival_time": test["arrival_time"], "actual": y_test}
    )
    for name, predictor in predictors.items():
        if name in ("Median", "Linear", "GBM"):
            predictor.fit(train_reg[FEATURE_COLUMNS], y_train_reg)
            preds = predictor.predict(test[FEATURE_COLUMNS])
        else:
            predictor.fit(train, y_train)
            preds = predictor.predict(test)

        predictions[name] = preds
        metrics = compute_metrics(y_test, preds)
        metrics["predictor"] = name
        rows.append(metrics)
        print(
            f"{name:8s} MAE={metrics['mae']:>10.2f}  RMSE={metrics['rmse']:>10.2f}  "
            f"under-prediction rate={metrics['under_prediction_rate']:.2%}"
        )

    results = pd.DataFrame(rows)[["predictor", "mae", "rmse", "under_prediction_rate"]]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}")
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote {PREDICTIONS_PATH} (per-task test predictions, for figures)")

    gbm = predictors["GBM"]
    print("\nGBM feature importances (leakage sanity check):")
    for feat, importance in sorted(gbm.feature_importances().items(), key=lambda kv: -kv[1]):
        print(f"  {feat:<16} {importance:.4f}")

    metrics_by_name = {r["predictor"]: r for r in rows}
    linear_mae = metrics_by_name["Linear"]["mae"]
    gbm_mae = metrics_by_name["GBM"]["mae"]
    oracle_mae = metrics_by_name["Oracle"]["mae"]
    if oracle_mae > 1e-6:
        print(f"\nWARNING: Oracle MAE is {oracle_mae:.4f}, expected ~0 (it returns the true runtime).")
    if linear_mae > 0 and gbm_mae < linear_mae / LEAK_SUSPICION_RATIO:
        print(
            f"\nWARNING: GBM MAE ({gbm_mae:.2f}) is more than {LEAK_SUSPICION_RATIO}x better than "
            f"Linear MAE ({linear_mae:.2f}). Per the Stage 3 spec, treat this as a likely leak "
            "until proven otherwise -- inspect the feature importances above for anything that "
            "should not be knowable at arrival time."
        )


if __name__ == "__main__":
    main()
