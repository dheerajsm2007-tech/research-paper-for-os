"""runs the grid, writes results

Orchestrates the full experiment grid and writes results/ outputs.

Responsibilities (README.md sections 5, 7-8 / Stage 5):
- Run all five schedulers (FCFS, RR, SRTF+ExpAvg, SRTF+Last2, SRTF+Linear,
  SRTF+GBM, SRTF+Oracle) over the chronological test split; write
  results/tables/scheduler_comparison.csv.
- Run the sensitivity analysis: inject multiplicative noise into true
  runtimes (bounded uniform at the spec's levels {0, .1, .2, .3, .5}, extended
  upward, plus a log-normal sweep), run SRTF under each level, write
  results/tables/sensitivity*.csv, and identify the crossover point where
  noisy-SRTF's average waiting time exceeds Round Robin's.
- Fix and print all random seeds (single source of truth, per CLAUDE.md).
- Compute the headline recovery percentage:
  recovery = (W_expavg - W_gbm) / (W_expavg - W_oracle) * 100.

CORRECTNESS GATE: SRTF+Oracle must have the lowest avg_waiting_time of every
scheduler in the grid -- this is a mathematical certainty for a correctly
implemented simulator (see src/scheduler.py's oracle-equivalence guarantee
and tests/test_scheduler.py), so a violation here means something upstream
is broken, not a modeling curiosity. This script raises rather than
continuing if that invariant fails, per the Stage 5 spec's own instruction.
"""

import os

import numpy as np
import pandas as pd

from evaluate_predictors import FEATURES_PATH, chronological_split
from predictors import (
    FEATURE_COLUMNS,
    ExponentialAveraging,
    GBMPredictor,
    Last2,
    LinearRegressionPredictor,
    MIN_RUNTIME,
    Oracle,
)
from scheduler import FCFS, RoundRobin, SRTF

SCHEDULER_TABLE_PATH = os.path.join("results", "tables", "scheduler_comparison.csv")
SENSITIVITY_TABLE_PATH = os.path.join("results", "tables", "sensitivity.csv")
SENSITIVITY_LOGNORMAL_TABLE_PATH = os.path.join("results", "tables", "sensitivity_lognormal.csv")

SEED = 42
QUANTUM = 4
# Bounded uniform sweep: the spec's {0, .1, .2, .3, .5}, extended upward because
# the fitted models' errors turned out to lie far beyond +/-50%.
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
# Log-normal multiplicative sweep: predicted = true * exp(sigma * N(0, 1)).
# Symmetric in log space, so it can produce the large under-predictions of
# long tasks that bounded uniform noise structurally cannot -- the failure
# mode the fitted models actually exhibit.
LOGNORMAL_SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


class NoisyOraclePredictor:
    """SENSITIVITY-ANALYSIS TOOL, NOT A REAL PREDICTOR. Reads the test row's
    own true runtime directly and perturbs it:

      model="uniform":   predicted = true * (1 + U(-level, level))
      model="lognormal": predicted = true * exp(level * N(0, 1))

    Predictions are floored at MIN_RUNTIME exactly as the real predictors
    are, so a uniform level above 1 cannot yield a negative burst. This
    intentionally does NOT satisfy CLAUDE.md's leakage-safety rules -- it
    exists only to ask "how does the SCHEDULER behave as prediction error
    grows," independent of whether any real feature-based model could
    achieve that error level. Never use it outside the sensitivity analysis.

    Uses its own numpy Generator (seeded explicitly, not global numpy
    random state) so results are reproducible per CLAUDE.md rule 5.
    """

    def __init__(self, level: float, seed: int, model: str = "uniform"):
        if model not in ("uniform", "lognormal"):
            raise ValueError(f"unknown noise model {model!r}")
        self.level = level
        self.model = model
        self.rng = np.random.default_rng(seed)

    def fit(self, X_train, y_train):
        return self

    def predict(self, X) -> np.ndarray:
        true_runtime = X["runtime"].to_numpy(dtype=float)
        if self.level == 0:
            return true_runtime.copy()
        if self.model == "uniform":
            factor = 1 + self.rng.uniform(-self.level, self.level, size=len(true_runtime))
        else:
            factor = np.exp(self.level * self.rng.standard_normal(size=len(true_runtime)))
        return np.maximum(true_runtime * factor, MIN_RUNTIME)


def run_sweep(model, levels, test, rr_avg_wait, oracle_wait, seed_base, label):
    print(f"Sensitivity analysis -- {label}:")
    rows = []
    for i, level in enumerate(levels):
        sub_seed = seed_base + i
        _, agg = SRTF(NoisyOraclePredictor(level, sub_seed, model)).run(test)
        rows.append({"level": level, "avg_waiting_time": agg["avg_waiting_time"]})
        print(f"  level={level:<5} (seed={sub_seed})  avg_waiting_time={agg['avg_waiting_time']:.2f}")

    # level=0 must exactly reproduce SRTF+Oracle -- internal consistency check.
    if not np.isclose(rows[0]["avg_waiting_time"], oracle_wait):
        raise AssertionError(
            f"level=0 noisy-SRTF avg_waiting_time ({rows[0]['avg_waiting_time']:.4f}) should equal "
            f"SRTF+Oracle ({oracle_wait:.4f}) since zero noise means predicted == true runtime."
        )

    sweep = pd.DataFrame(rows)
    crossed = sweep.loc[sweep["avg_waiting_time"] > rr_avg_wait]
    if crossed.empty:
        print(f"  No crossover: noisy SRTF stays below Round Robin ({rr_avg_wait:.2f}) up to level {levels[-1]}.")
    else:
        first = crossed.iloc[0]
        print(
            f"  Crossover: noisy SRTF first exceeds Round Robin ({rr_avg_wait:.2f}) at "
            f"level={first['level']} (avg_waiting_time={first['avg_waiting_time']:.2f})."
        )
    print()
    return sweep


def build_srtf_variants(train, y_train, train_reg, y_train_reg):
    exp_avg = ExponentialAveraging(alpha=0.5).fit(train, y_train)
    last2 = Last2().fit(train, y_train)
    linear = LinearRegressionPredictor().fit(train_reg[FEATURE_COLUMNS], y_train_reg)
    gbm = GBMPredictor(seed=SEED).fit(train_reg[FEATURE_COLUMNS], y_train_reg)
    oracle = Oracle().fit(train, y_train)
    return {
        "SRTF+ExpAvg": SRTF(exp_avg),
        "SRTF+Last2": SRTF(last2),
        "SRTF+Linear": SRTF(linear),
        "SRTF+GBM": SRTF(gbm),
        "SRTF+Oracle": SRTF(oracle),
    }


def main():
    print(f"Seed: {SEED}")
    print()

    features = pd.read_csv(FEATURES_PATH).sort_values("arrival_time", kind="mergesort").reset_index(drop=True)
    train, test = chronological_split(features)
    y_train = train["runtime"]
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")

    train_nan_mask = train[FEATURE_COLUMNS].isna().any(axis=1)
    test_nan_mask = test[FEATURE_COLUMNS].isna().any(axis=1)
    if test_nan_mask.any():
        raise ValueError(
            f"{int(test_nan_mask.sum())} test rows have NaN features -- investigate before "
            "running the experiment grid (see src/features.py)."
        )
    if train_nan_mask.any():
        print(f"Dropping {int(train_nan_mask.sum()):,} training rows with NaN features for LinearRegression/GBM.")
    train_reg = train.loc[~train_nan_mask]
    y_train_reg = y_train.loc[~train_nan_mask]
    print()

    schedulers = {"FCFS": FCFS(), "RR(q=4)": RoundRobin(quantum=QUANTUM)}
    schedulers.update(build_srtf_variants(train, y_train, train_reg, y_train_reg))

    rows = []
    print("Scheduler comparison (test set):")
    for name, sched in schedulers.items():
        _, agg = sched.run(test)
        rows.append({"scheduler": name, **agg})
        print(
            f"  {name:12s} avg_wait={agg['avg_waiting_time']:>12.2f}  "
            f"avg_turnaround={agg['avg_turnaround_time']:>12.2f}  "
            f"avg_response={agg['avg_response_time']:>12.2f}  "
            f"switches={agg['context_switches']:>7}"
        )

    comparison = pd.DataFrame(rows)[
        ["scheduler", "avg_waiting_time", "avg_turnaround_time", "avg_response_time", "context_switches"]
    ]
    os.makedirs(os.path.dirname(SCHEDULER_TABLE_PATH), exist_ok=True)
    comparison.to_csv(SCHEDULER_TABLE_PATH, index=False)
    print(f"\nWrote {SCHEDULER_TABLE_PATH}")

    # --- Correctness gate (Stage 5 spec's own check) ---
    by_name = comparison.set_index("scheduler")["avg_waiting_time"]
    oracle_wait = by_name["SRTF+Oracle"]
    min_wait = comparison["avg_waiting_time"].min()
    if not np.isclose(oracle_wait, min_wait):
        raise AssertionError(
            f"SRTF+Oracle avg_waiting_time ({oracle_wait:.4f}) is not the minimum across all "
            f"schedulers (min={min_wait:.4f}). A learned or noisy predictor cannot beat the true "
            "burst times -- this is mathematically impossible for a correct simulator, so the "
            "pipeline is broken upstream. Stopping per the Stage 5 spec."
        )
    worst_wait = comparison["avg_waiting_time"].max()
    if by_name["FCFS"] != worst_wait:
        print(
            f"\nNOTE: FCFS ({by_name['FCFS']:.2f}) is not the worst scheduler here "
            f"(worst={worst_wait:.2f}). The spec calls this a soft expectation ('should be worst "
            "or near-worst'), not a hard invariant, so this is reported rather than raised."
        )
    print()

    # --- Sensitivity analysis ---
    rr_avg_wait = by_name["RR(q=4)"]
    uniform = run_sweep(
        "uniform", NOISE_LEVELS, test, rr_avg_wait, oracle_wait, SEED,
        "bounded uniform, predicted = true * (1 + U(-eps, eps))",
    )
    uniform.insert(0, "noise_model", "uniform")
    uniform.to_csv(SENSITIVITY_TABLE_PATH, index=False)
    print(f"Wrote {SENSITIVITY_TABLE_PATH}\n")

    lognormal = run_sweep(
        "lognormal", LOGNORMAL_SIGMAS, test, rr_avg_wait, oracle_wait, SEED + 100,
        "log-normal, predicted = true * exp(sigma * N(0, 1))",
    )
    lognormal.insert(0, "noise_model", "lognormal")
    lognormal.to_csv(SENSITIVITY_LOGNORMAL_TABLE_PATH, index=False)
    print(f"Wrote {SENSITIVITY_LOGNORMAL_TABLE_PATH}\n")

    # Where do the fitted models actually sit on the log-normal axis? Their
    # empirical log-error spread, std(log(predicted / true)), is directly
    # comparable to sigma.
    true_runtime = test["runtime"].to_numpy(dtype=float)
    for name in ("SRTF+Linear", "SRTF+GBM"):
        preds = schedulers[name].predictor.predict(test)
        sigma_hat = float(np.std(np.log(preds / true_runtime)))
        print(f"Empirical log-error sigma of {name.split('+')[1]}: {sigma_hat:.2f} (compare with the log-normal sweep)")
    print()

    # --- Headline recovery number ---
    w_expavg = by_name["SRTF+ExpAvg"]
    w_gbm = by_name["SRTF+GBM"]
    w_oracle = by_name["SRTF+Oracle"]
    denominator = w_expavg - w_oracle
    if abs(denominator) < 1e-9:
        print(
            "WARNING: SRTF+ExpAvg and SRTF+Oracle have (numerically) identical avg waiting time "
            "-- the recovery percentage is undefined (0/0 gap). Skipping."
        )
    else:
        recovery = (w_expavg - w_gbm) / denominator * 100
        print(
            f"Headline: SRTF+GBM recovers {recovery:.1f}% of the waiting-time gap between "
            f"SRTF+ExpAvg ({w_expavg:.2f}) and SRTF+Oracle ({w_oracle:.2f})."
        )


if __name__ == "__main__":
    main()
