"""exp-avg, linear, GBM

Burst/runtime predictors sharing a common interface: fit(X_train, y_train)
and predict(X) -> np.ndarray.

Responsibilities (README.md section 5 / Stage 3):
- ExponentialAveraging(alpha): the textbook Silberschatz formula, using only
  previous_runtime; fit() is a no-op.
- Last2: mean of the same job's two most recent completed runtimes, with a
  global-mean fallback — the Tsafrir et al. (2007) baseline.
- LinearRegressionPredictor: scikit-learn linear regression on the full
  leakage-safe feature set.
- GBMPredictor: XGBoost on the full leakage-safe feature set.
- Oracle: returns the true runtime; used only as the performance ceiling,
  never as a competing predictor.

ON ExponentialAveraging AND Last2 BEING STATEFUL:
The textbook exponential-averaging recurrence (tau_next = alpha*t_last +
(1-alpha)*tau_prev) and "mean of the last k completed same-job runtimes" are
both inherently sequential/online estimators, unlike LinearRegression/GBM
which are ordinary batch regressors over static feature columns. To evaluate
them faithfully, fit(X_train, y_train) stores the training events and
predict(X_test) walks ONE merged arrival/completion timeline of training and
test events from an empty state -- exactly as a real scheduler would keep
updating its running estimate as bursts complete over time. A training task
still running when a test task arrives is not yet observed; an earlier
version replayed every training completion inside fit(), which leaked
future runtimes into test-time state (fixed; see
tests/test_predictor_causality.py).

This is why predict(X) for these two classes (and for Oracle) expects X to
carry a 'runtime' column, unlike LinearRegressionPredictor/GBMPredictor,
which never read 'runtime' out of X. This is NOT a leak: the shared event
walk (see _merged_events) always processes a row's own ARRIVAL event before
its own COMPLETION event, so a row's prediction is fixed at its arrival,
using only OTHER rows' completions that already happened by then -- a row's
own runtime only ever updates state for rows that arrive strictly *after*
it. evaluate_predictors.py passes the same full-column DataFrame to every
predictor's predict(); each predictor chooses which columns it actually
needs.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# The leakage-safe feature set built by src/features.py (Stage 2). Excludes
# task_id, job_id, arrival_time (identifiers) and runtime (the target).
FEATURE_COLUMNS = [
    "requested_cpu",
    "requested_mem",
    "n_instances",
    "hour_of_day",
    "day_of_week",
    "previous_runtime",
    "rolling_mean_3",
    "has_history",
]

_ARRIVAL, _COMPLETION = 0, 1


def _merged_events(df: pd.DataFrame):
    """Return (kinds, rows) for df's arrival/completion timeline, causally
    ordered: primarily by time, then arrivals (0) before completions (1) at
    an identical timestamp, then by original row position for determinism.

    df must have 'arrival_time' and 'runtime' columns. Completion time is
    arrival_time + runtime (arrival_time is a start_time proxy -- see
    src/preprocess.py -- so this matches Stage 1/2's own convention).
    Sorting is done here so callers never need to pre-sort df themselves.
    """
    n = len(df)
    arrival = df["arrival_time"].to_numpy(dtype=float)
    completion = arrival + df["runtime"].to_numpy(dtype=float)
    times = np.concatenate([arrival, completion])
    kinds = np.concatenate([np.full(n, _ARRIVAL), np.full(n, _COMPLETION)])
    rows = np.concatenate([np.arange(n), np.arange(n)])
    order = np.lexsort((rows, kinds, times))  # last key is primary
    return kinds[order], rows[order]


class Predictor:
    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError


def _event_frame(X, y) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": np.asarray(X["job_id"]),
            "arrival_time": np.asarray(X["arrival_time"], dtype=float),
            "runtime": np.asarray(y, dtype=float),
        }
    )


class _SequentialPredictor(Predictor):
    """Shared causal event walk for the online baselines.

    fit() only stores the training events. predict() walks ONE merged
    timeline of training and test events from an empty state, applying each
    completion (train or test) only when it happens, and fixes each test
    task's prediction at its own arrival. A training task that is still
    running when a test task arrives is therefore not yet observed
    (CLAUDE.md rule 2), and predict() is idempotent.

    Subclasses define the per-job state through two pure functions:
    _observe(state, runtime) -> new state, and _estimate(state) -> float|None.
    """

    def fit(self, X_train, y_train):
        self._train_events = _event_frame(X_train, y_train)
        return self

    def predict(self, X) -> np.ndarray:
        test_events = _event_frame(X, X["runtime"])
        combined = pd.concat([self._train_events, test_events], ignore_index=True)
        first_test_row = len(self._train_events)
        return self._walk(combined, first_test_row)[first_test_row:]

    def _observe(self, state, runtime):
        raise NotImplementedError

    def _estimate(self, state):
        raise NotImplementedError

    def _walk(self, df: pd.DataFrame, first_recorded_row: int) -> np.ndarray:
        job_ids = df["job_id"].to_numpy()
        runtimes = df["runtime"].to_numpy(dtype=float)
        kinds, rows = _merged_events(df)
        preds = np.full(len(df), np.nan)
        job_state: dict = {}
        global_sum, global_count = 0.0, 0

        for kind, row in zip(kinds, rows):
            job = job_ids[row]
            if kind == _COMPLETION:
                job_state[job] = self._observe(job_state.get(job), runtimes[row])
                global_sum += runtimes[row]
                global_count += 1
                continue
            if row < first_recorded_row:
                continue
            estimate = self._estimate(job_state.get(job))
            if estimate is not None:
                preds[row] = estimate
            elif global_count:
                preds[row] = global_sum / global_count
            # else: nothing has completed yet anywhere -- leave as NaN.
        return preds


class ExponentialAveraging(_SequentialPredictor):
    """tau_next = alpha * t_last + (1 - alpha) * tau_prev (Silberschatz &
    Galvin's textbook formula), one tau per job.

    Bootstrap choice: the first time a job is observed to complete, tau_prev
    is undefined, so it is seeded to that first observed burst itself (i.e.
    tau after the first completion equals that first runtime exactly). Any
    task whose job has no completed history yet falls back to the running
    global mean, consistent with has_history=False in Stage 2.
    """

    def __init__(self, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def _observe(self, tau_prev, runtime):
        if tau_prev is None:
            return runtime
        return self.alpha * runtime + (1 - self.alpha) * tau_prev

    def _estimate(self, tau):
        return tau


class _WindowedMeanPredictor(_SequentialPredictor):
    """Mean of the last <window> same-job completed runtimes, falling back to
    the running global mean when the job has no completed history yet."""

    def __init__(self, window: int):
        self.window = window

    def _observe(self, history, runtime):
        return ((history or ()) + (runtime,))[-self.window:]

    def _estimate(self, history):
        return sum(history) / len(history) if history else None


class Last2(_WindowedMeanPredictor):
    """Mean of the same job's two most recent completed runtimes, falling
    back to the global mean when the job has no completed history yet --
    the Tsafrir et al. (2007) baseline (LITERATURE.md #13).
    """

    def __init__(self):
        super().__init__(window=2)


# Runtimes are integer seconds >= 1 after Stage 1 filtering; no prediction is
# allowed below this, because a non-positive burst estimate is physically
# meaningless and (Stage 5 showed) turns SRTF into a starvation black hole.
MIN_RUNTIME = 1.0

# Heavily right-skewed inputs, log1p-compressed for the linear model only.
# Trees are invariant to monotonic input transforms, so GBM sees them raw.
SKEWED_FEATURES = ["n_instances", "previous_runtime", "rolling_mean_3"]


def _prepare_target(y, log_target: bool) -> np.ndarray:
    """The runtime target spans five orders of magnitude (median ~9s, 99th
    percentile ~309,000s). Squared-error regression on the raw scale is
    dominated by the rare giants and produced negative predictions for
    ordinary tasks. Modelling log1p(runtime) is a common remedy for
    heavy-tailed positive targets and cannot yield a negative burst. The target is still job-level runtime
    (CLAUDE.md rule 4) -- only the loss scale changes.
    """
    y = np.asarray(y, dtype=float)
    return np.log1p(y) if log_target else y


def _finish_prediction(p, log_target: bool) -> np.ndarray:
    p = np.expm1(p) if log_target else np.asarray(p, dtype=float)
    return np.maximum(p, MIN_RUNTIME)


def _linear_inputs(X: pd.DataFrame) -> pd.DataFrame:
    X = X[FEATURE_COLUMNS].astype(float).copy()
    X[SKEWED_FEATURES] = np.log1p(X[SKEWED_FEATURES])
    return X


class LinearRegressionPredictor(Predictor):
    """Ordinary least squares on log1p(runtime), with skewed inputs
    log1p-compressed and all inputs standardised -- a linear model, unlike a
    tree, is sensitive to input scale and skew.
    """

    def __init__(self, log_target: bool = True):
        self.log_target = log_target
        self.model = make_pipeline(StandardScaler(), LinearRegression())

    def fit(self, X_train, y_train):
        self.model.fit(_linear_inputs(X_train), _prepare_target(y_train, self.log_target))
        return self

    def predict(self, X) -> np.ndarray:
        return _finish_prediction(self.model.predict(_linear_inputs(X)), self.log_target)


class GBMPredictor(Predictor):
    """XGBoost on log1p(runtime). Hyperparameters were chosen on a
    chronological validation split INSIDE the training set (best RMSE/R^2 of
    four candidate configurations), never on the test set.
    """

    def __init__(self, seed: int = 42, log_target: bool = True, **xgb_kwargs):
        self.log_target = log_target
        params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="reg:squarederror",
            random_state=seed,
        )
        params.update(xgb_kwargs)
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train[FEATURE_COLUMNS].astype(float), _prepare_target(y_train, self.log_target))
        return self

    def predict(self, X) -> np.ndarray:
        return _finish_prediction(self.model.predict(X[FEATURE_COLUMNS].astype(float)), self.log_target)

    def feature_importances(self) -> dict:
        return dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))


class MedianBaseline(Predictor):
    """Predicts the training-set median runtime for every task. The constant
    reference that any feature-based model must beat before it can claim to
    have learned anything -- added after validation showed it is competitive
    on this trace slice.
    """

    def __init__(self):
        self.median = None

    def fit(self, X_train, y_train):
        self.median = float(np.median(np.asarray(y_train, dtype=float)))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.median)


class Oracle(Predictor):
    """Returns the true runtime. The performance ceiling, never a competing
    predictor (README.md section 3).
    """

    def fit(self, X_train, y_train):
        return self

    def predict(self, X) -> np.ndarray:
        return X["runtime"].to_numpy(dtype=float)
