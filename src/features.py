"""causal feature construction

Reads data/processed/tasks.csv and writes data/processed/features.csv.

Responsibilities (README.md section 5 / Stage 2, governed by CLAUDE.md):
- Direct features: requested_cpu, requested_mem, n_instances, hour_of_day,
  day_of_week.
- Historical features per job_id: previous_runtime (from the most recent
  same-job task that COMPLETED before this task's arrival_time) and
  rolling_mean_k (k=3) over such tasks, with a global-mean fallback and a
  has_history boolean when no history exists.
- Historical features must be computed in a single forward pass over tasks
  sorted by arrival_time, maintaining per-job state — never via
  groupby().transform() or a full-dataframe rolling window, which would look
  at future rows and leak.

WHY A NAIVE groupby().shift() IS UNSAFE HERE:
Tasks in the same job can run concurrently (a job is a DAG of tasks with many
instances). A task that *arrived* earlier is not guaranteed to have
*completed* before a later-arriving sibling task. Sorting by arrival_time and
shifting within each job_id group silently assumes the previous-arriving
sibling already finished, which is false whenever siblings overlap in time.
This module instead walks a single merged timeline of arrival and completion
events in true time order, so a task's features only ever reflect siblings
(or, for the fallback, any task) whose completion strictly preceded its own
arrival.
"""

import os

import numpy as np
import pandas as pd

INPUT_PATH = os.path.join("data", "processed", "tasks.csv")
OUTPUT_PATH = os.path.join("data", "processed", "features.csv")

ROLLING_K = 3
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400

# Event kinds, used as the sort key's tiebreak: at an identical timestamp,
# arrivals (0) are processed before completions (1), so a completion at the
# same instant as an arrival is NOT visible to that arrival. This enforces
# "completed strictly before arrival_time", not "at or before".
_ARRIVAL, _COMPLETION = 0, 1


def build_features(tasks: pd.DataFrame) -> pd.DataFrame:
    """Compute leakage-safe features for each task via one causal forward pass."""
    tasks = tasks.reset_index(drop=True)
    n = len(tasks)

    events = pd.concat(
        [
            pd.DataFrame({"time": tasks["arrival_time"], "kind": _ARRIVAL, "row": tasks.index}),
            pd.DataFrame({"time": tasks["end_time"], "kind": _COMPLETION, "row": tasks.index}),
        ],
        ignore_index=True,
    ).sort_values(["time", "kind", "row"], kind="mergesort")

    job_ids = tasks["job_id"].to_numpy()
    runtimes = tasks["runtime"].to_numpy(dtype=float)

    # job_id -> list of completed runtimes, in the order they completed.
    job_history: dict = {}
    global_sum = 0.0
    global_count = 0

    previous_runtime = np.full(n, np.nan)
    rolling_mean_3 = np.full(n, np.nan)
    has_history = np.zeros(n, dtype=bool)

    for time, kind, row in events[["time", "kind", "row"]].itertuples(index=False, name=None):
        if kind == _ARRIVAL:
            hist = job_history.get(job_ids[row])
            if hist:
                window = hist[-ROLLING_K:]
                previous_runtime[row] = hist[-1]
                rolling_mean_3[row] = sum(window) / len(window)
                has_history[row] = True
            else:
                fallback = (global_sum / global_count) if global_count else np.nan
                previous_runtime[row] = fallback
                rolling_mean_3[row] = fallback
                has_history[row] = False
        else:  # _COMPLETION
            job_history.setdefault(job_ids[row], []).append(runtimes[row])
            global_sum += runtimes[row]
            global_count += 1

    arrival_time = tasks["arrival_time"]
    return pd.DataFrame(
        {
            "task_id": tasks["task_id"],
            "job_id": tasks["job_id"],
            "arrival_time": arrival_time,
            "runtime": tasks["runtime"],
            # Direct features. Justification (CLAUDE.md rule 6): each is part
            # of the task's own submitted spec or a pure function of
            # arrival_time, so it is known the instant the task arrives.
            "requested_cpu": tasks["requested_cpu"],
            "requested_mem": tasks["requested_mem"],
            "n_instances": tasks["n_instances"],
            "hour_of_day": (arrival_time % SECONDS_PER_DAY) // SECONDS_PER_HOUR,
            "day_of_week": (arrival_time // SECONDS_PER_DAY) % 7,
            # Historical features. Justification: built only from same-job
            # (or, as fallback, any-job) completions with end_time strictly
            # before this task's arrival_time -- see build_features docstring.
            "previous_runtime": previous_runtime,
            "rolling_mean_3": rolling_mean_3,
            "has_history": has_history,
        }
    )


def main():
    tasks = pd.read_csv(INPUT_PATH)
    features = build_features(tasks)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    n_nan = features[["previous_runtime", "rolling_mean_3"]].isna().any(axis=1).sum()
    print(f"Wrote {len(features):,} rows to {OUTPUT_PATH}")
    print(f"has_history=True: {features['has_history'].sum():,} ({features['has_history'].mean():.2%})")
    print(f"Rows with no fallback available yet (global_count==0 at arrival): {n_nan:,}")


if __name__ == "__main__":
    main()
