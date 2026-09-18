"""Causality tests for src/features.py.

Synthetic timeline (all times in seconds, arrival_time == start_time as in
tasks.csv). Runtime = end_time - start_time in every row, matching Stage 1's
invariant.

  time:    0    1    3    5    10   20   25
  C1  A---C(1)
  A1  A--------------------C(10)
  B1            A----------------C(7)
  A2                 A----------------------C(15)
  A3                                       A----C(5)

  (A = arrival, C = completion, label = runtime)

- C1 (job C): arrives 0, runtime 1, completes 1. First-ever completion.
- A1 (job A): arrives 0, runtime 10, completes 10.
- B1 (job B): arrives 3, runtime 7, completes 10. Job B has no prior task, so
  it must fall back to the global mean of everything completed before t=3,
  which is only C1 (runtime 1) -> expected fallback = 1.0.
- A2 (job A): arrives 5, runtime 15, completes 20. A1 (job A's only other
  task) arrived earlier but has NOT completed yet at t=5 (it completes at
  t=10). This is the overlap trap: a naive groupby().shift() sorted by
  arrival would hand A2 A1's runtime (10). The correct answer is the global
  fallback again: only C1 has completed before t=5 -> expected fallback = 1.0.
- A3 (job A): arrives 25. By now A1 (completed 10, runtime 10) and A2
  (completed 20, runtime 15) have both completed, in that order. Expected
  previous_runtime = 15 (A2, the most recent), rolling_mean_3 = mean(10, 15)
  = 12.5 (only 2 available, k=3 is a cap not a requirement), has_history=True.

A second, later, non-overlapping job D (all times shifted well past the
above block) isolates the rolling_mean_3 k=3 cap: five sequential tasks
where only the most recent 3 of 4 completed runtimes should be averaged.

  D1 arrives 1000, runtime 1, completes 1001
  D2 arrives 1002, runtime 2, completes 1004
  D3 arrives 1005, runtime 3, completes 1008
  D4 arrives 1009, runtime 4, completes 1013
  D5 arrives 1014 -> job D history at arrival = [1, 2, 3, 4] (completion
     order). previous_runtime = 4 (D4, most recent). rolling_mean_3 must use
     only the last 3 -> mean(2, 3, 4) = 3.0, NOT mean(1, 2, 3, 4) = 2.5.

D1 additionally checks that the global fallback accumulates across *all*
earlier jobs, not just C: by t=1000 everything above (C1, A1, B1, A2, A3) has
completed, runtimes [1, 10, 7, 15, 5], mean = 38 / 5 = 7.6.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import build_features  # noqa: E402

COMMON = {"requested_cpu": 50.0, "requested_mem": 50.0, "n_instances": 1}


def _task(task_id, job_id, arrival, runtime):
    return {
        "task_id": task_id,
        "job_id": job_id,
        "arrival_time": arrival,
        "start_time": arrival,
        "end_time": arrival + runtime,
        "runtime": runtime,
        **COMMON,
    }


def _synthetic_tasks():
    rows = [
        _task("C1", "C", 0, 1),
        _task("A1", "A", 0, 10),
        _task("B1", "B", 3, 7),
        _task("A2", "A", 5, 15),
        _task("A3", "A", 25, 5),
        _task("D1", "D", 1000, 1),
        _task("D2", "D", 1002, 2),
        _task("D3", "D", 1005, 3),
        _task("D4", "D", 1009, 4),
        _task("D5", "D", 1014, 99),
    ]
    return pd.DataFrame(rows)


def _row(features, task_id):
    match = features.loc[features["task_id"] == task_id]
    assert len(match) == 1, f"expected exactly one row for {task_id}"
    return match.iloc[0]


def test_previous_runtime_uses_correct_earlier_completed_task():
    features = build_features(_synthetic_tasks())

    a3 = _row(features, "A3")
    assert a3["previous_runtime"] == 15.0  # A2 completed (t=20) most recently before A3 arrives (t=25)
    assert a3["rolling_mean_3"] == 12.5  # mean(A1=10, A2=15); only 2 completions exist
    assert bool(a3["has_history"]) is True

    d5 = _row(features, "D5")
    assert d5["previous_runtime"] == 4.0  # D4 is the most recently completed sibling
    assert d5["rolling_mean_3"] == 3.0  # mean(D2=2, D3=3, D4=4) -- last 3 only, D1=1 excluded
    assert bool(d5["has_history"]) is True


def test_overlapping_sibling_that_completes_after_arrival_uses_fallback_not_its_runtime():
    features = build_features(_synthetic_tasks())

    # A1 (runtime 10) arrives before A2 but does not COMPLETE until t=10,
    # which is after A2 arrives at t=5. A2 must therefore NOT see A1's
    # runtime -- it must fall back to the global mean of tasks completed
    # strictly before t=5, which is only C1 (runtime 1).
    a2 = _row(features, "A2")
    assert a2["previous_runtime"] == 1.0
    assert a2["previous_runtime"] != 10.0  # would be A1's runtime under a naive groupby().shift()
    assert a2["rolling_mean_3"] == 1.0
    assert bool(a2["has_history"]) is False


def test_global_fallback_accumulates_across_all_jobs():
    features = build_features(_synthetic_tasks())

    # B1 is job B's first-ever task: falls back to the global mean of
    # everything completed before t=3, which is only C1 (runtime 1).
    b1 = _row(features, "B1")
    assert b1["previous_runtime"] == 1.0
    assert bool(b1["has_history"]) is False

    # D1 is job D's first-ever task, arriving long after C1, A1, B1, A2, A3
    # have all completed (runtimes 1, 10, 7, 15, 5 -> mean 7.6).
    d1 = _row(features, "D1")
    assert np.isclose(d1["previous_runtime"], 7.6)
    assert bool(d1["has_history"]) is False


def test_first_ever_task_with_no_prior_completions_gets_nan_fallback():
    # C1 is the very first event in the whole trace: no task has completed
    # yet, so even the global fallback is undefined.
    features = build_features(_synthetic_tasks())
    c1 = _row(features, "C1")
    assert np.isnan(c1["previous_runtime"])
    assert np.isnan(c1["rolling_mean_3"])
    assert bool(c1["has_history"]) is False
