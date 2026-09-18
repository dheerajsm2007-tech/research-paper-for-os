"""Correctness tests for src/scheduler.py.

Two independent lines of evidence:

1. A hand-worked 4-process example (the classic Silberschatz & Galvin
   "Operating System Concepts" preemptive-SJF illustration) with literal
   expected average waiting times, computed by hand below and cross-checked
   against the textbook's own published answer for the SRTF case.

2. THE CRITICAL TEST: SRTF driven by an Oracle predictor must exactly match
   a second, independently-written classical SRTF implementation that reads
   true burst times directly (no Predictor abstraction at all). If this
   fails, nothing downstream (Stage 5's headline recovery-percentage number)
   can be trusted, since it depends on SRTF+Oracle being a genuine ceiling.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictors import Oracle  # noqa: E402
from src.scheduler import FCFS, SRTF  # noqa: E402


def _four_process_example():
    # P1 arrives 0, burst 8 | P2 arrives 1, burst 4
    # P3 arrives 2, burst 9 | P4 arrives 3, burst 5
    return pd.DataFrame(
        {
            "task_id": ["P1", "P2", "P3", "P4"],
            "job_id": ["P1", "P2", "P3", "P4"],
            "arrival_time": [0, 1, 2, 3],
            "runtime": [8, 4, 9, 5],
        }
    )


def test_fcfs_matches_hand_worked_average_waiting_time():
    # By hand, in arrival order (FCFS never preempts):
    #   P1: start=0,  completion=0+8=8   -> turnaround=8,  waiting=8-8=0
    #   P2: start=8,  completion=8+4=12  -> turnaround=11, waiting=11-4=7
    #   P3: start=12, completion=12+9=21 -> turnaround=19, waiting=19-9=10
    #   P4: start=21, completion=21+5=26 -> turnaround=23, waiting=23-5=18
    # avg waiting = (0+7+10+18)/4 = 35/4 = 8.75
    tasks = _four_process_example()
    results, aggregate = FCFS().run(tasks)

    results = results.set_index("task_id")
    assert results.loc["P1", "completion"] == 8
    assert results.loc["P2", "completion"] == 12
    assert results.loc["P3", "completion"] == 21
    assert results.loc["P4", "completion"] == 26
    assert aggregate["avg_waiting_time"] == pytest.approx(8.75)
    assert aggregate["context_switches"] == 3  # 4 dispatches, first one free


def test_srtf_matches_hand_worked_average_waiting_time():
    # By hand (preemptive, re-evaluating remaining burst at each arrival):
    #   t=0: only P1 ready (rem 8) -> run P1.
    #   t=1: P2 arrives (rem 4) < P1's remaining (7) -> preempt, run P2.
    #   t=2: P3 arrives (rem 9). Ready remaining: P1=7, P2=3, P3=9 -> keep P2.
    #   t=3: P4 arrives (rem 5). Ready remaining: P1=7, P2=2, P3=9, P4=5 -> keep P2.
    #   t=5: P2 completes (ran 1+1+1+2 = 4 total, started at t=1).
    #        Remaining: P1=7, P3=9, P4=5 -> run P4 (no more arrivals pending).
    #   t=10: P4 completes -> remaining: P1=7, P3=9 -> run P1.
    #   t=17: P1 completes -> remaining: P3=9 -> run P3.
    #   t=26: P3 completes.
    # waiting = turnaround - burst:
    #   P1: turnaround=17-0=17, waiting=17-8=9
    #   P2: turnaround=5-1=4,   waiting=4-4=0
    #   P3: turnaround=26-2=24, waiting=24-9=15
    #   P4: turnaround=10-3=7,  waiting=7-5=2
    # avg waiting = (9+0+15+2)/4 = 26/4 = 6.5
    # This matches the published textbook answer for this exact example.
    tasks = _four_process_example()
    results, aggregate = SRTF(Oracle()).run(tasks)

    results = results.set_index("task_id")
    assert results.loc["P1", "completion"] == 17
    assert results.loc["P2", "completion"] == 5
    assert results.loc["P3", "completion"] == 26
    assert results.loc["P4", "completion"] == 10
    assert results.loc["P2", "first_scheduled"] == 1  # P2 preempts P1 immediately on arrival
    assert results.loc["P3", "first_scheduled"] == 17  # P3 never runs until everything else is done
    assert aggregate["avg_waiting_time"] == pytest.approx(6.5)

    # SRTF should never do worse than FCFS on average waiting time for the
    # same instance -- a basic sanity property, not just a coincidence of
    # this particular example.
    _, fcfs_aggregate = FCFS().run(tasks)
    assert aggregate["avg_waiting_time"] < fcfs_aggregate["avg_waiting_time"]


def _classical_srtf(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Independent, deliberately simple (O(n^2)-ish) reference SRTF
    implementation using TRUE burst times directly -- no Predictor, no
    src/scheduler.py internals shared with the implementation under test.
    Used only to verify SRTF(Oracle()) produces identical per-task metrics.
    """
    order = tasks_df.sort_values(["arrival_time", "task_id"], kind="mergesort").reset_index(drop=True)
    n = len(order)
    arrival = order["arrival_time"].to_numpy(dtype=float)
    burst = order["runtime"].to_numpy(dtype=float)
    task_ids = order["task_id"].to_numpy()

    remaining = burst.copy()
    first_scheduled = np.full(n, np.nan)
    completion = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)

    now = 0.0
    completed = 0
    while completed < n:
        eligible = [i for i in range(n) if arrival[i] <= now + 1e-9 and not done[i]]
        if not eligible:
            now = min(arrival[i] for i in range(n) if not done[i] and arrival[i] > now)
            continue

        idx = min(eligible, key=lambda i: (remaining[i], arrival[i], i))
        if np.isnan(first_scheduled[idx]):
            first_scheduled[idx] = now

        future_arrivals = [arrival[i] for i in range(n) if not done[i] and arrival[i] > now]
        next_arrival = min(future_arrivals) if future_arrivals else math.inf
        step = min(remaining[idx], next_arrival - now)

        now += step
        remaining[idx] -= step
        if remaining[idx] <= 1e-9:
            completion[idx] = now
            done[idx] = True
            completed += 1

    df = pd.DataFrame(
        {
            "task_id": task_ids,
            "arrival": arrival,
            "burst": burst,
            "first_scheduled": first_scheduled,
            "completion": completion,
        }
    )
    df["waiting_time"] = (df["completion"] - df["arrival"]) - df["burst"]
    df["turnaround_time"] = df["completion"] - df["arrival"]
    df["response_time"] = df["first_scheduled"] - df["arrival"]
    return df.sort_values("task_id").reset_index(drop=True)


def _random_tasks(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arrival = np.sort(rng.integers(0, 100, size=n))
    burst = rng.integers(1, 30, size=n)
    task_ids = [f"T{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "task_id": task_ids,
            "job_id": task_ids,
            "arrival_time": arrival,
            "runtime": burst,
        }
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_srtf_oracle_equals_classical_srtf(seed):
    tasks = _random_tasks(seed=seed, n=25)

    oracle_results, _ = SRTF(Oracle()).run(tasks)
    oracle_results = oracle_results.sort_values("task_id").reset_index(drop=True)

    classical_results = _classical_srtf(tasks)

    for column in ["first_scheduled", "completion", "waiting_time", "turnaround_time", "response_time"]:
        np.testing.assert_allclose(
            oracle_results[column].to_numpy(),
            classical_results[column].to_numpy(),
            err_msg=f"SRTF(Oracle()) disagrees with classical SRTF on '{column}' (seed={seed})",
        )


def test_srtf_oracle_equals_classical_srtf_on_four_process_example():
    # Same check on the hand-worked example itself, for good measure.
    tasks = _four_process_example()
    oracle_results, _ = SRTF(Oracle()).run(tasks)
    oracle_results = oracle_results.sort_values("task_id").reset_index(drop=True)
    classical_results = _classical_srtf(tasks)

    for column in ["first_scheduled", "completion", "waiting_time", "turnaround_time", "response_time"]:
        np.testing.assert_allclose(
            oracle_results[column].to_numpy(), classical_results[column].to_numpy()
        )
