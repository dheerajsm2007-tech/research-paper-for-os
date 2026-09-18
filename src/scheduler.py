"""discrete-event simulator

A single-CPU, event-driven (not tick-by-tick) scheduling simulator taking a
pluggable Predictor for burst estimates.

Responsibilities (README.md sections 5-6 / Stage 4):
- Implement FCFS, RoundRobin(quantum), and SRTF(predictor). SRTF re-evaluates
  on every arrival; a task's prediction is made once at arrival and not
  revised afterwards.
- Track per task: arrival, first_scheduled, completion; derive waiting,
  turnaround, and response time; count context switches.
- Return a per-task results dataframe plus an aggregate dict.
- Correctness contract: SRTF driven by an Oracle predictor must reproduce
  classical SRTF (true burst times) exactly — this equivalence is the
  correctness test for the whole harness.

DESIGN NOTES:

Each policy has genuinely different preemption rules -- FCFS never preempts,
RoundRobin preempts only at fixed quantum boundaries, SRTF re-evaluates only
at arrival events -- so each is its own independently-reasoned event loop
(_run_fcfs / _run_round_robin / _run_srtf) rather than one generic engine
forced to cover all three. They share only the mechanical, low-risk parts:
results assembly (_finalize) and the context-switch convention (below).

SRTF and predictions: predictions are computed ONCE, up front, via a single
predictor.predict(tasks) call -- reusing the exact same Predictor objects
and batch-predict machinery already validated in src/predictors.py and
src/evaluate_predictors.py, rather than re-implementing per-row inference
inside the event loop. "Predicted once at arrival, never revised" is a
property of the INFORMATION used (each predictor's causal event walk only
ever uses data available at a row's own arrival -- see predictors.py), not
of when the Python call happens to execute. The event loop then tracks two
separate quantities per task: remaining_true (drives when the task actually
finishes in simulated time) and remaining_predicted = predicted_total -
elapsed (drives scheduling decisions only). When the predictor is Oracle,
predicted_total == true burst for every task, so remaining_predicted ==
remaining_true identically -- SRTF-with-Oracle reduces to classical SRTF by
construction, not by coincidence.

CONTEXT SWITCH CONVENTION: a context switch is counted every time the task
beginning a new CPU slice differs from the task that ran in the immediately
preceding slice. The very first slice of a simulation is never counted (there
is nothing to switch from). Idle CPU time does not itself count as a switch,
but does not reset the comparison either -- if task A finishes, the CPU sits
idle, and then task B starts, that is still counted as one switch (A -> B).
"""

import math

import numpy as np
import pandas as pd

EPS = 1e-9


def _sorted_arrays(tasks: pd.DataFrame):
    order = tasks.sort_values(["arrival_time", "task_id"], kind="mergesort").reset_index(drop=True)
    return (
        order["task_id"].to_numpy(),
        order["arrival_time"].to_numpy(dtype=float),
        order["runtime"].to_numpy(dtype=float),
    )


def _finalize(task_ids, arrival, burst, first_scheduled, completion, context_switches):
    assert not np.isnan(completion).any(), "internal error: some task never completed"
    assert not np.isnan(first_scheduled).any(), "internal error: some task never scheduled"

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

    aggregate = {
        "avg_waiting_time": float(df["waiting_time"].mean()),
        "avg_turnaround_time": float(df["turnaround_time"].mean()),
        "avg_response_time": float(df["response_time"].mean()),
        "context_switches": int(context_switches),
    }
    return df, aggregate


def _run_fcfs(tasks: pd.DataFrame):
    task_ids, arrival, burst = _sorted_arrays(tasks)
    n = len(task_ids)

    first_scheduled = np.full(n, np.nan)
    completion = np.full(n, np.nan)
    now = 0.0
    context_switches = 0

    for i in range(n):
        start = max(now, arrival[i])
        first_scheduled[i] = start
        completion[i] = start + burst[i]
        now = completion[i]
        if i > 0:
            context_switches += 1  # every FCFS dispatch after the first is a new task

    return _finalize(task_ids, arrival, burst, first_scheduled, completion, context_switches)


def _run_round_robin(tasks: pd.DataFrame, quantum: float):
    task_ids, arrival, burst = _sorted_arrays(tasks)
    n = len(task_ids)

    remaining = burst.copy()
    first_scheduled = np.full(n, np.nan)
    completion = np.full(n, np.nan)

    from collections import deque

    ready = deque()
    next_idx = 0
    now = 0.0
    completed = 0
    context_switches = 0
    last_run_idx = None

    def enqueue_arrivals_up_to(t):
        nonlocal next_idx
        while next_idx < n and arrival[next_idx] <= t + EPS:
            ready.append(next_idx)
            next_idx += 1

    if next_idx < n:
        now = arrival[0]
    enqueue_arrivals_up_to(now)

    while completed < n:
        if not ready:
            now = arrival[next_idx]
            enqueue_arrivals_up_to(now)
            continue

        idx = ready.popleft()
        if last_run_idx is not None and idx != last_run_idx:
            context_switches += 1
        if np.isnan(first_scheduled[idx]):
            first_scheduled[idx] = now

        run_for = min(quantum, remaining[idx])
        now += run_for
        remaining[idx] -= run_for
        last_run_idx = idx

        # Tasks that arrived during this slice join the queue before the
        # just-preempted task (standard "new arrivals ahead of the
        # round-robin requeue" convention).
        enqueue_arrivals_up_to(now)

        if remaining[idx] <= EPS:
            completion[idx] = now
            completed += 1
        else:
            ready.append(idx)

    return _finalize(task_ids, arrival, burst, first_scheduled, completion, context_switches)


def _run_srtf(tasks: pd.DataFrame, predicted_burst: dict):
    task_ids, arrival, burst = _sorted_arrays(tasks)
    n = len(task_ids)
    predicted_total = np.array([float(predicted_burst[t]) for t in task_ids])

    remaining_true = burst.copy()
    elapsed = np.zeros(n)
    first_scheduled = np.full(n, np.nan)
    completion = np.full(n, np.nan)

    ready = set()
    running = None
    next_idx = 0
    now = 0.0
    completed = 0
    context_switches = 0
    last_run_idx = None

    def enqueue_arrivals_up_to(t):
        nonlocal next_idx
        while next_idx < n and arrival[next_idx] <= t + EPS:
            ready.add(next_idx)
            next_idx += 1

    def remaining_predicted(idx):
        return predicted_total[idx] - elapsed[idx]

    def pick_best(candidates):
        return min(candidates, key=lambda idx: (remaining_predicted(idx), arrival[idx], idx))

    if next_idx < n:
        now = arrival[0]
    enqueue_arrivals_up_to(now)

    while completed < n:
        eligible = set(ready)
        if running is not None:
            eligible.add(running)

        if not eligible:
            now = arrival[next_idx]
            enqueue_arrivals_up_to(now)
            continue

        chosen = pick_best(eligible)

        if chosen != running:
            if running is not None:
                ready.add(running)
            ready.discard(chosen)
            if last_run_idx is not None and chosen != last_run_idx:
                context_switches += 1
            if np.isnan(first_scheduled[chosen]):
                first_scheduled[chosen] = now
            running = chosen

        next_arrival_time = arrival[next_idx] if next_idx < n else math.inf
        slice_len = min(remaining_true[running], next_arrival_time - now)

        now += slice_len
        remaining_true[running] -= slice_len
        elapsed[running] += slice_len
        last_run_idx = running

        enqueue_arrivals_up_to(now)

        if remaining_true[running] <= EPS:
            completion[running] = now
            completed += 1
            running = None

    return _finalize(task_ids, arrival, burst, first_scheduled, completion, context_switches)


class FCFS:
    def run(self, tasks: pd.DataFrame):
        return _run_fcfs(tasks)


class RoundRobin:
    def __init__(self, quantum: float = 4):
        self.quantum = quantum

    def run(self, tasks: pd.DataFrame):
        return _run_round_robin(tasks, self.quantum)


class SRTF:
    """Preemptive shortest-remaining-time-first, re-evaluating only at
    arrival events. `predictor` is any src/predictors.py Predictor; it is
    given the full `tasks` frame once (predictor.predict(tasks)) so every
    predictor -- stateful (ExponentialAveraging, Last2) or batch
    (LinearRegressionPredictor, GBMPredictor, Oracle) -- can use whatever
    columns it needs, exactly as in src/evaluate_predictors.py.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def run(self, tasks: pd.DataFrame):
        predicted = self.predictor.predict(tasks)
        predicted_burst = dict(zip(tasks["task_id"], predicted))
        return _run_srtf(tasks, predicted_burst)
