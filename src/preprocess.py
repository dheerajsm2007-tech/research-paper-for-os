"""trace -> tasks.csv

Reads a raw Alibaba cluster-trace-v2018 batch_task file in chunks and writes
data/processed/tasks.csv, one row per task, kept in arrival-time order.

Responsibilities (README.md section 5 / Stage 1):
- Parse --trace, --n (default 10000), --seed.
- Derive runtime = end_time - start_time; drop rows with missing timestamps,
  runtime <= 0, or tasks that did not terminate successfully.
- Take the first --n tasks chronologically (no random sampling), since the
  chronological train/test split later depends on this ordering.
- Print inferred column names/dtypes before filtering, and a summary of row
  counts and drop percentages after.

KNOWN LIMITATION (read before trusting arrival_time):
cluster-trace-v2018's batch_task table has no submit/queue timestamp field —
see https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt.
Only start_time and end_time are recorded per task. This script therefore uses
start_time as an arrival_time PROXY: the true queueing delay between a task's
submission and its start is not observable in this trace, so arrival_time here
actually means "the task became runnable," not "the task was submitted." This
should be stated as a threat to validity in the paper (README.md section 9).
"""

import argparse
import os
import sys

import pandas as pd

# Official column order for cluster-trace-v2018 batch_task.csv (the raw file has
# no header row). Source: cluster-trace-v2018/schema.txt in alibaba/clusterdata.
COLUMNS = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]

OUTPUT_PATH = os.path.join("data", "processed", "tasks.csv")

CHUNK_SIZE = 500_000
SCHEMA_PREVIEW_ROWS = 100_000

# Successful-completion marker for the `status` field. Rows with any other
# status (e.g. Failed, Cancelled, Interrupted, Running, Waiting) are dropped
# as "did not terminate successfully" per the Stage 1 spec.
SUCCESS_STATUS = "Terminated"

# How large the buffered candidate set is allowed to grow (as a multiple of
# --n) before it is trimmed back down. Keeps peak memory bounded regardless
# of trace size, instead of accumulating every surviving row before sorting.
CANDIDATE_TRIM_MULTIPLE = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess an Alibaba cluster-trace-v2018 batch_task file into tasks.csv"
    )
    parser.add_argument("--trace", required=True, help="Path to the raw batch_task.csv file")
    parser.add_argument(
        "--n", type=int, default=10000,
        help="Number of tasks to keep, taken chronologically by arrival_time (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed, recorded for reproducibility. Preprocessing itself is "
             "deterministic and never samples randomly (CLAUDE.md rule 3); this flag "
             "exists so the seed is declared here alongside the other stages.",
    )
    return parser.parse_args()


def print_inferred_schema(trace_path):
    sample = pd.read_csv(trace_path, header=None, names=COLUMNS, nrows=SCHEMA_PREVIEW_ROWS)
    print(f"Inferred columns and dtypes (first {SCHEMA_PREVIEW_ROWS:,} rows, before any filtering):")
    for col in COLUMNS:
        print(f"  {col:<14} {sample[col].dtype}")
    print("Compare this against cluster-trace-v2018/schema.txt's `batch task` table before proceeding.")
    print()


def trim_candidates(candidates, n):
    """Collapse buffered chunks to the n smallest-arrival_time rows seen so far."""
    merged = pd.concat(candidates, ignore_index=True)
    trimmed = merged.nsmallest(n, "arrival_time")
    return [trimmed]


def main():
    args = parse_args()

    if args.n <= 0:
        print(f"--n must be positive, got {args.n}", file=sys.stderr)
        sys.exit(1)

    print(f"Seed: {args.seed} (not used for sampling here; declared for consistency with later stages)")
    print()

    print_inferred_schema(args.trace)

    total_rows = 0
    dropped_missing_ts = 0
    dropped_nonpositive_runtime = 0
    dropped_not_terminated = 0

    candidates = []
    candidate_row_count = 0
    trim_threshold = max(args.n * CANDIDATE_TRIM_MULTIPLE, args.n + 1)

    reader = pd.read_csv(args.trace, header=None, names=COLUMNS, chunksize=CHUNK_SIZE)
    for chunk in reader:
        total_rows += len(chunk)

        missing_ts = chunk["start_time"].isna() | chunk["end_time"].isna()
        dropped_missing_ts += int(missing_ts.sum())
        chunk = chunk.loc[~missing_ts]

        runtime = chunk["end_time"] - chunk["start_time"]
        nonpositive_runtime = runtime <= 0
        dropped_nonpositive_runtime += int(nonpositive_runtime.sum())
        chunk = chunk.loc[~nonpositive_runtime]
        runtime = runtime.loc[chunk.index]

        not_terminated = chunk["status"] != SUCCESS_STATUS
        dropped_not_terminated += int(not_terminated.sum())
        chunk = chunk.loc[~not_terminated]
        runtime = runtime.loc[chunk.index]

        if chunk.empty:
            continue

        chunk = chunk.assign(runtime=runtime, arrival_time=chunk["start_time"])
        candidates.append(chunk)
        candidate_row_count += len(chunk)

        if candidate_row_count > trim_threshold:
            candidates = trim_candidates(candidates, args.n)
            candidate_row_count = len(candidates[0])

    surviving = total_rows - dropped_missing_ts - dropped_nonpositive_runtime - dropped_not_terminated

    if not candidates:
        print("No rows survived filtering -- aborting.", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(candidates, ignore_index=True)
    n = min(args.n, len(merged))
    tasks = (
        merged.nsmallest(n, "arrival_time")
        .sort_values(["arrival_time", "job_name", "task_name"], kind="mergesort")
        .reset_index(drop=True)
    )

    tasks["task_id"] = tasks["job_name"] + "::" + tasks["task_name"]
    tasks = tasks.rename(columns={
        "job_name": "job_id",
        "plan_cpu": "requested_cpu",
        "plan_mem": "requested_mem",
        "instance_num": "n_instances",
    })

    out_columns = [
        "task_id", "job_id", "arrival_time", "start_time", "end_time", "runtime",
        "requested_cpu", "requested_mem", "n_instances", "task_type",
    ]
    tasks = tasks[out_columns]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tasks.to_csv(OUTPUT_PATH, index=False)

    def pct(x):
        return f"{x / total_rows:.2%}" if total_rows else "n/a"

    print("Filtering summary (percentages are of total rows read from the trace):")
    print(f"  Rows read from trace:                 {total_rows:,}")
    print(f"  Dropped (missing start/end time):     {dropped_missing_ts:,} ({pct(dropped_missing_ts)})")
    print(f"  Dropped (runtime <= 0):                {dropped_nonpositive_runtime:,} ({pct(dropped_nonpositive_runtime)})")
    print(f"  Dropped (status != '{SUCCESS_STATUS}'):        {dropped_not_terminated:,} ({pct(dropped_not_terminated)})")
    print(f"  Rows surviving all filters:            {surviving:,} ({pct(surviving)})")
    print(f"  Tasks kept (--n={args.n}, chronological): {n:,}")
    print()

    print("Runtime distribution (seconds) over kept tasks:")
    print(f"  min:    {tasks['runtime'].min()}")
    print(f"  median: {tasks['runtime'].median()}")
    print(f"  mean:   {tasks['runtime'].mean():.2f}")
    print(f"  max:    {tasks['runtime'].max()}")
    print()

    print(f"Wrote {len(tasks):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
