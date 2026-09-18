# Learned Burst-Time Prediction for SRTF CPU Scheduling

An empirical study of whether machine-learned burst-time estimates can close the gap
between practical CPU scheduling and the theoretical optimum of Shortest Remaining
Time First.

---

## 1. Problem

Shortest Job First (SJF) and Shortest Remaining Time First (SRTF) are provably optimal
for average waiting time — but only if the scheduler knows each process's CPU burst
length in advance. No real operating system knows this.

The textbook workaround is **exponential averaging**:

```
τ(n+1) = α · t(n) + (1 − α) · τ(n)
```

where `t(n)` is the most recent observed burst and `τ(n)` the previous prediction.
This uses exactly one feature (the last burst) and one hand-tuned constant (`α`).
It is cheap, but it is not accurate.

**Research question.** If the burst-length predictor is replaced by a supervised
regression model trained on richer features available at task-arrival time, how much of
the performance gap between exponential-averaging SRTF and oracle SRTF can be recovered?

---

## 2. Contributions

1. A leakage-safe feature set for burst-time prediction using **only** attributes
   observable at arrival time (no post-execution fields).
2. A comparison of three predictors — exponential averaging (baseline), linear
   regression, and gradient-boosted trees — evaluated on prediction error.
3. Integration of each predictor into a discrete-event scheduling simulator, evaluated
   on *scheduling* metrics rather than prediction metrics alone.
4. A sensitivity analysis quantifying how much prediction error a learned scheduler can
   tolerate before its advantage over Round Robin disappears.

Point 4 is the part reviewers tend to care about most. Prediction accuracy is not the
contribution; the mapping from prediction accuracy to scheduling outcome is.

---

## 3. Experimental design

### Schedulers compared

| Scheduler | Burst knowledge | Role |
|---|---|---|
| FCFS | none | lower-bound reference |
| Round Robin (q = 4) | none | practical baseline |
| SRTF + exponential averaging | predicted, 1 feature | textbook baseline |
| **SRTF + learned predictor** | predicted, n features | proposed |
| SRTF + oracle | true burst | upper bound |

The oracle is not a competitor. It is the ceiling that defines the gap being measured.

### Metrics

- Average waiting time
- Average turnaround time
- Average response time
- Context switch count
- Prediction error: MAE, RMSE, and under-prediction rate

Under-prediction rate matters separately from MAE. A scheduler that systematically
under-estimates bursts behaves very differently from one that over-estimates by the same
magnitude, and the aggregate error metric hides this.

### Headline result format

> Learned-SRTF recovers **X%** of the waiting-time gap between exponential-averaging
> SRTF and oracle SRTF, and retains an advantage over Round Robin up to a prediction
> error of **Y%**.

---

## 4. Data

Primary candidate: **Alibaba Cluster Trace (cluster-trace-v2018)** —
`https://github.com/alibaba/clusterdata`

Alternative: **Google Cluster Traces** (2011 v2 or 2019 v3), or **GWA-T-4 AuverGrid**
from the Grid Workload Archive, which is smaller and used by several prior burst-time
prediction papers.

A subset of roughly 10,000 tasks is sufficient. The full traces are hundreds of GB and
downloading them is a known way to lose three weeks.

### Feature policy — leakage safety

The single most common fatal flaw in this class of paper is training on a feature that
is only knowable after the task has finished. Every feature must pass this test:

> Would the scheduler have this value at the moment the task enters the ready queue?

**Permitted:** requested CPU, requested memory, task priority, arrival timestamp, hour
of day, job/user identifier, number of sibling tasks in the job, rolling mean of the
last k completed bursts from the same user, previous burst from the same user.

**Forbidden:** actual CPU time used, end timestamp, exit status, peak memory observed,
any average computed over a window that includes the target task.

Historical features must be computed causally — for task *i*, only from tasks that
completed strictly before *i* arrived. Train/test splitting must be **chronological**,
not random. A random split lets a model see the future and inflates every number in
the results table.

---

## 5. Repository structure

```
.
├── README.md
├── LITERATURE.md              # annotated bibliography
├── data/
│   ├── raw/                   # downloaded trace slices (gitignored)
│   └── processed/             # tasks.csv — one row per task
├── src/
│   ├── preprocess.py          # trace → tasks.csv
│   ├── features.py            # causal feature construction
│   ├── predictors.py          # exp-avg, linear, GBM
│   ├── scheduler.py           # discrete-event simulator
│   └── experiments.py         # runs the grid, writes results
├── results/
│   ├── tables/
│   └── figures/
├── paper/
│   ├── main.tex
│   └── refs.bib
└── requirements.txt
```

---

## 6. Simulator contract

The simulator is adapted from a standard OS-lab implementation. One structural change
is required: burst length must be supplied by a pluggable predictor rather than read
from the process record.

```python
class Scheduler:
    def __init__(self, predictor):
        self.predictor = predictor      # .predict(task) -> float

    def select_next(self, ready_queue, now):
        ...
```

Setting `predictor` to an oracle that returns the true burst must reproduce classical
SRTF exactly. That equivalence is the correctness test for the whole harness — run it
before trusting any result.

---

## 7. Reproducing

```bash
pip install -r requirements.txt
python src/preprocess.py --trace data/raw/batch_task.csv --n 10000
python src/features.py
python src/experiments.py --all
```

Results land in `results/`. Random seeds are fixed in `experiments.py`.

---

## 8. Schedule

Approximately 45 hours total, spread across 13 weeks.

| Weeks | Phase | Deliverable |
|---|---|---|
| 1–2 | Data | `tasks.csv` with clean burst column |
| 3 | Features + linear baseline | MAE recorded |
| 4 | Gradient boosting | model comparison table |
| 5–6 | Simulator integration | oracle-equivalence test passes |
| 7–8 | Experiments + sensitivity | `results/` populated |
| 9 | Figures | all plots frozen |
| 10–12 | Writing | full draft |
| 13 | Buffer | — |

Results are frozen at week 9. Re-running experiments during the writing phase is the
most reliable way to miss the deadline.

---

## 9. Threats to validity

State these in the paper rather than waiting for a reviewer to raise them.

- **Trace-to-process mismatch.** Cluster traces record job-level runtimes, not
  fine-grained CPU bursts between I/O operations. The prediction target is job runtime;
  the paper must say so explicitly rather than claiming to predict true CPU bursts.
- **Simulation, not deployment.** No kernel integration, so predictor inference cost
  is not charged against scheduling latency. Report model inference time separately.
- **Single trace.** Results may not transfer across workload types. Evaluating on a
  second trace, even briefly, strengthens the paper considerably.
- **Preemption granularity.** SRTF re-evaluates on every arrival; the prediction is
  made once at arrival and not revised.
- **No true arrival/submit timestamp in the trace.** `batch_task.csv` in
  cluster-trace-v2018 records only `start_time` and `end_time` per task — there is no
  submit-time or queue-entry field (see `schema.txt`). `arrival_time` throughout this
  project is therefore `start_time`: the moment a task became runnable, not the moment
  it was submitted. Any queueing delay between submission and start is invisible to
  this dataset, so `hour_of_day` / `day_of_week` and the causal ordering of historical
  features are all computed relative to this proxy, not true submission time. State
  this explicitly rather than implying the trace has genuine arrival timestamps.
- **`hour_of_day` / `day_of_week` are locality indicators here, not periodicity.** The
  first 10,000 tasks span roughly 24 hours, so `day_of_week` only ever takes the values
  0 and 1, and every test-set task has `day_of_week = 1, hour_of_day = 0`. Both are
  legitimately known at arrival (no leakage), but on a slice this short they encode
  "near the train/test boundary" rather than any daily or weekly cycle — and they are
  the fitted GBM's top-ranked features. Either drop them or increase `--n` until the
  slice spans several days before interpreting them as periodicity.
- **The learned predictors barely beat a constant median.** With a log-transformed
  target the models are well-behaved (no negative predictions), but on the test set
  GBM, linear regression and a "predict the training median" baseline all land within
  a few percent of each other on MAE, with R² ≈ 0 for all three. The rare very long
  tasks (11 test tasks over 10,000 s) are feature-indistinguishable from typical
  6-second tasks and are predicted at 11–264 s. Consequently Round Robin, which needs
  no prediction at all, beats every prediction-driven SRTF variant. The paper must
  present the recovery percentage alongside this, not instead of it.
- **Bounded synthetic noise does not reproduce how real models fail.** Injecting
  `true × (1 ± ε)` with ε ≤ 0.5 barely moves SRTF and never crosses Round Robin, yet
  the fitted models sit far above Round Robin — their errors are large
  *under*-predictions of long tasks, which symmetric bounded noise cannot generate.
  The sensitivity sweep is therefore run twice: the bounded-uniform model extended
  well past ε = 0.5, and a log-normal model `true × exp(σ·N(0,1))` that can. Report
  where each fitted model's empirical log-error σ falls on the second curve.
- **Task-level, not job-level, runtime.** In the Alibaba trace a "task" is one stage of
  a job (with many instances). The prediction target is task runtime. The `runtime`
  column keeps its name, but the paper should say "task-level" precisely.

---

## 10. Author

Dheerajkumar Sasikumar — B.Tech AI/ML, VIT University
