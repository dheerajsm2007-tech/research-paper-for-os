# Notes for the sections you write yourself

> **Status (2026-09-23):** every section is now drafted in `main.tex`. These notes
> remain as the argument outline. Two corrections to the notes below: (1) the
> exponential-averaging and Last2 baselines previously leaked future training
> completions into test-time state; after the fix (see
> `tests/test_predictor_causality.py`) they under-predict about 18.5% of test tasks,
> not "almost never", and the headline moved from 59.0% to 58.5%; (2) GBM beats
> linear regression in scheduling because it ranks tasks better (Spearman 0.66 vs
> 0.09), not because of MAE. Before submission: verify each `% VERIFY` bib entry,
> fill the missing authors/venues BibTeX warns about, and check the
> `VERIFY-CLAIM` characterisations in Section II against the full texts.


Arguments only, derived from `results/tables/`. Not prose to paste. Every
number below is in a table; cite the table, do not retype the number.

## I. Introduction

- **The gap is real and large, but it is not where the textbook says it is.** On
  this workload oracle SRTF beats exponential-averaging SRTF by an order of
  magnitude in mean waiting time (Table: schedulers), so a better estimator has
  a lot of room. But round robin, which uses no estimate at all, also beats
  exponential-averaging SRTF by a wide margin. The interesting question is
  therefore not "can ML beat exponential averaging" (it can, easily) but "can
  any arrival-time predictor make SRTF competitive with a prediction-free
  policy".
- **Contribution list (draft §I.D before anything else):** (1) a leakage-safe,
  causally-tested feature pipeline on a real production trace, where the causal
  test demonstrably fails under the conventional grouped-shift construction;
  (2) an end-to-end mapping from prediction error to scheduling outcome — the
  recovery percentage — rather than prediction error alone; (3) a two-model
  sensitivity analysis (bounded uniform and log-normal) that locates the error
  level at which SRTF loses to round robin and shows that fitted-model error is
  structured, not random. If you cannot defend all three as specific, the paper
  does not yet have a result.
- **State the negative result up front, not as a caveat.** No prediction-driven
  SRTF beat round robin. Framing that as a limitation invites the reviewer to
  ask why the paper exists; framing it as the finding ("arrival-time features
  on this trace do not carry the information SRTF needs") is what makes the
  sensitivity analysis matter.

## II. Related Work

- **Position against Helmy et al. (2015) and Effah et al. (2025) on the axis
  they omit: scheduling outcome.** Both report prediction accuracy on grid data
  (Effah on a synthetic imitation of GWA-T-4). This paper reports prediction
  accuracy *and* the waiting time that accuracy buys, on a real trace, with a
  chronological split. You must actually read both to say precisely what they
  measured.
- **Borrow the evaluation stance of the HPC backfilling literature.** Tsafrir et
  al. (2007) is the source of the Last2 baseline and of the idea that
  system-generated predictions can replace user estimates; Chiang et al. (2002)
  established that better estimates improve scheduling, which is the premise the
  recovery metric depends on; TARE (2026) argues mean-based metrics hide tail
  behaviour, which is exactly why the under-prediction rate is reported
  separately. Say why single-CPU SRTF is a harder consumer of predictions than
  backfilling: one bad under-prediction becomes an unpreemptable task.
- **Explain why this is not a reinforcement-learning paper.** DeepRM, Decima,
  RLScheduler learn the policy; this paper keeps the (provably optimal, given
  the inputs) policy fixed and learns only its input, so the ceiling is known
  and the gap is measurable. That is the reason an oracle bound exists here and
  not in the RL work.

## VI. Results and Discussion

- **Why GBM recovers only part of the gap, and why MAE does not explain it.**
  In Table: prediction, GBM, linear regression and the constant median are
  within a few percent of each other on MAE and RMSE, yet in Table: schedulers
  their SRTF waiting times differ by a factor of about four. The scheduling
  outcome is governed by a handful of very long tasks, not by aggregate error:
  the scatter figure shows predictions saturating around 10–300 s regardless of
  actual runtime, and the largest test tasks are feature-indistinguishable from
  median tasks. Argue that MAE is the wrong yardstick for a scheduler, and that
  the recovery percentage is the right one.
- **What the crossover means for a real scheduler.** The log-normal sweep loses
  to round robin at σ = 2, and the fitted GBM's own σ is about 1.7 — just inside
  the safe region — yet its actual waiting time is roughly three times what the
  σ = 1.7 noise curve gives. The fitted error is *structured* (long tasks are
  systematically under-predicted), and structured error of a given spread hurts
  SRTF far more than random error of the same spread. Consequence: a deployment
  decision cannot be made from a prediction-error budget alone; it needs the
  error's correlation with runtime. This is the paragraph a reviewer will quote.
- **What the under-prediction rate implies about starvation.** Under
  (eq. srtf-pred) an under-predicted task's predicted remaining service goes
  negative and it can never be preempted. Exponential averaging and Last2
  almost never under-predict (their under-prediction rate is tiny because they
  predict close to the global mean), which is why they are *safer* than linear
  regression despite far worse MAE. The design flaw is "predict once, never
  revise": any deployment should revise the estimate once elapsed time exceeds
  it. That is the concrete engineering recommendation of the paper.

## VII. Threats to Validity (not drafted; README.md §9 has the full list)

- Arrival time is `start_time` — queueing delay before start is invisible, so
  "arrival" means "became runnable".
- The target is task-level runtime (one stage of a job), not a CPU burst and
  not, strictly, a whole job.
- The 10,000-task slice spans about 24 hours, so `hour_of_day`/`day_of_week`
  act as locality indicators, not periodicity — and they are the top GBM
  features. A longer slice is the obvious follow-up.
- Single trace; simulation, not deployment; inference cost not charged.

## VIII. Conclusion and Future Work

- **The honest summary:** learned prediction recovers a majority of the
  ExpAvg-to-oracle gap but does not make SRTF competitive with round robin on
  this trace, because the information SRTF needs — which rare tasks are the
  giants — is not present in arrival-time features.
- **The transferable result** is methodological: the recovery metric, the
  causal feature test, the oracle-equivalence test, and the structured-vs-random
  error contrast. Those survive a change of trace; the 58.5% number does not.
- **Future work that follows directly from the tables:** (1) revise predictions
  when elapsed time exceeds them (closes the starvation hole); (2) evaluate on a
  slice spanning several days and on a second trace (Google 2019 or GWA-T-4) so
  temporal features can mean something; (3) quantile or classification
  predictors that target "is this task a giant", since that is the decision
  the scheduler actually needs.

## Housekeeping

- Fig. 2 is a two-panel figure and may read better as `figure*` (full width);
  the spec asked for `\columnwidth`, so it is single-column for now.
- `[CITATION NEEDED]` markers in Sections III–V: SRPT optimality proof;
  Silberschatz et al. for exponential averaging; a prior log-runtime target;
  scikit-learn. None of these are in LITERATURE.md, so none were invented.
- Every `refs.bib` entry still carries a `% VERIFY:` line. Do not submit until
  each has been resolved or deleted.
