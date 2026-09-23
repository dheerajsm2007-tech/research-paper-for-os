"""Causality tests for the sequential baselines in src/predictors.py.

CLAUDE.md rule 2: a prediction for task i may use only tasks that COMPLETED
strictly before task i ARRIVED. For ExponentialAveraging and Last2 this must
hold across the train/test boundary too: a training task that is still
running when a test task arrives must not have been "observed" yet.

Synthetic timeline (seconds):

  train  X1 (job X): arrives 0, runtime 1000 -> completes 1000
  train  Y1 (job Y): arrives 0, runtime 2    -> completes 2
  test   X2 (job X): arrives 10
  test   Z1 (job Z): arrives 10
  test   Z2 (job Z): arrives 20   (Z1 completes at 13, before Z2 arrives)

At t=10 only Y1 has completed, so every test task arriving at 10 must see
the global fallback mean = 2.0 and no job-X history (X1 is still running).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictors import ExponentialAveraging, Last2  # noqa: E402

TRAIN = pd.DataFrame(
    {
        "job_id": ["X", "Y"],
        "arrival_time": [0, 0],
        "runtime": [1000.0, 2.0],
    }
)
TEST = pd.DataFrame(
    {
        "job_id": ["X", "Z", "Z"],
        "arrival_time": [10, 10, 20],
        "runtime": [5.0, 3.0, 4.0],
    }
)


@pytest.fixture(params=[ExponentialAveraging, Last2], ids=["ExpAvg", "Last2"])
def fitted(request):
    return request.param().fit(TRAIN, TRAIN["runtime"])


def test_training_task_still_running_at_test_arrival_is_not_observed(fitted):
    preds = fitted.predict(TEST)
    # X2: job X's only task (X1) has not completed at t=10 -> global fallback.
    assert preds[0] == pytest.approx(2.0)
    # Z1: no job-Z history; global mean over completions before t=10 is Y1 only.
    assert preds[1] == pytest.approx(2.0)


def test_test_completions_update_state_causally(fitted):
    preds = fitted.predict(TEST)
    # Z2 arrives at 20; Z1 (runtime 3) completed at 13, so job Z has history
    # [3] and both estimators must predict exactly 3.
    assert preds[2] == pytest.approx(3.0)


def test_predict_is_idempotent(fitted):
    first = fitted.predict(TEST)
    second = fitted.predict(TEST)
    np.testing.assert_array_equal(first, second)
