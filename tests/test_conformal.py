"""
Unit tests for Split Conformal Calibrator and finite-sample coverage guarantees.
"""

import numpy as np

from fraudsense.models.conformal import SplitConformalCalibrator


def test_split_conformal_calibration_and_set_generation():
    """Verify nonconformity quantile calibration and valid prediction set creation."""
    np.random.seed(42)
    n_cal = 500
    # Simulate predicted probabilities with reasonable confidence
    probs_cal = np.random.dirichlet([5, 2, 1], size=n_cal)
    y_cal = np.random.choice([0, 1, 2], size=n_cal, p=[0.7, 0.2, 0.1])

    calibrator = SplitConformalCalibrator()
    q = calibrator.calibrate(probs_cal, y_cal)

    assert 0.0 <= q <= 1.0
    assert calibrator.is_calibrated

    # Generate prediction set for a single test point
    test_prob = np.array([0.8, 0.15, 0.05])
    pred_set = calibrator.predict_set(test_prob)

    assert isinstance(pred_set, set)
    assert len(pred_set) >= 1
    assert pred_set.issubset({"ALLOW", "CHALLENGE", "FRAUD"})


def test_split_conformal_empirical_coverage():
    """Verify empirical coverage meets or exceeds nominal target coverage (90%)."""
    np.random.seed(42)
    n = 2000
    # Simulate informative probabilities
    y = np.random.choice(3, size=n, p=[0.7, 0.2, 0.1])
    probs = np.zeros((n, 3))
    for i in range(n):
        # High probability on true class, lower on others
        noise = np.random.uniform(0.01, 0.1, size=3)
        probs[i] = noise
        probs[i, y[i]] += np.random.uniform(0.6, 0.9)
        probs[i] /= probs[i].sum()

    calibrator = SplitConformalCalibrator()
    calibrator.calibrate(probs[:1000], y[:1000])

    eval_res = calibrator.evaluate_coverage(probs[1000:], y[1000:])

    # Check finite sample coverage is near or above 90%
    assert eval_res["empirical_coverage"] >= 0.89
    assert eval_res["target_coverage"] == 0.90
    assert eval_res["mean_set_size"] >= 1.0
