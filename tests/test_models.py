"""Change-point and linear model behaviour on synthetic data with a known balance point."""

import numpy as np

from euss_cooling.models import fit_changepoint, fit_linear


def _synthetic(bp=70.0, slope=0.5, base=0.1, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(50, 95, n)
    y = base + slope * np.maximum(x - bp, 0.0) + rng.normal(0, 0.02, n)
    return x, y


def test_changepoint_recovers_balance_point_and_slope():
    x, y = _synthetic(bp=70.0, slope=0.5)
    fit = fit_changepoint(x, y, grid=(55.0, 82.0, 0.5))
    assert abs(fit.params["balance_point"] - 70.0) <= 1.0
    assert abs(fit.params["slope_per_deg"] - 0.5) <= 0.05
    assert fit.params["slope_per_deg"] > 0


def test_linear_slope_positive_and_predicts():
    x, y = _synthetic()
    fit = fit_linear(x, y)
    assert fit.params["slope_per_deg"] > 0
    assert np.isfinite(fit.predict(np.array([60.0, 90.0]))).all()
