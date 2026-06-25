"""Three models of cooling electricity (kWh/hour) vs outdoor temperature.

All temperatures are in the configured units (default deg F), so slopes are kWh per deg F.

* ``fit_linear``       -- single-slope OLS baseline.
* ``fit_changepoint``  -- piecewise-linear: flat ``base_load`` below a cooling ``balance_point``
  temperature, then a ``slope`` above it. The balance point is found by grid search; this is the
  headline interpretable "sensitivity" model.
* ``fit_nonlinear``    -- gradient-boosted regressor capturing curvature/saturation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class FitResult:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]
    params: dict = field(default_factory=dict)


def fit_linear(x, y) -> FitResult:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    res = sm.OLS(y, sm.add_constant(x)).fit()
    intercept, slope = float(res.params[0]), float(res.params[1])
    ci_low, ci_high = (float(v) for v in res.conf_int()[1])

    def predict(xx):
        return intercept + slope * np.asarray(xx, float)

    return FitResult("linear", predict, {
        "intercept": intercept,
        "slope_per_deg": slope,
        "slope_ci_low": ci_low,
        "slope_ci_high": ci_high,
    })


def fit_changepoint(x, y, grid: tuple[float, float, float]) -> FitResult:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lo, hi, step = grid
    candidates = np.arange(lo, hi + 1e-9, step)

    best = None
    for t in candidates:
        feat = np.maximum(x - t, 0.0)
        design = np.column_stack([np.ones_like(feat), feat])
        beta, _res, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
        sse = float(np.sum((y - design @ beta) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(t), float(beta[0]), float(beta[1]))

    _sse, balance, base_load, slope = best

    def predict(xx):
        return base_load + slope * np.maximum(np.asarray(xx, float) - balance, 0.0)

    return FitResult("changepoint", predict, {
        "balance_point": balance,
        "base_load": base_load,
        "slope_per_deg": slope,
    })


def fit_nonlinear(x, y, seed: int = 0) -> FitResult:
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float)
    model = HistGradientBoostingRegressor(
        max_depth=3, max_iter=300, learning_rate=0.05,
        min_samples_leaf=50, random_state=seed)
    model.fit(x, y)

    def predict(xx):
        return model.predict(np.asarray(xx, float).reshape(-1, 1))

    return FitResult("nonlinear", predict, {"type": "HistGradientBoostingRegressor"})


def fit_all(x, y, grid, seed: int = 0) -> dict[str, FitResult]:
    return {
        "linear": fit_linear(x, y),
        "changepoint": fit_changepoint(x, y, grid),
        "nonlinear": fit_nonlinear(x, y, seed=seed),
    }
