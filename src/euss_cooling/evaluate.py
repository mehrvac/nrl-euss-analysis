"""Model comparison metrics and per-building sensitivity extraction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import Config
from .models import fit_all, fit_changepoint, fit_linear

log = logging.getLogger(__name__)


def metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def holdout_compare(x, y, cfg: Config) -> tuple[dict, pd.DataFrame]:
    """Train each model on a train split; report test metrics. Returns (fits_on_train, metrics_df)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.random_seed)

    fits = fit_all(x_tr, y_tr, cfg.changepoint_grid, seed=cfg.random_seed)
    rows = []
    for name, fit in fits.items():
        m = metrics(y_te, fit.predict(x_te))
        rows.append({"model": name, **m})
    return fits, pd.DataFrame(rows).set_index("model")


def per_building_sensitivity(table: pd.DataFrame, cfg: Config,
                             min_rows: int = 100) -> pd.DataFrame:
    """Fit a change-point (and linear) model per building; collect balance point + slope."""
    temp_col = cfg.temp_column
    rows = []
    for bldg_id, g in table.groupby("bldg_id"):
        if len(g) < min_rows or g["cooling_kwh"].std(ddof=0) == 0:
            continue
        x, y = g[temp_col].to_numpy(), g["cooling_kwh"].to_numpy()
        cp = fit_changepoint(x, y, cfg.changepoint_grid)
        lin = fit_linear(x, y)
        rows.append({
            "bldg_id": bldg_id,
            "balance_point": cp.params["balance_point"],
            "slope_per_deg": cp.params["slope_per_deg"],
            "base_load": cp.params["base_load"],
            "linear_slope": lin.params["slope_per_deg"],
            "annual_cooling_kwh": float(y.sum()),
        })
    out = pd.DataFrame(rows)
    log.info("Per-building sensitivities fit for %d buildings", len(out))
    return out
