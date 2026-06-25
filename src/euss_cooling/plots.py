"""Figures: pooled load-vs-temperature with fitted curves, and per-building sensitivity spread."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402

_UNIT = {"temp_f": "°F", "temp_c": "°C"}


def plot_load_vs_temp(table: pd.DataFrame, fits: dict, cfg: Config, dest: Path) -> Path:
    """Scatter of hourly cooling vs temperature with the three fitted curves overlaid."""
    temp_col = cfg.temp_column
    unit = _UNIT.get(temp_col, "")
    x = table[temp_col].to_numpy()
    y = table["cooling_kwh"].to_numpy()
    grid = np.linspace(np.nanmin(x), np.nanmax(x), 200)

    fig, ax = plt.subplots(figsize=(8, 6))
    # subsample scatter for legibility on large tables
    if len(x) > 40000:
        idx = np.random.default_rng(cfg.random_seed).choice(len(x), 40000, replace=False)
        ax.scatter(x[idx], y[idx], s=3, alpha=0.05, color="0.5")
    else:
        ax.scatter(x, y, s=3, alpha=0.05, color="0.5")

    period = cfg.period_label
    colors = {"linear": "tab:blue", "changepoint": "tab:red", "nonlinear": "tab:green"}
    for name, fit in fits.items():
        label = name
        if name == "changepoint":
            bp = fit.params["balance_point"]
            sl = fit.params["slope_per_deg"]
            label = f"changepoint (bp={bp:.1f}{unit}, {sl:.3f} kWh/{unit})"
        ax.plot(grid, fit.predict(grid), color=colors.get(name), lw=2.2, label=label)

    ax.set_xlabel(f"Outdoor dry-bulb temperature ({unit})")
    ax.set_ylabel(f"Cooling electricity (kWh / {period}, per dwelling)")
    ax.set_title(f"{cfg.resolution.capitalize()} cooling load vs temperature")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest


def plot_sensitivity_distributions(sens: pd.DataFrame, cfg: Config, dest: Path) -> Path:
    """Histograms of per-building balance point and slope (temperature sensitivity)."""
    unit = _UNIT.get(cfg.temp_column, "")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(sens["balance_point"], bins=24, color="tab:red", alpha=0.8)
    axes[0].axvline(sens["balance_point"].median(), color="k", ls="--", lw=1)
    axes[0].set_xlabel(f"Cooling balance point ({unit})")
    axes[0].set_ylabel("Buildings")
    axes[0].set_title("Per-building balance point")

    axes[1].hist(sens["slope_per_deg"], bins=24, color="tab:purple", alpha=0.8)
    axes[1].axvline(sens["slope_per_deg"].median(), color="k", ls="--", lw=1)
    axes[1].set_xlabel(f"Sensitivity (kWh per {unit} above balance point)")
    axes[1].set_ylabel("Buildings")
    axes[1].set_title("Per-building temperature sensitivity")
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest
