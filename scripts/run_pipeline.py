"""End-to-end pipeline: select buildings -> download -> build table -> fit/compare -> report.

Usage:
    python scripts/run_pipeline.py [path/to/config.yaml]
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from euss_cooling import download, evaluate, ingest, plots  # noqa: E402
from euss_cooling.config import Config  # noqa: E402
from euss_cooling.models import fit_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")

_UNIT = {"temp_f": "°F", "temp_c": "°C"}


def write_report(cfg: Config, selection, table, metrics_df, pooled_fits, sens, fig_paths) -> Path:
    unit = _UNIT.get(cfg.temp_column, "")
    cp = pooled_fits["changepoint"].params
    upgrade_name = selection["upgrade_name"].iloc[0] if "upgrade_name" in selection else f"upgrade {cfg.upgrade}"
    lines = [
        f"# Cooling-load temperature sensitivity — {cfg.county_name_contains or cfg.county_gisjoin}",
        "",
        "## Scope",
        f"- Dataset: NREL ResStock 2024.2 (TMY3), `{cfg.prefix.split('/')[-1]}`",
        f"- Geography: {cfg.county_name_contains or cfg.county_gisjoin} (`{cfg.county_gisjoin}`), state {cfg.state}",
        f"- Upgrade: {cfg.upgrade} — \"{upgrade_name}\"",
        f"- Buildings sampled: {selection.shape[0]} (post-retrofit, with cooling)",
        f"- Target: cooling electricity (kWh/{cfg.period_label}){' incl. fans/pumps' if cfg.include_fans_pumps else ''}",
        f"- Temperature: outdoor dry-bulb in {unit} (converted from source °C)",
        f"- Modeling resolution: {cfg.resolution} | rows (building-{cfg.period_label}s): {len(table):,}",
        "",
        "## Headline sensitivity (pooled change-point model)",
        f"- **Cooling balance point: {cp['balance_point']:.1f} {unit}**",
        f"- **Sensitivity above balance point: {cp['slope_per_deg']:.3f} kWh per {unit} per dwelling-{cfg.period_label}**",
        f"- Base load below balance point: {cp['base_load']:.3f} kWh/{cfg.period_label}",
        "",
        "## Model comparison (held-out test set)",
        "",
        metrics_df.round(4).to_markdown(),
        "",
        "## Per-building sensitivity distribution",
        f"- Balance point: median {sens['balance_point'].median():.1f} {unit} "
        f"(IQR {sens['balance_point'].quantile(.25):.1f}–{sens['balance_point'].quantile(.75):.1f})",
        f"- Slope: median {sens['slope_per_deg'].median():.3f} kWh/{unit} "
        f"(IQR {sens['slope_per_deg'].quantile(.25):.3f}–{sens['slope_per_deg'].quantile(.75):.3f})",
        "",
        "## Figures",
    ] + [f"- `{p.relative_to(cfg.report_path.parent.parent)}`" for p in fig_paths] + [
        "",
        "## Caveat — Catalina Island",
        "ResStock assigns TMY3 weather at the **county** level, so this uses Los Angeles County's "
        "representative weather station, **not** Catalina Island's maritime microclimate. Treat the "
        "result as a county-level first approximation; substitute a Catalina-specific weather file "
        "(re-run with a different weather CSV) for an island-specific estimate.",
        "",
    ]
    cfg.report_path.write_text("\n".join(lines), encoding="utf-8")
    return cfg.report_path


def main(config_path: str | None = None) -> None:
    cfg = Config.load(config_path)
    cfg.ensure_dirs()

    log.info("=== 1. Select buildings ===")
    selection = download.select_buildings(cfg)

    log.info("=== 2. Download weather ===")
    weather = download.fetch_weather(cfg, selection["in.county"].unique())

    log.info("=== 3. Build modeling table ===")
    hourly = ingest.build_modeling_table(cfg, selection, weather)
    table = ingest.resolve_modeling_frame(cfg, hourly)

    log.info("=== 4. Fit & compare models ===")
    x = table[cfg.temp_column].to_numpy()
    y = table["cooling_kwh"].to_numpy()
    _train_fits, metrics_df = evaluate.holdout_compare(x, y, cfg)
    pooled_fits = fit_all(x, y, cfg.changepoint_grid, seed=cfg.random_seed)  # full-data fit for plotting/report
    print("\nModel comparison (test set):\n", metrics_df.round(4), "\n")

    log.info("=== 5. Per-building sensitivity ===")
    sens = evaluate.per_building_sensitivity(table, cfg)
    sens.to_csv(cfg.processed_dir / "per_building_sensitivity.csv", index=False)

    log.info("=== 6. Figures & report ===")
    fig1 = plots.plot_load_vs_temp(table, pooled_fits, cfg, cfg.figures_dir / "load_vs_temp.png")
    fig2 = plots.plot_sensitivity_distributions(sens, cfg, cfg.figures_dir / "sensitivity_dist.png")

    with open(cfg.models_dir / "pooled_fits.pkl", "wb") as fh:
        pickle.dump({k: v.params for k, v in pooled_fits.items()}, fh)
    metrics_df.to_csv(cfg.models_dir / "model_metrics.csv")

    report = write_report(cfg, selection, table, metrics_df, pooled_fits, sens, [fig1, fig2])
    log.info("Done. Report: %s", report)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
