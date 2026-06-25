# nrl-euss-analysis

Quantifies the **sensitivity of residential HVAC cooling electricity load to outdoor dry-bulb
temperature** using NREL's **ResStock 2024.2** End-Use Load Profiles dataset (simulated on
**TMY3** weather).

## What it does

For a sample of simulated buildings in **Los Angeles County, CA** that received **upgrade
package 1** (the HEAT STAR heat-pump-with-electric-backup retrofit), the pipeline:

1. Selects buildings from the ResStock metadata and downloads their 15-minute timeseries Parquet
   files plus the county's TMY3 weather (EPW) from the public OEDI S3 bucket.
2. Builds an hourly modeling table of **cooling electricity (kWh)** vs **dry-bulb temperature (°F)**
   (source weather is °C and converted before modeling).
3. Fits and compares three models of load vs temperature:
   - **Linear** regression (single slope baseline),
   - **Change-point / balance-point** regression (flat below a cooling balance point, then a slope),
   - a **flexible nonlinear** model (spline / gradient-boosted).
4. Reports the **temperature sensitivity** (kWh per °F) and **cooling balance-point temperature**,
   per-building distributions, model comparison metrics, and figures.

> **Catalina Island note:** ResStock assigns weather at the *county* level, so LA County's TMY3
> file reflects a county-representative station, **not** Catalina's maritime microclimate. Treat
> the county result as a first approximation; a Catalina-specific weather file can be substituted
> later. See `outputs/report.md` after a run.

## Data source

Public OEDI data lake (anonymous access, region `us-west-2`):

```
s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_tmy3_release_2/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python scripts/run_pipeline.py            # uses config.yaml
```

Outputs land in `outputs/figures/`, `outputs/models/`, and `outputs/report.md`.
Downloaded data is cached under `data/raw/` (git-ignored) so re-runs are offline.

## Configuration

All scope/knobs live in [`config.yaml`](config.yaml): geography (state/county/upgrade), sample
size, target column, balance-point search grid, and units. Change `county_*` / `state` / `upgrade`
to retarget the analysis.

## Layout

```
src/euss_cooling/   # config, S3 IO, download, weather, ingest, models, evaluate, plots
scripts/            # run_pipeline.py
tests/              # weather-join + change-point unit tests
data/               # downloaded ResStock + weather (git-ignored)
outputs/            # figures, fitted models, report (git-ignored)
```
