"""Fetch ResStock metadata, select the building sample, and pull weather + timeseries.

Selection logic (confirmed against the 2024.2 / release_2 upgrade-01 metadata):
* ``in.county`` is the county GISJOIN code (e.g. ``G0600370`` = Los Angeles County, CA).
* ``in.*`` columns describe the *baseline* building; ``upgrade.*`` columns and ``upgrade_name``
  describe the post-retrofit building. Upgrade 1 = "ENERGY STAR heat pump with elec backup".
* ``applicability == True`` marks buildings that actually received the upgrade; post-upgrade these
  all have a heat pump providing cooling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import s3io
from .config import Config

log = logging.getLogger(__name__)

ANNUAL_COOLING_COL = "out.electricity.cooling.energy_consumption.kwh"

# Metadata columns kept for selection, joining, and reporting.
META_COLUMNS = [
    "applicability",
    "in.county",
    "in.county_name",
    "in.sqft",
    "in.ashrae_iecc_climate_zone_2004",
    "in.hvac_cooling_efficiency",
    "upgrade.hvac_cooling_efficiency",
    "upgrade_name",
    ANNUAL_COOLING_COL,
]


def fetch_metadata(cfg: Config) -> Path:
    """Download the (upgrade-specific) state metadata parquet to the raw cache."""
    client = s3io.anon_s3_client(cfg.region)
    dest = cfg.raw_dir / Path(cfg.metadata_key).name
    log.info("Fetching metadata: %s", dest.name)
    return s3io.download_file(client, cfg.bucket, cfg.metadata_key, dest)


def select_buildings(cfg: Config) -> pd.DataFrame:
    """Return a reproducible sample of eligible buildings (indexed by ``bldg_id``)."""
    meta_path = fetch_metadata(cfg)
    df = pd.read_parquet(meta_path, columns=META_COLUMNS)

    in_county = df["in.county"] == cfg.county_gisjoin
    if not in_county.any() and cfg.county_name_contains:
        log.warning("County code %s not found; falling back to name match '%s'",
                    cfg.county_gisjoin, cfg.county_name_contains)
        in_county = df["in.county_name"].astype(str).str.contains(
            cfg.county_name_contains, case=False, na=False)

    eligible = df[in_county & (df["applicability"]) & (df[ANNUAL_COOLING_COL] > 0)].copy()
    if eligible.empty:
        raise RuntimeError(
            f"No eligible buildings for county={cfg.county_gisjoin} upgrade={cfg.upgrade}. "
            "Check scope settings in config.yaml.")

    n = min(cfg.n_buildings, len(eligible))
    sample = eligible.sample(n=n, random_state=cfg.random_seed).sort_index()
    log.info("Selected %d of %d eligible buildings (upgrade=%d, '%s')",
             n, len(eligible), cfg.upgrade,
             sample["upgrade_name"].iloc[0] if "upgrade_name" in sample else "?")
    return sample


def fetch_weather(cfg: Config, county_gisjoins) -> dict[str, Path]:
    """Download the TMY3 weather CSV for each county code; returns code -> local path."""
    client = s3io.anon_s3_client(cfg.region)
    out: dict[str, Path] = {}
    for code in sorted(set(county_gisjoins)):
        key = cfg.weather_key(code)
        dest = cfg.raw_dir / Path(key).name
        s3io.download_file(client, cfg.bucket, key, dest)
        out[code] = dest
    log.info("Weather files ready: %s", ", ".join(p.name for p in out.values()))
    return out


def load_building_timeseries(cfg: Config, bldg_id: int, fs=None) -> pd.DataFrame:
    """Read only the timestamp + cooling columns for one building (cached locally as slim parquet)."""
    cache = cfg.interim_dir / f"{bldg_id}-{cfg.upgrade}_cooling.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    fs = fs or s3io.anon_arrow_fs(cfg.region)
    key = f"{cfg.bucket}/{cfg.timeseries_key(bldg_id)}"
    cols = ["timestamp"] + cfg.target_columns
    df = pq.read_table(key, columns=cols, filesystem=fs).to_pandas()
    df.to_parquet(cache, index=False)
    return df
