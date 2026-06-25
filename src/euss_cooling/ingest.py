"""Build the hourly cooling-load-vs-temperature modeling table.

Per building: the 15-minute cooling-electricity series (end-of-interval ``timestamp``, sim year
2018) is summed to clock-hour totals, then joined to hourly weather on (month, day, hour). The
join key uses the *interval start hour*: a 15-min row ending 00:15..01:00 belongs to clock hour 00,
which matches the weather row for the hour beginning 00:00.
"""

from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from . import s3io
from .config import Config
from .download import load_building_timeseries
from .weather import load_weather_hourly

log = logging.getLogger(__name__)


def building_hourly(df_ts: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Sum 15-min cooling kWh to clock-hour totals keyed by (month, day, hour)."""
    ts_end = pd.to_datetime(df_ts["timestamp"])
    hour_begin = (ts_end - pd.Timedelta(minutes=15)).dt.floor("h")
    cooling = df_ts[target_columns].sum(axis=1).to_numpy()

    hourly = (pd.DataFrame({"hour_begin": hour_begin.to_numpy(), "cooling_kwh": cooling})
              .groupby("hour_begin", as_index=False)["cooling_kwh"].sum())
    hourly["month"] = hourly["hour_begin"].dt.month
    hourly["day"] = hourly["hour_begin"].dt.day
    hourly["hour"] = hourly["hour_begin"].dt.hour
    return hourly


def build_modeling_table(cfg: Config, selection: pd.DataFrame,
                         weather_by_county: dict[str, "object"]) -> pd.DataFrame:
    """Assemble the tidy hourly table across all sampled buildings and persist it."""
    fs = s3io.anon_arrow_fs(cfg.region)
    temp_col = cfg.temp_column
    weather_cache = {code: load_weather_hourly(path, cfg.temp_units)
                     for code, path in weather_by_county.items()}

    frames = []
    for bldg_id, row in tqdm(list(selection.iterrows()), desc="buildings"):
        ts = load_building_timeseries(cfg, bldg_id, fs=fs)
        hourly = building_hourly(ts, cfg.target_columns)
        wx = weather_cache[row["in.county"]]
        merged = hourly.merge(wx, on=["month", "day", "hour"], how="left")
        merged["bldg_id"] = bldg_id
        merged["county"] = row["in.county"]
        merged["sqft"] = row.get("in.sqft")
        merged["climate_zone"] = row.get("in.ashrae_iecc_climate_zone_2004")
        frames.append(merged[["bldg_id", "hour_begin", "month", "day", "hour",
                              temp_col, "cooling_kwh", "county", "sqft", "climate_zone"]])

    table = pd.concat(frames, ignore_index=True)

    n_nan = int(table[temp_col].isna().sum())
    if n_nan:
        log.warning("%d rows have no matched temperature (dropped)", n_nan)
        table = table.dropna(subset=[temp_col])

    out = cfg.processed_dir / "modeling_table_hourly.parquet"
    table.to_parquet(out, index=False)
    log.info("Hourly modeling table: %d rows, %d buildings -> %s",
             len(table), table["bldg_id"].nunique(), out.name)
    return table


def to_daily(table: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    """Aggregate the hourly table to per-building-day: total cooling kWh vs mean temperature."""
    table = table.copy()
    table["date"] = table["hour_begin"].dt.floor("D")
    daily = (table.groupby(["bldg_id", "date"])
             .agg(cooling_kwh=("cooling_kwh", "sum"), **{temp_col: (temp_col, "mean")})
             .reset_index())
    meta = table.groupby("bldg_id")[["county", "sqft", "climate_zone"]].first()
    return daily.merge(meta, on="bldg_id", how="left")


def resolve_modeling_frame(cfg: Config, hourly: pd.DataFrame) -> pd.DataFrame:
    """Return the frame at the configured resolution and persist it."""
    if cfg.resolution == "daily":
        frame = to_daily(hourly, cfg.temp_column)
    else:
        frame = hourly
    out = cfg.processed_dir / f"modeling_table_{cfg.resolution}.parquet"
    frame.to_parquet(out, index=False)
    log.info("Modeling frame (%s): %d rows -> %s", cfg.resolution, len(frame), out.name)
    return frame
