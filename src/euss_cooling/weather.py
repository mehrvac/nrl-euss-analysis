"""Parse ResStock TMY3 weather CSVs into an hourly dry-bulb temperature table.

The ResStock weather CSV is hourly with an **end-of-interval** ``date_time`` and a placeholder
year (e.g. 1999). Dry-bulb is in degrees Celsius. We convert to the configured units and key each
row by the (month, day, hour) of the interval's *start hour*, so it can be joined to the building
timeseries (which uses a different placeholder year) on calendar position rather than absolute time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def c_to_f(celsius):
    return celsius * 9.0 / 5.0 + 32.0


def _temp_column(columns) -> str:
    for c in columns:
        if "dry bulb" in c.lower():
            return c
    raise KeyError(f"No dry-bulb temperature column found in {list(columns)}")


def load_weather_hourly(csv_path: str | Path, temp_units: str = "fahrenheit") -> pd.DataFrame:
    """Return columns ``month, day, hour, <temp_f|temp_c>`` for the hour beginning at that time."""
    w = pd.read_csv(csv_path)
    dt_end = pd.to_datetime(w["date_time"])
    hour_begin = dt_end - pd.Timedelta(hours=1)  # end-of-interval label -> hour beginning

    temp_c = w[_temp_column(w.columns)].astype(float)
    if temp_units == "fahrenheit":
        temp = c_to_f(temp_c)
        temp_col = "temp_f"
    else:
        temp = temp_c
        temp_col = "temp_c"

    return pd.DataFrame({
        "month": hour_begin.dt.month.to_numpy(),
        "day": hour_begin.dt.day.to_numpy(),
        "hour": hour_begin.dt.hour.to_numpy(),
        temp_col: temp.to_numpy(),
    })
