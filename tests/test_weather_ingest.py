"""Weather parsing/conversion and the 15-min -> hourly -> weather join."""

import numpy as np
import pandas as pd

from euss_cooling.ingest import building_hourly
from euss_cooling.weather import c_to_f, load_weather_hourly


def test_c_to_f():
    assert c_to_f(0) == 32.0
    assert c_to_f(100) == 212.0
    assert c_to_f(20) == 68.0


def test_weather_hourly_keys_and_conversion(tmp_path):
    # End-of-interval, placeholder year 1999, deg C.
    csv = tmp_path / "GX_TMY3.csv"
    pd.DataFrame({
        "date_time": ["1999-01-01 01:00:00", "1999-01-01 02:00:00"],
        "Dry Bulb Temperature [°C]": [10.0, 20.0],
        "Relative Humidity [%]": [80, 70],
    }).to_csv(csv, index=False)

    wx = load_weather_hourly(csv, temp_units="fahrenheit")
    # date_time 01:00 -> hour beginning 00:00; 02:00 -> 01:00
    assert list(zip(wx["month"], wx["day"], wx["hour"])) == [(1, 1, 0), (1, 1, 1)]
    assert wx["temp_f"].tolist() == [50.0, 68.0]


def test_building_hourly_join_no_nan(tmp_path):
    csv = tmp_path / "GX_TMY3.csv"
    pd.DataFrame({
        "date_time": ["1999-01-01 01:00:00", "1999-01-01 02:00:00"],
        "Dry Bulb Temperature [°C]": [10.0, 20.0],
    }).to_csv(csv, index=False)
    wx = load_weather_hourly(csv, temp_units="fahrenheit")

    # Building 15-min series (sim year 2018), first two clock hours, end-of-interval timestamps.
    stamps = pd.date_range("2018-01-01 00:15", periods=8, freq="15min")
    ts = pd.DataFrame({
        "timestamp": stamps,
        "out.electricity.cooling.energy_consumption": [0.1] * 4 + [0.5] * 4,
        "out.electricity.cooling_fans_pumps.energy_consumption": [0.0] * 8,
    })
    hourly = building_hourly(ts, [
        "out.electricity.cooling.energy_consumption",
        "out.electricity.cooling_fans_pumps.energy_consumption",
    ])

    assert list(zip(hourly["month"], hourly["day"], hourly["hour"])) == [(1, 1, 0), (1, 1, 1)]
    np.testing.assert_allclose(hourly["cooling_kwh"].to_numpy(), [0.4, 2.0])

    merged = hourly.merge(wx, on=["month", "day", "hour"], how="left")
    assert merged["temp_f"].notna().all()
    assert merged["temp_f"].tolist() == [50.0, 68.0]
