"""Typed access to ``config.yaml`` plus S3 key/path builders for the ResStock release.

All filesystem paths resolve relative to the repository root so the pipeline can be run from
anywhere. S3 keys are built from the confirmed ``resstock_tmy3_release_2`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    data: dict
    config_path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else REPO_ROOT / "config.yaml"
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(data=data, config_path=path)

    # --- release / S3 ---
    @property
    def bucket(self) -> str:
        return self.data["release"]["bucket"]

    @property
    def region(self) -> str:
        return self.data["release"].get("region", "us-west-2")

    @property
    def prefix(self) -> str:
        return self.data["release"]["prefix"].rstrip("/")

    # --- scope ---
    @property
    def state(self) -> str:
        return self.data["scope"]["state"]

    @property
    def upgrade(self) -> int:
        return int(self.data["scope"]["upgrade"])

    @property
    def county_gisjoin(self) -> str:
        return self.data["scope"]["county_gisjoin"]

    @property
    def county_name_contains(self) -> str:
        return self.data["scope"].get("county_name_contains", "")

    # --- sampling ---
    @property
    def n_buildings(self) -> int:
        return int(self.data["sampling"]["n_buildings"])

    @property
    def random_seed(self) -> int:
        return int(self.data["sampling"]["random_seed"])

    # --- target ---
    @property
    def cooling_column(self) -> str:
        return self.data["target"]["cooling_column"]

    @property
    def include_fans_pumps(self) -> bool:
        return bool(self.data["target"].get("include_fans_pumps", True))

    @property
    def fans_pumps_column(self) -> str:
        return self.data["target"]["fans_pumps_column"]

    @property
    def target_columns(self) -> list[str]:
        """Timeseries columns summed into the cooling-electricity target (kWh)."""
        cols = [self.cooling_column]
        if self.include_fans_pumps:
            cols.append(self.fans_pumps_column)
        return cols

    # --- modeling ---
    @property
    def temp_units(self) -> str:
        return self.data["modeling"].get("temp_units", "fahrenheit")

    @property
    def temp_column(self) -> str:
        return "temp_f" if self.temp_units == "fahrenheit" else "temp_c"

    @property
    def resolution(self) -> str:
        return self.data["modeling"].get("resolution", "daily")

    @property
    def period_label(self) -> str:
        """Per-period noun for labels/units (a slope is kWh per deg per dwelling-<period>)."""
        return "day" if self.resolution == "daily" else "hour"

    @property
    def changepoint_grid(self) -> tuple[float, float, float]:
        """(start, stop, step) balance-point search grid, in the configured temp units."""
        a, b, s = self.data["modeling"]["changepoint_grid_f"]
        return float(a), float(b), float(s)

    @property
    def test_size(self) -> float:
        return float(self.data["modeling"].get("test_size", 0.25))

    # --- paths (resolved against repo root) ---
    def _path(self, key: str) -> Path:
        return REPO_ROOT / self.data["paths"][key]

    @property
    def raw_dir(self) -> Path:
        return self._path("raw_dir")

    @property
    def interim_dir(self) -> Path:
        return self._path("interim_dir")

    @property
    def processed_dir(self) -> Path:
        return self._path("processed_dir")

    @property
    def figures_dir(self) -> Path:
        return self._path("figures_dir")

    @property
    def models_dir(self) -> Path:
        return self._path("models_dir")

    @property
    def report_path(self) -> Path:
        return self._path("report_path")

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.processed_dir,
                  self.figures_dir, self.models_dir, self.report_path.parent):
            d.mkdir(parents=True, exist_ok=True)

    # --- S3 key builders (confirmed against resstock_tmy3_release_2) ---
    @property
    def metadata_key(self) -> str:
        tag = f"upgrade{self.upgrade:02d}" if self.upgrade else "baseline"
        return (f"{self.prefix}/metadata_and_annual_results/by_state/state={self.state}/"
                f"parquet/{self.state}_{tag}_metadata_and_annual_results.parquet")

    def timeseries_key(self, bldg_id: int) -> str:
        return (f"{self.prefix}/timeseries_individual_buildings/by_state/"
                f"upgrade={self.upgrade}/state={self.state}/{bldg_id}-{self.upgrade}.parquet")

    def weather_key(self, county_gisjoin: str) -> str:
        return f"{self.prefix}/weather/state={self.state}/{county_gisjoin}_TMY3.csv"
