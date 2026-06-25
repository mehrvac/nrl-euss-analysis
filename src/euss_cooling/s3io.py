"""Anonymous (no-credentials) access to the public OEDI ``oedi-data-lake`` bucket.

Two access styles are used:
* ``boto3`` for whole-file downloads (metadata parquet, weather CSVs), with on-disk caching.
* ``pyarrow.fs.S3FileSystem`` for column-pruned reads of the per-building timeseries parquet,
  so we transfer only the handful of columns we model instead of the full ~4.6 MB files.
"""

from __future__ import annotations

from pathlib import Path

import boto3
import pyarrow.fs as pafs
from botocore import UNSIGNED
from botocore.client import Config as BotoConfig


def anon_s3_client(region: str = "us-west-2"):
    """boto3 S3 client that sends unsigned (anonymous) requests."""
    return boto3.client("s3", region_name=region, config=BotoConfig(signature_version=UNSIGNED))


def anon_arrow_fs(region: str = "us-west-2") -> pafs.S3FileSystem:
    """pyarrow S3 filesystem for anonymous column-pruned parquet reads."""
    return pafs.S3FileSystem(anonymous=True, region=region)


def download_file(client, bucket: str, key: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Download ``s3://bucket/key`` to ``dest`` (cached: skips if present unless ``overwrite``)."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    client.download_file(bucket, key, str(tmp))
    tmp.replace(dest)
    return dest
