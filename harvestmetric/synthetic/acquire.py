"""Download the official synthetic dataset from Kaggle and verify its checksum.

Run: python -m harvestmetric.synthetic.acquire

Kaggle served this public dataset without sign-in on 2026-11-12. If the
endpoint starts requiring authentication, download the file manually from
https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction
and place it in data/raw/synthetic/.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

from . import config

URL = f"https://www.kaggle.com/api/v1/datasets/download/{config.KAGGLE_DATASET}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dest_dir: Path = config.DATA_DIR) -> Path:
    dest = dest_dir / config.KAGGLE_FILENAME
    if dest.exists() and sha256(dest) == config.KAGGLE_SHA256:
        print(f"Already present and verified: {dest}")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as resp:  # noqa: S310 (fixed https URL)
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        dest.write_bytes(archive.read(config.KAGGLE_FILENAME))
    digest = sha256(dest)
    if digest != config.KAGGLE_SHA256:
        raise RuntimeError(f"Checksum mismatch for {dest}: {digest}. The upstream file may have changed.")
    print(f"Downloaded and verified {dest}")
    return dest


if __name__ == "__main__":
    download()
