"""Download the three public sources used in Part B and record their provenance.

Run: python -m harvestmetric.real.acquire [--skip-weather]

1. USDA NASS Quick Stats: Iowa county corn-for-grain yield (CC0). Retrieved
   through the Quick Stats web interface's spreadsheet export, which needs no
   API key. The query parameters are in config.NASS_QUERY.
2. US Census Bureau 2024 Gazetteer: county internal points (lat/lon), used to
   place weather queries. US federal government work (public domain).
3. NASA POWER daily point API: 2 m max/min temperature and corrected
   precipitation for each county's internal point. Citation required.

Raw files go to data/raw/real/ (git-ignored). Provenance (URLs, retrieval
time, checksums, versions) is written to data/processed/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

USER_AGENT = "harvestmetric-research-pipeline/0.2"


def _get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https hosts)
        return resp.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_nass() -> dict:
    query = urllib.parse.urlencode(config.NASS_QUERY, quote_via=urllib.parse.quote)
    uuid = _get(f"https://quickstats.nass.usda.gov/uuid/encode?{query}").decode().strip().strip('"')
    url = f"https://quickstats.nass.usda.gov/data/spreadsheet/{uuid}.csv"
    data = _get(url)
    if not data.startswith(b'"Program"'):
        raise RuntimeError("Unexpected Quick Stats response; the web export format may have changed.")
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.NASS_FILE.write_bytes(data)
    return {"source": "USDA NASS Quick Stats (web spreadsheet export)", "query": config.NASS_QUERY,
            "query_uuid": uuid, "url": url, "retrieved_utc": _now(), "bytes": len(data),
            "sha256": _sha256(data), "license": "CC0 1.0 (catalog.data.gov listing for NASS Quick Stats)"}


def fetch_gazetteer() -> dict:
    payload = _get(config.GAZETTEER_URL)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".txt"))
        table = pd.read_csv(archive.open(name), sep="\t", dtype={"GEOID": str})
    table.columns = [c.strip() for c in table.columns]
    iowa = table[table.GEOID.str.startswith(config.STATE_FIPS)][["GEOID", "NAME", "INTPTLAT", "INTPTLONG"]]
    iowa = iowa.rename(columns={"GEOID": "fips", "NAME": "county_name", "INTPTLAT": "lat", "INTPTLONG": "lon"})
    iowa.to_csv(config.GAZETTEER_FILE, index=False)
    return {"source": "US Census Bureau 2024 Gazetteer, counties", "url": config.GAZETTEER_URL,
            "retrieved_utc": _now(), "zip_sha256": _sha256(payload), "iowa_counties": int(len(iowa)),
            "license": "US federal government work, not subject to copyright (17 U.S.C. 105)"}


def fetch_power(sleep_s: float = 1.0) -> dict:
    centroids = pd.read_csv(config.GAZETTEER_FILE, dtype={"fips": str})
    config.POWER_DIR.mkdir(parents=True, exist_ok=True)
    version, digests = None, {}
    for row in centroids.itertuples():
        dest = config.POWER_DIR / f"{row.fips}.json"
        if not dest.exists():
            params = {"parameters": ",".join(config.POWER_PARAMETERS), "community": "AG",
                      "latitude": f"{row.lat:.4f}", "longitude": f"{row.lon:.4f}",
                      "start": config.POWER_START, "end": config.POWER_END, "format": "JSON"}
            dest.write_bytes(_get(f"{config.POWER_URL}?{urllib.parse.urlencode(params)}"))
            time.sleep(sleep_s)
        raw = dest.read_bytes()
        digests[row.fips] = _sha256(raw)
        version = json.loads(raw)["header"]["api"]["version"]
    return {"source": "NASA POWER daily point API (community=AG)", "url": config.POWER_URL,
            "parameters": config.POWER_PARAMETERS, "period": [config.POWER_START, config.POWER_END],
            "api_version": version, "retrieved_utc": _now(), "points": len(digests),
            "sha256_by_fips": digests,
            "citation": ("These data were obtained from the NASA Langley Research Center (LaRC) POWER Project "
                         "funded through the NASA Earth Science/Applied Science Program."),
            "terms": "No use restrictions stated; citation required; NASA asks to be notified of redistribution."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-weather", action="store_true", help="only NASS + Gazetteer")
    args = parser.parse_args()
    manifest = json.loads(config.MANIFEST_PATH.read_text()) if config.MANIFEST_PATH.exists() else {}
    print("NASS Quick Stats ..."); manifest["nass"] = fetch_nass()
    print("Census Gazetteer ..."); manifest["gazetteer"] = fetch_gazetteer()
    if not args.skip_weather:
        print("NASA POWER (99 points, cached per county) ..."); manifest["power"] = fetch_power()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {config.MANIFEST_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
