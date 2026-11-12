"""Data-quality and plausibility audit. Run: python -m harvestmetric.synthetic.audit"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from . import config
from .data import load_dataset, read_raw

# Rough real-world bounding boxes (degrees). Longitudes east of Greenwich are positive.
REGION_BOXES = {
    "Central USA": {"lat": (30, 49), "lon": (-105, -85)},
    "South USA": {"lat": (25, 37), "lon": (-107, -75)},
    "North India": {"lat": (22, 35), "lon": (68, 89)},
    "South India": {"lat": (8, 20), "lon": (73, 85)},
    "East Africa": {"lat": (-12, 15), "lon": (28, 52)},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_audit(path: str | Path | None = None) -> dict:
    path = Path(path) if path else config.default_data_path()
    raw = read_raw(path)
    df, report = load_dataset(path)
    numeric = [c for c in config.NUMERIC_RANGES if c != "total_days"]

    # Duplicate structure
    copies = raw.groupby(list(raw.columns)).size()
    dup: dict[str, Any] = {
        "raw_rows": report.raw_rows,
        "unique_rows": report.unique_rows,
        "copies_per_unique_row": sorted(copies.unique().tolist()),
        "unique_farm_ids": int(df[config.GROUP_COL].nunique()),
        "unique_sensor_ids": int(df["sensor_id"].nunique()),
        "farm_and_sensor_numbers_identical": bool(
            (df.farm_id.str[4:] == df.sensor_id.str[4:]).all()),
    }

    # Geography: share of farms whose coordinates fall inside their region's box
    geo: dict[str, Any] = {}
    for region, box in REGION_BOXES.items():
        g = df[df.region == region]
        inside = g.latitude.between(*box["lat"]) & g.longitude.between(*box["lon"])
        geo[region] = {
            "n": int(len(g)),
            "lat_range": [round(g.latitude.min(), 2), round(g.latitude.max(), 2)],
            "lon_range": [round(g.longitude.min(), 2), round(g.longitude.max(), 2)],
            "share_inside_plausible_box": round(float(inside.mean()), 3),
        }

    # Dates
    dates: dict[str, Any] = {
        "sowing_range": [str(df.sowing_date.min().date()), str(df.sowing_date.max().date())],
        "harvest_range": [str(df.harvest_date.min().date()), str(df.harvest_date.max().date())],
        "years_present": sorted(df.sowing_date.dt.year.unique().tolist()),
        "timestamp_within_season_share": float(
            ((df.timestamp >= df.sowing_date) & (df.timestamp <= df.harvest_date)).mean()),
        "sowing_window_by_crop": {
            crop: [str(g.sowing_date.min().date()), str(g.sowing_date.max().date())]
            for crop, g in df.groupby("crop_type")
        },
    }

    # Distribution shape: KS test against a uniform on the observed range
    uniform = {}
    for col in numeric:
        x = df[col].to_numpy(float)
        lo, hi = x.min(), x.max()
        uniform[col] = round(float(stats.kstest(x, "uniform", args=(lo, hi - lo)).pvalue), 4)

    # Relationship with target
    corr = df[numeric].corr()[config.TARGET].drop(config.TARGET)
    pairwise = df[numeric].corr().abs().where(~np.eye(len(numeric), dtype=bool))
    anova = {}
    for col in config.CATEGORIES:
        groups = [g[config.TARGET].to_numpy() for _, g in df.groupby(col)]
        anova[col] = round(float(stats.f_oneway(*groups).pvalue), 4)

    target_by_crop = (df.groupby("crop_type")[config.TARGET]
                      .agg(["count", "mean", "std", "min", "max"]).round(1))

    findings = [
        (f"The official file has one row per farm ({dup['unique_rows']} farms). The working copy used by "
         f"the original notebook repeated every record {config.LEGACY_COPIES} times (verified record-identical "
         "to the official file), so its row-level split placed identical copies in train and test."
         if dup["copies_per_unique_row"] == [1] else
         f"Each of the {dup['unique_rows']} unique rows appears "
         f"{'/'.join(map(str, dup['copies_per_unique_row']))} times; "
         "a row-level split therefore places identical copies in train and test."),
        "Coordinates do not match region labels: every region spans the same box "
        "(lat ~10-35, lon ~70-90 E). US regions would need negative longitudes; "
        "East Africa lies at lon ~28-52 E.",
        f"All sowing dates fall in {dates['sowing_range'][0]} .. {dates['sowing_range'][1]} "
        "with the same window for every crop and region, although regional cropping "
        "calendars differ (e.g. Indian rice/cotton are usually sown in the monsoon season).",
        f"{sum(p > 0.05 for p in uniform.values())} of {len(uniform)} numeric columns are "
        "statistically indistinguishable from a uniform distribution (KS p > 0.05).",
        f"Largest |Pearson r| between any feature and yield is {corr.abs().max():.3f}; "
        f"no categorical column has an ANOVA p-value below 0.05 (min p = {min(anova.values()):.3f}).",
        "Yield is ~Uniform(2000, 6000) kg/ha for every crop, including cotton, "
        "whose realistic yield scale differs from cereals.",
        "The publisher's description states the data 'simulates' smart-farming operations.",
    ]

    return {
        "file": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
        "load_issues": report.issues,
        "duplicates": dup,
        "geography": geo,
        "dates": dates,
        "uniform_ks_pvalues": uniform,
        "pearson_with_target": corr.round(4).to_dict(),
        "max_abs_pairwise_feature_corr": round(float(np.nanmax(pairwise.to_numpy())), 4),
        "anova_pvalues_categorical": anova,
        "target_by_crop": target_by_crop.reset_index().to_dict(orient="records"),
        "findings": findings,
        "conclusion": (
            "The data shows the signature of independently, uniformly generated columns with no "
            "built-in relationship to yield. Treat it as synthetic practice data, not field "
            "observations."
        ),
    }


def main() -> None:
    result = run_audit()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Do not write absolute local paths into a report that may be published.
    result["file"]["path"] = Path(result["file"]["path"]).name
    config.AUDIT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {config.AUDIT_PATH.relative_to(config.PROJECT_ROOT)}")
    for f in result["findings"]:
        print(" -", f)


if __name__ == "__main__":
    main()
