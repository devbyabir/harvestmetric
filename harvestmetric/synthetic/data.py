"""Loading, de-duplication and schema validation of the raw CSV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when the dataset violates the expected schema."""


@dataclass
class LoadReport:
    path: str
    raw_rows: int
    unique_rows: int
    exact_duplicates_removed: int
    issues: list[str] = field(default_factory=list)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse day-first (dd/mm/yyyy) or ISO (yyyy-mm-dd) dates; fail loudly on anything else."""
    text = series.astype(str).str.strip()
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}$")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    out[iso] = pd.to_datetime(text[iso], format="%Y-%m-%d", errors="coerce")
    out[~iso] = pd.to_datetime(text[~iso], format="%d/%m/%Y", errors="coerce")
    if out.isna().any():
        bad = text[out.isna()].head(3).tolist()
        raise DataValidationError(f"{series.name}: unparseable dates, e.g. {bad}")
    return out


def read_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else config.default_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See docs/DATA_PROVENANCE.md for how to obtain it, "
            "or set HARVESTMETRIC_DATA to its location."
        )
    # 'None' is a real category (no irrigation / healthy crop), so disable NA parsing.
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    issues: list[str] = []
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df.isna().any().any():
        issues.append(f"missing values in: {df.columns[df.isna().any()].tolist()}")
    for col, allowed in config.CATEGORIES.items():
        unknown = sorted(set(df[col].astype(str)) - set(allowed))
        if unknown:
            issues.append(f"{col}: unexpected categories {unknown}")
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            issues.append(f"{col}: non-numeric values")
        elif ((values < lo) | (values > hi)).any():
            issues.append(f"{col}: {int(((values < lo) | (values > hi)).sum())} values outside [{lo}, {hi}]")
    dup_farms = df.drop_duplicates().duplicated(config.GROUP_COL).sum()
    if dup_farms:
        issues.append(f"{dup_farms} farm_ids appear with conflicting records")
    return issues


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, derive sowing_month, and check internal date consistency."""
    out = df.copy()
    for col in config.DATE_COLS:
        out[col] = parse_dates(out[col])
    out["sowing_month"] = out["sowing_date"].dt.month
    days = (out["harvest_date"] - out["sowing_date"]).dt.days
    if (days != out["total_days"]).any():
        raise DataValidationError("total_days does not equal harvest_date - sowing_date")
    return out


def load_dataset(path: str | Path | None = None, strict: bool = True) -> tuple[pd.DataFrame, LoadReport]:
    """Load, drop exact duplicate rows, validate and prepare. Returns (frame, report)."""
    raw = read_raw(path)
    # Sort so the split does not depend on row order in the file.
    unique = raw.drop_duplicates().sort_values(config.GROUP_COL, kind="stable").reset_index(drop=True)
    issues = validate(unique)
    if issues and strict:
        raise DataValidationError("; ".join(issues))
    report = LoadReport(
        path=str(path or config.default_data_path()),
        raw_rows=len(raw),
        unique_rows=len(unique),
        exact_duplicates_removed=len(raw) - len(unique),
        issues=issues,
    )
    return prepare(unique), report
