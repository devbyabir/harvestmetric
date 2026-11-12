"""Check whether two copies of the dataset contain the same records.

Usage:
    python -m harvestmetric.synthetic.provenance OFFICIAL.csv OTHER.csv

Example: OFFICIAL = data/raw/synthetic/Smart_Farming_Crop_Yield_2024.csv and
OTHER = final_dataset_crop_yield.csv (the original working copy).

Both files are de-duplicated and their dates normalised before comparison, so
a 10x duplicated copy or a different date format still counts as a match.
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from . import config
from .data import parse_dates, read_raw


def normalised(path) -> pd.DataFrame:
    df = read_raw(path).drop_duplicates()
    for col in config.DATE_COLS:
        df[col] = parse_dates(df[col]).dt.strftime("%Y-%m-%d")
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(6)
    return df.sort_values(config.GROUP_COL).reset_index(drop=True)


def compare(path_a, path_b) -> dict:
    a, b = normalised(path_a), normalised(path_b)
    same_cols = list(a.columns) == list(b.columns)
    result: dict[str, Any] = {"rows_a": len(a), "rows_b": len(b), "same_columns": same_cols}
    if same_cols and len(a) == len(b):
        diff = (a != b).sum()
        result["cells_differing"] = int(diff.sum())
        result["columns_with_differences"] = diff[diff > 0].to_dict()
        result["identical_records"] = bool(diff.sum() == 0)
    else:
        result["identical_records"] = False
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out = compare(sys.argv[1], sys.argv[2])
    for k, v in out.items():
        print(f"{k}: {v}")
    sys.exit(0 if out["identical_records"] else 1)
