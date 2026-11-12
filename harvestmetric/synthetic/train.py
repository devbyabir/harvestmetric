"""End-to-end training and evaluation. Run: python -m harvestmetric.synthetic.train

Protocol
1. Load the CSV, drop exact duplicate rows (one row per farm remains).
2. Hold out 20% of farms (stratified by crop) as an untouched test set.
3. For each feature set and model: tune hyperparameters with grouped CV on
   the training farms only, and estimate generalisation with nested grouped CV.
4. Select the model with the best nested-CV RMSE on the at-sowing task.
5. Fit candidates on all training farms; evaluate once on the test farms.
6. Run a label-permutation test and bootstrap intervals to judge whether the
   selected model beats chance; this sets the `prediction_supported` gate.
7. Reproduce the original notebook's row-level split to show why its R² was invalid.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedGroupKFold,
    cross_val_score,
    cross_validate,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline

from ..metrics import bootstrap_ci, breakdown, paired_bootstrap_improvement, regression_metrics
from . import config
from .data import load_dataset, prepare, read_raw
from .features import AT_SOWING, FEATURE_SETS, FULL_SEASON, build_preprocessor
from .models import candidate_models
from .split import group_holdout_split


def yield_band(y: pd.Series) -> pd.Series:
    return pd.cut(y, bins=[2000, 3000, 4000, 5000, 6000.01], right=False,
                  labels=["2000-3000", "3000-4000", "4000-5000", "5000-6000"])


SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}


def grouped_cv(seed: int = config.RANDOM_STATE) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)


def cv_splits(frame: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialise grouped, crop-stratified folds so any estimator can reuse them."""
    return list(grouped_cv(seed).split(frame, frame["crop_type"], groups=frame[config.GROUP_COL]))


def tuned(estimator, grid: dict, inner_cv):
    if not grid:
        return clone(estimator)
    return GridSearchCV(clone(estimator), grid, cv=inner_cv,
                        scoring="neg_root_mean_squared_error", n_jobs=-1)


def evaluate_feature_set(fs, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    # One row per farm, so the plain KFold used for inner tuning folds is also farm-grouped.
    if not train[config.GROUP_COL].is_unique:
        raise ValueError("expected one row per farm after de-duplication")
    X_tr, y_tr = train[fs.columns], train[config.TARGET]
    X_te, y_te = test[fs.columns], test[config.TARGET]
    outer = cv_splits(train, config.RANDOM_STATE)
    results = {}
    for name, (est, grid) in candidate_models(fs).items():
        t0 = time.time()
        # Nested CV: inner folds (tuning) are derived from each outer training fold.
        inner = KFold(n_splits=4, shuffle=True, random_state=config.RANDOM_STATE)
        nested = cross_validate(tuned(est, grid, inner), X_tr, y_tr, cv=outer, scoring=SCORING)
        final = tuned(est, grid, cv_splits(train, config.RANDOM_STATE + 1)).fit(X_tr, y_tr)
        best_params = getattr(final, "best_params_", {})
        model = getattr(final, "best_estimator_", final)
        pred = model.predict(X_te)
        results[name] = {
            "cv": {k: float(abs(nested[f"test_{k}"]).mean()) if k != "r2" else float(nested["test_r2"].mean())
                   for k in SCORING},
            "cv_r2_std": float(nested["test_r2"].std()),
            "best_params": {k.replace("model__", ""): v for k, v in best_params.items()},
            "test": regression_metrics(y_te, pred),
            "seconds": round(time.time() - t0, 1),
            "_model": model,
            "_pred": pred,
        }
    return results


def legacy_split_demo(path) -> dict:
    """Mimic the notebook: row-level KFold on the 10x duplicated working copy vs grouped CV.

    The notebook's working copy held every official record exactly
    config.LEGACY_COPIES times (verified record-identical on 2026-11-12). If the
    input has no duplicates (the official Kaggle file), that copy is recreated
    here, so the demonstration does not depend on the unpublished local file.
    """
    raw = prepare(read_raw(path))
    recreated = len(raw) == raw.drop_duplicates().shape[0]
    if recreated:
        raw = pd.concat([raw] * config.LEGACY_COPIES, ignore_index=True)
    fs = FULL_SEASON
    model = Pipeline([("prep", build_preprocessor(fs)),
                      ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                                      random_state=config.RANDOM_STATE))])
    X, y = raw[fs.columns], raw[config.TARGET]
    row_level = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=config.RANDOM_STATE),
                                scoring="r2")
    grouped = cross_val_score(model, X, y, groups=raw[config.GROUP_COL],
                              cv=StratifiedGroupKFold(5, shuffle=True, random_state=config.RANDOM_STATE).split(
                                  X, raw["crop_type"], raw[config.GROUP_COL]),
                              scoring="r2")
    return {
        "rows": int(len(raw)),
        "working_copy_recreated_from_official_file": recreated,
        "model": "Random forest, full_season features",
        "row_level_kfold_r2": float(row_level.mean()),
        "grouped_by_farm_kfold_r2": float(grouped.mean()),
    }


def run(path=None, n_permutations: int = 200) -> dict:
    df, load_report = load_dataset(path)
    train_idx, test_idx = group_holdout_split(df)
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    out: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "data": {"source_file": Path(load_report.path).name,
                 "raw_rows": load_report.raw_rows, "unique_rows": load_report.unique_rows,
                 "original_working_copy_rows": load_report.unique_rows * config.LEGACY_COPIES,
                 "train_farms": int(len(train)), "test_farms": int(len(test)),
                 "test_farm_ids": sorted(test[config.GROUP_COL].tolist())},
        "prediction_point": AT_SOWING.description,
        "feature_sets": {n: {"description": f.description, "features": f.columns}
                         for n, f in FEATURE_SETS.items()},
        "results": {},
    }

    all_results = {}
    for name, fs in FEATURE_SETS.items():
        print(f"\n== Feature set: {name} ==")
        res = evaluate_feature_set(fs, train, test)
        all_results[name] = res
        out["results"][name] = {m: {k: v for k, v in r.items() if not k.startswith("_")} for m, r in res.items()}
        for m, r in res.items():
            print(f"  {m:20s} CV R2 {r['cv']['r2']:+.3f}  RMSE {r['cv']['rmse']:7.1f} | "
                  f"test R2 {r['test']['r2']:+.3f}  MAE {r['test']['mae']:7.1f}  RMSE {r['test']['rmse']:7.1f}")

    # ---- Selection on the primary (at-sowing) task, by nested-CV RMSE only.
    primary = all_results[AT_SOWING.name]
    learned = {m: r for m, r in primary.items() if "baseline" not in m}
    selected = min(learned, key=lambda m: learned[m]["cv"]["rmse"])
    best_baseline = min((m for m in primary if "baseline" in m), key=lambda m: primary[m]["cv"]["rmse"])
    sel, base = primary[selected], primary[best_baseline]
    y_te = test[config.TARGET].to_numpy()

    print(f"\nSelected model: {selected}; permutation test ({n_permutations} permutations)...")
    perm_cv = cv_splits(train, config.RANDOM_STATE + 2)
    score, perm_scores, p_value = permutation_test_score(
        clone(sel["_model"]), train[AT_SOWING.columns], train[config.TARGET], cv=perm_cv,
        n_permutations=n_permutations, scoring="r2", random_state=config.RANDOM_STATE, n_jobs=-1)

    r2_ci = bootstrap_ci(y_te, sel["_pred"], "r2")
    improvement = paired_bootstrap_improvement(y_te, sel["_pred"], base["_pred"])
    supported = bool(p_value < 0.05 and r2_ci[0] > 0 and improvement["ci_low"] > 0)

    out["selection"] = {
        "task": AT_SOWING.name,
        "selected_model": selected,
        "best_baseline": best_baseline,
        "rule": "lowest nested grouped-CV RMSE on training farms; test set used once afterwards",
        "test_r2_95ci": list(r2_ci),
        "rmse_vs_baseline": improvement,
        "permutation_test": {"cv_r2": float(score), "null_r2_mean": float(perm_scores.mean()),
                             "null_r2_95pct": float(np.quantile(perm_scores, 0.95)),
                             "p_value": float(p_value), "n_permutations": n_permutations},
        "prediction_supported": supported,
        "gate": "prediction shown only if permutation p < 0.05 AND test R² 95% CI > 0 "
                "AND RMSE reduction vs best baseline has 95% CI > 0",
    }
    print(f"  CV R2 {score:+.3f}, permutation p = {p_value:.3f}, test R2 95% CI {r2_ci[0]:+.3f}..{r2_ci[1]:+.3f}")
    print(f"  Prediction supported: {supported}")

    # ---- Breakdown and failure cases on the test set
    frame = test[[config.GROUP_COL, "crop_type", "region", config.TARGET]].copy()
    frame["predicted"] = sel["_pred"]
    frame["baseline"] = base["_pred"]
    frame["abs_error"] = (frame.predicted - frame[config.TARGET]).abs()
    frame["yield_band"] = yield_band(frame[config.TARGET])
    by_crop = breakdown(frame, "crop_type", config.TARGET)
    by_band = breakdown(frame, "yield_band", config.TARGET)
    worst = frame.sort_values("abs_error", ascending=False).head(10)
    out["selection"]["prediction_spread"] = {
        "test_pred_std": float(np.std(sel["_pred"])), "test_actual_std": float(np.std(y_te)),
        "pred_min": float(np.min(sel["_pred"])), "pred_max": float(np.max(sel["_pred"]))}

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    by_crop.round(2).to_csv(config.TABLES_DIR / "test_by_crop.csv", index=False)
    by_band.round(2).to_csv(config.TABLES_DIR / "test_by_yield_band.csv", index=False)
    worst.round(2).to_csv(config.TABLES_DIR / "test_worst_errors.csv", index=False)
    frame.round(2).to_csv(config.TABLES_DIR / "test_predictions.csv", index=False)
    rows = [{"feature_set": fs, "model": m, **{f"cv_{k}": v for k, v in r["cv"].items()},
             **{f"test_{k}": v for k, v in r["test"].items() if k != "n"}}
            for fs, res in out["results"].items() for m, r in res.items()]
    pd.DataFrame(rows).round(4).to_csv(config.TABLES_DIR / "model_comparison.csv", index=False)

    print("\nReproducing the original row-level evaluation for comparison...")
    out["legacy_split_demo"] = legacy_split_demo(path)
    print("  ", out["legacy_split_demo"])

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": sel["_model"], "feature_set": AT_SOWING.name, "model_name": selected,
                 "prediction_supported": supported, "sklearn_version": sklearn.__version__},
                config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {config.METRICS_PATH.relative_to(config.PROJECT_ROOT)} and "
          f"{config.MODEL_PATH.relative_to(config.PROJECT_ROOT)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="Path to the CSV (default: auto-detect)")
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    run(args.data, args.permutations)


if __name__ == "__main__":
    main()
