#!/usr/bin/env python3
"""Scale-robust country-held-out forecasting benchmark.

Compares persistence, a raw Poisson tree, and a log-change tree that can
extrapolate to countries whose event volume exceeds every training country.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from sociofm.evaluation.run_downstream_suite import (
    build_examples, calendarize, read_country_days, regression_metrics, split_masks,
)
from sociofm.evaluation.run_robustness import country_metrics, stratified_country_folds


def summarize(folds: list[dict], model_name: str) -> dict:
    keys = ("rmse", "mae", "rmsle", "r2")
    out = {}
    for key in keys:
        values = [x["models"][model_name]["micro"][key] for x in folds]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_sd"] = float(np.std(values, ddof=1))
    for key in ("median_country_nrmse", "median_country_nmae", "median_country_rmsle"):
        values = [x["models"][model_name]["macro_country"][key] for x in folds]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_sd"] = float(np.std(values, ddof=1))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    print("Loading country-day streams...", flush=True)
    raw = read_country_days(Path(args.data_root))
    frame, features, _ = build_examples(calendarize(raw))
    masks = split_masks(frame)
    assignments, _ = stratified_country_folds(frame, masks["train"], args.folds, args.seed)
    X = frame[features].to_numpy(np.float32)
    y = frame["target_events"].to_numpy(np.float32)
    current = frame["events"].to_numpy(np.float32)
    folds = []

    for fold in range(args.folds):
        heldout = {c for c, value in assignments.items() if value == fold}
        train = masks["train"] & ~frame["country"].isin(heldout).to_numpy()
        test = masks["test"] & frame["country"].isin(heldout).to_numpy()

        raw_model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=250, learning_rate=0.06,
            max_leaf_nodes=31, l2_regularization=1.0,
            random_state=args.seed + fold,
        ).fit(X[train], y[train])
        raw_pred = np.maximum(raw_model.predict(X[test]), 0)

        # Relative log change transfers across very different country scales.
        delta = np.log1p(y) - np.log1p(current)
        change_model = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=250, learning_rate=0.06,
            max_leaf_nodes=31, l2_regularization=1.0,
            random_state=args.seed + 100 + fold,
        ).fit(X[train], delta[train])
        change = np.clip(change_model.predict(X[test]), -4.0, 4.0)
        change_pred = np.maximum(
            np.expm1(np.log1p(current[test]) + change), 0
        )
        persistence = current[test]

        models = {}
        for name, pred in (
            ("persistence", persistence),
            ("raw_poisson_tree", raw_pred),
            ("log_change_tree", change_pred),
        ):
            models[name] = {
                "micro": regression_metrics(y[test], pred),
                "macro_country": country_metrics(frame, test, pred),
            }
        folds.append({
            "fold": fold,
            "heldout_countries": sorted(heldout),
            "test_rows": int(test.sum()),
            "models": models,
        })
        print(
            f"fold={fold} persistence_rmse={models['persistence']['micro']['rmse']:.2f} "
            f"raw_rmse={models['raw_poisson_tree']['micro']['rmse']:.2f} "
            f"log_change_rmse={models['log_change_tree']['micro']['rmse']:.2f}",
            flush=True,
        )

    result = {
        "protocol": (
            f"{args.folds}-fold country-held-out, stratified by pre-2025 event-volume decile; "
            "2025-07 onward test"
        ),
        "summary": {
            name: summarize(folds, name)
            for name in ("persistence", "raw_poisson_tree", "log_change_tree")
        },
        "folds": folds,
        "interpretation_guardrail": (
            "Geographic fragility is claimed only relative to persistence and the "
            "scale-transferable log-change model, not from raw-count RMSE alone."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
