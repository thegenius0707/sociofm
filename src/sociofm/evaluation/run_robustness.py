#!/usr/bin/env python3
"""Targeted robustness corrections for the SocioFM downstream benchmark.

Adds:
1) volume-stratified country-held-out folds with scale-robust metrics;
2) class-balanced next-day event-root classification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sociofm.evaluation.run_downstream_suite import (
    ROOTS,
    build_examples,
    calendarize,
    classification_metrics,
    read_country_days,
    regression_metrics,
    split_masks,
)


def country_metrics(frame: pd.DataFrame, mask: np.ndarray, pred: np.ndarray) -> dict:
    part = frame.loc[mask, ["country", "target_events"]].copy()
    part["prediction"] = np.maximum(pred, 0)
    rows = []
    for country, group in part.groupby("country"):
        y = group["target_events"].to_numpy()
        p = group["prediction"].to_numpy()
        rmse = math.sqrt(mean_squared_error(y, p))
        mae = mean_absolute_error(y, p)
        scale = float(np.mean(y)) + 1.0
        rows.append({
            "country": country,
            "n": int(len(group)),
            "mean_events": float(np.mean(y)),
            "rmse": float(rmse),
            "mae": float(mae),
            "nrmse_mean_plus_1": float(rmse / scale),
            "nmae_mean_plus_1": float(mae / scale),
            "rmsle": float(math.sqrt(mean_squared_error(np.log1p(y), np.log1p(p)))),
        })
    return {
        "median_country_nrmse": float(np.median([x["nrmse_mean_plus_1"] for x in rows])),
        "median_country_nmae": float(np.median([x["nmae_mean_plus_1"] for x in rows])),
        "median_country_rmsle": float(np.median([x["rmsle"] for x in rows])),
        "countries": sorted(rows, key=lambda x: -x["nrmse_mean_plus_1"]),
    }


def stratified_country_folds(frame: pd.DataFrame, train: np.ndarray, folds: int, seed: int):
    volume = (
        frame.loc[train]
        .groupby("country")["target_events"]
        .agg(["sum", "mean", "count"])
        .reset_index()
    )
    # Quantile strata prevent one fold from receiving most high-volume countries.
    rank = volume["sum"].rank(method="first")
    volume["stratum"] = pd.qcut(rank, q=min(10, len(volume)), labels=False)
    rng = np.random.default_rng(seed)
    assignments = {}
    for _, group in volume.groupby("stratum"):
        countries = group["country"].to_numpy().copy()
        rng.shuffle(countries)
        for i, country in enumerate(countries):
            assignments[str(country)] = i % folds
    return assignments, volume


def geographic_suite(frame, features, masks, folds: int, seed: int):
    X = frame[features].to_numpy(np.float32)
    y = frame["target_events"].to_numpy(np.float32)
    assignments, volume = stratified_country_folds(frame, masks["train"], folds, seed)
    result = []
    for fold in range(folds):
        heldout = {c for c, f in assignments.items() if f == fold}
        train = masks["train"] & ~frame["country"].isin(heldout).to_numpy()
        test = masks["test"] & frame["country"].isin(heldout).to_numpy()
        model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=250, learning_rate=0.06,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=seed + fold,
        ).fit(X[train], y[train])
        pred = np.maximum(model.predict(X[test]), 0)
        result.append({
            "fold": fold,
            "heldout_countries": sorted(heldout),
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "micro": regression_metrics(y[test], pred),
            "macro_country": country_metrics(frame, test, pred),
        })
    summary = {}
    for metric in ("rmse", "mae", "rmsle", "r2"):
        values = [x["micro"][metric] for x in result]
        summary[f"micro_{metric}_mean"] = float(np.mean(values))
        summary[f"micro_{metric}_sd"] = float(np.std(values, ddof=1))
    for metric in ("median_country_nrmse", "median_country_nmae", "median_country_rmsle"):
        values = [x["macro_country"][metric] for x in result]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_sd"] = float(np.std(values, ddof=1))
    return {
        "method": f"{folds}-fold country-held-out; countries stratified by pre-2025 total event volume deciles",
        "summary": summary,
        "folds": result,
        "country_training_volume": volume.to_dict(orient="records"),
    }


def class_weights(y: np.ndarray) -> dict[str, float]:
    classes, counts = np.unique(y, return_counts=True)
    # Square-root inverse frequency is less unstable than full inverse frequency.
    raw = np.sqrt(counts.sum() / (len(classes) * counts))
    raw = np.minimum(raw, 10.0)
    return {str(c): float(w) for c, w in zip(classes, raw)}


def balanced_type_suite(frame, features, masks, seeds):
    eligible = frame["target_root"].isin(ROOTS).to_numpy()
    train = masks["train"] & eligible
    validation = masks["validation"] & eligible
    test = masks["test"] & eligible
    X = frame[features].to_numpy(np.float32)
    y = frame["target_root"].to_numpy()
    weights = class_weights(y[train])
    sample_weight = np.array([weights.get(str(label), 1.0) for label in y[train]], dtype=np.float32)
    results = []
    test_proba = []
    validation_metrics = []
    for seed in seeds:
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(X[train], y[train], sample_weight=sample_weight)
        vproba = model.predict_proba(X[validation])
        vpred = model.classes_[vproba.argmax(axis=1)]
        validation_metrics.append({
            "seed": seed,
            **classification_metrics(y[validation], vpred, vproba, model.classes_),
        })
        proba = model.predict_proba(X[test])
        pred = model.classes_[proba.argmax(axis=1)]
        test_proba.append(proba)
        results.append({
            "seed": seed,
            **classification_metrics(y[test], pred, proba, model.classes_),
        })
    ensemble = np.mean(test_proba, axis=0)
    pred = model.classes_[ensemble.argmax(axis=1)]
    return {
        "weighting": "capped square-root inverse class frequency",
        "class_weights": weights,
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "test_rows": int(test.sum()),
        "validation_seeds": validation_metrics,
        "test_seeds": results,
        "test_ensemble": classification_metrics(y[test], pred, ensemble, model.classes_),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", default="7,17,29")
    args = p.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())

    print("Loading and calendarizing country-day streams...", flush=True)
    raw = read_country_days(Path(args.data_root))
    frame, features, _ = build_examples(calendarize(raw))
    masks = split_masks(frame)
    print(f"rows={len(frame)} features={len(features)}", flush=True)
    result = {
        "geographic_generalization": geographic_suite(frame, features, masks, args.folds, seeds[0]),
        "balanced_event_type": balanced_type_suite(frame, features, masks, seeds),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "geographic_summary": result["geographic_generalization"]["summary"],
        "balanced_event_type": result["balanced_event_type"]["test_ensemble"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
