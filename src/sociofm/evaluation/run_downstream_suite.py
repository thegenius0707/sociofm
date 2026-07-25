#!/usr/bin/env python3
"""Reproducible downstream benchmark suite for SocioFM country-day streams.

The suite is deliberately leakage-safe: features at day t predict outcomes at
day t+1, and all temporal splits are based on the target day.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

SEEDS = (7, 17, 29)
ROOTS = tuple(f"{i:02d}" for i in range(1, 21))

# GDELT uses FIPS country codes. This is an explicit analytical grouping, not
# a claim that development status is binary or immutable.
GLOBAL_NORTH_FIPS = {
    "AS", "AU", "BE", "CA", "DA", "EI", "FI", "FR", "GM", "GR", "IC",
    "IT", "JA", "LU", "NE", "NO", "NZ", "PO", "SP", "SW", "SZ", "UK",
    "US",
}


def stable_bin(value: str, bins: int = 16) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % bins


def dominant_root(meta: dict) -> str:
    roots = meta.get("top_event_roots") or []
    return str(roots[0][0]).zfill(2) if roots else "00"


def actor_features(meta: dict, bins: int = 16) -> tuple[list[float], float, float]:
    vec = [0.0] * bins
    actors = meta.get("top_actors") or []
    total = sum(float(x[1]) for x in actors) or 1.0
    for actor, count in actors:
        vec[stable_bin(str(actor), bins)] += float(count) / total
    concentration = sum((float(x[1]) / total) ** 2 for x in actors)
    return vec, float(len(actors)), concentration


def read_country_days(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(root.glob("*/country_day_sequences.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    meta = json.loads(line)["meta"]
                    day = pd.to_datetime(str(meta["date"]), format="%Y%m%d")
                    actor_vec, actor_unique, actor_hhi = actor_features(meta)
                    roots = {str(k).zfill(2): float(v) for k, v in meta.get("top_event_roots", [])}
                    row = {
                        "country": str(meta["country"]),
                        "date": day,
                        "events": float(meta.get("events", 0)),
                        "mentions": float(meta.get("mentions", 0)),
                        "tone": float(meta.get("avg_tone", 0)),
                        "goldstein": float(meta.get("avg_goldstein", 0)),
                        "dominant_root": dominant_root(meta),
                        "actor_unique": actor_unique,
                        "actor_hhi": actor_hhi,
                    }
                    row.update({f"actor_hash_{i}": value for i, value in enumerate(actor_vec)})
                    row.update({f"root_{code}": roots.get(code, 0.0) for code in ROOTS})
                    records.append(row)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    if not records:
        raise RuntimeError(f"No country_day_sequences.jsonl records found under {root}")
    return pd.DataFrame.from_records(records)


def calendarize(frame: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    numeric = [c for c in frame.columns if c not in {"country", "date", "dominant_root"}]
    for country, group in frame.groupby("country", sort=False):
        group = group.sort_values("date").set_index("date")
        idx = pd.date_range(group.index.min(), group.index.max(), freq="D")
        group = group.reindex(idx)
        group["country"] = country
        group["dominant_root"] = group["dominant_root"].fillna("00")
        group[numeric] = group[numeric].fillna(0.0)
        group.index.name = "date"
        outputs.append(group.reset_index())
    return pd.concat(outputs, ignore_index=True)


def build_examples(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    frame = frame.sort_values(["country", "date"]).copy()
    grouped = frame.groupby("country", sort=False)
    for lag in (1, 2, 3, 7, 14, 30):
        frame[f"events_lag_{lag}"] = grouped["events"].shift(lag - 1)
    frame["events_mean_7"] = grouped["events"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    frame["events_mean_30"] = grouped["events"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    frame["mentions_mean_7"] = grouped["mentions"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    frame["target_events"] = grouped["events"].shift(-1)
    frame["target_root"] = grouped["dominant_root"].shift(-1)
    frame["target_date"] = frame["date"] + pd.Timedelta(days=1)
    frame = frame.dropna(subset=["target_events", "target_root"]).reset_index(drop=True)

    target_date = frame["target_date"]
    frame["year"] = target_date.dt.year
    frame["month_sin"] = np.sin(2 * np.pi * target_date.dt.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * target_date.dt.month / 12)
    frame["dow_sin"] = np.sin(2 * np.pi * target_date.dt.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * target_date.dt.dayofweek / 7)
    frame["country_hash"] = frame["country"].map(stable_bin)
    frame["global_north"] = frame["country"].isin(GLOBAL_NORTH_FIPS).astype(float)
    frame["media_ratio"] = frame["mentions"] / np.maximum(frame["events"], 1)

    lag = [c for c in frame if c.startswith("events_lag_")] + [
        "events_mean_7", "events_mean_30", "mentions_mean_7", "events", "mentions",
    ]
    date_cols = ["year", "month_sin", "month_cos", "dow_sin", "dow_cos"]
    actor = [c for c in frame if c.startswith("actor_hash_")] + ["actor_unique", "actor_hhi"]
    roots = [f"root_{x}" for x in ROOTS]
    context = ["tone", "goldstein", "country_hash", "global_north", "media_ratio"]
    full = lag + date_cols + actor + roots + context
    groups = {
        "date": date_cols,
        "actor": actor,
        "tone": ["tone"],
        "goldstein": ["goldstein"],
        "event_roots": roots,
        "country": ["country_hash", "global_north"],
        "media": ["mentions", "mentions_mean_7", "media_ratio"],
    }
    frame[full] = frame[full].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame, full, groups


def split_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    day = frame["target_date"]
    return {
        "train": (day < "2025-01-01").to_numpy(),
        "validation": ((day >= "2025-01-01") & (day < "2025-07-01")).to_numpy(),
        "test": (day >= "2025-07-01").to_numpy(),
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.maximum(pred, 0)
    return {
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "rmsle": float(math.sqrt(mean_squared_error(np.log1p(y), np.log1p(pred)))),
        "r2": float(r2_score(y, pred)),
    }


def multiclass_brier(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    pos = {c: i for i, c in enumerate(classes)}
    one_hot = np.zeros_like(proba)
    for row, label in enumerate(y):
        if label in pos:
            one_hot[row, pos[label]] = 1
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def ece_score(y: np.ndarray, proba: np.ndarray, classes: np.ndarray, bins: int = 15) -> float:
    pred_idx = proba.argmax(axis=1)
    pred = classes[pred_idx]
    confidence = proba.max(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, proba, labels=classes)),
        "brier": multiclass_brier(y, proba, classes),
        "ece_15": ece_score(y, proba, classes),
    }


def cluster_bootstrap_rmse(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, clusters: np.ndarray,
    seed: int, reps: int,
) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    indices = {c: np.flatnonzero(clusters == c) for c in unique}
    delta = []
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([indices[c] for c in sampled])
        a = math.sqrt(mean_squared_error(y[idx], pred_a[idx]))
        b = math.sqrt(mean_squared_error(y[idx], pred_b[idx]))
        delta.append(a - b)
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return {
        "delta_rmse_a_minus_b": float(np.mean(delta)),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": float(min(
            1.0,
            2 * (min(np.sum(np.array(delta) <= 0), np.sum(np.array(delta) >= 0)) + 1)
            / (reps + 1),
        )),
        "bootstrap_unit": "country",
        "repetitions": reps,
    }


def fit_count_models(frame: pd.DataFrame, features: list[str], masks: dict, seeds: tuple[int, ...]) -> dict:
    X = frame[features].to_numpy(np.float32)
    y = frame["target_events"].to_numpy(np.float32)
    test = masks["test"]
    outputs = {
        "persistence": regression_metrics(y[test], frame.loc[test, "events"].to_numpy()),
        "seven_day_mean": regression_metrics(y[test], frame.loc[test, "events_mean_7"].to_numpy()),
        "seeds": [],
    }
    predictions = []
    for seed in seeds:
        model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(X[masks["train"]], y[masks["train"]])
        pred = np.maximum(model.predict(X[test]), 0)
        predictions.append(pred)
        outputs["seeds"].append({"seed": seed, **regression_metrics(y[test], pred)})
    ensemble = np.mean(predictions, axis=0)
    outputs["ensemble"] = regression_metrics(y[test], ensemble)
    outputs["paired_bootstrap_vs_persistence"] = cluster_bootstrap_rmse(
        y[test], ensemble, frame.loc[test, "events"].to_numpy(),
        frame.loc[test, "country"].to_numpy(), seeds[0], 1000,
    )
    outputs["_ensemble_predictions"] = ensemble
    return outputs


def fit_type_models(frame: pd.DataFrame, features: list[str], masks: dict, seeds: tuple[int, ...]) -> dict:
    eligible = frame["target_root"].isin(ROOTS).to_numpy()
    train = masks["train"] & eligible
    test = masks["test"] & eligible
    X = frame[features].to_numpy(np.float32)
    y = frame["target_root"].to_numpy()
    classes, counts = np.unique(y[train], return_counts=True)
    majority = classes[counts.argmax()]
    outputs = {
        "majority": {
            "accuracy": float(np.mean(y[test] == majority)),
            "macro_f1": float(f1_score(y[test], np.repeat(majority, test.sum()), average="macro", zero_division=0)),
        },
        "seeds": [],
        "test_rows": int(test.sum()),
    }
    all_proba = []
    for seed in seeds:
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(X[train], y[train])
        proba = model.predict_proba(X[test])
        pred = model.classes_[proba.argmax(axis=1)]
        all_proba.append(proba)
        outputs["seeds"].append({
            "seed": seed,
            **classification_metrics(y[test], pred, proba, model.classes_),
        })
    ensemble = np.mean(all_proba, axis=0)
    pred = model.classes_[ensemble.argmax(axis=1)]
    outputs["ensemble"] = classification_metrics(y[test], pred, ensemble, model.classes_)
    return outputs


def subgroup_regression(frame: pd.DataFrame, test: np.ndarray, pred: np.ndarray) -> dict:
    part = frame.loc[test, ["target_events", "events", "country", "global_north", "mentions"]].copy()
    part["prediction"] = pred
    part["north_south"] = np.where(part["global_north"] == 1, "Global North", "Global South")
    part["media_density"] = pd.qcut(part["mentions"].rank(method="first"), 3, labels=["low", "medium", "high"])
    result = {}
    for column in ("north_south", "media_density"):
        result[column] = {}
        for key, group in part.groupby(column, observed=True):
            result[column][str(key)] = {
                "n": int(len(group)),
                **regression_metrics(group["target_events"].to_numpy(), group["prediction"].to_numpy()),
            }
    country_rows = []
    for country, group in part.groupby("country"):
        if len(group) >= 30:
            country_rows.append({
                "country": country,
                "n": int(len(group)),
                **regression_metrics(group["target_events"].to_numpy(), group["prediction"].to_numpy()),
            })
    result["countries"] = sorted(country_rows, key=lambda x: -x["rmse"])
    return result


def heldout_country_test(frame: pd.DataFrame, features: list[str], masks: dict, seed: int) -> dict:
    countries = sorted(frame["country"].unique())
    heldout = {c for c in countries if stable_bin(f"{seed}:{c}", 5) == 0}
    train = masks["train"] & ~frame["country"].isin(heldout).to_numpy()
    test = masks["test"] & frame["country"].isin(heldout).to_numpy()
    X = frame[features].to_numpy(np.float32)
    y = frame["target_events"].to_numpy(np.float32)
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed,
    ).fit(X[train], y[train])
    pred = np.maximum(model.predict(X[test]), 0)
    return {
        "heldout_countries": sorted(heldout),
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        **regression_metrics(y[test], pred),
    }


def ablation_suite(frame: pd.DataFrame, full: list[str], groups: dict, masks: dict, seed: int) -> dict:
    y = frame["target_events"].to_numpy(np.float32)
    test = masks["test"]
    result = {}
    for name in ("full", *[f"without_{x}" for x in groups]):
        removed = set() if name == "full" else set(groups[name.removeprefix("without_")])
        features = [x for x in full if x not in removed]
        model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        ).fit(frame.loc[masks["train"], features], y[masks["train"]])
        pred = np.maximum(model.predict(frame.loc[test, features]), 0)
        result[name] = {"features": len(features), **regression_metrics(y[test], pred)}
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="7,17,29")
    args = p.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    print("Loading country-day streams...", flush=True)
    raw = read_country_days(Path(args.data_root))
    frame, full, groups = build_examples(calendarize(raw))
    masks = split_masks(frame)
    print(f"rows={len(frame)} train={masks['train'].sum()} validation={masks['validation'].sum()} test={masks['test'].sum()}", flush=True)

    count = fit_count_models(frame, full, masks, seeds)
    ensemble = count.pop("_ensemble_predictions")
    result = {
        "protocol": {
            "train": "target date before 2025-01-01",
            "validation": "2025-01-01 through 2025-06-30",
            "test": "2025-07-01 onward",
            "forecast_horizon": "next calendar day",
            "seeds": list(seeds),
        },
        "data": {
            "country_day_rows_observed": int(len(raw)),
            "examples_calendarized": int(len(frame)),
            "countries": int(frame["country"].nunique()),
        },
        "event_volume": count,
        "event_type": fit_type_models(frame, full, masks, seeds),
        "country_heldout": [heldout_country_test(frame, full, masks, seed) for seed in seeds],
        "ablations": ablation_suite(frame, full, groups, masks, seeds[0]),
        "fragility": subgroup_regression(frame, masks["test"], ensemble),
        "limitations": [
            "Global North/South is an explicit coarse grouping and must be sensitivity-tested.",
            "Source-host ablation is evaluated separately on event-level streams.",
            "Country-day missingness is interpreted as zero observed GDELT events.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "event_volume": result["event_volume"]["ensemble"],
        "event_type": result["event_type"]["ensemble"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
