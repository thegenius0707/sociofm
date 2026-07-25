#!/usr/bin/env python3
"""Final pre-figure statistics for SocioFM.

Produces probabilistic forecast metrics (CRPS approximation and interval
coverage), paired country-cluster bootstrap comparisons across frozen
representations, and compute/energy/cost accounting.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM

from sociofm.evaluation.run_representation_benchmark import (
    chronological_masks,
    load_examples,
    numeric_context,
)
from sociofm.evaluation.run_downstream_suite import regression_metrics


MODEL_CACHES = {
    "sociofm_25m_50k": "embeddings_25m_50k_full.npz",
    "sociofm_25m_100k": "embeddings_sociofm_25m_100k_full.npz",
    "sociofm_50m_100k": "embeddings_sociofm_50m_100k_full.npz",
    "sociofm_100m_100k": "embeddings_sociofm_100m_100k_full.npz",
    "sociofm_100m_200k": "embeddings_sociofm_100m_200k_full.npz",
    "distilgpt2_frozen": "embeddings_distilgpt2_frozen_full.npz",
    "gpt2_frozen": "embeddings_gpt2_frozen_full.npz",
}

COMPUTE_RUNS = {
    "sociofm_25m_50k": {
        "path": "checkpoints/sociofm_25m/checkpoint-50000",
        "steps": 50_000, "gpu_hours": 2.89 / 2, "avg_gpu_watts": 86,
    },
    "sociofm_25m_100k": {
        "path": "checkpoints/sociofm_25m",
        "steps": 100_000, "gpu_hours": 2.89, "avg_gpu_watts": 86,
    },
    "sociofm_50m_100k": {
        "path": "checkpoints/sociofm_50m",
        "steps": 100_000, "gpu_hours": 10.686, "avg_gpu_watts": 89,
    },
    "sociofm_100m_100k": {
        "path": "checkpoints/sociofm_100m/checkpoint-100000",
        "steps": 100_000, "gpu_hours": 12.198, "avg_gpu_watts": 81,
    },
    "sociofm_100m_200k": {
        "path": "checkpoints/sociofm_100m",
        "steps": 200_000, "gpu_hours": 27.386, "avg_gpu_watts": 81,
    },
}


def train_hybrid_predictions(frame, masks, cache_path: Path, seeds):
    embeddings = np.load(cache_path, mmap_mode="r")["embeddings"]
    numeric = numeric_context(frame)
    scaler = StandardScaler().fit(numeric[masks["train"]])
    numeric = scaler.transform(numeric).astype(np.float32)
    matrix = np.column_stack([np.asarray(embeddings, dtype=np.float32), numeric])
    current = frame["events"].to_numpy(np.float32)
    y = frame["target_events"].to_numpy(np.float32)
    delta = np.log1p(y) - np.log1p(current)
    predictions = []
    seed_metrics = []
    for seed in seeds:
        model = SGDRegressor(
            loss="huber", penalty="l2", alpha=1e-4, max_iter=300,
            tol=1e-4, random_state=seed, early_stopping=False,
        ).fit(matrix[masks["train"]], delta[masks["train"]])
        change = np.clip(model.predict(matrix[masks["test"]]), -4, 4)
        pred = np.maximum(
            np.expm1(np.log1p(current[masks["test"]]) + change), 0
        )
        predictions.append(pred)
        seed_metrics.append({"seed": seed, **regression_metrics(y[masks["test"]], pred)})
    ensemble = np.mean(predictions, axis=0)
    del embeddings, numeric, matrix
    gc.collect()
    return ensemble, seed_metrics


def cluster_bootstrap_difference(
    y, pred_a, pred_b, clusters, seed=7, reps=2000, metric="rmse",
):
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    index = {c: np.flatnonzero(clusters == c) for c in unique}
    values = []
    for _ in range(reps):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([index[c] for c in sampled])
        if metric == "rmse":
            a = math.sqrt(mean_squared_error(y[idx], pred_a[idx]))
            b = math.sqrt(mean_squared_error(y[idx], pred_b[idx]))
        else:
            a = float(np.mean(pred_a[idx]))
            b = float(np.mean(pred_b[idx]))
        values.append(a - b)
    values = np.asarray(values)
    low, high = np.percentile(values, [2.5, 97.5])
    smaller_tail = min(np.sum(values <= 0), np.sum(values >= 0))
    p = min(1.0, 2 * (smaller_tail + 1) / (reps + 1))
    return {
        "mean_difference_a_minus_b": float(values.mean()),
        "ci95": [float(low), float(high)],
        "p_two_sided": float(p),
        "repetitions": reps,
        "bootstrap_unit": "country",
    }


def pinball(y, q, tau):
    error = y - q
    return np.maximum(tau * error, (tau - 1) * error)


def probabilistic_forecast(frame, masks, seed=7):
    features = numeric_context(frame)
    scaler = StandardScaler().fit(features[masks["train"]])
    features = scaler.transform(features).astype(np.float32)
    current = frame["events"].to_numpy(np.float32)
    y = frame["target_events"].to_numpy(np.float32)
    delta = np.log1p(y) - np.log1p(current)
    taus = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    quantiles = []
    for tau in taus:
        print(f"CRPS quantile {tau:.2f}", flush=True)
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=float(tau), max_iter=250,
            learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=seed,
        ).fit(features[masks["train"]], delta[masks["train"]])
        change = np.clip(model.predict(features[masks["test"]]), -4, 4)
        quantiles.append(np.maximum(
            np.expm1(np.log1p(current[masks["test"]]) + change), 0
        ))
    q = np.sort(np.column_stack(quantiles), axis=1)
    yt = y[masks["test"]]
    losses = np.column_stack([pinball(yt, q[:, i], tau) for i, tau in enumerate(taus)])
    crps_row = 2 * np.trapezoid(losses, taus, axis=1)
    persistence_crps = np.abs(yt - current[masks["test"]])
    q10, q50, q90 = q[:, 2], q[:, 4], q[:, 6]
    return {
        "quantile_grid": taus.tolist(),
        "approximation": "2 x trapezoidal integral of pinball loss over quantile grid",
        "crps_mean": float(crps_row.mean()),
        "persistence_deterministic_crps_mean": float(persistence_crps.mean()),
        "crps_skill_vs_persistence": float(1 - crps_row.mean() / persistence_crps.mean()),
        "central_80_coverage": float(np.mean((yt >= q10) & (yt <= q90))),
        "central_80_mean_width": float(np.mean(q90 - q10)),
        "median_forecast": regression_metrics(yt, q50),
        "_crps_row": crps_row,
        "_persistence_crps_row": persistence_crps,
    }


def compute_accounting(gpu_price, pue, avg_tokens_per_sequence):
    rows = []
    for name, spec in COMPUTE_RUNS.items():
        model = AutoModelForCausalLM.from_pretrained(spec["path"])
        parameters = sum(x.numel() for x in model.parameters())
        del model
        gc.collect()
        gpu_energy = spec["gpu_hours"] * spec["avg_gpu_watts"] / 1000
        rows.append({
            "model": name,
            "parameters": int(parameters),
            "steps": spec["steps"],
            "effective_batch_sequences": 32,
            "estimated_token_exposures": int(
                spec["steps"] * 32 * avg_tokens_per_sequence
            ),
            "gpu_hours": spec["gpu_hours"],
            "estimated_average_gpu_watts": spec["avg_gpu_watts"],
            "gpu_energy_kwh": gpu_energy,
            "pue_adjusted_energy_kwh": gpu_energy * pue,
            "gpu_cost_usd": spec["gpu_hours"] * gpu_price,
        })
    return {
        "runs": rows,
        "assumptions": {
            "gpu_price_usd_per_hour": gpu_price,
            "pue": pue,
            "average_tokens_per_sequence": avg_tokens_per_sequence,
            "energy_scope": "GPU-only estimate and PUE-adjusted estimate; excludes embodied carbon",
            "token_scope": "estimated exposures from observed mean evaluation sequence length",
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="7,17,29")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--gpu-price", type=float, default=0.60)
    p.add_argument("--pue", type=float, default=1.2)
    args = p.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    results_root = Path(args.results_root)
    frame = load_examples(Path(args.data_root))
    masks = chronological_masks(frame)
    y = frame.loc[masks["test"], "target_events"].to_numpy(np.float32)
    clusters = frame.loc[masks["test"], "country"].to_numpy()

    predictions, seed_metrics = {}, {}
    for name, filename in MODEL_CACHES.items():
        cache = results_root / filename
        if not cache.exists():
            print(f"SKIP missing cache: {cache}", flush=True)
            continue
        print(f"Hybrid predictions: {name}", flush=True)
        predictions[name], seed_metrics[name] = train_hybrid_predictions(
            frame, masks, cache, seeds
        )

    reference = "sociofm_25m_50k"
    comparisons = {}
    if reference in predictions:
        for name, pred in predictions.items():
            if name == reference:
                continue
            comparisons[f"{reference}_vs_{name}"] = cluster_bootstrap_difference(
                y, predictions[reference], pred, clusters,
                seed=seeds[0], reps=args.bootstrap_reps, metric="rmse",
            )

    probabilistic = probabilistic_forecast(frame, masks, seeds[0])
    crps = probabilistic.pop("_crps_row")
    persistence_crps = probabilistic.pop("_persistence_crps_row")
    probabilistic["paired_country_bootstrap_vs_persistence"] = (
        cluster_bootstrap_difference(
            y, crps, persistence_crps, clusters, seed=seeds[0],
            reps=args.bootstrap_reps, metric="mean",
        )
    )
    result = {
        "paired_representation_bootstrap": comparisons,
        "representation_seed_metrics": seed_metrics,
        "probabilistic_forecast": probabilistic,
        "compute_accounting": compute_accounting(
            args.gpu_price, args.pue, 6_026_894 / 60_000
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "probabilistic_forecast": probabilistic,
        "paired_comparisons": comparisons,
    }, indent=2))


if __name__ == "__main__":
    main()
