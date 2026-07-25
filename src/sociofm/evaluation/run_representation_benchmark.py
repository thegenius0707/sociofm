#!/usr/bin/env python3
"""Frozen-representation downstream benchmark for SocioFM and open LMs.

Tasks:
  1. next-day event-volume forecasting (representation-only and hybrid);
  2. next-day dominant event-root classification;
  3. validation-fitted temperature calibration;
  4. Global North/South and media-density subgroup performance.

All models use the same probe family and chronological splits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from sociofm.evaluation.run_downstream_suite import (
    GLOBAL_NORTH_FIPS,
    ROOTS,
    classification_metrics,
    regression_metrics,
)


def dominant_root(meta: dict) -> str:
    roots = meta.get("top_event_roots") or []
    return str(roots[0][0]).zfill(2) if roots else "00"


def zero_text(country: str, day: pd.Timestamp) -> str:
    return (
        f"[COUNTRY={country}] [DATE={day.strftime('%Y%m%d')}] [EVENTS=0] "
        "[MENTIONS=0] [AVG_TONE=0] [AVG_GOLDSTEIN=0] "
        "[TOP_EVENT_ROOTS=[]] [TOP_ACTORS=[]]"
    )


def load_examples(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(root.glob("*/country_day_sequences.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    meta = obj["meta"]
                    records.append({
                        "country": str(meta["country"]),
                        "date": pd.to_datetime(str(meta["date"]), format="%Y%m%d"),
                        "text": obj["text"],
                        "events": float(meta.get("events", 0)),
                        "mentions": float(meta.get("mentions", 0)),
                        "tone": float(meta.get("avg_tone", 0)),
                        "goldstein": float(meta.get("avg_goldstein", 0)),
                        "root": dominant_root(meta),
                    })
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    raw = pd.DataFrame(records)
    if raw.empty:
        raise RuntimeError(f"No country-day streams under {root}")

    outputs = []
    for country, group in raw.groupby("country", sort=False):
        group = group.sort_values("date").set_index("date")
        idx = pd.date_range(group.index.min(), group.index.max(), freq="D")
        group = group.reindex(idx)
        group["country"] = country
        for col in ("events", "mentions", "tone", "goldstein"):
            group[col] = group[col].fillna(0.0)
        group["root"] = group["root"].fillna("00")
        missing = group["text"].isna()
        group.loc[missing, "text"] = [zero_text(country, day) for day in group.index[missing]]
        group.index.name = "date"
        outputs.append(group.reset_index())
    frame = pd.concat(outputs, ignore_index=True).sort_values(["country", "date"])

    grouped = frame.groupby("country", sort=False)
    frame["target_events"] = grouped["events"].shift(-1)
    frame["target_root"] = grouped["root"].shift(-1)
    frame["target_date"] = frame["date"] + pd.Timedelta(days=1)
    frame["events_mean_7"] = grouped["events"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    frame["events_mean_30"] = grouped["events"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    frame["mentions_mean_7"] = grouped["mentions"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    frame = frame.dropna(subset=["target_events", "target_root"]).reset_index(drop=True)
    frame["north_south"] = np.where(
        frame["country"].isin(GLOBAL_NORTH_FIPS), "Global North", "Global South"
    )
    frame["media_density"] = pd.qcut(
        frame["mentions"].rank(method="first"), 3, labels=["low", "medium", "high"]
    ).astype(str)
    return frame


def chronological_masks(frame: pd.DataFrame):
    day = frame["target_date"]
    return {
        "train": (day < "2025-01-01").to_numpy(),
        "validation": ((day >= "2025-01-01") & (day < "2025-07-01")).to_numpy(),
        "test": (day >= "2025-07-01").to_numpy(),
    }


def base_model(model):
    if hasattr(model, "transformer"):
        return model.transformer
    base = getattr(model, "base_model", None)
    if base is None:
        raise RuntimeError("Could not locate the model's base transformer")
    return base


def encode(
    model_path: str, texts: list[str], batch_size: int, device: str,
    cache: Path | None,
) -> np.ndarray:
    if cache and cache.exists():
        print(f"Loading cached embeddings: {cache}", flush=True)
        return np.load(cache, mmap_mode="r")["embeddings"]
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    encoder = base_model(model)
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tok(
            texts[start:start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device == "cuda"
        ):
            hidden = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=1)
        chunks.append(pooled.cpu().numpy().astype(np.float16))
        if start % (batch_size * 100) == 0:
            print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    matrix = np.concatenate(chunks)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embeddings=matrix)
    del model, encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    return matrix


def numeric_context(frame: pd.DataFrame) -> np.ndarray:
    day = frame["target_date"]
    values = np.column_stack([
        np.log1p(frame["events"]),
        np.log1p(frame["events_mean_7"]),
        np.log1p(frame["events_mean_30"]),
        np.log1p(frame["mentions"]),
        np.log1p(frame["mentions_mean_7"]),
        frame["tone"],
        frame["goldstein"],
        np.sin(2 * np.pi * day.dt.month / 12),
        np.cos(2 * np.pi * day.dt.month / 12),
        np.sin(2 * np.pi * day.dt.dayofweek / 7),
        np.cos(2 * np.pi * day.dt.dayofweek / 7),
    ])
    return values.astype(np.float32)


def temperature_scale(proba: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(proba, 1e-9, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    return out / out.sum(axis=1, keepdims=True)


def choose_temperature(y, proba, classes) -> float:
    position = {label: i for i, label in enumerate(classes)}
    idx = np.array([position.get(label, -1) for label in y])
    keep = idx >= 0
    best_t, best_nll = 1.0, float("inf")
    for temperature in np.linspace(0.5, 4.0, 71):
        scaled = temperature_scale(proba[keep], float(temperature))
        nll = -np.mean(np.log(np.clip(scaled[np.arange(keep.sum()), idx[keep]], 1e-9, 1)))
        if nll < best_nll:
            best_t, best_nll = float(temperature), float(nll)
    return best_t


def align_probabilities(
    proba: np.ndarray, source_classes: np.ndarray, target_classes: np.ndarray
) -> np.ndarray:
    """Align classifier probabilities, retaining unseen temporal classes."""
    aligned = np.full((len(proba), len(target_classes)), 1e-9, dtype=np.float64)
    target_pos = {label: i for i, label in enumerate(target_classes)}
    for source_col, label in enumerate(source_classes):
        if label in target_pos:
            aligned[:, target_pos[label]] = proba[:, source_col]
    return aligned / aligned.sum(axis=1, keepdims=True)


def volume_probe(X: np.ndarray, frame: pd.DataFrame, masks, seeds) -> dict:
    current = frame["events"].to_numpy(np.float32)
    y = frame["target_events"].to_numpy(np.float32)
    delta = np.log1p(y) - np.log1p(current)
    test = masks["test"]
    results = {
        "persistence": regression_metrics(y[test], current[test]),
        "representation_only": [],
        "hybrid": [],
    }
    numeric = numeric_context(frame)
    scaler = StandardScaler().fit(numeric[masks["train"]])
    numeric = scaler.transform(numeric).astype(np.float32)
    variants = {
        "representation_only": np.asarray(X, dtype=np.float32),
        "hybrid": np.column_stack([np.asarray(X, dtype=np.float32), numeric]),
    }
    predictions = {}
    for variant, matrix in variants.items():
        predictions[variant] = []
        for seed in seeds:
            reg = SGDRegressor(
                loss="huber", penalty="l2", alpha=1e-4, max_iter=300,
                tol=1e-4, random_state=seed, early_stopping=False,
            ).fit(matrix[masks["train"]], delta[masks["train"]])
            change = np.clip(reg.predict(matrix[test]), -4.0, 4.0)
            pred = np.maximum(np.expm1(np.log1p(current[test]) + change), 0)
            predictions[variant].append(pred)
            results[variant].append({"seed": seed, **regression_metrics(y[test], pred)})
        ensemble = np.mean(predictions[variant], axis=0)
        results[f"{variant}_ensemble"] = regression_metrics(y[test], ensemble)
        results[f"_{variant}_prediction"] = ensemble
    return results


def type_probe(X: np.ndarray, frame: pd.DataFrame, masks, seeds) -> dict:
    y = frame["target_root"].to_numpy()
    eligible = np.isin(y, ROOTS)
    train, validation, test = (
        masks["train"] & eligible,
        masks["validation"] & eligible,
        masks["test"] & eligible,
    )
    matrix = np.asarray(X, dtype=np.float32)
    all_classes = np.array(ROOTS)
    test_proba, rows = [], []
    for seed in seeds:
        clf = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-5, max_iter=300,
            tol=1e-4, class_weight="balanced", random_state=seed,
        ).fit(matrix[train], y[train])
        val_proba = align_probabilities(
            clf.predict_proba(matrix[validation]), clf.classes_, all_classes
        )
        temperature = choose_temperature(y[validation], val_proba, all_classes)
        proba = align_probabilities(
            clf.predict_proba(matrix[test]), clf.classes_, all_classes
        )
        proba = temperature_scale(proba, temperature)
        pred = all_classes[proba.argmax(axis=1)]
        test_proba.append(proba)
        rows.append({
            "seed": seed,
            "temperature": temperature,
            **classification_metrics(y[test], pred, proba, all_classes),
        })
    ensemble = np.mean(test_proba, axis=0)
    pred = all_classes[ensemble.argmax(axis=1)]
    return {
        "test_rows": int(test.sum()),
        "seeds": rows,
        "ensemble": classification_metrics(y[test], pred, ensemble, all_classes),
    }


def subgroup_volume(frame: pd.DataFrame, mask: np.ndarray, pred: np.ndarray) -> dict:
    part = frame.loc[mask, ["target_events", "north_south", "media_density"]].copy()
    part["prediction"] = pred
    result = {}
    for key in ("north_south", "media_density"):
        result[key] = {}
        for group, rows in part.groupby(key):
            result[key][str(group)] = {
                "n": int(len(rows)),
                **regression_metrics(rows["target_events"].to_numpy(), rows["prediction"].to_numpy()),
            }
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--embedding-cache", default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--seeds", default="7,17,29")
    args = p.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    frame = load_examples(Path(args.data_root))
    if args.max_examples:
        initial = chronological_masks(frame)
        quotas = {
            "train": int(args.max_examples * 0.6),
            "validation": int(args.max_examples * 0.2),
            "test": args.max_examples - int(args.max_examples * 0.8),
        }
        pieces = []
        for split, mask in initial.items():
            part = frame.loc[mask]
            pieces.append(part.sample(
                n=min(quotas[split], len(part)), random_state=7
            ))
        frame = pd.concat(pieces, ignore_index=True).sort_values(
            ["country", "date"]
        ).reset_index(drop=True)
    masks = chronological_masks(frame)
    print(
        f"examples={len(frame)} train={masks['train'].sum()} "
        f"validation={masks['validation'].sum()} test={masks['test'].sum()}",
        flush=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(args.embedding_cache) if args.embedding_cache else None
    X = encode(args.model, frame["text"].tolist(), args.batch_size, device, cache)
    volume = volume_probe(X, frame, masks, seeds)
    rep_pred = volume.pop("_representation_only_prediction")
    hybrid_pred = volume.pop("_hybrid_prediction")
    result = {
        "model_name": args.model_name,
        "model": args.model,
        "examples": len(frame),
        "splits": {k: int(v.sum()) for k, v in masks.items()},
        "event_volume": volume,
        "event_type": type_probe(X, frame, masks, seeds),
        "subgroups": {
            "representation_only": subgroup_volume(frame, masks["test"], rep_pred),
            "hybrid": subgroup_volume(frame, masks["test"], hybrid_pred),
        },
        "protocol": (
            "Frozen mean-pooled normalized representations; identical linear probes; "
            "train before 2025, validation 2025H1, test 2025H2."
        ),
        "device": device,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "model": args.model_name,
        "volume_representation": volume["representation_only_ensemble"],
        "volume_hybrid": volume["hybrid_ensemble"],
        "event_type": result["event_type"]["ensemble"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
