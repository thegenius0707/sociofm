#!/usr/bin/env python3
"""GPU audit: temporal/source/country fragility and field ablations for SocioFM."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

FIELDS = ("DATE", "COUNTRY", "ACTOR1", "ACTOR2", "EVENT_CODE", "EVENT_ROOT",
          "QUAD", "GOLDSTEIN", "MENTIONS", "SOURCES", "ARTICLES", "TONE")
NORTH = {"AS", "AU", "BE", "CA", "DA", "EI", "FI", "FR", "GM", "GR", "IC",
         "IT", "JA", "LU", "NE", "NO", "NZ", "PO", "SP", "SW", "SZ", "UK", "US"}


def mask_field(text: str, field: str) -> str:
    return re.sub(rf"\[{re.escape(field)}=[^\]]*\]", f"[{field}=MASK]", text)


def load_sample(root: Path, year: str, per_month: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for path in sorted(root.glob(f"{year}-*/event_sequences.jsonl")):
        reservoir = []
        seen = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen += 1
                if len(reservoir) < per_month:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < per_month:
                        reservoir[j] = item
        for item in reservoir:
            meta = item.get("meta", {})
            rows.append({
                "text": item["text"],
                "month": path.parent.name,
                "country": meta.get("country") or "UNK",
                "source": meta.get("source_host") or "unknown",
                "event_root": str(meta.get("event_root_code") or "00").zfill(2),
            })
    return rows


def score_texts(model, tok, texts: list[str], batch_size: int, device: str):
    sums, tokens, per_item = 0.0, 0, []
    for start in range(0, len(texts), batch_size):
        batch = tok(texts[start:start + batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device == "cuda"
        ):
            logits = model(**batch).logits[:, :-1]
        labels = batch["input_ids"][:, 1:]
        mask = batch["attention_mask"][:, 1:]
        loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        item_sum = (loss * mask).sum(1)
        item_n = mask.sum(1)
        sums += float(item_sum.sum())
        tokens += int(item_n.sum())
        per_item.extend((item_sum / item_n.clamp_min(1)).float().cpu().tolist())
    mean = sums / max(tokens, 1)
    return {"loss": mean, "perplexity": math.exp(min(mean, 20)), "tokens": tokens}, per_item


def grouped(rows, losses, key: str, minimum: int = 30):
    stats = defaultdict(list)
    for row, loss in zip(rows, losses):
        stats[row[key]].append(loss)
    return sorted(
        [{"group": k, "n": len(v), "loss": sum(v) / len(v),
          "perplexity": math.exp(min(sum(v) / len(v), 20))}
         for k, v in stats.items() if len(v) >= minimum],
        key=lambda x: (-x["n"], x["group"]),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--year", default="2025")
    p.add_argument("--events-per-month", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else None
    ).to(device).eval()
    rows = load_sample(Path(args.data_root), args.year, args.events_per_month, args.seed)
    base, losses = score_texts(model, tok, [x["text"] for x in rows], args.batch_size, device)
    for row in rows:
        row["north_south"] = "Global North" if row["country"] in NORTH else "Global South"
    ablations = {}
    for field in FIELDS:
        print(f"Scoring ablation: {field}", flush=True)
        result, _ = score_texts(
            model, tok, [mask_field(x["text"], field) for x in rows],
            args.batch_size, device,
        )
        result["delta_loss"] = result["loss"] - base["loss"]
        ablations[field] = result
    result = {
        "model": args.model,
        "year": args.year,
        "sample_events": len(rows),
        "sampling": "per-month reservoir sampling",
        "base": base,
        "ablations": ablations,
        "monthly": grouped(rows, losses, "month"),
        "countries": grouped(rows, losses, "country"),
        "sources": grouped(rows, losses, "source"),
        "event_roots": grouped(rows, losses, "event_root"),
        "north_south": grouped(rows, losses, "north_south"),
        "device": device,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"base": base, "north_south": result["north_south"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
