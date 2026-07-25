#!/usr/bin/env python3
"""Evaluate a SocioFM checkpoint on a bounded streaming split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, GPT2LMHeadModel


class EventStream(IterableDataset):
    def __init__(self, manifest: str, tokenizer, block_size: int):
        self.paths = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.tokenizer, self.block_size = tokenizer, block_size

    def __iter__(self):
        for path in self.paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        text = json.loads(line)["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue
                    ids = self.tokenizer(text, truncation=True, max_length=self.block_size)["input_ids"]
                    if len(ids) >= 8:
                        yield {"input_ids": ids}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=1000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(EventStream(args.manifest, tokenizer, args.block_size),
                        batch_size=args.batch_size, collate_fn=collator)
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            total_loss += float(loss)
            batches += 1
            if batches >= args.max_batches:
                break
    mean_loss = total_loss / max(batches, 1)
    print(json.dumps({"batches": batches, "loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
