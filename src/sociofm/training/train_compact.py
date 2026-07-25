#!/usr/bin/env python3
"""Train a compact causal Transformer directly from JSONL event shards."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments)


class EventStream(IterableDataset):
    def __init__(self, manifest: str, tokenizer, block_size: int, seed: int = 7):
        self.paths = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.tokenizer, self.block_size, self.seed = tokenizer, block_size, seed

    def __iter__(self):
        paths = list(self.paths)
        random.Random(self.seed + int(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0)).shuffle(paths)
        for path in paths:
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
    p.add_argument("--manifests", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--resume-from-checkpoint", default=None)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config = GPT2Config(vocab_size=len(tok), n_positions=args.block_size, n_ctx=args.block_size,
                        n_embd=args.hidden, n_layer=args.layers, n_head=args.heads,
                        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id)
    model = GPT2LMHeadModel(config)
    train = EventStream(str(Path(args.manifests) / "train.json"), tok, args.block_size)
    valid = EventStream(str(Path(args.manifests) / "validation.json"), tok, args.block_size, 17)
    training = TrainingArguments(output_dir=args.output, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, eval_strategy="no",
        save_strategy="steps", save_steps=2000, logging_steps=100, learning_rate=3e-4,
        warmup_steps=2000, weight_decay=0.1, bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(), report_to="none")
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    Trainer(model=model, args=training, train_dataset=train, eval_dataset=valid,
            data_collator=collator).train(
                resume_from_checkpoint=args.resume_from_checkpoint
            )
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
