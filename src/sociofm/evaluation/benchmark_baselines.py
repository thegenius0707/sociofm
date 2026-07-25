#!/usr/bin/env python3
"""Temporal baselines for next-day societal event forecasting."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = defaultdict(dict)
    for path in sorted(Path(args.data_root).glob("*/country_day_sequences.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    meta = json.loads(line)["meta"]
                    country, day = meta["country"], meta["date"]
                    rows[country][day] = meta
                except (json.JSONDecodeError, KeyError):
                    continue

    errors_prev, errors_week = [], []
    evaluated = 0
    for country, series in rows.items():
        observed = sorted(series)
        if len(observed) < 2:
            continue
        first, last = date.fromisoformat(observed[0][:4] + "-" + observed[0][4:6] + "-" + observed[0][6:]) , date.fromisoformat(observed[-1][:4] + "-" + observed[-1][4:6] + "-" + observed[-1][6:])
        calendar_days = []
        cursor = first
        while cursor <= last:
            calendar_days.append(cursor.strftime("%Y%m%d"))
            cursor += timedelta(days=1)
        values = {d: float(series[d].get("events", 0)) if d in series else 0.0 for d in calendar_days}
        for i in range(1, len(calendar_days)):
            prev_day, target_day = calendar_days[i - 1], calendar_days[i]
            y = values[target_day]
            pred_prev = values[prev_day]
            window = [values[d] for d in calendar_days[max(0, i - 7):i]]
            pred_week = sum(window) / len(window)
            errors_prev.append((pred_prev - y) ** 2)
            errors_week.append((pred_week - y) ** 2)
            evaluated += 1

    result = {
        "countries": len(rows),
        "country_day_rows": sum(len(x) for x in rows.values()),
        "next_day_pairs": evaluated,
        "persistence_rmse": math.sqrt(sum(errors_prev) / max(len(errors_prev), 1)),
        "seven_day_mean_rmse": math.sqrt(sum(errors_week) / max(len(errors_week), 1)),
        "note": "Chronological next-day count baselines; no future information used.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
