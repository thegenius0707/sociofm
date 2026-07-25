#!/usr/bin/env python3
"""SocioFM Figure 4 v2: downstream transfer, calibration and evidence strength.

Panels:
  a) multi-task frozen-representation transfer landscape;
  b) representation-only to hybrid context-fusion gains;
  c) event-type calibration landscape;
  d) seed-level stability of hybrid event-volume probes;
  e) paired country-cluster bootstrap forest plot;
  f) model-field ablation matrix.

Exports a 600-DPI PNG and an editable vector PDF. No footer and no internal
"Figure 4" label are added.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------- Shared visual system ----------
INK = "#202A35"
MUTED = "#697785"
GRID = "#DCE3E8"
BLUE = "#174A7E"
BLUE_LIGHT = "#78AFCB"
TEAL = "#148F88"
PURPLE = "#7656A5"
CORAL = "#D65A4A"
GOLD = "#D49A32"
GREY = "#87939E"
LIGHT_GREY = "#C8D0D7"

ORDER = [
    "sociofm_25m_50k",
    "sociofm_25m_100k",
    "sociofm_50m_100k",
    "sociofm_100m_100k",
    "sociofm_100m_200k",
    "distilgpt2_frozen",
    "gpt2_frozen",
]

SOCIO_ORDER = ORDER[:5]

LABELS = {
    "sociofm_25m_50k": "25M · 50k",
    "sociofm_25m_100k": "25M · 100k",
    "sociofm_50m_100k": "50M · 100k",
    "sociofm_100m_100k": "101M · 100k",
    "sociofm_100m_200k": "101M · 200k",
    "distilgpt2_frozen": "DistilGPT-2",
    "gpt2_frozen": "GPT-2",
}

SHORT = {
    "sociofm_25m_50k": "S25/50",
    "sociofm_25m_100k": "S25/100",
    "sociofm_50m_100k": "S50/100",
    "sociofm_100m_100k": "S101/100",
    "sociofm_100m_200k": "S101/200",
    "distilgpt2_frozen": "Distil",
    "gpt2_frozen": "GPT-2",
}

COLORS = {
    "sociofm_25m_50k": BLUE,
    "sociofm_25m_100k": BLUE_LIGHT,
    "sociofm_50m_100k": TEAL,
    "sociofm_100m_100k": PURPLE,
    "sociofm_100m_200k": CORAL,
    "distilgpt2_frozen": GREY,
    "gpt2_frozen": GOLD,
}

PARAMETERS_M = {
    "sociofm_25m_50k": 25.0,
    "sociofm_25m_100k": 25.0,
    "sociofm_50m_100k": 50.0,
    "sociofm_100m_100k": 101.25888,
    "sociofm_100m_200k": 101.25888,
    "distilgpt2_frozen": 82.0,
    "gpt2_frozen": 124.0,
}

REPRESENTATION_FILES = {
    "sociofm_25m_50k": [
        "representation_25m_50k.json",
        "representation_sociofm_25m_50k.json",
    ],
    "sociofm_25m_100k": ["representation_sociofm_25m_100k.json"],
    "sociofm_50m_100k": ["representation_sociofm_50m_100k.json"],
    "sociofm_100m_100k": ["representation_sociofm_100m_100k.json"],
    "sociofm_100m_200k": ["representation_sociofm_100m_200k.json"],
    "distilgpt2_frozen": ["representation_distilgpt2_frozen.json"],
    "gpt2_frozen": ["representation_gpt2_frozen.json"],
}

AUDIT_FILES = {
    "sociofm_25m_50k": "lm_audit_25m_50k.json",
    "sociofm_25m_100k": "lm_audit_25m_100k.json",
    "sociofm_50m_100k": "lm_audit_50m_100k.json",
    "sociofm_100m_100k": "lm_audit_100m_100k.json",
    "sociofm_100m_200k": "lm_audit_100m_200k.json",
}

FIELDS = [
    "DATE", "COUNTRY", "ACTOR1", "ACTOR2", "EVENT_CODE", "EVENT_ROOT",
    "QUAD", "GOLDSTEIN", "MENTIONS", "SOURCES", "ARTICLES", "TONE",
]

FIELD_LABELS = {
    "DATE": "Date",
    "COUNTRY": "Country",
    "ACTOR1": "Actor 1",
    "ACTOR2": "Actor 2",
    "EVENT_CODE": "Event code",
    "EVENT_ROOT": "Event root",
    "QUAD": "Quad",
    "GOLDSTEIN": "Goldstein",
    "MENTIONS": "Mentions",
    "SOURCES": "Sources",
    "ARTICLES": "Articles",
    "TONE": "Tone",
}

DIVERGING = LinearSegmentedColormap.from_list(
    "sociofm_ablation", [BLUE, "#C7DFE8", "#F7F7F4", "#F2C8BE", CORAL]
)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.edgecolor": INK,
        "axes.linewidth": .7,
        "axes.labelcolor": INK,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARN {path}: {exc}", flush=True)
        return {}


def load_first(root: Path, candidates: list[str]) -> dict:
    for filename in candidates:
        value = read_json(root / filename)
        if value:
            return value
    return {}


def load_representations(root: Path) -> dict[str, dict]:
    return {
        key: load_first(root, candidates)
        for key, candidates in REPRESENTATION_FILES.items()
    }


def load_audits(root: Path) -> dict[str, dict]:
    return {
        key: read_json(root / filename)
        for key, filename in AUDIT_FILES.items()
    }


def draw_header(ax, letter: str, title: str, subtitle: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        ax.text(0, .83, letter, fontsize=10, fontweight="bold",
                ha="left", va="top"),
        ax.text(.085, .83, title, fontsize=9, fontweight="bold",
                ha="left", va="top"),
    ]
    wrapped = "\n".join(textwrap.wrap(
        subtitle, width=50, break_long_words=False, break_on_hyphens=False
    ))
    items.append(ax.text(
        .085, .38, wrapped, fontsize=7.0, color=MUTED,
        ha="left", va="top", linespacing=1.18,
    ))
    for item in items:
        item.set_gid("export-audit")


def panel_cell(fig, spec, letter: str, title: str, subtitle: str):
    inner = spec.subgridspec(2, 1, height_ratios=[.25, .75], hspace=0)
    header = fig.add_subplot(inner[0])
    draw_header(header, letter, title, subtitle)
    return fig.add_subplot(inner[1])


def clean_axis(ax, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=.6)
    if grid:
        ax.grid(color=GRID, linewidth=.55, zorder=0)
    ax.set_axisbelow(True)


def unavailable(ax, message: str) -> None:
    ax.text(.5, .5, message, transform=ax.transAxes,
            ha="center", va="center", color=MUTED)
    ax.set_axis_off()


def valid(value) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def nested(obj: dict, *keys, default=np.nan):
    value = obj
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def volume_metrics(result: dict, variant: str) -> dict:
    return nested(result, "event_volume", f"{variant}_ensemble", default={})


def type_metrics(result: dict) -> dict:
    return nested(result, "event_type", "ensemble", default={})


def marker_for(key: str) -> str:
    return "o" if key.startswith("sociofm") else "D"


def point_size(key: str) -> float:
    return 38 + 1.05 * PARAMETERS_M[key]


def available_models(representations: dict[str, dict]) -> list[str]:
    return [key for key in ORDER if representations.get(key)]


def panel_a(ax, representations: dict[str, dict]) -> None:
    models = [
        key for key in available_models(representations)
        if valid(nested(volume_metrics(representations[key], "hybrid"), "rmse"))
        and valid(nested(type_metrics(representations[key]), "macro_f1"))
    ]
    if not models:
        unavailable(ax, "Representation benchmarks unavailable")
        return
    offsets = {
        "sociofm_25m_50k": (5, -10),
        "sociofm_25m_100k": (5, -10),
        "sociofm_50m_100k": (-5, 8),
        "sociofm_100m_100k": (5, -14),
        "sociofm_100m_200k": (5, 8),
        "distilgpt2_frozen": (-5, 8),
        "gpt2_frozen": (-5, 18),
    }
    for key in models:
        x = float(volume_metrics(representations[key], "hybrid")["rmse"])
        y = float(type_metrics(representations[key])["macro_f1"])
        ax.scatter(
            x, y, s=point_size(key), marker=marker_for(key),
            color=COLORS[key], alpha=.90, edgecolor="white",
            linewidth=.8, zorder=3,
        )
        dx, dy = offsets[key]
        ax.annotate(
            SHORT[key], (x, y), xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=5.8, color=COLORS[key], clip_on=True,
        )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK,
               markeredgecolor="white", markersize=5.5, label="SocioFM"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=GREY,
               markeredgecolor="white", markersize=5.2, label="Open LM"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.text(.02, .98, "← lower RMSE  |  higher Macro-F1 ↑",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.8, color=MUTED)
    ax.set_xlabel("Hybrid event-volume RMSE (lower is better)")
    ax.set_ylabel("Event-type Macro-F1 (higher is better)")
    clean_axis(ax)


def panel_b(ax, representations: dict[str, dict]) -> None:
    models = [
        key for key in available_models(representations)
        if valid(nested(volume_metrics(representations[key], "representation_only"), "rmse"))
        and valid(nested(volume_metrics(representations[key], "hybrid"), "rmse"))
    ]
    if not models:
        unavailable(ax, "Volume transfer benchmarks unavailable")
        return
    y = np.arange(len(models))[::-1]
    for ypos, key in zip(y, models):
        start = float(volume_metrics(representations[key], "representation_only")["rmse"])
        end = float(volume_metrics(representations[key], "hybrid")["rmse"])
        arrow = FancyArrowPatch(
            (start, ypos), (end, ypos),
            arrowstyle="-|>", mutation_scale=11,
            linewidth=1.7, color=COLORS[key],
            connectionstyle="arc3,rad=.045", zorder=2,
        )
        ax.add_patch(arrow)
        ax.scatter(
            start, ypos, s=34, facecolor="white", edgecolor=COLORS[key],
            linewidth=1.1, zorder=3,
        )
        ax.scatter(
            end, ypos, s=40, color=COLORS[key], edgecolor="white",
            linewidth=.7, zorder=3,
        )
        midpoint = (start + end) / 2
        ax.text(
            midpoint, ypos, f"{end-start:+.1f}",
            ha="center", va="center", fontsize=5.7, color=COLORS[key],
            bbox={
                "boxstyle": "round,pad=.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": .92,
            },
            zorder=5,
        )
    ax.set_yticks(y, [LABELS[key] for key in models])
    ax.set_xlabel("Event-volume RMSE")
    clean_axis(ax)


def panel_c(ax, representations: dict[str, dict]) -> None:
    models = [
        key for key in available_models(representations)
        if valid(nested(type_metrics(representations[key]), "brier"))
        and valid(nested(type_metrics(representations[key]), "ece_15"))
    ]
    if not models:
        unavailable(ax, "Calibration benchmarks unavailable")
        return
    offsets = {
        "sociofm_25m_50k": (5, -12),
        "sociofm_25m_100k": (5, -12),
        "sociofm_50m_100k": (5, 8),
        "sociofm_100m_100k": (5, -12),
        "sociofm_100m_200k": (-5, 8),
        "distilgpt2_frozen": (-5, -12),
        "gpt2_frozen": (5, -12),
    }
    for key in models:
        metrics = type_metrics(representations[key])
        x = float(metrics["brier"])
        y = float(metrics["ece_15"])
        accuracy = float(metrics.get("accuracy", 0))
        size = 45 + 150 * max(accuracy, 0)
        ax.scatter(
            x, y, s=size, marker=marker_for(key), color=COLORS[key],
            alpha=.90, edgecolor="white", linewidth=.8, zorder=3,
        )
        dx, dy = offsets[key]
        ax.annotate(
            SHORT[key], (x, y), xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=5.8, color=COLORS[key], clip_on=True,
        )
    ax.text(
        .02, .98, "← lower Brier  |  lower ECE ↓",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=5.8, color=MUTED,
    )
    ax.set_xlabel("Multiclass Brier score (lower is better)")
    ax.set_ylabel("ECE, 15 bins (lower is better)")
    clean_axis(ax)


def seed_rows(
    key: str, result: dict, final_stats: dict,
) -> list[dict]:
    rows = nested(final_stats, "representation_seed_metrics", key, default=[])
    if isinstance(rows, list) and rows:
        return rows
    fallback = nested(result, "event_volume", "hybrid", default=[])
    return fallback if isinstance(fallback, list) else []


def panel_d(ax, representations: dict[str, dict], final_stats: dict) -> None:
    models = []
    values = {}
    for key in available_models(representations):
        sample = [
            float(row["rmse"]) for row in seed_rows(
                key, representations[key], final_stats
            ) if valid(row.get("rmse"))
        ]
        if sample:
            models.append(key)
            values[key] = np.asarray(sample)
    if not models:
        unavailable(ax, "Seed-level metrics unavailable")
        return
    y = np.arange(len(models))[::-1]
    rng = np.random.default_rng(7)
    for ypos, key in zip(y, models):
        sample = values[key]
        mean = float(np.mean(sample))
        sd = float(np.std(sample, ddof=1)) if len(sample) > 1 else 0
        ax.plot(
            [mean - sd, mean + sd], [ypos, ypos], color=COLORS[key],
            linewidth=3, alpha=.38, solid_capstyle="round", zorder=1,
        )
        jitter = rng.uniform(-.07, .07, len(sample))
        ax.scatter(
            sample, np.full(len(sample), ypos) + jitter,
            s=22, color=COLORS[key], alpha=.68,
            edgecolor="white", linewidth=.45, zorder=2,
        )
        ax.scatter(
            mean, ypos, s=48, facecolor="white", edgecolor=COLORS[key],
            linewidth=1.2, zorder=3,
        )
    ax.set_yticks(y, [LABELS[key] for key in models])
    ax.set_xlabel("Hybrid event-volume RMSE across seeds")
    clean_axis(ax)


def bootstrap_comparator(key: str) -> str:
    prefix = "sociofm_25m_50k_vs_"
    return key[len(prefix):] if key.startswith(prefix) else key


def panel_e(ax, final_stats: dict) -> None:
    raw = nested(final_stats, "paired_representation_bootstrap", default={})
    rows = []
    if isinstance(raw, dict):
        for name, stats in raw.items():
            comparator = bootstrap_comparator(name)
            if comparator not in LABELS:
                continue
            if (
                valid(stats.get("mean_difference_a_minus_b"))
                and isinstance(stats.get("ci95"), list)
                and len(stats["ci95"]) == 2
            ):
                rows.append((comparator, stats))
    rows.sort(key=lambda item: ORDER.index(item[0]) if item[0] in ORDER else 99)
    if not rows:
        unavailable(ax, "Bootstrap comparisons unavailable")
        return
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK, linestyle=(0, (3, 2)), linewidth=.9, zorder=1)
    for ypos, (key, stats) in zip(y, rows):
        mean = float(stats["mean_difference_a_minus_b"])
        low, high = (float(x) for x in stats["ci95"])
        ax.plot([low, high], [ypos, ypos], color=COLORS[key],
                linewidth=2.1, solid_capstyle="round", zorder=2)
        ax.scatter(
            mean, ypos, s=42, color=COLORS[key], edgecolor="white",
            linewidth=.7, zorder=3,
        )
        p = float(stats.get("p_two_sided", np.nan))
        p_text = f"p={p:.3f}" if np.isfinite(p) else ""
        ax.annotate(
            p_text, (high, ypos), xytext=(4, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=5.7, color=MUTED,
        )
    ax.set_yticks(y, [LABELS[key] for key, _ in rows])
    ax.set_xlabel("RMSE difference: 25M · 50k − comparator")
    clean_axis(ax)


def panel_f(ax, audits: dict[str, dict]) -> None:
    available = [
        key for key in SOCIO_ORDER
        if isinstance(audits.get(key, {}).get("ablations"), dict)
    ]
    if not available:
        unavailable(ax, "Ablation audits unavailable")
        return
    matrix = np.full((len(available), len(FIELDS)), np.nan)
    for i, key in enumerate(available):
        values = audits[key].get("ablations", {})
        for j, field in enumerate(FIELDS):
            value = nested(values, field, "delta_loss")
            if valid(value):
                matrix[i, j] = float(value)
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(np.nanpercentile(finite, 95)), .002) if len(finite) else .01
    image = ax.imshow(
        matrix, cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto", interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(available)), [LABELS[key] for key in available])
    ax.set_xticks(
        np.arange(len(FIELDS)),
        [FIELD_LABELS[field] for field in FIELDS],
        rotation=55, ha="right", rotation_mode="anchor",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value) and abs(value) >= .70 * limit:
                ax.text(
                    j, i, f"{value:+.3f}", ha="center", va="center",
                    fontsize=4.7,
                    color="white" if abs(value) >= .84 * limit else INK,
                )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=.07)
    cb = plt.colorbar(image, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cb.outline.set_linewidth(.45)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def audit_text_bounds(fig, tolerance_px: float = 3) -> None:
    """Fail export if a panel header lies outside the fixed canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    offenders = []
    for item in fig.findobj(match=mpl.text.Text):
        if (
            not item.get_visible()
            or not item.get_text().strip()
            or item.get_gid() != "export-audit"
        ):
            continue
        box = item.get_window_extent(renderer=renderer)
        if (
            box.x0 < canvas.x0 - tolerance_px
            or box.y0 < canvas.y0 - tolerance_px
            or box.x1 > canvas.x1 + tolerance_px
            or box.y1 > canvas.y1 + tolerance_px
        ):
            offenders.append(item.get_text().replace("\n", " / "))
    if offenders:
        raise RuntimeError("Header text outside export canvas: " + "; ".join(offenders))


def render(results_root: Path, out_dir: Path):
    representations = load_representations(results_root)
    audits = load_audits(results_root)
    final_stats = read_json(results_root / "final_prefigure_statistics.json")

    fig = plt.figure(figsize=(12.0, 7.20))
    grid = fig.add_gridspec(
        2, 3,
        left=.065, right=.962, bottom=.088, top=.985,
        wspace=.20, hspace=.15,
    )
    ax_a = panel_cell(
        fig, grid[0, 0], "a", "Multi-task transfer landscape",
        "Frozen representations on forward event-volume and event-type tests",
    )
    ax_b = panel_cell(
        fig, grid[0, 1], "b", "Contribution of structured context",
        "Open markers: representation only; filled markers: hybrid probe",
    )
    ax_c = panel_cell(
        fig, grid[0, 2], "c", "Calibration–discrimination landscape",
        "Marker area represents event-type accuracy; lower-left is preferred",
    )
    ax_d = panel_cell(
        fig, grid[1, 0], "d", "Random-seed stability",
        "Three probe seeds; open markers show means and thick lines show ±1 SD",
    )
    ax_e = panel_cell(
        fig, grid[1, 1], "e", "Country-cluster bootstrap comparisons",
        "Paired 95% intervals from 2,000 country resamples; negative favors reference",
    )
    ax_f = panel_cell(
        fig, grid[1, 2], "f", "Information reliance under field masking",
        "Change in matched-audit loss after masking each structured event field",
    )

    panel_a(ax_a, representations)
    panel_b(ax_b, representations)
    panel_c(ax_c, representations)
    panel_d(ax_d, representations, final_stats)
    panel_e(ax_e, final_stats)
    panel_f(ax_f, audits)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "main_figure_4_transfer_calibration_v2_600dpi.png"
    pdf = out_dir / "main_figure_4_transfer_calibration_v2_editable.pdf"
    audit_text_bounds(fig)
    fig.savefig(png, dpi=600, bbox_inches=None)
    fig.savefig(pdf, bbox_inches=None)
    plt.close(fig)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--out", default="figures/figure4"
    )
    args = parser.parse_args()
    configure_style()
    png, pdf = render(Path(args.results_root), Path(args.out))
    print(json.dumps({
        "png": str(png),
        "pdf": str(pdf),
        "dpi": 600,
        "editable_vector_pdf": True,
        "footer": False,
        "internal_figure_label": False,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
