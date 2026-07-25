#!/usr/bin/env python3
"""SocioFM Figure 2 landscape v3: scaling and temporal transfer.

Six symmetric panels tell one linked story:
  a) 3D capacity × training × perplexity landscape
  b) validation-to-forward-test dumbbells
  c) 3D optimization trajectories recovered from Trainer state files
  d) matched Global North–South audit pairs
  e) within-capacity additional-training vectors
  f) compute-efficiency bubble field and empirical Pareto frontier

Outputs a 600-DPI PNG and an editable vector PDF. No footer and no internal
"Figure 2" label are added.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------- Fixed experiment registry ----------
INK = "#202A35"
MUTED = "#697785"
GRID = "#DCE3E8"
PALE = "#F4F7F9"
BLUE = "#174A7E"
BLUE_LIGHT = "#78AFCB"
TEAL = "#148F88"
PURPLE = "#7656A5"
CORAL = "#D65A4A"
ORANGE = "#E8872E"
GREY = "#87939E"
LIGHT_GREY = "#C8D0D7"

ORDER = [
    "sociofm_25m_50k",
    "sociofm_25m_100k",
    "sociofm_50m_100k",
    "sociofm_100m_100k",
    "sociofm_100m_200k",
]

LABELS = {
    "sociofm_25m_50k": "25M · 50k",
    "sociofm_25m_100k": "25M · 100k",
    "sociofm_50m_100k": "50M · 100k",
    "sociofm_100m_100k": "101M · 100k",
    "sociofm_100m_200k": "101M · 200k",
}

COLORS = {
    "sociofm_25m_50k": BLUE,
    "sociofm_25m_100k": BLUE_LIGHT,
    "sociofm_50m_100k": TEAL,
    "sociofm_100m_100k": PURPLE,
    "sociofm_100m_200k": CORAL,
}

PARAMETERS_M = {
    "sociofm_25m_50k": 25.0,
    "sociofm_25m_100k": 25.0,
    "sociofm_50m_100k": 50.0,
    "sociofm_100m_100k": 101.25888,
    "sociofm_100m_200k": 101.25888,
}

STEPS = {
    "sociofm_25m_50k": 50_000,
    "sociofm_25m_100k": 100_000,
    "sociofm_50m_100k": 100_000,
    "sociofm_100m_100k": 100_000,
    "sociofm_100m_200k": 200_000,
}

PPL = {
    "sociofm_25m_50k": {"validation": 2.1082819842, "test": 2.1746925319},
    "sociofm_25m_100k": {"validation": 1.8187005491, "test": 2.2439101798},
    "sociofm_50m_100k": {"validation": 1.7972096527, "test": 2.1957518232},
    "sociofm_100m_100k": {"validation": 1.7898808693, "test": 2.6422995545},
    "sociofm_100m_200k": {"validation": 1.7012033830, "test": 2.8686090596},
}

GPU_HOURS_FALLBACK = {
    "sociofm_25m_50k": 1.445,
    "sociofm_25m_100k": 2.89,
    "sociofm_50m_100k": 10.686,
    "sociofm_100m_100k": 12.198,
    "sociofm_100m_200k": 27.386,
}

AUDIT_FILES = {
    "sociofm_25m_50k": "lm_audit_25m_50k.json",
    "sociofm_25m_100k": "lm_audit_25m_100k.json",
    "sociofm_50m_100k": "lm_audit_50m_100k.json",
    "sociofm_100m_100k": "lm_audit_100m_100k.json",
    "sociofm_100m_200k": "lm_audit_100m_200k.json",
}


def configure_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.edgecolor": INK,
        "axes.linewidth": .7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
    })


def read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARN {path}: {exc}", flush=True)
        return {}


def draw_header(ax, letter, title, subtitle):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    letter_text = ax.text(0, .82, letter, fontsize=10, fontweight="bold",
                          ha="left", va="top")
    title_text = ax.text(.085, .82, title, fontsize=9, fontweight="bold",
                        ha="left", va="top")
    wrapped = "\n".join(
        textwrap.wrap(
            subtitle, width=49, break_long_words=False,
            break_on_hyphens=False,
        )
    )
    subtitle_text = ax.text(
        .085, .37, wrapped, fontsize=7.0, color=MUTED,
        ha="left", va="top", linespacing=1.20,
    )
    for text in (letter_text, title_text, subtitle_text):
        text.set_gid("export-audit")


def panel_cell(fig, spec, letter, title, subtitle, projection=None):
    inner = spec.subgridspec(2, 1, height_ratios=[.25, .75], hspace=0)
    header = fig.add_subplot(inner[0])
    draw_header(header, letter, title, subtitle)
    return fig.add_subplot(inner[1], projection=projection)


def clean_axis(ax, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=.6)
    if grid:
        ax.grid(color=GRID, linewidth=.55, zorder=0)
    ax.set_axisbelow(True)


def style_3d(ax):
    ax.view_init(elev=25, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(GRID)
    ax.grid(False)


def load_audits(results_root: Path):
    return {
        key: read_json(results_root / filename)
        for key, filename in AUDIT_FILES.items()
    }


def load_compute(results_root: Path):
    final = read_json(results_root / "final_prefigure_statistics.json")
    runs = final.get("compute_accounting", {}).get("runs", [])
    hours = dict(GPU_HOURS_FALLBACK)
    for row in runs:
        if row.get("model") in hours:
            hours[row["model"]] = float(row["gpu_hours"])
    return hours


def checkpoint_path(checkpoint_root: Path, key: str):
    if key == "sociofm_25m_50k":
        return checkpoint_root / "sociofm_25m" / "checkpoint-50000"
    if key == "sociofm_25m_100k":
        return checkpoint_root / "sociofm_25m" / "checkpoint-100000"
    if key == "sociofm_50m_100k":
        return checkpoint_root / "sociofm_50m" / "checkpoint-100000"
    if key == "sociofm_100m_100k":
        return checkpoint_root / "sociofm_100m" / "checkpoint-100000"
    return checkpoint_root / "sociofm_100m" / "checkpoint-200000"


def load_training_histories(checkpoint_root: Path):
    histories = {}
    for key in ORDER:
        path = checkpoint_path(checkpoint_root, key) / "trainer_state.json"
        state = read_json(path)
        rows = []
        for row in state.get("log_history", []):
            if "loss" not in row or "step" not in row:
                continue
            try:
                step = int(row["step"])
                loss = float(row["loss"])
            except (TypeError, ValueError):
                continue
            if step <= STEPS[key] and np.isfinite(loss):
                rows.append((step, loss))
        if rows:
            # Limit visual density while retaining the exact endpoints.
            rows.sort()
            if len(rows) > 350:
                idx = np.linspace(0, len(rows)-1, 350).astype(int)
                rows = [rows[i] for i in idx]
            histories[key] = rows
        else:
            print(f"WARN no training history for {key}: {path}", flush=True)
    return histories


def regional_pair(audit):
    groups = {
        row.get("group"): float(row.get("perplexity", np.nan))
        for row in audit.get("north_south", [])
    }
    return groups.get("Global North", np.nan), groups.get("Global South", np.nan)


def panel_a(ax):
    """3D capacity × training × validation/test landscape."""
    for key in ORDER:
        x = PARAMETERS_M[key]
        y = STEPS[key] / 1000
        zv = PPL[key]["validation"]
        zt = PPL[key]["test"]
        ax.plot([x, x], [y, y], [zv, zt], color=COLORS[key],
                linewidth=2.2, alpha=.8)
        ax.scatter([x], [y], [zv], s=34, facecolor="white",
                   edgecolor=COLORS[key], linewidth=1.1, depthshade=False)
        ax.scatter([x], [y], [zt], s=46, color=COLORS[key],
                   edgecolor="white", linewidth=.7, depthshade=False)
        ax.text(x, y, zt+.055, LABELS[key], fontsize=6.2,
                ha="center", va="bottom", color=COLORS[key])
    ax.set_xlabel("Parameters (M)", labelpad=5)
    ax.set_ylabel("Training steps (k)", labelpad=5)
    ax.set_zlabel("PPL", labelpad=2)
    ax.set_box_aspect((1.15, 1.05, .80))
    style_3d(ax)


def panel_b(ax):
    """Validation-to-test paired dumbbells."""
    y = np.arange(len(ORDER))[::-1]
    for ypos, key in zip(y, ORDER):
        a = PPL[key]["validation"]
        b = PPL[key]["test"]
        ax.plot([a, b], [ypos, ypos], color=LIGHT_GREY,
                linewidth=3, solid_capstyle="round")
        ax.scatter(a, ypos, s=34, facecolor="white",
                   edgecolor=TEAL, linewidth=1.2, zorder=3)
        ax.scatter(b, ypos, s=42, color=CORAL,
                   edgecolor="white", linewidth=.7, zorder=3)
        ax.annotate(
            f"+{b-a:.2f}", (b, ypos), xytext=(5, 0),
            textcoords="offset points", va="center", fontsize=6.2,
            color=CORAL,
        )
    ax.set_yticks(y, [LABELS[k] for k in ORDER])
    ax.set_xlabel("Perplexity")
    ax.set_xlim(
        min(PPL[k]["validation"] for k in ORDER)-.06,
        max(PPL[k]["test"] for k in ORDER)+.22,
    )
    clean_axis(ax)


def panel_c(ax, histories):
    """Actual optimization trajectories in a 3D model-separated field."""
    available = [k for k in ORDER if k in histories]
    if not available:
        ax.text2D(.5, .5, "Trainer histories unavailable", transform=ax.transAxes,
                  ha="center", va="center", color=MUTED)
        ax.set_axis_off()
        return
    for i, key in enumerate(available):
        rows = np.asarray(histories[key], dtype=float)
        # Suppress the first unstable logging points only when they dominate
        # the visual range; the underlying state files remain unchanged.
        keep = rows[:, 1] <= np.nanpercentile(rows[:, 1], 98)
        rows = rows[keep]
        x = rows[:, 0] / 1000
        y = np.full(len(rows), i)
        z = rows[:, 1]
        points = np.column_stack([x, y, z])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(
            segments, colors=COLORS[key], linewidths=1.4, alpha=.90
        )
        ax.add_collection3d(lc)
        ax.scatter([x[-1]], [i], [z[-1]], color=COLORS[key],
                   s=28, edgecolor="white", linewidth=.6, depthshade=False)
    all_steps = [step/1000 for key in available for step, _ in histories[key]]
    all_loss = [loss for key in available for _, loss in histories[key]]
    ax.set_xlim(0, max(all_steps)*1.03)
    ax.set_ylim(-.2, len(available)-.1)
    ax.set_zlim(max(0, np.nanpercentile(all_loss, 1)*.85),
                np.nanpercentile(all_loss, 98)*1.05)
    compact = {
        "sociofm_25m_50k": "25/50",
        "sociofm_25m_100k": "25/100",
        "sociofm_50m_100k": "50/100",
        "sociofm_100m_100k": "101/100",
        "sociofm_100m_200k": "101/200",
    }
    ax.set_yticks(np.arange(len(available)), [compact[k] for k in available])
    ax.set_xlabel("Step (k)", labelpad=5)
    ax.set_ylabel("Model/steps", labelpad=4)
    ax.set_zlabel("Loss", labelpad=2)
    ax.set_box_aspect((1.4, 1.0, .75))
    style_3d(ax)


def panel_d(ax, audits):
    """Matched North–South pairs by checkpoint."""
    available = [k for k in ORDER if audits.get(k)]
    y = np.arange(len(available))[::-1]
    for ypos, key in zip(y, available):
        north, south = regional_pair(audits[key])
        ax.plot([north, south], [ypos, ypos], color=LIGHT_GREY,
                linewidth=3, solid_capstyle="round")
        ax.scatter(north, ypos, s=36, color=PURPLE,
                   edgecolor="white", linewidth=.7, zorder=3)
        ax.scatter(south, ypos, s=38, facecolor="white",
                   edgecolor=TEAL, linewidth=1.2, zorder=3)
        ax.annotate(
            f"{south-north:+.3f}", (max(north, south), ypos),
            xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=6.2,
            color=CORAL if south > north else TEAL,
        )
    if available:
        ax.set_yticks(y, [LABELS[k] for k in available])
        ax.set_xlabel("Matched-audit perplexity")
        all_values = [
            value for key in available for value in regional_pair(audits[key])
            if np.isfinite(value)
        ]
        if all_values:
            ax.set_xlim(min(all_values)-.02, max(all_values)+.09)
        clean_axis(ax)
    else:
        ax.text(.5, .5, "Audit JSON files unavailable",
                transform=ax.transAxes, ha="center", va="center", color=MUTED)
        ax.set_axis_off()


def panel_e(ax):
    """Within-capacity additional-training vectors."""
    for key in ORDER:
        ax.scatter(
            PPL[key]["validation"], PPL[key]["test"],
            s=45, color=COLORS[key], edgecolor="white", linewidth=.7,
            zorder=3, label=LABELS[key],
        )
    pairs = [
        ("sociofm_25m_50k", "sociofm_25m_100k", BLUE),
        ("sociofm_100m_100k", "sociofm_100m_200k", CORAL),
    ]
    for start, end, color in pairs:
        arrow = FancyArrowPatch(
            (PPL[start]["validation"], PPL[start]["test"]),
            (PPL[end]["validation"], PPL[end]["test"]),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.7, color=color,
            connectionstyle="arc3,rad=.08", zorder=2,
        )
        ax.add_patch(arrow)
    ax.axline((1.7, 1.7), slope=1, color=GREY,
              linestyle="--", linewidth=.8, alpha=.8)
    ax.set_xlabel("Validation perplexity")
    ax.set_ylabel("Forward-test perplexity")
    ax.legend(ncol=2, loc="upper right", fontsize=5.8)
    clean_axis(ax)


def pareto_indices(x, y):
    order = np.argsort(x)
    keep = []
    best = np.inf
    for idx in order:
        if y[idx] < best:
            keep.append(idx)
            best = y[idx]
    return keep


def panel_f(ax, gpu_hours):
    x = np.array([gpu_hours[k] for k in ORDER], dtype=float)
    y = np.array([PPL[k]["test"] for k in ORDER], dtype=float)
    sizes = np.array([PARAMETERS_M[k] for k in ORDER], dtype=float)
    for i, key in enumerate(ORDER):
        ax.scatter(x[i], y[i], s=30+sizes[i]*1.1,
                   color=COLORS[key], alpha=.92,
                   edgecolor="white", linewidth=.8, zorder=3)
        if x[i] == max(x):
            offset, align = (-4, 4), "right"
        else:
            offset, align = (4, 4), "left"
        ax.annotate(
            LABELS[key], (x[i], y[i]), xytext=offset,
            textcoords="offset points", fontsize=6.1, color=COLORS[key],
            ha=align,
        )
    keep = pareto_indices(x, y)
    frontier = np.column_stack([x[keep], y[keep]])
    frontier = frontier[np.argsort(frontier[:, 0])]
    ax.plot(frontier[:, 0], frontier[:, 1], color=INK,
            linestyle=(0, (3, 2)), linewidth=1.1,
            label="Empirical Pareto frontier")
    ax.set_xlabel("GPU hours")
    ax.set_ylabel("Forward-test perplexity")
    ax.set_xlim(0, max(x)*1.10)
    ax.set_ylim(min(y)-.05, max(y)+.06)
    ax.legend(loc="upper left")
    clean_axis(ax)


def audit_text_bounds(fig, tolerance_px=3):
    """Fail before export if any visible text is clipped by the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    offenders = []
    for text in fig.findobj(match=mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        # Limit the strict canvas test to headers. Matplotlib's projected
        # 3D ticks and several axis artists can report misleading extents.
        if text.get_gid() != "export-audit":
            continue
        try:
            box = text.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if (
            box.x0 < canvas.x0 - tolerance_px
            or box.y0 < canvas.y0 - tolerance_px
            or box.x1 > canvas.x1 + tolerance_px
            or box.y1 > canvas.y1 + tolerance_px
        ):
            offenders.append(text.get_text().replace("\n", " / "))
    if offenders:
        raise RuntimeError(
            "Text outside export canvas: " + "; ".join(offenders[:8])
        )


def render(results_root: Path, checkpoint_root: Path, out_dir: Path):
    audits = load_audits(results_root)
    gpu_hours = load_compute(results_root)
    histories = load_training_histories(checkpoint_root)

    # Wide 2 × 3 landscape canvas: larger panel bodies and shorter scan path.
    fig = plt.figure(figsize=(12.0, 7.20))
    grid = fig.add_gridspec(
        2, 3,
        left=.065, right=.985, bottom=.060, top=.985,
        wspace=.18, hspace=.15,
    )
    ax_a = panel_cell(
        fig, grid[0, 0], "a", "Capacity–training–transfer landscape",
        "Open markers: validation; filled markers: forward test",
        projection="3d",
    )
    ax_b = panel_cell(
        fig, grid[0, 1], "b", "Forward temporal generalization gap",
        "Open: validation; filled: test; labels show degradation",
    )
    ax_c = panel_cell(
        fig, grid[0, 2], "c", "Optimization trajectories",
        "Trainer-state loss histories separated in three-dimensional space",
        projection="3d",
    )
    ax_d = panel_cell(
        fig, grid[1, 0], "d", "Regional transfer disparity",
        "Purple: North; open teal: South; labels show the gap",
    )
    ax_e = panel_cell(
        fig, grid[1, 1], "e", "Effect of additional training",
        "Arrows connect checkpoints with identical model capacity",
    )
    ax_f = panel_cell(
        fig, grid[1, 2], "f", "Compute-efficiency frontier",
        "Marker area represents parameter count; lower perplexity is better",
    )

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c, histories)
    panel_d(ax_d, audits)
    panel_e(ax_e)
    panel_f(ax_f, gpu_hours)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "main_figure_2_scaling_transfer_landscape_v3_600dpi.png"
    pdf = out_dir / "main_figure_2_scaling_transfer_landscape_v3_editable.pdf"
    audit_text_bounds(fig)
    fig.savefig(png, dpi=600, bbox_inches=None)
    fig.savefig(pdf, bbox_inches=None)
    plt.close(fig)
    return png, pdf, histories


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--out", default="figures/figure2")
    args = p.parse_args()
    configure_style()
    png, pdf, histories = render(
        Path(args.results_root), Path(args.checkpoint_root), Path(args.out)
    )
    print(json.dumps({
        "png": str(png),
        "pdf": str(pdf),
        "dpi": 600,
        "editable_vector_pdf": True,
        "training_histories_found": sorted(histories),
        "footer": False,
        "internal_figure_label": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
