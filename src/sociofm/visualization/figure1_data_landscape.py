#!/usr/bin/env python3
"""Premium SocioFM Figure 1 v3: symmetric societal-event data landscapes.

Reads monthly country_day_sequences.jsonl shards and produces a four-panel,
distribution-first scientific figure:

  a) 3D country × month × event-intensity landscape
  b) yearly ridgeline distributions of country-day event volume
  c) bivariate density of media attention and event tone
  d) event-root × year composition heatmap

Exports a 600-DPI PNG and an editable vector PDF. No footer and no internal
"Figure 1" label are added.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------- Shared publication style ----------
INK = "#202A35"
MUTED = "#697785"
GRID = "#DCE3E8"
BLUE = "#174A7E"
TEAL = "#148F88"
PURPLE = "#7656A5"
CORAL = "#D65A4A"
GOLD = "#C79A2B"
PALE = "#F4F7F9"

LANDSCAPE = LinearSegmentedColormap.from_list(
    "sociofm_landscape",
    ["#F4F7F9", "#BFD6E5", "#4F95B8", "#174A7E", "#512B71"],
)
HEAT = LinearSegmentedColormap.from_list(
    "sociofm_heat",
    ["#F7F8FA", "#C6E0DC", "#54A9A2", "#176A70", "#173F5F"],
)
RIDGE_COLORS = ["#174A7E", "#148F88", "#7656A5", "#D65A4A"]


def configure_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "savefig.pad_inches": 0.04,
    })


def draw_header(ax, letter, title, subtitle):
    """Draw a fixed-height header that can never overlap the data region."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, .82, letter, fontsize=10, fontweight="bold",
            color=INK, ha="left", va="top")
    ax.text(.085, .82, title, fontsize=9, fontweight="bold",
            color=INK, ha="left", va="top")
    ax.text(.085, .31, subtitle, fontsize=7.2, color=MUTED,
            ha="left", va="top")


def panel_cell(fig, outer_spec, letter, title, subtitle, projection=None):
    """Return a body axis beneath a dedicated, symmetric panel header."""
    inner = outer_spec.subgridspec(
        2, 1, height_ratios=[.20, .80], hspace=0.00
    )
    header = fig.add_subplot(inner[0])
    draw_header(header, letter, title, subtitle)
    body = fig.add_subplot(inner[1], projection=projection)
    return body


def clean_axis(ax, grid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=.6)
    if grid:
        ax.grid(axis="both", color=GRID, linewidth=.55, zorder=0)
    ax.set_axisbelow(True)


def parse_date(meta):
    raw = str(meta.get("date", "")).replace("-", "")
    if len(raw) < 6 or not raw[:6].isdigit():
        return None, None
    return int(raw[:4]), raw[:6]


def safe_float(value, default=np.nan):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def iter_country_days(data_root: Path, max_rows: int | None):
    paths = sorted(data_root.glob("20??-??/country_day_sequences.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No monthly country_day_sequences.jsonl files under {data_root}"
        )
    seen = 0
    for path in paths:
        print(f"READ {path.parent.name}", flush=True)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    meta = row.get("meta", row)
                except json.JSONDecodeError:
                    continue
                yield meta
                seen += 1
                if max_rows and seen >= max_rows:
                    return


def collect(data_root: Path, max_rows: int | None, sample_size: int, seed: int):
    rng = np.random.default_rng(seed)
    country_month = defaultdict(float)
    country_total = Counter()
    year_log_events = defaultdict(list)
    root_year = defaultdict(Counter)

    # Reservoir samples for the dense bivariate panel.
    attention_sample = np.empty(sample_size, dtype=np.float32)
    tone_sample = np.empty(sample_size, dtype=np.float32)
    sample_n = 0
    eligible_n = 0
    rows = 0

    for meta in iter_country_days(data_root, max_rows):
        year, month = parse_date(meta)
        country = str(meta.get("country", "")).upper().strip()
        events = safe_float(meta.get("events"), 0.0)
        mentions = safe_float(meta.get("mentions"), np.nan)
        tone = safe_float(meta.get("avg_tone"), np.nan)
        if year is None or not country or events <= 0:
            continue

        rows += 1
        country_month[(country, month)] += events
        country_total[country] += events
        year_log_events[year].append(math.log10(events + 1))

        roots = meta.get("top_event_roots", [])
        if isinstance(roots, list):
            for item in roots:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    root = str(item[0]).zfill(2)
                    count = safe_float(item[1], 0)
                    if count > 0:
                        root_year[year][root] += count

        if np.isfinite(mentions) and np.isfinite(tone) and mentions >= 0:
            # Attention is coverage per coded event, log-scaled for long tails.
            attention = math.log10(mentions / max(events, 1.0) + 1.0)
            eligible_n += 1
            if sample_n < sample_size:
                attention_sample[sample_n] = attention
                tone_sample[sample_n] = tone
                sample_n += 1
            else:
                j = int(rng.integers(0, eligible_n))
                if j < sample_size:
                    attention_sample[j] = attention
                    tone_sample[j] = tone

    return {
        "rows": rows,
        "country_month": country_month,
        "country_total": country_total,
        "year_log_events": year_log_events,
        "root_year": root_year,
        "attention": attention_sample[:sample_n],
        "tone": tone_sample[:sample_n],
    }


def smooth2d(matrix, passes=2):
    """Small dependency-free smoothing kernel for a calmer 3D surface."""
    out = matrix.astype(float, copy=True)
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            padded[:-2, :-2] + 2*padded[:-2, 1:-1] + padded[:-2, 2:] +
            2*padded[1:-1, :-2] + 4*padded[1:-1, 1:-1] + 2*padded[1:-1, 2:] +
            padded[2:, :-2] + 2*padded[2:, 1:-1] + padded[2:, 2:]
        ) / 16.0
    return out


def density_curve(values, grid, bandwidth=.12):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > 60_000:
        values = values[np.linspace(0, len(values)-1, 60_000).astype(int)]
    if len(values) < 2:
        return np.zeros_like(grid)
    # Vectorized Gaussian KDE in chunks to keep memory bounded.
    density = np.zeros_like(grid, dtype=float)
    for start in range(0, len(values), 5000):
        chunk = values[start:start+5000]
        u = (grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-.5*u*u).sum(axis=1)
    density /= len(values) * bandwidth * math.sqrt(2*math.pi)
    return density


def draw_landscape(ax, d, top_countries):
    months = sorted({m for _, m in d["country_month"]})
    matrix = np.zeros((len(top_countries), len(months)), dtype=float)
    month_index = {m: i for i, m in enumerate(months)}
    for ci, country in enumerate(top_countries):
        for month in months:
            matrix[ci, month_index[month]] = d["country_month"].get((country, month), 0)

    z = smooth2d(np.log10(matrix + 1), passes=1)
    x, y = np.meshgrid(np.arange(len(months)), np.arange(len(top_countries)))
    surface = ax.plot_surface(
        x, y, z, cmap=LANDSCAPE, rcount=len(top_countries),
        ccount=len(months), linewidth=.12, edgecolor=(1, 1, 1, .28),
        antialiased=True, alpha=.98,
    )
    ax.contour(x, y, z, zdir="z", offset=0, levels=10,
               cmap=LANDSCAPE, linewidths=.45, alpha=.7)
    ax.view_init(elev=27, azim=-59)
    ax.set_zlim(0, max(1, float(np.nanmax(z))*1.05))
    ticks = np.linspace(0, len(months)-1, min(5, len(months))).astype(int)
    ax.set_xticks(ticks, [months[i] for i in ticks], rotation=20, ha="right")
    y_ticks = np.arange(0, len(top_countries), max(1, len(top_countries)//7))
    ax.set_yticks(y_ticks, [top_countries[i] for i in y_ticks])
    ax.set_xlabel("Month", labelpad=7)
    ax.set_ylabel("Country rank", labelpad=5)
    ax.set_zlabel("Log event volume", labelpad=4)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((.96, .97, .98, .55))
    ax.xaxis.pane.set_edgecolor(GRID)
    ax.yaxis.pane.set_edgecolor(GRID)
    ax.zaxis.pane.set_edgecolor(GRID)
    ax.grid(False)
    # The z-axis already carries intensity; a second colorbar would duplicate
    # that information and break the four-panel symmetry.


def draw_ridgelines(ax, d):
    years = sorted(d["year_log_events"])
    all_values = np.concatenate([np.asarray(d["year_log_events"][y]) for y in years])
    upper = min(np.nanpercentile(all_values, 99.7), 5.2)
    grid = np.linspace(0, max(1.5, upper), 320)
    curves = {}
    for year in years:
        curves[year] = density_curve(d["year_log_events"][year], grid)

    # True 3D density ribbons: x=event volume, y=year, z=density.
    vertices = []
    colors = []
    for i, year in enumerate(years):
        curve = curves[year]
        curve = curve / max(curve.max(), 1e-9)
        vertices.append(
            [(grid[0], 0.0), *list(zip(grid, curve)), (grid[-1], 0.0)]
        )
        colors.append(RIDGE_COLORS[i % len(RIDGE_COLORS)])

    ribbons = PolyCollection(
        vertices, facecolors=colors, edgecolors=colors,
        linewidths=1.05, alpha=.76,
    )
    ax.add_collection3d(ribbons, zs=np.arange(len(years)), zdir="y")

    for i, year in enumerate(years):
        median = float(np.median(d["year_log_events"][year]))
        curve = curves[year] / max(curves[year].max(), 1e-9)
        height = float(np.interp(median, grid, curve))
        ax.scatter(
            [median], [i], [height], s=25, facecolor="white",
            edgecolor=RIDGE_COLORS[i % len(RIDGE_COLORS)],
            linewidth=.9, depthshade=False,
        )

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-.25, len(years)-.1)
    ax.set_zlim(0, 1.08)
    ax.set_yticks(np.arange(len(years)), years)
    ax.set_zticks([0, .5, 1], ["0", ".5", "1"])
    ax.set_xlabel("Log event volume", labelpad=5)
    ax.set_ylabel("Year", labelpad=5)
    ax.set_zlabel("Relative density", labelpad=4)
    ax.view_init(elev=25, azim=-62)
    ax.set_box_aspect((1.45, 1.0, .72))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(GRID)
    ax.grid(False)


def draw_attention_tone(ax, d):
    attention = d["attention"]
    tone = d["tone"]
    keep = (
        np.isfinite(attention) & np.isfinite(tone) &
        (tone >= np.nanpercentile(tone, .5)) &
        (tone <= np.nanpercentile(tone, 99.5))
    )
    attention, tone = attention[keep], tone[keep]
    hb = ax.hexbin(
        attention, tone, gridsize=(64, 48), mincnt=1,
        cmap=LANDSCAPE, norm=LogNorm(), linewidths=0,
    )
    # Central tendency is shown as thin density-compatible bin medians.
    bins = np.linspace(np.nanmin(attention), np.nanmax(attention), 22)
    centers, medians = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (attention >= lo) & (attention < hi)
        if idx.sum() >= 30:
            centers.append((lo+hi)/2)
            medians.append(np.median(tone[idx]))
    ax.scatter(centers, medians, s=13, facecolor="white", edgecolor=CORAL,
               linewidth=.65, zorder=4, label="Binned median tone")
    ax.axhline(0, color="white", linewidth=.8, linestyle="--", alpha=.8)
    ax.set_xlabel("Media attention, log₁₀(mentions per event + 1)")
    ax.set_ylabel("Average tone")
    ax.legend(loc="lower left", fontsize=6.3, frameon=True,
              facecolor="white", framealpha=.88)
    clean_axis(ax, grid=False)
    cax = inset_axes(
        ax, width="3.0%", height="55%", loc="center right",
        bbox_to_anchor=(-.025, 0, 1, 1), bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = ax.figure.colorbar(hb, cax=cax)
    cb.set_label("Country-day density", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def draw_root_heatmap(ax, d):
    years = sorted(d["root_year"])
    roots = sorted({
        root for counts in d["root_year"].values() for root in counts
    }, key=lambda r: int(r) if r.isdigit() else 999)
    matrix = np.array([
        [d["root_year"][year].get(root, 0) for root in roots]
        for year in years
    ], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, row_totals, out=np.zeros_like(matrix),
                       where=row_totals > 0) * 100
    im = ax.imshow(shares, aspect="auto", cmap=HEAT, interpolation="nearest")
    labels = [root if i % 2 == 0 else "" for i, root in enumerate(roots)]
    ax.set_xticks(np.arange(len(roots)), labels, rotation=0)
    ax.set_yticks(np.arange(len(years)), years)
    ax.set_xlabel("CAMEO event root")
    ax.set_ylabel("Year")
    ax.set_xlim(-.5, len(roots) + 1.2)
    for yi in range(len(years)):
        for xi in range(len(roots)):
            value = shares[yi, xi]
            if value >= 3.5:
                color = "white" if value > np.nanmax(shares)*.48 else INK
                ax.text(xi, yi, f"{value:.0f}", ha="center", va="center",
                        fontsize=5.4, color=color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = inset_axes(
        ax, width="3.0%", height="55%", loc="center right",
        bbox_to_anchor=(-.02, 0, 1, 1), bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("Within-year share (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def render(d, out_dir: Path, top_n: int):
    top_countries = [
        country for country, _ in d["country_total"].most_common(top_n)
    ]
    if not top_countries:
        raise RuntimeError("No valid country-day records were collected")

    # 183 mm wide, balanced for two-column journal reproduction.
    fig = plt.figure(figsize=(7.205, 7.90))
    gs = fig.add_gridspec(
        2, 2,
        left=.055, right=.975, bottom=.055, top=.985,
        wspace=.20, hspace=.16,
    )
    # Every cell has the same dimensions and an independent header band.
    # Titles therefore never occupy or resize the plotting area.
    ax_a = panel_cell(
        fig, gs[0, 0], "a", "Global event-intensity landscape",
        f"Top {len(top_countries)} countries; monthly retained-event totals",
        projection="3d",
    )
    ax_b = panel_cell(
        fig, gs[0, 1], "b", "Country-day volume distributions",
        "3D density ribbons; open markers are annual medians",
        projection="3d",
    )
    ax_c = panel_cell(
        fig, gs[1, 0], "c", "Media attention and event tone",
        f"Hexagonal log-density; sample n={len(d['attention']):,}",
    )
    ax_d = panel_cell(
        fig, gs[1, 1], "d", "Event-type composition across years",
        "Within-year distribution across represented CAMEO roots",
    )

    draw_landscape(ax_a, d, top_countries)
    draw_ridgelines(ax_b, d)
    draw_attention_tone(ax_c, d)
    draw_root_heatmap(ax_d, d)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "main_figure_1_data_landscape_v3_600dpi.png"
    pdf = out_dir / "main_figure_1_data_landscape_v3_editable.pdf"
    # Force one layout pass before freezing the exact export geometry.
    fig.canvas.draw()
    fig.savefig(png, dpi=600, bbox_inches=None)
    fig.savefig(pdf, bbox_inches=None)
    plt.close(fig)
    return png, pdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default="data/monthly_2022_2025",
        help="Directory containing YYYY-MM/country_day_sequences.jsonl shards",
    )
    p.add_argument("--out", default="figures")
    p.add_argument("--top-countries", type=int, default=28)
    p.add_argument("--sample-size", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Optional smoke-test row limit; omit for the full figure",
    )
    args = p.parse_args()
    configure_style()
    d = collect(
        Path(args.data_root), args.max_rows, args.sample_size, args.seed
    )
    print(
        f"COLLECTED rows={d['rows']:,} countries={len(d['country_total']):,} "
        f"attention_sample={len(d['attention']):,}",
        flush=True,
    )
    png, pdf = render(d, Path(args.out), args.top_countries)
    print(json.dumps({
        "png": str(png),
        "pdf": str(pdf),
        "dpi": 600,
        "footer": False,
        "internal_figure_label": False,
        "country_day_rows": d["rows"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
