#!/usr/bin/env python3
"""SocioFM Figure 3: geographic generalization and media fragility.

Six coordinated panels:
  a) country-level matched-audit perplexity on a world map;
  b) country coverage versus matched-audit perplexity;
  c) country-level North/South perplexity distributions by checkpoint;
  d) source frequency versus source-level perplexity;
  e) country-held-out RMSLE distributions across forecasting methods;
  f) event-root vulnerability relative to each checkpoint's base perplexity.

Exports a 600-DPI PNG and an editable vector PDF. No footer and no internal
"Figure 3" label are added.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ---------- Shared visual system ----------
INK = "#202A35"
MUTED = "#697785"
GRID = "#DCE3E8"
PALE = "#F4F7F9"
MISSING = "#E8EDF1"
BLUE = "#174A7E"
BLUE_LIGHT = "#78AFCB"
TEAL = "#148F88"
TEAL_LIGHT = "#A9D8D3"
PURPLE = "#7656A5"
PURPLE_LIGHT = "#C9B9E0"
CORAL = "#D65A4A"
GOLD = "#D49A32"
GREY = "#87939E"

WORLD_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)

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

AUDIT_FILES = {
    "sociofm_25m_50k": "lm_audit_25m_50k.json",
    "sociofm_25m_100k": "lm_audit_25m_100k.json",
    "sociofm_50m_100k": "lm_audit_50m_100k.json",
    "sociofm_100m_100k": "lm_audit_100m_100k.json",
    "sociofm_100m_200k": "lm_audit_100m_200k.json",
}

NORTH = {
    "AS", "AU", "BE", "CA", "DA", "EI", "FI", "FR", "GM", "GR", "IC",
    "IT", "JA", "LU", "NE", "NO", "NZ", "PO", "SP", "SW", "SZ", "UK", "US",
}

MAP_CMAP = LinearSegmentedColormap.from_list(
    "sociofm_map", ["#EAF3F6", "#9BC8D1", TEAL, BLUE, "#3D295F"]
)
HEX_CMAP = LinearSegmentedColormap.from_list(
    "sociofm_hex", ["#EDF4F6", BLUE_LIGHT, TEAL, BLUE, "#3D295F"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "sociofm_diverging", [BLUE, "#D4E6ED", "#F7F7F4", "#F1C7BE", CORAL]
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
        "legend.fontsize": 6.3,
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


def load_audits(results_root: Path) -> dict[str, dict]:
    return {
        key: read_json(results_root / filename)
        for key, filename in AUDIT_FILES.items()
    }


def draw_header(ax, letter: str, title: str, subtitle: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    texts = [
        ax.text(0, .83, letter, fontsize=10, fontweight="bold",
                ha="left", va="top"),
        ax.text(.085, .83, title, fontsize=9, fontweight="bold",
                ha="left", va="top"),
    ]
    wrapped = "\n".join(textwrap.wrap(
        subtitle, width=50, break_long_words=False, break_on_hyphens=False
    ))
    texts.append(ax.text(
        .085, .38, wrapped, fontsize=7.0, color=MUTED,
        ha="left", va="top", linespacing=1.18,
    ))
    for item in texts:
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


def unavailable(ax, text: str) -> None:
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center",
            color=MUTED, fontsize=8)
    ax.set_axis_off()


def audit_rows(audit: dict, key: str) -> list[dict]:
    rows = audit.get(key, [])
    return rows if isinstance(rows, list) else []


def valid_number(value) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def ensure_world_geojson(path: Path) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Natural Earth boundaries -> {path}", flush=True)
        request = urllib.request.Request(
            WORLD_URL, headers={"User-Agent": "SocioFM-Figure/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            path.write_bytes(payload)
        except Exception as exc:
            raise RuntimeError(
                "Natural Earth world boundaries could not be downloaded. "
                f"Download {WORLD_URL} manually and pass --world-geojson. {exc}"
            ) from exc
    return read_json(path)


def geometry_rings(geometry: dict):
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = coordinates
    else:
        return
    for polygon in polygons:
        if polygon and len(polygon[0]) >= 3:
            yield np.asarray(polygon[0], dtype=float)


def feature_code(properties: dict) -> str:
    for key in ("FIPS_10", "FIPS_10_", "POSTAL", "ISO_A2", "ISO_A2_EH"):
        value = str(properties.get(key, "")).upper().strip()
        if value and value not in {"-99", "NONE", "NAN"}:
            return value
    return ""


def panel_a(ax, audit: dict, world: dict) -> None:
    """Country-level perplexity choropleth using GDELT/FIPS country codes."""
    values = {
        str(row.get("group", "")).upper(): float(row["perplexity"])
        for row in audit_rows(audit, "countries")
        if valid_number(row.get("perplexity"))
    }
    if not values or not world.get("features"):
        unavailable(ax, "Country audit or world boundaries unavailable")
        return

    observed = np.asarray(list(values.values()), dtype=float)
    lo, hi = np.nanpercentile(observed, [5, 95])
    if math.isclose(lo, hi):
        hi = lo + .01
    norm = mpl.colors.Normalize(lo, hi)
    patches, colors = [], []
    missing_patches = []
    for feature in world["features"]:
        code = feature_code(feature.get("properties", {}))
        for ring in geometry_rings(feature.get("geometry", {})):
            patch = Polygon(ring, closed=True)
            if code in values:
                patches.append(patch)
                colors.append(values[code])
            else:
                missing_patches.append(patch)

    if missing_patches:
        missing_collection = PatchCollection(
            missing_patches, facecolor=MISSING, edgecolor="white",
            linewidth=.22, zorder=1,
        )
        ax.add_collection(missing_collection)
    collection = PatchCollection(
        patches, cmap=MAP_CMAP, norm=norm, edgecolor="white",
        linewidth=.25, zorder=2,
    )
    collection.set_array(np.asarray(colors))
    ax.add_collection(collection)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 88)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    cax = inset_axes(
        ax, width="38%", height="5%", loc="lower left",
        bbox_to_anchor=(.02, .02, .96, .96),
        bbox_transform=ax.transAxes, borderpad=0,
    )
    cb = plt.colorbar(collection, cax=cax, orientation="horizontal")
    cb.set_label("Country perplexity", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(.5)
    mapped = len({feature_code(f.get("properties", {}))
                  for f in world["features"]} & set(values))
    ax.text(
        .99, .02, f"Mapped countries: {mapped}/{len(values)}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.0, color=MUTED,
    )


def binned_median(x: np.ndarray, y: np.ndarray, bins: int = 8):
    if len(x) < bins:
        return np.array([]), np.array([])
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    centers, medians = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x <= right)
        if mask.sum() >= 3:
            centers.append(float(np.median(x[mask])))
            medians.append(float(np.median(y[mask])))
    return np.asarray(centers), np.asarray(medians)


def panel_b(ax, audit: dict) -> None:
    rows = [
        row for row in audit_rows(audit, "countries")
        if valid_number(row.get("n")) and valid_number(row.get("perplexity"))
    ]
    if not rows:
        unavailable(ax, "Country audit unavailable")
        return
    x = np.log10(np.asarray([float(row["n"]) for row in rows]) + 1)
    y = np.asarray([float(row["perplexity"]) for row in rows])
    codes = np.asarray([str(row.get("group", "")).upper() for row in rows])
    sizes = 14 + 38 * (x - x.min()) / max(np.ptp(x), .01)
    north = np.isin(codes, list(NORTH))
    ax.scatter(
        x[north], y[north], s=sizes[north], color=PURPLE, alpha=.72,
        edgecolor="white", linewidth=.45, label="Global North", zorder=2,
    )
    ax.scatter(
        x[~north], y[~north], s=sizes[~north], facecolor="white",
        edgecolor=TEAL, linewidth=.9, alpha=.88, label="Global South", zorder=2,
    )
    bx, by = binned_median(x, y)
    if len(bx):
        ax.plot(bx, by, color=INK, linewidth=1.4, zorder=3)
        ax.scatter(bx, by, s=13, color=INK, edgecolor="white",
                   linewidth=.35, zorder=4)
    priority = np.argsort(y)[-3:]
    midpoint = float(np.median(x))
    for rank, idx in enumerate(priority):
        right_side = x[idx] > midpoint
        ax.annotate(
            codes[idx], (x[idx], y[idx]),
            xytext=(-4 if right_side else 4, 3 + 3 * rank),
            textcoords="offset points", fontsize=5.8, color=INK,
            ha="right" if right_side else "left", clip_on=True,
        )
    ax.set_xlabel("Country sample size, log₁₀(n + 1)")
    ax.set_ylabel("Country perplexity")
    ax.legend(loc="lower left")
    clean_axis(ax)


def kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.zeros_like(grid)
    std = max(float(np.std(values, ddof=1)), .015)
    bandwidth = max(1.06 * std * len(values) ** (-1 / 5), .018)
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-.5 * z * z).sum(axis=1)
    density /= len(values) * bandwidth * math.sqrt(2 * math.pi)
    return density


def panel_c(ax, audits: dict[str, dict]) -> None:
    available = [key for key in ORDER if audit_rows(audits.get(key, {}), "countries")]
    if not available:
        unavailable(ax, "Country audits unavailable")
        return
    all_values = [
        float(row["perplexity"])
        for key in available
        for row in audit_rows(audits[key], "countries")
        if valid_number(row.get("perplexity"))
    ]
    lo, hi = np.nanpercentile(all_values, [1, 99])
    grid = np.linspace(max(.8, lo - .08), hi + .10, 250)
    y_positions = np.arange(len(available))[::-1]

    for ypos, key in zip(y_positions, available):
        rows = audit_rows(audits[key], "countries")
        north = np.asarray([
            float(row["perplexity"]) for row in rows
            if str(row.get("group", "")).upper() in NORTH
            and valid_number(row.get("perplexity"))
        ])
        south = np.asarray([
            float(row["perplexity"]) for row in rows
            if str(row.get("group", "")).upper() not in NORTH
            and valid_number(row.get("perplexity"))
        ])
        dn, ds = kde(north, grid), kde(south, grid)
        scale = .34 / max(dn.max(), ds.max(), 1e-9)
        ax.fill_between(grid, ypos, ypos + dn * scale, color=PURPLE,
                        alpha=.72, linewidth=0)
        ax.plot(grid, ypos + dn * scale, color=PURPLE, linewidth=.8)
        ax.fill_between(grid, ypos, ypos - ds * scale, color=TEAL_LIGHT,
                        alpha=.86, linewidth=0)
        ax.plot(grid, ypos - ds * scale, color=TEAL, linewidth=.8)
        if len(north):
            ax.scatter(np.median(north), ypos + .02, s=12, color=PURPLE,
                       edgecolor="white", linewidth=.4, zorder=4)
        if len(south):
            ax.scatter(np.median(south), ypos - .02, s=14, facecolor="white",
                       edgecolor=TEAL, linewidth=.8, zorder=4)
        ax.axhline(ypos, color=GRID, linewidth=.45, zorder=0)
    ax.set_yticks(y_positions, [LABELS[key] for key in available])
    ax.set_xlabel("Country-level perplexity")
    ax.set_xlim(grid.min(), grid.max())
    ax.text(.02, .97, "North", transform=ax.transAxes, color=PURPLE,
            fontsize=6.3, va="top")
    ax.text(.02, .89, "South", transform=ax.transAxes, color=TEAL,
            fontsize=6.3, va="top")
    clean_axis(ax, grid=False)


def panel_d(ax, audit: dict) -> None:
    rows = [
        row for row in audit_rows(audit, "sources")
        if valid_number(row.get("n")) and valid_number(row.get("perplexity"))
        and str(row.get("group", "")).lower() != "unknown"
    ]
    if not rows:
        unavailable(ax, "Source audit unavailable")
        return
    x = np.log10(np.asarray([float(row["n"]) for row in rows]) + 1)
    y = np.asarray([float(row["perplexity"]) for row in rows])
    hb = ax.hexbin(
        x, y, gridsize=(25, 17), mincnt=1, bins="log",
        cmap=HEX_CMAP, linewidths=.15, edgecolors="white",
    )
    bx, by = binned_median(x, y, bins=7)
    if len(bx):
        ax.plot(bx, by, color=CORAL, linewidth=1.3, zorder=4)
        ax.scatter(bx, by, s=13, facecolor="white", edgecolor=CORAL,
                   linewidth=.7, zorder=5)
    eligible = np.where(x >= np.quantile(x, .60))[0]
    extremes = []
    if len(eligible):
        extremes = [eligible[np.argmax(y[eligible])], eligible[np.argmin(y[eligible])]]
    for idx in extremes:
        label = str(rows[idx].get("group", ""))
        if len(label) > 22:
            label = label[:20] + "…"
        right_side = x[idx] > np.median(x)
        ax.annotate(
            label, (x[idx], y[idx]),
            xytext=(-4 if right_side else 4, 4),
            textcoords="offset points", fontsize=5.4, color=INK,
            ha="right" if right_side else "left", clip_on=True,
        )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="4.5%", pad=.08)
    cb = plt.colorbar(hb, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cb.outline.set_linewidth(.45)
    ax.set_xlabel("Source frequency, log₁₀(n + 1)")
    ax.set_ylabel("Source perplexity")
    clean_axis(ax)


def panel_e(ax, geographic: dict) -> None:
    folds = geographic.get("folds", [])
    names = ["persistence", "raw_poisson_tree", "log_change_tree"]
    labels = ["Persistence", "Raw-count", "Log-change"]
    colors = [GREY, PURPLE, TEAL]
    values = {
        name: [
            fold.get("models", {}).get(name, {}).get("micro", {}).get("rmsle")
            for fold in folds
        ]
        for name in names
    }
    if not folds or not all(any(valid_number(v) for v in values[name]) for name in names):
        unavailable(ax, "Geographic benchmark unavailable")
        return
    rng = np.random.default_rng(7)
    positions = np.arange(1, 4)
    for pos, name, color in zip(positions, names, colors):
        sample = np.asarray([v for v in values[name] if valid_number(v)], dtype=float)
        violin = ax.violinplot(
            sample, positions=[pos], widths=.62,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.24)
        jitter = rng.uniform(-.10, .10, len(sample))
        ax.scatter(
            np.full(len(sample), pos) + jitter, sample,
            s=20, color=color, alpha=.82, edgecolor="white",
            linewidth=.4, zorder=3,
        )
        ax.scatter(
            pos, np.median(sample), s=47, facecolor="white",
            edgecolor=color, linewidth=1.2, zorder=4,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Country-held-out RMSLE")
    ax.set_xlim(.5, 3.5)
    clean_axis(ax)


def panel_f(ax, audits: dict[str, dict]) -> None:
    available = [key for key in ORDER if audit_rows(audits.get(key, {}), "event_roots")]
    roots = sorted({
        str(row.get("group", "")).zfill(2)
        for key in available
        for row in audit_rows(audits[key], "event_roots")
        if str(row.get("group", ""))
    })
    if not available or not roots:
        unavailable(ax, "Event-root audits unavailable")
        return
    matrix = np.full((len(available), len(roots)), np.nan)
    for i, key in enumerate(available):
        base = float(audits[key].get("base", {}).get("perplexity", np.nan))
        lookup = {
            str(row.get("group", "")).zfill(2): float(row["perplexity"])
            for row in audit_rows(audits[key], "event_roots")
            if valid_number(row.get("perplexity"))
        }
        if not np.isfinite(base) or base <= 0:
            continue
        for j, root in enumerate(roots):
            if root in lookup:
                matrix[i, j] = 100 * (lookup[root] / base - 1)
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(np.nanpercentile(finite, 95)), 2.0) if len(finite) else 5.0
    image = ax.imshow(
        matrix, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto", interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(roots)), roots, rotation=0)
    ax.set_yticks(np.arange(len(available)), [LABELS[key] for key in available])
    ax.set_xlabel("CAMEO event root")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value) and abs(value) >= .72 * limit:
                ax.text(
                    j, i, f"{value:+.0f}",
                    ha="center", va="center", fontsize=5.0,
                    color="white" if abs(value) >= .86 * limit else INK,
                )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="4.5%", pad=.08)
    cb = plt.colorbar(image, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cb.outline.set_linewidth(.45)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def audit_text_bounds(fig, tolerance_px: float = 3) -> None:
    """Fail export if any panel header lies outside the fixed canvas."""
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


def render(
    results_root: Path,
    out_dir: Path,
    world_geojson: Path,
    map_model: str,
    detail_model: str,
):
    audits = load_audits(results_root)
    geographic = read_json(results_root / "geographic_v2.json")
    world = ensure_world_geojson(world_geojson)

    if map_model not in audits:
        raise ValueError(f"Unknown --map-model: {map_model}")
    if detail_model not in audits:
        raise ValueError(f"Unknown --detail-model: {detail_model}")

    fig = plt.figure(figsize=(12.0, 7.20))
    grid = fig.add_gridspec(
        2, 3,
        left=.065, right=.985, bottom=.060, top=.985,
        wspace=.18, hspace=.15,
    )
    ax_a = panel_cell(
        fig, grid[0, 0], "a", "Geography of model uncertainty",
        f"Country perplexity from the matched 2025 audit; {LABELS[map_model]}",
    )
    ax_b = panel_cell(
        fig, grid[0, 1], "b", "Coverage–fragility relationship",
        "Country sample size, model uncertainty and regional representation",
    )
    ax_c = panel_cell(
        fig, grid[0, 2], "c", "Regional uncertainty distributions",
        "Mirrored country densities: North above and South below each baseline",
    )
    ax_d = panel_cell(
        fig, grid[1, 0], "d", "Media-source fragility landscape",
        f"Hexagonal density of source frequency versus perplexity; {LABELS[detail_model]}",
    )
    ax_e = panel_cell(
        fig, grid[1, 1], "e", "Country-held-out robustness",
        "Five volume-stratified folds; open circles denote fold medians",
    )
    ax_f = panel_cell(
        fig, grid[1, 2], "f", "Event-type vulnerability matrix",
        "Relative perplexity against each checkpoint's overall matched-audit value",
    )

    panel_a(ax_a, audits[map_model], world)
    panel_b(ax_b, audits[detail_model])
    panel_c(ax_c, audits)
    panel_d(ax_d, audits[detail_model])
    panel_e(ax_e, geographic)
    panel_f(ax_f, audits)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "main_figure_3_geographic_media_v1_600dpi.png"
    pdf = out_dir / "main_figure_3_geographic_media_v1_editable.pdf"
    audit_text_bounds(fig)
    fig.savefig(png, dpi=600, bbox_inches=None)
    fig.savefig(pdf, bbox_inches=None)
    plt.close(fig)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--out", default="figures/figure3"
    )
    parser.add_argument(
        "--world-geojson",
        default="assets/ne_110m_admin_0_countries.geojson",
    )
    parser.add_argument("--map-model", choices=ORDER, default="sociofm_50m_100k")
    parser.add_argument("--detail-model", choices=ORDER, default="sociofm_50m_100k")
    args = parser.parse_args()
    configure_style()
    png, pdf = render(
        Path(args.results_root),
        Path(args.out),
        Path(args.world_geojson),
        args.map_model,
        args.detail_model,
    )
    print(json.dumps({
        "png": str(png),
        "pdf": str(pdf),
        "dpi": 600,
        "editable_vector_pdf": True,
        "map_model": args.map_model,
        "detail_model": args.detail_model,
        "footer": False,
        "internal_figure_label": False,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
