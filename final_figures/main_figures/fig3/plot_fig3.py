#!/usr/bin/env python3
"""Single entry point that regenerates every panel of Figure 3 (handwritten-digit classification).

Panels (caption a-j):
  a  network diagram (fig3_a.png / fig3_a_light.png) + per-class gate-rail V+/V- bars (fig3_a_gate_rail_fields.png)
  b  network state at init          -> fig3_b.png
  c  network state trained          -> fig3_c.png
  d  V_G distribution at init       -> fig3_d.png
  e  V_G distribution trained       -> fig3_e.png
  f  confusion matrix at init       -> fig3_f.png            (shared colorbar fig3_fg_colorbar.png)
  g  confusion matrix trained       -> fig3_g.png
  h  accuracy + hinge loss vs step  -> fig3_h.png            (per-sample variant fig3_h_sample.png)
  i  input-weighted sensitivity     -> fig3_i.png            (colorbar fig3_i_colorbar.png)
  j  misclassified digits           -> fig3_j.png            (all-misses variant fig3_j_all.png)

The panel-a *diagram* (fig3_a*.png) is a static asset assembled by the shared Fig. 2 network renderer
and is NOT regenerated here; this script regenerates the panel-a gate-rail bars plus panels b-j.

The sensitivity panel (i) measures 360 held-out digits on the real ngspice operating point using a
fragile subprocess `--worker` self-reinvocation pattern; that logic lives in
`plot_fig3_scikit_all_test_sensitivity.py` and is invoked here as an imported helper (it also still
writes Supplementary Fig. 4's signed map/colorbar into suppl_figures/suppl4/data/). The network-state
renders (b/c) are 3000-dpi and slow. Both are skipped by `--fast`.

Aggregated output .npz files are written to ./data/. Input data are read from paper_release/scikit_digits/.

Usage:
  python plot_fig3.py                 # regenerate every panel (SLOW: ngspice sensitivity + 3000-dpi nets)
  python plot_fig3.py --fast          # only the fast panels: a-bars, d, e, f, g, h, j
  python plot_fig3.py --panels f g h j
  python plot_fig3.py --panels i --sensitivity-workers 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors, ticker
from matplotlib.collections import LineCollection
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_PR = next(p for p in HERE.parents if (p / "device_model").is_dir())
DATA_DIR = HERE / "data"
TASK = _PR / "scikit_digits"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TASK))

# ------------------------------------------------------------------------------------------------
# Shared inputs (read-only; produced by the scikit_digits training pipeline)
# ------------------------------------------------------------------------------------------------
GATES_CACHE = DATA_DIR / "fig3_scikit_clean_final_step_gates.npz"       # derived init/final gate cache
UPDCURVE = TASK / "results" / "updcurve_clean.npz"                      # per-step train/test metrics
TOPOLOGY = TASK / "topology_1247.npz"                                   # edge (drain, source) topology

# ------------------------------------------------------------------------------------------------
# Aggregated outputs (written under ./data/)
# ------------------------------------------------------------------------------------------------
CONFUSION_CACHE = DATA_DIR / "fig3_scikit_confusion_matrices.npz"
GATE_RAIL_VALUES = DATA_DIR / "fig3_scikit_gate_rail_fields.npz"
NETWORK_META = DATA_DIR / "fig3_scikit_network_state_panels_meta.npz"
MISCLASSIFIED_SCORES = DATA_DIR / "fig3_scikit_misclassified_digits_scores.npz"
MISCLASSIFIED_ALL_SCORES = DATA_DIR / "fig3_scikit_misclassified_digits_all_scores.npz"

TEXT = "#34383D"
NEUTRAL = "#F3F4F4"


# ================================================================================================
# Shared data helpers
# ================================================================================================
def _test_split() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float64)
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    return Xte, yte.astype(int)


# ================================================================================================
# Panels d/e (V_G histograms) + b/c (network state) : from plot_fig3_scikit_network_state_panels.py
# ================================================================================================
NET_WIDTH_IN = 1.55
NET_HEIGHT_IN = 1.59
NET_XLIM = (-0.76, 1.68)
NET_YLIM = (-1.252, 1.252)
HIST_WIDTH_IN = 1.48
HIST_HEIGHT_IN = 0.53
GATE_VMIN = 0.4
GATE_VCENTER = 2.5
GATE_VMAX = 6.1
HIST_MIN = 0.4
HIST_MAX = 6.1
HIST_BINS = 29
HIST_Y_TICKS_STEP0 = [50, 100]
HIST_Y_TICKS_FINAL = [90, 180]
HIST_Y_TOP_STEP0 = 100.0
HIST_Y_TOP_FINAL = 180.0
HIST_LABEL_SIZE = 6.0
HIST_TICK_SIZE = 6.0
HIST_X_TICKS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
HIST_SPINE_LINEWIDTH = 0.55
NET_BORDER = "#B8BDC2"
INPUT_NODE_BORDER_COLOR = "#000000"
INPUT_NODE_BORDER_WIDTH = 0.30
OUTPUT_ARC_DEG = 70.0
OUTPUT_PAIR_SEPARATION = 0.20
NETWORK_DPI = 3000
HIST_DPI = 600


def _gate_cmap() -> colors.Colormap:
    return colors.LinearSegmentedColormap.from_list("gate_voltage", ["#4B86B4", NEUTRAL, "#ED9360"], N=256)


def _gate_norm() -> colors.TwoSlopeNorm:
    return colors.TwoSlopeNorm(vmin=GATE_VMIN, vcenter=GATE_VCENTER, vmax=GATE_VMAX)


def _gate_rgba(vg: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    clipped = np.clip(np.asarray(vg, dtype=float), GATE_VMIN, GATE_VMAX)
    rgba = _gate_cmap()(_gate_norm()(clipped))
    rgba[:, 3] = alpha
    return rgba


def _load_state(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    required = ["vg_init", "vg_final", "drain", "source"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"{path} missing required keys: {', '.join(missing)}")
    return {
        "vg_init": np.asarray(data["vg_init"], dtype=float).reshape(-1),
        "vg_final": np.asarray(data["vg_final"], dtype=float).reshape(-1),
        "drain": np.asarray(data["drain"], dtype=int).reshape(-1),
        "source": np.asarray(data["source"], dtype=int).reshape(-1),
        "final_step": int(data["final_step"]) if "final_step" in data.files else -1,
        "final_epoch": int(data["final_epoch"]) if "final_epoch" in data.files else -1,
        "final_sample": int(data["final_sample"]) if "final_sample" in data.files else -1,
        "final_test_acc": float(data["final_test_acc"]) if "final_test_acc" in data.files else float("nan"),
    }


def _input_positions() -> np.ndarray:
    xs = np.linspace(-0.70, 0.70, 8)
    ys = np.linspace(0.70, -0.70, 8)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=float)


def _output_positions() -> np.ndarray:
    theta = np.deg2rad(np.linspace(OUTPUT_ARC_DEG, -OUTPUT_ARC_DEG, 10))
    centers = np.column_stack([1.50 * np.cos(theta), 1.12 * np.sin(theta)])
    sep = OUTPUT_PAIR_SEPARATION
    out = np.zeros((20, 2), dtype=float)
    out[:10] = centers + np.array([sep / 2.0, 0.0])
    out[10:] = centers + np.array([-sep / 2.0, 0.0])
    return out


def _draw_network(drain, source, vg, out_prefix: Path, *, edge_width, edge_alpha, dpi) -> Path:
    if not (drain.size == source.size == vg.size):
        raise ValueError("drain, source, and vg must have the same length")
    input_pos = _input_positions()
    output_pos = _output_positions()
    rail = source.astype(int) - 64

    fig, ax = plt.subplots(figsize=(NET_WIDTH_IN, NET_HEIGHT_IN))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(*NET_XLIM)
    ax.set_ylim(*NET_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    segments, edge_vg = [], []
    for edge_idx, (d_node, rail_idx) in enumerate(zip(drain, rail)):
        d, r = int(d_node), int(rail_idx)
        if 0 <= d < 64 and 0 <= r < 20:
            segments.append([input_pos[d], output_pos[r]])
            edge_vg.append(float(vg[edge_idx]))
    if segments:
        ax.add_collection(LineCollection(
            segments, colors=_gate_rgba(np.asarray(edge_vg, dtype=float), alpha=edge_alpha),
            linewidths=edge_width, zorder=1, capstyle="round", joinstyle="round"))

    ax.scatter(input_pos[:, 0], input_pos[:, 1], marker="s", s=7.5, facecolors="#FFFFFF",
               edgecolors=INPUT_NODE_BORDER_COLOR, linewidths=INPUT_NODE_BORDER_WIDTH, zorder=4)
    for c in range(10):
        for rail_idx, label in [(c + 10, f"{c}-"), (c, f"{c}+")]:
            x0, y0 = output_pos[rail_idx]
            ax.scatter([x0], [y0], marker="o", s=31, facecolors="#FFFFFF",
                       edgecolors=TEXT, linewidths=INPUT_NODE_BORDER_WIDTH, zorder=5)
            ax.text(x0, y0, label, ha="center", va="center", fontsize=4.0, color=TEXT,
                    fontfamily="Open Sans", zorder=6)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def _draw_histogram(vg, out_prefix: Path, *, count_top, y_ticks, dpi) -> Path:
    bins = np.linspace(HIST_MIN, HIST_MAX, HIST_BINS)
    counts, edges = np.histogram(vg, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    bar_colors = _gate_rgba(centers, alpha=0.96)

    fig, ax = plt.subplots(figsize=(HIST_WIDTH_IN, HIST_HEIGHT_IN))
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.38, top=0.93)
    ax.bar(centers, counts, width=widths, color=bar_colors, edgecolor=TEXT, linewidth=0.25, align="center")
    ax.set_xlim(HIST_MIN, HIST_MAX)
    ax.set_ylim(0.0, count_top)
    ax.set_xticks(HIST_X_TICKS)
    ax.set_yticks(y_ticks)
    ax.set_xlabel("$V_G$ (V)", labelpad=0.2, fontsize=HIST_LABEL_SIZE)
    ax.set_ylabel("Count", labelpad=0.2, fontsize=HIST_LABEL_SIZE)
    ax.tick_params(axis="both", which="major", direction="out", width=0.35, length=1.25,
                   pad=0.6, colors=TEXT, labelsize=HIST_TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(HIST_SPINE_LINEWIDTH)
        spine.set_edgecolor(TEXT)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def render_network_states(*, edge_width=0.25, edge_alpha=0.50, network_dpi=NETWORK_DPI, hist_dpi=HIST_DPI):
    """Panels b, c (network diagrams) + d, e (V_G histograms)."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.0, "axes.labelsize": 6.0, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "ps.fonttype": 42,
    })
    state = _load_state(GATES_CACHE)
    drain = np.asarray(state["drain"], dtype=int)
    source = np.asarray(state["source"], dtype=int)
    vg_init = np.asarray(state["vg_init"], dtype=float)
    vg_final = np.asarray(state["vg_final"], dtype=float)

    outputs = [
        _draw_network(drain, source, vg_init, HERE / "fig3_b", edge_width=edge_width, edge_alpha=edge_alpha, dpi=network_dpi),
        _draw_network(drain, source, vg_final, HERE / "fig3_c", edge_width=edge_width, edge_alpha=edge_alpha, dpi=network_dpi),
        _draw_histogram(vg_init, HERE / "fig3_d", count_top=HIST_Y_TOP_STEP0, y_ticks=HIST_Y_TICKS_STEP0, dpi=hist_dpi),
        _draw_histogram(vg_final, HERE / "fig3_e", count_top=HIST_Y_TOP_FINAL, y_ticks=HIST_Y_TICKS_FINAL, dpi=hist_dpi),
    ]
    NETWORK_META.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        NETWORK_META, gates=str(GATES_CACHE), snapshot_policy="final training step",
        final_step=int(state["final_step"]), final_epoch=int(state["final_epoch"]),
        final_sample=int(state["final_sample"]), final_test_acc=float(state["final_test_acc"]),
        vg_init_min=float(np.min(vg_init)), vg_init_max=float(np.max(vg_init)),
        vg_final_min=float(np.min(vg_final)), vg_final_max=float(np.max(vg_final)),
        gate_norm=np.asarray([GATE_VMIN, GATE_VCENTER, GATE_VMAX], dtype=float),
        hist_range=np.asarray([HIST_MIN, HIST_MAX], dtype=float),
        hist_count_top=float(HIST_Y_TOP_FINAL), hist_count_top_step0=float(HIST_Y_TOP_STEP0),
    )
    for p in outputs + [NETWORK_META]:
        print(p)
    return outputs


def render_vg_histograms(*, hist_dpi=HIST_DPI):
    """Panels d, e only (fast; no 3000-dpi network render)."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.0, "axes.labelsize": 6.0, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "ps.fonttype": 42,
    })
    state = _load_state(GATES_CACHE)
    vg_init = np.asarray(state["vg_init"], dtype=float)
    vg_final = np.asarray(state["vg_final"], dtype=float)
    outputs = [
        _draw_histogram(vg_init, HERE / "fig3_d", count_top=HIST_Y_TOP_STEP0, y_ticks=HIST_Y_TICKS_STEP0, dpi=hist_dpi),
        _draw_histogram(vg_final, HERE / "fig3_e", count_top=HIST_Y_TOP_FINAL, y_ticks=HIST_Y_TICKS_FINAL, dpi=hist_dpi),
    ]
    for p in outputs:
        print(p)
    return outputs


# ================================================================================================
# Panel a bars (gate-rail V+/V- fields) : from plot_fig3_scikit_gate_rail_fields.py
# ================================================================================================
GR_WIDTH_IN = 2.96
GR_HEIGHT_IN = 0.70
GR_MAP_SIZE_IN = 0.27
GR_MAP_COLUMN_GAP_IN = 0.024
GR_MAP_ROW_GAP_IN = 0.025
GR_MAP_TOP_MARGIN_IN = 0.012
GR_CLASS_LABEL_Y_IN = 0.045
GR_CLASS_LABEL_SIZE = 6.0
GR_PRUNED = "#000000"


def _gate_rail_cmap() -> colors.Colormap:
    cmap = colors.LinearSegmentedColormap.from_list("gate_voltage", ["#4B86B4", NEUTRAL, "#ED9360"], N=256)
    cmap.set_bad(GR_PRUNED)
    return cmap


def _load_topology(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "drain" in data.files and "source" in data.files:
        return np.asarray(data["drain"], dtype=int).reshape(-1), np.asarray(data["source"], dtype=int).reshape(-1)
    if "edges_D" in data.files and "edges_S" in data.files:
        return np.asarray(data["edges_D"], dtype=int).reshape(-1), np.asarray(data["edges_S"], dtype=int).reshape(-1)
    raise KeyError(f"{path} does not look like a scikit topology NPZ")


def _build_gate_rail_maps(drain, source, vg) -> tuple[np.ndarray, np.ndarray]:
    if not (drain.size == source.size == vg.size):
        raise ValueError("drain, source, and vg must have the same length")
    plus = np.full((10, 64), np.nan, dtype=float)
    minus = np.full((10, 64), np.nan, dtype=float)
    rail = source.astype(int) - 64
    for edge_idx, (pixel, rail_idx) in enumerate(zip(drain.astype(int), rail)):
        if not 0 <= pixel < 64:
            continue
        if 0 <= rail_idx < 10:
            plus[rail_idx, pixel] = float(vg[edge_idx])
        elif 10 <= rail_idx < 20:
            minus[rail_idx - 10, pixel] = float(vg[edge_idx])
    return plus, minus


def render_gate_rail_fields(*, dpi=600):
    """Panel a bars: per-class V+ over V- gate-voltage maps."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 5, "xtick.labelsize": 5, "ytick.labelsize": 5, "ps.fonttype": 42,
    })
    drain, source = _load_topology(TOPOLOGY)
    data = np.load(GATES_CACHE, allow_pickle=True)
    vg = np.asarray(data["vg_final"], dtype=float).reshape(-1)
    plus, minus = _build_gate_rail_maps(drain, source, vg)

    cmap = _gate_rail_cmap()
    norm = _gate_norm()
    fig = plt.figure(figsize=(GR_WIDTH_IN, GR_HEIGHT_IN))
    total_map_width_in = 10 * GR_MAP_SIZE_IN + 9 * GR_MAP_COLUMN_GAP_IN
    map_left_in = 0.5 * (GR_WIDTH_IN - total_map_width_in)
    plus_bottom_in = GR_HEIGHT_IN - GR_MAP_TOP_MARGIN_IN - GR_MAP_SIZE_IN
    minus_bottom_in = plus_bottom_in - GR_MAP_ROW_GAP_IN - GR_MAP_SIZE_IN

    for digit in range(10):
        left_in = map_left_in + digit * (GR_MAP_SIZE_IN + GR_MAP_COLUMN_GAP_IN)
        for row_offset, values in [(0, plus[digit]), (1, minus[digit])]:
            bottom_in = plus_bottom_in if row_offset == 0 else minus_bottom_in
            ax = fig.add_axes([left_in / GR_WIDTH_IN, bottom_in / GR_HEIGHT_IN,
                               GR_MAP_SIZE_IN / GR_WIDTH_IN, GR_MAP_SIZE_IN / GR_HEIGHT_IN])
            ax.imshow(values.reshape(8, 8), cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_edgecolor(TEXT)
        fig.text((left_in + 0.5 * GR_MAP_SIZE_IN) / GR_WIDTH_IN, GR_CLASS_LABEL_Y_IN / GR_HEIGHT_IN,
                 str(digit), ha="center", va="center", fontsize=GR_CLASS_LABEL_SIZE,
                 fontfamily="Open Sans", color=TEXT)

    png = HERE / "fig3_a_gate_rail_fields.png"
    fig.savefig(png, dpi=dpi)
    plt.close(fig)

    GATE_RAIL_VALUES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GATE_RAIL_VALUES, plus=plus, minus=minus,
             missing_plus=np.isnan(plus), missing_minus=np.isnan(minus),
             gate_norm=np.asarray([GATE_VMIN, GATE_VCENTER, GATE_VMAX], dtype=float),
             topology=str(TOPOLOGY), gates=str(GATES_CACHE))
    print(png)
    print(GATE_RAIL_VALUES)
    return png


# ================================================================================================
# Panels f/g (confusion matrices + colorbar) : from plot_fig3_scikit_confusion_matrices.py
# ================================================================================================
CONFUSION_BOX_SIZE_IN = 1.1914
CONFUSION_LEFT_IN = 0.138
CONFUSION_BOTTOM_IN = 0.138
CONFUSION_FIGURE_SIZE_IN = 1.34
CB_WIDTH_IN = 0.30
CB_HEIGHT_IN = 1.30
CB_BODY_HEIGHT_IN = 1.20
CB_THICKNESS_IN = 0.041
CONFUSION_TICK_LABEL_SIZE = 6.0


def _compute_confusions(gates_path: Path, cache_path: Path) -> dict:
    import train_scikit as T
    from sklearn.metrics import confusion_matrix

    data = np.load(gates_path, allow_pickle=True)
    vg_init = np.asarray(data["vg_init"], dtype=float)
    vg_final = np.asarray(data["vg_final"], dtype=float)
    drain = np.asarray(data["drain"], dtype=int)
    source = np.asarray(data["source"], dtype=int)
    final_ep = int(data["final_epoch"]) if "final_epoch" in data.files else -1
    Xte, yte = _test_split()

    ng = _get_ng()

    def eval_preds(vg):
        ng.load_circuit(T.build_netlist(drain, source, vg, np.zeros(vg.size), np.zeros(vg.size)))
        out_nodes = [900 + j for j in range(20)]
        rng = np.random.default_rng(0)
        preds = []
        for x in Xte:
            cmds = [f"alter VIN{p} dc = {float(x[p]):.8f}" for p in range(64)]
            cmds += [f"alter VINB{p} dc = {float(x[p]):.8f}" for p in range(64)]
            T.exec_chunked(ng, cmds)
            vout = T.read_nodes(ng, out_nodes, rng, 0.0)
            preds.append(int(np.argmax(vout[:10] - vout[10:])))
        return np.asarray(preds, dtype=int)

    pred_init = eval_preds(vg_init)
    ng.exec_command("destroy all")
    pred_final = eval_preds(vg_final)

    cm_init = confusion_matrix(yte, pred_init, labels=np.arange(10))
    cm_final = confusion_matrix(yte, pred_final, labels=np.arange(10))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, cm_init=cm_init, cm_final=cm_final, pred_init=pred_init, pred_final=pred_final,
             y_true=yte, final_epoch=final_ep, snapshot_policy="final training step", gates=str(gates_path))
    return _load_confusion_cache(cache_path)


def _load_confusion_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "cm_init": np.asarray(data["cm_init"], dtype=float),
        "cm_final": np.asarray(data["cm_final"], dtype=float),
        "final_epoch": np.asarray(data["final_epoch"], dtype=int),
        "gates": np.asarray(data["gates"], dtype=str) if "gates" in data.files else np.asarray(""),
    }


def _load_or_compute_confusions(gates_path: Path, cache_path: Path, recompute: bool) -> dict:
    if recompute or not cache_path.exists():
        return _compute_confusions(gates_path, cache_path)
    with np.load(cache_path, allow_pickle=True) as data:
        if "final_epoch" not in data.files:
            return _compute_confusions(gates_path, cache_path)
    metrics = _load_confusion_cache(cache_path)
    if str(metrics.get("gates", "")) != str(gates_path):
        return _compute_confusions(gates_path, cache_path)
    return metrics


def _normalize_rows(cm: np.ndarray) -> np.ndarray:
    denom = np.sum(cm, axis=1, keepdims=True)
    return cm / np.where(denom > 0, denom, 1.0)


def _draw_confusion(cm, out_prefix: Path, vmax: float, dpi=600) -> Path:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 4.5, "xtick.labelsize": CONFUSION_TICK_LABEL_SIZE, "ytick.labelsize": CONFUSION_TICK_LABEL_SIZE,
    })
    fig = plt.figure(figsize=(CONFUSION_FIGURE_SIZE_IN, CONFUSION_FIGURE_SIZE_IN))
    ax = fig.add_axes([CONFUSION_LEFT_IN / CONFUSION_FIGURE_SIZE_IN, CONFUSION_BOTTOM_IN / CONFUSION_FIGURE_SIZE_IN,
                       CONFUSION_BOX_SIZE_IN / CONFUSION_FIGURE_SIZE_IN, CONFUSION_BOX_SIZE_IN / CONFUSION_FIGURE_SIZE_IN])
    ax.imshow(_normalize_rows(cm), cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest", origin="upper")
    ticks = np.arange(10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(width=0.35, length=1.2, pad=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_edgecolor(TEXT)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def _draw_confusion_colorbar(out_prefix: Path, vmax: float, dpi=600) -> Path:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 4.5, "ytick.labelsize": 5.0,
    })
    fig = plt.figure(figsize=(CB_WIDTH_IN, CB_HEIGHT_IN))
    cax = fig.add_axes([0.18, 0.5 * (1.0 - CB_BODY_HEIGHT_IN / CB_HEIGHT_IN),
                        CB_THICKNESS_IN / CB_WIDTH_IN, CB_BODY_HEIGHT_IN / CB_HEIGHT_IN])
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="magma")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_ticks([])
    cb.ax.tick_params(width=0.0, length=0.0, pad=0.0, colors=TEXT)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor(TEXT)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def render_confusions(*, recompute=False, dpi=600):
    """Panels f (init) and g (trained) + shared colorbar fig3_fg_colorbar.png."""
    metrics = _load_or_compute_confusions(GATES_CACHE, CONFUSION_CACHE, recompute)
    cm_init = metrics["cm_init"]
    cm_final = metrics["cm_final"]
    vmax = 1.0
    f_png = _draw_confusion(cm_init, HERE / "fig3_f", vmax, dpi)
    g_png = _draw_confusion(cm_final, HERE / "fig3_g", vmax, dpi)
    cbar_png = _draw_confusion_colorbar(HERE / "fig3_fg_colorbar", vmax, dpi)
    print(f"initial_acc={float(np.trace(cm_init) / cm_init.sum()):.6f}")
    print(f"final_acc={float(np.trace(cm_final) / cm_final.sum()):.6f}")
    for p in (f_png, g_png, cbar_png):
        print(p)
    return f_png, g_png, cbar_png


# ================================================================================================
# Panel h (accuracy + hinge loss vs step) : from plot_fig3_scikit_clean_acc_loss.py
# ================================================================================================
TRAIN_COLOR = "#274E87"
TEST_COLOR = "#D34F72"
AXIS_COLOR = "#34383D"
H_WIDTH_IN = 3.11
H_HEIGHT_IN = 2.55
H_DPI = 600
H_SUBPLOT_LEFT = 0.118
H_SUBPLOT_RIGHT = 0.997
H_SUBPLOT_BOTTOM = 0.125
H_SUBPLOT_TOP = 0.992
H_SUBPLOT_HSPACE = 0.075
CURVE_LINEWIDTH = 0.5
CURVE_ALPHA = 0.7
SNAPSHOT_STEP_DOT_SIZE = 0.5
X_SCALE = "log"
PLOT_START_STEP = 4.0
PLOT_END_EPOCH = 15
ACCURACY_YLIM = (0.0, 1.02)
ACCURACY_TICKS = [0.2, 0.4, 0.6, 0.8, 1.0]
LOG_X_MAJOR_TICKS = 5
H_FONT_SIZE = 6.0
H_LEGEND_FONT_SIZE = 5.2
H_SPINE_LINEWIDTH = 0.55


def _sci_tick(value: float, _pos: int) -> str:
    if abs(value) < 1e-15:
        return "0"
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _load_acc_loss_metrics(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    required = {"train_acc", "train_loss", "test_acc", "test_loss"}
    if not required.issubset(data.files):
        raise KeyError(f"{path} must contain {sorted(required)}")
    train_acc = np.asarray(data["train_acc"], dtype=float)
    train_loss = np.asarray(data["train_loss"], dtype=float)
    test_acc = np.asarray(data["test_acc"], dtype=float)
    test_loss = np.asarray(data["test_loss"], dtype=float)
    n = len(test_acc)
    return {
        "step": np.arange(1, n + 1, dtype=float),
        "train_acc": train_acc, "test_acc": test_acc, "train_loss": train_loss, "test_loss": test_loss,
        "epoch": np.asarray(data["epoch"], dtype=int) if "epoch" in data.files else np.full(n, -1, dtype=int),
        "final_idx": n - 1, "final_step": int(n),
        "final_epoch": int(data["epoch"][n - 1]) if "epoch" in data.files else -1,
    }


def _style_acc_loss_axis(ax) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", which="major", direction="out", width=0.45, length=2.0, pad=1.5, colors=AXIS_COLOR)
    ax.tick_params(axis="x", which="minor", direction="out", width=0.35, length=1.2, pad=1.5, colors=AXIS_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(H_SPINE_LINEWIDTH)
        spine.set_edgecolor(AXIS_COLOR)


def _draw_acc_loss(metrics: dict, out_prefix: Path, dpi=H_DPI) -> Path:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": H_FONT_SIZE, "axes.labelsize": H_FONT_SIZE, "xtick.labelsize": H_FONT_SIZE,
        "ytick.labelsize": H_FONT_SIZE, "legend.fontsize": H_LEGEND_FONT_SIZE, "ps.fonttype": 42,
    })
    step = np.asarray(metrics["step"], dtype=float)
    train_acc = np.asarray(metrics["train_acc"], dtype=float)
    test_acc = np.asarray(metrics["test_acc"], dtype=float)
    train_loss = np.clip(np.asarray(metrics["train_loss"], dtype=float), np.finfo(float).tiny, None)
    test_loss = np.clip(np.asarray(metrics["test_loss"], dtype=float), np.finfo(float).tiny, None)
    epoch = np.asarray(metrics["epoch"], dtype=int)
    plot_mask = (step >= float(PLOT_START_STEP)) & (epoch <= int(PLOT_END_EPOCH))
    if not bool(np.any(plot_mask)):
        raise ValueError("No training steps are in the requested plotting window")
    step_plot = step[plot_mask]
    snapshot_idx = int(metrics["final_idx"])
    snapshot_step = int(metrics["final_step"])
    snapshot_visible = bool(plot_mask[snapshot_idx])

    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(H_WIDTH_IN, H_HEIGHT_IN), sharex=True,
                                          gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": H_SUBPLOT_HSPACE})
    fig.subplots_adjust(left=H_SUBPLOT_LEFT, right=H_SUBPLOT_RIGHT, bottom=H_SUBPLOT_BOTTOM, top=H_SUBPLOT_TOP)

    ax_acc.plot(step_plot, train_acc[plot_mask], color=TRAIN_COLOR, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA, label="Train")
    ax_acc.plot(step_plot, test_acc[plot_mask], color=TEST_COLOR, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA, label="Test")
    if snapshot_visible:
        ax_acc.scatter([snapshot_step], [train_acc[snapshot_idx]], s=SNAPSHOT_STEP_DOT_SIZE, marker="o",
                       facecolors=TRAIN_COLOR, edgecolors="#000000", linewidths=0.0, zorder=5)
        ax_acc.scatter([snapshot_step], [test_acc[snapshot_idx]], s=SNAPSHOT_STEP_DOT_SIZE, marker="o",
                       facecolors=TEST_COLOR, edgecolors="#000000", linewidths=0.0, zorder=5)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(*ACCURACY_YLIM)
    ax_acc.set_yticks(ACCURACY_TICKS)
    ax_acc.legend(loc="upper left", frameon=False, handlelength=1.6, borderpad=0.25, labelspacing=0.25)

    ax_loss.plot(step_plot, train_loss[plot_mask], color=TRAIN_COLOR, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA, label="Train")
    ax_loss.plot(step_plot, test_loss[plot_mask], color=TEST_COLOR, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA, label="Test")
    if snapshot_visible:
        ax_loss.scatter([snapshot_step], [train_loss[snapshot_idx]], s=SNAPSHOT_STEP_DOT_SIZE, marker="o",
                        facecolors=TRAIN_COLOR, edgecolors="#000000", linewidths=0.0, zorder=5)
        ax_loss.scatter([snapshot_step], [test_loss[snapshot_idx]], s=SNAPSHOT_STEP_DOT_SIZE, marker="o",
                        facecolors=TEST_COLOR, edgecolors="#000000", linewidths=0.0, zorder=5)
    ax_loss.set_ylabel("Hinge Loss")
    ax_loss.set_xlabel("Training Step")
    ax_loss.set_yscale("linear")
    loss_max = float(np.nanmax([np.nanmax(train_loss[plot_mask]), np.nanmax(test_loss[plot_mask])]))
    ax_loss.set_ylim(0.0, loss_max * 1.08)
    ax_loss.yaxis.set_major_formatter(ticker.FuncFormatter(_sci_tick))

    for ax in (ax_acc, ax_loss):
        ax.set_xscale(X_SCALE)
        ax.set_xlim(float(step_plot[0]), float(step_plot[-1]))
        _style_acc_loss_axis(ax)
    ax_loss.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=LOG_X_MAJOR_TICKS))
    ax_loss.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax_loss.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax_acc.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    fig.align_ylabels([ax_acc, ax_loss])

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def render_acc_loss(*, also_sample_variant=True, dpi=H_DPI):
    """Panel h (fig3_h.png) + optional per-sample-styled variant (fig3_h_sample.png).

    Both are drawn from the same updcurve_clean.npz; the ``fig3_h_sample`` variant preserves the
    legacy step_acc_loss_sample output name so downstream assembly keeps working.
    """
    metrics = _load_acc_loss_metrics(UPDCURVE)
    h_png = _draw_acc_loss(metrics, HERE / "fig3_h", dpi)
    print(f"final_step={int(metrics['final_step'])} final_test_acc={float(metrics['test_acc'][metrics['final_idx']]):.6f}")
    print(h_png)
    if also_sample_variant:
        h_sample_png = _draw_acc_loss(metrics, HERE / "fig3_h_sample", dpi)
        print(h_sample_png)
        return h_png, h_sample_png
    return (h_png,)


# ================================================================================================
# Panel j (misclassified digits) : from plot_fig3_scikit_misclassified_digits.py
# ================================================================================================
J_WIDTH_IN = 3.11
J_HEIGHT_IN = 0.65
J_DPI = 600
J_PREFERRED_POSITIONS = (1, -2, 4, 5, 8)
J_DEFAULT_N_COLS = len(J_PREFERRED_POSITIONS)
J_BORDER = "#34383D"
J_BOX_BORDER_WIDTH = 0.55


def _load_misclassified(cache_path: Path, *, include_all: bool):
    data = np.load(cache_path, allow_pickle=True)
    Xte, split_y = _test_split()
    y_true = np.asarray(data["y_true"], dtype=int)
    y_pred = np.asarray(data["pred_final"], dtype=int)
    if not np.array_equal(split_y, y_true):
        raise ValueError("Cached labels do not match the expected scikit held-out split")
    miss = np.flatnonzero(y_true != y_pred)
    if include_all:
        positions = np.arange(miss.size, dtype=int)
        return Xte[miss], y_true[miss], y_pred[miss], miss, positions + 1
    positions = []
    for pos in J_PREFERRED_POSITIONS:
        resolved = int(pos)
        if resolved < 0:
            resolved = int(miss.size) + resolved + 1
        if 1 <= resolved <= miss.size and resolved not in positions:
            positions.append(resolved)
    for pos in range(int(miss.size), 0, -1):
        if len(positions) >= J_DEFAULT_N_COLS:
            break
        if pos not in positions:
            positions.append(pos)
    positions = np.asarray(positions[:J_DEFAULT_N_COLS], dtype=int) - 1
    selected = miss[positions]
    return Xte[selected], y_true[selected], y_pred[selected], selected, positions + 1


# A single NgSpiceShared instance is reused for every ngspice-touching render in this process.
# PySpice/cffi raise "duplicate declaration of struct ngcomplex" if NgSpiceShared() is constructed
# more than once in the same interpreter, so we build it lazily and share it.
_NG_SHARED = None


def _get_ng():
    global _NG_SHARED
    if _NG_SHARED is None:
        import train_scikit as T
        _NG_SHARED = T.NgSpiceShared(send_data=False)
    return _NG_SHARED


def _solve_scores(images, gates_path: Path):
    import train_scikit as T
    data = np.load(gates_path, allow_pickle=True)
    drain = np.asarray(data["drain"], dtype=int)
    source = np.asarray(data["source"], dtype=int)
    vg = np.asarray(data["vg_final"], dtype=float)
    if not (drain.size == source.size == vg.size):
        raise ValueError("drain, source, and vg_final must have the same length")
    ng = _get_ng()
    ng.load_circuit(T.build_netlist(drain, source, vg, np.zeros(vg.size), np.zeros(vg.size)))
    nodes = [900 + j for j in range(20)]
    rng = np.random.default_rng(0)
    vout = []
    for image in images:
        cmds = [f"alter VIN{p} dc = {float(image[p]):.8f}" for p in range(64)]
        cmds += [f"alter VINB{p} dc = {float(image[p]):.8f}" for p in range(64)]
        T.exec_chunked(ng, cmds)
        vout.append(T.read_nodes(ng, nodes, rng, 0.0))
    vout_arr = np.asarray(vout, dtype=float)
    scores = vout_arr[:, :10] - vout_arr[:, 10:]
    return vout_arr[:, :10], vout_arr[:, 10:], scores


def _draw_misclassified(images, y_true, y_pred, out: Path, *, n_cols, figure_width_in, figure_height_in, dpi=J_DPI) -> Path:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 5.0, "axes.titlesize": 5.0,
    })
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figure_width_in, figure_height_in))
    axes = np.asarray(axes).reshape(-1)
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.998, wspace=0.025, hspace=0.0)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    for ax, image in zip(axes, images):
        ax.imshow(image.reshape(8, 8), cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(J_BOX_BORDER_WIDTH)
            spine.set_edgecolor(J_BORDER)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def render_misclassified(*, include_all=False, dpi=J_DPI):
    """Panel j (fig3_j.png) or the all-misses variant (fig3_j_all.png)."""
    if not CONFUSION_CACHE.exists():
        # panel j depends on the confusion cache (holds pred_final); compute it if missing.
        _load_or_compute_confusions(GATES_CACHE, CONFUSION_CACHE, recompute=False)
    images, y_true, y_pred, test_idx, positions = _load_misclassified(CONFUSION_CACHE, include_all=include_all)
    v_plus, v_minus, scores = _solve_scores(images, GATES_CACHE)
    margins = scores[np.arange(len(y_true)), y_pred] - scores[np.arange(len(y_true)), y_true]
    n_cols = len(images) if include_all else J_DEFAULT_N_COLS
    n_rows = int(np.ceil(len(images) / n_cols))
    figure_width_in = (J_WIDTH_IN * n_cols / J_DEFAULT_N_COLS) if include_all else J_WIDTH_IN
    figure_height_in = J_HEIGHT_IN * n_rows
    out = HERE / ("fig3_j_all.png" if include_all else "fig3_j.png")
    values_path = MISCLASSIFIED_ALL_SCORES if include_all else MISCLASSIFIED_SCORES
    _draw_misclassified(images, y_true, y_pred, out, n_cols=n_cols,
                        figure_width_in=figure_width_in, figure_height_in=figure_height_in, dpi=dpi)
    values_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(values_path, positions=positions, test_indices=test_idx, y_true=y_true, y_pred=y_pred,
             v_plus=v_plus, v_minus=v_minus, scores=scores,
             delta_pred_minus_true=margins, delta_pred_minus_true_mV=margins * 1e3, gates=str(GATES_CACHE))
    print("true_pred=" + ",".join(f"{int(t)}->{int(p)}" for t, p in zip(y_true, y_pred)))
    print("delta_pred_minus_true_mV=" + ",".join(f"{float(v) * 1e3:.3f}" for v in margins))
    print(out)
    print(values_path)
    return out


# ================================================================================================
# Panel i (input-weighted sensitivity) : delegated to the fragile ngspice subprocess helper
# ================================================================================================
def render_sensitivity(*, workers=None, png_dpi=600):
    """Panel i (fig3_i.png + fig3_i_colorbar.png) via the ngspice subprocess-worker helper.

    This preserves the fragile `--worker` self-reinvocation pattern intact: we call the helper's
    main() with an argv it constructs its own subprocesses from. It also writes Supplementary Fig. 4's
    signed map/colorbar into suppl_figures/suppl4/data/ (unchanged)."""
    import plot_fig3_scikit_all_test_sensitivity as sens

    argv = [sys.argv[0]]
    if workers is not None:
        argv += ["--workers", str(int(workers))]
    argv += ["--png-dpi", str(int(png_dpi))]
    saved = sys.argv
    try:
        sys.argv = argv
        sens.main()
    finally:
        sys.argv = saved
    return HERE / "fig3_i.png"


# ================================================================================================
# Entry point
# ================================================================================================
ALL_PANELS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
FAST_PANELS = ["a", "d", "e", "f", "g", "h", "j"]  # excludes 3000-dpi nets (b,c) and ngspice sensitivity (i)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate Figure 3 panels (single entry point).")
    p.add_argument("--panels", nargs="+", choices=ALL_PANELS, default=None,
                   help="Subset of panels to render (default: all). Panel a renders the gate-rail bars.")
    p.add_argument("--fast", action="store_true",
                   help="Render only fast panels (a-bars, d, e, f, g, h, j); skip 3000-dpi nets (b,c) and ngspice sensitivity (i).")
    p.add_argument("--recompute-confusion", action="store_true",
                   help="Force ngspice recomputation of the confusion cache instead of reusing data/*.npz.")
    p.add_argument("--sensitivity-workers", type=int, default=None,
                   help="Worker count for the panel-i ngspice sensitivity sweep.")
    p.add_argument("--no-variants", action="store_true",
                   help="Skip the auxiliary variants (fig3_h_sample.png, fig3_j_all.png).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.panels is not None:
        panels = list(args.panels)
    elif args.fast:
        panels = list(FAST_PANELS)
    else:
        panels = list(ALL_PANELS)
    panels = [p for p in ALL_PANELS if p in panels]  # canonical order

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    made_variants = not args.no_variants

    for panel in panels:
        print(f"\n=== panel {panel} ===")
        if panel == "a":
            render_gate_rail_fields()  # panel-a bars (diagram fig3_a*.png is a static asset, not regenerated)
        elif panel in ("b", "c"):
            # b and c come from the same 3000-dpi render; run once when either is requested.
            if panel == "b" or "b" not in panels:
                render_network_states()
        elif panel in ("d", "e"):
            if panel == "d" or "d" not in panels:
                # if the network render already ran (b/c requested), d/e are already produced.
                if not (("b" in panels) or ("c" in panels)):
                    render_vg_histograms()
        elif panel in ("f", "g"):
            if panel == "f" or "f" not in panels:
                render_confusions(recompute=args.recompute_confusion)
        elif panel == "h":
            render_acc_loss(also_sample_variant=made_variants)
        elif panel == "i":
            render_sensitivity(workers=args.sensitivity_workers)
        elif panel == "j":
            render_misclassified(include_all=False)
            if made_variants:
                render_misclassified(include_all=True)

    print("\nFigure 3 panels complete:", ", ".join(panels))


if __name__ == "__main__":
    main()
