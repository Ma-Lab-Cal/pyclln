#!/usr/bin/env python3
"""Consolidated renderer for Figure 4 -- "Autoregressive language generation in a CLLN".

This single entry point produces every script-generated panel of Fig. 4:

  * panel a : the seven-token context -> 13,664-nMOS bipartite network diagram
              (``fig4_a_network.png``), its gate-voltage colorbar
              (``fig4_a_colorbar.png``), and the measured next-token probability
              bars shown beside the network (``fig4_a_next_token_hist.png``).
  * panel d : train/test QMass vs epoch (``fig4_d_qmass.png``).
  * panel e : train/test next-token cross-entropy vs epoch (``fig4_e_cross_entropy.png``).
  * panel f : validity / novelty / uniqueness vs readout temperature
              (``fig4_f_temperature_sweep.png``).

Panels b and c of the manuscript figure are typeset example sentences (not
produced by any script) and are therefore not generated here.

An auxiliary gate-voltage histogram (not used in the manuscript composite) is
also produced as ``fig4_gate_voltage_hist.png`` for completeness.

All input data is read from ``paper_release/language_model`` (gates, curves,
vocab, token classes, inference model) and from the shipped temperature-sweep
JSON in this folder. Every ``.json``/``.npz`` file that a panel *writes* is
placed in the ``data/`` subfolder next to this script.

The network diagram is drawn purely from the shipped endpoint gate voltages
(no circuit solve).  The next-token histogram calls ngspice once through the
paper-release inference model (a single free .op solve) and is the only panel
that touches the simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_PR = next(p for p in Path(__file__).resolve().parents if (p / "device_model").is_dir())  # robust paper_release root (self-contained)
def _relsrc(p):
    try: return str(Path(p).resolve().relative_to(_PR.resolve()))
    except Exception: return Path(p).name

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors, font_manager, ticker
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Shared locations
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"

PNG_DPI = 600

FONT_FAMILY = ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"]
YTICK_FONT_FAMILY = ["Roboto Mono", "DejaVu Sans Mono", "monospace"]
ROBOTO_MONO_FONT = OUT_DIR / "fonts" / "RobotoMono-Regular.ttf"
AXIS_COLOR = "#34383D"


# =========================================================================== #
# Panel a -- network diagram + colorbar
# =========================================================================== #
NET_DEFAULT_RUN_DIR = _PR / "language_model" / "runs" / "clean"
NET_GATE_FILE = _PR / "language_model" / "runs" / "clean" / "gates.npz"

NET_FIG_SIZE = (3.11, 2.5101)
COLORBAR_SIZE = (1.20, 0.52)
NETWORK_XLIM = (0.075, 0.9016)
NETWORK_YLIM = (0.060, 0.956)

TEXT = "#34383D"
BORDER = "#B8BDC2"
NEUTRAL = "#F3F4F4"
NETWORK_NEUTRAL = "#DDE2E4"
GATE_LOW = "#4B86B4"
GATE_LOW_NEAR_CENTER = "#B6CAD5"
GATE_CENTER = NETWORK_NEUTRAL
GATE_HIGH_NEAR_CENTER = "#EBC6B2"
GATE_HIGH = "#ED9360"

NET_TEXT_SIZE = 5.0
WORD_GROUP_TEXT_SIZE = 5.0

CONTEXT_LEN = 7
TOKEN_EMBED_DIM = 16
GATE_DISPLAY_VMIN = 0.4
GATE_DISPLAY_VMAX = 8.0
GATE_VMIN = 1.0
GATE_VCENTER = 3.54
GATE_VMAX = 6.5

CONTENT_LIST_ROLES = {
    "ADJECTIVES": "adjective",
    "PARTICLES": "particle",
    "PROPERTIES": "property",
    "APPARATUS": "apparatus",
    "MEDIA": "medium",
    "STATEFUL": "stateful",
    "OUTCOMES": "outcome",
}
FUNCTION_TOKEN_ROLES = {
    ".": "terminal",
    "?": "terminal",
    "the": "determiner",
    "a": "determiner",
    "an": "determiner",
    "this": "determiner",
    "that": "determiner",
    "each": "determiner",
    "every": "determiner",
    "what": "question",
    "why": "question",
    "in": "preposition",
    "of": "preposition",
    "is": "relation",
    "has": "relation",
    "can": "relation",
    "measure": "relation",
    "shows": "relation",
    "measures": "relation",
}
GRAMMAR_ROLE_ORDER = (
    "terminal",
    "determiner",
    "question",
    "preposition",
    "relation",
    "particle",
    "property",
    "apparatus",
    "medium",
    "stateful",
    "outcome",
    "adjective",
    "function",
)
ELLIPSE_ROLE_ORDER = (
    "terminal",
    "property",
    "adjective",
    "apparatus",
    "question",
    "particle",
    "relation",
    "medium",
    "preposition",
    "stateful",
    "determiner",
    "outcome",
    "function",
)
ROLE_LABELS = {
    "terminal": "Sentence endings",
    "determiner": "Determiners",
    "question": "Question words",
    "preposition": "Linking words",
    "relation": "Actions",
    "particle": "Physical things",
    "property": "Measured qualities",
    "apparatus": "Instruments",
    "medium": "Materials & places",
    "stateful": "States & systems",
    "outcome": "Effects & events",
    "adjective": "Descriptions",
    "function": "Grammar words",
}
ELLIPSE_ROLE_LABELS = {
    "terminal": "End marks",
    "determiner": "Determiners",
    "question": "Question\nwords",
    "preposition": "Linking\nwords",
    "relation": "Actions",
    "particle": "Things",
    "property": "Qualities",
    "apparatus": "Tools",
    "medium": "Materials",
    "stateful": "States",
    "outcome": "Effects",
    "adjective": "Descriptions",
    "function": "Grammar",
}


def _set_common_rc(font_size: float = NET_TEXT_SIZE) -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY[0],
            "font.sans-serif": FONT_FAMILY,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )


def _gate_cmap() -> colors.Colormap:
    return colors.LinearSegmentedColormap.from_list(
        "gate_voltage_lm",
        [
            (0.00, GATE_LOW),
            (0.44, GATE_LOW_NEAR_CENTER),
            (0.50, GATE_CENTER),
            (0.56, GATE_HIGH_NEAR_CENTER),
            (1.00, GATE_HIGH),
        ],
        N=512,
    )


def _gate_norm(gate_vmax: float, gate_vcenter: float) -> colors.TwoSlopeNorm:
    return colors.TwoSlopeNorm(vmin=GATE_VMIN, vcenter=gate_vcenter, vmax=gate_vmax)


def _symmetric_gate_vmax(gate_vcenter: float) -> float:
    eps = 1e-6
    return max(float(gate_vcenter + (gate_vcenter - GATE_VMIN)), gate_vcenter + eps)


def _gate_display_cmap(gate_vmax: float, gate_vcenter: float) -> colors.Colormap:
    display_values = np.linspace(GATE_DISPLAY_VMIN, GATE_DISPLAY_VMAX, 512)
    clipped_values = np.clip(display_values, GATE_VMIN, gate_vmax)
    rgba = _gate_cmap()(_gate_norm(gate_vmax, gate_vcenter)(clipped_values))
    return colors.ListedColormap(rgba, name="gate_voltage_lm_display")


def _gate_rgba(vg: np.ndarray, *, alpha: float, gate_vmax: float, gate_vcenter: float) -> np.ndarray:
    clipped = np.clip(np.asarray(vg, dtype=float), GATE_VMIN, gate_vmax)
    rgba = _gate_cmap()(_gate_norm(gate_vmax, gate_vcenter)(clipped))
    rgba[:, 3] = alpha
    return rgba


def _gate_vcenter(gate_matrix: np.ndarray, gate_vmax: float) -> float:
    finite = np.asarray(gate_matrix[np.isfinite(gate_matrix)], dtype=float)
    if finite.size == 0:
        return GATE_VCENTER
    mean = float(np.mean(finite))
    eps = 1e-6
    return float(np.clip(mean, GATE_VMIN + eps, gate_vmax - eps))


def _load_vocab(run_dir: Path) -> list[str]:
    vocab_txt = _PR / "language_model" / "vocab.txt"
    if vocab_txt.exists():
        vocab = [t for t in vocab_txt.read_text(encoding="utf-8").splitlines() if t != ""]
        if vocab[0] != "<BOS>":
            raise ValueError(f"Expected vocab[0] to be <BOS>, got {vocab[0]!r}")
        return vocab
    meta_path = run_dir / "run_meta.json"
    vocab: list[str] | None = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta.get("vocab"), list):
            vocab = [str(word) for word in meta["vocab"]]
    if vocab is None:
        manifest_path = run_dir / "embedding_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        vocab = [str(word) for word in manifest["vocab"]]
    if vocab[0] != "<BOS>":
        raise ValueError(f"Expected vocab[0] to be <BOS>, got {vocab[0]!r}")
    return vocab


def _load_gate_matrix(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "vg_final" not in loaded.files:
            raise KeyError(f"{path} must contain vg_final")
        return np.asarray(loaded["vg_final"], dtype=float)
    return np.asarray(loaded, dtype=float)


def _load_token_class_map() -> dict[str, str]:
    """Token -> grammatical role (for node coloring), read from the shipped token_classes.json."""
    path = _PR / "language_model" / "token_classes.json"
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _role_for_token(token: str, token_class: dict[str, str]) -> str:
    return FUNCTION_TOKEN_ROLES.get(token, token_class.get(token, "function"))


def _output_group_order(output_tokens: list[str]) -> tuple[list[int], list[dict[str, object]]]:
    token_class = _load_token_class_map()
    roles = [_role_for_token(token, token_class) for token in output_tokens]
    role_order = [role for role in GRAMMAR_ROLE_ORDER if role in roles]
    role_order.extend(sorted(set(roles).difference(role_order)))

    order: list[int] = []
    groups: list[dict[str, object]] = []
    for role in role_order:
        indices = [idx for idx, found_role in enumerate(roles) if found_role == role]
        if not indices:
            continue
        start = len(order)
        order.extend(indices)
        groups.append(
            {
                "role": role,
                "label": ROLE_LABELS.get(role, role.title()),
                "start": start,
                "stop": len(order),
            }
        )
    return order, groups


def _reorder_groups_for_ellipse(
    visual_order: list[int],
    groups: list[dict[str, object]],
) -> tuple[list[int], list[dict[str, object]]]:
    by_role = {str(group["role"]): group for group in groups}
    role_order = [role for role in ELLIPSE_ROLE_ORDER if role in by_role]
    role_order.extend(role for role in by_role if role not in role_order)

    reordered_indices: list[int] = []
    reordered_groups: list[dict[str, object]] = []
    for role in role_order:
        group = by_role[role]
        old_start = int(group["start"])
        old_stop = int(group["stop"])
        new_start = len(reordered_indices)
        reordered_indices.extend(visual_order[old_start:old_stop])
        new_group = dict(group)
        new_group["start"] = new_start
        new_group["stop"] = len(reordered_indices)
        reordered_groups.append(new_group)
    return reordered_indices, reordered_groups


def _input_positions(
    context_layout: str,
    *,
    y_min: float = 0.075,
    y_max: float = 0.940,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if context_layout == "vertical":
        x_pitch = 0.0120
        y_pitch = x_pitch * NET_FIG_SIZE[0] / NET_FIG_SIZE[1]
        token_centers = [
            (0.105, float(y))
            for y in np.linspace(y_max - 1.5 * y_pitch, y_min + 1.5 * y_pitch, CONTEXT_LEN)
        ]
    else:
        token_centers = [(float(x), 0.525) for x in np.linspace(0.075, 0.500, CONTEXT_LEN)]
        x_pitch = 0.0120
        y_pitch = 0.0152
    positions = []
    boxes: list[dict[str, float]] = []

    for x, y in token_centers:
        for row in range(4):
            for col in range(4):
                positions.append(
                    (
                        float(x + (col - 1.5) * x_pitch),
                        float(y + (1.5 - row) * y_pitch),
                    )
                )
        boxes.append(
            {
                "x0": float(x - 2.12 * x_pitch),
                "x1": float(x + 2.12 * x_pitch),
                "y0": float(y - 2.12 * y_pitch),
                "y1": float(y + 2.12 * y_pitch),
                "center_x": float(x),
                "center_y": float(y),
            }
        )
    return np.asarray(positions, dtype=float), boxes


def _grouped_output_y(groups: list[dict[str, object]], *, group_gap: float = 2.25) -> tuple[np.ndarray, list[dict[str, float]]]:
    raw: list[float] = []
    spans: list[tuple[int, int]] = []
    cursor = 0.0
    for group in groups:
        size = int(group["stop"]) - int(group["start"])
        start = len(raw)
        raw.extend(cursor + np.arange(size, dtype=float))
        stop = len(raw)
        spans.append((start, stop))
        cursor += size + group_gap

    arr = np.asarray(raw, dtype=float)
    arr = 0.940 - (arr - arr.min()) / (arr.max() - arr.min()) * 0.865

    bounds: list[dict[str, float]] = []
    for group, (start, stop) in zip(groups, spans):
        ys = arr[start:stop]
        bounds.append(
            {
                "role": str(group.get("role", "")),
                "label": str(group["label"]),
                "start": start,
                "stop": stop,
                "center": float(np.mean(ys)),
                "top": float(np.max(ys)),
                "bottom": float(np.min(ys)),
            }
        )
    return arr, bounds


def _output_positions(output_tokens: list[str], output_layout: str) -> tuple[np.ndarray, list[dict[str, float]], list[int]]:
    visual_order, groups = _output_group_order(output_tokens)
    if output_layout in {"stacked", "ellipse"}:
        visual_order, groups = _reorder_groups_for_ellipse(visual_order, groups)
    if len(visual_order) != len(output_tokens):
        raise ValueError(f"Output grouping returned {len(visual_order)} rows for {len(output_tokens)} tokens")
    y, group_bounds = _grouped_output_y(
        groups,
        group_gap=4.50 if output_layout in {"arc", "ellipse", "stacked"} else 2.25,
    )
    if output_layout == "arc":
        theta_max = np.deg2rad(83.0)
        theta = np.interp(y, [float(np.min(y)), float(np.max(y))], [-theta_max, theta_max])
        x = 0.166 + 0.195 * np.cos(theta)
        y = 0.525 + 0.445 * np.sin(theta)
        for group in group_bounds:
            start = int(group["start"])
            stop = int(group["stop"])
            ys = y[start:stop]
            group["center"] = float(np.mean(ys))
            group["top"] = float(np.max(ys))
            group["bottom"] = float(np.min(ys))
    elif output_layout == "ellipse":
        cx, cy = 0.300, 0.525
        rx, ry = 0.268, 0.362
        x = np.zeros(len(output_tokens), dtype=float)
        y = np.zeros(len(output_tokens), dtype=float)
        n_groups = len(group_bounds)
        cluster_step = 0.0122
        for group_idx, group in enumerate(group_bounds):
            start = int(group["start"])
            stop = int(group["stop"])
            size = stop - start
            theta = np.pi / 2.0 + np.pi / n_groups - 2.0 * np.pi * group_idx / n_groups
            anchor = np.asarray([cx + rx * np.cos(theta), cy + ry * np.sin(theta)], dtype=float)
            tangent = np.asarray([-rx * np.sin(theta), ry * np.cos(theta)], dtype=float)
            tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
            radial = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)

            cols = int(np.ceil(np.sqrt(size)))
            rows = int(np.ceil(size / cols))
            for local_idx, visual_idx in enumerate(range(start, stop)):
                row = local_idx // cols
                col = local_idx % cols
                used_cols = cols if row < rows - 1 else size - row * cols
                col_center = 0.5 * (used_cols - 1)
                row_center = 0.5 * (rows - 1)
                offset = (
                    tangent * ((col - col_center) * cluster_step)
                    + radial * ((row_center - row) * cluster_step)
                )
                x[visual_idx], y[visual_idx] = anchor + offset

            idx = np.arange(start, stop)
            xs = x[idx]
            ys = y[idx]
            mean_x = float(np.mean(xs))
            group["center"] = float(np.mean(ys))
            group["top"] = float(np.max(ys))
            group["bottom"] = float(np.min(ys))
            role = str(group["role"])
            ellipse_label = ELLIPSE_ROLE_LABELS.get(role, str(group["label"]))
            above_context = float(np.mean(ys)) >= cy
            label_y = float(np.max(ys) + 0.020 if above_context else np.min(ys) - 0.020)
            if mean_x < 0.115:
                label_x = float(np.max(xs) + 0.020)
                label_ha = "left"
            elif mean_x > 0.520:
                label_x = float(np.min(xs) - 0.020)
                label_ha = "right"
            else:
                label_x = float(np.mean(xs))
                label_ha = "center"

            if role == "outcome":
                label_x = float(np.mean(xs))
                label_ha = "center"
                label_y = float(np.max(ys) + 0.020)
                above_context = True
            elif role in {"determiner", "apparatus"}:
                label_x = float(np.min(xs))
                label_ha = "left"
                label_y = float(np.max(ys) + 0.020)
                above_context = True
            elif role == "stateful":
                label_x = float(np.min(xs))
                label_ha = "left"
                label_y = float(np.min(ys) - 0.020)
                above_context = False
            elif role in {"question", "preposition"}:
                label_x = float(np.mean(xs))
                label_ha = "center"
                label_y = float(np.min(ys) - 0.020)
                above_context = False

            group["ellipse_label"] = ellipse_label
            group["label_x"] = float(np.clip(label_x, 0.018, 0.760))
            group["label_y"] = float(np.clip(label_y, 0.065, 0.940))
            group["label_ha"] = label_ha
            group["label_va"] = "bottom" if above_context else "top"
            group["tick_x"] = float(np.mean(xs))
            group["tick_y"] = float(np.mean(ys))
    else:
        x = np.full(len(output_tokens), 0.735, dtype=float)
    return np.column_stack([x, y]), group_bounds, visual_order


def _segments_and_weights(
    input_pos: np.ndarray,
    output_pos: np.ndarray,
    gate_matrix: np.ndarray,
    output_order: list[int],
    gate_vcenter: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    segments: list[np.ndarray] = []
    weights: list[float] = []
    for visual_idx, out_xy in enumerate(output_pos):
        out_idx = output_order[visual_idx]
        for in_idx, in_xy in enumerate(input_pos):
            segments.append(np.asarray([in_xy, out_xy], dtype=float))
            weights.append(float(gate_matrix[out_idx, in_idx]))
    weights_arr = np.asarray(weights, dtype=float)

    order = np.argsort(np.abs(weights_arr - float(gate_vcenter)), kind="stable")
    return [segments[int(i)] for i in order], weights_arr[order]


def _draw_window_boxes(ax: plt.Axes, boxes: list[dict[str, float]]) -> None:
    for box in boxes:
        ax.add_patch(
            Rectangle(
                (box["x0"], box["y0"]),
                box["x1"] - box["x0"],
                box["y1"] - box["y0"],
                facecolor="none",
                edgecolor=NETWORK_NEUTRAL,
                linewidth=0.32,
                zorder=2,
            )
        )


def _draw_input_labels(
    ax: plt.Axes,
    boxes: list[dict[str, float]],
    context_layout: str,
    output_layout: str,
) -> None:
    return


def _draw_output_groups(ax: plt.Axes, group_bounds: list[dict[str, float]], output_layout: str) -> None:
    if output_layout == "ellipse":
        for group in group_bounds:
            label_x = float(group["label_x"])
            label_y = float(group["label_y"])
            ax.text(
                label_x,
                label_y,
                str(group.get("ellipse_label", group["label"])),
                ha=str(group["label_ha"]),
                va=str(group["label_va"]),
                fontsize=WORD_GROUP_TEXT_SIZE,
                color=TEXT,
                zorder=9,
                linespacing=0.92,
            )
        return

    if output_layout == "arc":
        x0 = 0.392
        x1 = 0.404
        label_x = 0.416
        label_size = WORD_GROUP_TEXT_SIZE
    else:
        x0 = 0.759
        x1 = 0.771
        label_x = 0.790
        label_size = WORD_GROUP_TEXT_SIZE
    label_centers = [float(group["center"]) for group in group_bounds]
    if output_layout == "arc":
        min_gap = 0.034
        for idx in range(1, len(label_centers)):
            label_centers[idx] = min(label_centers[idx], label_centers[idx - 1] - min_gap)

    for group, label_center in zip(group_bounds, label_centers):
        top = group["top"] + 0.003
        bottom = group["bottom"] - 0.003
        center = group["center"]
        ax.plot([x0, x1], [top, top], color=NETWORK_NEUTRAL, linewidth=0.34, zorder=2)
        ax.plot([x1, x1], [bottom, top], color=NETWORK_NEUTRAL, linewidth=0.34, zorder=2)
        ax.plot([x0, x1], [bottom, bottom], color=NETWORK_NEUTRAL, linewidth=0.34, zorder=2)
        ax.text(
            label_x,
            label_center,
            ELLIPSE_ROLE_LABELS.get(str(group["role"]), str(group["label"])),
            ha="left",
            va="center",
            fontsize=label_size,
            color=TEXT,
            linespacing=0.92,
        )


def _save_hires(fig: plt.Figure, out_prefix: Path, dpi: int) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    expected_px = tuple(int(round(v * dpi)) for v in fig.get_size_inches())
    fig.savefig(png, dpi=dpi, facecolor="white", transparent=False)
    plt.close(fig)
    with Image.open(png) as im:
        out = im.convert("RGB")
        if out.size != expected_px:
            out = out.resize(expected_px, Image.Resampling.LANCZOS)
        out.save(png, dpi=(dpi, dpi))
    return png


def draw_colorbar(out_prefix: Path, *, gate_vmax: float, gate_vcenter: float, dpi: int) -> Path:
    _set_common_rc()
    fig = plt.figure(figsize=COLORBAR_SIZE)
    cax = fig.add_axes([0.10, 0.62, 0.84, 0.12])
    mappable = plt.cm.ScalarMappable(
        norm=colors.Normalize(vmin=GATE_DISPLAY_VMIN, vmax=GATE_DISPLAY_VMAX),
        cmap=_gate_display_cmap(gate_vmax, gate_vcenter),
    )
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.outline.set_visible(True)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor(TEXT)
    cb.set_ticks([GATE_DISPLAY_VMIN, gate_vcenter, GATE_DISPLAY_VMAX])
    cb.set_ticklabels([f"{GATE_DISPLAY_VMIN:.1f}", f"{gate_vcenter:.2f}", f"{GATE_DISPLAY_VMAX:.1f}"])
    cb.ax.tick_params(length=1.4, width=0.28, pad=1.0, labelsize=NET_TEXT_SIZE, colors=TEXT)
    cb.set_label("final VG (V)", fontsize=NET_TEXT_SIZE, labelpad=1.2, color=TEXT)
    return _save_hires(fig, out_prefix, dpi)


def draw_network(
    gate_matrix: np.ndarray,
    vocab: list[str],
    out_prefix: Path,
    *,
    edge_width: float,
    edge_alpha: float,
    gate_vmax: float,
    dpi: int,
    context_layout: str,
    output_layout: str,
) -> Path:
    output_tokens = vocab[1:]
    if gate_matrix.shape != (len(output_tokens), CONTEXT_LEN * TOKEN_EMBED_DIM):
        raise ValueError(
            f"Expected gates shape {(len(output_tokens), CONTEXT_LEN * TOKEN_EMBED_DIM)}, "
            f"got {gate_matrix.shape}"
        )

    _set_common_rc()
    gate_vcenter = _gate_vcenter(gate_matrix, gate_vmax)
    gate_vmax = _symmetric_gate_vmax(gate_vcenter)
    output_pos, group_bounds, output_order = _output_positions(output_tokens, output_layout)
    input_pos, input_boxes = _input_positions(
        context_layout,
        y_min=float(np.min(output_pos[:, 1])),
        y_max=float(np.max(output_pos[:, 1])),
    )
    segments, weights = _segments_and_weights(input_pos, output_pos, gate_matrix, output_order, gate_vcenter)

    fig, ax = plt.subplots(figsize=NET_FIG_SIZE)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(*NETWORK_XLIM)
    ax.set_ylim(*NETWORK_YLIM)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    ax.add_collection(
        LineCollection(
            segments,
            colors=_gate_rgba(weights, alpha=edge_alpha, gate_vmax=gate_vmax, gate_vcenter=gate_vcenter),
            linewidths=edge_width,
            zorder=1,
            capstyle="round",
            joinstyle="round",
        )
    )

    _draw_window_boxes(ax, input_boxes)
    _draw_output_groups(ax, group_bounds, output_layout)

    ax.scatter(
        input_pos[:, 0],
        input_pos[:, 1],
        marker="s",
        s=8.0,
        facecolors="#FFFFFF",
        edgecolors=BORDER,
        linewidths=0.16,
        zorder=4,
    )
    ax.scatter(
        output_pos[:, 0],
        output_pos[:, 1],
        marker="o",
        s=4.5,
        facecolors="#FFFFFF",
        edgecolors=TEXT,
        linewidths=0.14,
        zorder=5,
    )

    _draw_input_labels(ax, input_boxes, context_layout, output_layout)

    return _save_hires(fig, out_prefix, dpi)


def render_panel_a_network(
    out_dir: Path,
    dpi: int,
    *,
    run_dir: Path = NET_DEFAULT_RUN_DIR,
    vg_file: Path = NET_GATE_FILE,
    edge_width: float = 0.065,
    edge_alpha: float = 0.40,
    context_layout: str = "vertical",
    output_layout: str = "stacked",
) -> list[Path]:
    """Panel a: the context -> vocab bipartite network diagram and its colorbar."""
    gate_matrix = _load_gate_matrix(vg_file)
    vocab = _load_vocab(run_dir)
    gate_vcenter = _gate_vcenter(gate_matrix, GATE_VMAX)
    gate_vmax = _symmetric_gate_vmax(gate_vcenter)
    network_png = draw_network(
        gate_matrix,
        vocab,
        out_dir / "fig4_a_network",
        edge_width=edge_width,
        edge_alpha=edge_alpha,
        gate_vmax=gate_vmax,
        dpi=dpi,
        context_layout=context_layout,
        output_layout=output_layout,
    )
    colorbar_png = draw_colorbar(
        out_dir / "fig4_a_colorbar",
        gate_vmax=gate_vmax,
        gate_vcenter=gate_vcenter,
        dpi=dpi,
    )
    return [network_png, colorbar_png]


# =========================================================================== #
# Panel a -- next-token probability bars (measured on real ngspice)
# =========================================================================== #
NEXT_TOKEN_RELEASE_RESULTS = _PR / "language_model" / "results"
NEXT_TOKEN_CONTEXT = "The eigenfunction shows dechoerence in the"
NEXT_TOKEN_TEXT_SIZE = 6.0
NEXT_TOKEN_BAR_COLOR = "#7A7A7A"
NEXT_TOKEN_HISTOGRAM_LAST_TOKEN = "conductor"
NEXT_TOKEN_DISPLAY_ROW_COUNT = 9
NEXT_TOKEN_CORRECTIONS = {
    "dechoerence": "decoherence",
}


def _set_next_token_rc() -> None:
    plt.style.use("default")
    if ROBOTO_MONO_FONT.exists():
        font_manager.fontManager.addfont(str(ROBOTO_MONO_FONT))
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY[0],
            "font.sans-serif": FONT_FAMILY,
            "font.size": NEXT_TOKEN_TEXT_SIZE,
            "axes.labelsize": NEXT_TOKEN_TEXT_SIZE,
            "xtick.labelsize": NEXT_TOKEN_TEXT_SIZE,
            "ytick.labelsize": NEXT_TOKEN_TEXT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _set_next_token_tick_font(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels():
        label.set_fontfamily(FONT_FAMILY[0])
        label.set_fontsize(NEXT_TOKEN_TEXT_SIZE)
    ytick_props = (
        font_manager.FontProperties(fname=str(ROBOTO_MONO_FONT), size=NEXT_TOKEN_TEXT_SIZE)
        if ROBOTO_MONO_FONT.exists()
        else font_manager.FontProperties(family=YTICK_FONT_FAMILY, size=NEXT_TOKEN_TEXT_SIZE)
    )
    for label in ax.get_yticklabels():
        label.set_fontproperties(ytick_props)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _next_token_default_temp(release_results_dir: Path) -> float:
    results_json = release_results_dir / "results.json"
    if results_json.exists():
        op_text = str(_load_json(results_json).get("operating_point", ""))
        match = re.search(r"T\s*=\s*([0-9.eE+-]+)", op_text)
        if match:
            return float(match.group(1))
    return 0.0006


def _next_token_probabilities(inf_module, run: str, context_text: str, temp: float) -> dict[str, object]:
    corrected = " ".join(NEXT_TOKEN_CORRECTIONS.get(w.lower(), w) for w in context_text.split())
    notes = [
        f"{w}->{NEXT_TOKEN_CORRECTIONS[w.lower()]}"
        for w in context_text.split()
        if w.lower() in NEXT_TOKEN_CORRECTIONS
    ]
    model = inf_module.Model(run, chip=None, temp=temp)
    ids = model.context_row(model.tokenize(corrected))
    probs = model.probs([ids], temp)[0]
    context_tokens = [model.vocab[i] for i in ids if i != inf_module.TL.BOS_ID]
    return {
        "source": "paper_release/language_model runs/clean endpoint gates; ngspice .op free solve, 122-token grammar (123 symbols incl. <BOS>)",
        "run": run,
        "context_tokens": context_tokens,
        "correction_notes": notes,
        "temp": temp,
        "probs": probs,
        "vocab": model.vocab,
    }


def _next_token_histogram_indices_and_other(vocab: list[str], probs: np.ndarray) -> tuple[np.ndarray, float]:
    non_bos = np.arange(1, len(vocab))
    ranked = non_bos[np.argsort(probs[non_bos])[::-1]]
    stop_positions = np.flatnonzero(
        [vocab[int(idx)] == NEXT_TOKEN_HISTOGRAM_LAST_TOKEN for idx in ranked]
    )
    if stop_positions.size == 0:
        raise ValueError(
            f"{NEXT_TOKEN_HISTOGRAM_LAST_TOKEN!r} is not present in the ranked next-token vocabulary."
        )
    shown = ranked[: int(stop_positions[0]) + 1]
    other = float(np.sum(probs[non_bos]) - np.sum(probs[shown]))
    return shown, other


def _save_next_token_table(result: dict[str, object], out_prefix: Path) -> Path:
    vocab = list(result["vocab"])
    probs = np.asarray(result["probs"], dtype=float)
    rows = [
        {"token": token, "vocab_index": idx, "probability": float(probs[idx])}
        for idx, token in enumerate(vocab)
    ]
    table_path = out_prefix.with_suffix(".json")
    top_idx, other = _next_token_histogram_indices_and_other(vocab, probs)
    summary = [
        {"token": vocab[int(idx)], "vocab_index": int(idx), "probability": float(probs[int(idx)])}
        for idx in top_idx
    ]
    summary.append({"token": "others", "vocab_index": None, "probability": other})
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "context_tokens": result["context_tokens"],
                "corrections": result["correction_notes"],
                "source": result["source"],
                "run": result["run"],
                "temperature": result["temp"],
                "shown_through_token": NEXT_TOKEN_HISTOGRAM_LAST_TOKEN,
                "shown_tokens_plus_others": summary,
                "rows": rows,
            },
            f,
            indent=2,
        )
    return table_path


def _plot_next_token(result: dict[str, object], out_prefix: Path, dpi: int) -> Path:
    vocab = list(result["vocab"])
    probs = np.asarray(result["probs"], dtype=float)
    top_idx, other = _next_token_histogram_indices_and_other(vocab, probs)
    labels = [vocab[int(idx)] for idx in top_idx] + ["others"]
    p = np.asarray([float(probs[int(idx)]) for idx in top_idx] + [other])
    x = np.linspace(0, NEXT_TOKEN_DISPLAY_ROW_COUNT - 1, len(labels))

    _set_next_token_rc()
    fig, ax = plt.subplots(figsize=(1.06, 2.50))
    fig.subplots_adjust(left=0.100, right=0.430, bottom=0.030, top=0.900)

    ax.barh(x, p, height=0.8, color=NEXT_TOKEN_BAR_COLOR)
    max_p = float(np.max(p))

    ax.set_ylim(NEXT_TOKEN_DISPLAY_ROW_COUNT - 0.5, -0.5)
    ax.set_xlim(min(1.0, max(0.4, max_p * 1.08)), 0.0)
    ax.set_xticks([0.0, 0.2, 0.4])
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, labelsize=NEXT_TOKEN_TEXT_SIZE)
    ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, labelsize=NEXT_TOKEN_TEXT_SIZE)
    _set_next_token_tick_font(ax)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def render_panel_a_next_token(
    out_dir: Path,
    data_dir: Path,
    dpi: int,
    *,
    run: str = "clean",
    context: str = NEXT_TOKEN_CONTEXT,
    temp: float | None = None,
    release_results_dir: Path = NEXT_TOKEN_RELEASE_RESULTS,
) -> list[Path]:
    """Panel a: measured next-token probability bars beside the network."""
    sys.path.insert(0, str(_PR / "language_model"))
    import infer_language_model as INF  # imported lazily -- pulls in ngspice tooling

    resolved_temp = _next_token_default_temp(release_results_dir) if temp is None else float(temp)
    result = _next_token_probabilities(INF, run, context, resolved_temp)
    png = _plot_next_token(result, out_dir / "fig4_a_next_token_hist", int(dpi))
    table = _save_next_token_table(result, data_dir / "fig4_a_next_token_hist")
    return [png, table]


# =========================================================================== #
# Panels d and e -- learning curves (QMass + cross-entropy)
# =========================================================================== #
CURVE_DEFAULT_DATA = _PR / "language_model" / "runs" / "clean" / "curve.npz"
CURVE_FIG_SIZE = (2.06, 1.73)
CURVE_TEXT_SIZE = 6.0
CURVE_LEGEND_TEXT_SIZE = 5.0
TRAIN_COLOR = "#274E87"
TEST_COLOR = "#D34F72"
CURVE_LINEWIDTH = 0.65
CURVE_ALPHA = 0.7
MARKER_SIZE = 2.2
MARKER_EDGE_WIDTH = 0.0


def _set_curve_rc() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY[0],
            "font.sans-serif": FONT_FAMILY,
            "font.size": CURVE_TEXT_SIZE,
            "axes.labelsize": CURVE_TEXT_SIZE,
            "xtick.labelsize": CURVE_TEXT_SIZE,
            "ytick.labelsize": CURVE_TEXT_SIZE,
            "legend.fontsize": CURVE_LEGEND_TEXT_SIZE,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY[0],
            "mathtext.it": FONT_FAMILY[0],
            "mathtext.bf": FONT_FAMILY[0],
            "mathtext.sf": FONT_FAMILY[0],
            "mathtext.tt": FONT_FAMILY[0],
            "mathtext.cal": FONT_FAMILY[0],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_curve_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=0.45,
        length=2.0,
        pad=1.5,
        colors=AXIS_COLOR,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor(AXIS_COLOR)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY[0])
        label.set_fontsize(CURVE_TEXT_SIZE)


def _padded_ylim(train: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([train, test]).astype(float)
    vals = vals[np.isfinite(vals)]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = max(hi - lo, 1e-6)
    return lo - 0.12 * span, hi + 0.12 * span


def _plot_curve(
    epochs: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ylabel: str,
    out_prefix: Path,
    dpi: int,
    ylim: tuple[float, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=CURVE_FIG_SIZE)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.245, top=0.965)

    ax.plot(
        epochs,
        train,
        "-o",
        color=TRAIN_COLOR,
        linewidth=CURVE_LINEWIDTH,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        alpha=CURVE_ALPHA,
        label="Train",
    )
    ax.plot(
        epochs,
        test,
        "-o",
        color=TEST_COLOR,
        linewidth=CURVE_LINEWIDTH,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        alpha=CURVE_ALPHA,
        label="Test",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    x_pad = max(0.35, 0.04 * float(epochs[-1] - epochs[0]))
    ax.set_xlim(float(epochs[0]) - x_pad, float(epochs[-1]) + x_pad)
    if epochs.size <= 13:
        ax.set_xticks(epochs)
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    if ylim is None:
        ax.set_ylim(*_padded_ylim(train, test))
    else:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    _style_curve_axis(ax)

    legend = ax.legend(
        loc="best",
        frameon=False,
        handlelength=1.5,
        borderpad=0.25,
        labelspacing=0.25,
    )
    for text in legend.get_texts():
        text.set_fontfamily(FONT_FAMILY[0])
        text.set_fontsize(CURVE_LEGEND_TEXT_SIZE)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def _load_curves(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing learning-curve data: {path}")
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("history", rows.get("points", rows.get("epochs", [])))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path} does not contain a non-empty history list")
        epochs = np.asarray([row["epoch"] for row in rows], dtype=float)
        curves = {
            "epochs": epochs,
            "ce_train": np.asarray([row["train_ce"] for row in rows], dtype=float),
            "ce_test": np.asarray([row["test_ce"] for row in rows], dtype=float),
            "qmass_train": np.asarray([row["train_qmass"] for row in rows], dtype=float),
            "qmass_test": np.asarray([row["test_qmass"] for row in rows], dtype=float),
        }
    elif path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        curves = {
            "epochs": np.asarray(z["epoch"], dtype=float),
            "ce_train": np.asarray(z["train_ce"], dtype=float),
            "ce_test": np.asarray(z["test_ce"], dtype=float),
            "qmass_train": np.asarray(z["train_qmass"], dtype=float),
            "qmass_test": np.asarray(z["test_qmass"], dtype=float),
        }
    else:
        raise ValueError(f"unsupported learning-curve data type: {path}")

    finite = np.ones_like(curves["epochs"], dtype=bool)
    for key in ("ce_train", "ce_test", "qmass_train", "qmass_test"):
        finite &= np.isfinite(curves[key])
    if not np.any(finite):
        raise ValueError(f"{path} has no finite train/test CE and qmass rows")
    return {key: np.asarray(value, dtype=float)[finite] for key, value in curves.items()}


def render_panels_d_e_curves(
    out_dir: Path,
    dpi: int,
    *,
    data: Path = CURVE_DEFAULT_DATA,
) -> list[Path]:
    """Panels d (QMass) and e (cross-entropy): train/test learning curves."""
    _set_curve_rc()
    curves = _load_curves(data.resolve())
    epochs = curves["epochs"]
    qmass_png = _plot_curve(
        epochs=epochs,
        train=curves["qmass_train"],
        test=curves["qmass_test"],
        ylabel="QMass",
        out_prefix=out_dir / "fig4_d_qmass",
        dpi=int(dpi),
    )
    ce_png = _plot_curve(
        epochs=epochs,
        train=curves["ce_train"],
        test=curves["ce_test"],
        ylabel="Cross Entropy Loss",
        out_prefix=out_dir / "fig4_e_cross_entropy",
        dpi=int(dpi),
    )
    return [qmass_png, ce_png]


# =========================================================================== #
# Panel f -- readout temperature sweep
# =========================================================================== #
TEMP_SWEEP_DATA = OUT_DIR / "fig4_lm_readout_temperature_sweep_release_final.json"
TEMP_FIG_SIZE = (2.06, 1.73)
TEMP_TEXT_SIZE = 6.0
TEMP_LEGEND_TEXT_SIZE = 5.0
T_LINEWIDTH = 0.65
SPINE_LINEWIDTH = 0.55
MAJOR_TICK_WIDTH = 0.45
MAJOR_TICK_LENGTH = 2.0
TICK_PAD = 1.5


def _set_temp_rc() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY[0],
            "font.sans-serif": FONT_FAMILY,
            "font.size": TEMP_TEXT_SIZE,
            "axes.labelsize": TEMP_TEXT_SIZE,
            "xtick.labelsize": TEMP_TEXT_SIZE,
            "ytick.labelsize": TEMP_TEXT_SIZE,
            "legend.fontsize": TEMP_LEGEND_TEXT_SIZE,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY[0],
            "mathtext.it": FONT_FAMILY[0],
            "mathtext.bf": FONT_FAMILY[0],
            "mathtext.sf": FONT_FAMILY[0],
            "mathtext.tt": FONT_FAMILY[0],
            "mathtext.cal": FONT_FAMILY[0],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_temp_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=MAJOR_TICK_WIDTH,
        length=MAJOR_TICK_LENGTH,
        pad=TICK_PAD,
        colors=AXIS_COLOR,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LINEWIDTH)
        spine.set_edgecolor(AXIS_COLOR)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY[0])
        label.set_fontsize(TEMP_TEXT_SIZE)


def _plot_temperature_sweep(payload: dict, out_prefix: Path, dpi: int) -> Path:
    points = sorted(payload["points"], key=lambda row: row["temperature"])
    x = np.asarray([row["temperature"] for row in points], dtype=float)
    validity = np.asarray([row["valid_pct"] for row in points], dtype=float)
    novelty = np.asarray([row["novelty_pct"] for row in points], dtype=float)
    uniqueness = np.asarray([row["uniqueness_pct"] for row in points], dtype=float)

    _set_temp_rc()
    fig, ax = plt.subplots(figsize=TEMP_FIG_SIZE)
    fig.subplots_adjust(left=0.195, right=0.970, bottom=0.245, top=0.965)

    ax.plot(x, validity, "-o", linewidth=CURVE_LINEWIDTH, markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH, alpha=CURVE_ALPHA, label="Validity", zorder=3)
    ax.plot(x, novelty, "-o", linewidth=CURVE_LINEWIDTH, markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH, alpha=CURVE_ALPHA, label="Novelty", zorder=3)
    ax.plot(x, uniqueness, "-o", linewidth=CURVE_LINEWIDTH, markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH, alpha=CURVE_ALPHA, label="Uniqueness", zorder=3)
    ax.axvline(float(payload["selected_temp"]), color="black", linewidth=T_LINEWIDTH,
               linestyle="-", alpha=CURVE_ALPHA, label="Selected T", zorder=1)
    ax.axvline(float(payload["training_temp"]), color="black", linewidth=T_LINEWIDTH,
               linestyle="--", alpha=CURVE_ALPHA, label="Training T", zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(max(1e-4, float(np.min(x))), float(np.max(x)) * 1.08)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Readout Temperature T")
    _style_temp_axis(ax)

    legend = ax.legend(loc="lower left", frameon=True, framealpha=0.78, facecolor="white",
                       edgecolor="none", handlelength=1.45, handletextpad=0.5, borderaxespad=0.45,
                       labelspacing=0.25, fontsize=TEMP_LEGEND_TEXT_SIZE)
    legend.get_frame().set_linewidth(0.0)
    for text in legend.get_texts():
        text.set_fontfamily(FONT_FAMILY[0])
        text.set_fontsize(TEMP_LEGEND_TEXT_SIZE)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def render_panel_f_temperature_sweep(
    out_dir: Path,
    dpi: int,
    *,
    data: Path = TEMP_SWEEP_DATA,
) -> list[Path]:
    """Panel f: validity / novelty / uniqueness vs readout temperature."""
    if not data.exists():
        raise SystemExit(f"sweep data not found: {data}")
    payload = _load_json(data)
    png = _plot_temperature_sweep(payload, out_dir / "fig4_f_temperature_sweep", int(dpi))
    return [png]


# =========================================================================== #
# Auxiliary -- gate-voltage distribution histogram (NOT in the composite)
# =========================================================================== #
GATE_HIST_GATE_FILE = _PR / "language_model" / "runs" / "clean" / "gates.npz"
GATE_HIST_FIG_SIZE = (2.12, 1.78)
GATE_HIST_TEXT_SIZE = 6.0
GATE_HIST_BAR_COLOR = "#6F879D"
GATE_HIST_MEAN_COLOR = "#111111"


def _set_gate_hist_rc() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY[0],
            "font.sans-serif": FONT_FAMILY,
            "font.size": GATE_HIST_TEXT_SIZE,
            "axes.labelsize": GATE_HIST_TEXT_SIZE,
            "xtick.labelsize": GATE_HIST_TEXT_SIZE,
            "ytick.labelsize": GATE_HIST_TEXT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_gate_hist_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=0.45,
        length=2.0,
        pad=1.5,
        colors=AXIS_COLOR,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor(AXIS_COLOR)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY[0])
        label.set_fontsize(GATE_HIST_TEXT_SIZE)


def _load_gate_values(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"missing gate file: {path}")
    z = np.load(path, allow_pickle=True)
    if isinstance(z, np.lib.npyio.NpzFile):
        if "vg_final" not in z.files:
            raise KeyError(f"{path} must contain vg_final")
        gates = np.asarray(z["vg_final"], dtype=float)
        metadata = {
            "source": _relsrc(path),
            "shape": list(gates.shape),
            "edges": int(z["edges"]) if "edges" in z.files else int(gates.size),
            "final_epoch": int(z["final_epoch"]) if "final_epoch" in z.files else None,
        }
    else:
        gates = np.asarray(z, dtype=float)
        metadata = {"source": _relsrc(path), "shape": list(gates.shape), "edges": int(gates.size), "final_epoch": None}
    return gates.ravel(), metadata


def _plot_gate_hist(values: np.ndarray, metadata: dict[str, object], out_prefix: Path, summary_prefix: Path, bins: int, dpi: int) -> tuple[Path, Path]:
    _set_gate_hist_rc()
    fig, ax = plt.subplots(figsize=GATE_HIST_FIG_SIZE)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.245, top=0.965)

    ax.hist(values, bins=int(bins), range=(0.4, 8.0), color=GATE_HIST_BAR_COLOR, alpha=0.82, edgecolor="none")
    mean_v = float(np.mean(values))
    ax.axvline(mean_v, color=GATE_HIST_MEAN_COLOR, linewidth=0.65, alpha=0.8)

    ax.set_xlim(0.4, 8.0)
    ax.set_xlabel("final gate voltage (V)")
    ax.set_ylabel("count")
    ax.set_xticks([0.4, 2.0, 4.0, 6.0, 8.0])
    _style_gate_hist_axis(ax)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    summary = summary_prefix.with_suffix(".json")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)

    payload = {
        **metadata,
        "n": int(values.size),
        "min": float(np.min(values)),
        "mean": mean_v,
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "bins": int(bins),
        "figure_size_in": list(GATE_HIST_FIG_SIZE),
    }
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return png, summary


def render_aux_gate_voltage_hist(
    out_dir: Path,
    data_dir: Path,
    dpi: int,
    *,
    gate_file: Path = GATE_HIST_GATE_FILE,
    bins: int = 48,
) -> list[Path]:
    """Auxiliary (not in composite): the final gate-voltage distribution histogram."""
    values, metadata = _load_gate_values(gate_file)
    png, summary = _plot_gate_hist(
        values,
        metadata,
        out_dir / "fig4_gate_voltage_hist",
        data_dir / "fig4_gate_voltage_hist",
        int(bins),
        int(dpi),
    )
    return [png, summary]


# =========================================================================== #
# Entry point
# =========================================================================== #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render every script-generated panel of Figure 4.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Where to write panel PNGs.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Where to write script-produced JSON/NPZ.")
    p.add_argument("--png-dpi", type=int, default=PNG_DPI)
    p.add_argument(
        "--skip-ngspice",
        action="store_true",
        help="Skip panels that call ngspice (network render + next-token histogram).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir
    data_dir = args.data_dir
    dpi = int(args.png_dpi)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []

    # Panel a (network diagram + colorbar).  Drawn purely from the shipped
    # endpoint gate voltages -- no circuit solve, so it always runs.
    produced += render_panel_a_network(out_dir, dpi)

    if not args.skip_ngspice:
        # Panel a (next-token probability bars).  The only panel that calls
        # ngspice (a single free .op solve via the inference model).
        produced += render_panel_a_next_token(out_dir, data_dir, dpi)

    # Panels d and e (learning curves).
    produced += render_panels_d_e_curves(out_dir, dpi)

    # Panel f (temperature sweep).
    produced += render_panel_f_temperature_sweep(out_dir, dpi)

    # Auxiliary gate-voltage histogram (not part of the manuscript composite).
    produced += render_aux_gate_voltage_hist(out_dir, data_dir, dpi)

    for path in produced:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
