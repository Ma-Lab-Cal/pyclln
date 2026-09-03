#!/usr/bin/env python3
"""Render Fig. 4-style panels for the ionosphere run."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
_PR = next(p for p in Path(__file__).resolve().parents if (p / "device_model").is_dir())  # robust paper_release root (self-contained)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors, ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


DEFAULT_RUN_DIR = _PR / "ionosphere" / "decision_plane_data"
DEFAULT_CURVE = _PR / "ionosphere" / "results" / "curve_clean.npz"
DEFAULT_GATES = _PR / "ionosphere" / "results" / "gates_clean.npz"
DEFAULT_TOPOLOGY = _PR / "ionosphere" / "topology_147.npz"
DEFAULT_DEVICE_LIB = _PR / "device_model" / "nmos_lvl1_ald1106.lib"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

PNG_DPI = 600
NETWORK_DPI = 1200

TEXT = "#34383D"
BLACK = "#000000"
BORDER = "#B8BDC2"
NEUTRAL = "#E7E9EA"
NETWORK_NEUTRAL = "#AEB8BF"
TRAIN_COLOR = "#274E87"
TEST_COLOR = "#D34F72"
BEST_STEP_COLOR = "#000000"
ACC_LOSS_CURVE_LINEWIDTH = 0.65
ACC_LOSS_CURVE_ALPHA = 0.7
BEST_STEP_LINEWIDTH = 0.5
GATE_LOW = "#1F5D8C"
GATE_HIGH = "#C9531F"
NODE_BORDER_COLOR = "#000000"
NODE_BORDER_WIDTH = 0.55
INPUT_NODE_BORDER_COLOR = NODE_BORDER_COLOR
INPUT_NODE_BORDER_WIDTH = NODE_BORDER_WIDTH
HIDDEN_NODE_BORDER_COLOR = NODE_BORDER_COLOR
HIDDEN_NODE_BORDER_WIDTH = NODE_BORDER_WIDTH
OUTPUT_NODE_BORDER_COLOR = NODE_BORDER_COLOR
OUTPUT_NODE_BORDER_WIDTH = NODE_BORDER_WIDTH
DIFF_LOW = "#8B57A4"
DIFF_HIGH = "#3B9B78"
PRUNED = "#C4C8CC"
DECISION_LOW = "#8B75B8"
DECISION_HIGH = "#7FA85F"
DECISION_REGION_ALPHA = 0.16
DECISION_POINT_ALPHA = 0.96
DECISION_NEUTRAL = "#F8F6EF"
DECISION_BOUNDARY = "#252A2F"
DECISION_POINT_EDGE = "#111111"
DECISION_PC_AXIS = "#111111"
DECISION_DOT_SIZE = 3.55
DECISION_LEGEND_FONT_SIZE = 5.0

FONT_FAMILY = ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"]
AXIS_LABEL_SIZE = 6.0
TICK_LABEL_SIZE = 6.0
PANEL_SPINE_LINEWIDTH = 0.55

ACC_LOSS_SIZE = (3.24, 2.55)
ACC_LOSS_LEFT = 0.135
ACC_LOSS_RIGHT = 0.985
ACC_LOSS_BOTTOM = 0.125
ACC_LOSS_TOP = 0.992
ACC_LOSS_HSPACE = 0.075
ACC_LOSS_AXIS_WIDTH_IN = ACC_LOSS_SIZE[0] * (ACC_LOSS_RIGHT - ACC_LOSS_LEFT)
ACC_LOSS_AXIS_HEIGHT_IN = ACC_LOSS_SIZE[1] * (ACC_LOSS_TOP - ACC_LOSS_BOTTOM) / (2.0 + ACC_LOSS_HSPACE)
ACC_LOSS_XMIN = 2.0
NETWORK_SIZE = (1.61, 1.41)
NETWORK_XLIM = (-0.84, 1.775)
NETWORK_YLIM = (-1.15, 1.15)
NETWORK_EDGE_WIDTH = 0.45
NETWORK_EDGE_ALPHA = 0.82
HIST_SIZE = (1.48, 0.53)
HIST_LEFT = 0.18
HIST_RIGHT = 0.985
HIST_BOTTOM = 0.38
HIST_TOP = 0.93
HIST_BOX_WIDTH_IN = round(HIST_SIZE[0] * (HIST_RIGHT - HIST_LEFT) * PNG_DPI) / PNG_DPI
CONFUSION_SIZE = (1.390001, 1.390001)
COLORBAR_SIZE = (0.30, 1.30)
COLORBAR_BODY_HEIGHT_IN = 1.20
COLORBAR_THICKNESS_IN = 0.041
DIFF_FIELDS_SIZE = (3.00, 1.36)
DIFF_PANEL_SIZE = (1.98, 1.73)
SAMPLE_STRIP_SIZE = (3.00, 0.78)
BASELINE_LEFT = ACC_LOSS_LEFT
BASELINE_RIGHT = ACC_LOSS_RIGHT
BASELINE_BOTTOM = 0.31
BASELINE_TOP = 0.93
BASELINE_YMIN = 60.0
BASELINE_YMAX = 102.0
BASELINE_TICK_LABEL_SIZE = 6.0
BASELINE_SIZE = (3.24, 1.32)
DECISION_GRID_N = 480
DECISION_KNN_NEIGHBORS = 2
DECISION_SMOOTH_SIGMA = 2.0
DECISION_MARGIN_FRAC = 0.045
DECISION_PC_AXIS_FRAC = 0.115
DECISION_PC_AXIS_ORIGIN_FRAC = (0.095, 0.105)
DECISION_LEGEND_ANCHOR = (0.018, 0.018)

GATE_VMIN = 0.40
GATE_VCENTER = 2.00
GATE_VMAX = 8.00
GATE_COLOR_GAMMA = 0.52
HIST_XMIN = 0.00
HIST_XMAX = 4.00
HIST_BINS = 30
HIST_INITIAL_COUNT_TOP = 160.0
HIST_FINAL_COUNT_TOP = 80.0
HIST_INITIAL_Y_TICKS = (75.0, 150.0)
HIST_FINAL_Y_TICKS = (40.0, 80.0)
HIST_LABEL_SIZE = 6.0
HIST_TICK_LABEL_SIZE = 6.0
FEATURE_VMAX = 0.077
RS_FREE = 1e9
CLASS_NAMES = np.asarray(["Negative", "Positive"], dtype=object)
OUTPUT_NODE_LABELS = ["N", "P"]
BASELINE_REFERENCE_ACCURACY = [
    ("XGBoost", 89.8, 82.9, 95.5),
    ("Support\nVector", 88.6, 81.8, 94.4),
    ("Random\nForest", 90.9, 85.2, 96.7),
    ("Neural\nNetwork", 87.5, 79.5, 94.3),
    ("Logistic\nRegression", 79.5, 70.4, 87.5),
]
BASELINE_REFERENCE_PRECISION = [
    ("XGBoost", 90.6, 84.0, 95.9),
    ("Support\nVector", 90.6, 84.6, 95.8),
    ("Random\nForest", 91.0, 84.4, 96.7),
    ("Neural\nNetwork", 89.8, 83.6, 95.3),
    ("Logistic\nRegression", 83.3, 75.0, 90.6),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ionosphere panels in the Fig. 4 scikit style")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--curve", type=Path, default=DEFAULT_CURVE)
    p.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    p.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    p.add_argument("--device-lib", type=Path, default=DEFAULT_DEVICE_LIB)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--state-epoch",
        type=int,
        default=None,
        help="Saved epoch state to use when exact gates are unavailable. Defaults to the final (endpoint) step.",
    )
    p.add_argument(
        "--use-cache-inference",
        action="store_true",
        help="Use cached per-epoch test outputs instead of exact ngspice inference from --gates.",
    )
    return p.parse_args()


def _set_common_rc(font_size: float = 5.0) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": max(font_size - 0.8, 3.5),
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out_prefix: Path, *, dpi: int = PNG_DPI) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    # Keep the canvas controlled only by Matplotlib figsize/subplots_adjust.
    # Post-save cropping changes exported dimensions and apparent line widths.
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def _png_size(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _remove_deprecated_outputs(out_dir: Path) -> None:
    for name in (
        "fig2_ionosphere_confusion_initial.png",
        "fig2_ionosphere_confusion_final.png",
        "fig2_ionosphere_confusion_colorbar.png",
        "fig2_ionosphere_confusion_matrices.npz",
        "fig2_ionosphere_differential_gate_fields.png",
        "fig2_ionosphere_differential_gate_fields.npz",
        "fig2_ionosphere_misclassified_digits.png",
        "fig2_ionosphere_misclassified_digits_scores.npz",
        "fig2_ionosphere_vg_hist_step0_finebins.png",
        "fig2_ionosphere_decision_boundary_initial_k2.png",
        "fig2_ionosphere_decision_boundary_final_k2.png",
        "fig2_ionosphere_decision_boundary_initial_k3.png",
        "fig2_ionosphere_decision_boundary_final_k3.png",
        "fig2_ionosphere_decision_boundaries_knn_variants.npz",
        # Old descriptive PNG names, superseded by panel-letter names (fig2_a ... fig2_h).
        "fig2_ionosphere_clean_acc_loss.png",
        "fig2_ionosphere_step_acc_loss_sample.png",
        "fig2_ionosphere_network_step0.png",
        "fig2_ionosphere_network_final_step.png",
        "fig2_ionosphere_vg_hist_step0.png",
        "fig2_ionosphere_vg_hist_final_step.png",
        "fig2_ionosphere_decision_boundary_initial.png",
        "fig2_ionosphere_decision_boundary_final.png",
        "fig2_ionosphere_network_differential_panel.png",
        "fig2_ionosphere_baseline_comparison.png",
        # Data files that formerly lived at the figure root, now relocated under data/.
        "fig2_ionosphere_exact_clean_outputs.npz",
        "fig2_ionosphere_decision_boundaries.npz",
        "fig2_ionosphere_network_differential_panel_scores.npz",
        "fig2_ionosphere_baseline_comparison_data.npz",
        "fig2_ionosphere_clean_final_step_gates.npz",
        "fig2_ionosphere_clean_step_metrics.npz",
        "fig2_ionosphere_network_state_panels_meta.npz",
    ):
        (out_dir / name).unlink(missing_ok=True)


def _load_meta(run_dir: Path) -> dict:
    with (run_dir / "run_meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_topology(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    drain = np.asarray(data["edges_D"], dtype=int).reshape(-1)
    source = np.asarray(data["edges_S"], dtype=int).reshape(-1)
    input_nodes = np.asarray(data["input_nodes"], dtype=int).reshape(-1)
    out_nodes = np.asarray(data["out_nodes"], dtype=int).reshape(-1)
    all_nodes = set(drain.tolist()) | set(source.tolist())
    hidden_nodes = np.asarray(
        sorted(n for n in all_nodes if n not in set(input_nodes.tolist()) and n not in set(out_nodes.tolist()) and n >= 300),
        dtype=int,
    )
    return {
        "drain": drain,
        "source": source,
        "input_nodes": input_nodes,
        "hidden_nodes": hidden_nodes,
        "out_nodes": out_nodes,
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_ionosphere_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 35:
                continue
            x_rows.append([float(v) for v in parts[:34]])
            y_rows.append(1 if parts[34].strip() == "g" else 0)
    if not x_rows:
        raise ValueError(f"No ionosphere rows loaded from {path}")
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int)


def _metadata_input_scale(ionosphere_dir: Path, run_dir: Path) -> float:
    metas = [
        _load_json(ionosphere_dir / "runs" / "clean" / "run_meta.json"),
        _load_json(ionosphere_dir / "results" / "results.json"),
        _load_json(run_dir / "run_meta.json"),
    ]
    for meta in metas:
        hp = meta.get("hyperparameters", {})
        if "input_scale_V" in hp:
            return float(hp["input_scale_V"])
        if "input_scale" in hp:
            return float(hp["input_scale"])
        recipe = meta.get("training", {}).get("recipe", {})
        if "input_scale" in recipe:
            return float(recipe["input_scale"])
        if "input_scale" in meta:
            return float(meta["input_scale"])
    return 1.0


def _metadata_seed(ionosphere_dir: Path, run_dir: Path) -> int:
    metas = [
        _load_json(ionosphere_dir / "runs" / "clean" / "run_meta.json"),
        _load_json(ionosphere_dir / "results" / "results.json"),
        _load_json(run_dir / "run_meta.json"),
    ]
    for meta in metas:
        for key in ("seed", "random_seed"):
            if key in meta:
                return int(meta[key])
        run = meta.get("run", {})
        if isinstance(run, dict):
            for key in ("seed", "random_seed"):
                if key in run:
                    return int(run[key])
        recipe = meta.get("training", {}).get("recipe", {})
        if "seed" in recipe:
            return int(recipe["seed"])
    return 0


def _load_eval_dataset(run_dir: Path, topology_path: Path) -> dict[str, np.ndarray | float | int | str]:
    ionosphere_dir = topology_path.resolve().parent
    data_path = ionosphere_dir / "ionosphere.data"
    if data_path.exists():
        input_scale = _metadata_input_scale(ionosphere_dir, run_dir)
        seed = _metadata_seed(ionosphere_dir, run_dir)
        x_all, y_all = _load_ionosphere_data(data_path)
        _, x_test, _, y_test = train_test_split(
            x_all,
            y_all,
            test_size=0.2,
            random_state=seed,
            stratify=y_all,
        )
        return {
            "test_x": x_test * float(input_scale),
            "test_y": y_test,
            "input_scale": float(input_scale),
            "seed": int(seed),
            "source": f"{data_path} seed={seed} input_scale={float(input_scale):g}",
        }

    x = np.load(run_dir / "test_x.npy").astype(float)
    y = np.load(run_dir / "test_y.npy").astype(int)
    return {
        "test_x": x,
        "test_y": y,
        "input_scale": np.nan,
        "seed": -1,
        "source": f"{run_dir}/test_x.npy",
    }


def _load_step_curve(curve_path: Path, run_dir: Path) -> dict[str, np.ndarray | int | float | str]:
    source_path = curve_path if curve_path.exists() else run_dir / "0_updcurve.npz"
    z = np.load(source_path, allow_pickle=True)
    required = {"train_acc", "test_acc", "train_loss", "test_loss", "epoch", "sample"}
    missing = sorted(required.difference(z.files))
    if missing:
        raise KeyError(f"{source_path} is missing keys: {missing}")

    test_acc = np.asarray(z["test_acc"], dtype=float)
    finite = np.isfinite(test_acc)
    if not np.any(finite):
        raise ValueError("No finite test accuracy values in 0_updcurve.npz")
    final_idx = int(test_acc.size - 1)
    return {
        "step": np.arange(1, test_acc.size + 1, dtype=float),
        "train_acc": np.asarray(z["train_acc"], dtype=float),
        "test_acc": test_acc,
        "train_loss": np.asarray(z["train_loss"], dtype=float),
        "test_loss": np.asarray(z["test_loss"], dtype=float),
        "epoch": np.asarray(z["epoch"], dtype=int),
        "sample": np.asarray(z["sample"], dtype=int),
        "final_idx": final_idx,
        "final_step": final_idx + 1,
        "final_epoch": int(np.asarray(z["epoch"], dtype=int)[final_idx]),
        "final_sample": int(np.asarray(z["sample"], dtype=int)[final_idx]),
        "final_test_acc": float(test_acc[final_idx]),
        "source_path": str(source_path),
    }


def _available_epochs(run_dir: Path, stem: str) -> list[int]:
    epochs: list[int] = []
    pattern = re.compile(rf"0_{re.escape(stem)}_epoch(\d+)\.npy$")
    for path in run_dir.glob(f"0_{stem}_epoch*.npy"):
        m = pattern.match(path.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def _nearest_epoch(run_dir: Path, stem: str, requested: int) -> int:
    epochs = _available_epochs(run_dir, stem)
    if not epochs:
        raise FileNotFoundError(f"No saved epoch files matching 0_{stem}_epoch*.npy in {run_dir}")
    if requested in epochs:
        return requested
    prior = [e for e in epochs if e <= requested]
    return prior[-1] if prior else epochs[0]


def _load_state(
    run_dir: Path,
    topo: dict[str, np.ndarray],
    curve: dict[str, np.ndarray | int | float | str],
    state_epoch: int | None,
    gates_path: Path,
) -> dict:
    meta = _load_meta(run_dir) if (run_dir / "run_meta.json").exists() else {}
    init_vg = float(meta.get("vg_init", {}).get("fixed", GATE_VCENTER))
    requested_epoch = int(curve["final_epoch"]) if state_epoch is None else int(state_epoch)

    if gates_path.exists():
        z = np.load(gates_path, allow_pickle=True)
        if "vg_final" in z.files:
            vg_final = np.asarray(z["vg_final"], dtype=float)
        elif "vg_best" in z.files:
            vg_final = np.asarray(z["vg_best"], dtype=float)
        else:
            raise KeyError(f"{gates_path} is missing vg_final/vg_best")

        if "vg_init" in z.files:
            vg0 = np.asarray(z["vg_init"], dtype=float)
        else:
            vg0 = np.full(topo["drain"].size, init_vg, dtype=float)

        if vg0.size != topo["drain"].size or vg_final.size != topo["drain"].size:
            raise ValueError(f"{gates_path} gate vector size does not match topology edge count")

        final_epoch = int(np.asarray(z["final_epoch"]).item()) if "final_epoch" in z.files else requested_epoch
        if "final_step_index" in z.files:
            final_step_index = int(np.asarray(z["final_step_index"]).item())
            final_sample = int(np.asarray(z["final_sample"]).item()) if "final_sample" in z.files else int(curve["final_sample"])
        else:
            final_step_index = int(curve["final_idx"])
            final_sample = int(curve["final_sample"])
        clean_test_acc = float(np.asarray(z["clean_test_acc"]).item()) if "clean_test_acc" in z.files else float(curve["final_test_acc"])
        return {
            "vg0": vg0,
            "vg_final": vg_final,
            "state_epoch": final_epoch,
            "requested_state_epoch": requested_epoch,
            "state_sample": final_sample,
            "state_step_index": final_step_index,
            "state_test_acc": clean_test_acc,
            "meta": meta,
            "source_path": str(gates_path),
            "source_kind": "exact_gates",
        }

    epoch = _nearest_epoch(run_dir, "vg_unique", requested_epoch)
    vg0 = np.full(topo["drain"].size, init_vg, dtype=float)
    vg_final = np.load(run_dir / f"0_vg_unique_epoch{epoch}.npy").astype(float)
    epoch_idx = np.flatnonzero(np.asarray(curve["epoch"], dtype=int) == epoch)
    final_idx = int(epoch_idx[-1]) if epoch_idx.size else int(curve["final_idx"])
    return {
        "vg0": vg0,
        "vg_final": vg_final,
        "state_epoch": epoch,
        "requested_state_epoch": requested_epoch,
        "state_sample": int(np.asarray(curve["sample"], dtype=int)[final_idx]),
        "state_step_index": final_idx,
        "state_test_acc": float(np.asarray(curve["test_acc"], dtype=float)[final_idx]),
        "meta": meta,
        "source_path": str(run_dir / f"0_vg_unique_epoch{epoch}.npy"),
        "source_kind": "epoch_cache",
    }


def _set_gate_scale(vg0: np.ndarray, vg_best: np.ndarray) -> None:
    global GATE_VCENTER
    init_values = np.asarray(vg0, dtype=float)
    finite = init_values[np.isfinite(init_values)]
    if finite.size:
        GATE_VCENTER = float(np.median(finite))
    if not GATE_VMIN < GATE_VCENTER < GATE_VMAX:
        raise ValueError("Gate color scale must satisfy GATE_VMIN < GATE_VCENTER < GATE_VMAX")


def _sci_tick(value: float, _pos: int) -> str:
    if abs(value) < 1e-15:
        return "0"
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", which="major", direction="out", width=0.45, length=2.0, pad=1.5, colors=TEXT, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", which="minor", direction="out", width=0.35, length=1.2, pad=1.5, colors=TEXT, labelsize=TICK_LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PANEL_SPINE_LINEWIDTH)
        spine.set_edgecolor(TEXT)


def _style_baseline_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=0.45,
        length=2.0,
        pad=1.5,
        colors=BLACK,
        labelcolor=BLACK,
    )
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(BLACK)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PANEL_SPINE_LINEWIDTH)
        spine.set_edgecolor(BLACK)
        spine.set_zorder(10)


def _draw_acc_loss(curve: dict[str, np.ndarray | int | float], out_prefix: Path) -> Path:
    _set_common_rc(6.0)
    step = np.asarray(curve["step"], dtype=float)
    train_acc = np.asarray(curve["train_acc"], dtype=float)
    test_acc = np.asarray(curve["test_acc"], dtype=float)
    train_loss = np.clip(np.asarray(curve["train_loss"], dtype=float), np.finfo(float).tiny, None)
    test_loss = np.clip(np.asarray(curve["test_loss"], dtype=float), np.finfo(float).tiny, None)
    visible = step >= ACC_LOSS_XMIN
    if not np.any(visible):
        visible = np.ones_like(step, dtype=bool)

    fig, (ax_acc, ax_loss) = plt.subplots(
        2,
        1,
        figsize=ACC_LOSS_SIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": ACC_LOSS_HSPACE},
    )
    fig.subplots_adjust(left=ACC_LOSS_LEFT, right=ACC_LOSS_RIGHT, bottom=ACC_LOSS_BOTTOM, top=ACC_LOSS_TOP)

    for ax, a_train, a_test, ylabel in (
        (ax_acc, train_acc, test_acc, "Accuracy"),
        (ax_loss, train_loss, test_loss, "Hinge Loss"),
    ):
        ax.plot(step[visible], a_train[visible], color=TRAIN_COLOR, linewidth=ACC_LOSS_CURVE_LINEWIDTH, alpha=ACC_LOSS_CURVE_ALPHA, label="Train")
        ax.plot(step[visible], a_test[visible], color=TEST_COLOR, linewidth=ACC_LOSS_CURVE_LINEWIDTH, alpha=ACC_LOSS_CURVE_ALPHA, label="Test")
        ax.set_ylabel(ylabel, labelpad=1.0)
        ax.set_xscale("log")
        ax.set_xlim(ACC_LOSS_XMIN, float(step[-1]) * 1.045)
        _style_axis(ax)

    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_acc.legend(
        loc="upper left",
        bbox_to_anchor=(0.018, 0.982),
        frameon=False,
        handlelength=1.6,
        borderpad=0.25,
        borderaxespad=0.0,
        labelspacing=0.25,
    )
    loss_max = float(np.nanmax([np.nanmax(train_loss[visible]), np.nanmax(test_loss[visible])]))
    ax_loss.set_ylim(0.0, loss_max * 1.08)
    ax_loss.yaxis.set_major_formatter(ticker.FuncFormatter(_sci_tick))
    ax_loss.set_xlabel("Training Step")
    ax_loss.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    ax_loss.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax_loss.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax_acc.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    fig.align_ylabels([ax_acc, ax_loss])
    return _save(fig, out_prefix)


def _gate_cmap(neutral: str = NEUTRAL) -> colors.Colormap:
    return colors.LinearSegmentedColormap.from_list("gate_voltage", [GATE_LOW, neutral, GATE_HIGH], N=256)


def _gate_norm() -> colors.TwoSlopeNorm:
    return colors.TwoSlopeNorm(vmin=GATE_VMIN, vcenter=GATE_VCENTER, vmax=GATE_VMAX)


def _gate_rgba(vg: np.ndarray, alpha: float = 1.0, neutral: str = NEUTRAL) -> np.ndarray:
    raw = np.asarray(vg, dtype=float)
    values = np.clip(raw.reshape(-1), GATE_VMIN, GATE_VMAX)
    neutral_rgb = np.asarray(colors.to_rgb(neutral), dtype=float)
    low_rgb = np.asarray(colors.to_rgb(GATE_LOW), dtype=float)
    high_rgb = np.asarray(colors.to_rgb(GATE_HIGH), dtype=float)
    rgb = np.tile(neutral_rgb, (values.size, 1))

    below = values < GATE_VCENTER
    above = values > GATE_VCENTER
    if np.any(below):
        frac = ((GATE_VCENTER - values[below]) / (GATE_VCENTER - GATE_VMIN)) ** GATE_COLOR_GAMMA
        rgb[below] = neutral_rgb + frac[:, None] * (low_rgb - neutral_rgb)
    if np.any(above):
        frac = ((values[above] - GATE_VCENTER) / (GATE_VMAX - GATE_VCENTER)) ** GATE_COLOR_GAMMA
        rgb[above] = neutral_rgb + frac[:, None] * (high_rgb - neutral_rgb)

    rgba = np.column_stack([rgb, np.full(values.size, alpha, dtype=float)])
    return rgba.reshape(raw.shape + (4,))


def _hist_color_values(vg: np.ndarray, edges: np.ndarray, centers: np.ndarray, counts: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=float).copy()
    gates = np.asarray(vg, dtype=float)
    for i, count in enumerate(counts):
        if count <= 0:
            continue
        if i == len(counts) - 1:
            in_bin = (gates >= edges[i]) & (gates <= edges[i + 1])
        else:
            in_bin = (gates >= edges[i]) & (gates < edges[i + 1])
        if np.any(in_bin):
            values[i] = float(np.mean(gates[in_bin]))
    return values


def _feature_cmap() -> colors.Colormap:
    return colors.LinearSegmentedColormap.from_list("feature_signed", [GATE_LOW, NEUTRAL, GATE_HIGH], N=256)


def _feature_norm() -> colors.TwoSlopeNorm:
    return colors.TwoSlopeNorm(vmin=-FEATURE_VMAX, vcenter=0.0, vmax=FEATURE_VMAX)


def _node_positions(topo: dict[str, np.ndarray]) -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    input_nodes = topo["input_nodes"]
    hidden_nodes = topo["hidden_nodes"]
    out_nodes = topo["out_nodes"]

    y_inputs = np.linspace(0.98, -0.98, 17)
    for i, node in enumerate(input_nodes):
        col = i // 17
        row = i % 17
        pos[int(node)] = (-0.74 + 0.20 * col, float(y_inputs[row]))

    y_hidden = np.linspace(0.98, -0.98, max(len(hidden_nodes), 1))
    for y, node in zip(y_hidden, hidden_nodes):
        pos[int(node)] = (0.38, float(y))

    y_out = np.linspace(0.34, -0.34, max(len(out_nodes), 1))
    for y, node in zip(y_out, out_nodes):
        pos[int(node)] = (1.46, float(y))
    return pos


def _draw_network(
    topo: dict[str, np.ndarray],
    vg: np.ndarray,
    out_prefix: Path,
    *,
    size: tuple[float, float],
    feature_values: np.ndarray | None = None,
    vout: np.ndarray | None = None,
    edge_width: float = NETWORK_EDGE_WIDTH,
    edge_alpha: float = NETWORK_EDGE_ALPHA,
    edge_neutral: str = NEUTRAL,
    save_dpi: int = PNG_DPI,
) -> Path:
    _set_common_rc(4.6)
    drain = topo["drain"]
    source = topo["source"]
    pos = _node_positions(topo)

    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(*NETWORK_XLIM)
    ax.set_ylim(*NETWORK_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    segments = []
    edge_vg = []
    for d_node, s_node, gate in zip(drain, source, vg):
        d = int(d_node)
        s = int(s_node)
        if d in pos and s in pos:
            segments.append([pos[d], pos[s]])
            edge_vg.append(float(gate))
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=_gate_rgba(np.asarray(edge_vg, dtype=float), alpha=edge_alpha, neutral=edge_neutral),
                linewidths=edge_width,
                zorder=1,
                capstyle="round",
                joinstyle="round",
            )
        )

    input_xy = np.asarray([pos[int(n)] for n in topo["input_nodes"]], dtype=float)
    if feature_values is None:
        ax.scatter(
            input_xy[:, 0],
            input_xy[:, 1],
            marker="s",
            s=7.5,
            facecolors="#FFFFFF",
            edgecolors=INPUT_NODE_BORDER_COLOR,
            linewidths=INPUT_NODE_BORDER_WIDTH,
            zorder=4,
        )
    else:
        ax.scatter(
            input_xy[:, 0],
            input_xy[:, 1],
            marker="s",
            s=9.2,
            c=np.asarray(feature_values, dtype=float),
            cmap=_feature_cmap(),
            norm=_feature_norm(),
            edgecolors=INPUT_NODE_BORDER_COLOR,
            linewidths=INPUT_NODE_BORDER_WIDTH,
            zorder=4,
        )

    hidden_xy = np.asarray([pos[int(n)] for n in topo["hidden_nodes"]], dtype=float)
    if hidden_xy.size:
        ax.scatter(
            hidden_xy[:, 0],
            hidden_xy[:, 1],
            marker="h",
            s=30,
            facecolors="#FFFFFF",
            edgecolors=HIDDEN_NODE_BORDER_COLOR,
            linewidths=HIDDEN_NODE_BORDER_WIDTH,
            zorder=5,
        )

    out_xy = np.asarray([pos[int(n)] for n in topo["out_nodes"]], dtype=float)
    for idx, (x, y) in enumerate(out_xy):
        ax.scatter([x], [y], s=31, facecolors="#FFFFFF", edgecolors=OUTPUT_NODE_BORDER_COLOR, linewidths=OUTPUT_NODE_BORDER_WIDTH, zorder=6)
        ax.text(
            x,
            y,
            OUTPUT_NODE_LABELS[idx] if idx < len(OUTPUT_NODE_LABELS) else str(idx),
            ha="center",
            va="center",
            fontsize=4.0,
            color=TEXT,
            fontfamily="Open Sans",
            zorder=7,
        )

    if vout is not None and len(vout) == len(out_xy):
        values = np.asarray(vout, dtype=float)
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = max(hi - lo, 1e-9)
        for idx, (x, y) in enumerate(out_xy):
            frac = (float(values[idx]) - lo) / span
            length = 0.06 + 0.22 * frac
            ax.plot([x + 0.11, x + 0.11 + length], [y - 0.17, y - 0.17], color="#111111", linewidth=0.95, solid_capstyle="butt", zorder=8)
            ax.plot([x + 0.11, x + 0.11], [y - 0.19, y - 0.15], color="#333333", linewidth=0.30, zorder=8)

    return _save(fig, out_prefix, dpi=save_dpi)


def _nice_top(value: float) -> float:
    value = max(float(value), 1.0)
    step = 10.0 ** np.floor(np.log10(value))
    for scale in (1.0, 2.0, 5.0, 10.0):
        top = scale * step
        if top >= value:
            return top
    return 10.0 * step


def _hist_xticks() -> np.ndarray:
    span = float(HIST_XMAX - HIST_XMIN)
    if span <= 1.0:
        step = 0.1
    elif span <= 3.0:
        step = 0.5
    else:
        step = 1.0
    ticks = [float(HIST_XMIN)]
    lo = np.ceil((HIST_XMIN - 1e-12) / step) * step
    hi = np.floor((HIST_XMAX + 1e-12) / step) * step
    ticks.extend(np.arange(lo, hi + 0.5 * step, step).tolist())
    ticks.append(float(HIST_XMAX))
    return np.asarray(sorted(set(np.round(ticks, 1))), dtype=float)


def _draw_histogram(
    vg: np.ndarray,
    out_prefix: Path,
    *,
    count_top: float,
    y_ticks: tuple[float, float] | None = None,
    bins_count: int = HIST_BINS,
    neutral_init_bin: bool = False,
) -> Path:
    _set_common_rc(HIST_LABEL_SIZE)
    bins = np.linspace(HIST_XMIN, HIST_XMAX, bins_count)
    counts, edges = np.histogram(vg, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    color_values = _hist_color_values(vg, edges, centers, counts)

    fig, ax = plt.subplots(figsize=HIST_SIZE)
    fig.subplots_adjust(left=HIST_LEFT, right=HIST_RIGHT, bottom=HIST_BOTTOM, top=HIST_TOP)
    bar_colors = _gate_rgba(color_values, alpha=0.96)
    if neutral_init_bin:
        init_bin = (edges[:-1] <= GATE_VCENTER) & (GATE_VCENTER < edges[1:])
        if np.any(init_bin):
            bar_colors[init_bin] = _gate_rgba(np.asarray([GATE_VCENTER]), alpha=0.96)[0]
    ax.bar(centers, counts, width=widths, color=bar_colors, edgecolor=TEXT, linewidth=0.25, align="center")
    ax.set_xlim(HIST_XMIN, HIST_XMAX)
    ax.set_ylim(0.0, count_top)
    ax.set_xticks(_hist_xticks())
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    tick_labels = ax.get_xticklabels()
    if tick_labels:
        tick_labels[0].set_ha("left")
        tick_labels[-1].set_ha("right")
    ax.set_yticks(list(y_ticks) if y_ticks is not None else [count_top / 2.0, count_top])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.set_xlabel("$V_G$ (V)", labelpad=0.2, fontsize=HIST_LABEL_SIZE)
    ax.set_ylabel("Count", labelpad=0.2, fontsize=HIST_LABEL_SIZE)
    ax.tick_params(axis="both", which="major", direction="out", width=0.35, length=1.25, pad=0.6, colors=TEXT, labelsize=HIST_TICK_LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PANEL_SPINE_LINEWIDTH)
        spine.set_edgecolor(TEXT)
    return _save(fig, out_prefix)


def _normalize_rows(cm: np.ndarray) -> np.ndarray:
    denom = np.sum(cm, axis=1, keepdims=True)
    return cm / np.where(denom > 0, denom, 1.0)


def _draw_confusion(cm: np.ndarray, out_prefix: Path, vmax: float = 1.0) -> Path:
    _set_common_rc(4.5)
    fig, ax = plt.subplots(figsize=CONFUSION_SIZE)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.imshow(_normalize_rows(cm), cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest", origin="upper")
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(width=0.35, length=1.2, pad=0.6, labelsize=TICK_LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(PANEL_SPINE_LINEWIDTH)
        spine.set_edgecolor(TEXT)
    return _save(fig, out_prefix)


def _draw_confusion_colorbar(out_prefix: Path, vmax: float = 1.0) -> Path:
    _set_common_rc(4.5)
    fig = plt.figure(figsize=COLORBAR_SIZE)
    cax = fig.add_axes(
        [
            0.18,
            0.5 * (1.0 - COLORBAR_BODY_HEIGHT_IN / COLORBAR_SIZE[1]),
            COLORBAR_THICKNESS_IN / COLORBAR_SIZE[0],
            COLORBAR_BODY_HEIGHT_IN / COLORBAR_SIZE[1],
        ]
    )
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.0, vmax=vmax), cmap="magma")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_ticks([])
    cb.ax.tick_params(width=0.0, length=0.0, pad=0.0, colors=TEXT)
    cb.outline.set_linewidth(PANEL_SPINE_LINEWIDTH)
    cb.outline.set_edgecolor(TEXT)
    return _save(fig, out_prefix)


def _feature_grid(values: np.ndarray, fill: float = np.nan) -> np.ndarray:
    out = np.full(34, fill, dtype=float)
    flat = np.asarray(values, dtype=float).reshape(-1)
    out[: min(out.size, flat.size)] = flat[: out.size]
    return out.reshape(2, 17)


def _topology_maps(topo: dict[str, np.ndarray], vg: np.ndarray) -> dict[str, np.ndarray]:
    input_idx = {int(n): i for i, n in enumerate(topo["input_nodes"])}
    hidden_idx = {int(n): i for i, n in enumerate(topo["hidden_nodes"])}
    out_idx = {int(n): i for i, n in enumerate(topo["out_nodes"])}
    direct = np.full((len(out_idx), len(input_idx)), np.nan, dtype=float)
    input_hidden = np.full((len(hidden_idx), len(input_idx)), np.nan, dtype=float)
    hidden_out = np.full((len(out_idx), len(hidden_idx)), np.nan, dtype=float)

    for d_node, s_node, gate in zip(topo["drain"], topo["source"], vg):
        d = int(d_node)
        s = int(s_node)
        if d in input_idx and s in hidden_idx:
            input_hidden[hidden_idx[s], input_idx[d]] = float(gate)
        elif d in input_idx and s in out_idx:
            direct[out_idx[s], input_idx[d]] = float(gate)
        elif d in hidden_idx and s in out_idx:
            hidden_out[out_idx[s], hidden_idx[d]] = float(gate)
    return {"direct": direct, "input_hidden": input_hidden, "hidden_out": hidden_out}


def _draw_gate_fields(topo: dict[str, np.ndarray], vg: np.ndarray, out_prefix: Path) -> Path:
    _set_common_rc(5.0)
    maps = _topology_maps(topo, vg)
    direct = maps["direct"]
    input_hidden = maps["input_hidden"]
    hidden_out = maps["hidden_out"]

    panels: list[np.ndarray] = []
    labels: list[str] = []
    for h in range(input_hidden.shape[0]):
        panels.append(_feature_grid(input_hidden[h] - GATE_VCENTER))
        labels.append(f"input_hidden_{h}")
    panels.append(_feature_grid(direct[0] - GATE_VCENTER))
    labels.append("direct_b")
    panels.append(_feature_grid(direct[1] - GATE_VCENTER))
    labels.append("direct_g")
    panels.append(_feature_grid(direct[1] - direct[0]))
    labels.append("direct_g_minus_b")
    valid_hidden = np.isfinite(input_hidden)
    hidden_counts = np.sum(valid_hidden, axis=0)
    hidden_mean = np.full(input_hidden.shape[1], np.nan, dtype=float)
    with np.errstate(invalid="ignore"):
        hidden_sum = np.nansum(input_hidden, axis=0)
    has_hidden = hidden_counts > 0
    hidden_mean[has_hidden] = hidden_sum[has_hidden] / hidden_counts[has_hidden]
    panels.append(_feature_grid(hidden_mean - GATE_VCENTER))
    labels.append("input_hidden_mean")
    hidden_delta = np.full(34, np.nan, dtype=float)
    if hidden_out.shape[0] >= 2:
        hidden_delta[: hidden_out.shape[1]] = hidden_out[1] - hidden_out[0]
    panels.append(_feature_grid(hidden_delta))
    labels.append("hidden_g_minus_b")
    panels = panels[:10]
    labels = labels[:10]

    cmap = colors.LinearSegmentedColormap.from_list("ionosphere_delta", [DIFF_LOW, NEUTRAL, DIFF_HIGH], N=256)
    cmap.set_bad(PRUNED)
    finite = np.asarray([v for panel in panels for v in panel.ravel() if np.isfinite(v)], dtype=float)
    if finite.size:
        diff_vmin = float(np.min(finite))
        diff_vmax = float(np.max(finite))
    else:
        diff_vmin, diff_vmax = -1.0, 1.0
    diff_vmin = min(diff_vmin, 0.0)
    diff_vmax = max(diff_vmax, 0.0)
    if not diff_vmin < 0.0 < diff_vmax:
        eps = max(abs(diff_vmin), abs(diff_vmax), 1.0) * 1e-6
        diff_vmin -= eps
        diff_vmax += eps
    norm = colors.TwoSlopeNorm(vmin=diff_vmin, vcenter=0.0, vmax=diff_vmax)

    fig = plt.figure(figsize=DIFF_FIELDS_SIZE)
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=[1, 1, 1, 1, 1, 0.075],
        hspace=0.16,
        wspace=0.08,
        left=0.005,
        right=0.992,
        top=0.982,
        bottom=0.018,
    )
    image = None
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(grid[i // 5, i % 5])
        image = ax.imshow(panel, cmap=cmap, norm=norm, interpolation="nearest", origin="upper", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(PANEL_SPINE_LINEWIDTH)
            spine.set_edgecolor(TEXT)

    if image is None:
        raise RuntimeError("No gate-field panels were drawn")
    cax = fig.add_subplot(grid[:, 5])
    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks([])
    cb.ax.tick_params(width=0.6, length=0.0)
    png = _save(fig, out_prefix)
    np.savez(
        out_prefix.with_suffix(".npz"),
        direct=direct,
        input_hidden=input_hidden,
        hidden_out=hidden_out,
        panel_labels=np.asarray(labels),
        vlim=max(abs(diff_vmin), abs(diff_vmax)),
        diff_vmin=diff_vmin,
        diff_vcenter=0.0,
        diff_vmax=diff_vmax,
        gate_center=GATE_VCENTER,
    )
    return png


def _exec_chunked(ng, cmds: list[str], max_len: int = 900) -> None:
    buf = ""
    for cmd in cmds:
        if len(buf) + len(cmd) + 2 > max_len:
            ng.exec_command(buf)
            buf = ""
        buf = cmd if not buf else buf + "; " + cmd
    if buf:
        ng.exec_command(buf)


def _clean_eval_netlist(topology_path: Path, vg: np.ndarray, device_lib: Path) -> tuple[str, np.ndarray]:
    topo = np.load(topology_path, allow_pickle=True)
    input_nodes = np.asarray(topo["input_nodes"], dtype=int).reshape(-1)
    out_nodes = np.asarray(topo["out_nodes"], dtype=int).reshape(-1)
    edges_d = np.asarray(topo["edges_D"], dtype=int).reshape(-1)
    edges_s = np.asarray(topo["edges_S"], dtype=int).reshape(-1)
    if vg.size != edges_d.size:
        raise ValueError(f"Gate vector size {vg.size} does not match topology edge count {edges_d.size}")

    negref = int(np.asarray(topo["negref"]).item())
    posref = int(np.asarray(topo["posref"]).item())
    node_pool = [negref, posref] + input_nodes.tolist() + out_nodes.tolist() + edges_d.tolist() + edges_s.tolist()
    sink0 = max(node_pool) + 1

    lines = [".title exact_ionosphere_clean_eval", f'.include "{device_lib}"', ".options klu"]
    # The topology is input-driven; these rails match the original trainer and are unconnected.
    lines.append(f"VMINUS {negref} 0 0.0000000000000000")
    lines.append(f"VPLUS  {posref} 0 0.0780000000000000")
    for i, node in enumerate(input_nodes):
        lines.append(f"VIN{i} {int(node)} 0 0")
    for i, out_node in enumerate(out_nodes, start=1):
        lines.append(f"RS{i} {int(out_node)} {sink0 + (i - 1)} {RS_FREE:.6g}")
    for j in range(out_nodes.size):
        lines.append(f"VOUT{j} {sink0 + j} 0 0")
    for e, (drain, source) in enumerate(zip(edges_d.tolist(), edges_s.tolist())):
        lines.append(f"VG{e} g{e} {int(drain)} {float(vg[e]):.16f}")
        lines.append(f"X{e} {int(drain)} g{e} {int(source)} b{e} NMOSWRAP")
    lines.extend(
        [
            ".options TEMP = 27C",
            ".options TNOM = 27C",
            ".options itl1=40 itl2=40 itl4=6 itl5=60",
            ".options gmin=1e-8 reltol=5e-3 abstol=1e-8 vntol=1e-5",
            ".options rshunt=1e9",
            ".op",
            ".end",
        ]
    )
    return "\n".join(lines) + "\n", out_nodes


def _eval_clean_vout(topology_path: Path, device_lib: Path, vg: np.ndarray, x: np.ndarray) -> np.ndarray:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    netlist, out_nodes = _clean_eval_netlist(topology_path, vg, device_lib)
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)
    vout = np.full((x.shape[0], out_nodes.size), np.nan, dtype=float)
    try:
        for row_idx, row in enumerate(x):
            _exec_chunked(ng, [f"alter VIN{i} dc = {float(v):.16f}" for i, v in enumerate(row)])
            ng.run()
            plot = ng.plot(None, ng.last_plot).to_analysis()
            vout[row_idx, :] = np.asarray([float(plot[str(int(n))].as_ndarray()[-1]) for n in out_nodes], dtype=float)
            ng.exec_command("destroy all")
    finally:
        try:
            ng.remove_circuit()
        except Exception:
            pass
    return vout


def _eval_clean_vout_initial_final(
    topology_path: Path,
    device_lib: Path,
    vg0: np.ndarray,
    vg_best: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    netlist, out_nodes = _clean_eval_netlist(topology_path, vg0, device_lib)
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    def eval_current_state() -> np.ndarray:
        vout = np.full((x.shape[0], out_nodes.size), np.nan, dtype=float)
        for row_idx, row in enumerate(x):
            _exec_chunked(ng, [f"alter VIN{i} dc = {float(v):.16f}" for i, v in enumerate(row)])
            ng.run()
            plot = ng.plot(None, ng.last_plot).to_analysis()
            vout[row_idx, :] = np.asarray([float(plot[str(int(n))].as_ndarray()[-1]) for n in out_nodes], dtype=float)
            ng.exec_command("destroy all")
        return vout

    try:
        vout_init = eval_current_state()
        _exec_chunked(ng, [f"alter VG{i} dc = {float(v):.16f}" for i, v in enumerate(vg_best)])
        vout_final = eval_current_state()
    finally:
        try:
            ng.remove_circuit()
        except Exception:
            pass
    return vout_init, vout_final


def _confusion_from_vout(y: np.ndarray, vout: np.ndarray, n_classes: int) -> np.ndarray:
    pred = np.argmax(vout, axis=1).astype(int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for y_true, y_pred in zip(y, pred):
        if 0 <= int(y_true) < n_classes and 0 <= int(y_pred) < n_classes:
            cm[int(y_true), int(y_pred)] += 1
    return cm


def _load_eval_outputs(
    run_dir: Path,
    topology_path: Path,
    device_lib: Path,
    data_dir: Path,
    vg0: np.ndarray,
    vg_best: np.ndarray,
    state_epoch: int,
    *,
    use_cache: bool,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    eval_data_source: str = "",
) -> dict[str, np.ndarray | str]:
    if x is None:
        x = np.load(run_dir / "test_x.npy").astype(float)
        eval_data_source = eval_data_source or f"{run_dir}/test_x.npy"
    else:
        x = np.asarray(x, dtype=float)
    if y is None:
        y = np.load(run_dir / "test_y.npy").astype(int)
    else:
        y = np.asarray(y, dtype=int)

    source = "exact_ngspice"
    if use_cache:
        vout_init = np.load(run_dir / "0_vout_test_epoch0.npy").astype(float)
        vout_final = np.load(run_dir / f"0_vout_test_epoch{state_epoch}.npy").astype(float)
        source = "epoch_cache"
    else:
        try:
            vout_init, vout_final = _eval_clean_vout_initial_final(topology_path, device_lib, vg0, vg_best, x)
        except Exception as exc:
            print(f"exact_ngspice_failed={exc!r}; falling back to cached epoch outputs")
            vout_init = np.load(run_dir / "0_vout_test_epoch0.npy").astype(float)
            vout_final = np.load(run_dir / f"0_vout_test_epoch{state_epoch}.npy").astype(float)
            source = "epoch_cache_after_exact_failure"

    pred_init = np.argmax(vout_init, axis=1).astype(int)
    pred_final = np.argmax(vout_final, axis=1).astype(int)
    np.savez(
        data_dir / "fig2_ionosphere_exact_clean_outputs.npz",
        test_x=x,
        test_y=y,
        vout_init=vout_init,
        vout_final=vout_final,
        pred_init=pred_init,
        pred_final=pred_final,
        class_names=CLASS_NAMES,
        output_node_labels=np.asarray(OUTPUT_NODE_LABELS, dtype=object),
        source=source,
        topology=str(topology_path),
        device_lib=str(device_lib),
        eval_data_source=eval_data_source,
        state_epoch=state_epoch,
        initial_acc=float(np.mean(pred_init == y)),
        final_acc=float(np.mean(pred_final == y)),
    )
    return {
        "test_x": x,
        "test_y": y,
        "vout_init": vout_init,
        "vout_final": vout_final,
        "pred_init": pred_init,
        "pred_final": pred_final,
        "source": source,
    }


def _decision_cmap() -> colors.Colormap:
    return colors.ListedColormap(
        [
            colors.to_rgba(DECISION_LOW, DECISION_REGION_ALPHA),
            colors.to_rgba(DECISION_HIGH, DECISION_REGION_ALPHA),
        ],
        name="ionosphere_decision",
    )


def _rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def _best_square_rotation(scores: np.ndarray) -> tuple[float, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    best_theta = 0.0
    best_side = np.inf
    for theta in np.linspace(0.0, np.pi, 18001):
        rotated = scores @ _rotation_matrix(float(theta)).T
        side = float(np.max(np.ptp(rotated, axis=0)))
        if side < best_side:
            best_side = side
            best_theta = float(theta)
    return best_theta, _rotation_matrix(best_theta)


def _make_decision_plane(x: np.ndarray, grid_n: int = DECISION_GRID_N) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    pca = PCA(n_components=2)
    scores_raw = pca.fit_transform(x)
    rotation_angle, rotation = _best_square_rotation(scores_raw)
    scores = scores_raw @ rotation.T
    lo = np.min(scores, axis=0)
    hi = np.max(scores, axis=0)
    span = hi - lo
    side = float(np.max(span))
    if side <= 1e-12:
        side = 1.0
    margin_frac = min(max(float(DECISION_MARGIN_FRAC), 0.0), 0.45)
    side = side / max(1.0 - 2.0 * margin_frac, 1e-6)
    center = 0.5 * (lo + hi)
    lo = center - 0.5 * side
    hi = center + 0.5 * side

    gx = np.linspace(lo[0], hi[0], grid_n)
    gy = np.linspace(lo[1], hi[1], grid_n)
    xx, yy = np.meshgrid(gx, gy)
    grid_scores = np.column_stack([xx.ravel(), yy.ravel()])
    return {
        "pca_mean": pca.mean_,
        "pca_components": pca.components_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_scores_raw": scores_raw,
        "rotation_angle_rad": np.asarray(rotation_angle, dtype=float),
        "rotation_matrix": rotation,
        "margin_frac": np.asarray(margin_frac, dtype=float),
        "scores": scores,
        "grid_scores": grid_scores,
        "xx": xx,
        "yy": yy,
        "extent": np.asarray([lo[0], hi[0], lo[1], hi[1]], dtype=float),
    }


def _draw_decision_boundary(
    plane: dict[str, np.ndarray],
    y: np.ndarray,
    pred_labels: np.ndarray,
    out_prefix: Path,
    n_neighbors: int = DECISION_KNN_NEIGHBORS,
    show_pc_axes: bool = False,
    show_legend: bool = False,
) -> tuple[Path, np.ndarray]:
    _set_common_rc(4.5)
    scores = np.asarray(plane["scores"], dtype=float)
    xx = np.asarray(plane["xx"], dtype=float)
    yy = np.asarray(plane["yy"], dtype=float)
    extent = np.asarray(plane["extent"], dtype=float)
    pred_labels = np.asarray(pred_labels, dtype=int)
    unique_pred = np.unique(pred_labels)
    cmap = _decision_cmap()
    if unique_pred.size == 1:
        region = np.full_like(xx, fill_value=float(unique_pred[0]), dtype=float)
        field = region
    else:
        knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(pred_labels)), weights="distance")
        knn.fit(scores, pred_labels)
        classes = knn.classes_.astype(int)
        proba = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        if 1 in classes:
            field = proba[:, int(np.flatnonzero(classes == 1)[0])].reshape(xx.shape)
        else:
            field = np.zeros_like(xx, dtype=float)
        field = gaussian_filter(field, sigma=DECISION_SMOOTH_SIGMA, mode="nearest")
        region = (field >= 0.5).astype(float)

    fig, ax = plt.subplots(figsize=CONFUSION_SIZE)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    ax.contourf(xx, yy, field, levels=[-0.001, 0.5, 1.001], cmap=cmap, zorder=0)
    if unique_pred.size > 1:
        ax.contour(xx, yy, field, levels=[0.5], colors=[DECISION_BOUNDARY], linewidths=0.26, alpha=0.95, zorder=1)

    y = np.asarray(y, dtype=int)
    for cls, face in ((0, DECISION_LOW), (1, DECISION_HIGH)):
        idx = np.flatnonzero(y == cls)
        if idx.size:
            ax.scatter(
                scores[idx, 0],
                scores[idx, 1],
                s=DECISION_DOT_SIZE,
                marker="o",
                facecolors=face,
                edgecolors=DECISION_POINT_EDGE,
                linewidths=0.20,
                alpha=DECISION_POINT_ALPHA,
                zorder=4,
            )

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=DECISION_LOW,
                markeredgecolor=DECISION_POINT_EDGE,
                markeredgewidth=0.30,
                markersize=3.0,
                alpha=DECISION_POINT_ALPHA,
                label=str(CLASS_NAMES[0]),
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=DECISION_HIGH,
                markeredgecolor=DECISION_POINT_EDGE,
                markeredgewidth=0.30,
                markersize=3.0,
                alpha=DECISION_POINT_ALPHA,
                label=str(CLASS_NAMES[1]),
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=DECISION_LEGEND_ANCHOR,
            frameon=False,
            prop={"family": "Open Sans", "size": DECISION_LEGEND_FONT_SIZE},
            handlelength=0.7,
            handletextpad=0.25,
            borderaxespad=0.0,
            labelspacing=0.12,
        )

    if show_pc_axes:
        side = float(extent[1] - extent[0])
        origin = np.asarray(
            [
                extent[0] + DECISION_PC_AXIS_ORIGIN_FRAC[0] * side,
                extent[2] + DECISION_PC_AXIS_ORIGIN_FRAC[1] * side,
            ],
            dtype=float,
        )
        arrow_len = DECISION_PC_AXIS_FRAC * side
        pc1 = np.asarray([1.0, 0.0], dtype=float)
        pc2 = np.asarray([0.0, 1.0], dtype=float)
        for direction in (pc1, pc2):
            end = origin + arrow_len * direction
            ax.annotate(
                "",
                xy=end,
                xytext=origin,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": DECISION_PC_AXIS,
                    "linewidth": 0.42,
                    "mutation_scale": 4.0,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                zorder=7,
            )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(width=0.35, length=1.2, pad=0.6, labelsize=TICK_LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(PANEL_SPINE_LINEWIDTH)
        spine.set_edgecolor(TEXT)
    return _save(fig, out_prefix), region


def _select_representative_sample(x: np.ndarray, y: np.ndarray, vout: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    pred = np.argmax(vout, axis=1).astype(int)
    margins = np.abs(vout[:, 1] - vout[:, 0])
    candidates = np.flatnonzero((y == 1) & (pred == y))
    if candidates.size == 0:
        candidates = np.flatnonzero(pred == y)
    if candidates.size == 0:
        candidates = np.arange(len(y))
    ordered = candidates[np.argsort(margins[candidates])]
    idx = int(ordered[len(ordered) // 2])
    return x[idx], int(y[idx]), idx, vout[idx], pred


def _draw_differential_panel(
    topo: dict[str, np.ndarray],
    vg: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vout_all: np.ndarray,
    state_epoch: int,
    source: str,
    out_prefix: Path,
    data_dir: Path,
) -> Path:
    sample_x, y_true, sample_idx, vout, pred_all = _select_representative_sample(x, y, vout_all)
    png = _draw_network(
        topo,
        vg,
        out_prefix,
        size=DIFF_PANEL_SIZE,
        feature_values=None,
        vout=vout,
        edge_width=0.66,
        edge_alpha=0.58,
    )
    scores_path = (data_dir / (out_prefix.name + "_scores")).with_suffix(".npz")
    np.savez(
        scores_path,
        sample_x=sample_x,
        sample_index=sample_idx,
        y_true=y_true,
        y_pred=int(np.argmax(vout)),
        vout=vout,
        pred_all=pred_all,
        state_epoch=state_epoch,
        source=source,
    )
    return png


def _draw_misclassified_strip(
    x: np.ndarray,
    y: np.ndarray,
    vout: np.ndarray,
    state_epoch: int,
    source: str,
    out_prefix: Path,
) -> Path:
    _set_common_rc(4.0)
    pred = np.argmax(vout, axis=1).astype(int)
    miss = np.flatnonzero(pred != y)
    margins = np.abs(vout[:, 1] - vout[:, 0])
    if miss.size >= 5:
        selected = miss[:5]
    else:
        correct = np.flatnonzero(pred == y)
        hard_correct = correct[np.argsort(margins[correct])]
        selected = np.concatenate([miss, hard_correct[: 5 - miss.size]])

    n_cols = 5
    fig, axes = plt.subplots(1, n_cols, figsize=SAMPLE_STRIP_SIZE)
    axes = np.asarray(axes).reshape(-1)
    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.010, top=0.84, wspace=0.03, hspace=0.0)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    for ax, idx in zip(axes, selected):
        image = _feature_grid(x[int(idx)])
        ax.imshow(image, cmap=_feature_cmap(), norm=_feature_norm(), interpolation="nearest", aspect="auto")
        ax.set_title(f"{int(y[int(idx)])}->{int(pred[int(idx)])}", color=TEXT, pad=0.2, fontsize=4.0)
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(PANEL_SPINE_LINEWIDTH)
            spine.set_edgecolor(TEXT)

    png = _save(fig, out_prefix)
    np.savez(
        out_prefix.with_name(out_prefix.name + "_scores").with_suffix(".npz"),
        test_indices=selected,
        y_true=y[selected],
        y_pred=pred[selected],
        vout=vout[selected],
        is_misclassified=pred[selected] != y[selected],
        margin_abs=margins[selected],
        state_epoch=state_epoch,
        source=source,
        n_misclassified=int(miss.size),
    )
    return png


def _draw_baseline_comparison(final_acc: float, final_precision: float, out_prefix: Path, data_dir: Path) -> Path:
    clln_acc_pct = 100.0 * float(final_acc)
    clln_precision_pct = 100.0 * float(final_precision)
    acc_rows = BASELINE_REFERENCE_ACCURACY + [("Transistor\nCLLN", clln_acc_pct, clln_acc_pct, clln_acc_pct)]
    precision_rows = BASELINE_REFERENCE_PRECISION + [("Transistor\nCLLN", clln_precision_pct, clln_precision_pct, clln_precision_pct)]
    labels = [row[0] for row in acc_rows]
    acc_center = np.asarray([row[1] for row in acc_rows], dtype=float)
    acc_low = np.asarray([row[2] for row in acc_rows], dtype=float)
    acc_high = np.asarray([row[3] for row in acc_rows], dtype=float)
    precision_center = np.asarray([row[1] for row in precision_rows], dtype=float)
    precision_low = np.asarray([row[2] for row in precision_rows], dtype=float)
    precision_high = np.asarray([row[3] for row in precision_rows], dtype=float)
    acc_best = acc_high.copy()
    precision_best = precision_high.copy()
    x = np.arange(len(acc_rows), dtype=float)
    bar_width = 0.24
    acc_x = x - 0.5 * bar_width
    precision_x = x + 0.5 * bar_width
    bar_colors = ["#858C94"] * (len(acc_rows) - 1) + [TRAIN_COLOR]

    _set_common_rc(6.0)
    fig, ax = plt.subplots(figsize=BASELINE_SIZE)
    fig.subplots_adjust(left=BASELINE_LEFT, right=BASELINE_RIGHT, bottom=BASELINE_BOTTOM, top=BASELINE_TOP)
    ax.bar(
        acc_x,
        acc_best - BASELINE_YMIN,
        bottom=BASELINE_YMIN,
        width=bar_width,
        color=bar_colors,
        edgecolor=BLACK,
        linewidth=0.35,
        zorder=2,
    )
    ax.bar(
        precision_x,
        precision_best - BASELINE_YMIN,
        bottom=BASELINE_YMIN,
        width=bar_width,
        color="white",
        edgecolor=BLACK,
        linewidth=0.35,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(BASELINE_YMIN, BASELINE_YMAX)
    ax.set_ylabel("Task Score (%)", labelpad=1.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.legend(
        handles=[
            Patch(facecolor=BLACK, edgecolor=BLACK, linewidth=0.35, label="Accuracy"),
            Patch(facecolor="white", edgecolor=BLACK, linewidth=0.35, label="Precision"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        frameon=False,
        ncol=2,
        fontsize=5.0,
        handlelength=0.85,
        handleheight=0.55,
        handletextpad=0.35,
        columnspacing=0.65,
        borderpad=0.0,
        labelspacing=0.20,
    )
    _style_baseline_axis(ax)

    png = _save(fig, out_prefix)
    np.savez(
        (data_dir / (out_prefix.name + "_data")).with_suffix(".npz"),
        labels=np.asarray(labels, dtype=object),
        center_accuracy_pct=acc_center,
        low_accuracy_pct=acc_low,
        high_accuracy_pct=acc_high,
        best_accuracy_pct=acc_best,
        center_precision_pct=precision_center,
        low_precision_pct=precision_low,
        high_precision_pct=precision_high,
        best_precision_pct=precision_best,
        clln_acc=float(clln_acc_pct / 100.0),
        clln_precision=float(clln_precision_pct / 100.0),
        source="UCI_baseline",
    )
    return png


def _write_dimensions(out_dir: Path) -> Path:
    rows = [
        ("fig2_a.png", DIFF_PANEL_SIZE),
        ("fig2_a_baseline.png", BASELINE_SIZE),
        ("fig2_b.png", NETWORK_SIZE),
        ("fig2_c.png", NETWORK_SIZE),
        ("fig2_d.png", HIST_SIZE),
        ("fig2_e.png", HIST_SIZE),
        ("fig2_f.png", ACC_LOSS_SIZE),
        ("fig2_f_sample.png", ACC_LOSS_SIZE),
        ("fig2_g.png", CONFUSION_SIZE),
        ("fig2_h.png", CONFUSION_SIZE),
    ]
    path = out_dir / "FIGURE_DIMENSIONS.md"
    lines = [
        "# Figure Dimensions",
        "",
        "Dimensions are reported in target placement units. Most panels are rendered at 600 dpi; standalone network panels are rendered at 1200 dpi.",
        "",
        "| File | Size (in) | Pixels |",
        "|---|---:|---:|",
    ]
    for name, (w, h) in rows:
        pixel_size = _png_size(out_dir / name)
        if pixel_size is not None:
            px_w, px_h = pixel_size
        else:
            px_w, px_h = int(round(w * PNG_DPI)), int(round(h * PNG_DPI))
        lines.append(f"| `{name}` | {w:.2f} x {h:.2f} | {px_w} x {px_h} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    curve_path = args.curve.resolve()
    gates_path = args.gates.resolve()
    topo_path = args.topology.resolve()
    device_lib = args.device_lib.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _remove_deprecated_outputs(out_dir)

    curve = _load_step_curve(curve_path, run_dir)
    topo = _load_topology(topo_path)
    state = _load_state(run_dir, topo, curve, args.state_epoch, gates_path)
    state_epoch = int(state["state_epoch"])
    vg0 = np.asarray(state["vg0"], dtype=float)
    vg_final = np.asarray(state["vg_final"], dtype=float)
    _set_gate_scale(vg0, vg_final)
    eval_dataset = _load_eval_dataset(run_dir, topo_path)
    eval_x = np.asarray(eval_dataset["test_x"], dtype=float)
    eval_y = np.asarray(eval_dataset["test_y"], dtype=int)
    eval_data_source = str(eval_dataset["source"])
    decision_plane = _make_decision_plane(eval_x)
    eval_outputs = _load_eval_outputs(
        run_dir,
        topo_path,
        device_lib,
        data_dir,
        vg0,
        vg_final,
        state_epoch,
        use_cache=bool(args.use_cache_inference),
        x=eval_x,
        y=eval_y,
        eval_data_source=eval_data_source,
    )
    eval_source = str(eval_outputs["source"])
    test_x = np.asarray(eval_outputs["test_x"], dtype=float)
    test_y = np.asarray(eval_outputs["test_y"], dtype=int)
    vout_init = np.asarray(eval_outputs["vout_init"], dtype=float)
    vout_final = np.asarray(eval_outputs["vout_final"], dtype=float)
    pred_init = np.asarray(eval_outputs["pred_init"], dtype=int)
    pred_final = np.asarray(eval_outputs["pred_final"], dtype=int)
    initial_acc = float(np.mean(np.argmax(vout_init, axis=1) == test_y))
    final_acc = float(np.mean(np.argmax(vout_final, axis=1) == test_y))
    final_tp = int(np.sum((pred_final == 1) & (test_y == 1)))
    final_fp = int(np.sum((pred_final == 1) & (test_y != 1)))
    final_precision = float(final_tp / max(final_tp + final_fp, 1))

    outputs: list[Path] = []
    outputs.append(_draw_acc_loss(curve, out_dir / "fig2_f"))
    outputs.append(_draw_acc_loss(curve, out_dir / "fig2_f_sample"))
    outputs.append(
        _draw_network(
            topo,
            vg0,
            out_dir / "fig2_b",
            size=NETWORK_SIZE,
            edge_neutral=NETWORK_NEUTRAL,
            save_dpi=NETWORK_DPI,
        )
    )
    outputs.append(
        _draw_network(
            topo,
            vg_final,
            out_dir / "fig2_c",
            size=NETWORK_SIZE,
            edge_neutral=NETWORK_NEUTRAL,
            save_dpi=NETWORK_DPI,
        )
    )

    outputs.append(_draw_histogram(vg0, out_dir / "fig2_d", count_top=HIST_INITIAL_COUNT_TOP, y_ticks=HIST_INITIAL_Y_TICKS))
    outputs.append(_draw_histogram(vg_final, out_dir / "fig2_e", count_top=HIST_FINAL_COUNT_TOP, y_ticks=HIST_FINAL_Y_TICKS, neutral_init_bin=True))

    decision_initial, region_init = _draw_decision_boundary(
        decision_plane,
        test_y,
        pred_init,
        out_dir / "fig2_g",
    )
    decision_final, region_final = _draw_decision_boundary(
        decision_plane,
        test_y,
        pred_final,
        out_dir / "fig2_h",
    )
    outputs.extend([decision_initial, decision_final])

    decision_scores = np.asarray(decision_plane["scores"], dtype=float)
    def _point_regions(pred: np.ndarray, n_neighbors: int = DECISION_KNN_NEIGHBORS) -> np.ndarray:
        pred = np.asarray(pred, dtype=int)
        unique = np.unique(pred)
        if unique.size == 1:
            return np.full(pred.shape, int(unique[0]), dtype=int)
        return (
            KNeighborsClassifier(n_neighbors=min(n_neighbors, len(pred)), weights="distance")
            .fit(decision_scores, pred)
            .predict(decision_scores)
            .astype(int)
        )

    def _nearest_grid_regions(region: np.ndarray) -> np.ndarray:
        xs = np.asarray(decision_plane["xx"], dtype=float)[0, :]
        ys = np.asarray(decision_plane["yy"], dtype=float)[:, 0]
        nearest = np.empty(decision_scores.shape[0], dtype=int)
        for i, (x, y) in enumerate(decision_scores):
            ix = int(np.argmin(np.abs(xs - x)))
            iy = int(np.argmin(np.abs(ys - y)))
            nearest[i] = int(region[iy, ix])
        return nearest

    point_region_init = _point_regions(pred_init)
    point_region_final = _point_regions(pred_final)
    nearest_grid_region_init = _nearest_grid_regions(region_init)
    nearest_grid_region_final = _nearest_grid_regions(region_final)
    model_wrong_init = np.flatnonzero(pred_init != test_y)
    model_wrong_final = np.flatnonzero(pred_final != test_y)
    wrong_zone_init = np.flatnonzero(point_region_init != test_y)
    wrong_zone_final = np.flatnonzero(point_region_final != test_y)
    np.savez(
        data_dir / "fig2_ionosphere_decision_boundaries.npz",
        pca_mean=np.asarray(decision_plane["pca_mean"], dtype=float),
        pca_components=np.asarray(decision_plane["pca_components"], dtype=float),
        pca_explained_variance_ratio=np.asarray(decision_plane["pca_explained_variance_ratio"], dtype=float),
        pca_scores_raw=np.asarray(decision_plane["pca_scores_raw"], dtype=float),
        pca_scores=np.asarray(decision_plane["scores"], dtype=float),
        rotation_angle_rad=np.asarray(decision_plane["rotation_angle_rad"], dtype=float),
        rotation_angle_deg=np.asarray(np.degrees(float(decision_plane["rotation_angle_rad"])), dtype=float),
        rotation_matrix=np.asarray(decision_plane["rotation_matrix"], dtype=float),
        margin_frac=np.asarray(decision_plane["margin_frac"], dtype=float),
        grid_scores=np.asarray(decision_plane["grid_scores"], dtype=float),
        grid_shape=np.asarray(decision_plane["xx"].shape, dtype=int),
        extent=np.asarray(decision_plane["extent"], dtype=float),
        region_init=region_init,
        region_final=region_final,
        pred_init=pred_init,
        pred_final=pred_final,
        point_region_init=point_region_init,
        point_region_final=point_region_final,
        nearest_grid_region_init=nearest_grid_region_init,
        nearest_grid_region_final=nearest_grid_region_final,
        point_region_matches_pred_init=bool(np.array_equal(point_region_init, pred_init)),
        point_region_matches_pred_final=bool(np.array_equal(point_region_final, pred_final)),
        nearest_grid_region_matches_pred_init=bool(np.array_equal(nearest_grid_region_init, pred_init)),
        nearest_grid_region_matches_pred_final=bool(np.array_equal(nearest_grid_region_final, pred_final)),
        model_wrong_indices_init=model_wrong_init,
        model_wrong_indices_final=model_wrong_final,
        wrong_zone_indices_init=wrong_zone_init,
        wrong_zone_indices_final=wrong_zone_final,
        test_y=test_y,
        class_names=CLASS_NAMES,
        output_node_labels=np.asarray(OUTPUT_NODE_LABELS, dtype=object),
        eval_source=eval_source,
        eval_data_source=eval_data_source,
        procedure="pca_2nn_distance_proba_pred_labels_smooth",
        knn_neighbors=min(DECISION_KNN_NEIGHBORS, len(pred_init)),
        knn_weights="distance",
        smooth_sigma=float(DECISION_SMOOTH_SIGMA),
    )

    outputs.append(
        _draw_differential_panel(
            topo,
            vg_final,
            test_x,
            test_y,
            vout_final,
            state_epoch,
            eval_source,
            out_dir / "fig2_a",
            data_dir,
        )
    )
    outputs.append(_draw_baseline_comparison(final_acc, final_precision, out_dir / "fig2_a_baseline", data_dir))

    final_gates_path = data_dir / "fig2_ionosphere_clean_final_step_gates.npz"
    np.savez(
        final_gates_path,
        vg0=vg0,
        vg_final=vg_final,
        drain=topo["drain"],
        source=topo["source"],
        input_nodes=topo["input_nodes"],
        hidden_nodes=topo["hidden_nodes"],
        out_nodes=topo["out_nodes"],
        class_names=CLASS_NAMES,
        output_node_labels=np.asarray(OUTPUT_NODE_LABELS, dtype=object),
        state_source=str(state["source_path"]),
        state_source_kind=str(state["source_kind"]),
        state_step_index=int(state["state_step_index"]),
        state_sample=int(state["state_sample"]),
        state_test_acc=float(state["state_test_acc"]),
        state_epoch=state_epoch,
        final_idx=int(curve["final_idx"]),
        final_step=int(curve["final_step"]),
        final_epoch=int(curve["final_epoch"]),
        final_sample=int(curve["final_sample"]),
        final_test_acc=float(curve["final_test_acc"]),
        requested_state_epoch=int(state["requested_state_epoch"]),
        topology=str(topo_path),
        run_dir=str(run_dir),
        curve=str(curve["source_path"]),
        gates=str(final_gates_path),
    )
    np.savez(
        data_dir / "fig2_ionosphere_clean_step_metrics.npz",
        step=np.asarray(curve["step"], dtype=float),
        train_acc=np.asarray(curve["train_acc"], dtype=float),
        test_acc=np.asarray(curve["test_acc"], dtype=float),
        train_loss=np.asarray(curve["train_loss"], dtype=float),
        test_loss=np.asarray(curve["test_loss"], dtype=float),
        epoch=np.asarray(curve["epoch"], dtype=int),
        sample=np.asarray(curve["sample"], dtype=int),
        final_idx=int(curve["final_idx"]),
        final_step=int(curve["final_step"]),
        final_epoch=int(curve["final_epoch"]),
        final_sample=int(curve["final_sample"]),
        final_test_acc=float(curve["final_test_acc"]),
    )
    meta_path = data_dir / "fig2_ionosphere_network_state_panels_meta.npz"
    np.savez(
        meta_path,
        gates=str(final_gates_path),
        gate_state_source=str(state["source_path"]),
        gate_state_source_kind=str(state["source_kind"]),
        eval_source=eval_source,
        eval_data_source=eval_data_source,
        eval_input_scale=float(eval_dataset["input_scale"]),
        eval_split_seed=int(eval_dataset["seed"]),
        class_names=CLASS_NAMES,
        output_node_labels=np.asarray(OUTPUT_NODE_LABELS, dtype=object),
        state_epoch=state_epoch,
        state_step_index=int(state["state_step_index"]),
        state_sample=int(state["state_sample"]),
        state_test_acc=float(state["state_test_acc"]),
        final_step=int(curve["final_step"]),
        final_epoch=int(curve["final_epoch"]),
        final_sample=int(curve["final_sample"]),
        final_test_acc=float(curve["final_test_acc"]),
        vg0_min=float(np.min(vg0)),
        vg0_max=float(np.max(vg0)),
        vg_final_min=float(np.min(vg_final)),
        vg_final_max=float(np.max(vg_final)),
        gate_norm=np.asarray([GATE_VMIN, GATE_VCENTER, GATE_VMAX], dtype=float),
        hist_range=np.asarray([HIST_XMIN, HIST_XMAX], dtype=float),
        hist_count_top_initial=HIST_INITIAL_COUNT_TOP,
        hist_count_top_final=HIST_FINAL_COUNT_TOP,
        hist_ticks_initial=np.asarray(HIST_INITIAL_Y_TICKS, dtype=float),
        hist_ticks_final=np.asarray(HIST_FINAL_Y_TICKS, dtype=float),
        hist_xticks=_hist_xticks(),
        hist_bins=int(HIST_BINS - 1),
    )
    dimensions_path = _write_dimensions(out_dir)

    print(f"run_dir={run_dir}")
    print(f"curve={curve['source_path']}")
    print(f"gates={state['source_path']}")
    print(f"topology={topo_path}")
    print(f"device_lib={device_lib}")
    print(f"eval_data={eval_data_source}")
    print(f"final_step={int(curve['final_step'])}")
    print(f"final_epoch={int(curve['final_epoch'])}")
    print(f"final_sample={int(curve['final_sample'])}")
    print(f"final_test_acc={float(curve['final_test_acc']):.6f}")
    print(f"state_epoch={state_epoch}")
    print(f"state_sample={int(state['state_sample'])}")
    print(f"state_step_index={int(state['state_step_index'])}")
    print(f"state_test_acc={float(state['state_test_acc']):.6f}")
    print(f"eval_source={eval_source}")
    print(f"initial_acc={initial_acc:.6f}")
    print(f"final_acc={final_acc:.6f}")
    for path in outputs:
        print(path)
    print(final_gates_path)
    print(meta_path)
    print(dimensions_path)


if __name__ == "__main__":
    main()
