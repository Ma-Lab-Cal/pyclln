#!/usr/bin/env python3
"""Render Supplemental Figure 2 as separate XOR/nonlinear-regression panels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
_PR = next(p for p in Path(__file__).resolve().parents if (p / "device_model").is_dir())  # robust paper_release root (self-contained)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def _default_release_bundle() -> Path:
    if (HERE / "XOR").is_dir() and (HERE / "nonlin_reg").is_dir():
        return HERE
    repo_root = Path(__file__).resolve().parents[3]
    return Path(__file__).resolve().parent / "data"


RELEASE_BUNDLE = _default_release_bundle()

# Data and output.
DEFAULT_XOR_ROOT = RELEASE_BUNDLE / "XOR" / "pnas_4x4_5runs_10000ep"
DEFAULT_NONLIN_RUN = RELEASE_BUNDLE / "nonlin_reg" / "pnas_4x4_gamma0p4_truepts_25000ep"
DEFAULT_OUT_DIR = HERE
DEFAULT_PANEL_PREFIX = "suppl2"
DEFAULT_SNAPSHOTS = [0, 100, 1000, 10000, 25000]
DEFAULT_XOR_INCLUDE_RUNS = ["seed-3"]

# Physical output size. Width is fixed to the requested 3.2 in.
PANEL_WIDTH_IN = 3.2
PANEL_HEIGHT_IN = 2.52
PNG_DPI = 600

# Tight, exact-size layout, matching the main Fig. 3 accuracy/loss curve style.
SUBPLOT_LEFT = 0.155
SUBPLOT_RIGHT = 0.992
SUBPLOT_BOTTOM = 0.155
SUBPLOT_TOP = 0.978

# Colors and lines.
BLACK = "#000000"
AXIS_COLOR = "#34383D"
CURVE_LINEWIDTH = 0.8
CURVE_ALPHA = 1.0
TRAIN_POINT_SIZE = 10
TRAIN_POINT_LINEWIDTH = 0.55

# Fonts and tick styling.
FONT_FAMILY = ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"]
FONT_SIZE = 6.0
AXES_LABEL_SIZE = 6.0
TICK_LABEL_SIZE = 6.0
LEGEND_FONT_SIZE = 5.0
SPINE_LINEWIDTH = 0.55
MAJOR_TICK_WIDTH = 0.45
MAJOR_TICK_LENGTH = 2.0
MINOR_TICK_WIDTH = 0.35
MINOR_TICK_LENGTH = 1.2
TICK_PAD = 1.5
EPOCH_XPAD_FRAC = 0.04
EPOCH_XPAD_MIN = 350

# Legend.
LEGEND_FRAME = False
LEGEND_HANDLE_LENGTH = 1.6
LEGEND_BORDER_PAD = 0.25
LEGEND_LABEL_SPACING = 0.25


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build separate Supplemental Figure 2 panels")
    p.add_argument("--xor-root", type=Path, default=DEFAULT_XOR_ROOT)
    p.add_argument("--nonlin-run", type=Path, default=DEFAULT_NONLIN_RUN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--panel-prefix", type=str, default=DEFAULT_PANEL_PREFIX)
    p.add_argument("--top-k-xor", type=int, default=1)
    p.add_argument("--xor-include-runs", nargs="+", default=DEFAULT_XOR_INCLUDE_RUNS)
    p.add_argument("--snapshots", type=int, nargs="+", default=DEFAULT_SNAPSHOTS)
    p.add_argument("--panel-width-in", type=float, default=PANEL_WIDTH_IN)
    p.add_argument("--panel-height-in", type=float, default=PANEL_HEIGHT_IN)
    return p.parse_args()


def _load_nonlin_dataset(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    dataset = meta["dataset"]
    return np.asarray(dataset["vins"], dtype=float), np.asarray(dataset["vous"], dtype=float)


def _load_xor_targets(run_dir: Path) -> np.ndarray:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    l0 = float(meta["dataset"]["L0"])
    return np.array([0.0, l0, l0, 0.0], dtype=float)


def _load_xor_runs(xor_root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for run_dir in sorted(p for p in xor_root.iterdir() if p.is_dir() and (p / "run_meta.json").exists()):
        acc = np.load(run_dir / "0_acc.npy")
        outputs = np.load(run_dir / "0_outputs.npy", allow_pickle=True)
        preds = np.stack(outputs).astype(float)
        targets = _load_xor_targets(run_dir)
        mse = ((preds - targets.reshape(1, -1)) ** 2).mean(axis=1)
        final_acc = float(acc[-1])
        final_mse = float(mse[-1])
        first_perfect = next((idx for idx, value in enumerate(acc) if value >= 1.0), int(1e9))
        runs.append(
            {
                "run_dir": run_dir,
                "label": run_dir.name,
                "acc": acc,
                "mse": mse,
                "final_acc": final_acc,
                "final_mse": final_mse,
                "first_perfect": first_perfect,
            }
        )

    if not runs:
        raise FileNotFoundError(f"No XOR run directories with run_meta.json found under {xor_root}")

    runs.sort(
        key=lambda item: (
            -float(item["final_acc"]),
            int(item["first_perfect"]),
            float(item["final_mse"]),
            str(item["label"]),
        )
    )
    return runs


def _validate_snapshots(preds_hist: np.ndarray, snapshots: list[int]) -> list[int]:
    max_epoch = preds_hist.shape[0] - 1
    out: list[int] = []
    for epoch in snapshots:
        if epoch < 0 or epoch > max_epoch:
            raise ValueError(f"snapshot epoch {epoch} is out of range 0..{max_epoch}")
        out.append(int(epoch))
    return out


def _style_axis(ax: plt.Axes) -> None:
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
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        width=MINOR_TICK_WIDTH,
        length=MINOR_TICK_LENGTH,
        pad=TICK_PAD,
        colors=AXIS_COLOR,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LINEWIDTH)
        spine.set_edgecolor(AXIS_COLOR)


def _set_xor_xticks(ax: plt.Axes, max_epoch: int) -> None:
    ticks = [0, 5000, 10000] if max_epoch >= 10000 else [0, max_epoch // 2, max_epoch]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(x)) for x in ticks])


def _set_epoch_xlim(ax: plt.Axes, max_epoch: int) -> None:
    pad = max(EPOCH_XPAD_MIN, EPOCH_XPAD_FRAC * float(max_epoch))
    ax.set_xlim(-pad, float(max_epoch) + pad)


def _configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.labelsize": AXES_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "ps.fonttype": 42,
        }
    )


def _make_panel(width_in: float, height_in: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP)
    return fig, ax


def _save(fig: plt.Figure, out_prefix: Path) -> list[Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_prefix.with_suffix(".png")]
    fig.savefig(paths[0], dpi=PNG_DPI)
    plt.close(fig)
    return paths


def _draw_panel_a(xor_runs: list[dict[str, object]], out_prefix: Path, width_in: float, height_in: float) -> list[Path]:
    fig, ax = _make_panel(width_in, height_in)
    for run in xor_runs:
        mse = np.clip(np.asarray(run["mse"], dtype=float), np.finfo(float).tiny, None)
        ax.plot(np.arange(1, mse.size + 1), mse, color=BLACK, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("XOR MSE")
    ax.set_yscale("log")
    xor_max_epoch = max(np.asarray(run["mse"]).size for run in xor_runs)
    _set_epoch_xlim(ax, xor_max_epoch)
    _set_xor_xticks(ax, xor_max_epoch)
    _style_axis(ax)
    return _save(fig, out_prefix)


def _draw_panel_b(xor_runs: list[dict[str, object]], out_prefix: Path, width_in: float, height_in: float) -> list[Path]:
    fig, ax = _make_panel(width_in, height_in)
    xor_max_epoch = max(np.asarray(run["acc"]).size for run in xor_runs) - 1
    for run in xor_runs:
        acc = np.asarray(run["acc"], dtype=float)
        ax.plot(np.arange(acc.size), acc, color=BLACK, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("XOR Accuracy")
    _set_epoch_xlim(ax, xor_max_epoch)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    _set_xor_xticks(ax, xor_max_epoch)
    _style_axis(ax)
    return _save(fig, out_prefix)


def _draw_panel_c(nonlin_mse: np.ndarray, out_prefix: Path, width_in: float, height_in: float) -> list[Path]:
    fig, ax = _make_panel(width_in, height_in)
    mse = np.clip(np.asarray(nonlin_mse, dtype=float), np.finfo(float).tiny, None)
    max_epoch = mse.size - 1
    ax.plot(np.arange(mse.size), mse, color=BLACK, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Regression MSE")
    ax.set_yscale("log")
    _set_epoch_xlim(ax, max_epoch)
    ax.set_xticks(np.arange(0, max_epoch + 1, 5000))
    _style_axis(ax)
    return _save(fig, out_prefix)


def _draw_panel_d(
    nonlin_x: np.ndarray,
    nonlin_y: np.ndarray,
    nonlin_preds: np.ndarray,
    snapshots: list[int],
    out_prefix: Path,
    width_in: float,
    height_in: float,
) -> list[Path]:
    fig, ax = _make_panel(width_in, height_in)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))
    ax.scatter(
        nonlin_x,
        nonlin_y,
        s=TRAIN_POINT_SIZE,
        facecolors="none",
        edgecolors=BLACK,
        linewidths=TRAIN_POINT_LINEWIDTH,
        label="_nolegend_",
    )
    for color, epoch in zip(colors, snapshots):
        ax.plot(
            nonlin_x,
            nonlin_preds[epoch],
            color=color,
            linewidth=CURVE_LINEWIDTH,
            alpha=CURVE_ALPHA,
            label=f"{epoch}",
        )
    ax.set_xlabel("Input Voltage (V)")
    ax.set_ylabel("Output Voltage (V)")
    legend = ax.legend(
        frameon=LEGEND_FRAME,
        loc="lower right",
        title="Epoch Number",
        title_fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        borderpad=LEGEND_BORDER_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    if legend is not None:
        legend.get_title().set_fontfamily(FONT_FAMILY[0])
    _style_axis(ax)
    return _save(fig, out_prefix)


def main() -> None:
    args = _parse_args()
    xor_root = args.xor_root.resolve()
    nonlin_run = args.nonlin_run.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_xor_labels = set(str(x) for x in args.xor_include_runs)
    xor_runs = [run for run in _load_xor_runs(xor_root) if str(run["label"]) in allowed_xor_labels]
    xor_runs = xor_runs[: max(1, int(args.top_k_xor))]
    if not xor_runs:
        raise FileNotFoundError(f"No selected XOR runs found under {xor_root} for labels={sorted(allowed_xor_labels)}")

    nonlin_mse = np.load(nonlin_run / "mse_history.npy")
    nonlin_preds = np.load(nonlin_run / "preds_history.npy")
    nonlin_x, nonlin_y = _load_nonlin_dataset(nonlin_run)
    snapshots = _validate_snapshots(nonlin_preds, [int(x) for x in args.snapshots])

    _configure_fonts()

    prefix = str(args.panel_prefix)
    width_in = float(args.panel_width_in)
    height_in = float(args.panel_height_in)
    written: list[Path] = []
    written.extend(_draw_panel_a(xor_runs, out_dir / f"{prefix}_a", width_in, height_in))
    written.extend(_draw_panel_b(xor_runs, out_dir / f"{prefix}_b", width_in, height_in))
    written.extend(_draw_panel_c(nonlin_mse, out_dir / f"{prefix}_c", width_in, height_in))
    written.extend(
        _draw_panel_d(
            nonlin_x,
            nonlin_y,
            nonlin_preds,
            snapshots,
            out_dir / f"{prefix}_d",
            width_in,
            height_in,
        )
    )

    print(f"xor_root={xor_root}")
    print(f"nonlin_run={nonlin_run}")
    print(f"selected_xor_runs={[str(run['label']) for run in xor_runs]}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
