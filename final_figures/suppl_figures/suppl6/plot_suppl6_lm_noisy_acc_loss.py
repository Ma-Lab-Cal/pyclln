#!/usr/bin/env python3
"""Supplementary Figure 6: noisy LM stacked QMass and cross-entropy curves.

Each noisy chip is rendered as one stacked figure: train/test QMass on top and
train/test cross-entropy on bottom. Curve styling follows the Fig. 4 LM QMass
and cross-entropy panels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


_PR = next(p for p in Path(__file__).resolve().parents if (p / "device_model").is_dir())
HERE = Path(__file__).resolve().parent
SOURCE_DIR = _PR / "language_model" / "runs"
TEMP_SWEEP_DIR = _PR / "final_figures" / "main_figures" / "fig4"
OUT_PREFIX = HERE / "data" / "suppl6_lm_noisy_curves"

PNG_DPI = 600
FIG2_ACC_LOSS_SIZE = (3.24, 2.55)
STACKED_SIZE = (FIG2_ACC_LOSS_SIZE[0] * 2.0 / 3.0, FIG2_ACC_LOSS_SIZE[1])

FONT_FAMILY = ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"]
TEXT_SIZE = 6.0
LEGEND_TEXT_SIZE = 5.0

TRAIN_COLOR = "#274E87"
TEST_COLOR = "#D34F72"
AXIS_COLOR = "#34383D"
CURVE_LINEWIDTH = 0.65
CURVE_ALPHA = 0.7
MARKER_SIZE = 2.2
MARKER_EDGE_WIDTH = 0.0

LEFT = 0.265
RIGHT = 0.94
BOTTOM = 0.125
TOP = 0.992
HSPACE = 0.075
AXIS_BOX_SIZE = (
    STACKED_SIZE[0] * (RIGHT - LEFT),
    STACKED_SIZE[1] * (TOP - BOTTOM) / (2.0 + HSPACE),
)
TEMP_LEFT = 0.195
TEMP_RIGHT = 0.970
TEMP_BOTTOM = 0.245
TEMP_TOP = 0.965
TEMP_FIG_SIZE = (
    AXIS_BOX_SIZE[0] / (TEMP_RIGHT - TEMP_LEFT),
    AXIS_BOX_SIZE[1] / (TEMP_TOP - TEMP_BOTTOM),
)
TEMP_SELECTED_LINEWIDTH = 0.65


def _set_rc() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY[0],
            "font.sans-serif": FONT_FAMILY,
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": LEGEND_TEXT_SIZE,
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


def _style_axis(ax: plt.Axes) -> None:
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
        label.set_fontsize(TEXT_SIZE)


def _padded_ylim(train: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([train, test]).astype(float)
    vals = vals[np.isfinite(vals)]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = max(hi - lo, 1e-6)
    return lo - 0.12 * span, hi + 0.12 * span


def _load_curve(path: Path) -> dict[str, np.ndarray | str]:
    data = np.load(path, allow_pickle=True)
    required = {"epoch", "train_qmass", "test_qmass", "train_ce", "test_ce"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")

    epochs = np.asarray(data["epoch"], dtype=float)
    qmass_train = np.asarray(data["train_qmass"], dtype=float)
    qmass_test = np.asarray(data["test_qmass"], dtype=float)
    ce_train = np.asarray(data["train_ce"], dtype=float)
    ce_test = np.asarray(data["test_ce"], dtype=float)
    if epochs.size and epochs[0] > 0:
        epochs = np.concatenate([[0.0], epochs])
        qmass_train = np.concatenate([[np.nan], qmass_train])
        qmass_test = np.concatenate([[np.nan], qmass_test])
        ce_train = np.concatenate([[np.nan], ce_train])
        ce_test = np.concatenate([[np.nan], ce_test])

    curves = {
        "source_path": str(path),
        "run": path.parent.name,
        "epochs": epochs,
        "qmass_train": qmass_train,
        "qmass_test": qmass_test,
        "ce_train": ce_train,
        "ce_test": ce_test,
    }
    finite = np.zeros_like(curves["epochs"], dtype=bool)
    for key in ("qmass_train", "qmass_test", "ce_train", "ce_test"):
        finite |= np.isfinite(curves[key])
    if not np.any(finite):
        raise ValueError(f"{path} has no finite train/test CE and QMass rows")
    return curves


def _plot_pair(
    ax: plt.Axes,
    epochs: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ylabel: str,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
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
    ax.set_ylabel(ylabel)
    ax.set_ylim(*(ylim if ylim is not None else _padded_ylim(train, test)))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    _style_axis(ax)


def _draw(curve: dict[str, np.ndarray | str], out_prefix: Path) -> Path:
    epochs = np.asarray(curve["epochs"], dtype=float)
    qmass_train = np.asarray(curve["qmass_train"], dtype=float)
    qmass_test = np.asarray(curve["qmass_test"], dtype=float)
    ce_train = np.asarray(curve["ce_train"], dtype=float)
    ce_test = np.asarray(curve["ce_test"], dtype=float)

    fig, (ax_qmass, ax_ce) = plt.subplots(
        2,
        1,
        figsize=STACKED_SIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": HSPACE},
    )
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)

    _plot_pair(ax_qmass, epochs, qmass_train, qmass_test, "QMass")
    _plot_pair(ax_ce, epochs, ce_train, ce_test, "Cross Entropy Loss")

    x_pad = max(0.35, 0.04 * float(epochs[-1] - epochs[0]))
    for ax in (ax_qmass, ax_ce):
        ax.set_xlim(float(epochs[0]) - x_pad, float(epochs[-1]) + x_pad)
        if epochs.size <= 12:
            ax.set_xticks(epochs)
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))

    ax_ce.set_xlabel("Epoch")
    ax_qmass.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    legend = ax_ce.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.5,
        borderpad=0.25,
        labelspacing=0.25,
    )
    for text in legend.get_texts():
        text.set_fontfamily(FONT_FAMILY[0])
        text.set_fontsize(LEGEND_TEXT_SIZE)
    fig.align_ylabels([ax_qmass, ax_ce])

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    return png


def _load_temp_sweep(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _draw_temp_sweep(payload: dict, out_prefix: Path) -> Path:
    points = sorted(payload["points"], key=lambda row: row["temperature"])
    x = np.asarray([row["temperature"] for row in points], dtype=float)
    validity = np.asarray([row["valid_pct"] for row in points], dtype=float)
    novelty = np.asarray([row["novelty_pct"] for row in points], dtype=float)
    uniqueness = np.asarray([row["uniqueness_pct"] for row in points], dtype=float)

    fig, ax = plt.subplots(figsize=TEMP_FIG_SIZE)
    fig.subplots_adjust(left=TEMP_LEFT, right=TEMP_RIGHT, bottom=TEMP_BOTTOM, top=TEMP_TOP)
    ax.plot(
        x,
        validity,
        "-o",
        linewidth=CURVE_LINEWIDTH,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        alpha=CURVE_ALPHA,
        label="Validity",
        zorder=3,
    )
    ax.plot(
        x,
        novelty,
        "-o",
        linewidth=CURVE_LINEWIDTH,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        alpha=CURVE_ALPHA,
        label="Novelty",
        zorder=3,
    )
    ax.plot(
        x,
        uniqueness,
        "-o",
        linewidth=CURVE_LINEWIDTH,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        alpha=CURVE_ALPHA,
        label="Uniqueness",
        zorder=3,
    )
    ax.axvline(
        float(payload["selected_temp"]),
        color="black",
        linewidth=TEMP_SELECTED_LINEWIDTH,
        linestyle="-",
        alpha=CURVE_ALPHA,
        label="Selected T",
        zorder=1,
    )
    ax.axvline(
        float(payload["training_temp"]),
        color="black",
        linewidth=TEMP_SELECTED_LINEWIDTH,
        linestyle="--",
        alpha=CURVE_ALPHA,
        label="Training T",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_xlim(max(1e-4, float(np.min(x))), float(np.max(x)) * 1.08)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Readout Temperature T")
    _style_axis(ax)

    legend = ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        handlelength=1.45,
        handletextpad=0.5,
        borderaxespad=0.45,
        labelspacing=0.25,
        fontsize=LEGEND_TEXT_SIZE,
    )
    legend.get_frame().set_linewidth(0.0)
    for text in legend.get_texts():
        text.set_fontfamily(FONT_FAMILY[0])
        text.set_fontsize(LEGEND_TEXT_SIZE)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    return png


def main() -> None:
    _set_rc()
    paths = [SOURCE_DIR / f"noisy_chip{i}" / "curve.npz" for i in (1, 2, 3)]
    temp_paths = [TEMP_SWEEP_DIR / f"fig4_lm_readout_temperature_sweep_noisy_chip{i}.json" for i in (1, 2, 3)]
    missing = [str(path) for path in paths if not path.exists()]
    missing += [str(path) for path in temp_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing noisy LM source files: " + ", ".join(missing))

    qmass_ce_names = {1: "suppl6_a", 2: "suppl6_b", 3: "suppl6_c"}
    temp_sweep_names = {1: "suppl6_d", 2: "suppl6_e", 3: "suppl6_f"}

    curves = [_load_curve(path) for path in paths]
    outputs = [
        _draw(curve, HERE / qmass_ce_names[i])
        for i, curve in enumerate(curves, start=1)
    ]
    temp_payloads = [_load_temp_sweep(path) for path in temp_paths]
    outputs.extend(
        _draw_temp_sweep(payload, HERE / temp_sweep_names[i])
        for i, payload in enumerate(temp_payloads, start=1)
    )

    np.savez(
        OUT_PREFIX.with_suffix(".npz"),
        source_paths=np.asarray([curve["source_path"] for curve in curves], dtype=object),
        temperature_sweep_paths=np.asarray([str(path) for path in temp_paths], dtype=object),
        final_test_ce=np.asarray([float(np.asarray(curve["ce_test"], dtype=float)[-1]) for curve in curves]),
        figure_size_in=np.asarray(STACKED_SIZE, dtype=float),
        axis_box_size_in=np.asarray(AXIS_BOX_SIZE, dtype=float),
        temperature_sweep_figure_size_in=np.asarray(TEMP_FIG_SIZE, dtype=float),
        fig4_single_panel_size_in=np.asarray((2.06, 1.73), dtype=float),
        style_source="paper_release/final_figures/main_figures/fig4/plot_fig4_lm_learning_curves.py",
        temperature_style_source="paper_release/final_figures/main_figures/fig4/plot_fig4_lm_temperature_sweep_release_final.py",
        epoch0_note="epoch 0 is included from the paper-release noisy LM curves.",
    )
    for path in outputs:
        print(path)
    print(OUT_PREFIX.with_suffix(".npz"))


if __name__ == "__main__":
    main()
