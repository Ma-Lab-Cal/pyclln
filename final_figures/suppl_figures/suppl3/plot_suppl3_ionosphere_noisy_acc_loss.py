#!/usr/bin/env python3
"""Supplementary Figure 3: noisy ionosphere epoch-wise acc/loss curves.

Uses the same acc/loss panel style as the released ionosphere panel in Fig. 2,
with the canvas width scaled to two thirds of the original. Each chip is
rendered as a separate figure.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np


_PR = next(p for p in Path(__file__).resolve().parents if (p / "device_model").is_dir())
HERE = Path(__file__).resolve().parent
SOURCE_DIR = _PR / "ionosphere" / "runs"
OUT_PREFIX = HERE / "data" / "suppl3_ionosphere_noisy_acc_loss"

PNG_DPI = 600

TEXT = "#34383D"
TRAIN_COLOR = "#274E87"
TEST_COLOR = "#D34F72"
ACC_LOSS_CURVE_LINEWIDTH = 0.65
ACC_LOSS_CURVE_ALPHA = 0.7
ACC_LOSS_MARKER_SIZE = 2.2

FONT_FAMILY = ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"]
AXIS_LABEL_SIZE = 6.0
TICK_LABEL_SIZE = 6.0
LEGEND_LABEL_SIZE = 5.0
PANEL_SPINE_LINEWIDTH = 0.55

FIG2_ACC_LOSS_SIZE = (3.24, 2.55)
ACC_LOSS_SIZE = (FIG2_ACC_LOSS_SIZE[0] * 2.0 / 3.0, FIG2_ACC_LOSS_SIZE[1])
ACC_LOSS_LEFT = 0.265
ACC_LOSS_RIGHT = 0.94
ACC_LOSS_BOTTOM = 0.125
ACC_LOSS_TOP = 0.992
ACC_LOSS_HSPACE = 0.075


def _set_common_rc(font_size: float = 5.0) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": font_size,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_LABEL_SIZE,
            "ps.fonttype": 42,
        }
    )


def _sci_tick(value: float, _pos: int) -> str:
    if abs(value) < 1e-15:
        return "0"
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=0.45,
        length=2.0,
        pad=1.5,
        colors=TEXT,
        labelsize=TICK_LABEL_SIZE,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        width=0.35,
        length=1.2,
        pad=1.5,
        colors=TEXT,
        labelsize=TICK_LABEL_SIZE,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PANEL_SPINE_LINEWIDTH)
        spine.set_edgecolor(TEXT)


def _load_curve(path: Path) -> dict[str, np.ndarray | int | float | str]:
    z = np.load(path, allow_pickle=True)
    required = {"train_acc", "test_acc", "train_loss", "test_loss", "epoch"}
    missing = sorted(required.difference(z.files))
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")

    epochs = np.asarray(z["epoch"], dtype=float)
    train_acc = np.asarray(z["train_acc"], dtype=float)
    test_acc = np.asarray(z["test_acc"], dtype=float)
    train_loss = np.asarray(z["train_loss"], dtype=float)
    test_loss = np.asarray(z["test_loss"], dtype=float)
    epoch0_source = "present"
    if epochs.size and epochs[0] > 0:
        epochs = np.concatenate([[0.0], epochs])
        train_acc = np.concatenate([[np.nan], train_acc])
        test_acc = np.concatenate([[np.nan], test_acc])
        train_loss = np.concatenate([[np.nan], train_loss])
        test_loss = np.concatenate([[np.nan], test_loss])
        epoch0_source = "prepended_unmeasured_nan"

    finite = np.isfinite(test_acc)
    if not np.any(finite):
        raise ValueError(f"{path} has no finite test accuracy values")
    return {
        "source_path": str(path),
        "run": path.parent.name,
        "epoch": epochs,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "epoch0_source": epoch0_source,
    }


def _save(fig: plt.Figure, out_prefix: Path) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    return png


def _draw(curve: dict[str, np.ndarray | int | float | str], out_prefix: Path) -> Path:
    _set_common_rc(6.0)
    fig, (ax_acc, ax_loss) = plt.subplots(
        2,
        1,
        figsize=ACC_LOSS_SIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": ACC_LOSS_HSPACE},
    )
    fig.subplots_adjust(left=ACC_LOSS_LEFT, right=ACC_LOSS_RIGHT, bottom=ACC_LOSS_BOTTOM, top=ACC_LOSS_TOP)

    epoch = np.asarray(curve["epoch"], dtype=float)
    train_acc = np.asarray(curve["train_acc"], dtype=float)
    test_acc = np.asarray(curve["test_acc"], dtype=float)
    train_loss = np.clip(np.asarray(curve["train_loss"], dtype=float), np.finfo(float).tiny, None)
    test_loss = np.clip(np.asarray(curve["test_loss"], dtype=float), np.finfo(float).tiny, None)
    for ax, a_train, a_test in (
        (ax_acc, train_acc, test_acc),
        (ax_loss, train_loss, test_loss),
    ):
        ax.plot(
            epoch,
            a_train,
            color=TRAIN_COLOR,
            linewidth=ACC_LOSS_CURVE_LINEWIDTH,
            alpha=ACC_LOSS_CURVE_ALPHA,
            marker="o",
            markersize=ACC_LOSS_MARKER_SIZE,
            markeredgewidth=0.0,
            label="Train",
        )
        ax.plot(
            epoch,
            a_test,
            color=TEST_COLOR,
            linewidth=ACC_LOSS_CURVE_LINEWIDTH,
            alpha=ACC_LOSS_CURVE_ALPHA,
            marker="o",
            markersize=ACC_LOSS_MARKER_SIZE,
            markeredgewidth=0.0,
            label="Test",
        )
        _style_axis(ax)

    ax_acc.set_ylabel("Accuracy", fontsize=AXIS_LABEL_SIZE, labelpad=1.0)
    ax_loss.set_ylabel("Hinge Loss", fontsize=AXIS_LABEL_SIZE, labelpad=1.0)
    x_pad = max(0.35, 0.04 * float(epoch[-1] - epoch[0]))
    for ax in (ax_acc, ax_loss):
        ax.set_xlim(float(epoch[0]) - x_pad, float(epoch[-1]) + x_pad)

    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_loss.legend(
        loc="upper right",
        frameon=False,
        fontsize=LEGEND_LABEL_SIZE,
        handlelength=1.45,
        borderpad=0.25,
        borderaxespad=0.25,
        labelspacing=0.25,
    )
    loss_max = float(np.nanmax([np.nanmax(train_loss), np.nanmax(test_loss)]))
    ax_loss.set_ylim(0.0, loss_max * 1.08)
    ax_loss.yaxis.set_major_formatter(ticker.FuncFormatter(_sci_tick))
    ax_loss.set_xlabel("Epoch", fontsize=AXIS_LABEL_SIZE)
    ax_acc.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    fig.align_ylabels([ax_acc, ax_loss])
    return _save(fig, out_prefix)


def main() -> None:
    paths = [SOURCE_DIR / f"noisy_chip{i}" / "curve.npz" for i in (1, 2, 3)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing noisy ionosphere curve files: " + ", ".join(missing))

    curves = [_load_curve(p) for p in paths]
    panel_names = {1: "suppl3_a", 2: "suppl3_b", 3: "suppl3_c"}
    pngs = []
    for i, curve in enumerate(curves, start=1):
        pngs.append(_draw(curve, HERE / panel_names[i]))
    np.savez(
        OUT_PREFIX.with_suffix(".npz"),
        source_paths=np.asarray([c["source_path"] for c in curves], dtype=object),
        figure_size_in=np.asarray(ACC_LOSS_SIZE, dtype=float),
        fig2_acc_loss_size_in=np.asarray(FIG2_ACC_LOSS_SIZE, dtype=float),
        width_scale=2.0 / 3.0,
        epoch_source="direct epoch-wise noisy-chip curve.npz",
        epoch0_source=np.asarray([c["epoch0_source"] for c in curves], dtype=object),
    )
    for png in pngs:
        print(png)
    print(OUT_PREFIX.with_suffix(".npz"))


if __name__ == "__main__":
    main()
