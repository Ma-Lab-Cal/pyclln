#!/usr/bin/env python3
"""Render all six panels of Figure 5 ("Robustness and design sweeps").

Run once, this regenerates every panel of Figure 5 from the CSV sweeps under
``paper_release/hparam_noise_sweeps/``:

    a  fig5_a.png  Digits test accuracy + hinge loss vs read-noise sigma
    b  fig5_b.png  accuracy + loss under device mismatch at inference
    c  fig5_c.png  test accuracy for MSE/CE/hinge objectives (Ionosphere + Digits)
    d  fig5_d.png  accuracy across nMOS operating modes (body x gate)
    e  fig5_e.png  accuracy vs init-distribution mean mu and half-width h
    f  fig5_f.png  accuracy vs learning-rate gamma and clamp delta

Filtered/derived data tables are written to the ``data/`` subfolder. Input CSVs
are read from ``paper_release/hparam_noise_sweeps/`` and are left untouched.

Usage:
    conda run -n p311env python plot_fig5.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

# Robust paper_release root (self-contained regardless of nesting depth).
_PR = next(p for p in Path(__file__).resolve().parents if (p / "device_model").is_dir())

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_OUT_DIR = DEFAULT_OUT_DIR / "data"
DATA_DIR = _PR / "hparam_noise_sweeps"
READ_NOISE_SOURCE = DATA_DIR / "read_noise" / "read_noise_sweep.csv"
MISMATCH_SOURCE = DATA_DIR / "mismatch_inference" / "mismatch_inference_summary.csv"

PNG_DPI = 600
FONT_FAMILY = ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"]
TEXT_SIZE = 6.0
TICK_SIZE = 6.0
LEGEND_SIZE = 5.0
SMALL_SIZE = 6.0
TITLE_SIZE = 6.0
AXIS_COLOR = "#000000"
ACC_COLOR = "#000000"
LOSS_COLOR = "#1F6EB3"
LINEWIDTH = 1.0
MARKER_SIZE = 2.6
ERROR_LINEWIDTH = 0.45
ERROR_CAPSIZE = 1.2
CURVE_ALPHA = 0.8


def _set_rc() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY[0],
            "font.sans-serif": FONT_FAMILY,
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "axes.labelcolor": AXIS_COLOR,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "xtick.color": AXIS_COLOR,
            "ytick.labelsize": TICK_SIZE,
            "ytick.color": AXIS_COLOR,
            "legend.fontsize": LEGEND_SIZE,
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
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        width=0.35,
        length=1.1,
        pad=1.2,
        colors=AXIS_COLOR,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor(AXIS_COLOR)


def _png_size(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


# ===========================================================================
# Panel a  --  Digits read-noise accuracy / loss
# (from plot_fig5_read_noise.py)
# ===========================================================================
RN_FIG_WIDTH_IN = 2.21
RN_AXES_RECT = (0.20, 0.14, 0.54, 0.78)
RN_FIG_HEIGHT_IN = RN_FIG_WIDTH_IN * RN_AXES_RECT[2] / RN_AXES_RECT[3]


def _rn_force_text_style(fig: plt.Figure) -> None:
    for item in fig.findobj(match=Text):
        item.set_fontfamily(FONT_FAMILY[0])
        item.set_fontsize(TEXT_SIZE)
        item.set_color(AXIS_COLOR)


def _rn_format_e_tick(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-15:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_value = int(exponent)
    if "." in mantissa:
        decimals = len(mantissa.split(".", 1)[1])
        mantissa = mantissa.replace(".", "")
        exponent_value -= decimals
    return f"{mantissa}e{exponent_value}"


def _rn_loss_ticks(top: float) -> np.ndarray:
    if top >= 0.1:
        return np.asarray([0.05, 0.10], dtype=float)
    step = 0.02 if top <= 0.12 else 0.05
    return np.arange(0.0, top + 0.5 * step, step)


def _rn_read_rows(source: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sweep") in ("read_noise", "read_only"):
                rows.append(row)
    rows.sort(key=lambda row: float(row["noise_mV"]))
    if not rows:
        raise ValueError(f"no read_only rows found in {source}")
    _rn_add_perseed_stds(rows, source.with_name("read_noise_perseed.csv"))
    return rows


def _rn_sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _rn_add_perseed_stds(rows: list[dict[str, str]], perseed_path: Path) -> None:
    if not perseed_path.exists():
        for row in rows:
            row.setdefault("test_loss_std", row.get("endpoint_test_loss_std", "0"))
        return

    acc_by_noise: dict[float, list[float]] = {}
    loss_by_noise: dict[float, list[float]] = {}
    with perseed_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sweep") not in ("read_noise", "read_only"):
                continue
            noise = float(row["noise_mV"])
            acc_by_noise.setdefault(noise, []).append(float(row["endpoint_test_acc"]))
            loss_by_noise.setdefault(noise, []).append(float(row["endpoint_test_loss"]))

    for row in rows:
        noise = float(row["noise_mV"])
        if row.get("test_acc_std", "") == "" and noise in acc_by_noise:
            row["test_acc_std"] = f"{_rn_sample_std(acc_by_noise[noise]):.17g}"
        row["test_loss_std"] = f"{_rn_sample_std(loss_by_noise.get(noise, [])):.17g}"


def _rn_float_or_zero(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _rn_write_filtered_csv(rows: list[dict[str, str]], out_path: Path) -> Path:
    fieldnames = [
        "noise_mV",
        "endpoint_test_acc",
        "test_acc_std",
        "endpoint_test_loss",
        "test_loss_std",
        "endpoint_epoch",
        "n_seeds",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "noise_mV": row["noise_mV"],
                    "endpoint_test_acc": row["endpoint_test_acc"],
                    "test_acc_std": row.get("test_acc_std", ""),
                    "endpoint_test_loss": row["endpoint_test_loss"],
                    "test_loss_std": row.get("test_loss_std", ""),
                    "endpoint_epoch": row["endpoint_epoch"],
                    "n_seeds": row.get("n_seeds", ""),
                }
            )
    return out_path


def _rn_save_tight_width_square_axes(
    fig: plt.Figure,
    out_path: Path,
    dpi: int,
    width_in: float,
    axes_rect: tuple[float, float, float, float],
) -> None:
    target_width_px = int(round(width_in * dpi))
    axes_aspect_height_per_width = axes_rect[2] / axes_rect[3]
    tmp_path = out_path.with_name(f".{out_path.stem}.tmp{out_path.suffix}")
    fig_width, _ = fig.get_size_inches()
    for _ in range(10):
        fig.set_size_inches(fig_width, fig_width * axes_aspect_height_per_width, forward=True)
        fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
        size = _png_size(tmp_path)
        if size is not None and size[0] == target_width_px:
            tmp_path.replace(out_path)
            return
        if size is None or size[0] <= 0:
            break
        fig_width *= target_width_px / size[0]
    fig.set_size_inches(fig_width, fig_width * axes_aspect_height_per_width, forward=True)
    fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    tmp_path.replace(out_path)


def _plot_read_noise(rows: list[dict[str, str]], out_path: Path, dpi: int) -> Path:
    x = np.asarray([float(row["noise_mV"]) for row in rows], dtype=float)
    acc = np.asarray([float(row["endpoint_test_acc"]) for row in rows], dtype=float)
    acc_std = np.asarray([_rn_float_or_zero(row.get("test_acc_std")) for row in rows], dtype=float)
    loss = np.asarray([float(row["endpoint_test_loss"]) for row in rows], dtype=float)
    loss_std = np.asarray([_rn_float_or_zero(row.get("test_loss_std")) for row in rows], dtype=float)

    fig = plt.figure(figsize=(RN_FIG_WIDTH_IN, RN_FIG_HEIGHT_IN))
    ax_acc = fig.add_axes(RN_AXES_RECT)
    ax_loss = ax_acc.twinx()

    ax_acc.errorbar(
        x,
        acc,
        yerr=acc_std,
        color=ACC_COLOR,
        linewidth=LINEWIDTH,
        elinewidth=ERROR_LINEWIDTH,
        capsize=ERROR_CAPSIZE,
        capthick=ERROR_LINEWIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgewidth=0.0,
        label="Acc",
    )
    ax_loss.errorbar(
        x,
        loss,
        yerr=loss_std,
        color=LOSS_COLOR,
        linewidth=LINEWIDTH,
        elinewidth=ERROR_LINEWIDTH,
        capsize=ERROR_CAPSIZE,
        capthick=ERROR_LINEWIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgewidth=0.0,
        label="Loss",
    )

    ax_acc.set_xlabel("Read Noise σ [mV]")
    ax_acc.set_ylabel("Test Accuracy", color=ACC_COLOR)
    ax_loss.set_ylabel("Test Loss", color=LOSS_COLOR)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    x_pad = 0.05 * (x_max - x_min)
    ax_acc.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_acc.set_xticks([0, 20, 40, 60, 80, 100])
    acc_floor = min(0.45, float(np.nanmin(acc - acc_std)) - 0.02)
    ax_acc.set_ylim(max(0.0, acc_floor), 1.02)
    ax_acc.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if acc_floor < 0.45 else [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    loss_top = max(0.11, float(np.nanmax(loss + loss_std)) * 1.06)
    ax_loss.set_ylim(0.0, loss_top)
    ax_loss.yaxis.set_major_formatter(FuncFormatter(_rn_format_e_tick))
    ax_loss.set_yticks(_rn_loss_ticks(loss_top))

    handles = [
        Line2D([0], [0], color=ACC_COLOR, linewidth=LINEWIDTH, marker="o", markersize=MARKER_SIZE, label="Acc"),
        Line2D([0], [0], color=LOSS_COLOR, linewidth=LINEWIDTH, marker="o", markersize=MARKER_SIZE, label="Loss"),
    ]
    legend = ax_acc.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        frameon=False,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.0,
        columnspacing=0.9,
        ncol=2,
    )
    for text, color in zip(legend.get_texts(), (ACC_COLOR, LOSS_COLOR)):
        text.set_fontsize(LEGEND_SIZE)
        text.set_color(color)

    _style_axis(ax_acc)
    ax_acc.spines["top"].set_position(("axes", 0.998))
    ax_acc.spines["right"].set_visible(False)
    _style_axis(ax_loss)
    ax_loss.spines["left"].set_visible(False)
    ax_loss.spines["bottom"].set_visible(False)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_loss.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True, colors=AXIS_COLOR)

    _rn_force_text_style(fig)
    ax_loss.yaxis.label.set_color(LOSS_COLOR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _rn_save_tight_width_square_axes(fig, out_path, dpi, RN_FIG_WIDTH_IN, RN_AXES_RECT)
    plt.close(fig)
    return out_path


def render_panel_a(out_dir: Path, data_out_dir: Path, dpi: int) -> Path:
    """Panel a: Digits test accuracy + hinge loss vs read-noise sigma."""
    rows = _rn_read_rows(READ_NOISE_SOURCE)
    _rn_write_filtered_csv(rows, data_out_dir / "scikit_read_noise_test_acc_loss.csv")
    return _plot_read_noise(rows, out_dir / "fig5_a.png", dpi)


# ===========================================================================
# Panel b  --  Digits inference-only mismatch-noise accuracy / loss
# (from plot_fig5_mismatch_noise.py)
# ===========================================================================
MM_FIG_WIDTH_IN = 2.21
MM_TARGET_HEIGHT_IN = 1.85
MM_AXES_RECT = (0.20, 0.14, 0.54, 0.78)
MM_FIG_HEIGHT_IN = MM_FIG_WIDTH_IN * MM_AXES_RECT[2] / MM_AXES_RECT[3]


def _mm_force_text_style(fig: plt.Figure) -> None:
    for item in fig.findobj(match=Text):
        item.set_fontfamily(FONT_FAMILY[0])
        item.set_fontsize(TEXT_SIZE)
        item.set_color(AXIS_COLOR)


def _mm_format_e_tick(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-15:
        return "0"
    mantissa, exponent = f"{value:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = str(int(exponent))
    return f"{mantissa}e{exponent}"


def _mm_nice_upper(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    target = value * 1.08
    decade = 10 ** np.floor(np.log10(target))
    for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        top = step * decade
        if top >= target:
            return float(top)
    return float(10.0 * decade)


def _mm_scale_ticks(scales: np.ndarray) -> list[float]:
    candidates = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    lo, hi = float(np.nanmin(scales)), float(np.nanmax(scales))
    ticks = [x for x in candidates if lo <= x <= hi]
    if len(ticks) > 6:
        ticks = [x for x in ticks if x in {0.2, 1.0, 10.0, hi}]
    if hi not in ticks:
        ticks.append(hi)
    return sorted(set(ticks))


def _mm_format_scale_tick(value: float) -> str:
    return f"{value:g}x"


def _mm_loss_axis(loss: np.ndarray, loss_std: np.ndarray) -> tuple[float, float, np.ndarray]:
    lower = float(np.nanmin(loss - loss_std))
    upper = float(np.nanmax(loss + loss_std))
    pad = max((upper - lower) * 0.10, upper * 0.02)
    bottom = max(0.0, lower - pad)
    top = _mm_nice_upper(upper)
    if top <= 0.01:
        tick_step = 0.002
    elif top <= 0.02:
        tick_step = 0.005
    elif top <= 0.12:
        tick_step = 0.02
    else:
        tick_step = 0.05
    first_tick = np.ceil(bottom / tick_step) * tick_step
    ticks = np.arange(first_tick, top + 0.5 * tick_step, tick_step)
    return bottom, top, ticks


def _mm_read_rows(source: Path) -> list[dict[str, object]]:
    if not source.exists():
        raise FileNotFoundError(f"missing mismatch source CSV: {source}")
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw = list(reader)
        fields = set(reader.fieldnames or [])
    if "test_acc_mean" in fields:
        rows = [
            {
                "noise_scale": float(row["noise_scale"]),
                "vt_sigma_mV": float(row["vt_sigma_mV"]),
                "beta_sigma_percent": float(row["beta_sigma_percent"]),
                "test_acc": float(row["test_acc_mean"]),
                "test_loss": float(row["test_loss_mean"]),
                "test_acc_std": float(row.get("test_acc_std", 0.0) or 0.0),
                "test_loss_std": float(row.get("test_loss_std", 0.0) or 0.0),
                "n_chips": int(float(row.get("n_chips", 1) or 1)),
                "status": row.get("status", "ok"),
            }
            for row in raw
            if row.get("status") == "ok" and row.get("test_acc_mean") not in (None, "")
        ]
    elif "test_acc" in fields:
        grouped: dict[float, list[dict[str, str]]] = {}
        for row in raw:
            if row.get("status") == "ok" and row.get("test_acc") not in (None, ""):
                grouped.setdefault(float(row["noise_scale"]), []).append(row)
        rows = []
        for scale, group in grouped.items():
            acc = np.asarray([float(row["test_acc"]) for row in group], dtype=float)
            loss = np.asarray([float(row["test_loss"]) for row in group], dtype=float)
            rows.append(
                {
                    "noise_scale": scale,
                    "vt_sigma_mV": float(group[0]["vt_sigma_mV"]),
                    "beta_sigma_percent": float(group[0]["beta_sigma_percent"]),
                    "test_acc": float(acc.mean()),
                    "test_loss": float(loss.mean()),
                    "test_acc_std": float(np.std(acc, ddof=1)) if acc.size > 1 else 0.0,
                    "test_loss_std": float(np.std(loss, ddof=1)) if loss.size > 1 else 0.0,
                    "n_chips": len(group),
                    "status": "ok",
                }
            )
    else:
        raise ValueError(f"{source} does not look like mismatch summary or detail CSV")
    rows.sort(key=lambda row: float(row["noise_scale"]))
    if not rows:
        raise ValueError(f"no ok mismatch rows found in {source}")
    return rows


def _mm_write_filtered_csv(rows: list[dict[str, object]], out_path: Path) -> Path:
    fieldnames = [
        "noise_scale",
        "vt_sigma_mV",
        "beta_sigma_percent",
        "test_acc",
        "test_loss",
        "test_acc_std",
        "test_loss_std",
        "n_chips",
        "status",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out_path


def _mm_save_tight_target_size(
    fig: plt.Figure,
    out_path: Path,
    dpi: int,
    width_in: float,
    height_in: float,
    axes_rect: tuple[float, float, float, float],
) -> None:
    target_width_px = int(round(width_in * dpi))
    target_height_px = int(round(height_in * dpi))
    axes_aspect_height_per_width = axes_rect[2] / axes_rect[3]
    tmp_path = out_path.with_name(f".{out_path.stem}.tmp{out_path.suffix}")
    fig_width, _ = fig.get_size_inches()
    ax = fig.axes[0]
    labelpad = float(ax.xaxis.labelpad)
    for _ in range(16):
        fig.set_size_inches(fig_width, fig_width * axes_aspect_height_per_width, forward=True)
        ax.xaxis.labelpad = labelpad
        fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
        size = _png_size(tmp_path)
        if size is None or size[0] <= 0 or size[1] <= 0:
            break
        width_px, height_px = size
        width_done = width_px == target_width_px
        height_done = height_px == target_height_px
        if width_done and height_done:
            tmp_path.replace(out_path)
            return
        fig_width *= target_width_px / width_px
        labelpad += (target_height_px - height_px) * 72.0 / dpi
        labelpad = max(0.0, labelpad)
    fig.set_size_inches(fig_width, fig_width * axes_aspect_height_per_width, forward=True)
    ax.xaxis.labelpad = labelpad
    fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    tmp_path.replace(out_path)


def _plot_mismatch(rows: list[dict[str, object]], out_path: Path, dpi: int) -> Path:
    x = np.asarray([float(row["noise_scale"]) for row in rows], dtype=float)
    acc = np.asarray([float(row["test_acc"]) for row in rows], dtype=float)
    acc_std = np.asarray([float(row.get("test_acc_std", 0.0) or 0.0) for row in rows], dtype=float)
    loss = np.asarray([float(row["test_loss"]) for row in rows], dtype=float)
    loss_std = np.asarray([float(row.get("test_loss_std", 0.0) or 0.0) for row in rows], dtype=float)

    fig = plt.figure(figsize=(MM_FIG_WIDTH_IN, MM_FIG_HEIGHT_IN))
    ax_acc = fig.add_axes(MM_AXES_RECT)
    ax_loss = ax_acc.twinx()

    ax_acc.errorbar(
        x,
        acc,
        yerr=acc_std,
        color=ACC_COLOR,
        linewidth=LINEWIDTH,
        elinewidth=ERROR_LINEWIDTH,
        capsize=ERROR_CAPSIZE,
        capthick=ERROR_LINEWIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgewidth=0.0,
        alpha=CURVE_ALPHA,
        label="Acc",
    )
    ax_acc.axvline(
        1.0,
        color="#555555",
        linewidth=0.75,
        linestyle="--",
        zorder=0.5,
        label="Nominal Device\nMismatch",
    )
    ax_loss.errorbar(
        x,
        loss,
        yerr=loss_std,
        color=LOSS_COLOR,
        linewidth=LINEWIDTH,
        elinewidth=ERROR_LINEWIDTH,
        capsize=ERROR_CAPSIZE,
        capthick=ERROR_LINEWIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgewidth=0.0,
        alpha=CURVE_ALPHA,
        label="Loss",
    )

    ax_acc.set_xlabel("Mismatch Noise Scale")
    ax_acc.set_ylabel("Test Accuracy", color=ACC_COLOR)
    ax_loss.set_ylabel("Test Loss", color=LOSS_COLOR)
    ax_acc.yaxis.labelpad = 0.6
    ax_acc.set_xscale("log")
    ax_acc.set_xlim(float(np.nanmin(x)) / 1.15, float(np.nanmax(x)) * 1.15)
    ticks = _mm_scale_ticks(x)
    ax_acc.set_xticks(ticks)
    ax_acc.set_xticklabels([_mm_format_scale_tick(t) for t in ticks])
    acc_floor = float(np.nanmin(acc - acc_std)) - 0.02
    if acc_floor < 0.5:
        ax_acc.set_ylim(max(0.0, min(0.38, acc_floor)), 1.02)
        ax_acc.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    else:
        # data-driven tight limit: round the lowest error-bar down to the nearest 0.05
        lo = math.floor(acc_floor * 20.0) / 20.0
        ax_acc.set_ylim(lo, 1.005)
        ax_acc.set_yticks(np.round(np.arange(lo, 1.0001, 0.05), 2))

    loss_bottom, loss_top, loss_ticks = _mm_loss_axis(loss, loss_std)
    ax_loss.set_ylim(loss_bottom, loss_top)
    ax_loss.yaxis.set_major_formatter(FuncFormatter(_mm_format_e_tick))
    ax_loss.set_yticks(loss_ticks)

    handles, labels = [], []
    for axis in (ax_acc, ax_loss):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    by_label = dict(zip(labels, handles))
    labels = ["Acc", "Loss", "Nominal Device\nMismatch"]
    handles = [by_label[label] for label in labels]
    legend = ax_acc.legend(
        handles=handles,
        labels=labels,
        loc="center",
        bbox_to_anchor=(0.56, 0.58),
        frameon=False,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.0,
        labelspacing=0.35,
        ncol=1,
    )
    for text, color in zip(legend.get_texts(), (ACC_COLOR, LOSS_COLOR, "#555555")):
        text.set_fontsize(LEGEND_SIZE)
        text.set_color(color)

    _style_axis(ax_acc)
    ax_acc.spines["top"].set_position(("axes", 0.999))
    ax_acc.spines["right"].set_visible(False)
    _style_axis(ax_loss)
    ax_loss.spines["left"].set_visible(False)
    ax_loss.spines["bottom"].set_visible(False)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_loss.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True, colors=AXIS_COLOR)

    _mm_force_text_style(fig)
    ax_loss.yaxis.label.set_color(LOSS_COLOR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _mm_save_tight_target_size(fig, out_path, dpi, MM_FIG_WIDTH_IN, MM_TARGET_HEIGHT_IN, MM_AXES_RECT)
    plt.close(fig)
    return out_path


def render_panel_b(out_dir: Path, data_out_dir: Path, dpi: int) -> Path:
    """Panel b: accuracy + loss under device mismatch at inference."""
    rows = _mm_read_rows(MISMATCH_SOURCE)
    _mm_write_filtered_csv(rows, data_out_dir / "scikit_mismatch_noise_test_acc_loss.csv")
    return _plot_mismatch(rows, out_dir / "fig5_b.png", dpi)


# ===========================================================================
# Panels c, d, e, f  --  hyperparameter / design sweeps
# (from plot_fig5_hparam_sweeps.py)
#   c = loss-comparison bars, d = device-mode matrix,
#   e = init heatmap, f = gamma x delta heatmap
# ===========================================================================
HATCH_EDGE = "#6C7075"
FAIL_FACE = "#F3F4F4"
CHOSEN_BORDER_COLOR = "#111111"
CHOSEN_BORDER_WIDTH = 1.6
CHOSEN_BORDER_INSET_FRAC = 0.0
ACCURACY_CMAP = "magma"
LOSS_CMAP = "magma_r"
DEVICE_CMAP = "gray"
DEFAULT_NORM = colors.Normalize(vmin=0.0, vmax=1.0)
# Internal dict-key labels for the accuracy/loss metric carried through the
# plotting helpers. The actual value read from every real CSV is the
# ``endpoint_test_acc`` column (see ``_hp_load_real_data``); these constants are
# only in-memory labels, so their string value is cosmetic.
ACCURACY_KEY = "endpoint_test_acc"
LOSS_KEY = "endpoint_test_loss"
STD_ACCURACY_KEY = "std_test_acc"
STD_LOSS_KEY = "std_test_loss"
ACCURACY_LABEL = "Test Accuracy (endpoint)"
LOSS_LABEL = "Test Loss (endpoint)"
SWEEP_PANEL_WIDTH_IN = 1.98
LOSS_PANEL_WIDTH_IN = 1.80
LOSS_AXES_RECT = (0.23, 0.16, 0.70, 0.80)
LOSS_PANEL_HEIGHT_IN = 1.85
DEVICE_PANEL_WIDTH_IN = 2.45
DEVICE_PANEL_HEIGHT_IN = 1.933
INIT_COLORBAR_PANEL_HEIGHT_IN = 2.08
GAMMA_COLORBAR_PANEL_HEIGHT_IN = INIT_COLORBAR_PANEL_HEIGHT_IN

BODY_ROWS = ["ground", "drain", "source", "floating"]
GATE_COLS = ["ground", "drain", "source", "body"]
INIT_MEANS = np.asarray([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], dtype=float)
INIT_HALF_SPREADS = np.asarray([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5], dtype=float)
GAMMAS = np.asarray([0.03, 0.1, 0.3, 1, 3, 10, 30], dtype=float)
DELTAS = np.asarray([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], dtype=float)
MARGINS = np.asarray([0.02, 0.05, 0.1], dtype=float)
TASKS = ["scikit", "ionosphere"]
LOSS_PLOT_TASKS = ["ionosphere", "scikit"]
LOSSES = ["MSE", "cross_entropy", "hinge"]
LOSS_BAR_COLORS = {
    "ionosphere": {
        "cross_entropy": "#C7D3E6",
        "MSE": "#7F98BF",
        "hinge": "#3F5F8F",
    },
    "scikit": {
        "cross_entropy": "#E8C7A7",
        "MSE": "#C99058",
        "hinge": "#8E5A2F",
    },
}
OK_STATUSES = {"", "ok", "complete", "completed", "success"}
CHOSEN_MARGIN = 0.02


def _status_ok(status: str) -> bool:
    return str(status).strip().lower() in OK_STATUSES


def _compact_float_label(value: float) -> str:
    text = f"{value:g}"
    if text.startswith("0."):
        return text[1:]
    return text


def _nice_tick_step(raw_step: float) -> float:
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    base = 10.0**exponent
    fraction = raw_step / base
    for nice_fraction in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= nice_fraction:
            return nice_fraction * base
    return 10.0 * base


def _nice_colorbar_ticks(vmin: float, vmax: float, count: int = 5) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        return np.linspace(0.0, 1.0, count)
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    span = vmax - vmin
    raw_step = span / float(count - 1)
    exponent = math.floor(math.log10(raw_step))
    candidate_steps = sorted(
        {
            nice_fraction * 10.0**candidate_exponent
            for candidate_exponent in range(exponent - 1, exponent + 2)
            for nice_fraction in (1.0, 2.0, 2.5, 5.0, 10.0)
        }
    )
    inside_candidates: list[tuple[float, float, float]] = []
    for step_candidate in candidate_steps:
        start_candidate = math.ceil((vmin - 1e-12) / step_candidate) * step_candidate
        end_candidate = start_candidate + step_candidate * (count - 1)
        if end_candidate <= vmax + 1e-12:
            inside_candidates.append((end_candidate - start_candidate, step_candidate, start_candidate))
    if inside_candidates:
        coverage, step, start = max(inside_candidates, key=lambda item: (item[0], item[1]))
        if coverage >= 0.65 * span:
            decimals = max(0, int(math.ceil(-math.log10(step))) + 1)
            return np.round(start + step * np.arange(count), decimals)

    step = _nice_tick_step(raw_step)
    start = math.floor(vmin / step) * step
    decimals = max(0, int(math.ceil(-math.log10(step))) + 1)
    return np.round(start + step * np.arange(count), decimals)


def _colorbar_tick_label(value: float, _pos: int | None = None) -> str:
    return f"{value:g}"


def _clean_loss_name(loss: str) -> str:
    value = str(loss).strip()
    low = value.lower()
    if low in {"mse", "mean_squared_error"}:
        return "MSE"
    if low in {"ce", "xent", "cross_entropy", "cross entropy"}:
        return "cross_entropy"
    if low == "hinge":
        return "hinge"
    return value


def _display_name(value: str) -> str:
    return str(value).replace("_", " ").title()


def _display_loss_name(loss: str) -> str:
    clean = _clean_loss_name(loss)
    if clean == "MSE":
        return "MSE"
    if clean == "cross_entropy":
        return "CE"
    if clean == "hinge":
        return "Hinge"
    return _display_name(clean)


def _display_task_name(task: str) -> str:
    if str(task).strip().lower() == "scikit":
        return "Digits"
    return _display_name(task)


def _display_metric_label(metric_label: str) -> str:
    if metric_label == ACCURACY_LABEL:
        return "Test Accuracy"
    if metric_label == LOSS_LABEL:
        return "Test Loss"
    return metric_label


def _float_cell(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def _first_existing(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value is not None and str(value).strip() != "":
            return str(value)
    return ""


def _float_first(row: dict[str, str], keys: tuple[str, ...]) -> float:
    value = _first_existing(row, keys)
    if not value:
        return float("nan")
    return float(value)


def _accuracy_fraction(value: float) -> float:
    if math.isfinite(value) and value > 1.5:
        return value / 100.0
    return value


def _accuracy_decimal_label(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{_accuracy_fraction(value):.3f}"


def _intish_cell(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _test_loss_from_row(row: dict[str, str], acc: float) -> float:
    value = _float_first(row, ("test_loss", "clean_loss"))
    if math.isfinite(value):
        return value
    if math.isfinite(acc):
        return 1.0 - acc
    return float("nan")


def _std_test_loss_from_row(row: dict[str, str], std_acc: float) -> float:
    value = _float_first(row, ("std_test_loss", "std_clean_loss"))
    if math.isfinite(value):
        return value
    return std_acc if math.isfinite(std_acc) else float("nan")


def _read_csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return [dict(row) for row in reader]


def _hp_sweep_files(data_dir: Path) -> dict[str, tuple[Path, set[str]]]:
    hyper = data_dir / "hyperparam"
    loss = data_dir / "loss"
    gamma_path = hyper / "gamma_delta_m0.02.csv"
    if not gamma_path.exists():
        gamma_path = hyper / "gamma_delta_margin.csv"
    return {
        "device": (
            hyper / "device_matrix.csv",
            {"body_tie", "gate_ref", "endpoint_test_acc", "status"},
        ),
        "init": (
            hyper / "init_heatmap.csv",
            {"init_mean", "init_half_spread", "endpoint_test_acc", "clip_fraction", "status"},
        ),
        "gamma": (
            gamma_path,
            {"margin", "gamma", "delta", "endpoint_test_acc", "status"},
        ),
        "loss": (
            loss / "loss_comparison.csv",
            {"task", "loss", "endpoint_test_acc"},
        ),
    }


def _hp_load_real_data(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    files = _hp_sweep_files(data_dir)
    missing_files = [str(path) for path, _ in files.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError("missing CSVs: " + ", ".join(missing_files))

    device = []
    for row in _read_csv_rows(*files["device"]):
        acc = _float_first(row, ("endpoint_test_acc",))
        device.append(
            {
                "body_tie": row["body_tie"].strip(),
                "gate_ref": row["gate_ref"].strip(),
                ACCURACY_KEY: acc,
                LOSS_KEY: _test_loss_from_row(row, acc),
                "status": row.get("status", "ok").strip(),
            }
        )
    init = []
    for row in _read_csv_rows(*files["init"]):
        acc = _float_first(row, ("endpoint_test_acc",))
        init.append(
            {
                "init_mean": _float_cell(row, "init_mean"),
                "init_half_spread": _float_cell(row, "init_half_spread"),
                ACCURACY_KEY: acc,
                LOSS_KEY: _test_loss_from_row(row, acc),
                "clip_fraction": _float_cell(row, "clip_fraction"),
                "status": row.get("status", "ok").strip(),
                "is_reference": row.get("is_reference", "").strip(),
            }
        )
    gamma = []
    for row in _read_csv_rows(*files["gamma"]):
        acc = _float_first(row, ("endpoint_test_acc",))
        gamma.append(
            {
                "margin": _float_cell(row, "margin"),
                "gamma": _float_cell(row, "gamma"),
                "delta": _float_cell(row, "delta"),
                ACCURACY_KEY: acc,
                LOSS_KEY: _test_loss_from_row(row, acc),
                "status": row.get("status", "ok").strip(),
            }
        )
    loss = []
    for row in _read_csv_rows(*files["loss"]):
        raw_acc = _float_first(row, ("endpoint_test_acc",))
        acc = _accuracy_fraction(raw_acc)
        std_acc = _float_first(row, ("std_test_acc",))
        loss.append(
            {
                "task": row["task"].strip(),
                "loss": _clean_loss_name(row["loss"]),
                ACCURACY_KEY: acc,
                "endpoint_test_acc_label": _accuracy_decimal_label(raw_acc),
                LOSS_KEY: _test_loss_from_row(row, acc),
                STD_ACCURACY_KEY: std_acc,
                STD_LOSS_KEY: _std_test_loss_from_row(row, std_acc),
                "n_trials": _intish_cell(row, "n_trials"),
                "status": row.get("status", "ok").strip(),
            }
        )
    return {"device": device, "init": init, "gamma": gamma, "loss": loss}


def _matrix_from_records(
    rows: list[dict[str, Any]],
    row_key: str,
    col_key: str,
    row_order: list[str] | np.ndarray,
    col_order: list[str] | np.ndarray,
    value_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    row_labels = [str(v) for v in row_order]
    col_labels = [str(v) for v in col_order]
    values = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)
    status = np.full((len(row_labels), len(col_labels)), "missing", dtype=object)
    row_index = {str(v): i for i, v in enumerate(row_labels)}
    col_index = {str(v): i for i, v in enumerate(col_labels)}
    for row in rows:
        r = str(row[row_key])
        c = str(row[col_key])
        if r not in row_index or c not in col_index:
            continue
        i = row_index[r]
        j = col_index[c]
        values[i, j] = float(row.get(value_key, np.nan))
        status[i, j] = str(row.get("status", "ok"))
    return values, status


def _grid_from_numeric_records(
    rows: list[dict[str, Any]],
    y_key: str,
    x_key: str,
    y_values: np.ndarray,
    x_values: np.ndarray,
    value_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    status = np.full((len(y_values), len(x_values)), "missing", dtype=object)
    for row in rows:
        y = float(row[y_key])
        x = float(row[x_key])
        y_matches = np.flatnonzero(np.isclose(y_values, y, rtol=1e-9, atol=1e-12))
        x_matches = np.flatnonzero(np.isclose(x_values, x, rtol=1e-9, atol=1e-12))
        if y_matches.size == 0 or x_matches.size == 0:
            continue
        i = int(y_matches[0])
        j = int(x_matches[0])
        values[i, j] = float(row.get(value_key, np.nan))
        status[i, j] = str(row.get("status", "ok"))
    return values, status


def _linear_edges(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 1:
        step = 1.0
        return np.asarray([vals[0] - step / 2, vals[0] + step / 2], dtype=float)
    mids = (vals[:-1] + vals[1:]) / 2.0
    first = vals[0] - (mids[0] - vals[0])
    last = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _log_edges(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 1:
        return np.asarray([vals[0] / math.sqrt(10.0), vals[0] * math.sqrt(10.0)], dtype=float)
    mids = np.sqrt(vals[:-1] * vals[1:])
    first = vals[0] ** 2 / mids[0]
    last = vals[-1] ** 2 / mids[-1]
    return np.concatenate([[first], mids, [last]])


def _norm_from_values(values: np.ndarray | list[float]) -> colors.Normalize:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return DEFAULT_NORM
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
        pad = max(0.005, abs(vmin) * 0.01)
        vmin -= pad
        vmax += pad
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _annotate_heatmap_values(
    ax: plt.Axes,
    values: np.ndarray,
    norm: colors.Normalize,
    value_format: str,
    cmap_name: str,
) -> None:
    cmap = plt.get_cmap(cmap_name)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "Fail", ha="center", va="center", fontsize=SMALL_SIZE, color=AXIS_COLOR)
                continue
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "#000000" if luminance > 0.56 else "#FFFFFF"
            ax.text(
                j,
                i,
                value_format.format(value),
                ha="center",
                va="center",
                fontsize=TEXT_SIZE,
                color=text_color,
            )


def _draw_plain_matrix_cells(ax: plt.Axes, values: np.ndarray, value_format: str) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor="#FFFFFF",
                    edgecolor="none",
                    zorder=0,
                )
            )
            value = values[i, j]
            label = value_format.format(value) if np.isfinite(value) else "Fail"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=TEXT_SIZE,
                color=AXIS_COLOR,
                zorder=4,
            )


def _draw_matrix_grid(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    x0 = -0.5
    x1 = n_cols - 0.5
    y0 = -0.5
    y1 = n_rows - 0.5
    inset = 0.014
    line_kwargs = {
        "color": "#202428",
        "linewidth": 0.55,
        "solid_capstyle": "butt",
        "clip_on": False,
        "zorder": 5,
    }
    for x in np.arange(0.5, n_cols - 0.5, 1.0):
        ax.plot([x, x], [y0 + inset, y1 - inset], **line_kwargs)
    for y in np.arange(0.5, n_rows - 0.5, 1.0):
        ax.plot([x0 + inset, x1 - inset], [y, y], **line_kwargs)
    ax.plot([x0 + inset, x1 - inset], [y0 + inset, y0 + inset], **line_kwargs)
    ax.plot([x0 + inset, x1 - inset], [y1 - inset, y1 - inset], **line_kwargs)
    ax.plot([x0 + inset, x0 + inset], [y0 + inset, y1 - inset], **line_kwargs)
    ax.plot([x1 - inset, x1 - inset], [y0 + inset, y1 - inset], **line_kwargs)


def _hatch_mesh_failures(
    ax: plt.Axes,
    status: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    for i in range(status.shape[0]):
        for j in range(status.shape[1]):
            if _status_ok(str(status[i, j])):
                continue
            ax.add_patch(
                Rectangle(
                    (x_edges[j], y_edges[i]),
                    x_edges[j + 1] - x_edges[j],
                    y_edges[i + 1] - y_edges[i],
                    facecolor=FAIL_FACE,
                    edgecolor=HATCH_EDGE,
                    linewidth=0.25,
                    hatch="//////",
                    zorder=5,
                )
            )


def _add_colorbar(
    fig: plt.Figure,
    mappable: Any,
    ax: Any,
    label: str,
    *,
    cax: plt.Axes | None = None,
    orientation: str = "vertical",
) -> None:
    if cax is not None:
        cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
    else:
        cb = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.025, orientation=orientation)
    if label:
        cb.set_label(label, fontsize=TEXT_SIZE, labelpad=1.0)
    if orientation == "horizontal":
        cb.set_ticks(_nice_colorbar_ticks(float(mappable.norm.vmin), float(mappable.norm.vmax), count=5))
        cb.ax.xaxis.set_major_formatter(ticker.FuncFormatter(_colorbar_tick_label))
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(
            axis="x",
            labelsize=TICK_SIZE,
            width=0.45,
            length=2.0,
            pad=1.0,
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
        )
    else:
        cb.ax.tick_params(labelsize=TICK_SIZE, width=0.45, length=2.0, pad=1.5)
    cb.outline.set_linewidth(0.55)


def _hp_force_figure_text_style(fig: plt.Figure) -> None:
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontfamily(FONT_FAMILY[0])
        text.set_fontsize(TEXT_SIZE)


def _hp_save(fig: plt.Figure, out_prefix: Path, dpi: int, *, tight: bool = True) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.0} if tight else {}
    _hp_force_figure_text_style(fig)
    fig.savefig(png, dpi=dpi, **save_kwargs)
    plt.close(fig)
    return png


def _hp_save_tight_exact(fig: plt.Figure, out_prefix: Path, dpi: int, width_in: float, height_in: float) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    tmp = out_prefix.with_name(f".{out_prefix.name}.tmp").with_suffix(".png")
    target_px = (int(round(width_in * dpi)), int(round(height_in * dpi)))
    _hp_force_figure_text_style(fig)
    for _ in range(8):
        fig.savefig(tmp, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
        size = _png_size(tmp)
        if size == target_px:
            tmp.replace(png)
            plt.close(fig)
            return png
        if size is None or size[0] <= 0 or size[1] <= 0:
            break
        cur_w, cur_h = fig.get_size_inches()
        fig.set_size_inches(cur_w * target_px[0] / size[0], cur_h * target_px[1] / size[1], forward=True)
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    tmp.replace(png)
    plt.close(fig)
    return png


def _plot_device_panel(
    rows: list[dict[str, Any]],
    out_prefix: Path,
    dpi: int,
    *,
    ax: plt.Axes | None = None,
    metric_key: str = ACCURACY_KEY,
    cmap_name: str = DEVICE_CMAP,
) -> Path | None:
    values, status = _matrix_from_records(rows, "body_tie", "gate_ref", BODY_ROWS, GATE_COLS, metric_key)
    own_fig = ax is None
    fig = plt.figure(figsize=(DEVICE_PANEL_WIDTH_IN, DEVICE_PANEL_HEIGHT_IN)) if own_fig else ax.figure
    if ax is None:
        ax = fig.add_axes((0.28, 0.36, 0.70, 0.60))

    ax.set_facecolor("#FFFFFF")
    _draw_plain_matrix_cells(ax, values, "{:.3f}")
    ax.set_xticks(np.arange(len(GATE_COLS)))
    ax.set_xticklabels([_display_name(v) for v in GATE_COLS], rotation=0, ha="center")
    ax.set_yticks(np.arange(len(BODY_ROWS)))
    ax.set_yticklabels([_display_name(v) for v in BODY_ROWS])
    ax.set_xlabel("Gate Reference", labelpad=0.5)
    ax.set_ylabel("Body Tie", labelpad=0.5)
    ax.set_xlim(-0.5, len(GATE_COLS) - 0.5)
    ax.set_ylim(len(BODY_ROWS) - 0.5, -0.5)
    _style_axis(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _draw_matrix_grid(ax, len(BODY_ROWS), len(GATE_COLS))
    if own_fig:
        return _hp_save(fig, out_prefix, dpi, tight=True)
    return None


def _plot_init_panel(
    rows: list[dict[str, Any]],
    out_prefix: Path,
    dpi: int,
    *,
    ax: plt.Axes | None = None,
    metric_key: str = ACCURACY_KEY,
    metric_label: str = ACCURACY_LABEL,
    cmap_name: str = ACCURACY_CMAP,
) -> Path | None:
    values, status = _grid_from_numeric_records(
        rows,
        "init_half_spread",
        "init_mean",
        INIT_HALF_SPREADS,
        INIT_MEANS,
        metric_key,
    )

    own_fig = ax is None
    cax = None
    if own_fig:
        fig, (cax, ax) = plt.subplots(
            2,
            1,
            figsize=(SWEEP_PANEL_WIDTH_IN, INIT_COLORBAR_PANEL_HEIGHT_IN),
            gridspec_kw={"height_ratios": [0.055, 1.0]},
        )
        fig.subplots_adjust(left=0.17, right=0.93, bottom=0.22, top=0.90, hspace=0.10)
    else:
        fig = ax.figure

    x_edges = _linear_edges(INIT_MEANS)
    y_edges = _linear_edges(INIT_HALF_SPREADS)
    if metric_key == ACCURACY_KEY:
        norm = colors.Normalize(vmin=0.94, vmax=float(np.nanmax(values)))
    else:
        norm = _norm_from_values(values)
    im = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values), cmap=cmap_name, norm=norm, shading="flat")
    _hatch_mesh_failures(ax, status, x_edges, y_edges)
    ax.set_xlabel("Initialization Mean [V]")
    ax.set_ylabel("Half-Width [V]")
    ax.set_xticks(INIT_MEANS)
    ax.set_xticklabels([_compact_float_label(v) for v in INIT_MEANS], rotation=45, ha="right")
    ax.set_yticks(INIT_HALF_SPREADS)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    _style_axis(ax)

    if own_fig:
        _add_colorbar(fig, im, ax, "", cax=cax, orientation="horizontal")
        return _hp_save_tight_exact(fig, out_prefix, dpi, SWEEP_PANEL_WIDTH_IN, INIT_COLORBAR_PANEL_HEIGHT_IN)
    return None


def _plot_gamma_delta_margin_panel(
    rows: list[dict[str, Any]],
    out_prefix: Path,
    dpi: int,
    *,
    ax: plt.Axes | None = None,
    metric_key: str = ACCURACY_KEY,
    metric_label: str = ACCURACY_LABEL,
    cmap_name: str = ACCURACY_CMAP,
    lower_is_better: bool = False,
) -> Path | None:
    own_fig = ax is None
    if own_fig:
        fig, (cax, ax) = plt.subplots(
            2,
            1,
            figsize=(SWEEP_PANEL_WIDTH_IN, GAMMA_COLORBAR_PANEL_HEIGHT_IN),
            gridspec_kw={"height_ratios": [0.055, 1.0]},
        )
        fig.subplots_adjust(left=0.19, right=0.93, bottom=0.22, top=0.90, hspace=0.10)
    else:
        fig = ax.figure
        cax = None

    x_edges = _log_edges(DELTAS)
    y_edges = _log_edges(GAMMAS)
    margin_rows = [row for row in rows if np.isclose(float(row["margin"]), CHOSEN_MARGIN)]
    values, status = _grid_from_numeric_records(margin_rows, "gamma", "delta", GAMMAS, DELTAS, metric_key)
    if metric_key == ACCURACY_KEY and not lower_is_better:
        gamma_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    else:
        gamma_norm = _norm_from_values(values)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(values),
        cmap=cmap_name,
        norm=gamma_norm,
        shading="flat",
    )
    _hatch_mesh_failures(ax, status, x_edges, y_edges)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xlabel("Clamp Amplitude [V]")
    ax.set_ylabel("Learning Rate")
    ax.set_xticks(DELTAS)
    ax.set_xticklabels([_compact_float_label(v) for v in DELTAS], rotation=35, ha="right")
    ax.set_yticks(GAMMAS)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _pos: _compact_float_label(value)))
    _style_axis(ax)
    if own_fig:
        _add_colorbar(fig, im, ax, "", cax=cax, orientation="horizontal")
        return _hp_save_tight_exact(fig, out_prefix, dpi, SWEEP_PANEL_WIDTH_IN, GAMMA_COLORBAR_PANEL_HEIGHT_IN)
    return None


def _plot_loss_bars(
    rows: list[dict[str, Any]],
    out_prefix: Path,
    dpi: int,
    *,
    ax: plt.Axes | None = None,
    metric_key: str = ACCURACY_KEY,
    metric_label: str = ACCURACY_LABEL,
    std_key: str = STD_ACCURACY_KEY,
    value_format: str = "{:.3f}",
    lower_is_better: bool = False,
) -> Path | None:
    own_fig = ax is None
    fig = plt.figure(figsize=(LOSS_PANEL_WIDTH_IN, LOSS_PANEL_HEIGHT_IN)) if own_fig else ax.figure
    if ax is None:
        ax = fig.add_axes(LOSS_AXES_RECT)

    y_positions = []
    labels = []
    values = []
    errors = []
    value_labels = []
    colors_list = []
    value_text_colors = []
    hatches = []
    group_bases = [0.95, 4.65]
    loss_offsets = [0.0, 1.15, 2.30]
    for task_idx, task in enumerate(LOSS_PLOT_TASKS):
        base = group_bases[task_idx]
        for loss_idx, loss_name in enumerate(LOSSES):
            match = [
                row
                for row in rows
                if str(row.get("task", "")).strip() == task
                and _clean_loss_name(str(row.get("loss", ""))) == loss_name
            ]
            row = match[0] if match else {}
            y_positions.append(base + loss_offsets[loss_idx])
            labels.append(_display_loss_name(loss_name))
            values.append(float(row.get(metric_key, np.nan)))
            errors.append(float(row.get(std_key, 0.0)))
            if metric_key == ACCURACY_KEY:
                value_labels.append(str(row.get("endpoint_test_acc_label", "")))
            else:
                value_labels.append("")
            colors_list.append(LOSS_BAR_COLORS.get(task, {}).get(loss_name, "#808080"))
            value_text_colors.append("#FFFFFF" if loss_idx == len(LOSSES) - 1 else "#000000")
            hatches.append("" if _status_ok(str(row.get("status", "missing"))) else "////")

    bars = ax.barh(
        y_positions,
        np.nan_to_num(values, nan=0.0),
        height=0.72,
        color=colors_list,
        edgecolor="#222222",
        linewidth=0.35,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if lower_is_better and finite_values.size:
        x_min = 0.0
        x_max = max(0.02, float(np.max(finite_values)) * 1.25)
    else:
        if finite_values.size and float(np.min(finite_values)) < 0.8:
            x_min = max(0.0, math.floor((float(np.min(finite_values)) - 0.10) * 10.0) / 10.0)
        else:
            x_min = 0.8
        x_max = 1.0
    ax.set_xlim(x_min, x_max)
    for y, value, label_text, text_color in zip(y_positions, values, value_labels, value_text_colors):
        if np.isfinite(value):
            offset = 0.012 * (x_max - x_min)
            if not label_text:
                label_text = value_format.format(value)
            ax.text(
                max(value - offset, x_min + 0.002 * (x_max - x_min)),
                y,
                label_text,
                ha="right",
                va="center",
                fontsize=SMALL_SIZE,
                color=text_color,
            )
        else:
            ax.text(x_min, y, "Fail", ha="left", va="center", fontsize=SMALL_SIZE)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(_display_metric_label(metric_label), labelpad=1.5)
    if lower_is_better and finite_values.size:
        pass
    else:
        if x_min < 0.8:
            ticks = [x_min] + [tick for tick in (0.4, 0.6, 0.8, 1.0) if tick > x_min]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])
        else:
            ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
            ax.set_xticklabels(["0.80", "0.85", "0.90", "0.95", "1.0"])
    ax.set_ylim(7.65, 0.25)
    ax.axhline(3.95, color=AXIS_COLOR, linewidth=0.55)
    _style_axis(ax)
    ax.spines["top"].set_position(("axes", 0.998))

    if own_fig:
        return _hp_save_tight_exact(fig, out_prefix, dpi, LOSS_PANEL_WIDTH_IN, LOSS_PANEL_HEIGHT_IN)
    return None


def render_panels_cdef(out_dir: Path, dpi: int) -> list[Path]:
    """Panels c-f: loss comparison, device modes, init heatmap, gamma x delta."""
    data = _hp_load_real_data(DATA_DIR)
    outputs = [
        _plot_loss_bars(data["loss"], out_dir / "fig5_c", dpi),
        _plot_device_panel(data["device"], out_dir / "fig5_d", dpi),
        _plot_init_panel(data["init"], out_dir / "fig5_e", dpi),
        _plot_gamma_delta_margin_panel(data["gamma"], out_dir / "fig5_f", dpi),
    ]
    return [p for p in outputs if p is not None]


# ===========================================================================
# Dimensions table + driver
# ===========================================================================
def _write_dimensions(out_dir: Path, dpi: int) -> Path:
    lines = [
        "# Figure Dimensions",
        "",
        f"Dimensions are the exported PNG physical sizes from the {dpi} dpi files.",
        "",
        "| File | Size (in) | Pixels |",
        "|---|---:|---:|",
    ]
    for path in sorted(out_dir.glob("fig5_*.png"), key=lambda p: p.name):
        size = _png_size(path)
        if size is None:
            continue
        width_px, height_px = size
        lines.append(
            f"| `{path.name}` | "
            f"{width_px / dpi:.2f} x {height_px / dpi:.2f} | {width_px} x {height_px} |"
        )
    out_path = out_dir / "FIGURE_DIMENSIONS.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render all six panels of Figure 5.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--data-out-dir", type=Path, default=DEFAULT_DATA_OUT_DIR)
    p.add_argument("--png-dpi", type=int, default=PNG_DPI)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _set_rc()

    out_dir = args.out_dir
    data_out_dir = args.data_out_dir
    dpi = int(args.png_dpi)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    outputs.append(render_panel_a(out_dir, data_out_dir, dpi))
    outputs.append(render_panel_b(out_dir, data_out_dir, dpi))
    outputs.extend(render_panels_cdef(out_dir, dpi))

    dimensions = _write_dimensions(out_dir, dpi)
    print(f"rendered {len(outputs)} Figure 5 panels")
    for path in outputs:
        print(f"saved {path}")
    print(f"wrote {dimensions}")


if __name__ == "__main__":
    main()
