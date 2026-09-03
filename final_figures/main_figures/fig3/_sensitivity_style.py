"""Shared scikit-digits sensitivity-field rendering.

Rendering only.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

DEFAULT_EVIDENCE_VLIM = 0.020
DEFAULT_EVIDENCE_SATURATION_VLIM = 0.016
DEFAULT_SIGNED_SATURATION_VLIM = 0.032
FIGURE_WIDTH_IN = 2.81
FIGURE_HEIGHT_IN = 1.31
PNG_DPI = 600
GRID_LEFT = 0.001
GRID_RIGHT = 0.999
GRID_BOTTOM = 0.004
GRID_TOP = 0.996
GRID_WSPACE = 0.06
GRID_HSPACE = 0.10
MAP_SPINE_LINEWIDTH = 0.55
COLORBAR_WIDTH_IN = 0.30
COLORBAR_HEIGHT_IN = 1.31
COLORBAR_BODY_HEIGHT_IN = 1.20
COLORBAR_THICKNESS_IN = 0.041
TEXT = "#34383D"
NEUTRAL = "#F3F4F4"
DIVERGING_PALETTE = ["#167A5B", NEUTRAL, "#6B2E83"]
MAGNITUDE_PALETTE = [NEUTRAL, "#3B9B78"]


class AsymmetricDivergingPowerNorm(colors.Normalize):
    """Diverging norm with a steeper near-zero visual slope on each side."""

    def __init__(self, vmin: float, vmax: float, gamma: float):
        if not (vmin < 0.0 < vmax):
            raise ValueError("AsymmetricDivergingPowerNorm requires vmin < 0 < vmax")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        super().__init__(vmin=vmin, vmax=vmax, clip=True)
        self.gamma = float(gamma)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        data = np.asarray(result.data, dtype=float)
        out = np.empty_like(data, dtype=float)
        neg = data < 0.0
        out[neg] = 0.5 - 0.5 * np.power(np.clip((-data[neg]) / abs(self.vmin), 0.0, 1.0), self.gamma)
        out[~neg] = 0.5 + 0.5 * np.power(np.clip(data[~neg] / self.vmax, 0.0, 1.0), self.gamma)
        out = np.ma.array(out, mask=result.mask, copy=False)
        if is_scalar:
            out = out[0]
        return out

    def inverse(self, value):
        value = np.asarray(value, dtype=float)
        out = np.empty_like(value, dtype=float)
        neg = value < 0.5
        out[neg] = -abs(self.vmin) * np.power(np.clip((0.5 - value[neg]) / 0.5, 0.0, 1.0), 1.0 / self.gamma)
        out[~neg] = self.vmax * np.power(np.clip((value[~neg] - 0.5) / 0.5, 0.0, 1.0), 1.0 / self.gamma)
        return out


def _load_final_state(path: Path, npz_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"drain", "source", npz_key}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    return (
        np.asarray(data["drain"], dtype=int).reshape(-1),
        np.asarray(data["source"], dtype=int).reshape(-1),
        np.asarray(data[npz_key], dtype=float).reshape(-1),
    )


def _test_split() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(float)
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    return Xte, yte.astype(int)


def _auto_abs_limit(values: np.ndarray) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return 1.0
    max_abs = float(np.max(np.abs(finite)))
    if max_abs <= 0.0:
        return 1.0
    return float(np.ceil(max_abs * 1000.0) / 1000.0)


def _saturated_diverging_cmap(display_vlim: float, saturation_vlim: float) -> colors.LinearSegmentedColormap:
    display_vlim = float(abs(display_vlim))
    saturation_vlim = float(abs(saturation_vlim))
    if display_vlim <= 0.0 or saturation_vlim >= display_vlim:
        return colors.LinearSegmentedColormap.from_list("sensitivity_diverging", DIVERGING_PALETTE, N=256)

    saturation_vlim = max(saturation_vlim, 1e-12)
    low_stop = 0.5 - 0.5 * saturation_vlim / display_vlim
    high_start = 0.5 + 0.5 * saturation_vlim / display_vlim
    return colors.LinearSegmentedColormap.from_list(
        "sensitivity_diverging_saturated",
        [
            (0.0, DIVERGING_PALETTE[0]),
            (low_stop, DIVERGING_PALETTE[0]),
            (0.5, DIVERGING_PALETTE[1]),
            (high_start, DIVERGING_PALETTE[2]),
            (1.0, DIVERGING_PALETTE[2]),
        ],
        N=256,
    )


def _cmap_norm(
    kind: str,
    vmin: float,
    vmax: float,
    saturation_vlim: float | None,
    contrast_gamma: float | None = None,
) -> tuple[colors.Colormap, colors.Normalize]:
    if kind == "magnitude":
        return (
            colors.LinearSegmentedColormap.from_list("sensitivity_magnitude", MAGNITUDE_PALETTE, N=256),
            colors.Normalize(vmin=vmin, vmax=vmax),
        )
    if contrast_gamma is not None:
        return (
            colors.LinearSegmentedColormap.from_list("sensitivity_diverging", DIVERGING_PALETTE, N=256),
            AsymmetricDivergingPowerNorm(vmin=vmin, vmax=vmax, gamma=float(contrast_gamma)),
        )
    if saturation_vlim is not None:
        display_vlim = max(abs(vmin), abs(vmax))
        return (
            _saturated_diverging_cmap(display_vlim, saturation_vlim),
            colors.TwoSlopeNorm(vmin=-display_vlim, vcenter=0.0, vmax=display_vlim),
        )
    return (
        colors.LinearSegmentedColormap.from_list("sensitivity_diverging", DIVERGING_PALETTE, N=256),
        colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
    )


def _draw_maps(
    maps: np.ndarray,
    out_prefix: Path,
    *,
    kind: str,
    vmin: float,
    vmax: float,
    dpi: int,
    saturation_vlim: float | None = None,
    contrast_gamma: float | None = None,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Open Sans", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
        }
    )

    cmap, norm = _cmap_norm(kind, vmin, vmax, saturation_vlim, contrast_gamma=contrast_gamma)

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    grid = fig.add_gridspec(
        2,
        5,
        hspace=GRID_HSPACE,
        wspace=GRID_WSPACE,
        left=GRID_LEFT,
        right=GRID_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
    )

    image = None
    for digit in range(10):
        ax = fig.add_subplot(grid[digit // 5, digit % 5])
        image = ax.imshow(maps[digit].reshape(8, 8), cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(MAP_SPINE_LINEWIDTH)
            spine.set_edgecolor(TEXT)

    if image is None:
        raise RuntimeError("No sensitivity maps were drawn")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def _draw_colorbar(
    out_prefix: Path,
    *,
    kind: str,
    vmin: float,
    vmax: float,
    dpi: int,
    saturation_vlim: float | None = None,
    contrast_gamma: float | None = None,
) -> Path:
    cmap, norm = _cmap_norm(kind, vmin, vmax, saturation_vlim, contrast_gamma=contrast_gamma)
    fig = plt.figure(figsize=(COLORBAR_WIDTH_IN, COLORBAR_HEIGHT_IN))
    cax = fig.add_axes(
        [
            0.18,
            0.5 * (1.0 - COLORBAR_BODY_HEIGHT_IN / COLORBAR_HEIGHT_IN),
            COLORBAR_THICKNESS_IN / COLORBAR_WIDTH_IN,
            COLORBAR_BODY_HEIGHT_IN / COLORBAR_HEIGHT_IN,
        ]
    )
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_ticks([])
    cb.ax.tick_params(width=0.0, length=0.0, pad=0.0, colors=TEXT)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor(TEXT)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png
