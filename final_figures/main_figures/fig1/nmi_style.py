"""Shared NMI figure style for the pyCLLN manuscript (figures_nmi).

Single source of truth for fonts, sizes, palette, and save conventions.
Every make_figN.py imports this FIRST (it sets the matplotlib backend).
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- fonts ------------------------------------------------------------
_FONT_DIRS = [Path.home() / ".local/share/fonts"]
for _d in _FONT_DIRS:
    if _d.is_dir():
        for _f in _d.glob("*.[ot]tf"):
            try:
                font_manager.fontManager.addfont(str(_f))
            except Exception:
                pass

FONT_STACK = ["Open Sans", "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

# ---- Nature-standard font sizes (final size; NMI allows 5-7 pt text,
#      8 pt bold panel letters) ------------------------------------------
FS_PANEL = 8      # panel letters (bold)
FS_TITLE = 7      # panel titles
FS_LABEL = 6      # axis labels, annotations, in-figure text
FS_TICK = 5       # tick labels
FS_SMALL = 5.5    # legends, footnotes (NMI minimum is 5)

# ---- palette ----------------------------------------------------------
ORANGE = "#d9892b"
GREEN = "#5f9d5d"
ORANGE_DARK, ORANGE_MID, ORANGE_LIGHT = "#c96b16", "#e68a1f", "#f4c27a"
GREEN_DARK, GREEN_MID, GREEN_LIGHT = "#3f7f2d", "#5ea241", "#cfe6b8"
GRAY = "#7a7a7a"
BLUE = "#4878a8"  # accent for accuracy curves where orange/green are taken

# gate-voltage colormap convention (VG clip range used across all tasks)
VG_CMAP = "viridis"
VG_MIN, VG_MAX = 0.4, 8.0

# ---- NMI geometry -----------------------------------------------------
FULL_W = 7.205   # in == 183 mm double column
COL_W = 3.465    # in == 88 mm single column
DPI = 400


def apply_nmi_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "legend.fontsize": FS_SMALL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # math rendered in Open Sans, sans-serif fallback for symbols
        "mathtext.fontset": "custom",
        "mathtext.rm": "Open Sans",
        "mathtext.it": "Open Sans:italic",
        "mathtext.bf": "Open Sans:bold",
        "mathtext.fallback": "stixsans",
    })


def style_axes(ax, grid=True):
    if grid:
        ax.grid(alpha=0.25, linewidth=0.4)
    ax.tick_params(labelsize=6, width=0.6, length=2.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def panel_label(ax, label, x=0.0, y=1.02, fontsize=8):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="left", clip_on=False)


def save_figure(fig, path, dpi=DPI):
    """Save <path>.png and <path>.pdf, then close."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(p.with_suffix(ext), dpi=dpi, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return p.with_suffix(".png")


def make_full_fig(height_mm):
    return plt.figure(figsize=(FULL_W, height_mm / 25.4))
