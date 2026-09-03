#!/usr/bin/env python3
"""Generate Figure 1 IV-curve panels for linear and NMOS edge elements.

Outputs:
  fig1_c_resistor.png
  fig1_c_nmos.png
  data/fig1_iv_curves_data.npz
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEVICE_LIB = REPO / "device_model" / "nmos_lvl1_ald1106.lib"

sys.path.insert(0, str(REPO / "figures_nmi"))
from nmi_style import apply_nmi_style, style_axes  # noqa: E402


DPI = 600
PANEL_W, PANEL_H = 1.16, 1.23
LINE_W = 0.55
RESISTOR_V_MIN, RESISTOR_V_MAX = -0.5, 5.0
ALD_V_MIN, ALD_V_MAX = -0.5, 5.0
N_V = 481
RESISTANCES_OHM = np.array([500.0, 1_000.0, 2_000.0, 5_000.0])
VGS_VALUES = np.array([1.00, 1.50, 2.00, 2.50])
TEXT_FONT = FontProperties(family="Open Sans", size=5)
LEGEND_FONT = FontProperties(family="Open Sans", size=5)

def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=DPI, facecolor="white")


def _format_resistance(r_ohm: float) -> str:
    if r_ohm >= 1000:
        return f"{r_ohm / 1000:g} kΩ"
    return f"{r_ohm:g} Ω"


def style_legend(legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.9)
    frame.set_edgecolor("none")
    frame.set_linewidth(0.0)


def resistor_curves(v: np.ndarray, resistances_ohm: np.ndarray) -> np.ndarray:
    return v[:, None] / resistances_ohm[None, :]


def curve_colors(n: int) -> np.ndarray:
    return plt.cm.viridis(np.linspace(0.12, 0.92, n))


def build_ald1106_netlist(vgs: float, v_min: float, v_max: float, v_step: float) -> str:
    """One NMOS device. Source and body share node s; gate is VGS above s."""
    return "\n".join(
        [
            ".title ald1106_bsource_gsource_iv",
            f'.include "{DEVICE_LIB}"',
            "VD d 0 0",
            "VS s 0 0",
            f"VG g s {vgs:.10f}",
            "X0 d g s s NMOSWRAP",
            ".options reltol=1e-4 abstol=1e-12 vntol=1e-8 gmin=1e-12 rshunt=1e12",
            ".options TEMP=27C TNOM=27C",
            f".dc VD {v_min:.10f} {v_max:.10f} {v_step:.10f}",
            ".print dc v(d) i(VD)",
            ".end",
            "",
        ]
    )


def parse_ngspice_dc_table(stdout: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ngspice .print dc v(d) i(VD); return VDS and drain current ID.

    ngspice reports current through the voltage source with positive current
    flowing from source positive to negative terminals. Drain current entering
    the MOSFET drain terminal is therefore -i(VD).
    """
    vds: list[float] = []
    ids: list[float] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            int(parts[0])
            v_d = float(parts[2])
            i_vd = float(parts[3])
        except ValueError:
            continue
        vds.append(v_d)
        ids.append(-i_vd)
    if not vds:
        raise RuntimeError("ngspice output did not contain a parseable DC table")
    return np.asarray(vds, dtype=float), np.asarray(ids, dtype=float)


def simulate_ald1106_curves(
    vgs_values: np.ndarray,
    v_min: float,
    v_max: float,
    n_v: int,
    ngspice_bin: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not DEVICE_LIB.exists():
        raise FileNotFoundError(f"missing device model: {DEVICE_LIB}")
    if shutil.which(ngspice_bin) is None:
        raise FileNotFoundError(f"ngspice executable not found: {ngspice_bin}")

    v_step = (v_max - v_min) / (n_v - 1)
    curves: list[np.ndarray] = []
    v_ref: np.ndarray | None = None
    with tempfile.TemporaryDirectory(prefix="fig1_ald1106_iv_") as tmp:
        tmpdir = Path(tmp)
        for vgs in vgs_values:
            netlist = build_ald1106_netlist(float(vgs), v_min, v_max, v_step)
            netlist_path = tmpdir / f"ald1106_vgs_{vgs:.2f}.cir"
            netlist_path.write_text(netlist)
            result = subprocess.run(
                [ngspice_bin, "-b", str(netlist_path)],
                cwd=tmpdir,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ngspice failed for VGS={vgs:.2f} V\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            vds, ids = parse_ngspice_dc_table(result.stdout)
            if v_ref is None:
                v_ref = vds
            elif len(vds) != len(v_ref) or not np.allclose(vds, v_ref):
                raise RuntimeError(f"VDS grid changed for VGS={vgs:.2f} V")
            curves.append(ids)
    assert v_ref is not None
    return v_ref, np.column_stack(curves)


def draw_zero_axes(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="black", lw=LINE_W, zorder=0)
    ax.axvline(0.0, color="black", lw=LINE_W, zorder=0)


def plot_resistor_iv(ax: plt.Axes, v: np.ndarray, currents_a: np.ndarray) -> None:
    colors = curve_colors(currents_a.shape[1])
    for idx, resistance in enumerate(RESISTANCES_OHM):
        ax.plot(
            v,
            1e3 * currents_a[:, idx],
            color=colors[idx],
            lw=LINE_W,
            label=_format_resistance(float(resistance)),
        )
    draw_zero_axes(ax)
    style_axes(ax, grid=False)
    ax.set_xlim(RESISTOR_V_MIN, RESISTOR_V_MAX)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)", labelpad=1)
    apply_text_style(ax)
    leg = ax.legend(
        frameon=True,
        loc="upper left",
        ncol=2,
        prop=LEGEND_FONT,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.15,
        labelspacing=0.12,
    )
    style_legend(leg)


def plot_ald1106_iv(ax: plt.Axes, v: np.ndarray, currents_a: np.ndarray) -> None:
    colors = curve_colors(currents_a.shape[1])[::-1]
    for idx, vgs in enumerate(VGS_VALUES):
        ax.plot(
            v,
            1e3 * currents_a[:, idx],
            color=colors[idx],
            lw=LINE_W,
            label=f"{vgs:g} V",
        )
    draw_zero_axes(ax)
    style_axes(ax, grid=False)
    ax.set_xlim(ALD_V_MIN, ALD_V_MAX)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)", labelpad=1)
    apply_text_style(ax)
    leg = ax.legend(
        frameon=True,
        loc="lower right",
        ncol=2,
        prop=LEGEND_FONT,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.15,
        labelspacing=0.12,
    )
    style_legend(leg)


def panel_axes(fig: plt.Figure) -> plt.Axes:
    return fig.add_axes([0.43, 0.28, 0.53, 0.65])


def apply_text_style(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontproperties(TEXT_FONT)
    ax.yaxis.label.set_fontproperties(TEXT_FONT)
    ax.tick_params(labelsize=5, pad=1, width=LINE_W)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_W)
        spine.set_color("black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(TEXT_FONT)


def make_figures(
    v: np.ndarray,
    resistor_i: np.ndarray,
    mos_v: np.ndarray,
    mos_i: np.ndarray,
) -> None:
    apply_nmi_style()

    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = panel_axes(fig)
    plot_resistor_iv(ax, v, resistor_i)
    _save(fig, "fig1_c_resistor")
    plt.close(fig)

    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = panel_axes(fig)
    plot_ald1106_iv(ax, mos_v, mos_i)
    _save(fig, "fig1_c_nmos")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice-bin", default="ngspice")
    args = parser.parse_args()

    v = np.linspace(RESISTOR_V_MIN, RESISTOR_V_MAX, N_V)
    resistor_i = resistor_curves(v, RESISTANCES_OHM)
    mos_v, mos_i = simulate_ald1106_curves(
        VGS_VALUES,
        ALD_V_MIN,
        ALD_V_MAX,
        N_V,
        args.ngspice_bin,
    )
    (HERE / "data").mkdir(exist_ok=True)
    np.savez_compressed(
        HERE / "data" / "fig1_iv_curves_data.npz",
        v_resistor=v,
        resistance_ohm=RESISTANCES_OHM,
        i_resistor_a=resistor_i,
        v_ald1106=mos_v,
        vgs=VGS_VALUES,
        i_ald1106_a=mos_i,
        device_model=str(DEVICE_LIB.relative_to(REPO)),
        ald1106_body="source",
        ald1106_gate_reference="source",
    )
    make_figures(v, resistor_i, mos_v, mos_i)
    print(f"saved IV panels in {HERE}")


if __name__ == "__main__":
    main()
