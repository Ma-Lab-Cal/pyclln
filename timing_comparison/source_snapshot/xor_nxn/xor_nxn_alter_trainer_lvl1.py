#!/usr/bin/env python3
from __future__ import annotations

"""
XOR N×N alter-based trainer (Ngspice shared backend) using an *ideal* Level‑1
NMOS model, for comparison against the default Level‑2 transistor model.

This module is a thin wrapper around ``xor_nxn_alter_trainer`` which replaces
its ``mk_switch_netlist`` function with a variant that builds the same
topology but with a simple ideal-NMOS model:

    .subckt eK t_D t_S
    V1 t_G 0 <gate_voltage>
    R1 t_D t_S 1e16
    M1 t_D t_G t_S t_S Ideal
    .model Ideal NMOS (level=1)
    .ends eK

All other training logic (dataset, graph, learning rule, timing) is reused
unchanged from ``xor_nxn_alter_trainer``.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

import xor_nxn_alter_trainer as _base


def mk_switch_netlist(
    edge_list: List[Tuple[int, int]],
    weights: np.ndarray,
    max_node: int,
    I_pos: float,
    I_neg: float,
    i_idxs: List[int],
    o_idxs: List[int],
    negref_idx: int,
    posref_idx: int,
    solver: str = "klu",
) -> str:
    """
    Build a transistor-level SPICE netlist for the given graph using an
    ideal Level‑1 NMOS model (body shorted to source).
    """
    weights = np.asarray(weights, dtype=float).reshape(-1)
    lines: List[str] = []
    lines.append(".title xor_nxn_alter_mse_lvl1")

    # Edge devices (ideal Level‑1 NMOS with a large series resistor between D/S)
    for edge_idx, (t_D, t_S) in enumerate(edge_list):
        gate_voltage = float(weights[edge_idx])
        lines.append(f".subckt e{edge_idx} t_D t_S")
        lines.append(f"V1 t_G 0 {gate_voltage:.16f}")
        # Very large resistor to keep Ngspice happy when the device is off.
        lines.append("R1 t_D t_S 1e16")
        # Body shorted to source; ideal NMOS.
        lines.append("M1 t_D t_G t_S t_S Ideal")
        lines.append(".model Ideal NMOS (level=1)")
        lines.append(f".ends e{edge_idx}")

    # Output resistors RS{i} from each output node to a unique sink node
    for i in range(1, len(o_idxs) + 1):
        lines.append(f"RS{i} {o_idxs[i-1]} {max_node + i} 1e9")

    # Positive / negative references
    lines.append(f"V5 {negref_idx} 0 {I_neg:.2f}")
    lines.append(f"V6 {posref_idx} 0 {I_pos:.2f}")

    # Input voltage sources (default 0V; updated via 'alter')
    for i in range(len(i_idxs)):
        lines.append(f"V{7 + i} {i_idxs[i]} 0 0")

    # Output clamp sources (initially 0V; updated via 'alter')
    for i in range(len(o_idxs)):
        lines.append(
            f"V{7 + len(i_idxs) + i} {max_node + i + 1} 0 0"
        )

    # Instantiate edge subcircuits
    for edge_idx, (t_D, t_S) in enumerate(edge_list):
        lines.append(f"X{edge_idx} {t_D} {t_S} e{edge_idx}")

    # Ngspice options: keep identical to the default trainer so that timing
    # comparisons isolate the effect of the device model.
    if solver.lower() == "klu":
        lines.append(".options klu")

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

    return "\n".join(lines) + "\n"


def main() -> None:
    """
    Entry point: reuse the original alter trainer but swap in the Level‑1
    netlist builder before calling its ``main``.
    """
    _base.mk_switch_netlist = mk_switch_netlist  # type: ignore[assignment]
    _base.main()


if __name__ == "__main__":
    main()

