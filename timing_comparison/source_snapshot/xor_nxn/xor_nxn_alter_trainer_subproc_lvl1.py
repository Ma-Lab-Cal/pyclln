#!/usr/bin/env python3
from __future__ import annotations

"""
XOR N×N alter-based trainer using Ngspice subprocess backend with an ideal
Level‑1 NMOS device model.

This is a thin wrapper around ``xor_nxn_alter_trainer_subproc`` that swaps its
``mk_switch_netlist`` function for the Level‑1 variant defined in
``xor_nxn_alter_trainer_lvl1`` so that:

  - the shared alter trainer (Level‑1) and this subprocess trainer use the
    *same* handwritten netlist, and
  - all training logic (dataset, graph, learning rule, timing) remains
    identical to the existing Level‑2 subprocess trainer.
"""

import xor_nxn_alter_trainer_subproc as _base
from xor_nxn_alter_trainer_lvl1 import mk_switch_netlist  # type: ignore[import]


def main() -> None:
    # Replace the netlist builder used inside the subprocess trainer module
    # before delegating to its main().
    _base.mk_switch_netlist = mk_switch_netlist  # type: ignore[assignment]
    _base.main()


if __name__ == "__main__":
    main()

