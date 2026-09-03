#!/usr/bin/env python3
"""
XOR N×N trainer using the *same* handwritten transistor-level netlist as
`xor_nxn_alter_trainer.py`, but driving Ngspice via a subprocess backend.

Goal
----
Provide a "perfect" subprocess baseline that matches the shared-library
alter trainer in:
  - dataset and accuracy logic,
  - graph topology,
  - transistor-level netlist (ALD NMOS),
  - contrastive local learning rule,
  - 2 solves per pattern (free + clamped),
  - timing arrays and run_meta format,

while differing *only* in how Ngspice is invoked:
  - here: fresh `ngspice -b` subprocesses per solve,
  - shared trainer: a single persistent NgSpiceShared context with `alter`
    commands and `destroy all` cleanups.

Outputs
-------
For each run directory, we write:
  - run_meta.json (backend="subprocess", variant="xor_nxn_alter_subproc")
  - netlist_initial.cir  (initial circuit with random gate voltages)
  - 0_epoch_times.npy
  - 0_epoch_train_run_times.npy
  - 0_epoch_eval_run_times.npy (always zeros here)
  - 0_epoch_destroy_times.npy (always zeros here)
  - 0_acc.npy, 0_val_acc.npy (accuracy history; includes Epoch 0 baseline)
  - 0_outputs.npy (free-phase predictions per epoch, shaped [epochs, 4])

Ngspice interface
-----------------
For each (epoch, sample) we do:
  1. FREE PHASE (one ngspice -b call)
     - Write a netlist with current gate voltages from `vg`.
     - .control:
         * alter rs{i} 1e12        (free outputs)
         * alter v7..v(7+K-1)      (inputs = x_row)
         * run
         * print allv > free file
  2. Read the free file in Python, compute free_y, and blend:
         v_target = eta * y_target + (1 - eta) * free_y
  3. CLAMPED PHASE (second ngspice -b call)
     - Same base netlist with same `vg`.
     - .control:
         * alter rs{i} 1.0         (strong clamp)
         * alter v7..v(7+K-1)      (inputs = x_row)
         * alter v_out_clamp       (single output clamp) to v_target
         * run
         * print allv > clamped file
  4. Read clamped file, get node voltages, apply the same contrastive
     update rule as in the shared trainer.

Epoch accuracy is computed from the cached free-phase predictions exactly
as in the shared trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Dataset + simple accuracy helper (copied from xor_nxn_alter_trainer.py)
# ---------------------------------------------------------------------------


def xor_dataset(
    I_pos: float = 0.33,
    I_neg: float = 0.11,
    I0: float = 0.45,
    L0: float = -0.087,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X, Y) for the 2-bit XOR as analog inputs/targets.

    X shape: (4, 4)
      Columns: [bias_neg, bias_pos, bit_a, bit_b]
    Y shape: (4, 1)
      Values: {0, L0}

    Truth table:
      (0,0) -> 0
      (0,1) -> L0
      (1,0) -> L0
      (1,1) -> 0
    """
    X = np.array(
        [
            [I_neg, I_pos, 0.0, 0.0],
            [I_neg, I_pos, 0.0, I0],
            [I_neg, I_pos, I0, 0.0],
            [I_neg, I_pos, I0, I0],
        ],
        dtype=float,
    )
    Y = np.array([0.0, L0, L0, 0.0], dtype=float).reshape(-1, 1)
    return X, Y


def accuracy_nearest_target(
    preds: np.ndarray,
    targets: np.ndarray,
    L0: float,
) -> float:
    """
    Accuracy for XOR where analog outputs are in {0, L0}:

      - For each prediction ŷ, compare squared distance to 0 and L0.
      - Predicted label is 0 if d(0) <= d(L0), else "L0".
      - True label is 0 if target ≈ 0, else "L0".
      - Returns mean correctness over samples.
    """
    p = np.asarray(preds, dtype=float).reshape(-1)
    t = np.asarray(targets, dtype=float).reshape(-1)

    d0 = (p - 0.0) ** 2
    d1 = (p - L0) ** 2
    pred_bits = np.where(d1 < d0, 1, 0)

    tol = 1e-6
    true_bits = np.where(np.abs(t - L0) <= tol, 1, 0)

    if true_bits.size == 0:
        return float("nan")
    return float(np.mean(pred_bits == true_bits))


# ---------------------------------------------------------------------------
# Graph topology for training (copied from xor_nxn_alter_trainer.py)
# ---------------------------------------------------------------------------


def _build_training_graph(side: int) -> Tuple[nx.Graph, List[int], List[int], int, int]:
    """
    Build the transistor-level graph for training.

    Node indexing matches xor_nxn_alter_trainer.py:
      - 0: (unused, global ground is node 0 in SPICE)
      - 1: negref_idx
      - 2: posref_idx
      - 3..: grid nodes laid out on a side×side torus.
    """
    if side < 3:
        raise ValueError("_build_training_graph requires side >= 3")

    negref_idx = 1
    posref_idx = 2
    base = 3  # first grid node index

    def gidx(i: int, j: int) -> int:
        return base + i * side + j

    G = nx.Graph()
    for i in range(side):
        for j in range(side):
            G.add_node(gidx(i, j))

    for i in range(side):
        for j in range(side):
            u = gidx(i, j)
            v_right = gidx(i, (j + 1) % side)
            v_down = gidx((i + 1) % side, j)
            G.add_edge(u, v_right)
            G.add_edge(u, v_down)

    # Input nodes: four corners of the grid.
    i_idxs = [
        gidx(0, 0),
        gidx(0, side - 1),
        gidx(side - 1, 0),
        gidx(side - 1, side - 1),
    ]

    # Single output at the grid center.
    o_idxs = [gidx(side // 2, side // 2)]

    return G, i_idxs, o_idxs, negref_idx, posref_idx


# ---------------------------------------------------------------------------
# Netlist builder (copied from xor_nxn_alter_trainer.py)
# ---------------------------------------------------------------------------


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
    Build a transistor-level SPICE netlist for the given graph with ALD NMOS
    devices. Gate voltages are set by the `weights` array.

    This is identical (modulo whitespace) to the netlist builder in
    xor_nxn_alter_trainer.py.
    """
    weights = np.asarray(weights, dtype=float).reshape(-1)
    lines: List[str] = []
    lines.append(".title xor_nxn_alter_mse")

    # Edge devices (NMOS with gate bias)
    for edge_idx, (t_D, t_S) in enumerate(edge_list):
        gate_voltage = float(weights[edge_idx])
        lines.append(f".subckt e{edge_idx} t_D t_S params: vtn=0.0")
        lines.append(".param vtx={vtn} cox=1.0 ires=0.41 pox=1.0 M=1")
        lines.append(f"V1 t_G 0 {gate_voltage:.16f}")
        lines.extend(
            [
                "RB t_B 0 10",
                "M1 t_D t_G t_S t_B ncg l=7.8e-6 w=0.138e-3 as=0.603e-8 ps=0.478e-3 ad=0.161e-8",
                "+                                  nrd=.3 nrs=1",
                ".model ncg  nmos  (level=2 ",
                "+ gamma=1.09 vto={.750+vtn}",
                "+ Uo=650 Ucrit=7000 Uexp=.1 Vmax=1.6e5",
                "+ phi=.70 tpg=+1",
                "+ nsub={1e16*ires} neff={10*ires} nss=7e10 nfs=1.17e11",
                "+ tox={.055u*cox} ",
                "+ Cgso={.94n*cox} Cgdo={.59n*cox} Cgbo={.138n*pox}",
                "+ cj=.39m cjsw=264p",
                "+ xj=2.0u ld=1.6u ",
                "+ pb=.9 js=20e-6  mj=.5 mjsw=0.18",
                "+ kf=.75e-28 rsh=10)",
                "d1 t_S t_Vp dps",
                ".model dps D (Is=2.61e-7 Isr=1.0e-5 Bv=34 Ibv=1e+4 Rs=2.74e-7 trs1=3e-3 Cjo=1.3e-4)",
                f".ends e{edge_idx}",
            ]
        )

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

    # Ngspice options
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


# ---------------------------------------------------------------------------
# Logging helper (similar to xor_nxn_alter_trainer._setup_logging)
# ---------------------------------------------------------------------------


def _setup_logging(run_dir: Path):
    """
    Tee stdout/stderr into run_dir/train_log.txt while preserving console IO.
    """
    log_path = run_dir / "train_log.txt"
    log_f = open(log_path, "a", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                except Exception:
                    pass

        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass

    out_streams = []
    err_streams = []
    try:
        if getattr(sys, "__stdout__", None):
            out_streams.append(sys.__stdout__)
    except Exception:
        pass
    out_streams.append(sys.stdout)
    out_streams.append(log_f)

    try:
        if getattr(sys, "__stderr__", None):
            err_streams.append(sys.__stderr__)
    except Exception:
        pass
    err_streams.append(sys.stderr)
    err_streams.append(log_f)

    sys.stdout = _Tee(*out_streams)  # type: ignore
    sys.stderr = _Tee(*err_streams)  # type: ignore
    return log_f


# ---------------------------------------------------------------------------
# Small helpers for reading Ngspice 'print allv' output
# ---------------------------------------------------------------------------


def _read_allv_file(path: Path) -> Dict[int, float]:
    """
    Parse a 'print allv' output file and return a mapping node_idx -> voltage.
    """
    nodemap: Dict[int, float] = {}
    try:
        text = path.read_text()
    except FileNotFoundError:
        return nodemap
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("v("):
            continue
        try:
            k, v = line.split(" = ")
            node_idx = int(k[2:-1])  # v(123) -> 123
            nodemap[node_idx] = float(v)
        except Exception:
            continue
    return nodemap


def _voltages_from_map(nodemap: Dict[int, float], nodes: Iterable[int]) -> np.ndarray:
    out: List[float] = []
    for n in nodes:
        out.append(float(nodemap.get(int(n), float("nan"))))
    return np.array(out, dtype=float)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "XOR N×N alter-based trainer using ngspice subprocess backend "
            "on the same handwritten netlist as xor_nxn_alter_trainer.py"
        )
    )
    p.add_argument("--side", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--solver",
        type=str,
        choices=["klu", "sparse"],
        default="klu",
        help="Ngspice linear solver (default: klu; 'sparse' means no .options klu).",
    )
    p.add_argument(
        "--ngspice-bin",
        type=str,
        default="ngspice",
        help="Ngspice executable to invoke (default: 'ngspice').",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Optional epoch-0 baseline evaluation (free-phase only)
# ---------------------------------------------------------------------------


def _evaluate_xor_subproc(
    run_dir: Path,
    X: np.ndarray,
    Y: np.ndarray,
    L0: float,
    G: nx.Graph,
    i_idxs: List[int],
    o_idxs: List[int],
    negref_idx: int,
    posref_idx: int,
    vg: np.ndarray,
    max_node: int,
    solver: str,
    ngspice_bin: str,
) -> float:
    """
    Compute XOR accuracy using free-phase-only solves via ngspice subprocess.
    We do NOT time this in the epoch timing arrays; it's just a baseline
    "Epoch 0" accuracy, matching the shared trainer's behavior.
    """
    K_out = len(o_idxs)
    preds = []

    for idx in range(X.shape[0]):
        x_row = X[idx]

        netlist = mk_switch_netlist(
            edge_list=list(G.edges()),
            weights=vg,
            max_node=max_node,
            I_pos=0.45,
            I_neg=0.0,
            i_idxs=i_idxs,
            o_idxs=o_idxs,
            negref_idx=negref_idx,
            posref_idx=posref_idx,
            solver=solver,
        )

        free_fname = f"eval_allv_idx{idx:02d}_free.txt"

        lines = netlist.strip().splitlines()
        if not lines or lines[-1].strip().lower() != ".end":
            raise RuntimeError("mk_switch_netlist must end with '.end'")
        body_lines = lines[:-1]
        end_line = lines[-1]

        control: List[str] = []
        control.append(".control")
        # Free phase: RS large, inputs set to x_row
        for i in range(1, K_out + 1):
            control.append(f"alter rs{i} 1e12")
        for i, v in enumerate(np.asarray(x_row, dtype=float).reshape(-1)):
            control.append(f"alter v{7 + i} dc = {float(v):.16f}")
        control.append("run")
        control.append(f"print allv > {free_fname}")
        control.append("quit")
        control.append(".endc")

        final_lines = body_lines + control + [end_line]
        cir_path = run_dir / f"eval_net_idx{idx:02d}.cir"
        cir_path.write_text("\n".join(final_lines) + "\n")

        try:
            subprocess.run(
                [ngspice_bin, "-b", str(cir_path.name)],
                cwd=run_dir,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] ngspice failed during eval on {cir_path}: {exc}", flush=True)
            preds.append(float("nan"))
            continue

        free_map = _read_allv_file(run_dir / free_fname)
        if o_idxs:
            out_idx = int(o_idxs[0])
            preds.append(float(free_map.get(out_idx, float("nan"))))
        else:
            preds.append(float("nan"))

    preds_arr = np.asarray(preds, dtype=float).reshape(-1, 1)
    return float(accuracy_nearest_target(preds_arr, Y, L0))


# ---------------------------------------------------------------------------
# Main training loop (subprocess backend, contrastive local rule)
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    side = int(args.side)
    epochs = int(args.epochs)
    gamma = float(args.gamma)
    eta = float(args.eta)
    seed = int(args.seed)
    solver = str(args.solver).lower()
    ngspice_bin = str(args.ngspice_bin)

    # Ensure we can actually invoke ngspice in this environment.
    if shutil.which(ngspice_bin) is None:
        raise SystemExit(
            f"ngspice executable '{ngspice_bin}' not found in PATH. "
            "Set --ngspice-bin to the correct binary name or path, "
            "or run inside the environment where ngspice is installed."
        )

    random.seed(seed)
    np.random.seed(seed)

    # Dataset
    X, Y = xor_dataset()
    L0 = float(-0.087)

    # Training graph + netlist meta
    G, i_idxs, o_idxs, negref_idx, posref_idx = _build_training_graph(side)
    edge_list = list(G.edges())
    n_edges = len(edge_list)

    vg = np.random.uniform(0.5, 3.0, size=n_edges).astype(float)
    for (u, v), w in zip(edge_list, vg):
        G[u][v]["weight"] = float(w)

    max_node = max(G.nodes())

    # Results directory
    this_dir = Path(__file__).resolve().parent
    results_root = this_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        runs_dir = results_root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = (
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + f"_N-{side}_seed-{seed}"
        )
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    # Save initial netlist (for reference; training runs use per-sample netlists with .control blocks).
    base_netlist = mk_switch_netlist(
        edge_list=edge_list,
        weights=vg,
        max_node=max_node,
        I_pos=0.45,
        I_neg=0.0,
        i_idxs=i_idxs,
        o_idxs=o_idxs,
        negref_idx=negref_idx,
        posref_idx=posref_idx,
        solver=solver,
    )
    (run_dir / "netlist_initial.cir").write_text(base_netlist)

    meta = {
        "script": str(Path(__file__).resolve()),
        "script_name": Path(__file__).name,
        "argv": list(os.sys.argv),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "grid_side": side,
        "backend": "subprocess",
        "solver": solver,
        "variant": "xor_nxn_alter_subproc",
        "epochs": epochs,
        "gamma": gamma,
        "eta": eta,
        "ngspice_bin": ngspice_bin,
    }
    try:
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    # Logging
    log_f = _setup_logging(run_dir)
    print("=== RUN START (xor_nxn_alter_subproc) ===", flush=True)

    # Node/index helpers for edge-wise updates
    nodes_list = np.asarray(sorted(G.nodes()), dtype=int)
    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)
    e1 = np.asarray([a for (a, _) in edge_list], dtype=int)
    e2 = np.asarray([b for (_, b) in edge_list], dtype=int)
    K_out = len(o_idxs)
    vout_base_idx = 7 + len(i_idxs)  # first output clamp source index

    # Epoch-wise timing arrays
    epoch_times: List[float] = []
    epoch_train_run_times: List[float] = []
    epoch_eval_run_times: List[float] = []
    epoch_destroy_times: List[float] = []  # always 0.0 here

    # Outputs per epoch (free-phase predictions for each pattern)
    outputs_list: List[np.ndarray] = []
    acc_hist: List[float] = []

    # Optional baseline (Epoch 0) via free-phase-only subprocess solves
    acc0 = _evaluate_xor_subproc(
        run_dir,
        X,
        Y,
        L0,
        G,
        i_idxs,
        o_idxs,
        negref_idx,
        posref_idx,
        vg,
        max_node,
        solver,
        ngspice_bin,
    )
    acc_hist.append(acc0)
    print(f"Epoch    0: acc={acc0:.3f}", flush=True)

    # Training loop
    for ep in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        train_run_time = 0.0

        order = np.arange(X.shape[0])
        np.random.shuffle(order)

        epoch_preds = np.full(X.shape[0], np.nan, dtype=float)

        for idx in order:
            x_row = X[idx]
            y_target = float(Y[idx, 0])

            # ---------- FREE PHASE (subprocess run) ----------
            netlist_free = mk_switch_netlist(
                edge_list=edge_list,
                weights=vg,
                max_node=max_node,
                I_pos=0.45,
                I_neg=0.0,
                i_idxs=i_idxs,
                o_idxs=o_idxs,
                negref_idx=negref_idx,
                posref_idx=posref_idx,
                solver=solver,
            )

            free_fname = f"allv_ep{ep:04d}_idx{idx:02d}_free.txt"

            lines = netlist_free.strip().splitlines()
            if not lines or lines[-1].strip().lower() != ".end":
                raise RuntimeError("mk_switch_netlist must end with '.end'")
            body_lines = lines[:-1]
            end_line = lines[-1]

            control_free: List[str] = []
            control_free.append(".control")
            # Free phase: RS large, inputs set to x_row
            for i in range(1, K_out + 1):
                control_free.append(f"alter rs{i} 1e12")
            for i, v in enumerate(np.asarray(x_row, dtype=float).reshape(-1)):
                control_free.append(f"alter v{7 + i} dc = {float(v):.16f}")
            control_free.append("run")
            control_free.append(f"print allv > {free_fname}")
            control_free.append("quit")
            control_free.append(".endc")

            final_lines_free = body_lines + control_free + [end_line]
            cir_free = run_dir / f"net_ep{ep:04d}_idx{idx:02d}_free.cir"
            cir_free.write_text("\n".join(final_lines_free) + "\n")

            t0 = time.perf_counter()
            try:
                subprocess.run(
                    [ngspice_bin, "-b", str(cir_free.name)],
                    cwd=run_dir,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] ngspice failed on FREE {cir_free}: {exc}", flush=True)
                continue
            t1 = time.perf_counter()
            train_run_time += (t1 - t0)

            free_map = _read_allv_file(run_dir / free_fname)
            free_nodes = _voltages_from_map(free_map, nodes_list)

            # Prediction from free phase: first output node.
            if o_idxs:
                out_idx = int(o_idxs[0])
                free_y = float(free_map.get(out_idx, float("nan")))
                epoch_preds[idx] = free_y
            else:
                free_y = float("nan")
                epoch_preds[idx] = float("nan")

            # ---------- CLAMPED PHASE (subprocess run) ----------
            # Blended target using free_y as in shared trainer:
            #   v_target = eta * y_target + (1 - eta) * free_y
            if not np.isnan(free_y):
                v_target = eta * y_target + (1.0 - eta) * free_y
            else:
                v_target = y_target

            netlist_clamped = mk_switch_netlist(
                edge_list=edge_list,
                weights=vg,
                max_node=max_node,
                I_pos=0.45,
                I_neg=0.0,
                i_idxs=i_idxs,
                o_idxs=o_idxs,
                negref_idx=negref_idx,
                posref_idx=posref_idx,
                solver=solver,
            )

            clamped_fname = f"allv_ep{ep:04d}_idx{idx:02d}_clamped.txt"

            lines_c = netlist_clamped.strip().splitlines()
            if not lines_c or lines_c[-1].strip().lower() != ".end":
                raise RuntimeError("mk_switch_netlist must end with '.end'")
            body_lines_c = lines_c[:-1]
            end_line_c = lines_c[-1]

            control_clamped: List[str] = []
            control_clamped.append(".control")
            # Clamp phase: RS small, inputs set to x_row, output clamp to v_target
            for i in range(1, K_out + 1):
                control_clamped.append(f"alter rs{i} 1.0")
            for i, v in enumerate(np.asarray(x_row, dtype=float).reshape(-1)):
                control_clamped.append(f"alter v{7 + i} dc = {float(v):.16f}")
            control_clamped.append(
                f"alter v{vout_base_idx} dc = {float(v_target):.16f}"
            )
            control_clamped.append("run")
            control_clamped.append(f"print allv > {clamped_fname}")
            control_clamped.append("quit")
            control_clamped.append(".endc")

            final_lines_c = body_lines_c + control_clamped + [end_line_c]
            cir_clamped = run_dir / f"net_ep{ep:04d}_idx{idx:02d}_clamped.cir"
            cir_clamped.write_text("\n".join(final_lines_c) + "\n")

            t0 = time.perf_counter()
            try:
                subprocess.run(
                    [ngspice_bin, "-b", str(cir_clamped.name)],
                    cwd=run_dir,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] ngspice failed on CLAMPED {cir_clamped}: {exc}", flush=True)
                continue
            t1 = time.perf_counter()
            train_run_time += (t1 - t0)

            clamped_map = _read_allv_file(run_dir / clamped_fname)
            clamped_nodes = _voltages_from_map(clamped_map, nodes_list)

            # Edge-wise contrastive update (same as shared trainer)
            free_e1 = free_nodes[index_of[e1]]
            free_e2 = free_nodes[index_of[e2]]
            clamped_e1 = clamped_nodes[index_of[e1]]
            clamped_e2 = clamped_nodes[index_of[e2]]

            free_diffs = free_e1 - free_e2
            clamped_diffs = clamped_e1 - clamped_e2
            update = -gamma * (clamped_diffs**2 - free_diffs**2)

            if np.any(update != 0.0):
                for k, du in enumerate(update):
                    nv = vg[k] + float(du)
                    if nv < 0.4:
                        nv = 0.4
                    elif nv > 10.0:
                        nv = 10.0
                    vg[k] = nv

        # Epoch accuracy from cached free-phase predictions.
        preds_arr = np.asarray(epoch_preds, dtype=float).reshape(-1, 1)
        acc_ep = accuracy_nearest_target(preds_arr, Y, L0)
        acc_hist.append(acc_ep)
        epoch_elapsed = time.perf_counter() - epoch_t0

        print(
            f"Epoch {ep:4d}: acc={acc_ep:.3f} "
            f"(elapsed={epoch_elapsed:.2f}s, train_run={train_run_time:.2f}s)",
            flush=True,
        )

        # Cache predictions for this epoch.
        outputs_list.append(np.array(epoch_preds, dtype=float))

        # Record timings
        epoch_times.append(epoch_elapsed)
        epoch_train_run_times.append(train_run_time)
        epoch_eval_run_times.append(0.0)
        epoch_destroy_times.append(0.0)

        print(
            f"[TIMING] Epoch {ep}: total={epoch_elapsed:.4f}s "
            f"train_run={train_run_time:.4f}s eval_run=0.0000s destroy=0.0000s",
            flush=True,
        )

        # Save timing arrays incrementally
        try:
            np.save(run_dir / "0_epoch_times.npy", np.asarray(epoch_times, dtype=float))
            np.save(
                run_dir / "0_epoch_train_run_times.npy",
                np.asarray(epoch_train_run_times, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_eval_run_times.npy",
                np.asarray(epoch_eval_run_times, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_destroy_times.npy",
                np.asarray(epoch_destroy_times, dtype=float),
            )
        except Exception as e:
            print(f"[WARN] failed to save epoch timing arrays: {e}", flush=True)

    final_acc = acc_hist[-1] if acc_hist else float("nan")
    print(f"FINAL acc={final_acc:.4f}", flush=True)
    print("=== RUN END (xor_nxn_alter_subproc) ===", flush=True)

    # Save outputs per epoch and accuracy history
    try:
        np.save(run_dir / "0_outputs.npy", np.array(outputs_list, dtype=object))
    except Exception as e:
        print(f"[WARN] failed to save 0_outputs.npy: {e}", flush=True)

    try:
        acc_arr = np.asarray(acc_hist, dtype=float)
        np.save(run_dir / "0_acc.npy", acc_arr)
        np.save(run_dir / "0_val_acc.npy", acc_arr)
    except Exception:
        pass

    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
