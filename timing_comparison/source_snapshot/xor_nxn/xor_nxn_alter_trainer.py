#!/usr/bin/env python3
"""
Alter-based XOR N×N trainer using an NgSpice shared backend.

Features
--------
- Pure-Python helpers:
    * xor_dataset()
    * accuracy_nearest_target()
    * build_xor_graph_nxn()

- Netlist builder:
    * mk_switch_netlist(..., solver='klu' | 'sparse')

- NgSpiceShared backend (single instance) with:
    * .alter-based updates for inputs and outputs
    * Edge-gate updates via a contrastive local rule
    * 'destroy all' after each Ngspice solve to keep analysis state lean

- Per-epoch timing breakdown:
    * Total epoch time
    * Aggregated "train_run", "eval_run" (currently 0), "destroy"
    * Fine-grained segments per epoch:
        - t_mk_free
        - t_alter_inputs
        - t_run_free
        - t_read_free
        - t_destroy_free
        - t_alter_outputs
        - t_run_clamped
        - t_read_clamped
        - t_destroy_clamped
        - t_weight_updates

- CLI flags:
    --side SIDE          (grid side length, >=3; default 4)
    --epochs E           (number of training epochs)
    --gamma GAMMA        (learning rate for gate updates)
    --eta ETA            (clamp blending factor)
    --seed SEED          (random seed)
    --solver {klu,sparse} (Ngspice linear solver; default klu)

RUN_DIR environment variable
----------------------------
If RUN_DIR is set, results are written directly into that directory:

    $ RUN_DIR=/tmp/xor_shared_klu python xor_nxn_alter_trainer.py ...

Otherwise, results are placed under:

    results/runs/<timestamp>_N-<side>_seed-<seed>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared  # backend class


# ---------------------------------------------------------------------------
# Dataset + simple accuracy helper
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
# Graph topology for tests (logical XOR grid)
# ---------------------------------------------------------------------------


def build_xor_graph_nxn(side: int) -> Tuple[nx.Graph, Tuple[np.ndarray, np.ndarray]]:
    """
    Build an N×N periodic grid + one extra node for input return.

    Nodes:
        0 .. side*side-1      : grid nodes on an N×N torus
        side*side             : extra node (input negative reference)

    Inputs (4×2 array):
        - Positions are four corners of the grid:
            [0, side-1, (side-1)*side, side*side-1]
        - Negative terminal is the extra node (side*side) for all inputs.

    Outputs (1×2 array):
        - Single output:
            [center_node, 0]
          where center_node = (side//2, side//2) in 2D indexing.
    """
    if side < 3:
        raise ValueError("build_xor_graph_nxn requires side >= 3")

    n_grid = side * side
    extra_node = n_grid  # side*side

    G = nx.Graph()
    G.add_nodes_from(range(n_grid + 1))

    def idx(i: int, j: int) -> int:
        return i * side + j

    # Periodic 2D grid (torus)
    for i in range(side):
        for j in range(side):
            u = idx(i, j)
            v_right = idx(i, (j + 1) % side)
            v_down = idx((i + 1) % side, j)
            G.add_edge(u, v_right)
            G.add_edge(u, v_down)

    # Inputs: four corners
    inputs_pos = np.array(
        [
            idx(0, 0),
            idx(0, side - 1),
            idx(side - 1, 0),
            idx(side - 1, side - 1),
        ],
        dtype=int,
    )
    inputs_neg = np.full(4, extra_node, dtype=int)
    inputs = np.stack([inputs_pos, inputs_neg], axis=1)

    # Single output at the grid center, referenced to global ground (0)
    center = idx(side // 2, side // 2)
    outputs = np.array([[center, 0]], dtype=int)

    return G, (inputs, outputs)


# ---------------------------------------------------------------------------
# Ngspice helper utilities (alter-based)
# ---------------------------------------------------------------------------


def _exec_chunked(ng, cmds: Iterable[str], max_len: int = 900, sep: str = "; ") -> None:
    """
    Execute a list of SPICE commands in chunks so that each exec_command()
    call stays under ~max_len characters.
    """
    buf: List[str] = []
    length = 0
    for c in cmds:
        c = str(c)
        cl = len(c)
        if buf and (length + (len(sep) if length else 0) + cl) > max_len:
            ng.exec_command(sep.join(buf))
            buf = [c]
            length = cl
        else:
            length = (length + len(sep) + cl) if buf else cl
            buf.append(c)
    if buf:
        ng.exec_command(sep.join(buf))


def get_voltages(ng, nodes: Iterable[int]) -> np.ndarray:
    """
    Query Ngspice for voltages using 'print allv' and return V(node)
    for the requested nodes. If a node is absent, returns NaN there.
    """
    s = ng.exec_command("print allv")
    nodemap: Dict[int, float] = {}
    for line in s.splitlines():
        line = line.strip()
        if not line.startswith("v("):
            continue
        try:
            k, v = line.split(" = ")
            node_idx = int(k[2:-1])  # v(123) -> 123
            nodemap[node_idx] = float(v)
        except Exception:
            continue
    out: List[float] = []
    for n in nodes:
        out.append(float(nodemap.get(int(n), float("nan"))))
    return np.array(out, dtype=float)


def alter_inputs(ng, values: np.ndarray, base_idx: int) -> None:
    """
    Alter the DC values of the input voltage sources V{base_idx + i}.
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    cmds = [
        f"alter v{base_idx + i} dc = {float(v):.16f}"
        for i, v in enumerate(vals)
    ]
    _exec_chunked(ng, cmds)


def alter_outputs(ng, vset: np.ndarray, vout_base_idx: int) -> None:
    """
    Clamp outputs by lowering RS{i} to ~1 Ω and setting V{vout_base_idx + i}.
    """
    vset = np.asarray(vset, dtype=float).reshape(-1)
    K = int(len(vset))
    cmds: List[str] = []
    for i in range(1, K + 1):
        cmds.append(f"alter rs{i} 1.0")  # 1 Ω clamp
    for i in range(K):
        cmds.append(
            f"alter v{vout_base_idx + i} dc = {float(vset[i]):.16f}"
        )
    _exec_chunked(ng, cmds)


def mk_free(ng, K: int) -> None:
    """
    Set RS{i} to a very large value to "free" each output node
    from its clamp source.
    """
    cmds = [f"alter rs{i} 1e12" for i in range(1, K + 1)]
    _exec_chunked(ng, cmds)


# ---------------------------------------------------------------------------
# Training graph (separate from test graph) + netlist builder
# ---------------------------------------------------------------------------


def _build_training_graph(side: int) -> Tuple[nx.Graph, List[int], List[int], int, int]:
    """
    Build the transistor-level graph for training.
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

    i_idxs = [
        gidx(0, 0),
        gidx(0, side - 1),
        gidx(side - 1, 0),
        gidx(side - 1, side - 1),
    ]

    o_idxs = [gidx(side // 2, side // 2)]

    return G, i_idxs, o_idxs, negref_idx, posref_idx


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
    Build a transistor-level SPICE netlist for the given graph.
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
# Evaluation helper (used only for epoch 0 accuracy right now)
# ---------------------------------------------------------------------------


def _evaluate_xor(
    ng,
    X: np.ndarray,
    Y: np.ndarray,
    o_idxs: List[int],
    L0: float,
    K: int,
) -> float:
    """
    Evaluate XOR accuracy over the 4 patterns, using nearest-target logic.

    This is only used for epoch 0 baseline; we don't time this in detail.
    """
    preds: List[float] = []

    for i in range(X.shape[0]):
        mk_free(ng, K)
        alter_inputs(ng, X[i], base_idx=7)
        ng.run()
        out = get_voltages(ng, o_idxs)
        if out.size > 0:
            preds.append(float(out[0]))
        else:
            preds.append(float("nan"))

    preds_arr = np.array(preds, dtype=float).reshape(-1, 1)
    acc = accuracy_nearest_target(preds_arr, Y, L0)
    return float(acc)


# ---------------------------------------------------------------------------
# CLI, logging, main
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


def _parse_args():
    p = argparse.ArgumentParser(
        description="XOR N×N alter-based trainer (NgSpice shared backend)"
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
        help="Ngspice linear solver (default: klu)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    side = int(args.side)
    epochs = int(args.epochs)
    gamma = float(args.gamma)
    eta = float(args.eta)
    seed = int(args.seed)
    solver = str(args.solver).lower()

    random.seed(seed)
    np.random.seed(seed)

    # Dataset
    X, Y = xor_dataset()
    L0 = float(-0.087)

    # Training graph + netlist
    G, i_idxs, o_idxs, negref_idx, posref_idx = _build_training_graph(side)
    edge_list = list(G.edges())
    n_edges = len(edge_list)

    vg = np.random.uniform(0.5, 3.0, size=n_edges).astype(float)
    for (u, v), w in zip(edge_list, vg):
        G[u][v]["weight"] = float(w)

    max_node = max(G.nodes())

    netlist = mk_switch_netlist(
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

    # Save initial netlist + meta
    (run_dir / "netlist_initial.cir").write_text(netlist)
    meta = {
        "script": str(Path(__file__).resolve()),
        "script_name": Path(__file__).name,
        "argv": list(os.sys.argv),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "grid_side": side,
        "backend": "shared",
        "solver": solver,
        "variant": "xor_nxn_alter",
        "epochs": epochs,
        "gamma": gamma,
        "eta": eta,
    }
    try:
        (run_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2)
        )
    except Exception:
        pass

    # Logging
    log_f = _setup_logging(run_dir)
    print("=== RUN START (xor_nxn_alter) ===", flush=True)

    # Single NgSpice shared backend instance
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    # Node/index helpers for edge-wise updates
    nodes_list = np.asarray(sorted(G.nodes()), dtype=int)
    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)
    e1 = np.asarray([a for (a, _) in edge_list], dtype=int)
    e2 = np.asarray([b for (_, b) in edge_list], dtype=int)

    # Baseline accuracy before training (untimed decomposition)
    acc_hist: List[float] = []
    acc0 = _evaluate_xor(ng, X, Y, o_idxs, L0, K=len(o_idxs))
    acc_hist.append(acc0)
    print(f"Epoch    0: acc={acc0:.3f}", flush=True)

    # Epoch-wise timing arrays (aggregated)
    epoch_times: List[float] = []
    epoch_train_run_times: List[float] = []
    epoch_eval_run_times: List[float] = []
    epoch_destroy_times: List[float] = []

    # Fine-grained timing arrays
    epoch_t_mk_free: List[float] = []
    epoch_t_alter_inputs: List[float] = []
    epoch_t_run_free: List[float] = []
    epoch_t_read_free: List[float] = []
    epoch_t_destroy_free: List[float] = []

    epoch_t_alter_outputs: List[float] = []
    epoch_t_run_clamped: List[float] = []
    epoch_t_read_clamped: List[float] = []
    epoch_t_destroy_clamped: List[float] = []

    epoch_t_weight_updates: List[float] = []

    # Outputs per epoch (free-phase preds)
    outputs_list: List[np.ndarray] = []

    # Training loop
    for ep in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()

        # Per-epoch accumulators
        t_mk_free = 0.0
        t_alter_inputs = 0.0
        t_run_free = 0.0
        t_read_free = 0.0
        t_destroy_free = 0.0

        t_alter_outputs = 0.0
        t_run_clamped = 0.0
        t_read_clamped = 0.0
        t_destroy_clamped = 0.0

        t_weight_updates = 0.0

        order = np.arange(X.shape[0])
        np.random.shuffle(order)

        epoch_preds = np.full(X.shape[0], np.nan, dtype=float)

        for idx in order:
            x_row = X[idx]
            y_target = float(Y[idx, 0])

            # Free phase
            t0 = time.perf_counter()
            mk_free(ng, len(o_idxs))
            t1 = time.perf_counter()
            t_mk_free += (t1 - t0)

            t0 = time.perf_counter()
            alter_inputs(ng, x_row, base_idx=7)
            t1 = time.perf_counter()
            t_alter_inputs += (t1 - t0)

            t0 = time.perf_counter()
            ng.run()
            t1 = time.perf_counter()
            t_run_free += (t1 - t0)

            t0 = time.perf_counter()
            free_out = get_voltages(ng, o_idxs)
            free_nodes = get_voltages(ng, nodes_list)
            t1 = time.perf_counter()
            t_read_free += (t1 - t0)

            if free_out.size > 0:
                epoch_preds[idx] = float(free_out[0])
                free_y = float(free_out[0])
            else:
                epoch_preds[idx] = float("nan")
                free_y = 0.0

            t0 = time.perf_counter()
            try:
                ng.exec_command("destroy all")
            except Exception:
                pass
            t1 = time.perf_counter()
            t_destroy_free += (t1 - t0)

            # Clamped phase
            v_target = eta * y_target + (1.0 - eta) * free_y

            t0 = time.perf_counter()
            alter_outputs(
                ng,
                np.array([v_target], dtype=float),
                vout_base_idx=7 + len(i_idxs),
            )
            t1 = time.perf_counter()
            t_alter_outputs += (t1 - t0)

            t0 = time.perf_counter()
            ng.run()
            t1 = time.perf_counter()
            t_run_clamped += (t1 - t0)

            t0 = time.perf_counter()
            clamped_out = get_voltages(ng, o_idxs)
            clamped_nodes = get_voltages(ng, nodes_list)
            t1 = time.perf_counter()
            t_read_clamped += (t1 - t0)

            t0 = time.perf_counter()
            try:
                ng.exec_command("destroy all")
            except Exception:
                pass
            t1 = time.perf_counter()
            t_destroy_clamped += (t1 - t0)

            # Edge updates
            t0 = time.perf_counter()
            free_e1 = free_nodes[index_of[e1]]
            free_e2 = free_nodes[index_of[e2]]
            clamped_e1 = clamped_nodes[index_of[e1]]
            clamped_e2 = clamped_nodes[index_of[e2]]

            free_diffs = free_e1 - free_e2
            clamped_diffs = clamped_e1 - clamped_e2
            update = -gamma * (clamped_diffs**2 - free_diffs**2)

            if np.any(update != 0.0):
                cmds: List[str] = []
                for k, du in enumerate(update):
                    nv = vg[k] + float(du)
                    if nv < 0.4:
                        nv = 0.4
                    elif nv > 10.0:
                        nv = 10.0
                    vg[k] = nv
                    cmds.append(f"alter v.x{k}.v1 dc = {nv:.16f}")
                _exec_chunked(ng, cmds)
            t1 = time.perf_counter()
            t_weight_updates += (t1 - t0)

            # Free outputs again for next pattern
            t0 = time.perf_counter()
            mk_free(ng, len(o_idxs))
            t1 = time.perf_counter()
            t_mk_free += (t1 - t0)

        # Accuracy from cached free-phase preds (no extra solves)
        preds_arr = np.asarray(epoch_preds, dtype=float).reshape(-1, 1)
        acc_ep = accuracy_nearest_target(preds_arr, Y, L0)
        acc_hist.append(acc_ep)
        epoch_elapsed = time.perf_counter() - epoch_t0

        print(
            f"Epoch {ep:4d}: acc={acc_ep:.3f} "
            f"(elapsed={epoch_elapsed:.2f}s)",
            flush=True,
        )

        # Cache predictions for this epoch.
        outputs_list.append(np.array(epoch_preds, dtype=float))

        # Aggregate timings
        train_run_time = (
            t_mk_free
            + t_alter_inputs
            + t_run_free
            + t_read_free
            + t_alter_outputs
            + t_run_clamped
            + t_read_clamped
            + t_weight_updates
        )
        destroy_time = t_destroy_free + t_destroy_clamped
        eval_run_time = 0.0

        epoch_times.append(epoch_elapsed)
        epoch_train_run_times.append(train_run_time)
        epoch_eval_run_times.append(eval_run_time)
        epoch_destroy_times.append(destroy_time)

        # Store per-epoch segment times
        epoch_t_mk_free.append(t_mk_free)
        epoch_t_alter_inputs.append(t_alter_inputs)
        epoch_t_run_free.append(t_run_free)
        epoch_t_read_free.append(t_read_free)
        epoch_t_destroy_free.append(t_destroy_free)

        epoch_t_alter_outputs.append(t_alter_outputs)
        epoch_t_run_clamped.append(t_run_clamped)
        epoch_t_read_clamped.append(t_read_clamped)
        epoch_t_destroy_clamped.append(t_destroy_clamped)

        epoch_t_weight_updates.append(t_weight_updates)

        print(
            f"[TIMING] Epoch {ep}: total={epoch_elapsed:.4f}s "
            f"train_run={train_run_time:.4f}s "
            f"eval_run={eval_run_time:.4f}s "
            f"destroy={destroy_time:.4f}s",
            flush=True,
        )

        # Save timing arrays incrementally so partial sweeps are still useful
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

            np.save(
                run_dir / "0_epoch_t_mk_free.npy",
                np.asarray(epoch_t_mk_free, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_t_alter_inputs.npy",
                np.asarray(epoch_t_alter_inputs, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_t_run_free.npy",
                np.asarray(epoch_t_run_free, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_t_read_free.npy",
                np.asarray(epoch_t_read_free, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_t_destroy_free.npy",
                np.asarray(epoch_t_destroy_free, dtype=float),
            )

            np.save(
                run_dir / "0_epoch_t_alter_outputs.npy",
                np.asarray(epoch_t_alter_outputs, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_t_run_clamped.npy",
                np.asarray(epoch_t_run_clamped, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_t_read_clamped.npy",
                np.asarray(epoch_t_read_clamped, dtype=float),
            )
            np.save(
                run_dir / "0_epoch_t_destroy_clamped.npy",
                np.asarray(epoch_t_destroy_clamped, dtype=float),
            )

            np.save(
                run_dir / "0_epoch_t_weight_updates.npy",
                np.asarray(epoch_t_weight_updates, dtype=float),
            )
        except Exception as e:
            print(f"[WARN] failed to save epoch timing arrays: {e}", flush=True)

    final_acc = acc_hist[-1] if acc_hist else float("nan")
    print(f"FINAL acc={final_acc:.4f}", flush=True)
    print("=== RUN END (xor_nxn_alter) ===", flush=True)

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
