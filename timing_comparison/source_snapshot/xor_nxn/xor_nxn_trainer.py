#!/usr/bin/env python3
"""
XOR N×N Grid — Training with switchable ngspice backends and basic timing.

This is a generalisation of the 4×4 / 30×30 XOR trainers. It uses the same
4-pattern XOR dataset (two logical inputs + two bias drives) and places the
input and output drives on selected nodes of an N×N periodic grid.

Backends
--------
--backend shared     : PySpice + ngspice shared library (single process)
--backend subprocess : PySpice + ngspice subprocess (new process per run)

--solver klu         : add '.options klu' (KLU linear solver)
--solver sparse      : default sparse solver (no extra option)

The actual backend / solver selection is implemented in sim.spice_net.AbstractNetwork.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
import resource
import sys

import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# NumPy 2.x compatibility patch for PySpice RawFile (subprocess backend)
# ---------------------------------------------------------------------------


def _patch_pyspice_rawfile_for_numpy2() -> None:
    """Monkey-patch PySpice's NgSpice RawFile._read_variable_data for NumPy 2.x.

    PySpice <= 1.5 uses ``np.fromstring(raw_data, dtype='f8')`` in binary mode,
    which raises ``ValueError: The binary mode of fromstring is removed, use
    frombuffer instead`` under NumPy 2.x. This replaces that call with an
    equivalent ``np.frombuffer(...)`` implementation on the actual
    PySpice.Spice.NgSpice.RawFile.RawFile class.
    """
    try:
        # In your repo, the failing path in the traceback is:
        #   PySpice/Spice/NgSpice/RawFile.py
        # so import from there.
        from PySpice.Spice.NgSpice import RawFile as RawFileMod  # type: ignore[import]
    except Exception:
        return

    # RawFileMod is either the class itself or a module exporting RawFile.
    RawFileCls = getattr(RawFileMod, "RawFile", RawFileMod)
    if RawFileCls is None:
        return

    def _read_variable_data(self, raw_data):  # type: ignore[override]
        import numpy as _np  # local import to avoid polluting module namespace

        # Determine how many columns (real vs complex).
        if self.flags == "real":
            number_of_columns = self.number_of_variables
        elif self.flags == "complex":
            number_of_columns = 2 * self.number_of_variables
        else:
            raise NotImplementedError

        # NumPy 2.x-compatible binary decode.
        input_data = _np.frombuffer(
            raw_data,
            count=number_of_columns * self.number_of_points,
            dtype="f8",
        )
        input_data = input_data.reshape((self.number_of_points, number_of_columns))
        input_data = input_data.transpose()

        # Complex variables: interleaved real/imag columns.
        if self.flags == "complex":
            raw_arr = input_data
            input_data = _np.array(raw_arr[0::2], dtype="complex128")
            input_data.imag = raw_arr[1::2]

        # Attach per-variable traces.
        for variable in self.variables.values():
            variable.data = input_data[variable.index]

    # Install patch (idempotent).
    RawFileCls._read_variable_data = _read_variable_data


_patch_pyspice_rawfile_for_numpy2()

# ---------------------------------------------------------------------------
# Remaining imports that use PySpice
# ---------------------------------------------------------------------------

# Make sure we can import the sim package (spice_net, etc.) from the repo root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sim.spice_net import GroundReferenceNetwork  # type: ignore[import]
from PySpice.Unit import u_V  # type: ignore[import]


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="XOR N×N training with switchable ngspice backends"
    )
    p.add_argument(
        "--side",
        type=int,
        default=30,
        help="Grid side length N (default: 30 => 30×30)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for weight initialisation",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="Target blending factor (0..1)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="Learning-rate prefactor",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="shared",
        choices=("shared", "subprocess"),
        help="Ngspice backend: 'shared' library or 'subprocess' (default: shared)",
    )
    p.add_argument(
        "--solver",
        type=str,
        default="klu",
        choices=("klu", "sparse"),
        help="Linear solver: 'klu' (.options klu) or 'sparse' (default ngspice).",
    )
    p.add_argument(
        "--body-mode",
        type=str,
        default="source",
        choices=("source", "ground", "floating"),
        help="NMOS body connection: 'source' (default), 'ground', or 'floating'.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Graph + dataset
# ---------------------------------------------------------------------------


def build_xor_graph_nxn(side: int):
    """Construct an N×N periodic grid and choose input/output nodes.

    Returns
    -------
    grid : nx.Graph
        Periodic grid graph with an extra reference node (index N*N).
    node_cfg : (inputs, outputs)
        Each is an array of shape (n_sources, 2) with [node_pos, node_neg]
        for each behavioural voltage source.
    """
    grid = nx.grid_graph([side, side], periodic=True)
    grid = nx.convert_node_labels_to_integers(grid, first_label=0, ordering="sorted")

    # Extra "reference" node used as the negative terminal of all input drives.
    extra_node = side * side
    grid.add_node(extra_node)

    # Random initial gate biases on every edge.
    for u, v in grid.edges():
        grid[u][v]["weight"] = float(np.random.uniform(0.1, 1.0))

    def idx(r: int, c: int) -> int:
        return r * side + c

    # Four input nodes roughly at the corners (one hop in from the border).
    n_in1 = idx(1, 1)
    n_in2 = idx(1, side - 2)
    n_in3 = idx(side - 2, 1)
    n_in4 = idx(side - 2, side - 2)

    # Single output near the centre.
    n_out = idx(side // 2, side // 2)

    # Node config: each row is [node_pos, node_neg].
    inputs = np.array(
        [
            [n_in1, extra_node],
            [n_in2, extra_node],
            [n_in3, extra_node],
            [n_in4, extra_node],
        ],
        dtype=int,
    )
    outputs = np.array([[n_out, 0]], dtype=int)
    node_cfg = (inputs, outputs)
    return grid, node_cfg


def xor_dataset(I_pos=0.33, I_neg=0.11, I0=0.45, L0=-0.087):
    """Four-pattern XOR dataset with two logical inputs and two bias drives.

    Inputs:
      - First two entries: fixed bias drives (I_neg, I_pos).
      - Last two entries: logical inputs i1, i2 scaled by I0.

    Targets:
      - 0       when i1 == i2
      - L0 (<0) when i1 != i2
    """
    X, Y = [], []
    for i1 in [0, 1]:
        for i2 in [0, 1]:
            X.append([I_neg, I_pos, I0 * i1, I0 * i2])
            Y.append([L0 * (i1 != i2)])
    return np.array(X, dtype=float), np.array(Y, dtype=float)


def accuracy_nearest_target(preds: np.ndarray, Y: np.ndarray, l0: float = -0.087) -> float:
    """Accuracy by nearest analogue target (0 vs L0).

    Parameters
    ----------
    preds : ndarray, shape (N, 1) or (N,)
        Predicted voltages at the output node.
    Y : ndarray, shape (N, 1) or (N,)
        Target voltages in {0, L0}.

    Returns
    -------
    float
        Fraction of samples for which the predicted voltage is closer to
        the correct target than to the other target.
    """
    p = preds.reshape(-1)
    t = Y.reshape(-1)
    d0 = np.abs(p - 0.0)
    d1 = np.abs(p - l0)
    yhat = (d1 < d0).astype(int)
    ytrue = (t != 0.0).astype(int)
    return float(np.mean(yhat == ytrue))


# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------


def _sim_timing_snapshot(net: GroundReferenceNetwork) -> dict:
    """Grab cumulative timing counters from the simulator, if available."""
    sim = getattr(net, "cached_simulator", None)
    if sim is None:
        return {}
    keys = [
        "t_total",
        "t_super",
        "t_remove_destroy",
        "t_load",
        "t_reset",
        "t_run",
        "t_plot",
        "t_dc_analysis",
        "n_runs",
    ]
    snap = {}
    for k in keys:
        if hasattr(sim, k):
            val = getattr(sim, k)
            snap[k] = float(val)
    if "n_runs" in snap:
        snap["n_runs"] = int(snap["n_runs"])
    return snap


def _mem_snapshot() -> dict:
    """Capture lightweight memory/cache-related stats for the current process."""
    rss_mb = 0.0
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_kb = float(parts[1])
                        rss_mb = rss_kb / 1024.0
                    break
    except FileNotFoundError:
        rss_mb = 0.0

    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        minflt = int(ru.ru_minflt)
        majflt = int(ru.ru_majflt)
    except Exception:
        minflt = 0
        majflt = 0

    return {
        "rss_mb": float(rss_mb),
        "minor_faults": int(minflt),
        "major_faults": int(majflt),
    }


def _outputs_from_analysis(net: GroundReferenceNetwork, analysis) -> np.ndarray:
    """Compute network outputs from an existing DC analysis object.

    Parameters
    ----------
    net : GroundReferenceNetwork
    analysis : PySpice.Probe.WaveForm.DcAnalysis

    Returns
    -------
    out : ndarray, shape (n_samples, n_out)
    """
    node0 = str(net.__nodes__[0])
    n_examples = len(analysis.nodes[node0])
    out = np.zeros((len(net.outputs), n_examples), dtype=float)
    for i, vsrc in enumerate(net.outputs):
        a, b = vsrc.node_names
        out[i] = u_V(analysis.nodes[a] - analysis.nodes[b])
    return out.T


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_loop(
    net: GroundReferenceNetwork,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int,
    eta: float,
    gamma: float,
    run_dir: Path,
):
    """Main XOR training loop (per-sample free/clamped solves).

    Parameters
    ----------
    net : GroundReferenceNetwork
    X : ndarray, shape (N, 4)
    Y : ndarray, shape (N, 1)
    epochs : int
    eta : float
    gamma : float
    run_dir : Path

    Returns
    -------
    losses, accs, outputs, epoch_total_times, epoch_sim_times
    """
    # Precompute edge endpoint indices (into the node array) for faster updates.
    e1, e2 = [], []
    for E in net.edges:
        a, b = list(map(int, E.circ.node_names[:2]))
        e1.append(a)
        e2.append(b)
    e1 = np.array(e1, dtype=int)
    e2 = np.array(e2, dtype=int)

    losses, accs, outputs = [], [], []
    epoch_total_times, epoch_sim_times = [], []

    # Initial metrics before any learning.
    preds = net.predict(X)
    losses.append(float(np.mean((Y - preds) ** 2)))
    accs.append(accuracy_nearest_target(preds, Y))
    outputs.append(preds.copy())

    for ep in range(epochs):
        epoch_idx = ep + 1
        epoch_t0 = time.time()
        mem_before = _mem_snapshot()

        # Per-epoch timing accumulators.
        t_train_setup = 0.0
        t_train_free = 0.0
        t_train_clamp = 0.0
        t_train_update = 0.0
        t_train_metrics = 0.0
        n_free_calls = 0
        n_clamp_calls = 0

        sim_before = _sim_timing_snapshot(net)

        # Shuffle sample order each epoch.
        setup_t0 = time.time()
        order = np.arange(len(X))
        np.random.shuffle(order)
        t_train_setup += float(time.time() - setup_t0)

        for idx in order:
            # Free phase: solve once and reuse the analysis to get outputs.
            free_t0 = time.time()
            free_analysis = net._solve(X[idx])
            t_train_free += float(time.time() - free_t0)
            n_free_calls += 1

            free_out = _outputs_from_analysis(net, free_analysis)[0]

            # Compute target blend and clamp outputs around it.
            nudges = eta * Y[idx] + (1.0 - eta) * free_out
            clamp_t0 = time.time()
            clamped_analysis = net._solve(X[idx], nudges.reshape(Y[idx].shape))
            t_train_clamp += float(time.time() - clamp_t0)
            n_clamp_calls += 1

            # Edge-wise update from free vs clamped edge voltages.
            upd_t0 = time.time()
            free_nodes = np.array(
                [float(u_V(free_analysis.nodes[str(i)])[0]) for i in net.__nodes__],
                dtype=float,
            )
            clamp_nodes = np.array(
                [float(u_V(clamped_analysis.nodes[str(i)])[0]) for i in net.__nodes__],
                dtype=float,
            )
            free_diffs = free_nodes[e1] - free_nodes[e2]
            clamped_diffs = clamp_nodes[e1] - clamp_nodes[e2]
            update = -gamma * (clamped_diffs**2 - free_diffs**2)
            net.update(update)
            t_train_update += float(time.time() - upd_t0)

        # Epoch-level metrics over the full dataset.
        metrics_t0 = time.time()
        preds = net.predict(X)
        losses.append(float(np.mean((Y - preds) ** 2)))
        accs.append(accuracy_nearest_target(preds, Y))
        outputs.append(preds.copy())
        t_train_metrics += float(time.time() - metrics_t0)

        epoch_total = float(time.time() - epoch_t0)
        epoch_sim = float(t_train_free + t_train_clamp)
        epoch_total_times.append(epoch_total)
        epoch_sim_times.append(epoch_sim)

        sim_after = _sim_timing_snapshot(net)
        mem_after = _mem_snapshot()

        sim_delta = {}
        if sim_before and sim_after:
            for k in sim_after:
                if k in sim_before:
                    sim_delta[k] = float(sim_after[k] - sim_before[k])

        minor_epoch = int(mem_after["minor_faults"] - mem_before["minor_faults"])
        major_epoch = int(mem_after["major_faults"] - mem_before["major_faults"])

        timing_epoch = {
            "epoch": int(epoch_idx),
            "t_epoch_total": epoch_total,
            "t_train_setup": float(t_train_setup),
            "t_train_free": float(t_train_free),
            "t_train_clamp": float(t_train_clamp),
            "t_train_update": float(t_train_update),
            "t_train_metrics": float(t_train_metrics),
            "epoch_sim_approx_free_plus_clamp": epoch_sim,
            "n_free_calls": int(n_free_calls),
            "n_clamp_calls": int(n_clamp_calls),
            "sim_timing_delta": sim_delta,
            "rss_mb_epoch_end": float(mem_after["rss_mb"]),
            "minor_faults_epoch": minor_epoch,
            "major_faults_epoch": major_epoch,
            "minor_faults_total": int(mem_after["minor_faults"]),
            "major_faults_total": int(mem_after["major_faults"]),
        }
        try:
            with open(run_dir / f"0_timing_epoch{epoch_idx}.json", "w") as f:
                json.dump(timing_epoch, f, indent=2)
        except Exception:
            pass

        # Checkpoint metrics each epoch.
        try:
            np.save(run_dir / "0_losses.npy", np.asarray(losses, dtype=float))
            np.save(run_dir / "0_acc.npy", np.asarray(accs, dtype=float))
            np.save(run_dir / "0_outputs.npy", np.array(outputs, dtype=object))
            np.save(run_dir / "0_epoch_times.npy", np.asarray(epoch_total_times, dtype=float))
            np.save(run_dir / "0_epoch_sim_times.npy", np.asarray(epoch_sim_times, dtype=float))
        except Exception:
            pass

        if (ep % 10) == 0:
            print(
                f"Epoch {epoch_idx:4d}: loss={losses[-1]:.6f} acc={accs[-1]:.3f} "
                f"(t_total={epoch_total:.4f}s, t_free={t_train_free:.4f}s, "
                f"t_clamp={t_train_clamp:.4f}s)",
                flush=True,
            )

    return losses, accs, outputs, epoch_total_times, epoch_sim_times


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    np.random.seed(int(args.seed))

    side = int(args.side)
    use_klu = args.solver == "klu"

    # Map body-mode string to flags.
    if args.body_mode == "source":
        body_to_ground = False
        floating_body = False
    elif args.body_mode == "ground":
        body_to_ground = True
        floating_body = False
    else:  # "floating"
        body_to_ground = False
        floating_body = True

    run_t0 = time.time()

    graph_t0 = time.time()
    grid, node_cfg = build_xor_graph_nxn(side)
    t_graph_build = float(time.time() - graph_t0)

    from sim.spice_net import GroundReferenceNetwork  # re-import to avoid circular issues

    net_t0 = time.time()
    net = GroundReferenceNetwork(
        name=f"xor{side}x{side}",
        con_graph=grid,
        node_cfg=node_cfg,
        body_to_ground=body_to_ground,
        floating_body=floating_body,
        epsilon=1e-9,
        backend=args.backend,
        use_klu=use_klu,
    )
    t_net_init = float(time.time() - net_t0)

    data_t0 = time.time()
    X, Y = xor_dataset()
    t_dataset_build = float(time.time() - data_t0)

    # Allow overriding results root via RUN_ROOT (for organising runs).
    env_root = os.environ.get("RUN_ROOT")
    if env_root:
        results_dir = Path(env_root)
    else:
        results_dir = Path(__file__).resolve().parent / "results"
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_N-{side}_seed-{args.seed}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "script": str(Path(__file__).resolve()),
        "script_name": Path(__file__).name,
        "argv": list(os.sys.argv),
        "seed": str(args.seed),
        "timestamp": datetime.now().isoformat(),
        "cwd": str(Path.cwd()),
        "python": os.sys.version,
        "epochs": int(args.epochs),
        "eta": float(args.eta),
        "gamma": float(args.gamma),
        "grid_side": int(side),
        "variant": "xor_nxn",
        "backend": args.backend,
        "solver": args.solver,
        "body_mode": args.body_mode,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    losses, accs, outputs, etot, esim = train_loop(
        net, X, Y, int(args.epochs), float(args.eta), float(args.gamma), run_dir
    )

    # Final checkpoint of arrays.
    np.save(run_dir / "0_losses.npy", np.asarray(losses, dtype=float))
    np.save(run_dir / "0_acc.npy", np.asarray(accs, dtype=float))
    np.save(run_dir / "0_outputs.npy", np.array(outputs, dtype=object))
    np.save(run_dir / "0_epoch_times.npy", np.asarray(etot, dtype=float))
    np.save(run_dir / "0_epoch_sim_times.npy", np.asarray(esim, dtype=float))

    # Store the final graph topology as GraphML.
    nx.write_graphml(grid, str(run_dir / "0.graphml"))
    nx.write_graphml(grid, str(run_dir / "0_best.graphml"))

    latest = results_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
    except Exception:
        pass
    latest.symlink_to(run_dir.resolve())

    run_total = float(time.time() - run_t0)
    timing_global = {
        "t_graph_build": float(t_graph_build),
        "t_net_init": float(t_net_init),
        "t_dataset_build": float(t_dataset_build),
        "t_run_total": run_total,
    }
    try:
        with open(run_dir / "0_timing_global.json", "w") as f:
            json.dump(timing_global, f, indent=2)
    except Exception:
        pass

    print(f"FINAL acc={accs[-1]:.4f}")


if __name__ == "__main__":
    main()
