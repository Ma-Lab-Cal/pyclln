#!/usr/bin/env python3
"""
Extended stack + integration tests for XOR N×N trainer + sim stack.

Covers:

  • xor_dataset / accuracy_nearest_target (basic + custom + boundary)
  • build_xor_graph_nxn (multiple sizes, connectivity sanity)
  • GroundReferenceNetwork:
      - both backends (shared / subprocess)
      - body modes (source / ground / floating)
      - KLU flag
      - repeated predict consistency
      - larger grids
      - input shape handling (1D vs 2D)
  • StatePreservingNgSpiceSimulator timing counters
  • _outputs_from_analysis vs predict (single + multi-sample)
  • _sim_timing_snapshot / _mem_snapshot progress
  • train_loop:
      - small and medium grids
      - both backends
      - reproducibility with fixed seed
      - gamma = 0 (no learning)
      - epochs = 0 (no training)
  • CLI-level integration:
      - calling xor_nxn_trainer.main() with RUN_ROOT
      - verifying on-disk outputs + run_meta consistency
      - shared + subprocess backends
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from xor_nxn_trainer import (  # type: ignore[import]
    build_xor_graph_nxn,
    xor_dataset,
    accuracy_nearest_target,
    _sim_timing_snapshot,
    _mem_snapshot,
    _outputs_from_analysis,
    train_loop,
)
import xor_nxn_trainer as trainer_mod  # for CLI integration
from sim.spice_net import GroundReferenceNetwork  # type: ignore[import]
from PySpice.Unit import u_V  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print("=" * 80)
    print(f"TEST: {title}")
    print("-" * 80)


def _run_test(title: str, fn):
    _banner(title)
    try:
        fn()
        print("  OK.\n")
    except Exception as exc:  # noqa: BLE001
        print(f"!!! TEST FAILED: {fn.__name__}")
        print(f"     {type(exc).__name__} : {exc}")
        traceback.print_exc()
        print()
        raise


def _make_net(
    side: int = 4,
    backend: str = "shared",
    use_klu: bool = False,
    body_mode: str = "source",
) -> GroundReferenceNetwork:
    """Factory for a GroundReferenceNetwork with various options."""
    grid, node_cfg = build_xor_graph_nxn(side)

    if body_mode == "source":
        body_to_ground = False
        floating_body = False
    elif body_mode == "ground":
        body_to_ground = True
        floating_body = False
    elif body_mode == "floating":
        body_to_ground = False
        floating_body = True
    else:
        raise ValueError(f"Unknown body_mode '{body_mode}'")

    net = GroundReferenceNetwork(
        name=f"xor{side}x{side}",
        con_graph=grid,
        node_cfg=node_cfg,
        body_to_ground=body_to_ground,
        floating_body=floating_body,
        epsilon=1e-9,
        backend=backend,
        use_klu=use_klu,
    )
    return net


def _expected_edge_count(side: int) -> int:
    """Expected number of edges in an N×N periodic grid_graph (undirected).

    For N >= 3: each node has degree 4 ⇒ E = N²·4 / 2 = 2 N².
    For N = 2: periodic wrap creates duplicate neighbour pairs that collapse
    in an undirected graph, so you only get 4 distinct edges.
    """
    if side == 2:
        return 4
    return 2 * side * side


def _run_cli_once(
    side: int,
    backend: str,
    solver: str,
    body_mode: str,
    epochs: int = 2,
    seed: int = 123,
):
    """Call xor_nxn_trainer.main() as if from CLI and validate on-disk outputs."""
    tmp_root = Path(__file__).resolve().parent / "tmp_cli_runs"
    tmp_root.mkdir(exist_ok=True)

    run_root = tmp_root / f"cli_{backend}_{solver}_{body_mode}_N{side}_{int(time.time() * 1e6)}"
    run_root.mkdir(exist_ok=False)

    # Route trainer to our temp root via RUN_ROOT.
    prev_run_root = os.environ.get("RUN_ROOT")
    os.environ["RUN_ROOT"] = str(run_root)

    # Patch sys.argv to simulate a real CLI call.
    argv_save = sys.argv[:]
    sys.argv = [
        "xor_nxn_trainer.py",
        "--side",
        str(side),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--eta",
        "0.5",
        "--gamma",
        "0.2",
        "--backend",
        backend,
        "--solver",
        solver,
        "--body-mode",
        body_mode,
    ]

    try:
        trainer_mod.main()
    finally:
        # Restore argv and RUN_ROOT.
        sys.argv = argv_save
        if prev_run_root is None:
            os.environ.pop("RUN_ROOT", None)
        else:
            os.environ["RUN_ROOT"] = prev_run_root

    # Now validate structure of outputs.
    runs_dir = run_root / "runs"
    assert runs_dir.is_dir(), "runs/ directory not created"

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
    rd = run_dirs[0]

    # Required files.
    required_files = [
        "0_losses.npy",
        "0_acc.npy",
        "0_outputs.npy",
        "0_epoch_times.npy",
        "0_epoch_sim_times.npy",
        "run_meta.json",
        "0_timing_global.json",
        "0.graphml",
        "0_best.graphml",
    ]
    for fname in required_files:
        path = rd / fname
        assert path.exists(), f"Missing expected file: {path}"

    meta = json.loads((rd / "run_meta.json").read_text())
    assert meta["grid_side"] == side
    assert meta["backend"] == backend
    assert meta["solver"] == solver
    assert meta["body_mode"] == body_mode
    assert meta["epochs"] == epochs

    losses = np.load(rd / "0_losses.npy")
    accs = np.load(rd / "0_acc.npy")
    outputs = np.load(rd / "0_outputs.npy", allow_pickle=True)
    etot = np.load(rd / "0_epoch_times.npy")
    esim = np.load(rd / "0_epoch_sim_times.npy")

    assert losses.shape[0] == epochs + 1
    assert accs.shape[0] == epochs + 1
    assert outputs.shape[0] == epochs + 1
    assert etot.shape[0] == epochs
    assert esim.shape[0] == epochs
    assert np.all(np.isfinite(losses))
    assert np.all((0.0 <= accs) & (accs <= 1.0))

    # Check 'latest' symlink.
    latest = run_root / "latest"
    assert latest.exists(), "'latest' symlink missing"
    assert latest.is_symlink(), "'latest' is not a symlink"
    assert latest.resolve() == rd.resolve(), "latest symlink does not point to newest run"

    return rd, meta


# ---------------------------------------------------------------------------
# Tests: xor_dataset / accuracy_nearest_target
# ---------------------------------------------------------------------------

def test_xor_dataset_basic():
    X, Y = xor_dataset()
    print("  X shape:", X.shape)
    print("  Y shape:", Y.shape)
    print("  X:\n", X)
    print("  Y:", Y.reshape(-1))

    assert X.shape == (4, 4)
    assert Y.shape == (4, 1)
    # XOR truth table check: patterns 01,10 => nonzero target
    assert Y[1, 0] != 0.0 and Y[2, 0] != 0.0
    assert Y[0, 0] == 0.0 and Y[3, 0] == 0.0

    acc_self = accuracy_nearest_target(Y, Y)
    print("  Self-accuracy accuracy_nearest_target(Y, Y):", acc_self)
    assert acc_self == 1.0


def test_xor_dataset_custom_params():
    X, Y = xor_dataset(I_pos=0.5, I_neg=0.2, I0=0.7, L0=-0.05)
    print("  X(0):", X[0])
    print("  Y:", Y.reshape(-1))

    assert np.allclose(X[0], [0.2, 0.5, 0.0, 0.0])
    # Targets should be {0, L0}
    assert set(np.unique(Y.round(6))) <= {0.0, -0.05}


def test_accuracy_nearest_target_boundary_behavior():
    # Preds exactly at halfway between 0 and L0 -> d0 == d1 -> yhat=0
    L0 = -0.1
    mid = L0 / 2.0
    preds = np.array([[mid], [mid]])
    Y0 = np.array([[0.0], [L0]])  # one should be 0, one should be L0

    acc = accuracy_nearest_target(preds, Y0, l0=L0)
    print("  boundary preds:", preds.reshape(-1))
    print("  targets:", Y0.reshape(-1))
    print("  accuracy at boundary:", acc)

    # yhat=(d1<d0) so ties go to 0; only the 0-target sample is correct
    assert acc == 0.5


# ---------------------------------------------------------------------------
# Tests: build_xor_graph_nxn
# ---------------------------------------------------------------------------

def test_build_xor_graph_small_sides():
    for side in (2, 3, 4):
        grid, node_cfg = build_xor_graph_nxn(side)
        inputs, outputs = node_cfg
        print(f"  side={side} -> nodes={grid.number_of_nodes()} edges={grid.number_of_edges()}")

        assert grid.number_of_nodes() == side * side + 1  # extra reference node
        assert grid.number_of_edges() == _expected_edge_count(side)

        assert inputs.shape == (4, 2)
        assert outputs.shape == (1, 2)

        # Check that all node indices in node_cfg are valid.
        max_node = grid.number_of_nodes() - 1
        assert np.all(inputs >= 0) and np.all(inputs <= max_node)
        assert np.all(outputs >= 0) and np.all(outputs <= max_node)


def test_build_xor_graph_medium_side():
    side = 8
    grid, node_cfg = build_xor_graph_nxn(side)
    inputs, outputs = node_cfg
    print(f"  side={side} -> nodes={grid.number_of_nodes()} edges={grid.number_of_edges()}")

    assert grid.number_of_nodes() == side * side + 1
    assert grid.number_of_edges() == _expected_edge_count(side)
    assert inputs.shape == (4, 2)
    assert outputs.shape == (1, 2)


# ---------------------------------------------------------------------------
# Tests: GroundReferenceNetwork (backends, body modes, shapes, etc.)
# ---------------------------------------------------------------------------

def test_ground_reference_network_backends_predict():
    X, _ = xor_dataset()
    for backend in ("shared", "subprocess"):
        print(f"  backend={backend}")
        net = _make_net(side=4, backend=backend, use_klu=False, body_mode="source")
        preds = net.predict(X)
        sol = net.solve(X[0])
        print("    preds shape:", preds.shape)
        print("    solve(X[0]) shape:", sol.shape)

        assert preds.shape == (4, 1)
        assert sol.shape[0] == net.__nodes__.shape[0]
        assert sol.shape[1] == 1
        assert np.all(np.isfinite(preds))
        assert np.all(np.isfinite(sol))


def test_ground_reference_network_body_modes():
    X, _ = xor_dataset()
    for body_mode in ("source", "ground", "floating"):
        print(f"  body_mode={body_mode}")
        net = _make_net(side=4, backend="shared", use_klu=False, body_mode=body_mode)
        preds = net.predict(X)
        assert preds.shape == (4, 1)
        assert np.all(np.isfinite(preds))


def test_ground_reference_network_repeated_predict_consistency():
    X, _ = xor_dataset()
    for backend in ("shared", "subprocess"):
        print(f"  backend={backend}")
        net = _make_net(side=4, backend=backend, use_klu=False, body_mode="source")

        p1 = net.predict(X)
        p2 = net.predict(X)
        diff = np.max(np.abs(p1 - p2))
        print("    max |p1 - p2|:", diff)
        # Allow tiny numerical drift.
        assert diff < 1e-9


def test_ground_reference_network_use_klu_flag():
    X, _ = xor_dataset()
    for backend in ("shared", "subprocess"):
        print(f"  backend={backend}, use_klu=True")
        net = _make_net(side=4, backend=backend, use_klu=True, body_mode="source")
        preds = net.predict(X)
        assert preds.shape == (4, 1)
        assert np.all(np.isfinite(preds))


def test_ground_reference_network_larger_grid_predict():
    X, _ = xor_dataset()
    for side in (6, 8):
        print(f"  side={side}")
        net = _make_net(side=side, backend="shared", use_klu=False, body_mode="source")
        preds = net.predict(X)
        sol = net.solve(X[0])
        print("    preds shape:", preds.shape, "solve shape:", sol.shape)
        assert preds.shape == (4, 1)
        assert sol.shape[0] == side * side + 1
        assert np.all(np.isfinite(preds))
        assert np.all(np.isfinite(sol))


def test_ground_reference_network_predict_input_shapes():
    """Check that predict() behaves correctly for (4,), (1,4), and (4,4) inputs."""
    X_full, _ = xor_dataset()
    x_vec = X_full[0]          # shape (4,)
    X_row = X_full[0:1, :]     # shape (1,4)

    net = _make_net(side=4, backend="shared", use_klu=False, body_mode="source")

    p_full = net.predict(X_full)    # (4,1)
    p_vec = net.predict(x_vec)      # (1,1)
    p_row = net.predict(X_row)      # (1,1)

    print("  p_full shape:", p_full.shape)
    print("  p_vec shape :", p_vec.shape)
    print("  p_row shape :", p_row.shape)

    assert p_full.shape == (4, 1)
    assert p_vec.shape == (1, 1)
    assert p_row.shape == (1, 1)

    # First sample of full batch should match 1-sample calls.
    assert np.allclose(p_vec[0, 0], p_full[0, 0])
    assert np.allclose(p_row[0, 0], p_full[0, 0])


# ---------------------------------------------------------------------------
# Tests: StatePreservingNgSpiceSimulator timing
# ---------------------------------------------------------------------------

def test_state_preserving_timing_multiple_runs():
    X, _ = xor_dataset()
    net = _make_net(side=4, backend="shared", use_klu=False, body_mode="source")
    sim = net.cached_simulator
    t0 = getattr(sim, "t_total", 0.0)
    n0 = getattr(sim, "n_runs", 0)
    print("  initial t_total:", t0, "n_runs:", n0)

    # Two predict calls -> two dc() calls.
    net.predict(X)
    net.predict(X)

    t1 = getattr(sim, "t_total", 0.0)
    n1 = getattr(sim, "n_runs", 0)
    print("  after two predicts t_total:", t1, "n_runs:", n1)

    assert n1 >= n0 + 2
    assert t1 > t0


# ---------------------------------------------------------------------------
# Tests: _outputs_from_analysis
# ---------------------------------------------------------------------------

def test_outputs_from_analysis_matches_predict_multi_sample():
    X, _ = xor_dataset()
    for backend in ("shared", "subprocess"):
        print(f"  backend={backend}")
        net = _make_net(side=4, backend=backend, use_klu=False, body_mode="source")

        analysis = net._solve(X)  # 4 samples in one DC sweep
        out_from_analysis = _outputs_from_analysis(net, analysis)
        preds = net.predict(X)

        diff = np.max(np.abs(out_from_analysis - preds))
        print("    max |out_from_analysis - preds|:", diff)

        assert out_from_analysis.shape == preds.shape
        assert diff < 1e-12


# ---------------------------------------------------------------------------
# Tests: _sim_timing_snapshot and _mem_snapshot
# ---------------------------------------------------------------------------

def test_sim_and_mem_snapshots_progress():
    X, _ = xor_dataset()
    net = _make_net(side=4, backend="shared", use_klu=False, body_mode="source")

    snap0 = _sim_timing_snapshot(net)
    mem0 = _mem_snapshot()
    print("  sim snap0:", snap0)
    print("  mem snap0:", mem0)

    net.predict(X)  # one run

    snap1 = _sim_timing_snapshot(net)
    mem1 = _mem_snapshot()
    print("  sim snap1:", snap1)
    print("  mem snap1:", mem1)

    assert snap1.get("n_runs", 0) >= snap0.get("n_runs", 0) + 1
    assert mem1["rss_mb"] >= 0.0
    assert mem1["minor_faults"] >= mem0["minor_faults"]


# ---------------------------------------------------------------------------
# Tests: train_loop (small / medium / edge cases)
# ---------------------------------------------------------------------------

def test_train_loop_small_run_shared_and_subprocess():
    X, Y = xor_dataset()
    tmp_root = Path(__file__).resolve().parent / "tmp_test_runs"
    tmp_root.mkdir(exist_ok=True)

    for backend in ("shared", "subprocess"):
        print(f"  backend={backend}")
        run_dir = tmp_root / f"train_small_{backend}_{int(time.time() * 1e6)}"
        run_dir.mkdir(exist_ok=False)

        net = _make_net(side=4, backend=backend, use_klu=False, body_mode="source")

        # Record initial gate voltages.
        v_init = np.array([e.get_val() for e in net.edges], dtype=float)

        losses, accs, outputs, etot, esim = train_loop(
            net=net,
            X=X,
            Y=Y,
            epochs=3,
            eta=0.5,
            gamma=0.2,
            run_dir=run_dir,
        )

        print("    losses:", losses)
        print("    accs:", accs)
        print("    epoch_total_times:", etot)
        print("    epoch_sim_times:", esim)
        print("    final acc:", accs[-1])

        assert len(losses) == 4  # initial + 3 epochs
        assert len(accs) == 4
        assert len(etot) == 3
        assert len(esim) == 3

        v_final = np.array([e.get_val() for e in net.edges], dtype=float)
        delta_v = np.max(np.abs(v_final - v_init))
        print("    max |Δv_gate|:", delta_v)
        assert delta_v > 0.0


def test_train_loop_medium_grid_shared():
    X, Y = xor_dataset()
    tmp_root = Path(__file__).resolve().parent / "tmp_test_runs"
    tmp_root.mkdir(exist_ok=True)

    side = 6
    run_dir = tmp_root / f"train_medium_shared_{side}_{int(time.time() * 1e6)}"
    run_dir.mkdir(exist_ok=False)

    net = _make_net(side=side, backend="shared", use_klu=False, body_mode="source")

    losses, accs, outputs, etot, esim = train_loop(
        net=net,
        X=X,
        Y=Y,
        epochs=3,
        eta=0.5,
        gamma=0.2,
        run_dir=run_dir,
    )

    print("  side:", side)
    print("  losses:", losses)
    print("  accs:", accs)
    assert len(losses) == 4
    assert len(accs) == 4
    assert all(np.isfinite(l) for l in losses)
    assert all(0.0 <= a <= 1.0 for a in accs)


def test_train_loop_reproducibility_shared():
    """Fixed seed → runs should be numerically identical (within tolerance)."""
    tmp_root = Path(__file__).resolve().parent / "tmp_test_runs"
    tmp_root.mkdir(exist_ok=True)

    def run_once(seed: int):
        np.random.seed(seed)
        X, Y = xor_dataset()
        side = 4
        run_dir = tmp_root / f"train_repro_{seed}_{int(time.time() * 1e6)}"
        run_dir.mkdir(exist_ok=False)
        net = _make_net(side=side, backend="shared", use_klu=False, body_mode="source")
        losses, accs, outputs, etot, esim = train_loop(
            net=net,
            X=X,
            Y=Y,
            epochs=3,
            eta=0.5,
            gamma=0.2,
            run_dir=run_dir,
        )
        return np.array(losses), np.array(accs)

    losses1, accs1 = run_once(seed=123)
    losses2, accs2 = run_once(seed=123)

    print("  losses1:", losses1)
    print("  losses2:", losses2)
    print("  accs1:", accs1)
    print("  accs2:", accs2)

    assert np.allclose(losses1, losses2, rtol=1e-6, atol=1e-8)
    assert np.allclose(accs1, accs2, rtol=1e-6, atol=1e-8)


def test_train_loop_gamma_zero_no_learning():
    """With gamma=0, gate voltages should not change across epochs."""
    X, Y = xor_dataset()
    tmp_root = Path(__file__).resolve().parent / "tmp_test_runs"
    tmp_root.mkdir(exist_ok=True)

    run_dir = tmp_root / f"train_gamma0_{int(time.time() * 1e6)}"
    run_dir.mkdir(exist_ok=False)

    net = _make_net(side=4, backend="shared", use_klu=False, body_mode="source")

    v_init = np.array([e.get_val() for e in net.edges], dtype=float)

    losses, accs, outputs, etot, esim = train_loop(
        net=net,
        X=X,
        Y=Y,
        epochs=3,
        eta=0.5,
        gamma=0.0,   # no learning
        run_dir=run_dir,
    )

    v_final = np.array([e.get_val() for e in net.edges], dtype=float)
    delta_v = np.max(np.abs(v_final - v_init))
    print("  gamma=0: max |Δv_gate|:", delta_v)

    assert delta_v < 1e-12  # effectively unchanged


def test_train_loop_zero_epochs():
    """epochs=0 → only initial metrics, no timing arrays."""
    X, Y = xor_dataset()
    tmp_root = Path(__file__).resolve().parent / "tmp_test_runs"
    tmp_root.mkdir(exist_ok=True)

    run_dir = tmp_root / f"train_epochs0_{int(time.time() * 1e6)}"
    run_dir.mkdir(exist_ok=False)

    net = _make_net(side=4, backend="shared", use_klu=False, body_mode="source")

    losses, accs, outputs, etot, esim = train_loop(
        net=net,
        X=X,
        Y=Y,
        epochs=0,
        eta=0.5,
        gamma=0.2,
        run_dir=run_dir,
    )

    print("  epochs=0 losses:", losses)
    print("  epochs=0 accs:", accs)
    assert len(losses) == 1
    assert len(accs) == 1
    assert len(etot) == 0
    assert len(esim) == 0


# ---------------------------------------------------------------------------
# Integration tests: CLI-level main()
# ---------------------------------------------------------------------------

def test_cli_integration_shared():
    """End-to-end: call trainer.main() (shared backend) and inspect artifacts."""
    rd, meta = _run_cli_once(
        side=4,
        backend="shared",
        solver="klu",
        body_mode="source",
        epochs=2,
        seed=321,
    )
    print("  shared CLI run_dir:", rd)
    print("  shared CLI meta:", meta)


def test_cli_integration_subprocess():
    """End-to-end: call trainer.main() (subprocess backend) and inspect artifacts."""
    rd, meta = _run_cli_once(
        side=4,
        backend="subprocess",
        solver="sparse",
        body_mode="ground",
        epochs=1,
        seed=42,
    )
    print("  subprocess CLI run_dir:", rd)
    print("  subprocess CLI meta:", meta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tests = [
        # xor_dataset / accuracy
        ("xor_dataset() basic", test_xor_dataset_basic),
        ("xor_dataset() custom params", test_xor_dataset_custom_params),
        ("accuracy_nearest_target() boundary behavior", test_accuracy_nearest_target_boundary_behavior),

        # build_xor_graph_nxn
        ("build_xor_graph_nxn() small sides", test_build_xor_graph_small_sides),
        ("build_xor_graph_nxn() medium side", test_build_xor_graph_medium_side),

        # GroundReferenceNetwork
        ("GroundReferenceNetwork backends predict", test_ground_reference_network_backends_predict),
        ("GroundReferenceNetwork body modes", test_ground_reference_network_body_modes),
        ("GroundReferenceNetwork repeated predict consistency", test_ground_reference_network_repeated_predict_consistency),
        ("GroundReferenceNetwork use_klu flag", test_ground_reference_network_use_klu_flag),
        ("GroundReferenceNetwork larger grid predict", test_ground_reference_network_larger_grid_predict),
        ("GroundReferenceNetwork predict input shapes", test_ground_reference_network_predict_input_shapes),

        # StatePreservingNgSpiceSimulator
        ("StatePreservingNgSpiceSimulator timing multiple runs", test_state_preserving_timing_multiple_runs),

        # outputs_from_analysis
        ("_outputs_from_analysis() matches predict (multi-sample)", test_outputs_from_analysis_matches_predict_multi_sample),

        # sim/mem snapshots
        ("_sim_timing_snapshot() and _mem_snapshot() progress", test_sim_and_mem_snapshots_progress),

        # train_loop
        ("train_loop() small run (shared + subprocess)", test_train_loop_small_run_shared_and_subprocess),
        ("train_loop() medium grid (shared)", test_train_loop_medium_grid_shared),
        ("train_loop() reproducibility (shared)", test_train_loop_reproducibility_shared),
        ("train_loop() gamma=0 no learning", test_train_loop_gamma_zero_no_learning),
        ("train_loop() epochs=0", test_train_loop_zero_epochs),

        # CLI integration
        ("CLI integration (shared backend)", test_cli_integration_shared),
        ("CLI integration (subprocess backend)", test_cli_integration_subprocess),
    ]

    any_fail = False
    for title, fn in tests:
        try:
            _run_test(title, fn)
        except Exception:
            any_fail = True  # continue to show all failures

    print("=" * 80)
    print("TEST RUN SUMMARY")
    if any_fail:
        print("One or more tests FAILED (see logs above).")
    else:
        print("If no 'TEST FAILED' messages appeared above, extended stack + integration tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
