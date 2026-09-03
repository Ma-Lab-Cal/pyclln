#!/usr/bin/env python3
"""
Analyze timing breakdowns for xor_nxn (subprocess vs shared) runs.

For each run under a root directory, we look for a run_meta.json and the
timing arrays:

  - 0_epoch_times.npy
  - 0_epoch_train_run_times.npy
  - 0_epoch_eval_run_times.npy
  - 0_epoch_destroy_times.npy

and (for the shared alter trainer) also:

  - 0_epoch_t_mk_free.npy
  - 0_epoch_t_alter_inputs.npy
  - 0_epoch_t_run_free.npy
  - 0_epoch_t_read_free.npy
  - 0_epoch_t_destroy_free.npy
  - 0_epoch_t_alter_outputs.npy
  - 0_epoch_t_run_clamped.npy
  - 0_epoch_t_read_clamped.npy
  - 0_epoch_t_destroy_clamped.npy
  - 0_epoch_t_weight_updates.npy

Usage example:

  python analyze_xor_nxn_timings.py \
    --root results/30x30_shared_klu_sweep_20251208-122255 \
    --backend shared \
    --solver klu \
    --side 30 \
    --variant xor_nxn_alter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _summary(arr: np.ndarray, epoch_times: np.ndarray | None = None) -> Tuple[float, float, float, float]:
    """
    Return (total, mean, std, frac_of_epoch_times).
    If epoch_times is provided, frac_of_epoch_times = total / sum(epoch_times),
    else 0.0.
    """
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    total = float(np.sum(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if epoch_times is not None and epoch_times.size > 0:
        frac = float(total / float(np.sum(epoch_times)))
    else:
        frac = 0.0
    return total, mean, std, frac


def analyze_run(run_dir: Path, meta: Dict) -> Dict:
    """
    Analyze a single run, returning a dict of summary stats plus printing them.
    """
    print(f"\n=== RUN: {run_dir} ===")
    backend = meta.get("backend", "unknown")
    solver = meta.get("solver", "unknown")
    side = meta.get("grid_side", "unknown")
    variant = meta.get("variant", "unknown")
    seed = meta.get("seed", "unknown")
    epochs = meta.get("epochs", "unknown")

    print(
        f"  backend={backend}  solver={solver}  side={side}  "
        f"variant={variant}  seed={seed}  epochs={epochs}"
    )

    epoch_times = np.load(run_dir / "0_epoch_times.npy")
    train_run = np.load(run_dir / "0_epoch_train_run_times.npy")
    eval_run = np.load(run_dir / "0_epoch_eval_run_times.npy")
    destroy = np.load(run_dir / "0_epoch_destroy_times.npy")

    tot, mean, std, _ = _summary(epoch_times)
    print(f"  epoch_times: total={tot:.4f}s  mean={mean:.4f}s  std={std:.4f}s")

    t_tr_tot, t_tr_mean, t_tr_std, t_tr_frac = _summary(train_run, epoch_times)
    print(
        "  0_epoch_train_run_times.npy: "
        f"total={t_tr_tot:.4f}s  mean={t_tr_mean:.4f}s  std={t_tr_std:.4f}s  "
        f"frac_of_epoch_times={t_tr_frac:.3f}"
    )

    t_ev_tot, t_ev_mean, t_ev_std, t_ev_frac = _summary(eval_run, epoch_times)
    print(
        "  0_epoch_eval_run_times.npy: "
        f"total={t_ev_tot:.4f}s  mean={t_ev_mean:.4f}s  std={t_ev_std:.4f}s  "
        f"frac_of_epoch_times={t_ev_frac:.3f}"
    )

    t_de_tot, t_de_mean, t_de_std, t_de_frac = _summary(destroy, epoch_times)
    print(
        "  0_epoch_destroy_times.npy: "
        f"total={t_de_tot:.4f}s  mean={t_de_mean:.4f}s  std={t_de_std:.4f}s  "
        f"frac_of_epoch_times={t_de_frac:.3f}"
    )

    # Try to load fine-grained segments; some runs might not have them (old runs)
    segments = {
        "t_mk_free": "0_epoch_t_mk_free.npy",
        "t_alter_inputs": "0_epoch_t_alter_inputs.npy",
        "t_run_free": "0_epoch_t_run_free.npy",
        "t_read_free": "0_epoch_t_read_free.npy",
        "t_destroy_free": "0_epoch_t_destroy_free.npy",
        "t_alter_outputs": "0_epoch_t_alter_outputs.npy",
        "t_run_clamped": "0_epoch_t_run_clamped.npy",
        "t_read_clamped": "0_epoch_t_read_clamped.npy",
        "t_destroy_clamped": "0_epoch_t_destroy_clamped.npy",
        "t_weight_updates": "0_epoch_t_weight_updates.npy",
    }

    seg_summaries: Dict[str, Tuple[float, float, float, float]] = {}

    # Only print segment breakdown if at least one exists
    have_any_segment = False
    for name, fname in segments.items():
        fpath = run_dir / fname
        if fpath.exists():
            have_any_segment = True
            arr = np.load(fpath)
            seg_summaries[name] = _summary(arr, epoch_times)
        else:
            seg_summaries[name] = (0.0, 0.0, 0.0, 0.0)

    if have_any_segment:
        print("  -- fine-grained segments (per epoch) --")
        for name in [
            "t_mk_free",
            "t_alter_inputs",
            "t_run_free",
            "t_read_free",
            "t_destroy_free",
            "t_alter_outputs",
            "t_run_clamped",
            "t_read_clamped",
            "t_destroy_clamped",
            "t_weight_updates",
        ]:
            tot_s, mean_s, std_s, frac_s = seg_summaries[name]
            print(
                f"    {name}: total={tot_s:.4f}s  mean={mean_s:.4f}s  "
                f"std={std_s:.4f}s  frac_of_epoch_times={frac_s:.3f}"
            )

    return {
        "backend": backend,
        "solver": solver,
        "side": side,
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "epoch_times_total": tot,
        "epoch_times_mean": mean,
        "epoch_times_std": std,
        "train_total": t_tr_tot,
        "train_frac": t_tr_frac,
        "eval_total": t_ev_tot,
        "eval_frac": t_ev_frac,
        "destroy_total": t_de_tot,
        "destroy_frac": t_de_frac,
        "segments": seg_summaries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing run_meta.json files",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Filter on meta['backend'] (e.g., shared, subprocess)",
    )
    ap.add_argument(
        "--solver",
        type=str,
        default=None,
        help="Filter on meta['solver'] (e.g., klu, sparse)",
    )
    ap.add_argument(
        "--side",
        type=int,
        default=None,
        help="Filter on meta['grid_side']",
    )
    ap.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Filter on meta['variant'] (e.g., xor_nxn_alter)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"[INFO] Searching under root: {root}")
    print(
        f"[INFO] Filters -> backend={args.backend}, solver={args.solver}, "
        f"side={args.side}, variant={args.variant}"
    )

    meta_files = list(root.rglob("run_meta.json"))
    runs: List[Dict] = []
    for mf in meta_files:
        meta = _load_json(mf)
        if args.backend is not None and meta.get("backend") != args.backend:
            continue
        if args.solver is not None and meta.get("solver") != args.solver:
            continue
        if args.side is not None and int(meta.get("grid_side", -1)) != int(args.side):
            continue
        if args.variant is not None and meta.get("variant") != args.variant:
            continue

        run_dir = mf.parent
        # require the core timing files to exist
        if not (run_dir / "0_epoch_times.npy").exists():
            continue
        if not (run_dir / "0_epoch_train_run_times.npy").exists():
            continue
        if not (run_dir / "0_epoch_destroy_times.npy").exists():
            continue

        runs.append({"dir": run_dir, "meta": meta})

    print(f"[INFO] Found {len(runs)} matching runs.")

    if not runs:
        return

    summaries: List[Dict] = []
    for r in runs:
        summaries.append(analyze_run(r["dir"], r["meta"]))

    # Grouped summary
    print("\n================ GROUPED SUMMARY ================\n")
    # Group by backend, solver, side, variant
    groups: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    for s in summaries:
        key = (
            str(s["backend"]),
            str(s["solver"]),
            str(s["side"]),
            str(s["variant"]),
        )
        groups.setdefault(key, []).append(s)

    for (backend, solver, side, variant), ss in groups.items():
        print(f"--- Group: backend={backend}, solver={solver}, side={side}, variant={variant} ---")
        print(f"  #runs: {len(ss)}")

        total_time_per_run = np.array([s["epoch_times_total"] for s in ss], dtype=float)
        mean_epoch_time_per_run = np.array([s["epoch_times_mean"] for s in ss], dtype=float)

        print(
            f"  total_time_per_run: mean={np.mean(total_time_per_run):.4f}s "
            f"std={np.std(total_time_per_run):.4f}s"
        )
        print(
            f"  mean_epoch_time_per_run: mean={np.mean(mean_epoch_time_per_run):.4f}s "
            f"std={np.std(mean_epoch_time_per_run):.4f}s"
        )

        train_total = np.array([s["train_total"] for s in ss], dtype=float)
        train_frac = np.array([s["train_frac"] for s in ss], dtype=float)
        eval_total = np.array([s["eval_total"] for s in ss], dtype=float)
        eval_frac = np.array([s["eval_frac"] for s in ss], dtype=float)
        destroy_total = np.array([s["destroy_total"] for s in ss], dtype=float)
        destroy_frac = np.array([s["destroy_frac"] for s in ss], dtype=float)

        print(
            f"  0_epoch_train_run_times.npy: total_per_run mean={np.mean(train_total):.4f}s "
            f"std={np.std(train_total):.4f}s mean_frac_of_epoch_times={np.mean(train_frac):.3f}"
        )
        print(
            f"  0_epoch_eval_run_times.npy: total_per_run mean={np.mean(eval_total):.4f}s "
            f"std={np.std(eval_total):.4f}s mean_frac_of_epoch_times={np.mean(eval_frac):.3f}"
        )
        print(
            f"  0_epoch_destroy_times.npy: total_per_run mean={np.mean(destroy_total):.4f}s "
            f"std={np.std(destroy_total):.4f}s mean_frac_of_epoch_times={np.mean(destroy_frac):.3f}"
        )

        # Aggregate segments if present
        seg_names = [
            "t_mk_free",
            "t_alter_inputs",
            "t_run_free",
            "t_read_free",
            "t_destroy_free",
            "t_alter_outputs",
            "t_run_clamped",
            "t_read_clamped",
            "t_destroy_clamped",
            "t_weight_updates",
        ]
        # Only print if at least one run has nonzero segments
        any_seg = any(
            (s["segments"].get(name, (0.0, 0.0, 0.0, 0.0))[0] > 0.0)
            for s in ss
            for name in seg_names
        )
        if any_seg:
            print("  -- grouped fine-grained segments --")
            for name in seg_names:
                totals = np.array(
                    [s["segments"].get(name, (0.0, 0.0, 0.0, 0.0))[0] for s in ss],
                    dtype=float,
                )
                fracs = np.array(
                    [s["segments"].get(name, (0.0, 0.0, 0.0, 0.0))[3] for s in ss],
                    dtype=float,
                )
                if np.all(totals == 0.0):
                    continue
                print(
                    f"    {name}: total_per_run mean={np.mean(totals):.4f}s "
                    f"std={np.std(totals):.4f}s mean_frac_of_epoch_times={np.mean(fracs):.3f}"
                )


if __name__ == "__main__":
    main()
