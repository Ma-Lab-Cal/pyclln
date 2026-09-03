#!/usr/bin/env python3
"""Deployment inference for a trained scikit-digits run: loads a chip fingerprint + the
endpoint gates from a run archive and evaluates the test set N times, each draw with a
fresh, independent read-noise stream (device mismatch stays fixed — it is the chip).
Usage: python infer_scikit.py --chip chip_1.npz --run results/run_chip1_gates.npz [--ninf 5]"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import train_scikit as T

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True)
    ap.add_argument("--run", required=True, help="run archive npz (uses vg_final endpoint gates)")
    ap.add_argument("--ninf", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meas-rel", type=float, default=5e-3)
    a = ap.parse_args()

    topo = np.load(HERE / "topology_1247.npz")
    drain, source = topo["drain"], topo["source"]
    z = np.load(a.chip)
    _z = np.load(a.run)
    vg = np.asarray(_z["vg_final"], dtype=np.float64)
    sigma = a.meas_rel * 1.0

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float64)
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed, stratify=y)

    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(T.build_netlist(drain, source, vg, z["vto_free"], z["vto_clamp"]))
    outA = [900 + j for j in range(20)]

    accs = []
    for j in range(a.ninf):
        rng = np.random.default_rng(555000 + 7919 * j)
        correct = 0
        for xt, yt in zip(Xte, yte):
            T.exec_chunked(ng, [f"alter VIN{p} dc = {xt[p]:.8f}" for p in range(64)])
            V = T.read_nodes(ng, outA, rng, sigma)
            correct += int(int(np.argmax(V[:10] - V[10:])) == int(yt))
        accs.append(correct / len(yte))
    print(f">> INFER{a.ninf} {Path(a.chip).name} @ {Path(a.run).name}: "
          f"{[round(x, 4) for x in accs]} mean={np.mean(accs):.4f}", flush=True)


if __name__ == "__main__":
    main()
