#!/usr/bin/env python3
"""Deployment inference for a trained ionosphere run: loads a chip fingerprint + the
endpoint gates from a run archive and evaluates the validation set N times, each draw with
a fresh, independent read-noise stream (device mismatch stays fixed; it is the chip).
Usage: python infer_ionosphere.py --chip chip_1.npz --run results/gates_chip1.npz [--ninf 5]"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import train_ionosphere as T

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True)
    ap.add_argument("--run", required=True, help="run archive npz (uses vg_final or vg_best)")
    ap.add_argument("--ninf", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meas-rel", type=float, default=5e-3)
    ap.add_argument("--input-scale", type=float, default=0.8)
    a = ap.parse_args()

    topo = np.load(HERE / "topology_147.npz")
    out = topo["out_nodes"]
    z = np.load(a.chip)
    run = np.load(a.run)
    if "vg_final" in run.files:
        vg = np.asarray(run["vg_final"], dtype=np.float64)
    else:
        vg = np.asarray(run["vg_best"], dtype=np.float64)
    sigma = a.meas_rel * float(a.input_scale)

    X, y = T.load_ionosphere(str(HERE / "ionosphere.data"))
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed, stratify=y)

    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(T.build_netlist(topo, vg, True, z["vto_free"], z["vto_clamp"]))

    accs = []
    for j in range(a.ninf):
        rng = np.random.default_rng(555000 + 7919 * j)
        correct = 0
        for xt, yt in zip(Xte, yte):
            T.exec_chunked(ng, [f"alter VIN{i} dc = {float(v*a.input_scale):.16f}" for i, v in enumerate(xt)])
            ng.run()
            plot = ng.plot(None, ng.last_plot).to_analysis()
            V = np.array([float(plot[str(n)].as_ndarray()[-1]) for n in out.tolist()])
            ng.exec_command("destroy all")
            V = V + rng.standard_normal(len(V)) * sigma
            correct += int(int(np.argmax(V)) == int(yt))
        accs.append(correct / len(yte))
    print(f">> INFER{a.ninf} {Path(a.chip).name} @ {Path(a.run).name}: "
          f"{[round(x, 4) for x in accs]} mean={np.mean(accs):.4f}", flush=True)


if __name__ == "__main__":
    main()
