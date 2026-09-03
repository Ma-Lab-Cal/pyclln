#!/usr/bin/env python3
"""scikit-digits (8x8, 10 classes) -- coupled-learning CLLN classifier trained end-to-end on ngspice.

64 pixel inputs x 20 differential rails (10 classes x {+,-}); prediction = argmax of the 10 rail-pair
voltage differences. Training rule: free solve -> loss clamp (--loss hinge|mse|cross_entropy) -> per-edge
contrastive gate update, per-sample SGD. Device: NMOS, drain at the
pixel node, source at the rail, gate = learned weight (body tied to drain). Network from topology_1247.npz.
Noise (--chip chips/chip_N.npz): a fixed per-device mismatch fingerprint (VTO sigma 10 mV + drive-strength
sigma 0.5%) plus Gaussian read noise; the free and clamp circuits are independent device sets.

Run:  python train_scikit.py [--chip chips/chip_1.npz] [--loss hinge]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from noise_model import KP_NOM, NMOSWRAP_PARAM, VTO_NOM  # parametrized NMOS wrapper

from PySpice.Spice.NgSpice.Shared import NgSpiceShared

RS_FREE, RS_CLAMP = 1e9, 0.01
VG_LO, VG_HI = 0.4, 8.0
OFF = 20000                      # node offset of the clamp-circuit copy
NC = 10                          # classes


def exec_chunked(ng, cmds, max_len=900):
    buf = ""
    for c in cmds:
        if len(buf) + len(c) + 2 > max_len:
            ng.exec_command(buf); buf = ""
        buf = c if not buf else buf + "; " + c
    if buf:
        ng.exec_command(buf)


def read_nodes(ng, nodes, rng, sigma):
    ng.run()
    plot = ng.plot(None, ng.last_plot).to_analysis()
    v = np.array([float(plot[str(n)].as_ndarray()[-1]) for n in nodes])
    ng.exec_command("destroy all")
    if sigma > 0:
        v = v + rng.standard_normal(len(v)) * sigma
    return v


def build_netlist(drain, source, vg, vth_free, vth_clamp, beta_free=None, beta_clamp=None):
    """One netlist, two electrically separate copies: A (free phase + all reads),
    B (clamp phase). Gates referenced to ground; body tied to the drain node."""
    E = len(drain)
    if beta_free is None:
        beta_free = np.ones(E, dtype=np.float64)
    if beta_clamp is None:
        beta_clamp = np.ones(E, dtype=np.float64)
    L = [".title scikit_digits_clln", NMOSWRAP_PARAM, ".options klu"]
    for copy, off, vth, beta, tag in (
        ("A", 0, vth_free, beta_free, ""),
        ("B", OFF, vth_clamp, beta_clamp, "B"),
    ):
        for p in range(64):
            L.append(f"VIN{tag}{p} {100 + p + off} 0 0")
        for j in range(20):
            L.append(f"RS{tag}{j} {900 + j + off} {500 + j + off} {RS_FREE:.6g}")
            L.append(f"VOUT{tag}{j} {500 + j + off} 0 0")
        for e in range(E):
            dnode = 100 + int(drain[e]) + off                 # drain (= pixel) node
            L.append(f"VG{tag}{e} g{tag}{e} 0 {float(vg[e]):.10f}")
            L.append(f"X{tag}{e} {dnode} g{tag}{e} "
                     f"{900 + int(source[e]) - 64 + off} {dnode} NMOSWRAP "  # body = drain
                     f"vth={VTO_NOM + float(vth[e]):.6f} kpval={KP_NOM * float(beta[e]):.12g}")
    L += [".options TEMP = 27C", ".options TNOM = 27C",
          ".options itl1=40 itl2=40 itl4=6 itl5=60",
          ".options gmin=1e-8 reltol=5e-3 abstol=1e-8 vntol=1e-5",
          ".options rshunt=1e9", ".op", ".end"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=str, default=None,
                    help="chip fingerprint npz (vto_free/vto_clamp + beta_free/beta_clamp); enables the 1x noise model")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--loss", choices=["hinge", "mse", "cross_entropy"], default="hinge",
                    help="training loss: hinge (margin-gated true/rival clamp), mse (one-hot rail targets), "
                         "cross_entropy (softmax-probability nudge)")
    ap.add_argument("--soft-t", type=float, default=0.02, help="softmax temperature for --loss cross_entropy")
    ap.add_argument("--mse-target", type=float, default=1.0, help="one-hot target magnitude for --loss mse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meas-rel", type=float, default=5e-3)
    ap.add_argument("--out", type=str, default="run_out")
    a = ap.parse_args()

    topo = np.load(HERE / "topology_1247.npz")
    drain, source = topo["drain"], topo["source"]
    E = len(drain)

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float64)                      # pixels -> [0,1] V
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed, stratify=y)

    noisy = a.chip is not None
    if noisy:
        z = np.load(a.chip)
        vth_f, vth_c = np.asarray(z["vto_free"]), np.asarray(z["vto_clamp"])
        # 1x mismatch model: per-device beta (strength) spread alongside VTO. Fall back to nominal
        # (ones) for legacy VTO-only chips so old fingerprints still reproduce.
        beta_f = np.asarray(z["beta_free"]) if "beta_free" in z.files else np.ones(E)
        beta_c = np.asarray(z["beta_clamp"]) if "beta_clamp" in z.files else np.ones(E)
        sigma = a.meas_rel * 1.0
    else:
        vth_f = vth_c = np.zeros(E)
        beta_f = beta_c = np.ones(E)
        sigma = 0.0
    rngA = np.random.default_rng(10_000)                   # read-noise streams (canonical)
    rngB = np.random.default_rng(20_000)

    import torch
    g = torch.Generator().manual_seed(a.seed)
    vg = torch.clamp(2.5 + 2.0 * (2 * torch.rand(E, generator=g) - 1), VG_LO, VG_HI) \
        .numpy().astype(np.float64)

    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(build_netlist(drain, source, vg, vth_f, vth_c, beta_f, beta_c))
    outA = [900 + j for j in range(20)]
    rail = source.astype(int) - 64                         # owning output rail of each edge

    def set_inputs(x):
        exec_chunked(ng, [f"alter VIN{p} dc = {x[p]:.8f}" for p in range(64)]
                       + [f"alter VINB{p} dc = {x[p]:.8f}" for p in range(64)])

    def eval_acc():
        correct = 0
        for xt, yt in zip(Xte, yte):
            set_inputs(xt)
            V = read_nodes(ng, outA, rngA, sigma)
            correct += int(int(np.argmax(V[:NC] - V[NC:])) == int(yt))
        return correct / len(yte)

    nodesA = [100 + p for p in range(64)] + outA
    nodesB = [n + OFF for n in nodesA]
    idxA = {n: i for i, n in enumerate(nodesA)}
    eDi = np.array([idxA[100 + int(d)] for d in drain])
    eSi = np.array([idxA[900 + int(s) - 64] for s in source])

    run_dir = HERE / "results" / a.out
    run_dir.mkdir(parents=True, exist_ok=True)
    acc_hist = [eval_acc()]
    print(f"[ep 0] untrained acc={acc_hist[0]:.4f} (chip={a.chip or 'clean'})", flush=True)
    order_rng = np.random.default_rng(0)
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        for i in order_rng.permutation(len(ytr)):
            x, yv = Xtr[i], int(ytr[i])
            set_inputs(x)
            Vf = read_nodes(ng, nodesA, rngA, sigma)
            sc = Vf[64:64 + NC] - Vf[64 + NC:]
            s2 = sc.copy(); s2[yv] = -np.inf
            r = int(np.argmax(s2))
            h = 0.5 * a.delta
            if a.loss == "hinge":
                if sc[yv] - sc[r] >= a.margin:
                    continue
                tg = {yv: Vf[64 + yv] + h, NC + yv: Vf[64 + NC + yv] - h,
                      r: Vf[64 + r] - h, NC + r: Vf[64 + NC + r] + h}
            elif a.loss == "mse":                                  # clamp every rail toward its one-hot target
                tg = {}
                for c in range(NC):
                    n = a.delta * ((a.mse_target if c == yv else 0.0) - sc[c])
                    tg[c] = Vf[64 + c] + 0.5 * n
                    tg[NC + c] = Vf[64 + NC + c] - 0.5 * n
            else:                                                  # cross_entropy: nudge by (one-hot - softmax)
                z = sc / a.soft_t; z = z - z.max(); pcls = np.exp(z); pcls = pcls / pcls.sum()
                grad = -pcls; grad[yv] += 1.0
                tg = {}
                for c in range(NC):
                    tg[c] = Vf[64 + c] + h * grad[c]
                    tg[NC + c] = Vf[64 + NC + c] - h * grad[c]
            cmds = []
            for j in range(20):
                if j in tg:
                    cmds += [f"alter RSB{j} {RS_CLAMP:.6g}", f"alter VOUTB{j} dc = {tg[j]:.10f}"]
                else:
                    cmds.append(f"alter RSB{j} {RS_FREE:.6g}")
            exec_chunked(ng, cmds)
            Vc = read_nodes(ng, nodesB, rngB, sigma)
            dVf = Vf[eDi] - Vf[eSi]
            dVc = Vc[eDi] - Vc[eSi]
            upd = np.flatnonzero(np.isin(rail, list(tg)))  # only the clamped rails' edges update
            dvg = -a.gamma * (dVc[upd] ** 2 - dVf[upd] ** 2)
            vg[upd] = np.clip(vg[upd] + dvg, VG_LO, VG_HI)
            exec_chunked(ng, [f"alter VG{e} dc = {vg[e]:.10f}" for e in upd]
                           + [f"alter VGB{e} dc = {vg[e]:.10f}" for e in upd])
        acc = eval_acc()
        acc_hist.append(acc)
        np.save(run_dir / f"vg_epoch{ep}.npy", vg.astype(np.float32))
        np.save(run_dir / "val_acc.npy", np.array(acc_hist))
        print(f"[ep {ep}/{a.epochs}] acc={acc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.save(run_dir / "vg_final.npy", vg)
    json.dump({"test_acc": acc, "endpoint_epoch": a.epochs, "chip": a.chip,
               "gamma": a.gamma, "delta": a.delta, "margin": a.margin,
               "epochs": a.epochs, "seed": a.seed},
              open(run_dir / "run_meta.json", "w"), indent=2)
    print(f">> DONE endpoint acc={acc:.4f} @ epoch {a.epochs}", flush=True)


if __name__ == "__main__":
    main()
