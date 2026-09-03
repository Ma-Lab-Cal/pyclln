#!/usr/bin/env python3
"""UCI ionosphere (binary good/bad, 34 features) -- coupled-learning CLLN classifier trained on ngspice.

34 inputs -> 5 hidden -> 2 outputs (147 edges); prediction = argmax of the two output-node voltages.
Training rule: free solve -> loss clamp (--loss hinge|mse|cross_entropy) -> per-edge
contrastive gate update. Device: NMOS, gate referenced to drain,
body floating. Network from topology_147.npz. Noise (--chip chips/chip_N.npz): a fixed per-device mismatch
fingerprint (VTO sigma 10 mV + drive-strength sigma 0.5%) plus Gaussian read noise; the free and clamp
circuits are independent device sets.

Run:  python train_ionosphere.py [--chip chips/chip_1.npz] [--loss hinge]
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from noise_model import NMOSWRAP_PARAM, VTO_NOM, KP_NOM  # parametrized NMOS wrapper (vth + kp)

from PySpice.Spice.NgSpice.Shared import NgSpiceShared

RS_FREE, RS_CLAMP = 1e9, 0.01
VG_LO, VG_HI = 0.4, 8.0
OFF = 20000                      # node offset of the clamp-circuit copy (noise mode)
# NOTE: this network is purely INPUT-DRIVEN — no device connects to the supply rail (negref/posref),
# so no VMINUS/VPLUS sources are emitted. The network is driven entirely through the 34 input sources.
DEVICE_LIB = HERE.parent / "device_model" / "nmos_lvl1_ald1106.lib"


def load_ionosphere(path):
    X, y = [], []
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) < 35:
            continue
        X.append([float(v) for v in parts[:34]])
        y.append(1 if parts[34].strip() == "g" else 0)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)


def exec_chunked(ng, cmds, max_len=900):
    buf = ""
    for c in cmds:
        if len(buf) + len(c) + 2 > max_len:
            ng.exec_command(buf); buf = ""
        buf = c if not buf else buf + "; " + c
    if buf:
        ng.exec_command(buf)


def _device_subckt(noisy, vth_delta, beta=1.0):
    if noisy:
        # 1x model: per-device VTO shift AND beta (kp strength) multiplier 1+eps
        return f"NMOSWRAP vth={VTO_NOM + float(vth_delta):.6f} kpval={KP_NOM * float(beta):.12g}"
    return "NMOSWRAP"


def build_netlist(topo, vg, noisy, vth_free, vth_clamp, beta_free=None, beta_clamp=None):
    inp, out = topo["input_nodes"], topo["out_nodes"]
    eD, eS = topo["edges_D"], topo["edges_S"]
    if beta_free is None:
        beta_free = np.ones(len(eD))
    if beta_clamp is None:
        beta_clamp = np.ones(len(eD))
    neg, pos = int(topo["negref"]), int(topo["posref"])
    sink0 = max([neg, pos] + inp.tolist() + out.tolist() + eD.tolist() + eS.tolist()) + 1
    K = len(out)
    L = [".title ionosphere_clln", NMOSWRAP_PARAM, ".options klu"]
    copies = ([("", 0, vth_free, beta_free)]
              + ([("B", OFF, vth_clamp, beta_clamp)] if noisy else []))
    for tag, off, vth, beta in copies:
        for i, n in enumerate(inp):
            L.append(f"VIN{tag}{i} {int(n) + off} 0 0")
        for i, on in enumerate(out, start=1):
            L.append(f"RS{tag}{i} {int(on) + off} {sink0 + off + (i - 1)} {RS_FREE:.6g}")
        for j in range(K):
            L.append(f"VOUT{tag}{j} {sink0 + off + j} 0 0")
        for e, (D, S) in enumerate(zip(eD.tolist(), eS.tolist())):
            L.append(f"VG{tag}{e} g{tag}{e} {D + off} {float(vg[e]):.16f}")   # gate ref = drain
            dev = _device_subckt(noisy, vth[e], beta[e])
            L.append(f"X{tag}{e} {D + off} g{tag}{e} {S + off} b{tag}{e} {dev}")
    L += [".options TEMP = 27C", ".options TNOM = 27C",
          ".options itl1=40 itl2=40 itl4=6 itl5=60",
          ".options gmin=1e-8 reltol=5e-3 abstol=1e-8 vntol=1e-5",
          ".options rshunt=1e9", ".op", ".end"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=str, default=None,
                    help="chip fingerprint npz (vto_free/vto_clamp); enables the noise model")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--loss", choices=["hinge", "mse", "cross_entropy"], default="hinge",
                    help="training loss: hinge (margin-gated true/rival clamp), mse (one-hot output targets), "
                         "cross_entropy (softmax-probability nudge)")
    ap.add_argument("--soft-t", type=float, default=0.02, help="softmax temperature for --loss cross_entropy")
    ap.add_argument("--mse-target", type=float, default=1.0, help="one-hot target magnitude for --loss mse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vg-init", type=float, default=2.0)
    ap.add_argument("--input-scale", type=float, default=0.8,
                    help="scales the [-1,1] features to +-0.8 V input node voltages")
    ap.add_argument("--meas-rel", type=float, default=5e-3)
    ap.add_argument("--data", type=str, default=str(HERE / "ionosphere.data"))
    ap.add_argument("--out", type=str, default="run_out")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    topo = np.load(os.environ.get("TOPO_PATH", str(HERE / "topology_147.npz")))
    eD, eS, out = topo["edges_D"], topo["edges_S"], topo["out_nodes"]
    E, K = len(eD), len(out)
    vg_lo = np.full(E, VG_LO, dtype=float)
    vg_hi = np.full(E, VG_HI, dtype=float)
    nodes = sorted(set(topo["input_nodes"].tolist()) | set(out.tolist())
                   | set(eD.tolist()) | set(eS.tolist()))
    idx = {n: i for i, n in enumerate(nodes)}
    eDi = np.array([idx[int(d)] for d in eD])
    eSi = np.array([idx[int(s)] for s in eS])
    oi = np.array([idx[int(n)] for n in out])

    X, y = load_ionosphere(a.data)                         # features already in [-1,1] V
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed, stratify=y)

    noisy = a.chip is not None
    if noisy:
        z = np.load(a.chip)
        vth_f, vth_c = np.asarray(z["vto_free"]), np.asarray(z["vto_clamp"])
        # 1x model: per-device beta (strength) spread alongside VTO; legacy VTO-only chips -> ones.
        beta_f = np.asarray(z["beta_free"]) if "beta_free" in z.files else np.ones(E)
        beta_c = np.asarray(z["beta_clamp"]) if "beta_clamp" in z.files else np.ones(E)
        sigma = a.meas_rel * float(a.input_scale)
    else:
        vth_f = vth_c = np.zeros(E)
        beta_f = beta_c = np.ones(E)
        sigma = 0.0
    rng = np.random.default_rng(424242)                    # canonical read-noise stream

    def nz(v):
        return v + rng.standard_normal(np.shape(v)) * sigma if sigma > 0 else v

    vg = np.full(E, float(a.vg_init))
    netlist = build_netlist(topo, vg, noisy, vth_f, vth_c, beta_free=beta_f, beta_clamp=beta_c)
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)
    B = "B" if noisy else ""                               # clamp acts on copy B when noisy

    def read_raw(node_list):
        ng.run()
        plot = ng.plot(None, ng.last_plot).to_analysis()
        v = np.array([float(plot[str(n)].as_ndarray()[-1]) for n in node_list])
        ng.exec_command("destroy all")
        if not np.all(np.isfinite(v)):
            raise RuntimeError("non-finite solve")
        return v

    def reload_circuit():
        try:
            ng.remove_circuit()
        except Exception:
            pass
        ng.load_circuit(netlist)
        upd = [f"alter VG{e} dc = {vg[e]:.16f}" for e in range(E)]
        if noisy:
            upd += [f"alter VGB{e} dc = {vg[e]:.16f}" for e in range(E)]
        exec_chunked(ng, upd)

    def set_inputs(x):
        xs = x * a.input_scale                       # scale [-1,1] features to +-0.8 V
        cmds = [f"alter VIN{i} dc = {float(v):.16f}" for i, v in enumerate(xs)]
        if noisy:
            cmds += [f"alter VINB{i} dc = {float(v):.16f}" for i, v in enumerate(xs)]
        exec_chunked(ng, cmds)

    def free_rs():
        exec_chunked(ng, [f"alter RS{i} {RS_FREE:.6g}" for i in range(1, K + 1)])

    def eval_acc():
        free_rs()
        correct = 0
        for xt, yt in zip(Xte, yte):
            set_inputs(xt)
            V = nz(read_raw(out.tolist()))
            correct += int(int(np.argmax(V)) == int(yt))
        return correct / len(yte)

    def eval_full(Xset, yset):
        """Epoch-level accuracy + mean binary-hinge loss on a dataset (test or train)."""
        free_rs()
        correct = 0
        loss = 0.0
        for xt, yt in zip(Xset, yset):
            set_inputs(xt)
            V = nz(read_raw(out.tolist()))
            yt = int(yt)
            correct += int(int(np.argmax(V)) == yt)
            rival = int(np.argmax([V[k] if k != yt else -1e30 for k in range(K)]))
            loss += max(0.0, a.margin - (float(V[yt]) - float(V[rival])))
        n = max(len(yset), 1)
        return correct / n, loss / n

    run_dir = HERE / "results" / a.out
    run_dir.mkdir(parents=True, exist_ok=True)
    _te0a, _te0l = eval_full(Xte, yte)
    _st = rng.bit_generator.state; _tr0a, _tr0l = eval_full(Xtr, ytr); rng.bit_generator.state = _st
    acc_hist = [_te0a]
    curve = {"epoch": [0], "train_acc": [_tr0a], "test_acc": [_te0a], "train_loss": [_tr0l], "test_loss": [_te0l]}
    print(f"[ep 0] untrained acc={acc_hist[0]:.4f} (chip={a.chip or 'clean'})", flush=True)
    _updc = os.environ.get("UPD_CURVE") == "1"       # per-gate-update acc/loss curve (fig2 style)
    updcurve = {"step": [], "epoch": [], "sample": [], "train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    if _updc:
        for _k, _v in zip(("step", "epoch", "sample", "train_acc", "test_acc", "train_loss", "test_loss"),
                          (0, 0, 0, _tr0a, _te0a, _tr0l, _te0l)):
            updcurve[_k].append(_v)
    _ustep = 0
    _traj = os.environ.get("TRAJ") == "1"           # record gate trajectory only (for parallel per-step eval)
    vg_traj = [vg.astype(np.float32).copy()]; traj_ep = [0]; traj_sa = [0]
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        order = np.arange(len(ytr))
        np.random.shuffle(order)
        for i in order:
            x, yv = Xtr[i], int(ytr[i])
            free_rs()
            set_inputs(x)
            try:
                raw = read_raw(nodes)
            except Exception:
                reload_circuit(); free_rs(); continue
            Vout = nz(raw[oi])
            Vf = nz(raw)
            r = int(np.argmax(np.where(np.arange(K) == yv, -np.inf, Vout)))
            h = 0.5 * a.delta
            if a.loss == "hinge":
                if Vout[yv] - Vout[r] >= a.margin:
                    continue
                tg = {yv: float(Vout[yv]) + h, r: float(Vout[r]) - h}
            elif a.loss == "mse":                                  # clamp every output toward its one-hot target
                tg = {c: float(Vout[c]) + a.delta * ((a.mse_target if c == yv else 0.0) - float(Vout[c])) for c in range(K)}
            else:                                                  # cross_entropy: nudge by (one-hot - softmax)
                z = Vout / a.soft_t; z = z - z.max(); pcls = np.exp(z); pcls = pcls / pcls.sum()
                tg = {c: float(Vout[c]) + h * ((1.0 if c == yv else 0.0) - pcls[c]) for c in range(K)}
            cmds = [f"alter RS{B}{j} {RS_FREE:.6g}" for j in range(1, K + 1)]
            for c, v in tg.items():
                cmds += [f"alter RS{B}{c+1} {RS_CLAMP:.6g}", f"alter VOUT{B}{c} dc = {v:.16f}"]
            exec_chunked(ng, cmds)
            try:
                Vc = nz(read_raw([n + OFF for n in nodes] if noisy else nodes))
            except Exception:
                reload_circuit(); free_rs(); continue
            dVf = Vf[eDi] - Vf[eSi]
            dVc = Vc[eDi] - Vc[eSi]
            contrast = dVc ** 2 - dVf ** 2
            vg = np.clip(vg - a.gamma * contrast, vg_lo, vg_hi)
            upd = [f"alter VG{e} dc = {vg[e]:.16f}" for e in range(E)]
            if noisy:
                upd += [f"alter VGB{e} dc = {vg[e]:.16f}" for e in range(E)]
            exec_chunked(ng, upd)
            if _traj:                                # record gate state after each update (eval later in parallel)
                vg_traj.append(vg.astype(np.float32).copy()); traj_ep.append(ep); traj_sa.append(int(i))
            elif _updc:                              # log acc/loss after each actual gate update
                _ustep += 1
                _stU = rng.bit_generator.state
                try:
                    _tea, _tel = eval_full(Xte, yte)
                    _tra, _trl = eval_full(Xtr, ytr)
                    for _k, _v in zip(("step", "epoch", "sample", "train_acc", "test_acc", "train_loss", "test_loss"),
                                      (_ustep, ep, int(i), _tra, _tea, _trl, _tel)):
                        updcurve[_k].append(_v)
                except Exception:
                    reload_circuit()                 # skip this step's log on a solve hiccup, keep training
                rng.bit_generator.state = _stU
                free_rs(); set_inputs(x)              # restore free-solve state for the next sample
        te_acc, te_loss = eval_full(Xte, yte)
        _st = rng.bit_generator.state; tr_acc, tr_loss = eval_full(Xtr, ytr); rng.bit_generator.state = _st
        acc = te_acc
        acc_hist.append(acc)
        for _k, _v in zip(("epoch", "train_acc", "test_acc", "train_loss", "test_loss"),
                          (ep, tr_acc, te_acc, tr_loss, te_loss)):
            curve[_k].append(_v)
        np.save(run_dir / f"vg_epoch{ep}.npy", vg.astype(np.float32))
        np.save(run_dir / "val_acc.npy", np.array(acc_hist))
        np.savez(run_dir / "curve.npz", **{_k: np.asarray(_v) for _k, _v in curve.items()})
        if _updc:
            np.savez(run_dir / "updcurve.npz", **{_k: np.asarray(_v) for _k, _v in updcurve.items()})
        if _traj:
            np.savez(run_dir / "vg_traj.npz", vg_states=np.array(vg_traj, dtype=np.float32),
                     epochs=np.array(traj_ep, dtype=np.int64), samples=np.array(traj_sa, dtype=np.int64))
        print(f"[ep {ep}/{a.epochs}] acc={acc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.save(run_dir / "vg_final.npy", vg)
    json.dump({"test_acc": acc, "endpoint_epoch": a.epochs, "chip": a.chip,
               "gamma": a.gamma, "delta": a.delta, "margin": a.margin,
               "epochs": a.epochs, "seed": a.seed, "edges": E,
               "device": "NMOS, gate referenced to drain, body floating",
               "update_rule": "contrastive gate update dVg = -gamma*(dVc^2 - dVf^2)"},
              open(run_dir / "run_meta.json", "w"), indent=2)
    print(f">> DONE endpoint acc={acc:.4f} @ epoch {a.epochs}", flush=True)


if __name__ == "__main__":
    main()
