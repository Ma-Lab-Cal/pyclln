#!/usr/bin/env python3
"""Nonlinear 1-D regression (4x4 NMOS input-output network) -- coupled-learning CLLN trained end-to-end on ngspice.

Metric: test MSE. Training rule: free solve -> output nudged toward the target by eta -> per-edge
contrastive gate update. Network loaded from
topology.npz. Noise (--chip chips/chip_N.npz): a fixed per-device mismatch fingerprint (VTO sigma 10 mV +
drive-strength sigma 0.5%) plus Gaussian read noise; the free and clamp circuits are independent device
sets. Omit --chip for a clean run.

Run:  python train_regression.py [--chip chips/chip_1.npz]
"""
import os, sys, argparse, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import reg_circuit as T
sys.path.insert(0, str(T.REPO_ROOT / "common"))
import noise_model as NZ
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


def mk_dual_netlist(edge_list, weights, node_map, vminus, vplus, d_free, d_clamp, clamp_res, max_node,
                    solver="klu", b_free=None, b_clamp=None):
    """One netlist, two electrically-separate copies: free (d_free, floating out) + clamp (d_clamp, clamped
    out). Clamp-copy nodes are offset by OFF; ground (0) and the per-edge gate weights are shared in value."""
    OFF = max_node + 100
    on, vm, vp, vi = (int(node_map[k]) for k in ("out", "vminus", "vplus", "vin"))
    if b_free is None: b_free = np.ones(len(edge_list))
    if b_clamp is None: b_clamp = np.ones(len(edge_list))
    L = [".title noisy_reg_dual", NZ.NMOSWRAP_PARAM.rstrip("\n")]
    for k, (tD, tS) in enumerate(edge_list):
        w = float(weights[k])
        vf = NZ.VTO_NOM + float(d_free[k]); vc = NZ.VTO_NOM + float(d_clamp[k])
        kf = NZ.KP_NOM * float(b_free[k]); kc = NZ.KP_NOM * float(b_clamp[k])   # 1x: per-device beta
        L += [f".subckt e{k} t_D t_S", f"V1 t_G 0 {w:.16f}", f"XNMOS t_D t_G t_S 0 NMOSWRAP vth={vf:.6f} kpval={kf:.12g}", f".ends e{k}"]
        L += [f".subckt e{k}c t_D t_S", f"V1 t_G 0 {w:.16f}", f"XNMOS t_D t_G t_S 0 NMOSWRAP vth={vc:.6f} kpval={kc:.12g}", f".ends e{k}c"]
    sf, sc = OFF * 2 + 1, OFF * 2 + 2
    L += [f"RSO_F {on} {sf} 1e12", f"VCLO_F {sf} 0 0",
          f"RSO_C {on + OFF} {sc} {float(clamp_res):.16f}", f"VCLO_C {sc} 0 0",
          f"VMINUS_F {vm} 0 {float(vminus):.16f}", f"VPLUS_F {vp} 0 {float(vplus):.16f}", f"VVIN_F {vi} 0 0",
          f"VMINUS_C {vm + OFF} 0 {float(vminus):.16f}", f"VPLUS_C {vp + OFF} 0 {float(vplus):.16f}", f"VVIN_C {vi + OFF} 0 0"]
    for k, (tD, tS) in enumerate(edge_list):
        L.append(f"X{k} {tD} {tS} e{k}")
        L.append(f"X{k}c {tD + OFF} {tS + OFF} e{k}c")
    if solver == "klu":
        L.append(".options klu")
    L += [".options TEMP = 27C", ".options TNOM = 27C", ".options itl1=40 itl2=40 itl4=6 itl5=60",
          ".options gmin=1e-8 reltol=5e-3 abstol=1e-8 vntol=1e-5", ".options rshunt=1e9", ".op", ".end"]
    return "\n".join(L) + "\n", OFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25000)
    ap.add_argument("--gamma", type=float, default=0.4)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clamp-res", type=float, default=0.01)
    ap.add_argument("--vminus", type=float, default=0.0)
    ap.add_argument("--vplus", type=float, default=0.45)
    ap.add_argument("--chip", type=str, default=None,
                    help="device-mismatch fingerprint file (chip_1/2/3.npz). Omit for the clean (noise-free) run.")
    ap.add_argument("--meas-rel", type=float, default=NZ.MEAS_REL)                   # 5e-3 -> *V_FS
    ap.add_argument("--vg-init-lo", type=float, default=3.0)
    ap.add_argument("--vg-init-hi", type=float, default=3.0)
    ap.add_argument("--eval-every", type=int, default=50)
    a = ap.parse_args()

    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed + 12345)
    X, Y = T.VINS.copy(), T.VOUS.copy()
    G, node_map, edge_list = T._build_training_graph(4)
    n_edges = len(edge_list)
    vg = (np.random.uniform(a.vg_init_lo, a.vg_init_hi, n_edges) if a.vg_init_hi > a.vg_init_lo
          else np.full(n_edges, a.vg_init_lo)).astype(float)
    vg = np.clip(vg, T.VG_CLIP_LO, T.VG_CLIP_HI)
    max_node = int(max(G.nodes()))
    meas_sigma = float(a.meas_rel) * float(a.vplus)
    if a.chip is not None:                                               # noisy: load the fixed chip fingerprint
        z = np.load(a.chip)
        d_free, d_clamp = np.asarray(z["vto_free"]), np.asarray(z["vto_clamp"])
        b_free = np.asarray(z["beta_free"]) if "beta_free" in z.files else np.ones(n_edges)
        b_clamp = np.asarray(z["beta_clamp"]) if "beta_clamp" in z.files else np.ones(n_edges)
    else:                                                                # clean: nominal devices
        d_free = d_clamp = np.zeros(n_edges); b_free = b_clamp = np.ones(n_edges)

    net, OFF = mk_dual_netlist(edge_list, vg, node_map, a.vminus, a.vplus, d_free, d_clamp, a.clamp_res, max_node,
                               b_free=b_free, b_clamp=b_clamp)
    ng = NgSpiceShared(send_data=False); ng.load_circuit(net)

    nodes = np.asarray(sorted(G.nodes()), dtype=int)
    index_of = np.full(nodes.max() + 1, -1, dtype=int); index_of[nodes] = np.arange(nodes.size)
    e1 = np.asarray([u for (u, _) in edge_list]); e2 = np.asarray([v for (_, v) in edge_list])
    on = int(node_map["out"])
    tag = os.path.splitext(os.path.basename(a.chip))[0] if a.chip else "clean"
    rd = os.environ.get("RUN_DIR", f"{_HERE}/results/noisy/reg_{tag}_s{a.seed}")
    os.makedirs(rd, exist_ok=True)

    def rd_nodes(nlist):
        v = T._require_finite("nodes", T.get_voltages(ng, list(nlist))).astype(float)
        return v + rng.standard_normal(np.shape(v)) * meas_sigma if meas_sigma > 0 else v

    def set_input(vin):
        T._exec_chunked(ng, [f"alter VVIN_F dc = {vin:.16f}", f"alter VVIN_C dc = {vin:.16f}"])

    def eval_mse():
        preds = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            set_input(float(X[i])); ng.run()
            preds[i] = rd_nodes([on])[0]
            try: ng.exec_command("destroy all")
            except Exception: pass
        return preds, float(np.mean((preds - Y) ** 2))

    p0, mse0 = eval_mse(); hist = [mse0]; t0 = time.time()
    print(f"[reg {tag} s{a.seed} meas={meas_sigma*1e3:.2f}mV] ep0 mse={mse0:.6f}", flush=True)

    for ep in range(1, a.epochs + 1):
        for idx in range(X.shape[0]):
            vin = float(X[idx]); target = float(Y[idx])
            set_input(vin); ng.run()                                  # solve 1 -> free read (free copy)
            free_out = rd_nodes([on])[0]
            free_all = rd_nodes(nodes)
            try: ng.exec_command("destroy all")
            except Exception: pass
            clamped_out = a.eta * target + (1.0 - a.eta) * free_out
            T._exec_chunked(ng, [f"alter VCLO_C dc = {clamped_out:.16f}"]); ng.run()   # solve 2 -> clamp read (clamp copy)
            clamp_all = rd_nodes(nodes + OFF)
            try: ng.exec_command("destroy all")
            except Exception: pass
            fd = free_all[index_of[e1]] - free_all[index_of[e2]]
            cd = clamp_all[index_of[e1]] - clamp_all[index_of[e2]]
            upd = -a.gamma * (cd ** 2 - fd ** 2)
            cmds = []
            for k, du in enumerate(upd):
                nv = float(np.clip(vg[k] + du, T.VG_CLIP_LO, T.VG_CLIP_HI)); vg[k] = nv
                cmds.append(f"alter v.x{k}.v1 dc = {nv:.16f}")
                cmds.append(f"alter v.x{k}c.v1 dc = {nv:.16f}")
            T._exec_chunked(ng, cmds)
        if ep % a.eval_every == 0 or ep == 1 or ep == a.epochs:
            _, mse_ep = eval_mse(); hist.append(mse_ep)
            np.save(f"{rd}/mse_history.npy", np.asarray(hist)); np.save(f"{rd}/vg_final.npy", vg)
            if ep % (a.eval_every * 20) == 0 or ep == a.epochs:
                print(f"[reg {tag} s{a.seed}] ep{ep} mse={mse_ep:.6f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[reg {tag} s{a.seed}] FINAL mse={hist[-1]:.6f}  clean~1.7e-5  meas={meas_sigma*1e3:.2f}mV", flush=True)


if __name__ == "__main__":
    main()
