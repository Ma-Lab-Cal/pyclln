#!/usr/bin/env python3
"""XOR (4x4 differential NMOS input-output network) -- coupled-learning CLLN trained end-to-end on ngspice.

Metric: solves XOR (all 4 input cases correct at the differential output). Training rule: free solve ->
output nudged toward the target by eta -> per-edge contrastive gate update, one update per
sample. Network loaded from topology.npz. Noise (--chip
chips/chip_N.npz): a fixed per-device mismatch fingerprint (VTO sigma 10 mV + drive-strength sigma 0.5%)
plus Gaussian read noise; the free and clamp circuits are independent device sets. Omit --chip for a
clean (noise-free) run.

Run:  python train_xor.py [--chip chips/chip_1.npz]
"""
import os, sys, argparse, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import xor_circuit as T
sys.path.insert(0, str(T.REPO_ROOT / "common"))
import noise_model as NZ
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


def mk_dual(edge_list, weights, node_map, vminus, vplus, d_free, d_clamp, clamp_res, max_node,
            b_free=None, b_clamp=None):
    OFF = max_node + 100
    op, om, vm, vp, v1n, v2n = (int(node_map[k]) for k in ("oplus", "ominus", "vminus", "vplus", "v1", "v2"))
    if b_free is None: b_free = np.ones(len(edge_list))
    if b_clamp is None: b_clamp = np.ones(len(edge_list))
    L = [".title noisy_xor_dual", NZ.NMOSWRAP_PARAM.rstrip("\n")]
    for k, (tD, tS) in enumerate(edge_list):
        w = float(weights[k]); vf = NZ.VTO_NOM + float(d_free[k]); vc = NZ.VTO_NOM + float(d_clamp[k])
        kf = NZ.KP_NOM * float(b_free[k]); kc = NZ.KP_NOM * float(b_clamp[k])   # 1x: per-device beta
        L += [f".subckt e{k} t_D t_S", f"V1 t_G 0 {w:.16f}", f"XNMOS t_D t_G t_S 0 NMOSWRAP vth={vf:.6f} kpval={kf:.12g}", f".ends e{k}"]
        L += [f".subckt e{k}c t_D t_S", f"V1 t_G 0 {w:.16f}", f"XNMOS t_D t_G t_S 0 NMOSWRAP vth={vc:.6f} kpval={kc:.12g}", f".ends e{k}c"]
    spf, smf, spc, smc = OFF * 2 + 1, OFF * 2 + 2, OFF * 2 + 3, OFF * 2 + 4
    L += [f"RS1_F {op} {spf} 1e12", f"RS2_F {om} {smf} 1e12", f"VCLP_F {spf} 0 0", f"VCLM_F {smf} 0 0",
          f"VMINUS_F {vm} 0 {float(vminus):.16f}", f"VPLUS_F {vp} 0 {float(vplus):.16f}", f"VV1_F {v1n} 0 0", f"VV2_F {v2n} 0 0",
          f"RS1_C {op + OFF} {spc} {float(clamp_res):.16f}", f"RS2_C {om + OFF} {smc} {float(clamp_res):.16f}",
          f"VCLP_C {spc} 0 0", f"VCLM_C {smc} 0 0",
          f"VMINUS_C {vm + OFF} 0 {float(vminus):.16f}", f"VPLUS_C {vp + OFF} 0 {float(vplus):.16f}",
          f"VV1_C {v1n + OFF} 0 0", f"VV2_C {v2n + OFF} 0 0"]
    for k, (tD, tS) in enumerate(edge_list):
        L.append(f"X{k} {tD} {tS} e{k}"); L.append(f"X{k}c {tD + OFF} {tS + OFF} e{k}c")
    L += [".options klu", ".options TEMP = 27C", ".options TNOM = 27C", ".options itl1=40 itl2=40 itl4=6 itl5=60",
          ".options gmin=1e-8 reltol=5e-3 abstol=1e-8 vntol=1e-5", ".options rshunt=1e9", ".op", ".end"]
    return "\n".join(L) + "\n", OFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clamp-res", type=float, default=0.01)
    ap.add_argument("--vminus", type=float, default=0.11)
    ap.add_argument("--vplus", type=float, default=0.33)
    ap.add_argument("--vmax", type=float, default=0.45)
    ap.add_argument("--L0", type=float, default=-0.087)
    ap.add_argument("--chip", type=str, default=None,
                    help="device-mismatch fingerprint file (chip_1/2/3.npz). Omit for the clean (noise-free) run.")
    ap.add_argument("--meas-rel", type=float, default=NZ.MEAS_REL)
    ap.add_argument("--vg-init-lo", type=float, default=0.5)
    ap.add_argument("--vg-init-hi", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=50)
    a = ap.parse_args()

    np.random.seed(a.seed); rng = np.random.default_rng(a.seed + 12345)
    X, Y = T.xor_dataset_pnas(Vmax=a.vmax, L0=a.L0)
    G, node_map, edge_list = T._build_training_graph(4)
    n_edges = len(edge_list)
    vg = np.clip(np.random.uniform(a.vg_init_lo, a.vg_init_hi, n_edges), T.VG_CLIP_LO, T.VG_CLIP_HI).astype(float)
    max_node = int(max(G.nodes()))
    meas_sigma = float(a.meas_rel) * float(a.vmax)
    if a.chip is not None:                                               # noisy: load the fixed chip fingerprint
        z = np.load(a.chip)
        d_free, d_clamp = np.asarray(z["vto_free"]), np.asarray(z["vto_clamp"])
        b_free = np.asarray(z["beta_free"]) if "beta_free" in z.files else np.ones(n_edges)
        b_clamp = np.asarray(z["beta_clamp"]) if "beta_clamp" in z.files else np.ones(n_edges)
    else:                                                                # clean: nominal devices
        d_free = d_clamp = np.zeros(n_edges); b_free = b_clamp = np.ones(n_edges)

    net, OFF = mk_dual(edge_list, vg, node_map, a.vminus, a.vplus, d_free, d_clamp, a.clamp_res, max_node,
                       b_free=b_free, b_clamp=b_clamp)
    ng = NgSpiceShared(send_data=False); ng.load_circuit(net)

    nodes = np.asarray(sorted(G.nodes()), dtype=int)
    index_of = np.full(nodes.max() + 1, -1, dtype=int); index_of[nodes] = np.arange(nodes.size)
    e1 = np.asarray([u for (u, _) in edge_list]); e2 = np.asarray([v for (_, v) in edge_list])
    op, om = int(node_map["oplus"]), int(node_map["ominus"])
    tag = os.path.splitext(os.path.basename(a.chip))[0] if a.chip else "clean"
    rd = os.environ.get("RUN_DIR", f"{_HERE}/results/noisy/xor_{tag}_s{a.seed}")
    os.makedirs(rd, exist_ok=True)

    def rd_nodes(nlist):
        v = T.get_voltages(ng, list(nlist)).astype(float)
        return v + rng.standard_normal(np.shape(v)) * meas_sigma if meas_sigma > 0 else v

    def set_inputs(v1, v2):
        T._exec_chunked(ng, [f"alter VV1_F dc = {v1:.16f}", f"alter VV2_F dc = {v2:.16f}",
                             f"alter VV1_C dc = {v1:.16f}", f"alter VV2_C dc = {v2:.16f}"])

    def eval_acc():
        preds = []
        for i in range(X.shape[0]):
            set_inputs(float(X[i, 0]), float(X[i, 1])); ng.run()
            fa = rd_nodes(nodes)
            preds.append(float(fa[index_of[op]] - fa[index_of[om]]))
            try: ng.exec_command("destroy all")
            except Exception: pass
        return float(T.accuracy_nearest_target(np.array(preds).reshape(-1, 1), Y, a.L0))

    acc0 = eval_acc(); hist = [acc0]; t0 = time.time()
    print(f"[xor {tag} s{a.seed} meas={meas_sigma*1e3:.2f}mV] ep0 acc={acc0:.3f}", flush=True)

    for ep in range(1, a.epochs + 1):
        for idx in range(X.shape[0]):
            v1, v2 = float(X[idx, 0]), float(X[idx, 1])
            set_inputs(v1, v2); ng.run()
            free_all = rd_nodes(nodes)
            try: ng.exec_command("destroy all")
            except Exception: pass
            fop, fom = free_all[index_of[op]], free_all[index_of[om]]
            free_O = fop - fom
            d = a.eta * (float(Y[idx, 0]) - free_O)            # eta*(L - O), target label L = Y[idx]
            T._exec_chunked(ng, [f"alter VCLP_C dc = {fop + 0.5*d:.16f}", f"alter VCLM_C dc = {fom - 0.5*d:.16f}"]); ng.run()
            clamp_all = rd_nodes(nodes + OFF)
            try: ng.exec_command("destroy all")
            except Exception: pass
            fd = free_all[index_of[e1]] - free_all[index_of[e2]]
            cd = clamp_all[index_of[e1]] - clamp_all[index_of[e2]]
            upd = -a.gamma * (cd ** 2 - fd ** 2)
            cmds = []
            for k, du in enumerate(upd):
                nv = float(np.clip(vg[k] + du, T.VG_CLIP_LO, T.VG_CLIP_HI)); vg[k] = nv
                cmds.append(f"alter v.x{k}.v1 dc = {nv:.16f}"); cmds.append(f"alter v.x{k}c.v1 dc = {nv:.16f}")
            T._exec_chunked(ng, cmds)
        if ep % a.eval_every == 0 or ep == 1 or ep == a.epochs:
            acc = eval_acc(); hist.append(acc)
            np.save(f"{rd}/acc_history.npy", np.asarray(hist)); np.save(f"{rd}/vg_final.npy", vg)
            if ep % (a.eval_every * 20) == 0 or ep == a.epochs:
                print(f"[xor {tag} s{a.seed}] ep{ep} acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[xor {tag} s{a.seed}] FINAL acc={hist[-1]:.3f}  meas={meas_sigma*1e3:.2f}mV", flush=True)


if __name__ == "__main__":
    main()
