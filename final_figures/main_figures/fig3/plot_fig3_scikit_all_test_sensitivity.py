#!/usr/bin/env python3
"""Device-faithful all-test input-sensitivity fields for the scikit-digits network (Fig. 3 + Suppl. 4).

The sensitivity DATA is measured on the REAL ngspice operating point. For every held-out test digit x and
pixel p, set VINp to a unit small-signal source and run a one-point AC transfer solve. In this DC transistor
network, that is the same local linearized derivative returned by ngspice `.tf`, while reading all 20 output
rails in one solve:

    J[c, p](x) = d(V(rail c+) - V(rail c-)) / dx_p .

Aggregate maps over all test samples:
    signed[c, p]   = E_test[ J[c, p] ]        (unweighted; Suppl. 4)
    evidence[c, p] = E_test[ x_p * J[c, p] ]  (input-weighted attribution)
Every value comes from ngspice operating-point solves on the released clean gates.
The plotting style is shared through _sensitivity_style.

  python plot_fig3_scikit_all_test_sensitivity.py [--workers 30]
"""
import argparse, os, sys, subprocess, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_PR = next(p for p in HERE.parents if (p / "device_model").is_dir())
SUPPL4_DIR = _PR / "final_figures" / "suppl_figures" / "suppl4"   # the signed (unweighted) map is Suppl. Fig. 4's asset
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_PR / "scikit_digits"))
from _sensitivity_style import (
    _draw_maps,
    _draw_colorbar,
    _auto_abs_limit,
    _test_split,
    _load_final_state,
    DEFAULT_SIGNED_SATURATION_VLIM,
)

OUTA = [900 + j for j in range(20)]
TMP = HERE / "_all_test_sens_partials"
DATA_DIR = HERE / "data"                                           # fig3 output npz live under data/
GATES = _PR / "scikit_digits" / "runs" / "clean" / "gates.npz"
AC_FREQ_HZ = 1.0
EVIDENCE_VMIN = -0.020
EVIDENCE_VMAX = 0.010
EVIDENCE_CONTRAST_GAMMA = 0.65


def _worker(lo, hi, out_path):
    import train_scikit as T
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    Xte, _ = _test_split()
    drain, source, vg = _load_final_state(GATES, "vg_final")
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(T.build_netlist(drain, source, vg, np.zeros(vg.size), np.zeros(vg.size)))
    rng = np.random.default_rng(0)
    ssum = np.zeros((10, 64)); esum = np.zeros((10, 64)); msum = np.zeros((10, 64)); vsum = np.zeros((10, 2)); n = 0
    T.exec_chunked(ng, [f"alter vin{p} ac 0" for p in range(64)]
                   + [f"alter vinb{p} ac 0" for p in range(64)])

    def rails(x):
        T.exec_chunked(ng, [f"alter VIN{p} dc = {x[p]:.8f}" for p in range(64)]
                         + [f"alter VINB{p} dc = {x[p]:.8f}" for p in range(64)])
        return T.read_nodes(ng, OUTA, rng, 0.0)

    def transfer_for_pixel(pixel):
        T.exec_chunked(ng, [f"alter vin{pixel} ac 1"])
        ng.exec_command(f"ac lin 1 {AC_FREQ_HZ:.8g} {AC_FREQ_HZ:.8g}")
        plot = ng.plot(None, ng.last_plot).to_analysis()
        v = np.array([complex(plot[str(node)].as_ndarray()[-1]) for node in OUTA], dtype=np.complex128)
        ng.exec_command("destroy all")
        T.exec_chunked(ng, [f"alter vin{pixel} ac 0"])
        return np.real(v[:10] - v[10:])

    for x in Xte[lo:hi]:
        V0 = rails(x)
        J = np.zeros((20, 64))
        for p in range(64):
            J[:10, p] = transfer_for_pixel(p)
        j = J[:10]
        ssum += j; esum += x[None, :] * j; msum += np.abs(j)
        vsum[:, 0] += V0[:10]; vsum[:, 1] += V0[10:]; n += 1
    np.savez(out_path, ssum=ssum, esum=esum, msum=msum, vsum=vsum, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(30, (os.cpu_count() or 4) - 2))
    ap.add_argument("--png-dpi", type=int, default=600)
    ap.add_argument("--worker", nargs=3, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        _worker(int(a.worker[0]), int(a.worker[1]), a.worker[2]); return

    Xte, yte = _test_split(); n_test = len(Xte)
    TMP.mkdir(exist_ok=True)
    for f in glob.glob(str(TMP / "*.npz")):
        os.remove(f)
    bounds = np.linspace(0, n_test, a.workers + 1).astype(int)
    procs = []
    for w in range(a.workers):
        lo, hi = int(bounds[w]), int(bounds[w + 1])
        if lo >= hi:
            continue
        procs.append(subprocess.Popen([sys.executable, str(Path(__file__)),
                     "--worker", str(lo), str(hi), str(TMP / f"part_{w}.npz")]))
    failures = []
    for w, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failures.append((w, rc))
    if failures:
        raise RuntimeError(f"sensitivity worker failure(s): {failures}")
    ssum = np.zeros((10, 64)); esum = np.zeros((10, 64)); msum = np.zeros((10, 64)); vsum = np.zeros((10, 2)); n = 0
    for f in glob.glob(str(TMP / "*.npz")):
        z = np.load(f); ssum += z["ssum"]; esum += z["esum"]; msum += z["msum"]; vsum += z["vsum"]; n += int(z["n"])
    if n != n_test:
        raise RuntimeError(f"partial sensitivity data covered {n} samples, expected {n_test}")
    signed, magnitude, evidence, output_voltage = ssum / n, msum / n, esum / n, vsum / n

    # Panel i is the input-weighted "evidence" map; its PNG + colorbar land in the fig3 folder
    # as fig3_i.png / fig3_i_colorbar.png. The aggregated .npz goes under data/.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    evidence_png_prefix = HERE / "fig3_i"
    npz_path = DATA_DIR / "fig3_scikit_all_test_sensitivity.npz"
    dpi = int(a.png_dpi)
    signed_vlim = _auto_abs_limit(signed)
    evidence_vmin = float(EVIDENCE_VMIN)
    evidence_vmax = float(EVIDENCE_VMAX)
    evidence_display = np.clip(evidence, evidence_vmin, evidence_vmax)
    evidence_contrast_gamma = float(EVIDENCE_CONTRAST_GAMMA)
    evidence_vlim = max(abs(evidence_vmin), abs(evidence_vmax))
    evidence_saturation_vlim = np.nan
    signed_saturation_vlim = min(float(DEFAULT_SIGNED_SATURATION_VLIM), signed_vlim)
    magnitude_vmax = float(np.ceil(np.max(magnitude) * 1000.0) / 1000.0)

    _draw_maps(evidence_display, evidence_png_prefix, kind="evidence",
               vmin=evidence_vmin, vmax=evidence_vmax, dpi=dpi, saturation_vlim=None,
               contrast_gamma=evidence_contrast_gamma)
    _draw_colorbar(evidence_png_prefix.with_name(evidence_png_prefix.name + "_colorbar"), kind="evidence",
                   vmin=evidence_vmin, vmax=evidence_vmax, dpi=dpi, saturation_vlim=None,
                   contrast_gamma=evidence_contrast_gamma)
    # The signed (unweighted) map is Supplementary Fig. 4's figure; render it into suppl4/data/
    # (fig3 keeps only the weighted evidence map for panel i).
    _signed_dir = SUPPL4_DIR / "data"
    _signed_dir.mkdir(parents=True, exist_ok=True)
    _draw_maps(signed, _signed_dir / "suppl4_signed_map", kind="signed",
               vmin=-signed_vlim, vmax=signed_vlim, dpi=dpi, saturation_vlim=signed_saturation_vlim)
    _draw_colorbar(_signed_dir / "suppl4_signed_colorbar", kind="signed",
                   vmin=-signed_vlim, vmax=signed_vlim, dpi=dpi, saturation_vlim=signed_saturation_vlim)

    np.savez(npz_path, signed_sensitivity=signed, magnitude_sensitivity=magnitude,
             evidence_attribution=evidence, test_labels=yte, n_test=n_test,
             class_counts=np.bincount(yte, minlength=10), mean_output_voltage=output_voltage,
             signed_vlim=signed_vlim, signed_saturation_vlim=signed_saturation_vlim,
             evidence_vlim=evidence_vlim, evidence_vmin=evidence_vmin, evidence_vmax=evidence_vmax,
             evidence_saturation_vlim=evidence_saturation_vlim,
             evidence_display_clip_min=evidence_vmin, evidence_display_clip_max=evidence_vmax,
             evidence_contrast_gamma=evidence_contrast_gamma,
             magnitude_vmax=magnitude_vmax, gates=str(GATES),
             surface=str("ngspice .op + one-point AC small-signal transfer solve"),
             transfer_method="one-point AC equivalent to ngspice .tf for this DC circuit",
             ac_frequency_hz=AC_FREQ_HZ,
             n_accumulated=n,
             definition=("all-test averages: signed=E_test[J], magnitude=E_test[abs(J)], "
                         "evidence=E_test[x_p*J]; J from ngspice small-signal transfer, "
                         "units V differential output per V input"))
    print(f"device-faithful all-test sensitivity over {n} test samples -> {npz_path} "
          f"(signed_vlim={signed_vlim:.4g}, evidence_vlim={evidence_vlim:.4g})")


if __name__ == "__main__":
    main()
