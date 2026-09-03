"""
Physical non-ideality model shared across all released CLLN task trainers, so the noise is identical
everywhere. Both effects are OFF for a clean run and ON for a noisy run.

(1) MEASUREMENT NOISE -- additive Gaussian on each read-out node voltage, fresh per measurement. The floor
    is set by the read-out resolution (ADC / instrumentation-amp LSB), a fixed fraction of the task's
    FULL-SCALE operating range: sigma_meas = MEAS_REL * V_FS, with MEAS_REL = 5e-3 (5 mV at a 1 V full-scale
    range), scaling linearly with V_FS. Per task: XOR / regression V_FS ~ 0.45 V -> ~2.25 mV; ionosphere
    input full-scale 0.8 V -> ~4.0 mV. An absolute thermal floor MEAS_FLOOR can be added in quadrature.

(2) DEVICE MISMATCH -- shipped as a FIXED per-chip fingerprint FILE (chip_1/2/3.npz), loaded by each
    trainer; it is data, not generated at run time. A fingerprint stores, per device: a threshold-voltage
    offset (`vto_free`/`vto_clamp`, added to VTO_NOM) and a drive-strength factor (`beta_free`/`beta_clamp`,
    multiplying KP_NOM). An E-edge network has 2E transistors -- the E devices in the free circuit and the E
    in the clamped circuit are independent, so a fingerprint carries separate free/clamp arrays. Each edge
    instantiates a parametrized NMOSWRAP whose .model is templated `vth={vth} kpval={kpval}`, so
    every instance is a genuinely different device from one shared .lib. Reference spreads: VTO sigma = 10 mV
    (representative NMOS-array Vth spread), drive-strength sigma = 0.5%.
"""
import numpy as np

MEAS_REL = 5.0e-3              # read-out noise = 5 mV at full-scale 1 V; scales linearly with V_FS
MEAS_FLOOR = 0.0              # optional absolute thermal floor (V); ~sub-uV here, off by default
VTO_NOM = 0.75               # nominal zero-bias threshold voltage (V)
KP_NOM = 6.249608378152027e-05   # nominal transconductance parameter

# Parametrized device wrapper for ngspice: per-instance VTO + drive strength via a templated .model (one
# shared .lib). Each edge instantiates `X... NMOSWRAP vth=<VTO_NOM+offset> kpval=<KP_NOM*factor>`.
NMOSWRAP_PARAM = (
    ".subckt NMOSWRAP D G S B vth=0.75 kpval=6.249608378152027e-05\n"
    ".model m_loc nmos(level=1 vto={vth} gamma=1.09 phi=0.9499477708465783 tpg=1 "
    "kp={kpval} lambda=0.19998491299329302 rsh=73.21306042358299)\n"
    "M0 D G S B m_loc l=7.8e-6 w=0.138e-3 as=0.603e-8 ps=0.478e-3 ad=0.161e-8 nrd=.3 nrs=1\n"
    ".ends NMOSWRAP\n"
)


def meas_sigma(v_scale, rel=MEAS_REL, floor=MEAS_FLOOR):
    """Measurement-noise sigma (V) for a read-out FULL-SCALE v_scale (V): rel*v_scale in quadrature with the
    absolute floor. e.g. v_scale=1.0 -> 5 mV; v_scale=0.45 (XOR/reg) -> ~2.25 mV."""
    return float((float(rel) * float(v_scale)) ** 2 + float(floor) ** 2) ** 0.5


def add_measurement_noise(V, rng, v_scale=None, sigma=None):
    """Additive Gaussian read-out noise on voltages V (numpy), fresh per call. Provide the task's read-out
    FULL-SCALE v_scale (-> sigma = rel*v_scale, the physical readout-resolution floor) OR an explicit absolute
    sigma. No-op if the resulting sigma<=0 (noise off)."""
    s = float(sigma) if sigma is not None else (meas_sigma(v_scale) if v_scale else 0.0)
    if s <= 0:
        return V
    return V + rng.standard_normal(np.shape(V)).astype(getattr(V, "dtype", np.float64)) * s
