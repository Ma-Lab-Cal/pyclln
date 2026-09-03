# scikit-digits (8×8) on an analog NMOS network

A 1,247-edge NMOS network classifies the sklearn 8×8 handwritten digits (10 classes), trained
end-to-end **in ngspice** by **coupled learning** — the local, two-phase contrastive rule. No
backpropagation, no feature front-end: the 64 pixels are applied directly as node voltages.

## Circuit
- 64 pixel inputs ([0,1] V) × 20 output rails — 10 classes × {+,−}; the class score is the
  **differential** rail voltage, score_c = V(c+) − V(c−), prediction = argmax.
- Each edge is one NMOS transistor: drain at a pixel node, source at a rail, learned
  weight = gate voltage (referenced to ground), **body tied to the drain node**. Rails read out
  through high-Z (1 GΩ) sense resistors.
- Training step (one per violating sample, strict SGD): solve the **free** circuit and read all
  nodes → multiclass hinge (margin 0.02 V) → solve the **clamped** circuit with the true rail
  pair nudged +δ/2 / −δ/2 and the top rival ∓δ/2 through stiff clamps → update each gate
  locally: ΔVg = −γ·(ΔV_clamp² − ΔV_free²), clip [0.4, 8.0] V.

## Noise model (`--chip chip_N.npz`)
- **Device mismatch**: per-transistor VTO offset (σ = 10 mV) and drive-strength (kp) factor (σ = 0.5%). The free and clamp phases run on two
  electrically separate copies of the network — independent physical devices, as on a real chip. A **chip** is its mismatch fingerprint, shipped as data (`chip_1/2/3.npz`:
  `vto_free`/`vto_clamp` + `beta_free`/`beta_clamp`, 1,247 devices each).
- **Read noise**: Gaussian σ = 5 mV (0.5% of the 1 V input full scale) on **every** measurement —
  training, per-epoch evaluation, and inference.

## Result (measured in ngspice)
| run | untrained | test acc (endpoint) |
|---|---|---|
| clean | 0.1361 | **0.9806** @ epoch 15 |
| chip 1 | 0.1417 | **0.9694** @ epoch 30 |
| chip 2 | 0.1417 | **0.9694** @ epoch 30 |
| chip 3 | 0.1417 | **0.9694** @ epoch 30 |

Training starts at chance (random Uniform(0.5, 4.5) V gate init, seed 0) and the network, recipe,
and split are identical across all runs. Endpoint gates and accuracy curves for every run are in
`results/run_*_gates.npz`.

## Files
- `train_scikit.py` — the complete trainer: builds the full monolithic netlist (both circuit
  copies) and runs coupled learning in ngspice. Clean by default; `--chip` enables the noise model.
- `infer_scikit.py` — deployment inference: N fresh-read draws at the endpoint gates.
- `topology_1247.npz` — the network (drain/source node per edge).
- `chip_1/2/3.npz` — the three characterized device-mismatch fingerprints.
- `results/run_{clean,chip1,chip2,chip3}_gates.npz` — recorded runs: endpoint gates `vg_final`,
  the untrained baseline (`vg0`/`acc0`), the per-epoch test-accuracy curve, and `drain`/`source`
  (noisy archives also carry the chip fingerprint `vto_*`/`beta_*`).
- `results/updcurve_clean.npz` — the **clean** run at **per-update** resolution (for plotting):
  `train_acc`, `train_loss`, `test_acc`, `test_loss` measured after **every applied gate update**
  (index 0 = untrained), plus `epoch` and `sample` (the epoch and training-sample index of each
  update, so points can be placed on a cumulative-update or epoch axis). 3,422 points.
- `results/results.json` — summary metrics and protocol.

## Run
```
conda run -n p311env python train_scikit.py                      # clean
conda run -n p311env python train_scikit.py --chip chip_1.npz --epochs 30    # full 30-epoch noise budget, chip 1
conda run -n p311env python infer_scikit.py --chip chip_1.npz --run results/run_chip1_gates.npz
```
Hyperparameters: γ = 1.0, δ = 0.05 V, margin = 0.02 V, seed-0 stratified 80/20
split (1437/360). The shipped archives are the recorded realizations; a re-run draws fresh read
noise per measurement, so noisy trajectories agree to within a ±1–2 sample (~0.3–0.6%) read-noise
variation at each epoch.
