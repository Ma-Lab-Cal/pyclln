# XOR on an analog NMOS input-output network

A 4×4 NMOS-transistor input-output network with a **differential output** (O = V(o+) − V(o−)) is trained to compute
XOR by **coupled learning** — the local, two-phase contrastive rule (reproducing PNAS-2024 Fig. 3A on a
device-faithful SPICE circuit).

## Circuit
- Inputs are two source voltages (the two XOR operands); the trainable weights are the NMOS **gate
  voltages**; the prediction is the differential output-node voltage at the DC operating point.
- Training step: solve the **free** circuit (read all node voltages), then solve the **clamped** circuit
  (the output nudged toward its target through a stiff clamp), then update each gate locally by the
  contrastive difference of the two phases. No backpropagation.

## Result (measured in ngspice)
The clean network solves XOR (all four input cases correct at the differential output); 2 of 5 canonical seeds
converge to a solving network. Under a device chip the single-shot differential output is unstable near
threshold, so noisy XOR is not claimed as a result; the noisy runs ship as data for provenance.

## Files
- `train_xor.py` — builds the full input-output netlist and runs coupled learning in ngspice.
- `xor_circuit.py` — circuit/graph helpers (dataset, netlist nodes, voltage readout).
- `xor_solution.npy` — primary clean solving gate voltages, from canonical seed 3.
- `gate_sets/` — clean seed-3 solution plus the archived noisy seed-1 gate vector retained for provenance.
- `results.json` — summary metrics.

## Run
```
conda run -n p311env python train_xor.py --seed 0          # clean (noise off by default)
conda run -n p311env python train_xor.py --seed 0 --chip chips/chip_1.npz --meas-rel 5e-3   # with device noise
```
Hyperparameters: η = 0.5, γ = 0.3 (clamp nudge).
