# Nonlinear regression on an analog NMOS input-output network

A 4×4 NMOS input-output network trained by **coupled learning** to fit a nonlinear 1-D target function, demonstrating
the same local two-phase rule on a continuous-output regression task (companion to the XOR classification
task).

## Circuit
- Input is a source voltage; the trainable weights are the NMOS **gate voltages**; the prediction is the
  differential output-node voltage at the DC operating point.
- Training step (identical rule to XOR): **free** solve → **clamped** solve (output nudged toward target
  through a clamp) → local contrastive gate update. No backpropagation.

## Result (measured in ngspice)
| | clean | + device noise (10 mV mismatch, scaled meas) |
|---|---|---|
| test MSE | 1.7e-5 | 1.8e-5 / 8.6e-5 / 1.2e-4 (3 chips) |

Device noise adds a modest error floor — the network still fits the target tightly, chip MSE staying at or below ~1e-4.

## Files
- `train_regression.py` — builds the input-output netlist and runs coupled learning in ngspice.
- `reg_circuit.py` — circuit/graph helpers (target dataset, netlist nodes, voltage readout).
- `reg_solution.npy` — trained gate voltages (seed-0 clean run, MSE 1.7e-5).
- `results.json` — summary metrics.

## Run
```
conda run -n p311env python train_regression.py --seed 0          # clean (noise off by default)
conda run -n p311env python train_regression.py --seed 0 --chip chips/chip_1.npz --meas-rel 5e-3   # noisy
```
Hyperparameters: η = 1.0, γ = 0.4 (clamp nudge).
