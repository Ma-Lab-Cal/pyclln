# Nonlinear-regression figure-data manifest

Source in the working repository:

- `nonlin_reg/results/pnas_4x4_gamma0p4_truepts_25000ep`

Original generating script and command from `run_meta.json`:

```bash
python nonlin_reg/nonlin_reg_trainer.py --side 4 --epochs 25000 --gamma 0.4 --eta 1.0 --solver klu --vminus 0.0 --vplus 0.45 --mse-stop -1 --init uniform --vg-init-lo 3.0 --vg-init-hi 3.0 --seed 0
```

Release-native equivalent script:

```bash
cd paper_release
RUN_DIR=figure_data/suppl2_xor_nonlin/nonlin_reg/pnas_4x4_gamma0p4_truepts_25000ep_reproduced \
  python nonlinear_regression/reg_circuit.py --side 4 --epochs 25000 --gamma 0.4 --eta 1.0 --solver klu --vminus 0.0 --vplus 0.45 --mse-stop -1 --init uniform --vg-init-lo 3.0 --vg-init-hi 3.0 --seed 0
```

Bundled files:

| File | Shape / role |
|---|---|
| `pnas_4x4_gamma0p4_truepts_25000ep/run_meta.json` | run metadata, dataset points, node map, original command |
| `pnas_4x4_gamma0p4_truepts_25000ep/mse_history.npy` | `(25001,)` MSE history |
| `pnas_4x4_gamma0p4_truepts_25000ep/preds_history.npy` | `(25001, 8)` predicted output voltages by epoch and input point |
| `pnas_4x4_gamma0p4_truepts_25000ep/netlist_initial.cir` | initial ngspice netlist for provenance |

The plotting script reads the training input/output points from `run_meta.json` and overlays selected
prediction snapshots from `preds_history.npy`.
