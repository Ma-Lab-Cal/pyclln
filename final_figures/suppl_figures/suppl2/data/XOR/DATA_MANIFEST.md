# XOR figure-data manifest

Source in the working repository:

- `xor/results/pnas_4x4_5runs_10000ep/seed-3`

Original generating script and command from `run_meta.json`:

```bash
python xor/xor_trainer.py --side 4 --epochs 10000 --gamma 0.3 --eta 0.5 --solver klu --vminus 0.11 --vplus 0.33 --vmax 0.45 --L0 -0.087 --seed 3
```

Release-native equivalent script:

```bash
cd paper_release
RUN_DIR=figure_data/suppl2_xor_nonlin/XOR/pnas_4x4_5runs_10000ep/seed-3_reproduced \
  python xor/xor_circuit.py --side 4 --epochs 10000 --gamma 0.3 --eta 0.5 --solver klu --vminus 0.11 --vplus 0.33 --vmax 0.45 --L0 -0.087 --seed 3
```

Bundled files:

| File | Shape / role |
|---|---|
| `pnas_4x4_5runs_10000ep/seed-3/run_meta.json` | run metadata, dataset constants, node map, original command |
| `pnas_4x4_5runs_10000ep/seed-3/0_acc.npy` | `(10001,)` accuracy history |
| `pnas_4x4_5runs_10000ep/seed-3/0_outputs.npy` | `(10000, 4)` XOR output voltages by epoch and truth-table row |
| `pnas_4x4_5runs_10000ep/seed-3/vg_final.npy` | `(32,)` final clean seed-3 gate voltages used for the plotted run |
| `pnas_4x4_5runs_10000ep/seed-3/netlist_initial.cir` | initial ngspice netlist for provenance |

The plotting script derives XOR MSE from `0_outputs.npy` and the target levels stored in `run_meta.json`.
