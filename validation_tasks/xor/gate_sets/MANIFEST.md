# XOR gate-set manifest

This folder preserves the two XOR gate vectors discussed during release assembly.

## Primary clean solution

- File: `clean_seed3_vg_final.npy`
- Also copied to: `../xor_solution.npy`
- Source: `xor/results/pnas_4x4_5runs_10000ep/seed-3/vg_history.npy`, final row
- Original run metadata: `clean_seed3_run_meta.json`
- Clean deterministic ngspice evaluation: `4/4` XOR cases correct (`accuracy = 1.0`)
- Clean training curve: `clean_seed3_acc.npy`
- Distribution summary: 32 gates, min `0.400000 V`, max `2.938915 V`, mean `1.402848 V`,
  median `1.320885 V`; `11/32` gates are below `0.75 V`.

Original command:

```bash
python xor/xor_trainer.py --side 4 --epochs 10000 --gamma 0.3 --eta 0.5 --solver klu --vminus 0.11 --vplus 0.33 --vmax 0.45 --L0 -0.087 --seed 3
```

## Archived noisy seed-1 gate vector

- File: `noisy_seed1_vg_final.npy`
- Source: `xor/results/noisy/xor_s1_mm0.01_mr0.005/vg_final.npy`
- Noisy training accuracy history: `noisy_seed1_acc_history.npy`
- Training condition: seed `1`, VTO mismatch sigma `10 mV`, measurement noise relative scale `0.005`
- Distribution summary: 32 gates, min `0.400000 V`, max `3.805677 V`, mean `1.730590 V`,
  median `1.362680 V`; `9/32` gates are below `0.75 V`.

This archived gate vector was the previous provenance source for `xor_solution.npy`. It is kept for
traceability, but the final saved gates do not solve the clean deterministic XOR evaluation (`accuracy =
0.5`). Its noisy eval history reaches `1.0` at some evaluation points but ends at `0.5` under noisy reads.

## Clean seed 1 note

The canonical clean run with seed `1` is not a solving solution: its recorded clean accuracy is `0.75`
and its final clean accuracy is `0.0`.
