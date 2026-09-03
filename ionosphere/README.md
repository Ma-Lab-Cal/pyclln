# UCI ionosphere on an analog MOS network

A 147-edge NMOS network classifies UCI ionosphere radar returns (binary good/bad, 351 samples ×
34 features), trained end-to-end **in ngspice** by **coupled learning** — the local, two-phase
contrastive rule. No backpropagation, no feature front-end: the 34 features are applied directly as
node voltages, with a clean, **label-independent topology**.

## Circuit (147 edges)
- **34 inputs → 5 hidden → 2 outputs.** Prediction = argmax of the two output-node voltages.
- Edge breakdown (each edge = one NMOS, gate voltage = learned weight, gate ref = drain, body float):
  - **68 input → output** (direct): every input drives both output rails — the linear/additive path.
  - **68 input → hidden** (sparse, **uniform**): **every input sends exactly 2 edges** to the hidden
    layer, assigned round-robin over the 5 hidden nodes (balanced fan-in 14/14/14/13/13). This is
    **label-independent** — no feature is favored; coupled learning decides what matters. (This
    replaces an earlier design where only the 9 most label-correlated inputs fed hidden.)
  - **10 hidden → output**: all 5 hidden nodes drive both rails.
  - **1 output ↔ output**: a single lateral edge between the two class rails.
- **Purely input-driven** — no device connects to the supply rail (negref/posref), so no VMINUS/VPLUS
  sources are emitted; the network is driven entirely through the input sources.
- Operating point: inputs ±0.8 V, **fixed** gate init 2.0 V (a single scalar applied to all 147 gates).

## Result (measured in ngspice)
All four runs are reported at the **same round endpoint epoch (50)**. The learning rate (γ=4) is
deliberately gentle, giving a **smooth learning curve** (see `results/curves.png`) while the fixed
2.0 V gate init keeps the network confident enough for the noisy chips to hold their accuracy.

| run | network (epoch 50) | test accuracy |
|---|---|---|
| **clean** | epoch-50 endpoint | **0.9718 = 69/71 (>97%)** |
| chip 1 | epoch-50 endpoint | 0.9577 = 68/71 (>95%) |
| chip 2 | epoch-50 endpoint | 0.9577 = 68/71 (>95%) |
| chip 3 | epoch-50 endpoint | 0.9437 = 67/71 (>94%) |

**Noise convention (same as the digits task).** Each chip is a fixed device-mismatch fingerprint
(`chip_N.npz`) plus read noise, applied consistently in training and inference. Clean is deterministic
(69/71 = 0.9718); the three chips give 68/71, 68/71, 67/71 at epoch 50, each reproducing exactly.


## Files
- `train_ionosphere.py` — complete trainer (147-edge recipe as defaults). `--chip chip_N.npz` enables noise.
- `topology_147.npz` — the network. `ionosphere.data` — the UCI dataset.
- `chip_1/2/3.npz` — the three device-mismatch fingerprints (data only).
- `runs/{clean,noisy_chip1,noisy_chip2,noisy_chip3}/` — one folder per run, each with `run_meta.json`
  (all hyperparameters + result + noise spec), `gates.npz` (`vg_init` untrained + `vg_final` trained),
  `curve.npz` (learning curve), and `chip.npz` (the fingerprint; noisy runs only).
- `results/results.json` — top-level metrics.

## Run
```
conda run -n p311env python train_ionosphere.py                      # clean (0.9718 at epoch 50)
conda run -n p311env python train_ionosphere.py --chip chips/chip_1.npz    # full noise model, chip 1
```
Recipe defaults: input-scale 0.8, γ = 4, δ = 0.01, margin = 0.01, vg-init 2.0,
clip [0.4, 8.0], 50 epochs, seed-0 split.

## Design lineage
The two outputs share **one** lateral edge (not two), and every input feeds the hidden layer
**uniformly** (2 edges each, by index) — so nothing in the architecture depends on the labels. This
is the most defensible of the three ionosphere designs.
