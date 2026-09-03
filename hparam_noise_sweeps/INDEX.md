# scikit-digits — all sweeps (plotting index)

**One place for every robustness sweep** (hyperparameter, loss, device-mismatch, and read-noise). The
fixed **operating point** is **drain/ground** (body tied to drain, gate referenced to ground), clean
**0.9806**. Every run is device-faithful ngspice on the 1,247-edge differential input-output
network, seed-0 data split. **All accuracies are the endpoint (end-of-budget) test accuracy** — the network
state after the last training epoch, never a peak-epoch selection. Paths are relative to `scikit_digits/`.

```
hparam_noise_sweeps/
├── hyperparam/          device / init / gamma-delta grids around the operating point, endpoint acc
├── loss/                loss comparison (hinge / mse / cross-entropy), both tasks
├── mismatch_inference/  device-mismatch robustness of the fixed clean operating point (+ per-draw raw)
├── read_noise/          measurement read-noise degradation of the clean operating point (+ per-run raw)
└── INDEX.md             (this file)
```

## hyperparam/ — robustness grids (operating point drain/ground, endpoint@ep15)
Each grid CSV carries `endpoint_test_acc` (end-of-budget test accuracy), `endpoint_epoch`, `untrained_acc`
(accuracy of the untrained network, chance-level), `test_loss` (final multiclass-hinge test loss), and
`test_acc_recheck` (re-evaluated accuracy — a sanity column that matches `endpoint_test_acc`).

| file | columns (key) | cells | plot |
|---|---|---|---|
| `device_matrix.csv` | `body_tie`, `gate_ref`, `endpoint_test_acc`, `untrained_acc`, `status` | 16 | **4×4 heatmap** body×gate. Operating point **drain/ground 0.9806**; every mode 0.967–0.981 (robust to the body/gate choice). |
| `init_heatmap.csv` | `init_mean`, `init_half_spread`, `endpoint_test_acc`, `clip_fraction`, `is_reference` | 121 | **11×11 heatmap** of the gate-init distribution U(mean−h, mean+h). Reference init U(0.5, 4.5) (mean 2.5, half-spread 2.0) → **0.9806**. |
| `gamma_delta_m0.02.csv` | `gamma`, `delta`, `endpoint_test_acc`, `status` | 49 | **γ×δ heatmap** at the operating margin (0.02), γ∈[0.03…30], δ∈[0.005…0.5] V. **0.9806** at the operating point (γ=1, δ=0.05), within a broad high-accuracy region of the grid. |

## loss/ — loss comparison (endpoint, both tasks)
Each row is a single run of one training objective (`--loss`) on one task, reported at the endpoint
(end-of-budget network state). Every run carries its endpoint **test accuracy, test loss, train accuracy,
train loss**, and the full hyperparameters for that run (γ, δ, margin, softmax temperature or MSE target,
gate-init distribution, body/gate reference, input range, epochs).

| file | contents | plot |
|---|---|---|
| `loss_comparison.csv` | one row per run: `task`, `loss`, `endpoint_test_acc`, `test_loss`, `train_acc`, `train_loss`, `endpoint_epoch`, `epochs`, and all hyperparameter columns (6 rows) | **grouped bar chart** (2 tasks × 3 losses) — table below |
| `runs/<task>_<loss>.json` | the full per-run record (metrics + hyperparameters) for each of the 6 runs | — |

Clamp variants: **hinge** = margin-gated true/rival rail clamp (stops once the margin is met); **mse** =
each output nudged toward its one-hot target in proportion to the error, `n_k = δ·(t_k − V_k)` (correct
class `t = +target`, all others 0); **cross_entropy** = softmax-probability nudge `n_k = δ·(q_k − p_k)` at
temperature T. Endpoint test accuracy:

| task | hinge | cross_entropy | mse |
|---|---|---|---|
| scikit-digits (ep15) | **0.9806** | 0.9556 | 0.9194 |
| ionosphere (ep50) | **0.9718** | 0.9718 | 0.7887 |

Hinge is strongest on both tasks: it stops perturbing an output once its margin is satisfied, so the
endpoint network is stable. The proportional mse nudge keeps a restoring push on every output through the
budget, leaving a lower endpoint accuracy — most sharply on ionosphere.

## mismatch_inference/ — device-mismatch robustness (fixed clean gates)
No retraining: the clean operating-point gates are evaluated under increasing device-mismatch magnitude, with
**5 device-mismatch draws** per scale (read noise off).

| file | contents | plot |
|---|---|---|
| `mismatch_inference_summary.csv` | per scale: `n_chips`, `test_acc_mean/std/min/max`, `test_loss_mean/std` | **accuracy/loss vs mismatch magnitude** (mean ± s.d. over 5 draws) |
| `mismatch_inference_perdraw.csv` | every individual draw: `noise_scale`, `mismatch_seed`, `test_acc`, `test_loss` | the individual points behind each summary row |

Sweep magnitudes (1× = the nominal device fingerprint): VTO σ and drive-strength (kp) σ scale together, from
0.2× (VTO 2 mV, kp 0.1 %) through 50× (VTO 500 mV, kp 25 %). Accuracy holds at 0.9806 through 2×, is 0.973 at
10×, 0.961 at 20×, and 0.89 at 50×.

## read_noise/ — measurement read-noise degradation
The clean operating-point network is retrained under Gaussian read noise of σ (mV) added to every node
measurement, mismatch off, **5 read-noise runs** per level, 30-epoch budget, endpoint.

| file | contents | plot |
|---|---|---|
| `read_noise_sweep.csv` | per σ: `endpoint_test_acc`, `test_acc_std/min/max`, `endpoint_test_loss`, `n_seeds` | **accuracy/loss vs read-noise σ** (mean ± s.d. over 5 runs) |
| `read_noise_perseed.csv` | every run: `noise_mV`, `read_seed`, `endpoint_test_acc`, `endpoint_test_loss` | the individual points behind each summary row |

Endpoint accuracy is 0.958 noise-free, 0.948 at 10 mV, 0.90 at 30 mV, and falls to chance by ~100 mV.

## Notes for the plotter
- Every `*.csv` is a plain header+rows table (`csv.DictReader`).
- All accuracies are **endpoint** (end-of-budget), not peak-epoch.
- Reference only the **3** shipped chips.
- The operating-point (drain/ground) grids are the primary figures.
