# Loss comparison (Fig. 5 loss-bars panel)

Endpoint (end-of-budget) **test accuracy** for each task × loss, on the released task trainers. Each loss is
run at its task's **main config** with matched clamp perturbations — the comparison isolates the loss
function, not any per-loss tuning.

| file | what |
|---|---|
| `loss_comparison.csv` | the panel data: `task`, `loss`, `endpoint_test_acc`, `endpoint_epoch`, `epochs`, `source`, `config_params` (6 rows) |

Reproduce by running each task's released trainer with the matching `--loss`:
`train_scikit.py --loss {hinge,mse,cross_entropy}` (15 epochs) and `train_ionosphere.py --loss {…}`
(50 epochs), reading the final-epoch test accuracy.

The `--loss` clamp variants: **hinge** = margin-gated true/rival rail clamp (stops perturbing an output once
its margin is met); **mse** = each output nudged toward its one-hot target in proportion to the error, n_k = δ·(t_k − V_k) (correct class t = +target, all others 0);
**cross_entropy** = softmax-probability nudge n_k = δ·(q_k − p_k) at readout temperature T. Hinge wins on both tasks (scikit-digits
**0.9806**, ionosphere **0.9718**); mse and cross-entropy train but the endpoint network settles lower —
the proportional mse nudge keeps a restoring push on every output even after a sample is
already classified correctly.
