# Compute-time benchmark (circuit-solve time vs network size)

A supplementary benchmark measuring how the per-step SPICE solve time scales with
network size, comparing two ways of driving ngspice on the same N×N analog XOR grids:

- **PyCLLN** — one shared-process ngspice instance, updated in place (`alter`) each step.
- **Naive Spice** — a fresh ngspice subprocess per step.

Each is run across the KLU and sparse solvers. The manuscript figure is **Supplemental
Figure 1** (`final_figures/suppl_figures/suppl1/`), which reads the shipped summary data
and plots mean wall-clock time per sample versus edge count on log axes.

## Contents

- `source_snapshot/` — the timing-benchmark trainers and small ngspice solver used to
  measure the timing rows. These are imported by the Supplemental Figure 1 script only
  in its optional `--measure-timing` mode; the figure otherwise regenerates from the
  shipped CSVs.

## Regenerate the figure

```bash
python final_figures/suppl_figures/suppl1/plot_suppl1_xor_scaling.py
```
