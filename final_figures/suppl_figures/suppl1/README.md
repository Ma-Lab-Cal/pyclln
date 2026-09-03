# Supplemental Figure 1

XOR scaling comparison for one per-sample update on the `xor_nxn`
alter benchmark.

The manuscript uses a single panel: the log-log timing plot (`suppl1.png`).

Outputs (PNGs written to this folder):

- `suppl1.png`: **manuscript figure** — 50-epoch mean wall-clock time per
  sample update with log x- and y-axes, loaded from
  `data/suppl1_xor_timing_summary.csv`.
- `suppl1_linear_timing.png`: same timing data with linear x- and y-axes
  (not used in the manuscript; kept for reference).
- `suppl1_memory.png`: peak RSS during one free+clamped XOR update, loaded
  from `data/suppl1_xor_memory_summary.csv` (not used in the manuscript;
  kept for reference).
- `suppl1_memory_timing_combined.png`: legacy two-panel composite
  (a: linear timing, b: memory); not used in the manuscript.

Source data (written to `data/`):

- `data/suppl1_xor_timing_summary.csv`
- `data/suppl1_xor_memory_summary.csv`
- `data/suppl1_methodology.json`

Both timing and memory panels use sides:

```text
3 5 7 10 15 20 30 40 50 60 70 80 90 100
```

Regenerate plots from saved source data:

```bash
python final_figures/suppl_figures/suppl1/plot_suppl1_xor_scaling.py
```

Measure or refresh additional timing rows, merge into the timing CSV, then plot:

```bash
python final_figures/suppl_figures/suppl1/plot_suppl1_xor_scaling.py \
  --measure-timing \
  --timing-sides 3 5 7 15
```

Measure or refresh memory rows, merge into the memory CSV, then plot:

```bash
python final_figures/suppl_figures/suppl1/plot_suppl1_xor_scaling.py \
  --measure-memory \
  --memory-sides 3 5 7 15
```
