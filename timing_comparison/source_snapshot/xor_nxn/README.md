# XOR N×N timing trainers

Trainers for the N×N analog XOR compute-time benchmark (see `../../README.md`).

- `xor_nxn_trainer.py` — reference trainer built on the small ngspice solver in
  `../sim/spice_net.py`.
- `xor_nxn_alter_trainer.py` — shared-instance trainer: one ngspice process, updated
  with `alter` each step (the **PyCLLN** backend).
- `xor_nxn_alter_trainer_subproc.py` — subprocess trainer: a fresh `ngspice -b` call
  per step on the identical circuit (the **Naive Spice** backend).
- `*_lvl1.py` — ideal-NMOS wrappers around the shared / subprocess trainers.
- `analyze_xor_nxn_timings.py`, `xor_nxn_stack_tests.py` — timing aggregation and a
  quick smoke test.

Each backend/solver combination logs standardized per-epoch timing arrays; the
aggregated summary drives Supplemental Figure 1.
