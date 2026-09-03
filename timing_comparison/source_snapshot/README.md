# Timing-benchmark source

Trainers and a small ngspice solver for the N×N analog XOR compute-time benchmark
(see `../README.md`). Two backends are compared on identical circuits:

- **shared** (`xor_nxn/xor_nxn_alter_trainer*.py`) — one persistent ngspice instance,
  updated with `alter` each step (labelled **PyCLLN** in the figure).
- **subprocess** (`xor_nxn/xor_nxn_alter_trainer_subproc*.py`) — a fresh `ngspice -b`
  call per step (labelled **Naive Spice**).

Both support the KLU and sparse solvers. These modules are used by the Supplemental
Figure 1 script in its optional `--measure-timing` mode; the figure otherwise
regenerates from the shipped summary CSVs.
