# Contrastive local learning in large nonlinear analog networks

This is the repository associated with the manuscript *"Contrastive local learning in large nonlinear analog networks."*

`PyCLLN` is a simulation platform for building and training nonlinear analog transistor networks by
contrastive local learning. Networks are defined as parameterized circuits and are trained and evaluated
directly in **SPICE** (ngspice). This repository provides the network definitions, trainers, trained
results, and the figure-generation code for the tasks in the manuscript.

For correspondence: pragyanpandey05@berkeley.edu.

## Layout

```
device_model/            NMOS device model (.lib reference; trainers use a matching inline wrapper)
common/                  shared SPICE solver + coupled-learning update + noise model
validation_tasks/
  xor/                   trainer, topology, chips, clean + 3 noisy runs, results, README
  nonlinear_regression/  "
scikit_digits/           trainer, topology, chips, clean + 3 noisy runs, results, README
ionosphere/              "
language_model/          trainer, vocabulary + sentences + embeddings, chips, runs, results, README
timing_comparison/       compute-time scaling benchmark (circuit-solve time vs network size)
hparam_noise_sweeps/     hyperparameter, loss, device-mismatch, and read-noise sweeps
final_figures/           scripts + compact source data for the paper and supplement figures
```

Each task folder is self-contained: a `train_*.py` that builds the full circuit netlist and runs coupled
learning, the topology and chip fingerprints as data files, `runs/{clean,noisy_chip1,2,3}/` (per-run
metadata + training curve + trained gates), the final results, and a short README describing the circuit
and how to reproduce it.

## Installation

The code needs Python 3.11, the scientific-Python stack, and **ngspice**, the native SPICE
engine that PySpice drives (ngspice is a system library, not a pip package). The one-command,
fully-pinned path uses conda and installs ngspice for you:

```
conda env create -f environment.yml     # creates the `clln` env: Python 3.11 + ngspice 41 + pinned deps
conda activate clln
```

Alternatively, install ngspice yourself (`conda install -c conda-forge ngspice=41`, or a system
package such as `apt-get install ngspice`) and the Python dependencies with pip:

```
pip install -r requirements.txt
```

All versions in `requirements.txt` are pinned to the tested environment (ngspice 41; PySpice 1.5,
which prints a harmless "Unsupported Ngspice version 41" on import). `transformers` is needed only
to regenerate the SciBERT embeddings in `language_model/embeddings/`; training and inference load
the shipped embeddings and do not require it.

## Running

```
conda activate clln

# Train + SPICE-validate a task. Run each from inside its task folder so relative data paths
# (topology, chips/) resolve -- the subshells below cd in and return you to the repo root:
( cd validation_tasks/xor && python train_xor.py )                        # trains + SPICE-validates XOR
( cd scikit_digits && python train_scikit.py --chip chips/chip_1.npz )    # a noisy chip

# Regenerate a paper figure (run from anywhere -- each figure script locates the repo root itself):
python final_figures/main_figures/fig5/plot_fig5.py
```
(See each task's README for arguments, including the `--chip` device-mismatch fingerprint and the
`--meas-rel` read-noise lever.)
