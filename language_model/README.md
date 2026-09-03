# Physics Language Model on an analog NMOS network

An autoregressive next-token language model over short quantum-physics sentences, run entirely as an
**input-output NMOS network** and trained end-to-end **in ngspice** by **coupled learning**
— the local, two-phase contrastive rule. No backpropagation. A 7-token context is turned into gate-input
voltages by a fixed SciBERT-FA embedding; the network's branch voltages are the next-token logits.

## Circuit (13,664 edges)
- **122 output branches × 112 gate inputs.** Each edge (an input-output connection) is one NMOS transistor: gate =
  learned weight, source = a context-embedding input voltage, drain = the shared per-branch output node.
- **Input:** a 7-token context × 16-D SciBERT-FA embedding = 112 gate-input voltages, applied in
  [0.0, 0.45] V (no analog front end).
- **Output:** the 122 branch drain voltages are the 122 next-token logits (`<BOS>` is never an output
  branch) via a masked softmax at the readout temperature.
- Prediction/generation samples the next token from that softmax; sentences are generated autoregressively.

Vocabulary: **122 tokens**, plus a `<BOS>` start-of-sequence marker (123 symbols; `<BOS>` is never generated); 7 sentence-opener determiners (`the/a/an/this/that/each/every`); 4
balanced grammatical shapes, including the `what is the … ?` question family.

## Results (device-faithful, real ngspice; mixed-opener autoregressive generation, five 1000-sentence batches, T=0.0006)

| run | endpoint epoch | valid-string rate |
|---|---|---|
| **clean** | 12 | **1.0000** |
| chip 1 | 12 | 0.9986 |
| chip 2 | 12 | 0.9970 |
| chip 3 | 12 | 0.9982 |
| **3-chip mean** | | **0.9979** |

Noise model (`chips/chip_{1,2,3}.npz`): per-transistor threshold-voltage mismatch (σ = 10 mV) + per-device
drive-strength mismatch (σ = 0.5%), fixed per chip, with free and clamp as electrically-separate device
sets; plus additive read noise (σ = 2.25 mV) on every measurement. Noise costs only ~0.2% validity.

## Layout
```
topology.npz          dense input-output net: branches=122, gate_inputs=112, context_len=7,
                      embed_dim=16, physical_output_ids
vocab.txt             the 122 vocabulary tokens plus `<BOS>` (123 lines; `<BOS>` first)
sentences.txt         the self-valid quantum-physics corpus (one sentence per line)
embeddings/
  scibert_fa16.json   the 123 x 16 embedding matrix the trainer loads
  build_embeddings.py reproducibility: rebuilds scibert_fa16.json from vocab.txt via SciBERT + FactorAnalysis
token_classes.json    token -> grammatical role (used only for the network figure's node coloring)
chips/chip_{1,2,3}.npz  three device-mismatch fingerprints (VTO + drive-strength, free/clamp device sets)
runs/
  clean/               run_meta.json + curve.npz (per-epoch) + gates.npz (vg_init, vg_final = 122x112)
  noisy_chip{1,2,3}/   "                                                        (trained under chip N)
results/
  results.json             full summary (network, corpus, headline, training recipe)
  valid_generations_clean.json  unique valid on-circuit generations from the clean model
train_language_model.py  coupled-learning trainer in ngspice (corpus generated from the grammar; loads embeddings)
infer_language_model.py  read next-token distributions / generate sentences on ngspice from a trained run
```

`runs/<run>/curve.npz` holds per-epoch metrics (`train_ce`, `test_ce`, `test_support_acc`, `test_qmass`,
`gen_validity_test_T0006`; the clean run also has `train_qmass`, `test_exact`) over the fixed held-out
window set — these are plotted in `final_figures/main_figures/fig4`.

## Running
```
# read the next-token distribution for a context on real ngspice
conda run -n p311env python infer_language_model.py --run clean --context "the electron has spin in the"

# generate sentences from a trained run
conda run -n p311env python infer_language_model.py --run noisy_chip1 --chip chips/chip_1.npz --generate 20

# retrain from scratch: the corpus is generated from the grammar, embeddings are loaded (optionally under a chip)
conda run -n p311env python train_language_model.py [--chip chips/chip_1.npz] [--epochs 12]
```

## Corpus design (why it works)
- Subject determiners span all 7 openers; the demonstratives/quantifiers (`this/that/each/every`) are
  agreement-free, so *"every electron has charge"* reads as a physical law.
- Object determiners are fixed to `the`, which removes the autoregressive a/an agreement trap (the model
  would otherwise have to emit `a`/`an` before seeing the noun) while keeping full opener diversity.
- The reduced 122-token vocabulary (plus `<BOS>` = 123 symbols) drops unused tokens and adds the what-is question family, giving a
  compact 13,664-edge network with no dead tokens.
