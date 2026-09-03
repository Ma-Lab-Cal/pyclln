"""Shared helpers for the language-model trainer and inference: token/id maps, the gate range,
the embedding matrix, context encoding, and the training-target distributions."""
import os
from typing import Dict, Tuple

import numpy as np

from clm.embeddings import EmbeddingManifest
from clm.vocab import VOCAB

# token <-> id maps and the output-branch index set (<BOS> is masked, not an output branch)
TOKEN_TO_ID = {w: i for i, w in enumerate(VOCAB)}
ID_TO_TOKEN = {i: w for i, w in enumerate(VOCAB)}
BOS_ID = TOKEN_TO_ID["<BOS>"]
OUTPUT_DIM = len(VOCAB)
PHYSICAL_OUTPUT_IDS = [i for i in range(OUTPUT_DIM) if i != BOS_ID]
PHYSICAL_OUTPUT_DIM = len(PHYSICAL_OUTPUT_IDS)

# gate programming range (volts); clip keeps every gate in strong, well-resolved conduction
VG_CLIP_LO = float(os.environ.get("LM_VG_CLIP_LO", "0.4"))
VG_CLIP_HI = float(os.environ.get("LM_VG_CLIP_HI", "8.0"))


def manifest_matrix(manifest: EmbeddingManifest) -> np.ndarray:
    """The (vocab, dim) embedding matrix, checked against the active vocabulary."""
    M = np.asarray(manifest.matrix, dtype=float)
    if M.shape[0] != OUTPUT_DIM:
        raise ValueError(f"manifest vocab {M.shape[0]} != OUTPUT_DIM {OUTPUT_DIM}")
    if list(manifest.vocab) != list(VOCAB):
        raise ValueError("manifest vocab order does not match VOCAB")
    return M


def encode_contexts(ctx: np.ndarray, M: np.ndarray, vminus: float, vplus: float) -> np.ndarray:
    """ctx (N, CONTEXT_LEN) int ids -> (N, CONTEXT_LEN*dim) rail-scaled source voltages."""
    rows = M[ctx]                                   # (N, ctx_len, dim)
    flat = rows.reshape(ctx.shape[0], -1)
    return vminus + flat * (vplus - vminus)


def build_context_target_distributions(ctx: np.ndarray, y: np.ndarray) -> Dict[Tuple[int, ...], np.ndarray]:
    """Empirical next-token distribution for every distinct context window (the soft training target)."""
    counts: Dict[Tuple[int, ...], np.ndarray] = {}
    for c, t in zip(ctx, y):
        key = tuple(int(v) for v in c.tolist())
        if key not in counts:
            counts[key] = np.zeros(OUTPUT_DIM, dtype=float)
        counts[key][int(t)] += 1.0
    return {k: v / float(v.sum()) for k, v in counts.items()}


def unigram_distribution(y: np.ndarray) -> np.ndarray:
    """Fallback next-token distribution (corpus unigram) for unseen contexts."""
    c = np.zeros(OUTPUT_DIM, dtype=float)
    for t in y:
        c[int(t)] += 1.0
    return c / float(c.sum())
