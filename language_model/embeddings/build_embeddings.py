#!/usr/bin/env python3
"""Build the SciBERT-FA embedding matrix the language model uses to turn tokens into gate-input voltages.

For each vocabulary token: bare-token SciBERT (allenai/scibert_scivocab_uncased) mean-pooled + L2-normalized
-> sklearn FactorAnalysis reduction to `--dim` dimensions -> per-dimension min/max scaled to [0,1]; <BOS> is
the zero vector. Reads the shipped vocab.txt and writes scibert_fa<dim>.json (the file the trainer loads).
This is a reproducibility script; the trained model ships with scibert_fa16.json already built.

  python build_embeddings.py --dim 16
"""
import argparse, json, os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOCAB_TXT = HERE.parent / "vocab.txt"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"


def _load_vocab():
    return [t for t in VOCAB_TXT.read_text().splitlines() if t != ""]


def _scibert_raw_embeddings(tokens):
    """(N, 768) bare-token mean-pooled, L2-normalized SciBERT embeddings; <BOS> -> zero vector."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SCIBERT_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(SCIBERT_MODEL, local_files_only=True)
    model.eval()
    out = np.zeros((len(tokens), model.config.hidden_size), dtype=float)
    with torch.no_grad():
        for i, word in enumerate(tokens):
            if word == "<BOS>":
                continue
            enc = tok(word, return_tensors="pt")
            hs = model(**enc).last_hidden_state[0]
            ids = enc["input_ids"][0].tolist()
            special = set(tok.all_special_ids)
            keep = torch.tensor([j for j, t in enumerate(ids) if t not in special]) or torch.arange(hs.shape[0])
            vec = hs[keep].mean(dim=0)
            n = float(vec.norm())
            out[i] = (vec / n).numpy() if n > 0 else vec.numpy()
    return out


def build_manifest(dim):
    from sklearn.decomposition import FactorAnalysis
    from sklearn.preprocessing import StandardScaler

    vocab = _load_vocab()
    bos = vocab.index("<BOS>")
    raw = _scibert_raw_embeddings(vocab)
    nonbos = [i for i in range(len(vocab)) if i != bos]
    feats = StandardScaler().fit_transform(raw[nonbos])
    reduced = FactorAnalysis(n_components=dim, random_state=0).fit_transform(feats)
    lo = reduced.min(0, keepdims=True); hi = reduced.max(0, keepdims=True)
    span = np.where((hi - lo) <= 1e-12, 1.0, hi - lo)
    scaled = (reduced - lo) / span
    matrix = np.zeros((len(vocab), dim))
    for row, vi in zip(scaled, nonbos):
        matrix[vi] = row
    matrix[bos] = 0.0
    return {
        "name": f"scibert_fa{dim}_qm{len(vocab)}",
        "kind": "baseline",
        "vocab": vocab,
        "matrix": [[float(v) for v in r] for r in matrix.tolist()],
        "dim": int(dim),
        "value_range": [0.0, 1.0],
        "derivation": (f"SciBERT {SCIBERT_MODEL} bare-token mean-pooled + L2-normalized, reduced via sklearn "
                       f"FactorAnalysis to {dim}D, dim-wise min/max scaled to [0,1]; <BOS> forced to zero."),
        "feature_basis": "scibert_scivocab_uncased",
        "reducer": "factor_analysis",
        "dimension_names": [f"fa{i + 1}" for i in range(dim)],
        "metadata": {"vocab_size": len(vocab), "knowledge_source": "external_pretrained_reduction"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    man = build_manifest(a.dim)
    out = a.out or HERE / f"scibert_fa{a.dim}.json"
    out.write_text(json.dumps(man, indent=2))
    mat = np.asarray(man["matrix"])
    print(f"wrote {out}  matrix {mat.shape} range [{mat.min():.3f},{mat.max():.3f}]")


if __name__ == "__main__":
    main()
