#!/usr/bin/env python3
"""Run a trained coupled-learning language model on real ngspice and measure it.

Loads the endpoint gates of a run (`runs/<run>/gates.npz`), builds the NMOS network, and solves
every free phase as an ngspice DC operating point. Three modes:

  # valid-string rate (the headline metric): generate N sentences from seeded openers and score
  # each for grammaticality on-circuit
  python infer_language_model.py --run clean
  python infer_language_model.py --run noisy_chip1 --chip chips/chip_1.npz

  # top next-token distribution for a context
  python infer_language_model.py --run clean --context "the electron has spin in the"

  # free-form generation
  python infer_language_model.py --run clean --generate 20

Openers seed a determiner + subject + verb for one of the grammatical sentence shapes; the network
then fills the remainder token by token. Validity is judged by the same grammar used to build the
corpus. Run from inside `language_model/`.
"""
import argparse
import collections
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import train_language_model as NG          # ngspice solver + network machinery
from clm import helpers as T
from clm import evaluate as EV
from clm.embeddings import load_embedding_manifest
from clm.grammar import (
    CONTEXT_LEN, GRAMMAR_VERSIONS, START_TOKENS, TERMINAL_PUNCT, SentenceRecord,
    as_train_test_bundle, records_from_maybe_bundle, split_sentence_pools,
    write_train_test_pool_files, _cls, _det_indef,
)
from clm.vocab import VOCAB

SEED = 1                       # corpus split seed (matches the shipped champion)
READ_NOISE_SIGMA = 2.25e-3     # V, read noise applied to measured free voltages under a device chip


# ---- grammatical sentence shapes: (name, kind, spec) --------------------------------------------
# kind "decl": spec = (subject_class | tuple, verb_tokens) -> seed [det, subject] + verb; the network
#              fills the object. kind "q": spec = fixed wh-prefix -> the network fills the body + "?".
_DECL = [
    ("measure-of", "decl", ("apparatus", ["measures"])),
    ("has-in",     "decl", ("particle",  ["has"])),
    ("shows-in",   "decl", ("stateful",  ["shows"])),
]
_EXTRA = [
    ("is-adj",      "decl", (("particle", "stateful", "medium"), ["is"])),
    ("can-measure", "decl", ("apparatus", ["can", "measure"])),
    ("q-what",      "q",    ["what", "is", "the"]),
    ("q-why",       "q",    ["why", "is", "the"]),
]


def build_shapes(all_shapes):
    shapes = _DECL + (_EXTRA if all_shapes else [])
    vset = set(VOCAB)
    # drop shapes whose fixed tokens are absent from the (reduced) vocabulary
    shapes = [s for s in shapes if all(t in vset for t in (list(s[2]) if s[1] == "q" else list(s[2][1])))]
    cls_needed = set()
    for _nm, _k, _sp in shapes:
        if _k == "decl":
            cl = _sp[0]
            for c in ((cl,) if isinstance(cl, str) else cl):
                cls_needed.add(c)
    words = {c: [w for w in VOCAB if _cls(w) == c] for c in cls_needed}
    return shapes, words


def shape_of(text):
    w = text.split()
    if w and w[0] == "what": return "q-what"
    if w and w[0] == "why": return "q-why"
    if "measures" in w and "of" in w: return "measure-of"
    if "can" in w and "measure" in w: return "can-measure"
    if "has" in w: return "has-in"
    if "is" in w and "in" not in w: return "is-adj"
    if "shows" in w: return "shows-in"
    if "is" in w and "in" in w: return "is-in"
    return "?"


def load_pools():
    """Reconstruct the champion train / held-out test pools deterministically from the grammar."""
    bundle = as_train_test_bundle(
        split_sentence_pools(grammar_version=GRAMMAR_VERSIONS[0], seed=SEED, dev_frac=0.15, test_frac=0.15))
    with tempfile.TemporaryDirectory() as td:
        write_train_test_pool_files(Path(td), bundle)
        train_pool = records_from_maybe_bundle(bundle_path=Path(td) / "sentence_pools.json", key="train_pool")
        test_pool = records_from_maybe_bundle(bundle_path=Path(td) / "sentence_pools.json", key="test_pool")
    return train_pool, test_pool


class LM:
    def __init__(self, run, chip=None, workers=None, det_mode="mixed", all_shapes=False):
        self.man = load_embedding_manifest(_HERE / "embeddings" / "scibert_fa16.json")
        self.M = T.manifest_matrix(self.man)
        self.dim = int(self.man.dim)
        self.F = CONTEXT_LEN * self.dim
        self.NB, self.OUT, self.BOS = T.PHYSICAL_OUTPUT_DIM, T.OUTPUT_DIM, T.BOS_ID
        self.det_mode = det_mode
        self.more_dets = os.environ.get("LM_MORE_DETS") == "1"
        gates = np.load(_HERE / "runs" / run / "gates.npz")["vg_final"].reshape(self.NB, self.F).astype(np.float64)
        self.shapes, self.words = build_shapes(all_shapes)

        w = workers or min(16, (os.cpu_count() or 4))
        self.solver = NG.NgspiceSolver(self.F, self.NB, w)
        self.solver.set_gates(gates)
        self.meas_sigma = 0.0
        if chip is not None:
            z = np.load(chip if Path(chip).is_absolute() else _HERE / chip)
            df = z["vto_free"].reshape(self.NB, self.F).astype(np.float64)
            dc = z["vto_clamp"].reshape(self.NB, self.F).astype(np.float64)
            bf = z["beta_free"].reshape(self.NB, self.F).astype(np.float64) if "beta_free" in z.files else np.ones((self.NB, self.F))
            bc = z["beta_clamp"].reshape(self.NB, self.F).astype(np.float64) if "beta_clamp" in z.files else np.ones((self.NB, self.F))
            self.solver.set_mismatch(df, dc, bf, bc)
            self.meas_sigma = READ_NOISE_SIGMA
        self._mrng = np.random.default_rng(0)

    def close(self):
        self.solver.close()

    def _probs(self, ctx_rows, temp):
        Xs = T.encode_contexts(np.asarray(ctx_rows, dtype=int), self.M, 0.0, 0.45).astype(np.float64)
        Vf = self.solver.free(Xs)
        if self.meas_sigma > 0:
            Vf = Vf + self._mrng.normal(0.0, self.meas_sigma, Vf.shape)
        Vfull = np.zeros((Xs.shape[0], self.OUT))
        Vfull[:, T.PHYSICAL_OUTPUT_IDS] = Vf
        Vfull[:, self.BOS] = -1e30
        z = Vfull / float(temp); z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return p

    def _pick_subj(self, cl, rng):
        pool = self.words[cl] if isinstance(cl, str) else [w for c in cl for w in self.words[c]]
        return pool[int(rng.integers(len(pool)))]

    def _seed_det(self, subj, rng):
        if self.det_mode == "the": return "the"
        if self.det_mode == "indef": return _det_indef(subj)
        opts = ["the", _det_indef(subj)]
        if self.more_dets:
            opts += ["this", "that", "each", "every"]
        return opts[int(rng.integers(len(opts)))]

    def generate(self, n, temp, seed, max_steps=12):
        grng = np.random.default_rng(seed)
        order = [self.shapes[i % len(self.shapes)] for i in range(n)]
        seeds = []
        for _nm, _k, _sp in order:
            if _k == "q":
                seeds.append(list(_sp))
            else:
                cl, vb = _sp
                subj = self._pick_subj(cl, grng)
                seeds.append([self._seed_det(subj, grng), subj] + vb)
        ctx = np.full((n, CONTEXT_LEN), self.BOS, dtype=int)
        outs = [[] for _ in range(n)]; done = np.zeros(n, bool)
        for i, sd in enumerate(seeds):
            for tok in sd:
                outs[i].append(tok); ctx[i] = np.append(ctx[i][1:], T.TOKEN_TO_ID[tok])
        for _ in range(max_steps):
            act = np.flatnonzero(~done)
            if act.size == 0: break
            p = self._probs(ctx[act], temp)
            for r, gi in enumerate(act):
                y = int(grng.choice(self.OUT, p=p[r] / p[r].sum())); tok = VOCAB[y]
                outs[gi].append(tok); ctx[gi] = np.append(ctx[gi][1:], y)
                if tok in TERMINAL_PUNCT: done[gi] = True
        return outs, [nm for nm, _, _ in order]

    def context_probs(self, text, temp):
        ids = [T.TOKEN_TO_ID[t] for t in text.lower().split() if t in T.TOKEN_TO_ID]
        ids = ids[-CONTEXT_LEN:]
        row = [self.BOS] * (CONTEXT_LEN - len(ids)) + ids
        return self._probs([row], temp)[0]


def report_validity(m, temp, n, reruns, train_texts, ref_texts, gv):
    vrates = []
    saved = None
    for r in range(reruns):
        outs, order = m.generate(n, temp, 1000 + r)
        rep = EV.evaluate_generated_sentences(
            outs, grammar_version=gv, train_pool_texts=train_texts,
            reference_pool_texts=ref_texts, reference_pool_name="test",
            start_tokens=[o[0] for o in outs])
        vrates.append(rep["valid_rate"])
        if r == 0:
            cases = rep.get("cases", [])
            valid = [c.get("generated_text", "") for c in cases if str(c.get("label", "")).startswith("valid")]
            per = {nm: [0, 0] for nm, _, _ in m.shapes}
            for i, c in enumerate(cases):
                nm = order[i]; per[nm][1] += 1
                if str(c.get("label", "")).startswith("valid"): per[nm][0] += 1
            cnt = collections.Counter(shape_of(t) for t in valid); tot = max(len(valid), 1)
            saved = (valid, per, cnt, tot)
    va = np.array(vrates)
    print(f"valid-string rate (T={temp}, {n} sentences x {reruns} reruns): "
          f"{va.mean():.4f} +- {va.std():.4f}   runs={[f'{v:.3f}' for v in vrates]}")
    valid, per, cnt, tot = saved
    print("  per-shape valid:", {nm: f"{v[0]/max(v[1],1):.2f}" for nm, v in per.items()})
    print("  shape mix:", {k: f"{v/tot*100:.0f}%" for k, v in cnt.most_common()})
    random.seed(3)
    for pp in random.sample(valid, min(6, len(valid))):
        print("    [valid]", pp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="clean", help="runs/<run>/gates.npz to load (clean, noisy_chip1, ...)")
    ap.add_argument("--chip", default=None, help="device-mismatch fingerprint npz (e.g. chips/chip_1.npz)")
    ap.add_argument("--temp", type=float, default=0.0006, help="readout softmax temperature")
    ap.add_argument("--n", type=int, default=1000, help="sentences per validity run")
    ap.add_argument("--reruns", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--det", default="mixed", choices=["mixed", "the", "indef"], help="opener determiner policy")
    ap.add_argument("--all-shapes", action="store_true", help="include is-adjective + question shapes")
    ap.add_argument("--context", default=None, help="print the top next-token distribution for a context")
    ap.add_argument("--generate", type=int, default=0, help="print N freely generated sentences")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    m = LM(a.run, a.chip, a.workers, det_mode=a.det, all_shapes=a.all_shapes)
    try:
        if a.context is not None:
            p = m.context_probs(a.context, a.temp)
            order = np.argsort(p)[::-1]
            print(f"context: {a.context}\ntop next tokens (T={a.temp}):")
            for j in order[:10]:
                print(f"  {VOCAB[j]:16s} {p[j]:.4f}")
            return
        if a.generate:
            outs, _ = m.generate(a.generate, a.temp, a.seed)
            for o in outs:
                print(" ".join(o))
            return
        train_pool, test_pool = load_pools()
        gv = train_pool[0].grammar_version
        report_validity(m, a.temp, a.n, a.reruns,
                        EV._pool_text_set(train_pool), EV._pool_text_set(test_pool), gv)
    finally:
        m.close()


if __name__ == "__main__":
    main()
