"""Quantum-physics vocabulary for the CLLN language task.

This extends the base 32-token vocabulary. The 14 function words are kept identical;
the content words are ~210 genuine physics / quantum-mechanics terms grouped
into semantic classes. Real words are required because each token string is fed
verbatim to SciBERT to build its embedding, and the grammar relies on
class-clustered embeddings (particles near particles, properties near
properties, ...) to stay learnable at scale.

Design rule for separability / learnability: the grammar (see grammar.py)
is *class-regular* -- a token's grammatical role is a function of its class,
not of an idiosyncratic per-word list. So a token must belong to exactly one
content class, and the classes must be pairwise disjoint. The assertions at the
bottom enforce this.
"""
from __future__ import annotations

from typing import Dict, List

# --- fixed function words (identical to the 32-token grammar) -----------------
FUNCTION_WORDS: List[str] = [
    "<BOS>", ".", "?", "the", "a", "this", "that", "each", "every", "what", "why", "in", "of",
    "is", "has", "can", "measure", "shows",
]
# NOTE: this/that/each/every added as agreement-free determiners (7 sentence starters w/ the/a/an).
# The total vocabulary is trimmed to the released token set via LM_VOCAB_SIZE.

# --- content classes (genuine QM / physics terms) -----------------------------
ADJECTIVES: List[str] = [
    "pure", "mixed", "coherent", "incoherent", "classical", "quantum",
    "degenerate", "excited", "stationary", "normalized", "orthogonal",
    "thermal", "entangled", "localized",
]

PARTICLES: List[str] = [
    "electron", "photon", "atom", "qubit", "proton", "neutron", "muon",
    "boson", "fermion", "quark", "neutrino", "positron", "hadron", "lepton",
    "gluon", "pion", "kaon", "baryon", "meson", "nucleon", "nucleus", "ion",
    "molecule", "isotope", "deuteron", "antiproton", "exciton", "polariton",
    "magnon", "plasmon", "phonon", "graviton", "dipole", "soliton", "hydrogen",
    "helium", "positronium", "antineutrino", "photoelectron", "antimuon",
    "hole", "vacancy", "impurity", "defect",
]

PROPERTIES: List[str] = [
    "spin", "phase", "energy", "basis", "momentum", "polarization", "helicity",
    "parity", "charge", "mass", "frequency", "amplitude", "wavelength",
    "coherence", "entropy", "eigenvalue", "position", "velocity",
    "acceleration", "potential", "current", "voltage", "flux", "temperature",
    "pressure", "density", "symmetry", "chirality", "isospin", "magnetization",
    "conductance", "resistance", "capacitance", "wavenumber", "torque",
    "power", "spectrum", "lifetime", "bandwidth", "fidelity", "purity",
    "occupation", "population", "magnitude",
]

APPARATUS: List[str] = [
    "detector", "measurement", "interferometer", "polarizer", "spectrometer",
    "cavity", "laser", "oscilloscope", "sensor", "amplifier", "resonator",
    "transducer", "accelerator", "collider", "calorimeter", "magnetometer",
    "photodetector", "beamsplitter", "modulator", "attenuator", "waveguide",
    "antenna", "electrode", "capacitor", "transistor", "apparatus",
]

MEDIA: List[str] = [
    "wave", "field", "vacuum", "lattice", "ensemble", "manifold", "medium",
    "plasma", "condensate", "crystal", "conductor", "insulator",
    "semiconductor", "superconductor", "dielectric", "fluid", "gas", "beam",
    "pulse", "packet", "mode", "channel", "reservoir", "bath",
]

STATEFUL: List[str] = [
    "state", "system", "eigenstate", "wavefunction", "hamiltonian", "operator",
    "observable", "lagrangian", "propagator", "spinor", "eigenfunction",
    "wavepacket", "register", "qudit", "ket", "bra", "configuration",
    "orbital", "shell", "subsystem", "mixture", "distribution", "matrix",
    "tensor", "functional", "hilbert", "gauge", "ansatz",
]

OUTCOMES: List[str] = [
    "superposition", "entanglement", "decoherence", "tunneling", "interference",
    "diffraction", "collapse", "scattering", "emission", "absorption",
    "excitation", "relaxation", "resonance", "oscillation", "transition",
    "decay", "fluctuation", "correlation", "dispersion", "refraction",
    "reflection", "ionization", "recombination", "annihilation", "condensation",
    "thermalization",
]

# class name -> ordered word list
CONTENT_CLASSES: Dict[str, List[str]] = {
    "adjective": ADJECTIVES,
    "particle": PARTICLES,
    "property": PROPERTIES,
    "apparatus": APPARATUS,
    "medium": MEDIA,
    "stateful": STATEFUL,
    "outcome": OUTCOMES,
}

# Optional low-edge knob: LM_VOCAB_SIZE trims the TOTAL vocab to N tokens by dropping words
# from the END of each content class proportionally (function words always kept; class order and
# separability invariants preserved). Edges = (N-1) * ctx * dim: trades vocab for dim at fixed budget.
import os as _os_vt
_TARGET = int(_os_vt.environ.get("LM_VOCAB_SIZE", "0"))
if _TARGET:
    _content_target = _TARGET - len(FUNCTION_WORDS)
    _total = sum(len(v) for v in CONTENT_CLASSES.values())
    if not (7 <= _content_target <= _total):
        raise ValueError(f"LM_VOCAB_SIZE={_TARGET} out of range")
    _keep = {c: max(2, int(round(len(w) * _content_target / _total))) for c, w in CONTENT_CLASSES.items()}
    _drift = _content_target - sum(_keep.values())
    for _c in sorted(_keep, key=lambda c: -len(CONTENT_CLASSES[c])):
        if _drift == 0:
            break
        _step = 1 if _drift > 0 else -1
        if 2 <= _keep[_c] + _step <= len(CONTENT_CLASSES[_c]):
            _keep[_c] += _step; _drift -= _step
    CONTENT_CLASSES = {c: w[:_keep[c]] for c, w in CONTENT_CLASSES.items()}


# content words in a stable order (grouped by class)
CONTENT_WORDS: List[str] = [
    w for cls in ("particle", "property", "apparatus", "medium", "stateful", "outcome", "adjective")
    for w in CONTENT_CLASSES[cls]
]

# Grammar v2 ("sci", env LM_V2=1): adds the phonological indefinite article "an"
# (a/an agreement) as an extra function word. v1 vocab/ids are untouched when unset.
import os as _os_v2
V2 = _os_v2.environ.get("LM_V2", "") == "1"
if V2:
    FUNCTION_WORDS = FUNCTION_WORDS[:5] + ["an"] + FUNCTION_WORDS[5:]  # after "a"
    FUNCTION_WORDS = FUNCTION_WORDS + ["measures"]  # finite verb for the 8-token of-genitive family
    if _os_v2.environ.get("LM_NEG", "") == "1":
        FUNCTION_WORDS = FUNCTION_WORDS + ["no"]  # negation particle (FD19: lexicon-FALSE frames)

# Reduced-vocab mode (LM_DROP_TOKENS): remove tokens left unused by the active shape set so the
# network shrinks with the corpus (branches = vocab-1, edges = branches * ctx * dim). Applied AFTER the
# VOCAB_SIZE trim + V2 additions, so it removes exactly the named tokens; content classes that end up
# empty are dropped. Used to prune the adjective class + unused function words (why/can/measure) once
# the adjective-dependent shapes are dropped -> fewer edges at no cost to the surviving shapes.
_DROP_TOK = set(w for w in _os_v2.environ.get("LM_DROP_TOKENS", "").split(",") if w)
if _DROP_TOK:
    FUNCTION_WORDS = [w for w in FUNCTION_WORDS if w not in _DROP_TOK]
    # keep emptied classes present (as []) so class_words(name) still resolves for dropped-family
    # builders that reference the pool unconditionally; the empty pool just yields no records.
    CONTENT_CLASSES = {c: [w for w in ws if w not in _DROP_TOK] for c, ws in CONTENT_CLASSES.items()}
    CONTENT_WORDS = [w for w in CONTENT_WORDS if w not in _DROP_TOK]
    TOKEN_CLASS = {w: cls for cls, words in CONTENT_CLASSES.items() for w in words}

# full vocabulary: function words first, then content (grouped by class)
VOCAB: List[str] = list(FUNCTION_WORDS) + list(CONTENT_WORDS)

# token -> class (only content words; function words are not in this map)
TOKEN_CLASS: Dict[str, str] = {
    w: cls for cls, words in CONTENT_CLASSES.items() for w in words
}


def class_words(name: str) -> List[str]:
    return list(CONTENT_CLASSES[name])


# --- integrity checks (separability invariants) -------------------------------
def _validate() -> None:
    # every content word appears in exactly one class
    seen: Dict[str, str] = {}
    for cls, words in CONTENT_CLASSES.items():
        for w in words:
            if w in seen:
                raise ValueError(f"token {w!r} appears in both {seen[w]!r} and {cls!r}")
            seen[w] = cls
    # function words disjoint from content
    overlap = set(FUNCTION_WORDS) & set(CONTENT_WORDS)
    if overlap:
        raise ValueError(f"function/content overlap: {sorted(overlap)}")
    # whole vocab unique
    if len(VOCAB) != len(set(VOCAB)):
        from collections import Counter
        dups = [w for w, c in Counter(VOCAB).items() if c > 1]
        raise ValueError(f"duplicate tokens in VOCAB: {dups}")


_validate()


if __name__ == "__main__":
    print(f"total vocab        : {len(VOCAB)}")
    print(f"function words     : {len(FUNCTION_WORDS)}")
    print(f"content words      : {len(CONTENT_WORDS)}")
    for cls, words in CONTENT_CLASSES.items():
        print(f"  {cls:10s}: {len(words)}")
    # edge count at a few embedding dims (BOS has no output branch)
    branches = len(VOCAB) - 1
    for d in (16, 18, 20):
        print(f"edges @ {d}D ctx6 : {branches * 6 * d}  (branches={branches})")
