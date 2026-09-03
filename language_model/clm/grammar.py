"""Grammar and corpus generation for the reduced-vocab physics language model.

Enumerates the grammatical sentences, splits train / held-out pools, samples the
training corpus, and windows sentences into fixed-length contexts. Public API:
``enumerate_sentence_records``, ``valid_sentence_texts``, ``is_valid_sentence``,
``split_sentence_pools``, ``sample_training_sentences``,
``build_windows_from_sentences``, ``SentenceRecord``, ``START_TOKENS`` ...

The grammar is the same 8 ``grammar_v1_core`` templates as the 32-token task
(so context length stays 6 and no generation-rate is lost), but the rule
relations are *class-regular*: a token's grammatical role is determined by its
semantic class, not an idiosyncratic per-word list. This keeps the task
separable and learnable at scale -- the model only has to route by class, which
SciBERT's class-clustered embeddings make tractable -- while the large content
classes give a big novel-sentence space (high uniqueness / heldout).

A per-family ``fanout`` knob optionally bounds how many class members each
subject relates to (deterministic, seeded) to keep support sizes moderate; the
default (0) means full class-regular.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

# allow running this module directly for a grammar self-check
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clm.vocab import (
    VOCAB,
    TOKEN_CLASS,
    class_words,
)

from clm.vocab import V2 as _V2
import os as _os_ctx
CONTEXT_LEN = int(_os_ctx.environ.get("LM_CTX", "7" if _V2 else "6"))  # v2-sci default 7; LM_CTX overrides (FD18: 6)
START_TOKENS = ["the", "a", "what", "why"]
TERMINAL_PUNCT = {".", "?"}
DETERMINERS = ("the", "a")

TOKEN_TO_ID: Dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {idx: tok for tok, idx in TOKEN_TO_ID.items()}

GRAMMAR_VERSIONS = ("grammar_v2_sci",) if _V2 else ("grammar_v1_core",)
CURRICULUM_STAGES = ("core",)
_DEFAULT_START_TOKENS = ["the", "a", "what", "why"]

# Optional simplification knob: set env LM_FAMILIES to a comma list to restrict
# which sentence families are generated (e.g. "stmt" = declarative only, dropping the
# what/why question families). Default: all families.
import os as _os
_FAMILY_FILTER = {f.strip() for f in _os.environ.get("LM_FAMILIES", "").split(",") if f.strip()}


# Optional knob: drop the determiner before OBJECTS so object-class routing becomes a
# clean 1-token-back decision (last token = the verb) instead of 2-back. Set
# LM_NO_OBJ_DET=1 to enable. Subjects keep their determiner.
_OBJ_DET = _os.environ.get("LM_NO_OBJ_DET", "") == ""


def _objslot(det2, obj) -> List[str]:
    return [det2, obj] if (det2 is not None) else [obj]


def _family_enabled(name: str) -> bool:
    if not _FAMILY_FILTER:
        return True
    if "stmt" in _FAMILY_FILTER and name.startswith("stmt_"):
        return True
    if "q" in _FAMILY_FILTER and name.startswith("q_"):
        return True
    return name in _FAMILY_FILTER


def _derive_start_tokens() -> List[str]:
    if _V2:
        return ["the", "a", "an"]   # v2-sci: declaratives only; 'an' sentences exist
    starts: List[str] = []
    if any(_family_enabled(f) for f in
           ("stmt_is_adj", "stmt_has", "stmt_can_measure", "stmt_shows", "stmt_is_in")):
        starts += ["the", "a"]
    if _family_enabled("q_what_simple") or _family_enabled("q_what_of"):
        starts.append("what")
    if _family_enabled("q_why"):
        starts.append("why")
    return starts or ["the", "a", "what", "why"]


START_TOKENS = _derive_start_tokens()

# -----------------------------------------------------------------------------
# Class-regular relation configuration.
#   subjects  : which classes can be the subject of the family
#   targets   : which classes can fill the object/location/tail slot
#   fanout    : 0 -> every subject relates to ALL target-class members;
#               k>0 -> each subject relates to a deterministic size-k subset
#                      (keeps support sizes moderate without losing vocab size)
# -----------------------------------------------------------------------------
RELATION_FANOUT_SEED = 20260620

# Design for learnability ("easily solved"): SUBJECTS are universal -- any noun may
# take any verb -- so the high-precision "which verb after this noun" decision does
# NOT depend on fuzzy noun-class discrimination in embedding space. The remaining
# structure is OBJECT class, which is determined by the VERB (a clean, well-separated
# function-word context): has/can-measure -> property, shows -> outcome, is-in ->
# location, is -> adjective. This keeps a genuine, semantically-flavoured QM grammar
# (entities HAVE/MEASURE properties, SHOW outcomes, are IN media/outcomes) with a huge
# novel-sentence space, while routing off clean function-word contexts.
RELATIONS: Dict[str, Dict[str, object]] = {
    "has":         {"subjects": ["__all_nouns__"], "targets": ["property"],            "fanout": 0},
    "can_measure": {"subjects": ["__all_nouns__"], "targets": ["property"],            "fanout": 0},
    "shows":       {"subjects": ["__all_nouns__"], "targets": ["outcome"],             "fanout": 0},
    "is_in":       {"subjects": ["__all_nouns__"], "targets": ["medium", "outcome"],   "fanout": 0},
    "is_adj":      {"subjects": ["__all_nouns__"], "targets": ["adjective"],           "fanout": 0},
    "why":         {"subjects": ["__all_nouns__"], "targets": ["adjective"],           "fanout": 0},
    "of":          {"subjects": ["property"],      "targets": ["particle", "stateful"], "fanout": 0},
    "what_simple": {"subjects": ["__all_nouns__"], "targets": [],                      "fanout": 0},
}

_NOUN_CLASSES = ("particle", "property", "apparatus", "medium", "stateful", "outcome")


def _members(classes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in classes:
        if c == "__all_nouns__":
            for nc in _NOUN_CLASSES:
                out.extend(class_words(nc))
        else:
            out.extend(class_words(c))
    # stable de-dup
    seen = set()
    uniq = []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def _targets_for(subject: str, spec: Mapping[str, object]) -> List[str]:
    targets = _members(list(spec["targets"]))  # type: ignore[arg-type]
    fanout = int(spec.get("fanout", 0))  # type: ignore[arg-type]
    if fanout <= 0 or fanout >= len(targets):
        return targets
    # deterministic per-subject subset
    rng = random.Random(f"{RELATION_FANOUT_SEED}:{subject}")
    return sorted(rng.sample(targets, fanout))


@lru_cache(maxsize=None)
def _relation(name: str) -> Dict[str, List[str]]:
    """subject -> ordered list of valid targets for one family."""
    spec = RELATIONS[name]
    rel: Dict[str, List[str]] = {}
    for subj in _members(list(spec["subjects"])):  # type: ignore[arg-type]
        rel[subj] = _targets_for(subj, spec)
    return rel


# -----------------------------------------------------------------------------
# SentenceRecord
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SentenceRecord:
    tokens: Tuple[str, ...]
    family: str
    grammar_version: str
    complexity: int

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def start_token(self) -> str:
        return self.tokens[0]

    @property
    def length(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> Dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "family": self.family,
            "grammar_version": self.grammar_version,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SentenceRecord":
        tokens = tuple(str(tok) for tok in payload["tokens"])
        return cls(
            tokens=tokens,
            family=str(payload["family"]),
            grammar_version=str(payload["grammar_version"]),
            complexity=int(payload.get("complexity", 1)),
        )


def sentence_text(tokens: Sequence[str]) -> str:
    return " ".join(str(tok) for tok in tokens)


def _rec(tokens: Sequence[str], family: str, version: str, complexity: int) -> SentenceRecord:
    return SentenceRecord(tokens=tuple(tokens), family=family, grammar_version=version, complexity=complexity)


def _unique_records(records: Iterable[SentenceRecord]) -> Tuple[SentenceRecord, ...]:
    dedup: Dict[str, SentenceRecord] = {}
    for record in records:
        dedup.setdefault(record.text, record)
    return tuple(sorted(dedup.values(), key=lambda r: (r.family, r.tokens)))


def _counter_dict(records: Sequence[SentenceRecord]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.tokens)
    return {tok: int(counter.get(tok, 0)) for tok in VOCAB}


def _group_key(record: SentenceRecord) -> Tuple[str, str, int]:
    return record.family, record.start_token, record.length


# -----------------------------------------------------------------------------
# Grammar v2 "sci": scientifically-typed relations, a/an agreement, 7-10 token
# sentences with adjective modifiers, in-PPs and of-genitives. CONTEXT_LEN = 8.
# Subjects are CLASS-RESTRICTED (the model must use class information):
#   has:          particle|stateful|medium|apparatus -> property   (mass-noun object, no det)
#   can measure:  apparatus ONLY                     -> property
#   shows:        particle|stateful|medium           -> outcome
#   is  <adj>:    particle|stateful|medium           -> adjective
#   is in:        particle|stateful                  -> medium|apparatus
# Properties/outcomes never head a declarative (fixes "the power has current").
# -----------------------------------------------------------------------------
_V2_AN_EXCEPTIONS = {"universe", "university", "uniform", "unitary", "union"}  # 'yu'-onset
# LM_SIMPLE=1 -> v4 "sci-simple": keep only the 5 structurally-simple families
# (has/shows/is-adj @5 tok, can-measure @6, measures-of @8). The corpus stays class-typed
# scientifically correct; the POS validator (metric) is unchanged and still accepts everything
# grammatical. Drops the positional-aliasing families (adj-modifier, PP-subject, is_in).
_V2_SIMPLE_MODE = _os.environ.get("LM_SIMPLE", "")
_V2_SIMPLE = _V2_SIMPLE_MODE in ("1", "2")
_V2_MIDDLE = _V2_SIMPLE_MODE == "3"  # middle-hard: simple families + adj-modified + locative (no PP subjects)
_V2_DIVERSE = _V2_SIMPLE_MODE == "4"  # shape-diverse: all 4 visually-distinct shapes, NO adj-modifier aliasing
_V2_DIVERSE_FAMILIES = {"s2_has", "s2_shows", "s2_is_adj", "s2_measures_short", "s2_can_measure",
                        "s2_measure_of", "s2_is_in", "s2_has_pp", "s2_shows_pp"}
_V2_MIDDLE_FAMILIES = {"s2_has", "s2_shows", "s2_is_adj", "s2_measures_short", "s2_can_measure",
                       "s2_measure_of", "s2_has_adj", "s2_shows_adj", "s2_is_in"}
_V2_SIMPLE_FAMILIES = ({"s2_has", "s2_shows", "s2_is_adj", "s2_can_measure", "s2_measures_short"}
                       if _V2_SIMPLE_MODE == "2" else
                       {"s2_has", "s2_shows", "s2_is_adj", "s2_can_measure", "s2_measure_of"})
# LM_SIMPLE_ISIN=1: add the locative family (6th visible shape) to simple mode
if _V2_SIMPLE and _os.environ.get("LM_SIMPLE_ISIN", "") == "1":
    _V2_SIMPLE_FAMILIES = _V2_SIMPLE_FAMILIES | {"s2_is_in"}
# LM_SIMPLE_PP=1: add the PP-subject families (7th/8th shapes) to simple mode
if _V2_SIMPLE and _os.environ.get("LM_SIMPLE_PP", "") == "1":
    _V2_SIMPLE_FAMILIES = _V2_SIMPLE_FAMILIES | {"s2_has_pp", "s2_shows_pp"}
# LM_LEX_FAMILIES_ONLY=1: keep only lexicon-bearing families (no class-typed retreat frames)
if _V2_SIMPLE and _os.environ.get("LM_LEX_FAMILIES_ONLY", "") == "1":
    _V2_SIMPLE_FAMILIES = _V2_SIMPLE_FAMILIES & {"s2_has", "s2_shows", "s2_measure_of", "s2_has_pp", "s2_shows_pp"}
# LM_FAMSET: explicit comma-separated family set (overrides all above), e.g.
# "s2_has,s2_shows,s2_is_adj,s2_is_in" for a particle-subject-only set (similar margins -> no single
# family dominates cold-read generation via the amplification law -> diverse output).
if _V2_SIMPLE and _os.environ.get("LM_FAMSET", ""):
    _V2_SIMPLE_FAMILIES = set(_os.environ["LM_FAMSET"].split(","))
if _V2_SIMPLE and _os.environ.get("LM_NEG", "") == "1":
    _V2_SIMPLE_FAMILIES = _V2_SIMPLE_FAMILIES | {"s2_has_no", "s2_shows_no"}


def _det_indef(word: str) -> str:
    w = str(word).lower()
    if w in _V2_AN_EXCEPTIONS:
        return "a"
    return "an" if w[:1] in "aeiou" else "a"


def _v2_dets(word: str) -> Tuple[str, ...]:
    return ("the", _det_indef(word))

# LM_MORE_DETS=1: expand the SUBJECT determiner to 7 agreement-safe openers — the/a/an (with agreement)
# + demonstratives this/that + universal quantifiers each/every ("every electron has charge"). Objects stay
# "the"-only (OF_THE/LOC_THE), so no autoregressive object-agreement burden. Gives 7 sentence starts.
_MORE_DETS = _os.environ.get("LM_MORE_DETS") == "1"
_EXTRA_DETS = ("this", "that", "each", "every")
def _subj_dets(word: str) -> Tuple[str, ...]:
    return _v2_dets(word) + _EXTRA_DETS if _MORE_DETS else _v2_dets(word)


_V2_SUBJ = {
    "has": ("particle", "stateful", "medium", "apparatus"),
    "measure": ("apparatus",),
    "shows": ("particle", "stateful", "medium"),
    "is_adj": ("particle", "stateful", "medium"),
    "is_in": ("particle", "stateful"),
}
# LM_DISJOINT_SUBJ=1: give each structure a DISJOINT subject class so the subject unambiguously
# determines the shape (like apparatus->measure-of, which hits 1.00). Kills cross-family bleed that caps
# has/shows/is-in at ~0.86-0.93 -> lets forced-SUBJECT generation reach measure-of-level validity on all shapes.
if _os.environ.get("LM_DISJOINT_SUBJ") == "1":
    _V2_SUBJ = {
        "has": ("particle",),      # particle  -> has [prop] in [medium]   (has_loc)
        "measure": ("apparatus",), # apparatus -> measures [prop] of [particle]
        "shows": ("stateful",),    # stateful  -> shows [outcome] in [medium]  (shows_loc)
        "is_adj": ("medium",),
        # is_in uses PARTICLE subjects too (not media): "the electron is in the lattice ." — media then
        # only ever appear as trailing LOCATIONS (never subjects), killing the double-duty collapse.
        # particle does both has_loc and is_in, disambiguated by the forced verb seed ("has" vs "is in").
        "is_in": ("particle",),
    }
_V2_FAN = {"pp_medium": 6, "of_tail": 8, "adj_mod": 4, "obj": 12}
if _V2_SIMPLE:
    _V2_FAN.update({"obj": 20, "of_tail": 12, "adj_mod": 6})  # fewer families -> denser pairings
    _V2_FAN["of_tail"] = int(_os.environ.get("LM_FAN_OF", _V2_FAN["of_tail"]))  # family-balance knob
    _V2_FAN["pp_medium"] = int(_os.environ.get("LM_FAN_PP", _V2_FAN["pp_medium"]))  # is_in size knob
    # adjectives are 8 brand-new output classes seen only in is_adj + q_why; raise adj_mod to pair
    # every subject with ALL adjectives -> more unique adjective examples so their output branches
    # can compete with the dominant noun prior (LM_FAN_ADJ=99 for full pairing).
    _V2_FAN["adj_mod"] = int(_os.environ.get("LM_FAN_ADJ", _V2_FAN["adj_mod"]))
if _V2_SIMPLE_MODE == "2":
    _V2_FAN.update({"obj": 999, "adj_mod": 999})  # v5 short-only: FULL class pairings (max coverage)


def _v2_pick(kind: str, key: str, pool: Sequence[str]) -> List[str]:
    k = _V2_FAN[kind]
    if k >= len(pool):
        return list(pool)
    rng = random.Random(f"{RELATION_FANOUT_SEED}:v2:{kind}:{key}")
    return sorted(rng.sample(list(pool), k))



# LM_PHYS=1 -> physics-fact corpus filter: 37 pairings confirmed as textbook violations by a
# two-pass multi-agent physics audit (2026-07-07; e.g. neutral particles x charge, spin-0 x
# polarization, mixed x ket, classical x qubit, photon x excitation), plus manual style
# exclusions ('measurement' as measuring subject / locative). Filters TRAINING corpus only;
# the POS-grammar validity metric is unaffected.
_V2_PHYS = _os.environ.get("LM_PHYS", "") == "1"
_V2_PHYS_EXCLUDE = {
 "App_meas_P": {("polarizer","charge"),("polarizer","acceleration"),("polarizer","velocity"),
   ("polarizer","position"),("polarizer","entropy"),("resonator","basis"),("sensor","basis"),
   ("spectrometer","basis"),("transducer","basis")},
 "S_has_P": {("eigenfunction","temperature"),("eigenfunction","pressure"),("eigenfunction","voltage"),
   ("eigenfunction","acceleration"),("eigenstate","voltage"),("kaon","polarization"),
   ("manifold","voltage"),("neutrino","charge"),("photon","charge"),("pion","polarization"),
   ("pion","helicity"),("kaon","helicity"),("pion","spin"),("kaon","spin"),
   ("spinor","pressure"),("spinor","voltage")},
 "Adj_S": {("classical","photon"),("classical","qubit"),("classical","ket"),("excited","photon"),
   ("mixed","ket"),("mixed","bra"),("mixed","wavefunction")},
 "S_is_Adj": {("ket","mixed"),("eigenstate","mixed"),("observable","pure"),("observable","mixed"),
   ("operator","excited")},
 "P_of_T": {("polarization","pion"),("polarization","kaon"),("helicity","pion"),("helicity","kaon"),("spin","pion"),("spin","kaon")},
 "S_shows_O": {("photon","excitation"),("photon","relaxation")},
}
_V2_PHYS_EXCLUDE_STRICT = {
 "Adj_S": {("classical","ket"),("classical","photon"),("classical","qubit"),("excited","photon"),("mixed","bra"),("mixed","ket"),("mixed","wavefunction")},
 "App_meas_P": {("polarizer","acceleration"),("polarizer","charge"),("polarizer","entropy"),("polarizer","position"),("polarizer","potential"),("polarizer","pressure"),("polarizer","temperature"),("polarizer","velocity"),("polarizer","voltage"),("resonator","basis"),("sensor","basis"),("spectrometer","basis"),("transducer","basis")},
 "P_of_T": {("acceleration","gluon"),("acceleration","photon"),("mass","gluon"),("mass","photon"),("polarization","kaon"),("polarization","pion")},
 "S_has_P": {("eigenfunction","acceleration"),("eigenfunction","charge"),("eigenfunction","mass"),("eigenfunction","pressure"),("eigenfunction","temperature"),("eigenfunction","voltage"),("eigenstate","voltage"),("gluon","mass"),("kaon","polarization"),("manifold","voltage"),("neutrino","charge"),("neutron","charge"),("photon","charge"),("photon","mass"),("pion","polarization"),("polarizer","acceleration"),("polarizer","charge"),("polarizer","entropy"),("polarizer","potential"),("polarizer","voltage"),("resonator","basis"),("spinor","pressure"),("spinor","voltage"),("transducer","basis")},
 "S_in_L": {("bra","amplifier"),("bra","collider"),("bra","condensate"),("bra","interferometer"),("bra","plasma"),("bra","semiconductor")},
 "S_is_Adj": {("eigenstate","mixed"),("ket","mixed"),("observable","mixed"),("observable","pure"),("operator","excited")},
 "S_shows_O": {("bra","absorption"),("bra","emission"),("bra","scattering"),("eigenfunction","absorption"),("photon","excitation"),("photon","relaxation")},
}
if _V2_SIMPLE_MODE == "4":
    # DIVERSE: rebalance shapes — cap PP/of combinatorics, boost simple-family pairings
    _V2_FAN.update({"pp_medium": 2, "of_tail": 3, "obj": 24})
    _V2_FAN["pp_medium"] = int(_os.environ.get("LM_FAN_PP", _V2_FAN["pp_medium"]))
    _V2_FAN["of_tail"] = int(_os.environ.get("LM_FAN_OF", _V2_FAN["of_tail"]))
    _V2_FAN["obj"] = int(_os.environ.get("LM_FAN_OBJ", _V2_FAN["obj"]))
# LM_FAN_PPOBJ: separate object fanout for the PP-subject families. Only defined when
# the env is set — otherwise the call sites keep kind "obj" (same RNG stream), so every
# pre-existing corpus (incl. the mode-4 goal corpus) enumerates unchanged.
if _os.environ.get("LM_FAN_PPOBJ", ""):
    _V2_FAN["pp_obj"] = int(_os.environ["LM_FAN_PPOBJ"])
if _os.environ.get("LM_PHYS", "") == "2":
    _V2_PHYS = True
    for _k, _v in _V2_PHYS_EXCLUDE_STRICT.items():
        _V2_PHYS_EXCLUDE[_k] = _V2_PHYS_EXCLUDE.get(_k, set()) | _v

_V2_PHYS_SUBJ_BAN_MEAS = {"measurement"}   # 'the measurement measures ...' (style)
_V2_PHYS_LOC_BAN = {"measurement"}         # '... is in the measurement' (style)


# LM_FACT=1: word-level physics-fact CORPUS filter (25x25 particle-property truth table,
# 50-agent audited 2026-07-10, fact_table_v1.json). Keeps only fact-TRUE pairings in the three
# fact-bearing frames (has / has-PP with particle subjects, measures-of tails). The grammar
# validity METRIC is deliberately unchanged (user directive: validity = grammar only; fact
# accuracy of generations is reported as a separate statistic).
_V2_FACT = _os.environ.get("LM_FACT", "") == "1"
_FACT_TRUE = frozenset()
_FACT_PARTS = frozenset()
if _V2_FACT:
    import json as _json
    with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fact_table_v1.json")) as _fh:
        _ft_tab = _json.load(_fh)["table"]
    _FACT_TRUE = frozenset((p, pr) for p, row in _ft_tab.items()
                           for pr, v in row.items() if v["verdict"] == "TRUE")
    _FACT_PARTS = frozenset(_ft_tab)


def _fact_ok(t):
    if not _V2_FACT:
        return True
    n = len(t)
    if n == 5 and t[2] == "has" and t[1] in _FACT_PARTS:
        return (t[1], t[3]) in _FACT_TRUE
    if n == 8 and t[2] == "in" and t[5] == "has" and t[1] in _FACT_PARTS:
        return (t[1], t[6]) in _FACT_TRUE
    if n == 8 and t[2] == "measures" and t[6] in _FACT_PARTS:
        return (t[6], t[3]) in _FACT_TRUE
    return True


# LM_LEX=1: FULL-LEXICON corpus filter (70 subjects x 25 properties 'has' + 15 outcomes 'shows',
# 140-agent audited 2026-07-10, full_lexicon_v1.json). Word-level subcategorization: has/has-PP,
# shows/shows-PP and measures-of frames keep only lexicon-TRUE pairings. Supersedes LM_FACT
# (particle-only table). Metric handling is decided at scoring time, not here.
_V2_LEX = _os.environ.get("LM_LEX", "") == "1"
_LEX_HAS_TRUE = frozenset()
_LEX_SHOWS_TRUE = frozenset()
_LEX_SUBJ = frozenset()
if _V2_LEX:
    import json as _json2
    _lex_fname = _os.environ.get("LM_LEX_FILE", "full_lexicon_v1.json")
    with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _lex_fname)) as _fh2:
        _lx = _json2.load(_fh2)["lexicon"]
    _LEX_HAS_TRUE = frozenset((s, p) for s, row in _lx.items()
                              for p, v in row["has"].items() if v["verdict"] == "TRUE")
    _LEX_SHOWS_TRUE = frozenset((s, o) for s, row in _lx.items()
                                for o, v in row["shows"].items() if v["verdict"] == "TRUE")
    _LEX_HAS_FALSE = frozenset((s, p) for s, row in _lx.items()
                               for p, v in row["has"].items() if v["verdict"] == "FALSE")
    _LEX_SHOWS_FALSE = frozenset((s, o) for s, row in _lx.items()
                                 for o, v in row["shows"].items() if v["verdict"] == "FALSE")
    _LEX_SUBJ = frozenset(_lx)
else:
    _LEX_HAS_FALSE = frozenset()
    _LEX_SHOWS_FALSE = frozenset()

# LM_NEG=1 (requires LM_LEX=1): negation families — "[det] S has no P ." legal iff the
# lexicon marks (S,P) FALSE (true negative physics statements: "the photon has no mass").
_V2_NEG = _V2_LEX and _os.environ.get("LM_NEG", "") == "1"


def _lex_ok(t):
    if not _V2_LEX:
        return True
    n = len(t)
    if n == 6 and t[3] == "no" and t[2] == "has":
        return (t[1], t[4]) in _LEX_HAS_FALSE
    if n == 6 and t[3] == "no" and t[2] == "shows":
        return (t[1], t[4]) in _LEX_SHOWS_FALSE
    if n == 5 and t[2] == "has" and t[1] in _LEX_SUBJ:
        return (t[1], t[3]) in _LEX_HAS_TRUE
    if n == 5 and t[2] == "shows" and t[1] in _LEX_SUBJ:
        return (t[1], t[3]) in _LEX_SHOWS_TRUE
    if n == 8 and t[2] == "in" and t[5] == "has" and t[1] in _LEX_SUBJ:
        return (t[1], t[6]) in _LEX_HAS_TRUE
    if n == 8 and t[2] == "in" and t[5] == "shows" and t[1] in _LEX_SUBJ:
        return (t[1], t[6]) in _LEX_SHOWS_TRUE
    if n == 8 and t[2] == "measures" and t[6] in _LEX_SUBJ:
        return (t[6], t[3]) in _LEX_HAS_TRUE
    return True


def _phys_ok(t):
    n = len(t)
    X = _V2_PHYS_EXCLUDE
    def bad(rel, a, b): return (a, b) in X.get(rel, ())
    if n == 5 and t[2] == "has":    return not bad("S_has_P", t[1], t[3])
    if n == 5 and t[2] == "shows":  return not bad("S_shows_O", t[1], t[3])
    if n == 5 and t[2] == "is":     return not bad("S_is_Adj", t[1], t[3])
    if n == 5 and t[2] == "measures":
        return t[1] not in _V2_PHYS_SUBJ_BAN_MEAS and not bad("App_meas_P", t[1], t[3])
    if n == 6 and t[2] == "can":
        return t[1] not in _V2_PHYS_SUBJ_BAN_MEAS and not bad("App_meas_P", t[1], t[4])
    if n == 6 and t[3] == "has":    return not (bad("Adj_S", t[1], t[2]) or bad("S_has_P", t[2], t[4]))
    if n == 6 and t[3] == "shows":  return not (bad("Adj_S", t[1], t[2]) or bad("S_shows_O", t[2], t[4]))
    if n == 7 and t[2] == "is" and t[3] == "in":
        return t[5] not in _V2_PHYS_LOC_BAN and not bad("S_in_L", t[1], t[5])
    if n == 8 and t[2] == "in" and t[5] == "has":
        return t[4] not in _V2_PHYS_LOC_BAN and not (bad("S_in_L", t[1], t[4]) or bad("S_has_P", t[1], t[6]))
    if n == 8 and t[2] == "in" and t[5] == "shows":
        return t[4] not in _V2_PHYS_LOC_BAN and not (bad("S_in_L", t[1], t[4]) or bad("S_shows_O", t[1], t[6]))
    if n == 8 and t[2] == "measures":
        return (t[1] not in _V2_PHYS_SUBJ_BAN_MEAS and not bad("App_meas_P", t[1], t[3])
                and not bad("P_of_T", t[3], t[6]))
    return True


@lru_cache(maxsize=None)
def _enumerate_v2(version: str) -> Tuple[SentenceRecord, ...]:
    recs: List[SentenceRecord] = []
    props = class_words("property")
    outs = class_words("outcome")
    adjs = class_words("adjective")
    media = class_words("medium")
    appar = class_words("apparatus")
    parts = class_words("particle")
    states = class_words("stateful")

    def subs(rel: str) -> List[str]:
        out: List[str] = []
        for c in _V2_SUBJ[rel]:
            out.extend(class_words(c))
        return out

    # F1 has (5-6 tok): [det] subj has prop .
    for subj in subs("has"):
        for det in _v2_dets(subj):
            for prop in _v2_pick("obj", f"has:{subj}", props):
                recs.append(_rec([det, subj, "has", prop, "."], "s2_has", version, 1))

    # F2 has+adj (6-7): [det] adj subj has prop .  (no adjectives on apparatus)
    for subj in subs("is_adj"):
        for adj in _v2_pick("adj_mod", f"adj:{subj}", adjs):
            for det in _v2_dets(adj):
                for prop in _v2_pick("obj", f"hasadj:{subj}", props):
                    recs.append(_rec([det, adj, subj, "has", prop, "."], "s2_has_adj", version, 2))

    # F3 has+PP (8): [det] subj in [det2] medium has prop .  (particles only: physical containment)
    for subj in parts:
        for med in _v2_pick("pp_medium", f"pp:{subj}", media):
            for det in _v2_dets(subj):
                for det2 in _v2_dets(med):
                    for prop in _v2_pick("pp_obj" if "pp_obj" in _V2_FAN else "obj", f"haspp:{subj}:{med}", props):
                        recs.append(_rec([det, subj, "in", det2, med, "has", prop, "."], "s2_has_pp", version, 3))

    # F4a measure-of (8): [det] apparatus measures prop of [det2] tail .
    # LM_OF_THE=1: the of-genitive tail uses only "the" ("of the [particle]", natural physics English:
    # "the mass of the electron"). Removes the deep a/an-agreement determiner that flips to "an" on ngspice
    # (transfer-fragile); subject det keeps a/an. Makes measure-of robust on-circuit (0.69 -> 1.00).
    _of_dets = (lambda w: ("the",)) if _os.environ.get("LM_OF_THE") == "1" else _v2_dets
    for app in appar:
        for prop in _v2_pick("obj", f"meas:{app}", props):
            for tail in _v2_pick("of_tail", f"of:{app}:{prop}", parts):  # physical carriers only
                for det in _subj_dets(app):
                    for det2 in _of_dets(tail):
                        recs.append(_rec([det, app, "measures", prop, "of", det2, tail, "."], "s2_measure_of", version, 3))
    # F4c measures-short (5): [det] apparatus measures prop .
    for app in appar:
        for prop in _v2_pick("obj", f"meass:{app}", props):
            for det in _v2_dets(app):
                recs.append(_rec([det, app, "measures", prop, "."], "s2_measures_short", version, 1))
    # F4b can-measure short (6): [det] apparatus can measure prop .
    for app in appar:
        for prop in _v2_pick("obj", f"measb:{app}", props):
            for det in _v2_dets(app):
                recs.append(_rec([det, app, "can", "measure", prop, "."], "s2_can_measure", version, 1))

    # F5 shows (+adj) (5-7): [det] [adj?] subj shows outcome .
    for subj in subs("shows"):
        for det in _v2_dets(subj):
            for out_ in _v2_pick("obj", f"show:{subj}", outs):
                recs.append(_rec([det, subj, "shows", out_, "."], "s2_shows", version, 1))
        for adj in _v2_pick("adj_mod", f"showadj:{subj}", adjs):
            for det in _v2_dets(adj):
                for out_ in _v2_pick("obj", f"showadj2:{subj}", outs):
                    recs.append(_rec([det, adj, subj, "shows", out_, "."], "s2_shows_adj", version, 2))

    # F6 shows+PP (9-10): [det] subj in [det2] medium shows outcome .  (particles only)
    for subj in parts:
        for med in _v2_pick("pp_medium", f"showpp:{subj}", media):
            for det in _v2_dets(subj):
                for det2 in _v2_dets(med):
                    for out_ in _v2_pick("pp_obj" if "pp_obj" in _V2_FAN else "obj", f"showppo:{subj}", outs):
                        recs.append(_rec([det, subj, "in", det2, med, "shows", out_, "."], "s2_shows_pp", version, 3))

    # F7 is_adj (5): [det] subj is adj .
    for subj in subs("is_adj"):
        for det in _v2_dets(subj):
            for adj in _v2_pick("adj_mod", f"isadj:{subj}", adjs):
                recs.append(_rec([det, subj, "is", adj, "."], "s2_is_adj", version, 1))

    # F9 negation (6, LM_NEG=1): [det] subj has|shows no X .  — X sampled from the subject's
    # lexicon-FALSE column (true negative physics: "the photon has no mass").
    if _V2_NEG:
        _negk = int(_os.environ.get("LM_FAN_NEG", "10"))
        for subj in sorted({s for s, _ in _LEX_HAS_FALSE}):
            fp = sorted(p for s, p in _LEX_HAS_FALSE if s == subj)
            rngn = random.Random(f"{RELATION_FANOUT_SEED}:v2:neg_has:{subj}")
            for prop in (fp if _negk >= len(fp) else sorted(rngn.sample(fp, _negk))):
                for det in _v2_dets(subj):
                    recs.append(_rec([det, subj, "has", "no", prop, "."], "s2_has_no", version, 1))
        for subj in sorted({s for s, _ in _LEX_SHOWS_FALSE}):
            fo = sorted(o for s, o in _LEX_SHOWS_FALSE if s == subj)
            rngn = random.Random(f"{RELATION_FANOUT_SEED}:v2:neg_shows:{subj}")
            for out_ in (fo if _negk >= len(fo) else sorted(rngn.sample(fo, _negk))):
                for det in _v2_dets(subj):
                    recs.append(_rec([det, subj, "shows", "no", out_, "."], "s2_shows_no", version, 1))

    # F8 is_in (7): [det] subj is in [det2] loc .
    for subj in subs("is_in"):
        for loc in _v2_pick("pp_medium", f"isin:{subj}", media + appar):
            for det in _v2_dets(subj):
                for det2 in _v2_dets(loc):
                    recs.append(_rec([det, subj, "is", "in", det2, loc, "."], "s2_is_in", version, 2))

    # F-loc: LENGTH-EQUALIZED has/shows with a trailing locative PP so they END in "the [medium] ." exactly
    # like measure-of ("of the [particle] .") and is-in ("in the [loc] ."). Uniform stop cue => forced-gen
    # validity of has/shows matches the long shapes (fixes over-gen/truncation of the bare-object versions).
    _nloc = int(_os.environ.get("LM_FAN_LOC", "8"))
    _nlocm = int(_os.environ.get("LM_FAN_LOCM", "6"))
    # LM_LOC_THE=1: the locative-PP object determiner uses only "the" ("in the [medium]"), mirroring
    # LM_OF_THE for measure-of. Removes the autoregressive a/an-agreement burden in OBJECT position (the
    # model must commit to a|an BEFORE seeing the noun -> "a ensemble" slips). SUBJECT det stays mixed the/a/an
    # (_v2_dets) so the model still trains on a/an BEGINNINGS -> high mixed-test validity on every opener.
    _loc_dets = (lambda w: ("the",)) if _os.environ.get("LM_LOC_THE") == "1" else _v2_dets
    for subj in subs("has"):
        for det in _subj_dets(subj):
            for prop in _v2_pick("obj", f"hasloc:{subj}", props)[:_nloc]:
                for med in _v2_pick("pp_medium", f"haslocm:{subj}:{prop}", media)[:_nlocm]:
                    for det2 in _loc_dets(med):
                        recs.append(_rec([det, subj, "has", prop, "in", det2, med, "."], "s2_has_loc", version, 2))
    for subj in subs("shows"):
        for det in _subj_dets(subj):
            for out_ in _v2_pick("obj", f"showloc:{subj}", outs)[:_nloc]:
                for med in _v2_pick("pp_medium", f"showlocm:{subj}:{out_}", media)[:_nlocm]:
                    for det2 in _loc_dets(med):
                        recs.append(_rec([det, subj, "shows", out_, "in", det2, med, "."], "s2_shows_loc", version, 2))

    # F10 is_a (6): [det] subj is a|an TYPE .  — physics-correct particle taxonomy (hardcoded, so it is
    # exempt from _phys_ok). Particle-subject => does not dominate cold-read generation; adds a distinct
    # "is a X" silhouette and makes the post-"is" decision a genuine 3-way (adj | in | a).
    _ISA_MAP = {
        "electron": ("lepton", "fermion"), "muon": ("lepton", "fermion"),
        "neutrino": ("lepton", "fermion"), "positron": ("lepton", "fermion"),
        "quark": ("fermion",),
        "proton": ("baryon", "hadron", "fermion", "nucleon"),
        "neutron": ("baryon", "hadron", "fermion", "nucleon"),
        "nucleon": ("baryon", "hadron", "fermion"),
        "pion": ("meson", "hadron", "boson"), "kaon": ("meson", "hadron", "boson"),
        "deuteron": ("hadron", "boson"),
        "photon": ("boson",), "gluon": ("boson",),
        "baryon": ("hadron", "fermion"), "meson": ("hadron", "boson"), "lepton": ("fermion",),
    }
    for subj, types in _ISA_MAP.items():
        for det in _v2_dets(subj):
            for ty in types:
                recs.append(_rec([det, subj, "is", _det_indef(ty), ty, "."], "s2_is_a", version, 1))

    # F3b/F6b adj+PP (9 tok): [det] adj subj in [det2] medium has|shows obj .  — long structure
    # that FILLS the full 7-context (9 tokens => final prediction sees 7 real tokens). Particle-subject so
    # it does not dominate cold-read generation. Adjectives restricted to those physically sensible for a
    # particle in a medium; subj-obj pairing still passes _phys_ok. _det_indef agrees with the adjective.
    _PP_ADJ = ("coherent", "incoherent", "classical", "quantum", "excited", "degenerate")
    _pp_adj = [a for a in _PP_ADJ if a in adjs]
    _nppo = int(_os.environ.get("LM_FAN_PPADJOBJ", "2"))
    for subj in parts:
        for med in _v2_pick("pp_medium", f"ppadj:{subj}", media):
            for adj in _pp_adj:
                det = _det_indef(adj)
                for det2 in _v2_dets(med):
                    for prop in _v2_pick("obj", f"hasppadj:{subj}:{med}:{adj}", props)[:_nppo]:
                        recs.append(_rec([det, adj, subj, "in", det2, med, "has", prop, "."], "s2_has_pp_adj", version, 3))
                    for out_ in _v2_pick("obj", f"showppadj:{subj}:{med}:{adj}", outs)[:_nppo]:
                        recs.append(_rec([det, adj, subj, "in", det2, med, "shows", out_, "."], "s2_shows_pp_adj", version, 3))

    # QUESTIONS (v2 interrogatives): exercise ?, what, why, is with short forms; internal det = "the".
    # q_what_simple "what is the [head] ?" (head = any content noun, so also covers e.g. measurement);
    # q_why "why is the [subj] [adj] ?" (physics-valid subject-adjective pairs, same pool as s2_is_adj).
    for head in (props + parts + outs + media + appar + states):
        recs.append(_rec(["what", "is", "the", head, "?"], "q_what_simple", version, 1))
    for subj in subs("is_adj"):
        for adj in _v2_pick("adj_mod", f"qwhy:{subj}", adjs):
            recs.append(_rec(["why", "is", "the", subj, adj, "?"], "q_why", version, 1))

    if _V2_SIMPLE:
        recs = [r for r in recs if r.family in _V2_SIMPLE_FAMILIES]
    elif _V2_MIDDLE:
        recs = [r for r in recs if r.family in _V2_MIDDLE_FAMILIES]
    elif _V2_DIVERSE:
        recs = [r for r in recs if r.family in _V2_DIVERSE_FAMILIES]
    if _V2_PHYS:
        recs = [r for r in recs if r.family == "s2_is_a" or r.family.startswith("q_") or _phys_ok(r.tokens)]
    if _V2_FACT:
        recs = [r for r in recs if _fact_ok(r.tokens)]
    if _V2_LEX:
        recs = [r for r in recs if _lex_ok(r.tokens)]
    return _unique_records(recs)


@lru_cache(maxsize=None)
def enumerate_sentence_records(grammar_version: str) -> Tuple[SentenceRecord, ...]:
    if grammar_version not in GRAMMAR_VERSIONS:
        raise ValueError(f"unknown grammar version: {grammar_version}")
    version = grammar_version
    records: List[SentenceRecord] = []

    if version == "grammar_v2_sci":
        return _enumerate_v2(version)

    has = _relation("has")
    measure = _relation("can_measure")
    shows = _relation("shows")
    isin = _relation("is_in")
    isadj = _relation("is_adj")
    why = _relation("why")
    of = _relation("of")
    what_heads = _members(["__all_nouns__"])

    # is_adj : [det] subj "is" adj "."
    for det in DETERMINERS:
        for subj, adjs in isadj.items():
            for adj in adjs:
                records.append(_rec([det, subj, "is", adj, "."], "stmt_is_adj", version, 1))

    det2_list = DETERMINERS if _OBJ_DET else (None,)
    for det1 in DETERMINERS:
        for det2 in det2_list:
            # has : [det1] subj "has" [det2] obj "."
            for subj, objs in has.items():
                for obj in objs:
                    records.append(_rec([det1, subj, "has", *_objslot(det2, obj), "."], "stmt_has", version, 1))
            # can measure : [det1] subj "can" "measure" [det2] obj "."
            for subj, objs in measure.items():
                for obj in objs:
                    records.append(_rec([det1, subj, "can", "measure", *_objslot(det2, obj), "."], "stmt_can_measure", version, 1))
            # shows : [det1] subj "shows" [det2] obj "."
            for subj, objs in shows.items():
                for obj in objs:
                    records.append(_rec([det1, subj, "shows", *_objslot(det2, obj), "."], "stmt_shows", version, 1))
            # is in : [det1] subj "is" "in" [det2] loc "."
            for subj, locs in isin.items():
                for loc in locs:
                    records.append(_rec([det1, subj, "is", "in", *_objslot(det2, loc), "."], "stmt_is_in", version, 1))
            # what simple : "what" "is" [det1] head "?"
            for head in what_heads:
                records.append(_rec(["what", "is", det1, head, "?"], "q_what_simple", version, 1))
            # what of : "what" "is" [det1] head "of" [det2] tail "?"
            for head, tails in of.items():
                for tail in tails:
                    records.append(_rec(["what", "is", det1, head, "of", *_objslot(det2, tail), "?"], "q_what_of", version, 1))
            # why : "why" "is" [det1] subj adj "?"
            for subj, adjs in why.items():
                for adj in adjs:
                    records.append(_rec(["why", "is", det1, subj, adj, "?"], "q_why", version, 1))

    if _FAMILY_FILTER:
        records = [r for r in records if _family_enabled(r.family)]
    return _unique_records(records)


@lru_cache(maxsize=None)
def valid_sentence_texts(grammar_version: str) -> frozenset[str]:
    return frozenset(record.text for record in enumerate_sentence_records(grammar_version))


def _cls(w: str) -> str:
    return TOKEN_CLASS.get(str(w), "")


def _v2_det_ok(d: str, w: str) -> bool:
    # the / agreement-correct a|an, plus (MORE_DETS) agreement-free this/that/each/every (valid before any singular noun)
    return d == "the" or d == _det_indef(w) or (_MORE_DETS and d in _EXTRA_DETS)


def _v2_sci_correct(toks: Sequence[str]) -> bool:
    """STRICT scientific well-typedness (class-typed relations). Secondary metric only —
    NOT the validity criterion (per user: validity = general grammar, science = corpus bias)."""
    t = [str(x) for x in toks]
    n = len(t)
    HAS_S = {"particle", "stateful", "medium", "apparatus"}
    ADJ_S = {"particle", "stateful", "medium"}
    if n == 5 and t[4] == ".":
        d, S, v, O = t[0], t[1], t[2], t[3]
        if not _v2_det_ok(d, S):
            return False
        if v == "has":
            return _cls(S) in HAS_S and _cls(O) == "property"
        if v == "shows":
            return _cls(S) in ADJ_S and _cls(O) == "outcome"
        if v == "measures":
            return _cls(S) == "apparatus" and _cls(O) == "property"
        if v == "is":
            return _cls(S) in ADJ_S and _cls(O) == "adjective"
        return False
    if n == 6 and t[5] == ".":
        d, A, S, v, O = t[0], t[1], t[2], t[3], t[4]
        if _cls(A) != "adjective" or not _v2_det_ok(d, A) or _cls(S) not in ADJ_S:
            return False
        if v == "has":
            return _cls(O) == "property"
        if v == "shows":
            return _cls(O) == "outcome"
        return False
    if n == 7 and t[2] == "is" and t[3] == "in" and t[6] == ".":
        d, S, d2, L = t[0], t[1], t[4], t[5]
        return (_v2_det_ok(d, S) and _cls(S) in {"particle", "stateful"}
                and _v2_det_ok(d2, L) and _cls(L) in {"medium", "apparatus"})
    if n == 8 and t[2] == "in" and t[7] == ".":
        d, S, d2, M, v, O = t[0], t[1], t[3], t[4], t[5], t[6]
        if not (_v2_det_ok(d, S) and _cls(S) == "particle" and _v2_det_ok(d2, M) and _cls(M) in {"medium", "apparatus"}):
            return False
        if v == "has":
            return _cls(O) == "property"
        if v == "shows":
            return _cls(O) == "outcome"
        return False
    if n == 8 and t[2] == "measures" and t[4] == "of" and t[7] == ".":
        d, App, P, d2, T = t[0], t[1], t[3], t[5], t[6]
        return (_v2_det_ok(d, App) and _cls(App) == "apparatus" and _cls(P) == "property"
                and _v2_det_ok(d2, T) and _cls(T) == "particle")
    if n == 6 and t[2] == "can" and t[3] == "measure" and t[5] == ".":
        d, App, P = t[0], t[1], t[4]
        return _v2_det_ok(d, App) and _cls(App) == "apparatus" and _cls(P) == "property"
    return False


_V2_NOUNS = {"particle", "property", "apparatus", "medium", "stateful", "outcome"}


def _v2_rule_valid(toks: Sequence[str]) -> bool:
    """General-GRAMMAR v2 validity: template structure + parts of speech + a/an agreement.
    Class typing among nouns is NOT required (scientifically-odd but grammatical counts
    valid; the training corpus biases toward scientific correctness — that bias is measured
    separately by _v2_sci_correct)."""
    t = [str(x) for x in toks]
    n = len(t)

    def N(w):
        return _cls(w) in _V2_NOUNS

    def A(w):
        return _cls(w) == "adjective"

    # questions: "what is the [head] ?"  /  "why is the [subj] [adj] ?"
    if n == 5 and t[0] == "what" and t[1] == "is" and t[2] == "the" and t[4] == "?":
        return N(t[3])
    if n == 6 and t[0] == "why" and t[1] == "is" and t[2] == "the" and t[5] == "?":
        return N(t[3]) and A(t[4])

    if n == 5 and t[4] == ".":
        d, S, v, O = t[0], t[1], t[2], t[3]
        if not (_v2_det_ok(d, S) and N(S)):
            return False
        if v in ("has", "shows", "measures"):
            return N(O)
        if v == "is":
            return A(O)
        return False
    if n == 6 and t[5] == ".":
        if t[2] in ("has", "shows") and t[3] == "no":
            d, S, X = t[0], t[1], t[4]
            return _v2_det_ok(d, S) and N(S) and N(X)
        if t[2] == "can" and t[3] == "measure":
            d, S, P = t[0], t[1], t[4]
            return _v2_det_ok(d, S) and N(S) and N(P)
        d, Aj, S, v, O = t[0], t[1], t[2], t[3], t[4]
        return (A(Aj) and _v2_det_ok(d, Aj) and N(S)
                and v in ("has", "shows") and N(O))
    if n == 7 and t[2] == "is" and t[3] == "in" and t[6] == ".":
        d, S, d2, L = t[0], t[1], t[4], t[5]
        return _v2_det_ok(d, S) and N(S) and _v2_det_ok(d2, L) and N(L)
    if n == 8 and t[2] == "in" and t[7] == ".":
        d, S, d2, M, v, O = t[0], t[1], t[3], t[4], t[5], t[6]
        return (_v2_det_ok(d, S) and N(S) and _v2_det_ok(d2, M) and N(M)
                and v in ("has", "shows") and N(O))
    if n == 8 and t[2] == "measures" and t[4] == "of" and t[7] == ".":
        d, S, P, d2, T = t[0], t[1], t[3], t[5], t[6]
        return (_v2_det_ok(d, S) and N(S) and N(P) and _v2_det_ok(d2, T) and N(T))
    # length-equalized has_loc/shows_loc: [det] S has|shows [O] in [det2] [M] .
    if n == 8 and t[2] in ("has", "shows") and t[4] == "in" and t[7] == ".":
        d, S, O, d2, M = t[0], t[1], t[3], t[5], t[6]
        return (_v2_det_ok(d, S) and N(S) and N(O) and _v2_det_ok(d2, M) and N(M))
    return False


def is_valid_sentence(tokens: Sequence[str], grammar_version: str) -> bool:
    if grammar_version == "grammar_v2_sci":
        return _v2_rule_valid(tokens)
    return sentence_text(tokens) in valid_sentence_texts(grammar_version)


def sci_correct_rate(token_seqs: Sequence[Sequence[str]]) -> float:
    """Fraction of sequences that are ALSO scientifically well-typed (secondary metric)."""
    if not token_seqs:
        return 0.0
    return sum(1 for t in token_seqs if _v2_sci_correct(t)) / len(token_seqs)


def grammar_summary(grammar_version: str) -> Dict[str, object]:
    records = enumerate_sentence_records(grammar_version)
    family_counts: Dict[str, int] = Counter(record.family for record in records)
    start_counts: Dict[str, int] = Counter(record.start_token for record in records)
    length_counts: Dict[str, int] = Counter(record.length for record in records)
    return {
        "grammar_version": grammar_version,
        "sentence_count": len(records),
        "family_counts": dict(sorted(family_counts.items())),
        "start_token_counts": dict(sorted(start_counts.items())),
        "length_counts": {str(k): int(v) for k, v in sorted(length_counts.items())},
        "token_coverage": _counter_dict(records),
    }


def summarize_records(records: Sequence[SentenceRecord]) -> Dict[str, object]:
    family_counts: Dict[str, int] = Counter(record.family for record in records)
    start_counts: Dict[str, int] = Counter(record.start_token for record in records)
    length_counts: Dict[str, int] = Counter(record.length for record in records)
    complexity_counts: Dict[str, int] = Counter(record.complexity for record in records)
    return {
        "count": len(records),
        "family_counts": dict(sorted(family_counts.items())),
        "start_token_counts": dict(sorted(start_counts.items())),
        "length_counts": {str(k): int(v) for k, v in sorted(length_counts.items())},
        "complexity_counts": {str(k): int(v) for k, v in sorted(complexity_counts.items())},
        "token_coverage": _counter_dict(records),
    }


def records_to_text_set(records: Sequence[SentenceRecord]) -> frozenset[str]:
    return frozenset(record.text for record in records)


# -----------------------------------------------------------------------------
# Train/dev/test split with guaranteed token coverage (same logic as qm32_spec)
# -----------------------------------------------------------------------------
def split_sentence_pools(
    *,
    grammar_version: str,
    seed: int,
    dev_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Dict[str, object]:
    if not (0.0 < dev_frac < 0.5 and 0.0 < test_frac < 0.5 and dev_frac + test_frac < 1.0):
        raise ValueError("dev_frac and test_frac must be in (0, 0.5) and sum to < 1.0")

    rng = random.Random(seed)
    records = list(enumerate_sentence_records(grammar_version))
    grouped: MutableMapping[Tuple[str, str, int], List[SentenceRecord]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record)].append(record)

    train_pool: List[SentenceRecord] = []
    dev_pool: List[SentenceRecord] = []
    test_pool: List[SentenceRecord] = []

    for key in sorted(grouped.keys()):
        items = grouped[key]
        rng.shuffle(items)
        n = len(items)
        n_dev = int(round(n * dev_frac))
        n_test = int(round(n * test_frac))
        if n >= 6:
            n_dev = max(1, n_dev)
            n_test = max(1, n_test)
        elif n >= 4:
            n_dev = max(1, min(n_dev, n - 2))
            n_test = max(1, min(n_test, n - n_dev - 1))
        else:
            n_dev = min(n_dev, max(0, n - 2))
            n_test = min(n_test, max(0, n - n_dev - 1))

        while n_dev + n_test > n - 1:
            if n_test >= n_dev and n_test > 0:
                n_test -= 1
            elif n_dev > 0:
                n_dev -= 1
            else:
                break

        dev_pool.extend(items[:n_dev])
        test_pool.extend(items[n_dev:n_dev + n_test])
        train_pool.extend(items[n_dev + n_test:])

    # only enforce coverage for tokens that actually occur in the active grammar
    present_tokens = {t for r in records for t in r.tokens}
    excluded_from_coverage = {"<BOS>"} | (set(VOCAB) - present_tokens)
    for tok in VOCAB:
        if tok in excluded_from_coverage:
            continue
        if any(tok in record.tokens for record in train_pool):
            continue
        moved = False
        for source in (dev_pool, test_pool):
            for idx, record in enumerate(source):
                if tok in record.tokens:
                    train_pool.append(source.pop(idx))
                    moved = True
                    break
            if moved:
                break
        if not moved:
            raise RuntimeError(f"could not ensure train coverage for token {tok!r}")

    train_pool = sorted(train_pool, key=lambda r: (r.family, r.tokens))
    dev_pool = sorted(dev_pool, key=lambda r: (r.family, r.tokens))
    test_pool = sorted(test_pool, key=lambda r: (r.family, r.tokens))

    return {
        "grammar_version": grammar_version,
        "seed": int(seed),
        "summary": grammar_summary(grammar_version),
        "train_pool": [record.to_dict() for record in train_pool],
        "dev_novel_pool": [record.to_dict() for record in dev_pool],
        "test_novel_pool": [record.to_dict() for record in test_pool],
        "pool_sizes": {
            "train": len(train_pool),
            "dev_novel": len(dev_pool),
            "test_novel": len(test_pool),
        },
        "train_coverage": _counter_dict(train_pool),
        "dev_coverage": _counter_dict(dev_pool),
        "test_coverage": _counter_dict(test_pool),
    }


def as_train_test_bundle(bundle: Mapping[str, object]) -> Dict[str, object]:
    """Expose the figure-evaluation split as the sole held-out test split.

    The final reduced-vocabulary paper run evaluated its epoch-wise CE and QMass
    on the split historically stored as ``dev_novel_pool``.  This adapter keeps
    those exact sentences—and therefore the published curves—but removes the
    unused third split from the final data contract.
    """
    return {
        "grammar_version": bundle["grammar_version"],
        "seed": bundle["seed"],
        "summary": bundle["summary"],
        "train_pool": bundle["train_pool"],
        "test_pool": bundle["dev_novel_pool"],
        "pool_sizes": {
            "train": int(bundle["pool_sizes"]["train"]),
            "test": int(bundle["pool_sizes"]["dev_novel"]),
        },
        "train_coverage": bundle["train_coverage"],
        "test_coverage": bundle["dev_coverage"],
    }


def _records_from_payload(items: Sequence[Mapping[str, object]]) -> List[SentenceRecord]:
    return [SentenceRecord.from_dict(item) for item in items]


def save_sentence_pool_bundle(path: Path, bundle: Mapping[str, object]) -> None:
    path.write_text(json.dumps(bundle, indent=2))


def load_sentence_pool_bundle(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def save_sentence_records(path: Path, records: Sequence[SentenceRecord]) -> None:
    path.write_text(json.dumps([record.to_dict() for record in records], indent=2))


def load_sentence_records(path: Path) -> List[SentenceRecord]:
    return _records_from_payload(json.loads(path.read_text()))


def write_pool_bundle_files(out_dir: Path, bundle: Mapping[str, object]) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "sentence_pools.json"
    save_sentence_pool_bundle(bundle_path, bundle)
    train_path = out_dir / "train_pool.json"
    dev_path = out_dir / "dev_novel_pool.json"
    test_path = out_dir / "test_novel_pool.json"
    meta_path = out_dir / "pool_meta.json"
    save_sentence_records(train_path, _records_from_payload(bundle["train_pool"]))
    save_sentence_records(dev_path, _records_from_payload(bundle["dev_novel_pool"]))
    save_sentence_records(test_path, _records_from_payload(bundle["test_novel_pool"]))
    meta = {
        "grammar_version": bundle["grammar_version"],
        "seed": bundle["seed"],
        "pool_sizes": bundle["pool_sizes"],
        "summary": bundle["summary"],
        "train_coverage": bundle["train_coverage"],
        "dev_coverage": bundle["dev_coverage"],
        "test_coverage": bundle["test_coverage"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"bundle": bundle_path, "train": train_path, "dev": dev_path, "test": test_path, "meta": meta_path}


def write_train_test_pool_files(out_dir: Path, bundle: Mapping[str, object]) -> Dict[str, Path]:
    """Write a final-release bundle containing only train and test pools."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "sentence_pools.json"
    save_sentence_pool_bundle(bundle_path, bundle)
    train_path = out_dir / "train_pool.json"
    test_path = out_dir / "test_pool.json"
    meta_path = out_dir / "pool_meta.json"
    save_sentence_records(train_path, _records_from_payload(bundle["train_pool"]))
    save_sentence_records(test_path, _records_from_payload(bundle["test_pool"]))
    meta = {
        "grammar_version": bundle["grammar_version"],
        "seed": bundle["seed"],
        "pool_sizes": bundle["pool_sizes"],
        "summary": bundle["summary"],
        "train_coverage": bundle["train_coverage"],
        "test_coverage": bundle["test_coverage"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"bundle": bundle_path, "train": train_path, "test": test_path, "meta": meta_path}


def records_from_maybe_bundle(
    *,
    bundle_path: Path | None = None,
    records_path: Path | None = None,
    key: str | None = None,
) -> List[SentenceRecord]:
    if records_path is not None:
        return load_sentence_records(records_path)
    if bundle_path is None or key is None:
        raise ValueError("provide either records_path or both bundle_path and key")
    bundle = load_sentence_pool_bundle(bundle_path)
    return _records_from_payload(bundle[key])


# full-vocab v2 corpus: the primary shapes (s2_measure_of/has_loc/shows_loc) have huge pools
# (2910/3000/1800 records) while the aux + question families are tiny (is_adj 156, q_why 78).
# Since sampling is proportional to pool size, questions get ~1% of the sample and never learn
# the "?" ending or short structures. LM_BALANCE_AUX=1 upweights each rare family so it
# contributes ~800 effective samples (enough to learn) while the primary shapes stay dominant.
_BALANCE_AUX = _os.environ.get("LM_BALANCE_AUX") == "1"
_AUX_FAMILY_WEIGHT = {
    # is_adj + q_why (adjective emitters) and can_measure need extra exposure: adjectives are new
    # output classes and "is"/"can measure" sit AFTER the shared [det][subj] prefix, so the primary
    # noun-prior overrides them. q_what learned fine at weight 8 ("what" leads, no interference).
    "s2_is_adj": 10.0,       # 156 recs x adj_mod-8 -> more unique adj examples, ~heavily upsampled
    "s2_shows": 2.0,         # short "shows outcome" (part of shows-in, already clean)
    "s2_can_measure": 5.0,   # emit [prop] then "." (model bleeds has/shows-loc "in the" tail)
    "q_what_simple": 8.0,    # already clean at 8 (0.97)
    "q_why": 16.0,           # adjective emitter + 3-token fill; smallest pool (78) -> highest weight
}


def _family_weight(stage: str, record: SentenceRecord) -> float:
    # gently downweight the longer (q_what_of) template so short families stay
    # well represented; otherwise uniform across families in the core stage.
    if stage != "core":
        raise ValueError(f"unknown curriculum stage: {stage}")
    if _BALANCE_AUX and record.family in _AUX_FAMILY_WEIGHT:
        return _AUX_FAMILY_WEIGHT[record.family]
    base = {
        "stmt_is_adj": 1.2,
        "stmt_has": 1.1,
        "stmt_can_measure": 1.1,
        "stmt_shows": 1.1,
        "stmt_is_in": 1.0,
        "q_what_simple": 1.1,
        "q_what_of": 0.85,
        "q_why": 1.0,
    }.get(record.family, 1.0)
    return base


def sample_training_sentences(
    train_pool: Sequence[SentenceRecord],
    *,
    num_sentences: int,
    curriculum_stage: str,
    min_target_count: int,
    seed: int,
) -> Tuple[List[List[str]], Dict[str, int]]:
    if curriculum_stage not in CURRICULUM_STAGES:
        raise ValueError(f"unknown curriculum stage: {curriculum_stage}")
    if num_sentences <= 0:
        raise ValueError("num_sentences must be > 0")

    rng = random.Random(seed)
    pool = list(train_pool)
    weights = [max(1e-6, _family_weight(curriculum_stage, record)) for record in pool]

    sampled: List[List[str]] = []
    counter: Counter[str] = Counter()
    for _ in range(int(num_sentences)):
        record = rng.choices(pool, weights=weights, k=1)[0]
        sampled.append(list(record.tokens))
        counter.update(record.tokens)

    exempt = {"<BOS>"}
    for tok in VOCAB:
        if tok in exempt:
            continue
        while counter.get(tok, 0) < min_target_count:
            candidates = [record for record in pool if tok in record.tokens]
            if not candidates:
                break
            record = rng.choice(candidates)
            sampled.append(list(record.tokens))
            counter.update(record.tokens)

    return sampled, {tok: int(counter.get(tok, 0)) for tok in VOCAB}


def build_windows_from_sentences(
    sentences: Sequence[Sequence[str]],
    *,
    context_len: int = CONTEXT_LEN,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    bos_id = TOKEN_TO_ID["<BOS>"]
    contexts: List[List[int]] = []
    targets: List[int] = []
    counter: Counter[str] = Counter()
    for sentence in sentences:
        ctx = [bos_id] * context_len
        for tok in sentence:
            targets.append(TOKEN_TO_ID[str(tok)])
            contexts.append(list(ctx))
            counter[str(tok)] += 1
            ctx = ctx[1:] + [TOKEN_TO_ID[str(tok)]]
    return (
        np.asarray(contexts, dtype=int),
        np.asarray(targets, dtype=int),
        {tok: int(counter.get(tok, 0)) for tok in VOCAB},
    )


if __name__ == "__main__":
    summ = grammar_summary("grammar_v1_core")
    print("sentence_count:", summ["sentence_count"])
    print("family_counts:", json.dumps(summ["family_counts"], indent=1))
    print("length_counts:", summ["length_counts"])
    bundle = split_sentence_pools(grammar_version="grammar_v1_core", seed=1)
    print("pool_sizes:", bundle["pool_sizes"])
    cov = bundle["train_coverage"]
    missing = [t for t in VOCAB if t != "<BOS>" and cov.get(t, 0) == 0]
    print("train tokens missing:", missing)
