from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

CONTEXT_LEN = 6
START_TOKENS = ["the", "a", "what", "why"]
TERMINAL_PUNCT = {".", "?"}

VOCAB: List[str] = [
    "<BOS>", ".", "?", "the", "a", "what", "why", "in", "of",
    "is", "has", "can", "measure", "shows",
    "electron", "photon", "atom", "qubit",
    "spin", "phase", "energy", "basis",
    "wave", "field", "state", "system",
    "detector", "measurement", "superposition", "entanglement",
    "pure", "mixed",
]

TOKEN_TO_ID: Dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {idx: tok for tok, idx in TOKEN_TO_ID.items()}

DETERMINERS = ("the", "a")
GRAMMAR_VERSIONS = ("grammar_v1_core", "grammar_v2_expanded", "grammar_v3_novelty_push")
CURRICULUM_STAGES = ("core", "expanded", "novelty_push")

NOUN_CLASSES: Dict[str, str] = {
    "electron": "particle",
    "photon": "particle",
    "atom": "particle",
    "qubit": "particle",
    "spin": "property",
    "phase": "property",
    "energy": "property",
    "basis": "property",
    "wave": "medium",
    "field": "medium",
    "state": "stateful",
    "system": "stateful",
    "detector": "apparatus",
    "measurement": "apparatus",
    "superposition": "outcome",
    "entanglement": "outcome",
}

IS_ADJ_HEADS: Dict[str, List[str]] = {
    "state": ["pure", "mixed"],
    "system": ["pure", "mixed"],
    "qubit": ["pure", "mixed"],
    "superposition": ["pure", "mixed"],
    "entanglement": ["pure", "mixed"],
}

HAS_OBJECTS: Dict[str, List[str]] = {
    "electron": ["spin", "energy", "phase"],
    "photon": ["energy", "phase"],
    "atom": ["state", "energy", "phase"],
    "qubit": ["state", "phase", "basis"],
    "state": ["phase", "basis", "energy"],
    "system": ["state", "energy", "basis", "entanglement"],
}

CAN_MEASURE_OBJECTS: Dict[str, List[str]] = {
    "detector": ["spin", "phase", "energy", "basis", "state"],
    "system": ["spin", "phase", "state", "energy"],
    "measurement": ["spin", "phase", "energy", "basis", "state"],
}

SHOWS_OBJECTS: Dict[str, List[str]] = {
    "measurement": ["state", "superposition", "entanglement", "phase"],
    "detector": ["state", "superposition", "entanglement", "measurement"],
    "wave": ["phase", "energy", "state"],
    "field": ["phase", "energy", "state"],
    "system": ["state", "superposition", "entanglement"],
}

IN_OBJECTS: Dict[str, List[str]] = {
    "qubit": ["superposition", "basis"],
    "system": ["superposition", "entanglement", "basis"],
    "state": ["basis", "phase"],
    "wave": ["phase", "field"],
    "field": ["phase", "wave"],
    "measurement": ["system", "state"],
}

WHAT_SIMPLE_HEADS = [
    "electron", "photon", "atom", "qubit",
    "spin", "phase", "energy", "basis",
    "wave", "field", "state", "system",
    "detector", "measurement", "superposition", "entanglement",
]

OF_HEAD_TO_TAIL: Dict[str, List[str]] = {
    "spin": ["electron", "atom", "qubit", "system"],
    "phase": ["photon", "wave", "field", "system", "state"],
    "energy": ["electron", "photon", "atom", "system", "field"],
    "basis": ["qubit", "state", "system"],
    "state": ["atom", "qubit", "system", "measurement"],
    "superposition": ["qubit", "system", "measurement"],
    "entanglement": ["system", "measurement"],
}

WHY_HEADS = {
    "state": ["pure", "mixed"],
    "system": ["pure", "mixed"],
    "qubit": ["pure", "mixed"],
    "superposition": ["pure", "mixed"],
    "entanglement": ["pure", "mixed"],
}

HEAD_IN_LOCS: Dict[str, List[str]] = {
    "state": ["atom", "qubit", "system", "measurement"],
    "phase": ["wave", "field", "state", "system"],
    "superposition": ["qubit", "system", "measurement"],
    "basis": ["qubit", "state", "system"],
    "energy": ["atom", "system", "field"],
}

OBJECT_LOCATIONS: Dict[str, List[str]] = {
    "spin": ["electron", "atom", "qubit", "system"],
    "phase": ["wave", "field", "state", "system"],
    "energy": ["electron", "photon", "atom", "field", "system"],
    "basis": ["qubit", "state", "system"],
    "state": ["atom", "qubit", "system", "measurement"],
    "superposition": ["qubit", "system", "measurement"],
    "entanglement": ["system", "measurement"],
    "measurement": ["detector", "system"],
}


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


def _unique_records(records: Iterable[SentenceRecord]) -> Tuple[SentenceRecord, ...]:
    dedup: Dict[str, SentenceRecord] = {}
    for record in records:
        dedup.setdefault(record.text, record)
    return tuple(sorted(dedup.values(), key=lambda r: (r.family, r.tokens)))


def _rec(tokens: Sequence[str], family: str, version: str, complexity: int) -> SentenceRecord:
    return SentenceRecord(tokens=tuple(tokens), family=family, grammar_version=version, complexity=complexity)


def _counter_dict(records: Sequence[SentenceRecord]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.tokens)
    return {tok: int(counter.get(tok, 0)) for tok in VOCAB}


def _group_key(record: SentenceRecord) -> Tuple[str, str, int]:
    return record.family, record.start_token, record.length


def _family_weight(stage: str, record: SentenceRecord) -> float:
    if stage == "core":
        family_weights = {
            "stmt_is_adj": 1.4,
            "stmt_has": 1.3,
            "stmt_can_measure": 1.2,
            "stmt_shows": 1.1,
            "stmt_is_in": 0.9,
            "q_what_simple": 1.1,
            "q_what_of": 0.9,
            "q_why": 0.9,
        }
        base = family_weights.get(record.family, 0.45)
        return base / max(1.0, 0.18 * (record.length - 5))

    if stage == "expanded":
        family_weights = {
            "stmt_is_adj": 1.0,
            "stmt_has": 1.0,
            "stmt_can_measure": 1.0,
            "stmt_shows": 1.0,
            "stmt_is_in": 0.95,
            "stmt_has_of": 0.9,
            "stmt_shows_of": 0.9,
            "q_what_simple": 0.95,
            "q_what_of": 0.95,
            "q_what_adj": 0.9,
            "q_what_in": 0.9,
            "q_why": 0.9,
        }
        return family_weights.get(record.family, 0.8)

    if stage == "novelty_push":
        family_weights = {
            "stmt_has_of": 1.1,
            "stmt_shows_of": 1.1,
            "stmt_has_in": 1.2,
            "stmt_measure_in": 1.2,
            "stmt_shows_in": 1.2,
            "q_what_of_in": 1.3,
            "q_what_in": 1.1,
            "q_what_adj": 1.0,
            "q_what_of": 1.0,
        }
        return family_weights.get(record.family, 0.8) * (1.0 + 0.08 * max(0, record.length - 5))

    raise ValueError(f"unknown curriculum stage: {stage}")


def _simple_np(noun: str, det: str) -> List[str]:
    return [det, noun]


@lru_cache(maxsize=None)
def enumerate_sentence_records(grammar_version: str) -> Tuple[SentenceRecord, ...]:
    if grammar_version not in GRAMMAR_VERSIONS:
        raise ValueError(f"unknown grammar version: {grammar_version}")

    records: List[SentenceRecord] = []
    version = grammar_version

    for det in DETERMINERS:
        for subj, adjs in IS_ADJ_HEADS.items():
            for adj in adjs:
                records.append(_rec([det, subj, "is", adj, "."], "stmt_is_adj", version, 1))

    for det1 in DETERMINERS:
        for det2 in DETERMINERS:
            for subj, objects in HAS_OBJECTS.items():
                for obj in objects:
                    records.append(_rec([det1, subj, "has", det2, obj, "."], "stmt_has", version, 1))
            for subj, objects in CAN_MEASURE_OBJECTS.items():
                for obj in objects:
                    records.append(_rec([det1, subj, "can", "measure", det2, obj, "."], "stmt_can_measure", version, 1))
            for subj, objects in SHOWS_OBJECTS.items():
                for obj in objects:
                    records.append(_rec([det1, subj, "shows", det2, obj, "."], "stmt_shows", version, 1))
            for subj, locs in IN_OBJECTS.items():
                for loc in locs:
                    records.append(_rec([det1, subj, "is", "in", det2, loc, "."], "stmt_is_in", version, 1))
            for head in WHAT_SIMPLE_HEADS:
                records.append(_rec(["what", "is", det1, head, "?"], "q_what_simple", version, 1))
            for head, tails in OF_HEAD_TO_TAIL.items():
                for tail in tails:
                    records.append(_rec(["what", "is", det1, head, "of", det2, tail, "?"], "q_what_of", version, 1))
            for subj, adjs in WHY_HEADS.items():
                for adj in adjs:
                    records.append(_rec(["why", "is", det1, subj, adj, "?"], "q_why", version, 1))

    if grammar_version in {"grammar_v2_expanded", "grammar_v3_novelty_push"}:
        for det1 in DETERMINERS:
            for det2 in DETERMINERS:
                for det3 in DETERMINERS:
                    for subj, objects in HAS_OBJECTS.items():
                        for head in objects:
                            if head not in OF_HEAD_TO_TAIL:
                                continue
                            for tail in OF_HEAD_TO_TAIL[head]:
                                records.append(_rec(
                                    [det1, subj, "has", det2, head, "of", det3, tail, "."],
                                    "stmt_has_of",
                                    version,
                                    2,
                                ))
                    for subj, objects in SHOWS_OBJECTS.items():
                        for head in objects:
                            if head not in OF_HEAD_TO_TAIL:
                                continue
                            for tail in OF_HEAD_TO_TAIL[head]:
                                records.append(_rec(
                                    [det1, subj, "shows", det2, head, "of", det3, tail, "."],
                                    "stmt_shows_of",
                                    version,
                                    2,
                                ))
                    for head, locs in HEAD_IN_LOCS.items():
                        for loc in locs:
                            records.append(_rec(
                                ["what", "is", det1, head, "in", det2, loc, "?"],
                                "q_what_in",
                                version,
                                2,
                            ))
                for head, adjs in IS_ADJ_HEADS.items():
                    for adj in adjs:
                        records.append(_rec(["what", "is", det1, adj, head, "?"], "q_what_adj", version, 2))

    if grammar_version == "grammar_v3_novelty_push":
        for det1 in DETERMINERS:
            for det2 in DETERMINERS:
                for det3 in DETERMINERS:
                    for subj, objects in HAS_OBJECTS.items():
                        for obj in objects:
                            for loc in OBJECT_LOCATIONS.get(obj, []):
                                records.append(_rec(
                                    [det1, subj, "has", det2, obj, "in", det3, loc, "."],
                                    "stmt_has_in",
                                    version,
                                    3,
                                ))
                    for subj, objects in CAN_MEASURE_OBJECTS.items():
                        for obj in objects:
                            for loc in OBJECT_LOCATIONS.get(obj, []):
                                records.append(_rec(
                                    [det1, subj, "can", "measure", det2, obj, "in", det3, loc, "."],
                                    "stmt_measure_in",
                                    version,
                                    3,
                                ))
                    for subj, objects in SHOWS_OBJECTS.items():
                        for obj in objects:
                            for loc in OBJECT_LOCATIONS.get(obj, []):
                                records.append(_rec(
                                    [det1, subj, "shows", det2, obj, "in", det3, loc, "."],
                                    "stmt_shows_in",
                                    version,
                                    3,
                                ))
                    for head, tails in OF_HEAD_TO_TAIL.items():
                        for tail in tails:
                            for loc in OBJECT_LOCATIONS.get(head, []):
                                records.append(_rec(
                                    ["what", "is", det1, head, "of", det2, tail, "in", det3, loc, "?"],
                                    "q_what_of_in",
                                    version,
                                    3,
                                ))

    return _unique_records(records)


@lru_cache(maxsize=None)
def valid_sentence_texts(grammar_version: str) -> frozenset[str]:
    return frozenset(record.text for record in enumerate_sentence_records(grammar_version))


def is_valid_sentence(tokens: Sequence[str], grammar_version: str) -> bool:
    return sentence_text(tokens) in valid_sentence_texts(grammar_version)


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

    excluded_from_coverage = {"<BOS>"}
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


def _records_from_payload(items: Sequence[Mapping[str, object]]) -> List[SentenceRecord]:
    return [SentenceRecord.from_dict(item) for item in items]


def save_sentence_pool_bundle(path: Path, bundle: Mapping[str, object]) -> None:
    path.write_text(json.dumps(bundle, indent=2))


def load_sentence_pool_bundle(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def save_sentence_records(path: Path, records: Sequence[SentenceRecord]) -> None:
    payload = [record.to_dict() for record in records]
    path.write_text(json.dumps(payload, indent=2))


def load_sentence_records(path: Path) -> List[SentenceRecord]:
    payload = json.loads(path.read_text())
    return _records_from_payload(payload)


def write_pool_bundle_files(out_dir: Path, bundle: Mapping[str, object]) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "sentence_pools.json"
    save_sentence_pool_bundle(bundle_path, bundle)
    train_path = out_dir / "train_pool.json"
    dev_path = out_dir / "dev_novel_pool.json"
    test_path = out_dir / "test_novel_pool.json"
    meta_path = out_dir / "pool_meta.json"
    train_records = _records_from_payload(bundle["train_pool"])
    dev_records = _records_from_payload(bundle["dev_novel_pool"])
    test_records = _records_from_payload(bundle["test_novel_pool"])
    save_sentence_records(train_path, train_records)
    save_sentence_records(dev_path, dev_records)
    save_sentence_records(test_path, test_records)
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
    return {
        "bundle": bundle_path,
        "train": train_path,
        "dev": dev_path,
        "test": test_path,
        "meta": meta_path,
    }


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
