from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.decomposition import FactorAnalysis, NMF, PCA

from clm.grammar_embed import (
    CAN_MEASURE_OBJECTS,
    CONTEXT_LEN,
    GRAMMAR_VERSIONS,
    HEAD_IN_LOCS,
    ID_TO_TOKEN,
    IN_OBJECTS,
    IS_ADJ_HEADS,
    NOUN_CLASSES,
    OBJECT_LOCATIONS,
    OF_HEAD_TO_TAIL,
    SHOWS_OBJECTS,
    START_TOKENS,
    TOKEN_TO_ID,
    VOCAB,
    WHY_HEADS,
    enumerate_sentence_records,
)

HAND15_DIMENSION_NAMES = [
    "bos",
    "terminal",
    "determiner",
    "question_word",
    "linker",
    "copula",
    "have",
    "modal",
    "lexical_verb",
    "noun",
    "adjective",
    "c1",
    "c2",
    "c3",
    "c4",
]

HAND15_TOKEN_EMBED: Dict[str, List[float]] = {
    "<BOS>":         [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.05],
    ".":             [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 0.10, 0.06, 0.32],
    "?":             [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.12, 0.88, 0.68],
    "the":           [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.12, 0.12, 0.12, 0.35],
    "a":             [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.14, 0.14, 0.12, 0.65],
    "what":          [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.18, 0.85, 0.35],
    "why":           [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.17, 0.20, 0.88, 0.65],
    "in":            [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.22, 0.78, 0.20, 0.35],
    "of":            [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.84, 0.24, 0.65],
    "is":            [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.55, 0.18, 0.88, 0.50],
    "has":           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.56, 0.50, 0.28, 0.48],
    "can":           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.58, 0.80, 0.22, 0.46],
    "measure":       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.84, 0.82, 0.24, 0.33],
    "shows":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.78, 0.82, 0.62, 0.74],
    "electron":      [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.82, 0.35, 0.15, 0.25],
    "photon":        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.78, 0.45, 0.15, 0.28],
    "atom":          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.75, 0.30, 0.25, 0.35],
    "qubit":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.70, 0.40, 0.55, 0.45],
    "spin":          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.25, 0.92, 0.20, 0.40],
    "phase":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.30, 0.88, 0.25, 0.45],
    "energy":        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.35, 0.78, 0.30, 0.55],
    "basis":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.25, 0.72, 0.35, 0.70],
    "wave":          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.45, 0.55, 0.35, 0.50],
    "field":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.40, 0.50, 0.30, 0.60],
    "state":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.54, 0.58, 0.95, 0.60],
    "system":        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.66, 0.50, 0.78, 0.66],
    "detector":      [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.70, 0.32, 0.24, 0.80],
    "measurement":   [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.48, 0.46, 0.34, 0.90],
    "superposition": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.35, 0.70, 0.75, 0.90],
    "entanglement":  [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.30, 0.68, 0.78, 0.95],
    "pure":          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.90, 0.22, 0.90, 0.34],
    "mixed":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.94, 0.26, 0.86, 0.68],
}


@dataclass
class EmbeddingManifest:
    name: str
    kind: str
    vocab: List[str]
    matrix: List[List[float]]
    dim: int
    value_range: List[float]
    derivation: str
    feature_basis: str
    reducer: str
    grammar_version: str | None = None
    dimension_names: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "vocab": self.vocab,
            "matrix": self.matrix,
            "dim": self.dim,
            "value_range": self.value_range,
            "derivation": self.derivation,
            "feature_basis": self.feature_basis,
            "reducer": self.reducer,
            "grammar_version": self.grammar_version,
            "dimension_names": self.dimension_names,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EmbeddingManifest":
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            vocab=[str(tok) for tok in payload["vocab"]],
            matrix=[[float(v) for v in row] for row in payload["matrix"]],
            dim=int(payload["dim"]),
            value_range=[float(v) for v in payload["value_range"]],
            derivation=str(payload["derivation"]),
            feature_basis=str(payload["feature_basis"]),
            reducer=str(payload["reducer"]),
            grammar_version=None if payload.get("grammar_version") is None else str(payload["grammar_version"]),
            dimension_names=[str(v) for v in payload.get("dimension_names", [])],
            metadata=dict(payload.get("metadata", {})),
        )


def save_embedding_manifest(path: Path, manifest: EmbeddingManifest) -> None:
    path.write_text(json.dumps(manifest.to_dict(), indent=2))


def load_embedding_manifest(path: Path) -> EmbeddingManifest:
    return EmbeddingManifest.from_dict(json.loads(path.read_text()))


def manifest_to_matrix(manifest: EmbeddingManifest) -> np.ndarray:
    matrix = np.asarray(manifest.matrix, dtype=float)
    if matrix.shape != (len(manifest.vocab), manifest.dim):
        raise ValueError(f"manifest matrix shape {matrix.shape} does not match vocab/dim")
    return matrix


def manifest_to_token_map(manifest: EmbeddingManifest) -> Dict[str, List[float]]:
    return {tok: [float(v) for v in row] for tok, row in zip(manifest.vocab, manifest.matrix)}


def encode_context_ids(ctx_ids: Sequence[int], manifest: EmbeddingManifest) -> np.ndarray:
    token_map = manifest_to_token_map(manifest)
    values: List[float] = []
    for tid in ctx_ids:
        values.extend(token_map[ID_TO_TOKEN[int(tid)]])
    return np.asarray(values, dtype=float)


def _dimwise_minmax(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    lo = np.min(matrix, axis=0, keepdims=True)
    hi = np.max(matrix, axis=0, keepdims=True)
    span = np.where((hi - lo) <= 1e-12, 1.0, hi - lo)
    scaled = (matrix - lo) / span
    scaled[:, (hi - lo).reshape(-1) <= 1e-12] = 0.0
    return scaled


def _matrix_to_manifest(
    *,
    name: str,
    kind: str,
    matrix: np.ndarray,
    derivation: str,
    feature_basis: str,
    reducer: str,
    grammar_version: str | None,
    dimension_names: Sequence[str],
    metadata: Mapping[str, object] | None = None,
) -> EmbeddingManifest:
    matrix = _dimwise_minmax(np.asarray(matrix, dtype=float))
    return EmbeddingManifest(
        name=name,
        kind=kind,
        vocab=list(VOCAB),
        matrix=[[float(v) for v in row] for row in matrix.tolist()],
        dim=int(matrix.shape[1]),
        value_range=[0.0, 1.0],
        derivation=derivation,
        feature_basis=feature_basis,
        reducer=reducer,
        grammar_version=grammar_version,
        dimension_names=[str(v) for v in dimension_names],
        metadata=dict(metadata or {}),
    )


def _hand15_matrix() -> np.ndarray:
    return np.asarray([HAND15_TOKEN_EMBED[token] for token in VOCAB], dtype=float)


def build_hand15_manifest() -> EmbeddingManifest:
    return _matrix_to_manifest(
        name="hand15_formalized",
        kind="baseline",
        matrix=_hand15_matrix(),
        derivation="Exact export of the existing 15D handcrafted QM embedding.",
        feature_basis="handcrafted_15d",
        reducer="none",
        grammar_version=None,
        dimension_names=HAND15_DIMENSION_NAMES,
        metadata={
            "source": "build_embeddings.py",
            "knowledge_source": "handcrafted_prior",
            "strict_methodology_ok": True,
        },
    )


def _bits5_matrix() -> np.ndarray:
    rows: List[List[float]] = []
    for idx in range(len(VOCAB)):
        rows.append([float(ch) for ch in f"{idx:05b}"])
    return np.asarray(rows, dtype=float)


def build_bits5_manifest() -> EmbeddingManifest:
    return _matrix_to_manifest(
        name="bits5_hamming",
        kind="baseline",
        matrix=_bits5_matrix(),
        derivation="5-bit binary code from the canonical QM32 token index.",
        feature_basis="binary_token_id",
        reducer="none",
        grammar_version=None,
        dimension_names=[f"bit{i}" for i in range(5)],
        metadata={
            "coding": "canonical_index_binary",
            "knowledge_source": "token_index_code",
            "strict_methodology_ok": True,
        },
    )


def _concept_feature_matrix() -> Tuple[np.ndarray, List[str]]:
    noun_classes = sorted(set(NOUN_CLASSES.values()))
    dim_names = [
        "bos",
        "terminal",
        "question_punct",
        "determiner",
        "question_word",
        "link_in",
        "link_of",
        "copula_is",
        "verb_has",
        "modal_can",
        "verb_measure",
        "verb_shows",
        "noun",
        "adjective",
        "can_start",
        "can_take_pred_adj",
        "of_head",
        "of_tail",
        "in_subject",
        "in_location",
    ] + [f"class_{cls_name}" for cls_name in noun_classes]

    matrix = np.zeros((len(VOCAB), len(dim_names)), dtype=float)
    class_offset = len(dim_names) - len(noun_classes)

    of_tail_set = {tail for tails in OF_HEAD_TO_TAIL.values() for tail in tails}
    in_location_set = {loc for locs in IN_OBJECTS.values() for loc in locs}

    for token, idx in TOKEN_TO_ID.items():
        matrix[idx, 0] = float(token == "<BOS>")
        matrix[idx, 1] = float(token in {".", "?"})
        matrix[idx, 2] = float(token == "?")
        matrix[idx, 3] = float(token in {"the", "a"})
        matrix[idx, 4] = float(token in {"what", "why"})
        matrix[idx, 5] = float(token == "in")
        matrix[idx, 6] = float(token == "of")
        matrix[idx, 7] = float(token == "is")
        matrix[idx, 8] = float(token == "has")
        matrix[idx, 9] = float(token == "can")
        matrix[idx, 10] = float(token == "measure")
        matrix[idx, 11] = float(token == "shows")
        matrix[idx, 12] = float(token in NOUN_CLASSES)
        matrix[idx, 13] = float(token in {"pure", "mixed"})
        matrix[idx, 14] = float(token in START_TOKENS)
        matrix[idx, 15] = float(token in IS_ADJ_HEADS)
        matrix[idx, 16] = float(token in OF_HEAD_TO_TAIL)
        matrix[idx, 17] = float(token in of_tail_set)
        matrix[idx, 18] = float(token in IN_OBJECTS)
        matrix[idx, 19] = float(token in in_location_set)

        if token in NOUN_CLASSES:
            cls_name = NOUN_CLASSES[token]
            matrix[idx, class_offset + noun_classes.index(cls_name)] = 1.0

    return matrix, dim_names


def _transition_feature_matrix(grammar_version: str) -> Tuple[np.ndarray, List[str]]:
    if grammar_version not in GRAMMAR_VERSIONS:
        raise ValueError(f"unknown grammar version: {grammar_version}")

    records = enumerate_sentence_records(grammar_version)
    family_names = sorted({record.family for record in records})
    family_index = {name: idx for idx, name in enumerate(family_names)}

    prev_counts = np.zeros((len(VOCAB), len(VOCAB)), dtype=float)
    next_counts = np.zeros((len(VOCAB), len(VOCAB)), dtype=float)
    family_counts = np.zeros((len(VOCAB), len(family_names)), dtype=float)
    stats = np.zeros((len(VOCAB), 4), dtype=float)
    # stats columns: occurrence_count, start_count, end_count, mean_position_numerator

    bos_id = TOKEN_TO_ID["<BOS>"]
    for record in records:
        tokens = list(record.tokens)
        for pos, token in enumerate(tokens):
            tid = TOKEN_TO_ID[token]
            prev_tok = bos_id if pos == 0 else TOKEN_TO_ID[tokens[pos - 1]]
            next_tok = bos_id if pos == len(tokens) - 1 else TOKEN_TO_ID[tokens[pos + 1]]
            prev_counts[tid, prev_tok] += 1.0
            next_counts[tid, next_tok] += 1.0
            family_counts[tid, family_index[record.family]] += 1.0
            stats[tid, 0] += 1.0
            stats[tid, 1] += float(pos == 0)
            stats[tid, 2] += float(pos == len(tokens) - 1)
            if len(tokens) > 1:
                stats[tid, 3] += float(pos) / float(len(tokens) - 1)

    occ = np.where(stats[:, [0]] <= 0.0, 1.0, stats[:, [0]])
    prev_norm = prev_counts / occ
    next_norm = next_counts / occ
    family_norm = family_counts / occ
    stat_norm = np.concatenate(
        [
            occ / max(1.0, float(np.max(occ))),
            stats[:, [1]] / occ,
            stats[:, [2]] / occ,
            stats[:, [3]] / occ,
        ],
        axis=1,
    )

    dim_names = (
        [f"prev_{tok}" for tok in VOCAB]
        + [f"next_{tok}" for tok in VOCAB]
        + [f"family_{name}" for name in family_names]
        + ["occurrence_rate", "start_rate", "end_rate", "mean_rel_position"]
    )
    matrix = np.concatenate([prev_norm, next_norm, family_norm, stat_norm], axis=1)
    return matrix, dim_names


def _hybrid_feature_matrix(grammar_version: str) -> Tuple[np.ndarray, List[str]]:
    concept, concept_names = _concept_feature_matrix()
    transition, transition_names = _transition_feature_matrix(grammar_version)
    matrix = np.concatenate([concept, transition], axis=1)
    names = concept_names + transition_names
    return matrix, names


def _pretrained_best_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    ranking = repo_root / "linear_net" / "embedding_tests" / "qm_embedding_scibert_reducer_sweep_20260608" / "best_overall_sorted.csv"
    default_path = repo_root / "linear_net" / "embedding_tests" / "qm_embedding_scibert_reducer_sweep_20260608" / "embeddings" / "scibert-scivocab__bare__N16_factor_analysis_circuit.npy"
    if not ranking.exists():
        return default_path

    try:
        with ranking.open(newline="") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        model = row["model"]
        reducer = row["reducer"]
        dim = int(row["N"])
        base_dir = ranking.parent / "embeddings"
        candidate = base_dir / f"{model}__bare__N{dim:02d}_{reducer}_circuit.npy"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    return default_path


def build_best_pretrained_manifest() -> EmbeddingManifest:
    path = _pretrained_best_path()
    matrix = np.load(path)
    if matrix.shape[0] != len(VOCAB):
        raise ValueError(f"pretrained baseline row count {matrix.shape[0]} does not match vocab size {len(VOCAB)}")
    return _matrix_to_manifest(
        name="best_pretrained_baseline",
        kind="baseline",
        matrix=np.asarray(matrix, dtype=float),
        derivation="Loaded from the best existing reduced SciBERT baseline sweep artifact.",
        feature_basis="existing_scibert_reduction",
        reducer="artifact",
        grammar_version=None,
        dimension_names=[f"d{i + 1}" for i in range(int(matrix.shape[1]))],
        metadata={
            "source_path": str(path),
            "knowledge_source": "external_pretrained_reduction",
            "strict_methodology_ok": True,
        },
    )


def _reduce_features(
    features: np.ndarray,
    raw_names: Sequence[str],
    *,
    dim: int,
    reducer: str,
) -> Tuple[np.ndarray, List[str]]:
    features = np.asarray(features, dtype=float)
    if dim <= 0:
        raise ValueError("dim must be > 0")
    if reducer == "none":
        if features.shape[1] != dim:
            raise ValueError(f"reducer=none requires raw feature width {features.shape[1]} to equal dim {dim}")
        return features.copy(), list(raw_names)

    if dim > min(features.shape[0], features.shape[1]) and reducer in {"pca", "factor_analysis", "nmf"}:
        raise ValueError(f"dim {dim} is too large for reducer {reducer} with features {features.shape}")

    if reducer == "pca":
        model = PCA(n_components=dim, random_state=0)
        reduced = model.fit_transform(features)
        names = [f"pc{i + 1}" for i in range(dim)]
    elif reducer == "factor_analysis":
        model = FactorAnalysis(n_components=dim, random_state=0)
        reduced = model.fit_transform(features)
        names = [f"fa{i + 1}" for i in range(dim)]
    elif reducer == "nmf":
        safe = np.clip(features, 0.0, None)
        model = NMF(n_components=dim, init="nndsvda", random_state=0, max_iter=1000)
        reduced = model.fit_transform(safe)
        names = [f"nmf{i + 1}" for i in range(dim)]
    else:
        raise ValueError(f"unsupported reducer: {reducer}")

    return reduced, names


def build_structured_manifest(
    *,
    family: str,
    grammar_version: str,
    dim: int,
    reducer: str,
) -> EmbeddingManifest:
    if grammar_version not in GRAMMAR_VERSIONS:
        raise ValueError(f"unknown grammar version: {grammar_version}")
    if family == "grammar_concept":
        features, raw_names = _concept_feature_matrix()
        feature_basis = "grammar_concept_features"
    elif family == "grammar_transition":
        features, raw_names = _transition_feature_matrix(grammar_version)
        feature_basis = "grammar_transition_features"
    elif family == "grammar_hybrid":
        features, raw_names = _hybrid_feature_matrix(grammar_version)
        feature_basis = "grammar_hybrid_features"
    elif family == "pretrained_assisted_hybrid":
        hybrid, raw_names = _hybrid_feature_matrix(grammar_version)
        pretrained = manifest_to_matrix(build_best_pretrained_manifest())
        features = np.concatenate([hybrid, pretrained], axis=1)
        raw_names = list(raw_names) + [f"pretrained_{i + 1}" for i in range(pretrained.shape[1])]
        feature_basis = "grammar_hybrid_plus_pretrained"
    else:
        raise ValueError(f"unsupported structured family: {family}")

    reduced, names = _reduce_features(features, raw_names, dim=dim, reducer=reducer)
    return _matrix_to_manifest(
        name=f"{family}_{grammar_version}_{reducer}_d{dim}",
        kind="structured",
        matrix=reduced,
        derivation=f"{family} built from exact {grammar_version} grammar features and compressed with {reducer}.",
        feature_basis=feature_basis,
        reducer=reducer,
        grammar_version=grammar_version,
        dimension_names=names,
        metadata={
            "family": family,
            "raw_feature_dim": int(features.shape[1]),
            "requested_dim": int(dim),
            "knowledge_source": (
                "grammar_rule_prior"
                if family == "grammar_concept"
                else "full_grammar_sentence_statistics"
            ),
            "strict_methodology_ok": bool(family == "grammar_concept"),
        },
    )


def build_embedding_manifest(
    embedding_name: str,
    *,
    grammar_version: str = "grammar_v1_core",
    dim: int | None = None,
    reducer: str = "nmf",
) -> EmbeddingManifest:
    if embedding_name == "hand15_formalized":
        return build_hand15_manifest()
    if embedding_name == "bits5_hamming":
        return build_bits5_manifest()
    if embedding_name == "best_pretrained_baseline":
        return build_best_pretrained_manifest()
    if embedding_name in {"grammar_concept", "grammar_transition", "grammar_hybrid", "pretrained_assisted_hybrid"}:
        if dim is None:
            raise ValueError(f"embedding {embedding_name} requires dim")
        return build_structured_manifest(
            family=embedding_name,
            grammar_version=grammar_version,
            dim=int(dim),
            reducer=str(reducer),
        )
    raise ValueError(f"unknown embedding name: {embedding_name}")
