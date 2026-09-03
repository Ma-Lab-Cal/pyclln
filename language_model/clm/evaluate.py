from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

from clm.grammar import (
    START_TOKENS,
    TERMINAL_PUNCT,
    SentenceRecord,
    is_valid_sentence,
    records_to_text_set,
    sentence_text,
)


@dataclass
class GenerationCase:
    start_token: str
    generated_tokens: List[str]
    label: str
    reason: str

    @property
    def text(self) -> str:
        return sentence_text(self.generated_tokens)

    @property
    def completed(self) -> bool:
        return bool(self.generated_tokens) and self.generated_tokens[-1] in TERMINAL_PUNCT


def save_report(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2))


def forced_start_schedule(num_samples: int, start_tokens: Sequence[str] = START_TOKENS) -> List[str]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    starts = list(start_tokens)
    return [starts[i % len(starts)] for i in range(num_samples)]


def _pool_text_set(
    pool: Sequence[SentenceRecord] | Sequence[Sequence[str]] | Sequence[str] | set[str] | frozenset[str] | None,
) -> set[str]:
    if pool is None:
        return set()
    if isinstance(pool, (set, frozenset)):
        return {str(item) for item in pool}
    items = list(pool)
    if not items:
        return set()
    first = items[0]
    if isinstance(first, SentenceRecord):
        return records_to_text_set(items)  # type: ignore[arg-type]
    if isinstance(first, str):
        return {str(item) for item in items}
    return {sentence_text(tokens) for tokens in items}  # type: ignore[arg-type]


def classify_generation(
    generated_tokens: Sequence[str],
    *,
    grammar_version: str,
    train_pool_texts: Sequence[str] | set[str] | frozenset[str],
    reference_pool_texts: Sequence[str] | set[str] | frozenset[str] | None = None,
) -> tuple[str, str]:
    tokens = [str(tok) for tok in generated_tokens]
    text = sentence_text(tokens)
    train_set = set(train_pool_texts)
    reference_set = _pool_text_set(reference_pool_texts)
    reference_mode = reference_pool_texts is not None

    if not tokens:
        return "invalid", "empty"
    if tokens[-1] not in TERMINAL_PUNCT:
        return "invalid", "not_terminated"
    if not is_valid_sentence(tokens, grammar_version):
        return "invalid", "not_in_grammar"
    if text in train_set:
        return "valid_seen_train", "train_pool_member"
    if not reference_mode:
        return "valid_novel", "heldout_valid"
    if text in reference_set:
        return "valid_reference", "reference_pool_member"
    return "valid_other_heldout", "valid_but_not_in_reference_pool"


def evaluate_generated_sentences(
    generated_sequences: Sequence[Sequence[str]],
    *,
    grammar_version: str,
    train_pool_texts: Sequence[str] | set[str] | frozenset[str],
    reference_pool_texts: Sequence[str] | set[str] | frozenset[str] | None = None,
    reference_pool_name: str = "heldout",
    start_tokens: Sequence[str] | None = None,
) -> Dict[str, object]:
    reference_set = _pool_text_set(reference_pool_texts)
    reference_mode = reference_pool_texts is not None

    if start_tokens is None:
        starts = [""] * len(generated_sequences)
    else:
        starts = [str(tok) for tok in start_tokens]
        if len(starts) != len(generated_sequences):
            raise ValueError("start token list must align with generated sequences length")

    cases: List[GenerationCase] = []
    per_start: Dict[str, Counter[str]] = defaultdict(Counter)
    unique_novel = set()
    unique_valid = set()
    unique_heldout = set()
    total_length = 0
    completed_count = 0

    for start, seq in zip(starts, generated_sequences):
        tokens = [str(tok) for tok in seq]
        label, reason = classify_generation(
            tokens,
            grammar_version=grammar_version,
            train_pool_texts=train_pool_texts,
            reference_pool_texts=reference_set if reference_mode else None,
        )
        case = GenerationCase(start_token=start, generated_tokens=tokens, label=label, reason=reason)
        cases.append(case)
        per_start[start][label] += 1
        per_start[start][reason] += 1
        total_length += len(tokens)
        completed_count += int(case.completed)
        if label in {"valid_seen_train", "valid_novel", "valid_reference", "valid_other_heldout"}:
            unique_valid.add(case.text)
        if label == "valid_novel":
            unique_novel.add(case.text)
        if label == "valid_reference":
            unique_novel.add(case.text)
            unique_heldout.add(case.text)
        if label == "valid_other_heldout":
            unique_heldout.add(case.text)

    label_counts = Counter(case.label for case in cases)
    reason_counts = Counter(case.reason for case in cases)
    n = max(1, len(cases))
    per_start_summary = {}
    for start, counter in sorted(per_start.items()):
        total = sum(counter[label] for label in {"valid_seen_train", "valid_novel", "valid_reference", "valid_other_heldout", "invalid"})
        total = max(total, 1)
        start_summary = {
            "count": int(total),
            "valid_seen_train": int(counter.get("valid_seen_train", 0)),
            "invalid": int(counter.get("invalid", 0)),
            "valid_rate": float(
                (
                    counter.get("valid_seen_train", 0)
                    + counter.get("valid_novel", 0)
                    + counter.get("valid_reference", 0)
                    + counter.get("valid_other_heldout", 0)
                )
                / total
            ),
        }
        if reference_mode:
            start_summary["valid_reference"] = int(counter.get("valid_reference", 0))
            start_summary["valid_other_heldout"] = int(counter.get("valid_other_heldout", 0))
            start_summary["reference_valid_rate"] = float(counter.get("valid_reference", 0) / total)
            start_summary[f"{reference_pool_name}_valid_rate"] = float(counter.get("valid_reference", 0) / total)
            start_summary["valid_other_heldout_rate"] = float(counter.get("valid_other_heldout", 0) / total)
            start_summary["heldout_valid_rate"] = float(
                (counter.get("valid_reference", 0) + counter.get("valid_other_heldout", 0)) / total
            )
        else:
            start_summary["valid_novel"] = int(counter.get("valid_novel", 0))
            start_summary["valid_novel_rate"] = float(counter.get("valid_novel", 0) / total)
        per_start_summary[start] = start_summary
    payload = {
        "num_samples": int(len(cases)),
        "grammar_version": grammar_version,
        "label_counts": dict(label_counts),
        "reason_counts": dict(reason_counts),
        "valid_rate": float(
            (
                label_counts.get("valid_seen_train", 0)
                + label_counts.get("valid_novel", 0)
                + label_counts.get("valid_reference", 0)
                + label_counts.get("valid_other_heldout", 0)
            )
            / n
        ),
        "completion_rate": float(completed_count / n),
        "mean_generation_length": float(total_length / n),
        "unique_valid_count": int(len(unique_valid)),
        "per_start_token": per_start_summary,
        "cases": [
            {
                "start_token": case.start_token,
                "generated_text": case.text,
                "generated_tokens": case.generated_tokens,
                "label": case.label,
                "reason": case.reason,
            }
            for case in cases
        ],
    }
    if reference_mode:
        reference_rate = float(label_counts.get("valid_reference", 0) / n)
        heldout_count = label_counts.get("valid_reference", 0) + label_counts.get("valid_other_heldout", 0)
        heldout_rate = float(heldout_count / n)
        payload["reference_pool_name"] = reference_pool_name
        payload["reference_pool_size"] = int(len(reference_set))
        payload["reference_valid_rate"] = reference_rate
        payload[f"{reference_pool_name}_valid_rate"] = reference_rate
        payload["valid_other_heldout_rate"] = float(label_counts.get("valid_other_heldout", 0) / n)
        payload["unique_reference_valid_count"] = int(len(unique_novel))
        payload[f"unique_{reference_pool_name}_valid_count"] = int(len(unique_novel))
        payload["heldout_valid_rate"] = heldout_rate
        payload["train_excluded_valid_rate"] = heldout_rate
        payload["unique_heldout_valid_count"] = int(len(unique_heldout))
        payload["unique_train_excluded_valid_count"] = int(len(unique_heldout))
    else:
        payload["valid_novel_rate"] = float(label_counts.get("valid_novel", 0) / n)
        payload["heldout_valid_rate"] = float(label_counts.get("valid_novel", 0) / n)
        payload["train_excluded_valid_rate"] = float(label_counts.get("valid_novel", 0) / n)
        payload["unique_valid_novel_count"] = int(len(unique_novel))
        payload["unique_heldout_valid_count"] = int(len(unique_novel))
        payload["unique_train_excluded_valid_count"] = int(len(unique_novel))
    return payload


def run_generation_benchmark(
    *,
    generate_fn: Callable[[str], Sequence[str]],
    grammar_version: str,
    train_pool: Sequence[SentenceRecord] | Sequence[Sequence[str]],
    reference_pool: Sequence[SentenceRecord] | Sequence[Sequence[str]] | None = None,
    reference_pool_name: str = "heldout",
    num_samples: int,
    start_tokens: Sequence[str] = START_TOKENS,
) -> Dict[str, object]:
    train_pool_texts = _pool_text_set(train_pool)
    reference_pool_texts = _pool_text_set(reference_pool)

    schedule = forced_start_schedule(num_samples, start_tokens=start_tokens)
    sequences: List[List[str]] = []
    for start in schedule:
        try:
            sequences.append([str(tok) for tok in generate_fn(start)])
        except Exception as exc:
            sequences.append([f"<error:{type(exc).__name__}>"])

    report = evaluate_generated_sentences(
        sequences,
        grammar_version=grammar_version,
        train_pool_texts=train_pool_texts,
        reference_pool_texts=reference_pool_texts if reference_pool is not None else None,
        reference_pool_name=reference_pool_name,
        start_tokens=schedule,
    )
    report["forced_start_schedule"] = schedule
    return report


def benchmark_pass(report: Mapping[str, object], *, valid_rate: float, novel_rate: float) -> bool:
    observed_novel = float(
        report.get(
            "heldout_valid_rate",
            report.get("train_excluded_valid_rate", report.get("reference_valid_rate", report.get("valid_novel_rate", 0.0))),
        )
    )
    return float(report.get("valid_rate", 0.0)) >= float(valid_rate) and observed_novel >= float(novel_rate)


def report_score(report: Mapping[str, object]) -> float:
    novel_rate = float(
        report.get(
            "heldout_valid_rate",
            report.get("train_excluded_valid_rate", report.get("reference_valid_rate", report.get("valid_novel_rate", 0.0))),
        )
    )
    unique_novel = float(
        report.get(
            "unique_heldout_valid_count",
            report.get("unique_train_excluded_valid_count", report.get("unique_reference_valid_count", report.get("unique_valid_novel_count", 0))),
        )
    )
    return 1000.0 * float(report.get("valid_rate", 0.0)) + 100.0 * novel_rate + unique_novel
