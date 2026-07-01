"""Utilities for representing and scoring KG candidate paths.

The helpers in this file are intentionally lightweight and operate on the
per-sample SubgraphRAG data dictionaries produced by ``RetrieverDataset``.
They keep all identifiers local to a sample so that path traces can be saved
and audited later.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math


@dataclass(frozen=True)
class PathTriple:
    """A single KG triple inside a candidate path."""

    local_triple_id: int
    h_id: int
    r_id: int
    t_id: int
    h: str
    r: str
    t: str
    triple_logit: float
    triple_score: float
    edge_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePath:
    """A path-level evidence candidate for select-then-generation."""

    path_index: int
    method: str
    path_score: float
    path_cost: float
    path_length: int
    source_entity_id: int
    source_entity: str
    terminal_entity_id: int
    terminal_entity: str
    node_ids: List[int]
    node_names: List[str]
    triples: List[PathTriple]

    @property
    def signature(self) -> Tuple[int, ...]:
        return tuple(triple.local_triple_id for triple in self.triples)

    @property
    def relation_signature(self) -> Tuple[str, ...]:
        return tuple(triple.r for triple in self.triples)

    def linearized(self) -> str:
        return " -> ".join(
            f"({triple.h}, {triple.r}, {triple.t})" for triple in self.triples
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["signature"] = list(self.signature)
        data["relation_signature"] = list(self.relation_signature)
        data["linearized"] = self.linearized()
        return data


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalize_scores(values: Sequence[float]) -> List[float]:
    """Min-max normalize scores; return 0.5 for ties."""

    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def aggregate_path_score(
    triple_scores: Sequence[float],
    agg: str = "mean_minus_len",
    length_penalty: float = 0.05,
) -> float:
    """Aggregate per-triple scores into a path-level score."""

    if not triple_scores:
        return float("-inf")
    agg = agg.lower()
    length = len(triple_scores)
    if agg == "mean":
        return float(sum(triple_scores) / length)
    if agg == "sum":
        return float(sum(triple_scores))
    if agg == "min":
        return float(min(triple_scores))
    if agg == "sum_sqrt_len":
        return float(sum(triple_scores) / math.sqrt(length))
    if agg == "mean_minus_len":
        return float(sum(triple_scores) / length - length_penalty * length)
    raise ValueError(f"Unsupported path score aggregation: {agg}")


def deduplicate_paths(paths: Iterable[CandidatePath]) -> List[CandidatePath]:
    """Deduplicate by triple-id signature while preserving score order."""

    seen = set()
    unique: List[CandidatePath] = []
    for path in paths:
        signature = path.signature
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(path)
    return unique


def reindex_paths(paths: Sequence[CandidatePath]) -> List[CandidatePath]:
    for idx, path in enumerate(paths):
        path.path_index = idx
    return list(paths)
