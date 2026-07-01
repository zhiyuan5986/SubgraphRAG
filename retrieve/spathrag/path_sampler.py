"""S-Path-RAG-inspired path proposal over SubgraphRAG scored triples.

This sampler mirrors the prompt-mode path construction idea in
``ref/spathrag/llm_integration/llm_wrapper.py``: candidate paths are made
explicit as textual KG evidence before a query is answered.  Unlike the
reference code, path scoring here is driven by SubgraphRAG's MLP triple logits.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import random

from .path_utils import (
    CandidatePath,
    PathTriple,
    aggregate_path_score,
    deduplicate_paths,
    normalize_scores,
    reindex_paths,
    sigmoid,
)


class CandidatePathSampler:
    """Build and sample candidate paths from query-conditioned triple scores."""

    def __init__(
        self,
        sample: Dict[str, Any],
        triple_logits: Sequence[float],
        path_score_agg: str = "mean_minus_len",
        length_penalty: float = 0.05,
        score_cost_weight: float = 1.0,
    ):
        self.sample = sample
        self.triple_logits = [float(x) for x in triple_logits]
        self.triple_scores = [sigmoid(x) for x in self.triple_logits]
        self.normalized_scores = normalize_scores(self.triple_logits)
        self.path_score_agg = path_score_agg
        self.length_penalty = length_penalty
        self.score_cost_weight = score_cost_weight

        self.entity_list = sample["text_entity_list"] + sample["non_text_entity_list"]
        self.relation_list = sample["relation_list"]
        self.h_ids = [int(x) for x in sample["h_id_list"]]
        self.r_ids = [int(x) for x in sample["r_id_list"]]
        self.t_ids = [int(x) for x in sample["t_id_list"]]
        self.adj = self._build_adjacency()

    def _build_adjacency(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = defaultdict(list)
        for triple_id, h_id in enumerate(self.h_ids):
            adj[h_id].append(triple_id)
        for h_id in adj:
            adj[h_id].sort(key=lambda tid: self.triple_logits[tid], reverse=True)
        return adj

    def _triple_to_path_triple(self, triple_id: int) -> PathTriple:
        h_id = self.h_ids[triple_id]
        r_id = self.r_ids[triple_id]
        t_id = self.t_ids[triple_id]
        score = self.triple_scores[triple_id]
        edge_cost = 1.0 + self.score_cost_weight * (1.0 - score)
        return PathTriple(
            local_triple_id=triple_id,
            h_id=h_id,
            r_id=r_id,
            t_id=t_id,
            h=self.entity_list[h_id],
            r=self.relation_list[r_id],
            t=self.entity_list[t_id],
            triple_logit=self.triple_logits[triple_id],
            triple_score=score,
            edge_cost=edge_cost,
        )

    def _make_candidate_path(self, triple_ids: Sequence[int], method: str) -> CandidatePath:
        triples = [self._triple_to_path_triple(tid) for tid in triple_ids]
        node_ids = [triples[0].h_id] + [triple.t_id for triple in triples]
        node_names = [self.entity_list[node_id] for node_id in node_ids]
        triple_scores = [triple.triple_score for triple in triples]
        path_score = aggregate_path_score(
            triple_scores,
            agg=self.path_score_agg,
            length_penalty=self.length_penalty,
        )
        path_cost = sum(triple.edge_cost for triple in triples)
        return CandidatePath(
            path_index=-1,
            method=method,
            path_score=path_score,
            path_cost=path_cost,
            path_length=len(triples),
            source_entity_id=node_ids[0],
            source_entity=node_names[0],
            terminal_entity_id=node_ids[-1],
            terminal_entity=node_names[-1],
            node_ids=node_ids,
            node_names=node_names,
            triples=triples,
        )

    def beam_paths(
        self,
        source_ids: Iterable[int],
        max_path_length: int = 4,
        beam_width: int = 16,
        expand_top_k: int = 32,
    ) -> List[CandidatePath]:
        """Query-aware beam expansion from topic entities without answer entities."""

        completed: List[CandidatePath] = []
        beams: List[Tuple[float, List[int], int, Tuple[int, ...]]] = []
        for source_id in source_ids:
            beams.append((0.0, [], int(source_id), (int(source_id),)))

        for _ in range(max_path_length):
            next_beams: List[Tuple[float, List[int], int, Tuple[int, ...]]] = []
            for _, triple_path, last_node, visited_nodes in beams:
                for triple_id in self.adj.get(last_node, [])[:expand_top_k]:
                    next_node = self.t_ids[triple_id]
                    if next_node in visited_nodes:
                        continue
                    new_triple_path = triple_path + [triple_id]
                    path = self._make_candidate_path(new_triple_path, method="beam")
                    completed.append(path)
                    next_beams.append(
                        (
                            path.path_score,
                            new_triple_path,
                            next_node,
                            visited_nodes + (next_node,),
                        )
                    )
            if not next_beams:
                break
            next_beams.sort(key=lambda item: item[0], reverse=True)
            beams = next_beams[:beam_width]
        return completed

    def random_walk_paths(
        self,
        source_ids: Iterable[int],
        num_walks: int = 32,
        walk_length: int = 4,
        restart_prob: float = 0.0,
        seed: int = 0,
    ) -> List[CandidatePath]:
        """Sample bounded random walks to keep some noisy/diverse evidence."""

        rng = random.Random(seed)
        sources = [int(x) for x in source_ids]
        if not sources:
            return []
        paths: List[CandidatePath] = []
        for _ in range(num_walks):
            current = rng.choice(sources)
            visited = {current}
            triple_path: List[int] = []
            for _ in range(walk_length):
                choices = [tid for tid in self.adj.get(current, []) if self.t_ids[tid] not in visited]
                if not choices:
                    break
                if rng.random() < restart_prob and triple_path:
                    current = rng.choice(sources)
                    visited = {current}
                    continue
                # Bias random walks toward higher MLP scores without making them deterministic.
                top_window = choices[: min(len(choices), 64)]
                triple_id = rng.choice(top_window)
                triple_path.append(triple_id)
                current = self.t_ids[triple_id]
                visited.add(current)
                paths.append(self._make_candidate_path(triple_path, method="random_walk"))
        return paths

    def build_path_pool(
        self,
        source_ids: Iterable[int],
        max_path_length: int = 4,
        beam_width: int = 16,
        expand_top_k: int = 32,
        num_random_walks: int = 32,
        random_seed: int = 0,
        pool_size: int = 100,
    ) -> List[CandidatePath]:
        candidates = []
        candidates.extend(
            self.beam_paths(
                source_ids,
                max_path_length=max_path_length,
                beam_width=beam_width,
                expand_top_k=expand_top_k,
            )
        )
        candidates.extend(
            self.random_walk_paths(
                source_ids,
                num_walks=num_random_walks,
                walk_length=max_path_length,
                seed=random_seed,
            )
        )
        candidates.sort(key=lambda path: path.path_score, reverse=True)
        return reindex_paths(deduplicate_paths(candidates)[:pool_size])

    @staticmethod
    def _take_unique(paths: Iterable[CandidatePath], limit: int, used: set) -> List[CandidatePath]:
        selected = []
        for path in paths:
            if path.signature in used:
                continue
            selected.append(path)
            used.add(path.signature)
            if len(selected) >= limit:
                break
        return selected

    def make_path_set(
        self,
        path_pool: Sequence[CandidatePath],
        paths_per_set: int,
        policy: str,
        seed: int = 0,
    ) -> List[CandidatePath]:
        """Construct one candidate path set from a larger pool."""

        if not path_pool:
            return []
        rng = random.Random(seed)
        policy = policy.lower()
        sorted_pool = sorted(path_pool, key=lambda path: path.path_score, reverse=True)
        used = set()
        selected: List[CandidatePath] = []

        if policy == "top_heavy":
            top_n = max(1, int(paths_per_set * 0.75))
            selected.extend(self._take_unique(sorted_pool, top_n, used))
        elif policy == "diverse":
            by_relation = {}
            for path in sorted_pool:
                key = path.relation_signature[:1]
                by_relation.setdefault(key, path)
            selected.extend(self._take_unique(by_relation.values(), max(1, paths_per_set // 2), used))
        elif policy == "noisy_light":
            top_n = max(1, int(paths_per_set * 0.50))
            selected.extend(self._take_unique(sorted_pool, top_n, used))
            tail = sorted_pool[top_n:]
            rng.shuffle(tail)
            selected.extend(self._take_unique(tail, max(1, int(paths_per_set * 0.25)), used))
        else:
            raise ValueError(f"Unsupported path-set policy: {policy}")

        # Fill remaining slots by score order.
        selected.extend(self._take_unique(sorted_pool, paths_per_set - len(selected), used))
        return reindex_paths(selected[:paths_per_set])
