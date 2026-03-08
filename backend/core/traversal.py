"""
traversal.py — Multi-hop KG traversal engine.

Executes the traversal plan produced by the decomposer,
scores each path, and returns the top-k ranked paths.
"""
from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .kg_adapter.base import KGAdapterBase
from .entity_linker import EntityLinker, EntityCandidate
from .relation_linker import RelationLinker
from .decomposer import DecompositionPlan, SubQuestion

_WEIGHTS_FILE = Path(__file__).resolve().parents[1] / "configs" / "weights.yaml"


def _load_weights() -> Dict[str, Any]:
    if _WEIGHTS_FILE.exists():
        with open(_WEIGHTS_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


@dataclass
class Triple:
    subj: str
    rel: str
    obj: str
    time: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "subj": self.subj,
            "rel": self.rel,
            "obj": self.obj,
            "time": self.time,
            "source": self.source,
        }


@dataclass
class ReasoningPath:
    path_id: str
    hops: int
    triples: List[Triple]
    node_names: List[str]
    score: float
    entity_link_score: float = 0.0
    relation_link_score: float = 0.0
    temporal_match_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "path_id": self.path_id,
            "hops": self.hops,
            "triples": [t.to_dict() for t in self.triples],
            "score": round(self.score, 4),
        }


@dataclass
class TraversalResult:
    paths: List[ReasoningPath]
    answer_candidates: List[str]       # terminal node names from top path
    execution_ms: float
    query_plan: Dict[str, Any]


class KGTraversalEngine:
    """
    Multi-hop KG traversal with transparent scoring.

    Scoring formula (configurable via weights.yaml):
      score = w_ent * entity_link_score
            + w_rel * relation_link_score
            + w_tmp * temporal_match_score
            - w_pen * path_length_penalty
    """

    def __init__(self, adapter: KGAdapterBase):
        self._adapter = adapter
        self._linker = EntityLinker(adapter)
        self._rel_linker = RelationLinker()
        cfg = _load_weights()
        sc = cfg.get("scoring", {})
        self._w_ent = sc.get("entity_link_weight", 0.35)
        self._w_rel = sc.get("relation_link_weight", 0.25)
        self._w_tmp = sc.get("temporal_match_weight", 0.20)
        self._w_pen = sc.get("path_length_penalty_weight", 0.20)
        self._pen_per_hop = cfg.get("path_length_penalty_per_hop", 0.08)
        self._max_hops = cfg.get("max_hops", 4)
        self._top_k = cfg.get("top_k_paths", 3)

    def execute(
        self,
        plan: DecompositionPlan,
        query: str,
    ) -> TraversalResult:
        t0 = time.perf_counter()
        all_paths: List[ReasoningPath] = []
        answer_candidates: List[str] = []

        # Build query keyword set for relevance scoring
        query_keywords = set(
            w.lower() for w in re.sub(r"[^\w\s]", " ", query).split()
            if len(w) > 3
        )

        # Process each sub-question sequentially (chain)
        carry_node_ids: List[str] = []   # carry forward from previous hop answer

        for sq in plan.subquestions:
            paths, candidates = self._execute_subquestion(
                sq, plan.constraints, carry_node_ids, query_keywords
            )
            all_paths.extend(paths)
            # Answer nodes from this step become anchors for the next
            carry_node_ids = [
                node for p in paths[:1] for node in p.node_names[-1:]
            ]
            # Convert names → ids for the next step
            carry_node_ids_resolved: List[str] = []
            for name in carry_node_ids:
                hits = self._adapter.search_nodes(name, top_k=1)
                if hits:
                    carry_node_ids_resolved.append(hits[0]["id"])
            carry_node_ids = carry_node_ids_resolved
            answer_candidates = candidates

        # Deduplicate paths by path_id
        seen_ids: set = set()
        unique_paths: List[ReasoningPath] = []
        for p in all_paths:
            if p.path_id not in seen_ids:
                seen_ids.add(p.path_id)
                unique_paths.append(p)

        # Sort by score descending
        unique_paths.sort(key=lambda p: p.score, reverse=True)
        top_paths = unique_paths[: self._top_k]

        return TraversalResult(
            paths=top_paths,
            answer_candidates=answer_candidates,
            execution_ms=round((time.perf_counter() - t0) * 1000, 2),
            query_plan={
                "question_type": plan.question_type,
                "subquestions": [sq.text for sq in plan.subquestions],
                "constraints": plan.constraints,
            },
        )

    # ── private ──────────────────────────────────────────────────────────────

    def _execute_subquestion(
        self,
        sq: SubQuestion,
        global_constraints: Dict[str, Any],
        carry_node_ids: List[str],
        query_keywords: Optional[set] = None,
    ) -> Tuple[List[ReasoningPath], List[str]]:

        query_keywords = query_keywords or set()

        # Step 1: Entity linking — prefer explicit anchor_entity from decomposer
        entity_candidates = []
        entity_link_score = 0.6

        if sq.anchor_entity:
            # Search the KG specifically for the anchor entity name
            anchor_hits = self._adapter.search_nodes(sq.anchor_entity, top_k=5)
            if anchor_hits:
                from .entity_linker import EntityCandidate
                entity_candidates = [
                    EntityCandidate(
                        id=h["id"],
                        name=h.get("name", h["id"]),
                        label=h.get("label", "?"),
                        score=h.get("score", 0.75),
                        match_type="anchor",
                    )
                    for h in anchor_hits
                ]
                entity_link_score = entity_candidates[0].score if entity_candidates else 0.6

        # Fall back to full entity linker if anchor search found nothing
        if not entity_candidates:
            entity_candidates = self._linker.link(sq.text)
            if entity_candidates:
                entity_link_score = entity_candidates[0].score

        if not entity_candidates and carry_node_ids:
            start_ids = carry_node_ids
        elif entity_candidates:
            start_ids = [c.id for c in entity_candidates[:3]]
        else:
            return [], []

        # Step 2: Relation linking
        rel_result = self._rel_linker.link(sq.text)
        relation_link_score = rel_result.confidence

        # Step 3: Build constraints
        constraints: Dict[str, Any] = dict(global_constraints)
        if sq.rel_hint:
            constraints.setdefault("rel_types", [sq.rel_hint])
        if rel_result.matched_relations:
            constraints.setdefault("rel_types", rel_result.matched_relations)

        time_filter: Optional[Dict] = None
        if global_constraints.get("time_after") or global_constraints.get("time_before"):
            time_filter = {
                "after":  global_constraints.get("time_after"),
                "before": global_constraints.get("time_before"),
            }

        # Step 4a: Target entity IDs for bidirectional search
        end_ids: Optional[List[str]] = None
        if sq.target_entity:
            target_hits = self._adapter.search_nodes(sq.target_entity, top_k=3)
            if target_hits:
                end_ids = [h["id"] for h in target_hits]

        # Step 4b: KG traversal
        raw_paths = self._adapter.find_paths(
            start_node_ids=start_ids,
            end_node_ids=end_ids,
            constraints=constraints,
            max_hops=self._max_hops,
            top_k=self._top_k * 3,   # fetch more, we'll rerank
        )

        # If no multi-hop paths found, fall back to direct neighbor expansion
        if not raw_paths:
            raw_paths = self._expand_neighbors(
                start_ids, constraints.get("rel_types"), time_filter
            )
            # Also try from the target side if we have one
            if not raw_paths and end_ids:
                raw_paths = self._expand_neighbors(end_ids, constraints.get("rel_types"), time_filter)

        # Step 5: Score and wrap — add query relevance bonus
        paths: List[ReasoningPath] = []
        for i, rp in enumerate(raw_paths):
            temporal_match = self._score_temporal(rp, global_constraints)
            hops = rp.get("hops", len(rp.get("triples", [])))
            length_penalty = self._pen_per_hop * max(0, hops - 1)

            # Query relevance: count keyword hits in node names and relation types
            relevance_bonus = self._score_path_relevance(rp, query_keywords)

            score = (
                self._w_ent * entity_link_score
                + self._w_rel * relation_link_score
                + self._w_tmp * temporal_match
                - self._w_pen * length_penalty
                + 0.15 * relevance_bonus  # bonus for keyword-matching paths
            )
            score = max(0.0, min(1.0, score))

            triples = [
                Triple(
                    subj=t.get("subj", ""),
                    rel=t.get("rel", ""),
                    obj=t.get("obj", ""),
                    time=t.get("time"),
                    source=t.get("source"),
                )
                for t in rp.get("triples", [])
            ]
            paths.append(
                ReasoningPath(
                    path_id=rp.get("path_id", f"p{i}"),
                    hops=hops,
                    triples=triples,
                    node_names=rp.get("node_names", []),
                    score=score,
                    entity_link_score=entity_link_score,
                    relation_link_score=relation_link_score,
                    temporal_match_score=temporal_match,
                )
            )

        paths.sort(key=lambda p: p.score, reverse=True)

        # Answer candidates = terminal nodes of top path
        answer_candidates: List[str] = []
        if paths:
            top = paths[0]
            answer_candidates = (
                [top.node_names[-1]] if top.node_names else
                [top.triples[-1].obj] if top.triples else []
            )

        return paths, answer_candidates

    @staticmethod
    def _score_path_relevance(path: Dict[str, Any], query_keywords: set) -> float:
        """
        Score how well a path matches the query keywords.
        Returns 0.0–1.0. Keywords found in node names or relation types score higher.
        """
        if not query_keywords:
            return 0.0

        hits = 0
        total_tokens = 0

        node_names = path.get("node_names", [])
        for name in node_names:
            if not name:
                continue
            tokens = name.lower().split()
            total_tokens += len(tokens)
            for tok in tokens:
                if any(kw in tok or tok in kw for kw in query_keywords):
                    hits += 1

        for triple in path.get("triples", []):
            for field_val in (triple.get("subj", ""), triple.get("obj", ""), triple.get("rel", "")):
                if not field_val:
                    continue
                tokens = field_val.lower().replace("_", " ").split()
                total_tokens += len(tokens)
                for tok in tokens:
                    if any(kw in tok or tok in kw for kw in query_keywords):
                        hits += 1

        if total_tokens == 0:
            return 0.0
        return min(1.0, hits / max(1, len(query_keywords)))

    def _expand_neighbors(
        self,
        node_ids: List[str],
        rel_types: Optional[List[str]],
        time_filter: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        """Single-hop fallback: expand all neighbors and wrap as 1-hop paths."""
        paths = []
        for nid in node_ids:
            neighbors = self._adapter.get_neighbors(
                nid, rel_types=rel_types, time_filter=time_filter
            )
            for i, nb in enumerate(neighbors):
                paths.append({
                    "path_id": f"nb_{nid}_{i}",
                    "hops": 1,
                    "node_ids": [nb.get("from_id", nid), nb.get("to_id", "")],
                    "node_names": [nb.get("from_name", ""), nb.get("to_name", "")],
                    "node_labels": ["?", nb.get("to_label", "?")],
                    "triples": [{
                        "subj": nb.get("from_name", ""),
                        "rel":  nb.get("rel", ""),
                        "obj":  nb.get("to_name", ""),
                        "time": nb.get("t_announce") or nb.get("t_effective"),
                        "source": nb.get("source"),
                    }],
                })
        return paths

    @staticmethod
    def _score_temporal(
        path: Dict[str, Any], constraints: Dict[str, Any]
    ) -> float:
        """Check if any triple's time satisfies the constraint."""
        if not constraints.get("time_after") and not constraints.get("time_before"):
            return 1.0  # no temporal constraint → full score

        triples = path.get("triples", [])
        for t in triples:
            t_val = t.get("time")
            if not t_val:
                continue
            try:
                if constraints.get("time_after") and t_val >= constraints["time_after"]:
                    return 1.0
                if constraints.get("time_before") and t_val <= constraints["time_before"]:
                    return 1.0
            except TypeError:
                pass
        return 0.2  # triples exist but no match
