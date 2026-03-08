"""
decomposer.py — Multi-hop question decomposer / planner.

Primary path: Gemini classifies & plans (structured JSON).
Fallback: pure rule-based regex (used when Gemini is unavailable or fails).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .relation_linker import RelationLinker

logger = logging.getLogger(__name__)


# ── Question-type patterns ────────────────────────────────────────────────
_MULTI_HOP_CHAIN_KEYWORDS = [
    r"which\s+\w+\s+benefited\s+from",
    r"companies?\s+(that|which)\s+(were|are|got)\s+(affected|impacted|benefited)",
    r"what\s+(changed|happened)\s+after",
    r"(who|which)\s+(was|were|is|are)\s+(involved|affected)\s+(in|by|after)",
    r"following\s+the\s+\w+\s+(of|by|from)",
    r"as\s+a\s+result\s+of",
    # Explicit "connection / relationship / link between X and Y"
    r"(connection|relationship|link|relation|impact|effect|influence)\s+(between|of|from)",
    r"how\s+(does|did|do|has|have|is|are)\s+.{3,60}\s+(affect|impact|influence|relate\s+to|connect)",
    r"what\s+(is|was|are|were)\s+the\s+(impact|effect|influence|connection|relationship|link|role)",
    r"(why|how)\s+(did|does|has|have|is|are)\s+.{3,40}\s+(cause|lead|contribute|result)",
    r"(impact|effect|influence|role)\s+of\s+.{3,40}\s+on\s+",
]
_MULTI_HOP_INTERSECTION_KEYWORDS = [
    r"(both|all)\s+\w+\s+and\s+\w+",
    r"common\s+(between|across)",
    r"also\s+\w+",
]
_TEMPORAL_KEYWORDS = [
    r"\b(after|before|since|during|in|on|prior to|following)\b.{2,30}\b(\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\b(q[1-4]|quarter|fy|fiscal year|financial year)\b",
]
_SINGLE_HOP_KEYWORDS = [
    r"^who\s+is",
    r"^what\s+is",
    r"^when\s+did",
    r"^which\s+\w+\s+(is|are|was|were)",
    r"ceo\s+of",
    r"founded\s+(by|in)",
    r"headquarters?\s+of",
]


@dataclass
class SubQuestion:
    index: int
    text: str
    anchor_entity: Optional[str]   # entity to start traversal from
    target_entity: Optional[str]   # entity to reach (may be None)
    rel_hint: Optional[str]        # suggested KG relation type


@dataclass
class DecompositionPlan:
    question_type: str              # "single_hop" | "multi_hop_chain" | "multi_hop_intersection" | "temporal"
    original_query: str
    subquestions: List[SubQuestion]
    constraints: Dict[str, Any]     # time_after, time_before, entity_types, rel_types
    is_multi_turn_followup: bool = False


class QuestionDecomposer:
    """
    Classifies a question and generates an ordered traversal plan.
    Pure rule-based (no LLM) for determinism and speed.
    """

    def __init__(self):
        self._rel_linker = RelationLinker()

    def decompose(
        self,
        query: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> DecompositionPlan:
        session_context = session_context or {}
        q = query.strip()

        # Resolve coreferences from session context
        q_resolved = self._resolve_coref(q, session_context)
        is_followup = q_resolved != q

        rel_result = self._rel_linker.link(q_resolved)
        constraints: Dict[str, Any] = {}
        if rel_result.temporal_op:
            constraints[f"time_{rel_result.temporal_op}"] = rel_result.temporal_ref
        if rel_result.matched_relations:
            constraints["rel_types"] = rel_result.matched_relations

        # ── Try Gemini first ────────────────────────────────────────────────
        qtype, subquestions = self._gemini_decompose(q_resolved, session_context, rel_result, constraints)

        return DecompositionPlan(
            question_type=qtype,
            original_query=q,
            subquestions=subquestions,
            constraints=constraints,
            is_multi_turn_followup=is_followup,
        )

    def _gemini_decompose(
        self,
        query: str,
        session_context: Dict,
        rel_result,
        constraints: Dict,
    ):
        """Try Gemini decomposition; fall back to rule-based on failure."""
        try:
            from .gemini_client import decompose as gemini_decompose
            result = gemini_decompose(query, session_context)
            if result and isinstance(result, dict):
                qtype = result.get("question_type", "single_hop")
                if qtype not in ("single_hop", "multi_hop_chain", "multi_hop_intersection", "temporal"):
                    qtype = "single_hop"
                # Merge Gemini constraints (rel_types, time windows)
                if "constraints" in result and isinstance(result["constraints"], dict):
                    for k, v in result["constraints"].items():
                        if v and k not in constraints:
                            constraints[k] = v
                # Build subquestions from Gemini's list
                raw_subqs = result.get("subquestions") or [query]
                # Use anchor_entities from Gemini if provided
                anchor_entities: List[Optional[str]] = result.get("anchor_entities") or []
                last_ents = session_context.get("last_entities") or []
                subquestions = []
                for i, sq in enumerate(raw_subqs):
                    # anchor: Gemini-provided > session context fallback
                    anchor = None
                    if i < len(anchor_entities) and anchor_entities[i]:
                        anchor = anchor_entities[i]
                    elif i == 0 and last_ents:
                        anchor = last_ents[0]
                    # target entity for multi-hop: the OTHER anchor entity
                    target = None
                    if len(anchor_entities) >= 2:
                        other_idx = 1 - i if i < 2 else None
                        if other_idx is not None and other_idx < len(anchor_entities):
                            target = anchor_entities[other_idx]
                    subquestions.append(SubQuestion(
                        index=i,
                        text=sq if isinstance(sq, str) else str(sq),
                        anchor_entity=anchor,
                        target_entity=target,
                        rel_hint=rel_result.matched_relations[min(i, len(rel_result.matched_relations) - 1)]
                               if rel_result.matched_relations else None,
                    ))
                logger.debug("Gemini decompose: type=%s subqs=%d anchors=%s", qtype, len(subquestions), anchor_entities)
                return qtype, subquestions
        except Exception as exc:
            logger.warning("Gemini decompose error: %s — falling back to rules", exc)

        # ── Rule-based fallback ─────────────────────────────────────────────
        qtype = self._classify(query)
        subquestions = self._build_subquestions(query, qtype, rel_result, session_context)
        return qtype, subquestions

    # ── private ──────────────────────────────────────────────────────────────

    def _classify(self, query: str) -> str:
        q = query.lower()

        for pattern in _MULTI_HOP_CHAIN_KEYWORDS:
            if re.search(pattern, q):
                return "multi_hop_chain"

        for pattern in _MULTI_HOP_INTERSECTION_KEYWORDS:
            if re.search(pattern, q):
                return "multi_hop_intersection"

        for pattern in _TEMPORAL_KEYWORDS:
            if re.search(pattern, q):
                return "temporal"

        return "single_hop"

    # ── Entity extraction helpers ────────────────────────────────────────────

    @staticmethod
    def _extract_between_entities(query: str) -> Optional[List[str]]:
        """
        Extract [entity_a, entity_b] from patterns like:
          "connection between <A> and <B>"
          "relationship between <A> and <B>"
          "impact of <A> on <B>"
          "how does <A> affect <B>"
        Returns None when no clear split is found.
        """
        q = query.strip()

        # Pattern 1: "... between <A> and <B>"
        m = re.search(
            r"\b(?:connection|relationship|link|relation|impact|difference)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            q, re.IGNORECASE
        )
        if m:
            return [m.group(1).strip(), m.group(2).strip()]

        # Pattern 2: "impact/effect/role of <A> on <B>"
        m = re.search(
            r"\b(?:impact|effect|influence|role|consequence)\s+of\s+(.+?)\s+on\s+(.+?)(?:\?|$)",
            q, re.IGNORECASE
        )
        if m:
            return [m.group(1).strip(), m.group(2).strip()]

        # Pattern 3: "how does/did <A> affect/impact/influence <B>"
        m = re.search(
            r"\bhow\s+(?:does|did|has|do)\s+(.+?)\s+(?:affect|impact|influence|relate\s+to|connect\s+to)\s+(.+?)(?:\?|$)",
            q, re.IGNORECASE
        )
        if m:
            return [m.group(1).strip(), m.group(2).strip()]

        return None

    def _build_subquestions(
        self,
        query: str,
        qtype: str,
        rel_result,
        session_context: Dict,
    ) -> List[SubQuestion]:
        subqs: List[SubQuestion] = []
        last_entities = session_context.get("last_entities", [])

        if qtype == "single_hop":
            subqs.append(SubQuestion(
                index=0,
                text=query,
                anchor_entity=last_entities[0] if last_entities else None,
                target_entity=None,
                rel_hint=rel_result.matched_relations[0] if rel_result.matched_relations else None,
            ))

        elif qtype in ("multi_hop_chain", "temporal"):
            # First try "between A and B" / "impact of A on B" splitting
            between_ents = self._extract_between_entities(query)
            if between_ents and len(between_ents) == 2:
                ent_a, ent_b = between_ents
                rel_hint = rel_result.matched_relations[0] if rel_result.matched_relations else None
                subqs.append(SubQuestion(
                    index=0,
                    text=f"What is {ent_a}?",
                    anchor_entity=ent_a,
                    target_entity=ent_b,
                    rel_hint=rel_hint,
                ))
                subqs.append(SubQuestion(
                    index=1,
                    text=f"How does {ent_a} relate to {ent_b}?",
                    anchor_entity=ent_b,
                    target_entity=ent_a,
                    rel_hint=rel_hint,
                ))
                return subqs

            # Split on causal / temporal connectors
            connectors = [
                r"\s+after\s+", r"\s+following\s+", r"\s+because\s+of\s+",
                r"\s+as\s+a\s+result\s+of\s+", r"\s+due\s+to\s+",
                r"\s+that\s+(benefited|were affected)\s+by\s+",
            ]
            parts = [query]
            for conn in connectors:
                new_parts = []
                for part in parts:
                    split = re.split(conn, part, maxsplit=1, flags=re.IGNORECASE)
                    new_parts.extend(split)
                parts = new_parts

            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                rel_hint = None
                if rel_result.matched_relations:
                    rel_hint = rel_result.matched_relations[min(i, len(rel_result.matched_relations)-1)]
                subqs.append(SubQuestion(
                    index=i,
                    text=part,
                    anchor_entity=last_entities[i] if i < len(last_entities) else None,
                    target_entity=None,
                    rel_hint=rel_hint,
                ))

        elif qtype == "multi_hop_intersection":
            # Each sub-question targets one branch
            # Simple split on "and" or "both"
            parts = re.split(r"\band\b|\bboth\b", query, flags=re.IGNORECASE)
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    subqs.append(SubQuestion(
                        index=i, text=part,
                        anchor_entity=None, target_entity=None,
                        rel_hint=rel_result.matched_relations[0] if rel_result.matched_relations else None,
                    ))

        if not subqs:
            subqs.append(SubQuestion(
                index=0, text=query,
                anchor_entity=None, target_entity=None, rel_hint=None,
            ))
        return subqs

    def _resolve_coref(self, query: str, ctx: Dict[str, Any]) -> str:
        """Replace pronouns / generic references with session memory values."""
        q = query
        last_entities: List[str] = ctx.get("last_entities", [])
        last_entity_name: Optional[str] = ctx.get("last_entity_name")
        last_event_name: Optional[str] = ctx.get("last_event_name")

        if last_entity_name:
            q = re.sub(r"\b(they|their|them|it|its|the company|this company|that company)\b",
                       last_entity_name, q, flags=re.IGNORECASE)
        if last_event_name:
            q = re.sub(r"\b(that event|this event|the event|it)\b",
                       last_event_name, q, flags=re.IGNORECASE)
        return q
