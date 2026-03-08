"""
relation_linker.py — Maps query predicates → KG relation types.

Supports a curated predicate synonym dictionary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ── Predicate synonym map ────────────────────────────────────────────────
# Keys are lower-cased surface patterns (substrings / exact words).
# Values are KG relation types as used in Neo4j.
_PREDICATE_MAP: Dict[str, List[str]] = {
    # ── Actual KG relation types (from Neo4j) ──────────────────────────────
    # REPORTS, ANNOUNCES, IMPACTS, IS_IN_FOCUS, OPERATES_IN are the top 5

    # Reporting / filing
    "report":           ["REPORTS"],
    "reports":          ["REPORTS"],
    "reported":         ["REPORTS"],
    "filing":           ["REPORTS"],
    "filed":            ["REPORTS"],
    "quarterly":        ["REPORTS"],
    "earnings":         ["REPORTS"],
    "results":          ["REPORTS"],

    # Announcements / corporate events
    "announced":        ["ANNOUNCES"],
    "announces":        ["ANNOUNCES"],
    "announcement":     ["ANNOUNCES"],
    "declared":         ["ANNOUNCES"],
    "launched":         ["ANNOUNCES"],
    "ipo":              ["ANNOUNCES"],
    "listing":          ["ANNOUNCES"],
    "dividend":         ["ANNOUNCES"],
    "buyback":          ["ANNOUNCES"],
    "merger":           ["ANNOUNCES"],
    "acquired":         ["ANNOUNCES"],

    # Impact / effect
    "impacted":         ["IMPACTS"],
    "impact":           ["IMPACTS"],
    "affected":         ["IMPACTS"],
    "affect":           ["IMPACTS"],
    "benefited":        ["IMPACTS"],
    "influenced":       ["IMPACTS"],
    "driven by":        ["IMPACTS"],
    "caused by":        ["IMPACTS"],
    "surged":           ["IMPACTS"],
    "fell":             ["IMPACTS"],
    "dropped":          ["IMPACTS"],
    "declined":         ["IMPACTS"],

    # Focus / watchlist
    "in focus":         ["IS_IN_FOCUS"],
    "focus":            ["IS_IN_FOCUS"],
    "spotlight":        ["IS_IN_FOCUS"],
    "watch":            ["IS_IN_FOCUS"],
    "trending":         ["IS_IN_FOCUS"],

    # Sector
    "operates in":      ["OPERATES_IN"],
    "sector":           ["OPERATES_IN"],
    "part of":          ["OPERATES_IN"],
    "belongs to":       ["OPERATES_IN"],
    "industry":         ["OPERATES_IN"],

    # Leadership (may exist as lower-frequency rels)
    "ceo":              ["LEADS", "ANNOUNCES"],
    "chief executive":  ["LEADS"],
    "head of":          ["LEADS"],
    "led by":           ["LEADS"],
    "founded":          ["ANNOUNCES"],
    "founder":          ["ANNOUNCES"],

    # Regulation / policy
    "regulated by":     ["IMPACTS"],
    "rbi policy":       ["ANNOUNCES", "IMPACTS"],
    "rbi rate":         ["ANNOUNCES", "IMPACTS"],
    "rate cut":         ["ANNOUNCES", "IMPACTS"],
    "repo rate":        ["ANNOUNCES", "IMPACTS"],
    "sebi":             ["ANNOUNCES", "IMPACTS"],
    "policy":           ["ANNOUNCES"],

    # Macro
    "related to":       ["IMPACTS"],
    "linked to":        ["IMPACTS"],
}

# Temporal operators — these are not relation types but query constraints
_TEMPORAL_KEYWORDS: Dict[str, str] = {
    "after":    "after",
    "before":   "before",
    "since":    "after",
    "following":"after",
    "prior to": "before",
    "during":   "during",
    "in":       "during",
}


@dataclass
class RelationLinkResult:
    matched_relations: List[str]       # KG rel types
    temporal_op: Optional[str]         # "after" | "before" | "during" | None
    temporal_ref: Optional[str]        # raw date/period string extracted
    confidence: float


class RelationLinker:
    """Maps natural language predicates to KG relation types."""

    def link(self, query: str) -> RelationLinkResult:
        q = query.lower()
        matched: List[str] = []
        seen: Set[str] = set()

        # Longest-match: try longest patterns first
        sorted_patterns = sorted(_PREDICATE_MAP.keys(), key=len, reverse=True)
        for pattern in sorted_patterns:
            if pattern in q:
                for rel in _PREDICATE_MAP[pattern]:
                    if rel not in seen:
                        seen.add(rel)
                        matched.append(rel)

        # Temporal
        temporal_op, temporal_ref = self._extract_temporal(q)

        confidence = 0.9 if matched else 0.3
        # Lower confidence when only temporal matched
        if not matched and temporal_op:
            confidence = 0.4

        return RelationLinkResult(
            matched_relations=matched,
            temporal_op=temporal_op,
            temporal_ref=temporal_ref,
            confidence=confidence,
        )

    # ── private ──────────────────────────────────────────────────────────────

    def _extract_temporal(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract temporal operator and reference from query."""
        # Detect operator
        op: Optional[str] = None
        ref: Optional[str] = None

        # Sort by length descending to prefer longer matches
        for kw, op_val in sorted(_TEMPORAL_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
            if kw in query:
                op = op_val
                # Try to extract a date/period that follows the keyword
                pattern = re.compile(
                    rf"{re.escape(kw)}\s+([a-z0-9\s/,-]{{3,30}})",
                    re.IGNORECASE,
                )
                m = pattern.search(query)
                if m:
                    ref = m.group(1).strip().rstrip(".,?")
                break

        return op, ref
