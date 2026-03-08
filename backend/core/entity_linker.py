"""
entity_linker.py — Maps query mentions → KG node IDs.

Supports:
- Multi-span candidate extraction (longest-span first)
- Alias lookup from financial_kg._ALIASES
- Fuzzy matching via difflib
- Scores each candidate (exact > alias > fuzzy)
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make financial_kg importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.kg_adapter.base import KGAdapterBase

# Import alias map from the existing kg_query module
try:
    from kg_query import _ALIASES, _STOPWORDS, _is_noise_result  # type: ignore
except ImportError:
    # Fallback if not importable (e.g., during unit tests)
    _ALIASES: Dict[str, List[str]] = {}
    _STOPWORDS: set = {
        "tell","me","about","what","who","is","the","a","an","in","of",
        "are","was","were","has","have","had","how","why","when","which",
        "that","this","it","its","from","to","for","on","by","at","find",
        "give","show","list","get","explain","describe","and","or","do","you",
    }
    def _is_noise_result(name: str) -> bool:  # type: ignore
        return False


_SCORE_EXACT   = 1.0
_SCORE_ALIAS   = 0.85
_SCORE_FUZZY   = 0.65


class EntityCandidate:
    __slots__ = ("id", "label", "name", "score", "match_type")

    def __init__(self, id: str, name: str, label: str,
                 score: float, match_type: str):
        self.id = id
        self.name = name
        self.label = label
        self.score = score
        self.match_type = match_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "name": self.name,
            "score": round(self.score, 4),
            "match_type": self.match_type,
        }


class EntityLinker:
    """
    Maps free-text mentions to KG node IDs.

    Usage:
        linker = EntityLinker(adapter)
        result = linker.link(query)
        # result: List[EntityCandidate]  (sorted by score desc)
    """

    def __init__(self, adapter: KGAdapterBase, top_k: int = 5):
        self._adapter = adapter
        self._top_k = top_k

    def link(self, query: str) -> List[EntityCandidate]:
        """
        Extract entity mentions from the query and link them to KG nodes.
        Returns a flat list of candidates (deduplicated, sorted by score).
        """
        candidates: List[EntityCandidate] = []
        seen_ids: set = set()
        spans = self._extract_spans(query)

        for span in spans:
            hits = self._resolve_span(span)
            for h in hits:
                if h.id not in seen_ids:
                    seen_ids.add(h.id)
                    candidates.append(h)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: self._top_k]

    def resolve_mention(self, mention: str) -> List[EntityCandidate]:
        """Direct resolution of a single mention (for /resolve endpoint)."""
        hits = self._resolve_span(mention.lower().strip())
        hits.sort(key=lambda c: c.score, reverse=True)
        return hits[: self._top_k]

    # ── private ──────────────────────────────────────────────────────────────

    def _extract_spans(self, query: str) -> List[str]:
        """
        Extract candidate spans (longest-first) skipping stopword-only spans.
        """
        tokens = re.sub(r"[?.!,]", "", query.lower()).split()
        spans: List[str] = []
        seen: set = set()
        for size in (4, 3, 2, 1):
            for i in range(len(tokens) - size + 1):
                chunk = tokens[i: i + size]
                if all(t in _STOPWORDS for t in chunk):
                    continue
                span = " ".join(chunk)
                if span not in seen:
                    seen.add(span)
                    spans.append(span)
        return spans

    def _resolve_span(self, span: str) -> List[EntityCandidate]:
        # 1) Exact substring
        hits = self._adapter.search_nodes(span, top_k=self._top_k)
        results: List[EntityCandidate] = []
        if hits:
            for h in hits:
                if not _is_noise_result(h.get("name", "")):
                    results.append(
                        EntityCandidate(
                            id=h["id"], name=h["name"],
                            label=h.get("label", "?"),
                            score=_SCORE_EXACT,
                            match_type="exact_substring",
                        )
                    )
            if results:
                return results

        # 2) Alias lookup
        aliases = self._get_aliases(span)
        for alias in aliases:
            hits = self._adapter.search_nodes(alias, top_k=self._top_k)
            for h in hits:
                if not _is_noise_result(h.get("name", "")):
                    results.append(
                        EntityCandidate(
                            id=h["id"], name=h["name"],
                            label=h.get("label", "?"),
                            score=_SCORE_ALIAS,
                            match_type=f"alias:{alias}",
                        )
                    )
            if results:
                return results

        # 3) Fuzzy (difflib)
        results = self._fuzzy_resolve(span)
        return results

    def _fuzzy_resolve(self, span: str) -> List[EntityCandidate]:
        # We need all names; pull from adapter via a broad search
        # (this is a best-effort fallback, performance is secondary)
        candidates: List[EntityCandidate] = []
        # Use adapter search_nodes with individual tokens
        tokens = span.split()
        for tok in tokens:
            if tok in _STOPWORDS or len(tok) < 3:
                continue
            hits = self._adapter.search_nodes(tok[:4], top_k=10)
            for h in hits:
                name = h.get("name", "")
                if _is_noise_result(name):
                    continue
                ratio = difflib.SequenceMatcher(None, span, name.lower()).ratio()
                if ratio >= 0.55:
                    candidates.append(
                        EntityCandidate(
                            id=h["id"], name=name,
                            label=h.get("label", "?"),
                            score=_SCORE_FUZZY * ratio,
                            match_type="fuzzy",
                        )
                    )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: self._top_k]

    @staticmethod
    def _get_aliases(query: str) -> List[str]:
        q = query.lower().strip()
        for key, aliases in _ALIASES.items():
            if key == q or q.startswith(key + " ") or q.endswith(" " + key):
                return aliases
            if f" {key} " in f" {q} ":
                return aliases
        return []
