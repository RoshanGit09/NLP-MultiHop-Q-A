"""
faithfulness_verifier.py — Checks that every explanation sentence
is grounded in at least one triple from the selected path.

Primary: Gemini judges each sentence ("YES" / "NO").
Fallback: lexical overlap check (2-of-3 subj/rel/obj match).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from backend.core.traversal import ReasoningPath, Triple

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    passed: bool
    aligned_count: int
    total_count: int
    alignment_ratio: float
    unaligned_sentences: List[str]
    verified_steps: List[str]   # only the sentences that passed


class FaithfulnessVerifier:
    """
    Verifies that explanation sentences are supported by path triples.
    Uses Gemini for semantic verification; falls back to lexical overlap.
    """

    def __init__(self, min_aligned_ratio: float = 0.80):
        self._threshold = min_aligned_ratio

    def verify(
        self,
        explanation_steps: List[str],
        path: ReasoningPath,
    ) -> VerificationResult:
        aligned: List[str] = []
        unaligned: List[str] = []

        triple_dicts = [
            {"subj": t.subj, "rel": t.rel, "obj": t.obj, "time": t.time}
            for t in path.triples
        ]

        for sentence in explanation_steps:
            if self._is_supported(sentence, path.triples, triple_dicts):
                aligned.append(sentence)
            else:
                unaligned.append(sentence)

        total = len(explanation_steps)
        ratio = len(aligned) / total if total > 0 else 1.0

        return VerificationResult(
            passed=ratio >= self._threshold,
            aligned_count=len(aligned),
            total_count=total,
            alignment_ratio=round(ratio, 3),
            unaligned_sentences=unaligned,
            verified_steps=aligned,
        )

    def _is_supported(
        self,
        sentence: str,
        triples: List[Triple],
        triple_dicts: List[dict],
    ) -> bool:
        # ── Try Gemini semantic verification ─────────────────────────────
        try:
            from backend.core.gemini_client import verify_sentence
            grounded, _ = verify_sentence(sentence, triple_dicts)
            return grounded
        except Exception as exc:
            logger.warning("Gemini verify error: %s — falling back to lexical", exc)

        # ── Lexical fallback: 2-of-3 subj/rel/obj appear in sentence ──────
        return self._lexical_check(sentence, triples)

    @staticmethod
    def _lexical_check(sentence: str, triples: List[Triple]) -> bool:
        s = sentence.lower()
        for t in triples:
            hits = 0
            for part in (t.subj, t.obj):
                if part and len(part) > 2 and part.lower() in s:
                    hits += 1
            if t.rel:
                rel_phrase = t.rel.lower().replace("_", " ")
                if rel_phrase in s:
                    hits += 1
            if hits >= 2:
                return True
        return False
