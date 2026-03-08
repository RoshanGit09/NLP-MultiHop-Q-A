"""
synthesizer.py — Converts KG traversal results to a final answer string.

Primary: Gemini verbalizes the KG triples into fluent, grounded English.
Fallback: template-based generation (used when Gemini is unavailable).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .traversal import TraversalResult, ReasoningPath

logger = logging.getLogger(__name__)


class AnswerSynthesizer:
    """
    Converts structured traversal results into a concise natural-language answer.
    Strictly grounded — only uses facts present in the top-path triples.
    """

    def synthesize(
        self,
        result: TraversalResult,
        original_query: str,
        question_type: str = "single_hop",
    ) -> str:
        if not result.paths:
            return (
                "I could not find a relevant answer in the Knowledge Graph for your query. "
                "Please try rephrasing with a specific company name, event, or indicator."
            )

        top_path = result.paths[0]
        triples = top_path.triples

        if not triples:
            candidates = result.answer_candidates
            if candidates:
                return f"Based on the KG, the answer is: **{', '.join(candidates)}**."
            return "No specific answer found in the Knowledge Graph."

        # ── Collect triples from ALL top paths for richer context ───────────
        # For multi-hop, include triples from all top-k paths (deduped)
        seen_triples: set = set()
        all_triples = []
        for path in result.paths:
            for t in path.triples:
                key = (t.subj, t.rel, t.obj)
                if key not in seen_triples:
                    seen_triples.add(key)
                    all_triples.append(t)

        triple_dicts = [
            {"subj": t.subj, "rel": t.rel, "obj": t.obj, "time": t.time}
            for t in all_triples
        ]

        # ── Try Gemini first ────────────────────────────────────────────────
        try:
            from .gemini_client import synthesize as gemini_synthesize
            gemini_answer = gemini_synthesize(triple_dicts, original_query, question_type)
            if gemini_answer:
                logger.debug("Gemini synthesize: %s", gemini_answer[:80])
                return gemini_answer
        except Exception as exc:
            logger.warning("Gemini synthesize error: %s — falling back to template", exc)

        # ── Template fallback ───────────────────────────────────────────────
        return self._template_synthesize(all_triples, question_type, result)

    def _template_synthesize(self, triples, question_type: str, result: TraversalResult) -> str:
        """Original template-based synthesis."""
        if question_type == "single_hop" and len(triples) == 1:
            t = triples[0]
            time_str = f" (as of {t.time})" if t.time else ""
            return f"**{t.subj}** {_humanize_rel(t.rel)} **{t.obj}**{time_str}."

        terminal_node = (
            triples[-1].obj if triples
            else result.answer_candidates[0] if result.answer_candidates
            else "unknown"
        )
        chain_str = "  ->  ".join(
            f"{t.subj} -[{t.rel}]-> {t.obj}" for t in triples
        )
        return (
            f"Following the reasoning chain:\n{chain_str}\n\n"
            f"**Answer: {terminal_node}**"
        )


def _humanize_rel(rel: str) -> str:
    """Convert KG relation type to a human-readable verb phrase."""
    mapping = {
        # Actual KG relation types
        "REPORTS":          "reported",
        "ANNOUNCES":        "announced",
        "IMPACTS":          "impacted",
        "IS_IN_FOCUS":      "is in focus alongside",
        "OPERATES_IN":      "operates in",
        # Legacy / lower-frequency
        "CEO_OF":           "is the CEO of",
        "LEADS":            "leads",
        "FOUNDED":          "founded",
        "PART_OF":          "is part of",
        "REGULATES":        "is regulated by",
        "REGULATED_BY":     "is regulated by",
        "INFLUENCES":       "influences",
        "RATE_CHANGE":      "changed rate affecting",
        "POLICY_CHANGE":    "introduced policy change for",
        "ACQUIRED":         "acquired",
        "MERGED_WITH":      "merged with",
        "INVESTED_IN":      "invested in",
        "IMPACTED_BY":      "was impacted by",
        "BENEFITED_FROM":   "benefited from",
        "ANNOUNCED":        "announced",
        "FILED":            "filed",
        "IPO":              "had its IPO on",
        "LISTED_ON":        "is listed on",
        "DIVIDEND":         "declared dividend",
        "BUYBACK":          "conducted buyback",
        "RELATED_TO":       "is related to",
        "LINKED_TO":        "is linked to",
    }
    return mapping.get(rel.upper(), rel.lower().replace("_", " "))
