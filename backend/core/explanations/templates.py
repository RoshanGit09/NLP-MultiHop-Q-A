"""
templates.py — Explanation generator.

Primary: Gemini generates fluent step-by-step explanation from KG triples.
Fallback: template-based (one mechanical sentence per triple).
"""
from __future__ import annotations

import logging
from typing import List

from backend.core.traversal import ReasoningPath, Triple

logger = logging.getLogger(__name__)


class TemplateExplainer:
    """
    Produces a list of natural-language explanation steps,
    one sentence per triple in the selected path.
    Every sentence is directly grounded in one KG triple.
    """

    def explain(self, path: ReasoningPath) -> List[str]:
        if not path.triples:
            return []

        # ── Try Gemini first ──────────────────────────────────────────────
        triple_dicts = [
            {"subj": t.subj, "rel": t.rel, "obj": t.obj, "time": t.time}
            for t in path.triples
        ]
        try:
            from backend.core.gemini_client import explain as gemini_explain
            steps = gemini_explain(triple_dicts)
            if steps:
                logger.debug("Gemini explain: %d steps", len(steps))
                return steps
        except Exception as exc:
            logger.warning("Gemini explain error: %s — falling back to templates", exc)

        # ── Template fallback ─────────────────────────────────────────────
        return [
            self._triple_to_sentence(i + 1, triple)
            for i, triple in enumerate(path.triples)
        ]

    @staticmethod
    def _triple_to_sentence(step: int, t: Triple) -> str:
        time_str = f" (recorded: {t.time})" if t.time else ""
        source_str = f" [Source: {t.source}]" if t.source else ""
        rel_phrase = _rel_phrase(t.rel)
        return (
            f"Step {step}: **{t.subj}** {rel_phrase} **{t.obj}**"
            f"{time_str}{source_str}."
        )


def _rel_phrase(rel: str) -> str:
    """Convert a KG relation type to a short English verb phrase."""
    _MAP = {
        # Actual KG relation types
        "REPORTS":          "reported",
        "ANNOUNCES":        "announced",
        "IMPACTS":          "impacted",
        "IS_IN_FOCUS":      "is in focus alongside",
        "OPERATES_IN":      "operates in the",
        # Legacy / lower-frequency
        "CEO_OF":           "serves as CEO of",
        "LEADS":            "leads",
        "FOUNDED":          "founded",
        "PART_OF":          "is part of",
        "REGULATES":        "is regulated by",
        "REGULATED_BY":     "is regulated by",
        "INFLUENCES":       "influences",
        "RATE_CHANGE":      "underwent a rate change affecting",
        "POLICY_CHANGE":    "was subject to a policy change from",
        "ACQUIRED":         "acquired",
        "MERGED_WITH":      "merged with",
        "INVESTED_IN":      "invested in",
        "IMPACTED_BY":      "was impacted by",
        "BENEFITED_FROM":   "benefited from",
        "ANNOUNCED":        "announced",
        "FILED":            "filed",
        "IPO":              "went public via IPO on",
        "LISTED_ON":        "is listed on",
        "DIVIDEND":         "declared a dividend",
        "BUYBACK":          "conducted a share buyback",
        "RELATED_TO":       "is related to",
        "LINKED_TO":        "is linked to",
        "JOINT_VENTURE":    "entered a joint venture with",
        "PARTNERSHIP_WITH": "entered a partnership with",
    }
    return _MAP.get(rel.upper(), rel.lower().replace("_", " "))
