"""
lang_router.py — Language detection, normalization, and routing.

Detects English and 6 Indian languages from queries and normalises them
for downstream processing (KG search always runs in English).

Supported languages:
  "en"  — English
  "hi"  — Hindi    (Devanagari: U+0900–U+097F)
  "mr"  — Marathi  (Devanagari: same block as Hindi)
  "ta"  — Tamil    (U+0B80–U+0BFF)
  "te"  — Telugu   (U+0C00–U+0C7F)
  "kn"  — Kannada  (U+0C80–U+0CFF)
  "ml"  — Malayalam(U+0D00–U+0D7F)
  "mixed" — Latin + any Indic script
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


# ── Unicode script ranges ─────────────────────────────────────────────────
_DEVANAGARI_RE  = re.compile(r"[\u0900-\u097F]")   # Hindi + Marathi
_TAMIL_RE       = re.compile(r"[\u0B80-\u0BFF]")
_TELUGU_RE      = re.compile(r"[\u0C00-\u0C7F]")
_KANNADA_RE     = re.compile(r"[\u0C80-\u0CFF]")
_MALAYALAM_RE   = re.compile(r"[\u0D00-\u0D7F]")
_LATIN_RE       = re.compile(r"[a-zA-Z]")

# All Indic patterns in priority order (most distinguishable first)
_INDIC_PATTERNS = [
    ("ta", _TAMIL_RE),
    ("te", _TELUGU_RE),
    ("kn", _KANNADA_RE),
    ("ml", _MALAYALAM_RE),
    ("hi", _DEVANAGARI_RE),   # hi/mr both Devanagari — resolved by langdetect below
]

# langdetect code → our code
_LD_MAP = {
    "hi": "hi", "mr": "mr",
    "ta": "ta", "te": "te",
    "kn": "kn", "ml": "ml",
}

# Languages we translate through — anything not English
_NON_ENGLISH = {"hi", "mr", "ta", "te", "kn", "ml", "mixed"}


@dataclass
class LanguageRouterResult:
    detected_lang: str           # "en"|"hi"|"mr"|"ta"|"te"|"kn"|"ml"|"mixed"
    normalized_query_en: str     # always populated (transliterated if needed)
    normalized_query_orig: Optional[str] = None   # original in native script
    is_mixed: bool = False
    original: str = ""


class LanguageRouter:
    """
    Detects script/language, normalises tokens, and routes to English
    for KG processing.

    Strategy:
      1. If hint is a known lang code, trust it.
      2. Count characters per Unicode block.
         - Dominant block (>50%) → that language.
         - Mixed Indic+Latin or multiple Indic → "mixed".
         - Pure Latin → use langdetect if available, else "en".
      3. For any non-English result, use DeepSeek to produce a clean
         English translation (via translator module) instead of a
         simple character map — gives much better KG search results.
      4. Strip noise tokens, normalise whitespace/case.
    """

    # Simple Devanagari → Latin phonetic fallback (used only if DeepSeek unavailable)
    _DEVA_MAP: dict[str, str] = {
        "अ":"a","आ":"aa","इ":"i","ई":"ii","उ":"u","ऊ":"uu",
        "ए":"e","ऐ":"ai","ओ":"o","औ":"au","क":"k","ख":"kh",
        "ग":"g","घ":"gh","च":"ch","छ":"chh","ज":"j","झ":"jh",
        "ट":"t","ठ":"th","ड":"d","ढ":"dh","त":"t","थ":"th",
        "द":"d","ध":"dh","न":"n","प":"p","फ":"ph","ब":"b",
        "भ":"bh","म":"m","य":"y","र":"r","ल":"l","व":"v",
        "श":"sh","ष":"sh","स":"s","ह":"h","ं":"n","ा":"a",
        "ि":"i","ी":"i","ु":"u","ू":"u","े":"e","ै":"ai",
        "ो":"o","ौ":"au","्":"","ँ":"n","ः":"h",
    }

    def route(self, query: str, lang_hint: str = "auto") -> LanguageRouterResult:
        query = query.strip()
        original = query

        # 1 — Honour explicit hint
        if lang_hint in ("en", "hi", "mr", "ta", "te", "kn", "ml"):
            detected = lang_hint
        else:
            detected = self._detect(query)

        # 2 — Produce English normalised form
        if detected in _NON_ENGLISH:
            normalized_en = self._to_english(query, detected)
        else:
            normalized_en = self._normalize_en(query)

        return LanguageRouterResult(
            detected_lang=detected,
            normalized_query_en=normalized_en,
            normalized_query_orig=query if detected in _NON_ENGLISH else None,
            is_mixed=(detected == "mixed"),
            original=original,
        )

    # ── private helpers ───────────────────────────────────────────────────

    def _detect(self, text: str) -> str:
        counts: dict[str, int] = {}
        for lang, pattern in _INDIC_PATTERNS:
            c = len(pattern.findall(text))
            if c:
                counts[lang] = c
        latin_count = len(_LATIN_RE.findall(text))

        total_indic = sum(counts.values())
        total = total_indic + latin_count or 1

        if total_indic == 0:
            # Pure Latin — use langdetect for hi/mr written in Latin (Hinglish etc.)
            if _LANGDETECT_AVAILABLE:
                try:
                    ld = detect(text)
                    if ld in _LD_MAP:
                        return _LD_MAP[ld]
                except LangDetectException:
                    pass
            return "en"

        indic_ratio = total_indic / total

        if indic_ratio < 0.40:
            return "mixed"

        # Dominant Indic script
        dominant_lang, dominant_count = max(counts.items(), key=lambda x: x[1])

        # Devanagari: disambiguate hi vs mr via langdetect
        if dominant_lang == "hi" and _LANGDETECT_AVAILABLE:
            try:
                ld = detect(text)
                if ld == "mr":
                    return "mr"
            except LangDetectException:
                pass

        return dominant_lang

    def _to_english(self, text: str, detected_lang: str) -> str:
        """
        Translate native-script query to English using DeepSeek.
        Falls back to character-map transliteration (Devanagari only) on error.
        """
        try:
            from .translator import translate
            en = translate(text, source_lang=detected_lang, target_lang="en")
            if en and en.strip():
                return self._normalize_en(en)
        except Exception:
            pass
        # Fallback: character-map for Devanagari (Hindi/Marathi)
        transliterated = self._transliterate_deva(text)
        return self._normalize_en(transliterated)

    def _normalize_en(self, text: str) -> str:
        """Lowercase, strip control chars, normalise whitespace."""
        text = text.lower()
        text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
        text = re.sub(r"[''`]", "'", text)
        text = re.sub(r"[–—]", "-", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _transliterate_deva(self, text: str) -> str:
        """Transliterate Devanagari characters to Latin approximation."""
        return "".join(self._DEVA_MAP.get(ch, ch) for ch in text)
