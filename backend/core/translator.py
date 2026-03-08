"""
translator.py — Multilingual translation for FinTraceQA using DeepSeek.

Translates between English and 6 Indian languages via DeepSeek-V3.
Uses the same DeepSeek client configured in gemini_client.py.

Supported language codes (ISO 639-1):
  "en"  — English
  "hi"  — Hindi       (Devanagari)
  "mr"  — Marathi     (Devanagari)
  "ta"  — Tamil
  "te"  — Telugu
  "kn"  — Kannada
  "ml"  — Malayalam
  "mixed" — treated as Hindi for translation output
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# Language display names used inside translation prompts
_LANG_NAMES: dict[str, str] = {
    "en":    "English",
    "hi":    "Hindi",
    "mr":    "Marathi",
    "ta":    "Tamil",
    "te":    "Telugu",
    "kn":    "Kannada",
    "ml":    "Malayalam",
    "mixed": "Hindi",   # fallback for mixed Devanagari
}

# All non-English codes that trigger translation back to the user's language
_NON_ENGLISH_LANGS = {"hi", "mr", "ta", "te", "kn", "ml", "mixed"}

_TRANSLATE_SYSTEM = (
    "You are a professional financial translator specialising in Indian stock markets. "
    "Translate the given text accurately into {target_lang}. "
    "Keep financial terms, company names, ticker symbols, numbers, and proper nouns unchanged. "
    "Preserve **bold** markdown formatting. "
    "Output ONLY the translated text — no explanations, no notes, no prefixes."
)


def _canonical(lang: str) -> str:
    """Normalise 'mixed' → 'hi'; pass through everything else."""
    return "hi" if lang == "mixed" else lang


def translate(
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """
    Translate *text* from *source_lang* to *target_lang*.

    Strategy (in priority order):
      1. Use the locally trained Transformer model (MultilingualInference).
      2. Fall back to DeepSeek-V3 if the local model is unavailable or fails.

    Returns the translated string, or the original text on any error
    (fail-safe — never crashes the caller).

    Args:
        text:        Text to translate.
        source_lang: ISO 639-1 source code or "mixed".
        target_lang: ISO 639-1 target code or "mixed".
    """
    if not text or not text.strip():
        return text

    source_lang = _canonical(source_lang)
    target_lang = _canonical(target_lang)

    if source_lang == target_lang:
        return text

    # ── 1. Try local trained model ─────────────────────────────────────────
    try:
        from .local_translator import translate_local
        local_result = translate_local(text, source_lang=source_lang, target_lang=target_lang)
        if local_result:
            logger.debug("Translator: used local model (%s→%s).", source_lang, target_lang)
            return local_result
    except Exception as exc:
        logger.debug("Translator: local model unavailable (%s), falling back to DeepSeek.", exc)

    # ── 2. Fall back to DeepSeek ───────────────────────────────────────────
    logger.info("Translator: using DeepSeek fallback (%s→%s).", source_lang, target_lang)
    target_name = _LANG_NAMES.get(target_lang, target_lang)
    system = _TRANSLATE_SYSTEM.format(target_lang=target_name)

    try:
        from .gemini_client import _generate
        result = _generate(
            system=system,
            user=text,
            temperature=0.1,
            max_tokens=800,
        )
        if not result or not result.strip():
            logger.warning("Translator: DeepSeek returned empty — keeping original.")
            return text
        return result.strip()
    except Exception as exc:
        logger.warning("Translator: DeepSeek fallback also failed (%s) — keeping original.", exc)
        return text


def translate_answer(
    answer: str,
    explanation_steps: List[str],
    detected_lang: str,
) -> tuple[str, List[str], str]:
    """
    Translate an English answer + reasoning steps back into the user's
    detected language (if non-English).

    Returns:
        (translated_answer, translated_steps, answer_lang)

        *answer_lang* is the ISO code of the output language
        (e.g. "hi", "ta", "te", "mr", "kn", "ml"), or "en" if no
        translation was needed.
    """
    if detected_lang not in _NON_ENGLISH_LANGS:
        return answer, explanation_steps, "en"

    target = _canonical(detected_lang)
    logger.info("Translator: translating answer to %s (detected=%s).", target, detected_lang)

    translated_answer = translate(answer, source_lang="en", target_lang=target)

    translated_steps: List[str] = [
        translate(step, source_lang="en", target_lang=target)
        for step in explanation_steps
    ]

    return translated_answer, translated_steps, target
