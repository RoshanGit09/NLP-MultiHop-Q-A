"""
local_translator.py — Wrapper around the locally trained Transformer model.

Loads the custom Encoder-Decoder Transformer trained on
Samanantar EN↔Indic parallel corpus.  The model is loaded ONCE on first
use (lazy singleton) so the backend startup is not slowed down.

Model paths (relative to the repo root):
  Model    : translation_model/checkpoints/final_model.pt
  Tokenizer: translation_model/tokenizer/multilingual_indic-3.model

Supported translation directions:
  English  → hi / mr / ta / te / kn / ml
  hi / mr / ta / te / kn / ml → English
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────
# This file lives at  backend/core/local_translator.py
# The repo root is two levels up.
_HERE = Path(__file__).resolve().parent          # backend/core/
_REPO_ROOT = _HERE.parent.parent                 # repo root

_MODEL_PATH = _REPO_ROOT / "translation_model" / "checkpoints" / "final_model.pt"
_TOKENIZER_PATH = _REPO_ROOT / "translation_model" / "tokenizer" / "multilingual_indic-3.model"

# ── Singleton state ────────────────────────────────────────────────────────
_inference_engine = None
_load_lock = threading.Lock()
_load_failed = False   # set to True permanently if load fails, to skip retries


def _get_engine():
    """
    Return the (lazily loaded) MultilingualInference singleton.
    Returns None if the model files are missing or loading fails.
    """
    global _inference_engine, _load_failed

    if _inference_engine is not None:
        return _inference_engine
    if _load_failed:
        return None

    with _load_lock:
        # Double-checked locking
        if _inference_engine is not None:
            return _inference_engine
        if _load_failed:
            return None

        if not _MODEL_PATH.exists():
            logger.warning(
                "Local translator: model file not found at %s — "
                "translation will fall back to DeepSeek.",
                _MODEL_PATH,
            )
            _load_failed = True
            return None

        if not _TOKENIZER_PATH.exists():
            logger.warning(
                "Local translator: tokenizer not found at %s — "
                "translation will fall back to DeepSeek.",
                _TOKENIZER_PATH,
            )
            _load_failed = True
            return None

        try:
            # Add the translation_model/ folder to sys.path so
            # 'from models.transformer import ...' works inside inference.py
            _tm_dir = str(_REPO_ROOT / "translation_model")
            if _tm_dir not in sys.path:
                sys.path.insert(0, _tm_dir)

            from inference import MultilingualInference  # type: ignore[import]  # runtime path

            logger.info("Local translator: loading trained model …")
            engine = MultilingualInference(
                model_path=str(_MODEL_PATH),
                tokenizer_path=str(_TOKENIZER_PATH),
            )
            _inference_engine = engine
            logger.info("Local translator: model loaded successfully ✓")
            return _inference_engine

        except Exception as exc:
            logger.warning(
                "Local translator: failed to load model (%s) — "
                "translation will fall back to DeepSeek.",
                exc,
            )
            _load_failed = True
            return None


def translate_local(
    text: str,
    source_lang: str,
    target_lang: str,
    use_beam_search: bool = False,
    max_length: int = 256,
) -> Optional[str]:
    """
    Translate *text* using the locally trained Transformer.

    Returns the translated string on success, or None if the model is
    unavailable (caller should fall back to DeepSeek).

    Args:
        text:           Text to translate.
        source_lang:    ISO 639-1 source code  ("en"|"hi"|"mr"|"ta"|"te"|"kn"|"ml").
        target_lang:    ISO 639-1 target code  ("en"|"hi"|"mr"|"ta"|"te"|"kn"|"ml").
        use_beam_search: Use beam search (slower, better quality) vs greedy.
        max_length:     Maximum output token length.
    """
    if not text or not text.strip():
        return text

    if source_lang == target_lang:
        return text

    engine = _get_engine()
    if engine is None:
        return None

    try:
        if use_beam_search:
            result = engine.beam_search_decode(
                src_text=text,
                target_lang=target_lang,
                max_length=max_length,
                verbose=False,
            )
        else:
            result = engine.greedy_decode(
                src_text=text,
                target_lang=target_lang,
                max_length=max_length,
                verbose=False,
            )

        if result and result.strip():
            # Strip leaked language-direction tokens like <2hi>, <2ta>, <2en> etc.
            import re as _re
            result = _re.sub(r"<2[a-z]{2}>\s*", "", result).strip()
            return result if result else None
        return None

    except Exception as exc:
        logger.warning("Local translator: inference error (%s).", exc)
        return None


def is_available() -> bool:
    """Return True if the local model loaded successfully."""
    return _get_engine() is not None
