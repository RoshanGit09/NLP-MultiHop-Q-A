"""
gemini_client.py — AI wrapper using DeepSeek-V3 (non-thinking) via api.deepseek.com.

DeepSeek uses an OpenAI-compatible API.  Model: deepseek-chat (= DeepSeek-V3, non-thinking).

Usage:
    from backend.core.gemini_client import decompose, synthesize, explain, verify_sentence

    plan   = decompose(query, context)
    answer = synthesize(triples, query, qtype)
    steps  = explain(triples)
    ok, r  = verify_sentence(sentence, triples)
"""
from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_MODEL_NAME        = "deepseek-chat"   # DeepSeek-V3, non-thinking mode

# ── Lazy singleton ────────────────────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. Add it to backend/.env"
            )
        from openai import OpenAI
        _client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
    return _client


# ── Core generate helper ──────────────────────────────────────────────────

def _generate(
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Call DeepSeek-V3 with a system + user message.
    Returns the response text, or '' on any error.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("DeepSeek call failed: %s", exc)
        return ""


# ── Prompts ───────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = textwrap.dedent("""
You are a financial question analyser for an Indian stock market Q&A system.

Given the user question and optional conversation context, output a JSON object with:
  "question_type": one of "single_hop" | "multi_hop_chain" | "multi_hop_intersection" | "temporal"
  "subquestions": list of strings (ordered sub-questions to answer)
  "anchor_entities": list of entity strings (one per subquestion, the key entity to search for)
  "constraints": object with optional keys "rel_types" (list), "time_after" (string), "time_before" (string)
  "reasoning": one sentence explanation of your classification

Rules:
- "single_hop": answerable with ONE fact from the KG (who, what, when about a single entity)
- "multi_hop_chain": requires following a chain of relations (A→B→C), OR asks about "connection/relationship/impact between X and Y", OR asks "how does X affect Y"
- "multi_hop_intersection": requires finding entities satisfying TWO conditions simultaneously
- "temporal": involves time-bounded filtering (after Q3, before FY24, since rate cut)
- For "multi_hop_chain" with "connection between X and Y": generate TWO subquestions, one starting from X and one starting from Y
- Subquestions should be atomic, each answerable independently
- "anchor_entities": extract the SPECIFIC named entity that should be the traversal start for each subquestion
- Output ONLY valid JSON, no markdown fences, no commentary

Examples:
Q: "What is the connection between US-India trade deal and the Indian stock market?"
→ {"question_type":"multi_hop_chain","subquestions":["What events relate to the US-India trade deal?","How did the US-India trade deal affect Indian stock markets?"],"anchor_entities":["US-India trade deal","Indian stock market"],"constraints":{}}

Q: "Who is the CEO of Infosys?"
→ {"question_type":"single_hop","subquestions":["Who is the CEO of Infosys?"],"anchor_entities":["Infosys"],"constraints":{}}
""").strip()

_SYNTHESIZE_SYSTEM = textwrap.dedent("""
You are a senior financial analyst assistant answering questions about Indian stock markets.
Your ONLY source of truth is the Knowledge Graph (KG) triples provided by the user.
Do NOT add any information not present in the triples. Do NOT hallucinate.

Instructions:
- Read the KG triples carefully — they represent real facts extracted from financial news.
- For "multi_hop_chain" or "connection/impact/relationship" questions:
    Write 2–4 sentences explaining the FULL chain of causation from start to end.
    Example: "The RBI's rate cut reduced borrowing costs, which boosted HDFC Bank's lending margins and led to strong quarterly results."
- For "single_hop" questions: give a direct, concise 1–2 sentence answer.
- Bold key entity names using **entity** markdown syntax.
- Mention dates/times if present in the triples.
- If the triples show A→B→C→D, explain each link in plain English.
- NEVER just list the triples. NEVER output "Answer: [node name]". Write proper sentences.
- If no relevant triples are provided, say: "I could not find a direct answer in the Knowledge Graph. Please try a more specific query."
""").strip()

_EXPLAIN_SYSTEM = textwrap.dedent("""
You are explaining the step-by-step reasoning behind a financial Q&A answer.
Use ONLY the KG triples provided. Generate a numbered list where each step = one triple.
Be concise. Bold entity names. Output ONLY the numbered steps, no intro text.

Format:
Step 1: **Entity A** [relation] **Entity B**.
Step 2: ...
""").strip()

_VERIFY_SYSTEM = (
    "You check whether an explanation sentence is grounded in KG triples. "
    "Answer with exactly 'YES' or 'NO'."
)


# ── Public helpers ────────────────────────────────────────────────────────

def _format_triples(triples: List[Dict[str, Any]]) -> str:
    lines = []
    for i, t in enumerate(triples, 1):
        time_part = f" [{t.get('time') or t.get('t', '')}]" if (t.get("time") or t.get("t")) else ""
        lines.append(f"  {i}. {t.get('subj','?')} --[{t.get('rel','?')}]--> {t.get('obj','?')}{time_part}")
    return "\n".join(lines) or "  (none)"


def decompose(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Ask DeepSeek to classify and decompose the question.
    Returns parsed dict or None if the call fails (caller falls back to rules).
    """
    ctx_str = json.dumps(context or {}, ensure_ascii=False)
    user_msg = f"User question: {query}\nConversation context: {ctx_str}"
    raw = _generate(_DECOMPOSE_SYSTEM, user_msg, temperature=0.1, max_tokens=400)
    if not raw:
        return None
    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("DeepSeek decompose: JSON parse failed. raw=%s", raw[:200])
        return None


def synthesize(
    triples: List[Dict[str, Any]],
    query: str,
    qtype: str = "single_hop",
) -> str:
    """
    Ask DeepSeek to verbalize triples into a fluent, grounded answer.
    Falls back to empty string (caller uses template synthesizer).
    """
    if not triples:
        return ""
    triples_str = _format_triples(triples)
    user_msg = (
        f"Question type: {qtype}\n"
        f"User question: {query}\n\n"
        f"Knowledge Graph triples:\n{triples_str}\n\n"
        f"Write a fluent, accurate answer based ONLY on the triples above."
    )
    return _generate(_SYNTHESIZE_SYSTEM, user_msg, temperature=0.3, max_tokens=400)


def explain(triples: List[Dict[str, Any]]) -> List[str]:
    """
    Ask DeepSeek to generate a step-by-step explanation from triples.
    Returns list of step strings, or [] on failure.
    """
    if not triples:
        return []
    triples_str = _format_triples(triples)
    user_msg = f"Triples:\n{triples_str}"
    raw = _generate(_EXPLAIN_SYSTEM, user_msg, temperature=0.1, max_tokens=300)
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    steps = [ln for ln in lines if re.match(r"^Step\s+\d+", ln, re.IGNORECASE)]
    return steps if steps else lines


def verify_sentence(
    sentence: str,
    triples: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    Ask DeepSeek whether the sentence is grounded in the triples.
    Returns (is_grounded: bool, raw_response: str).
    Falls back to True (conservative) on error.
    """
    if not triples:
        return True, "no triples"
    triples_str = _format_triples(triples)
    user_msg = f"Sentence: {sentence}\n\nTriples:\n{triples_str}"
    raw = _generate(_VERIFY_SYSTEM, user_msg, temperature=0.0, max_tokens=10)
    if not raw:
        return True, "api_error"
    grounded = raw.strip().upper().startswith("YES")
    return grounded, raw.strip()
