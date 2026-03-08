"""
chat.py — /chat and /resolve router for FinTraceQA chatbot.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.lang_router import LanguageRouter
from backend.core.decomposer import QuestionDecomposer
from backend.core.entity_linker import EntityLinker
from backend.core.traversal import KGTraversalEngine
from backend.core.synthesizer import AnswerSynthesizer
from backend.core.explanations import TemplateExplainer, FaithfulnessVerifier
from backend.core.memory import get_session
from backend.core.translator import translate, translate_answer

# ── Auto-reconnecting KG adapter ─────────────────────────────────────────
import logging as _logging
import threading as _threading
import time as _time

_log = _logging.getLogger(__name__)


class _SmartAdapter:
    """
    Transparent proxy around Neo4jAdapter / MockAdapter.

    - On every call, tries to use Neo4j.
    - If Neo4j throws ServiceUnavailable / SessionExpired, falls back to Mock
      for that call and schedules a reconnect attempt in the background.
    - Once Neo4j is reachable again, silently switches back — no server restart needed.
    """

    _RETRY_INTERVAL = 30   # seconds between background reconnect attempts

    def __init__(self):
        self._lock = _threading.Lock()
        self._neo4j = None          # Neo4jAdapter instance (None if unavailable)
        self._mock = None           # MockAdapter (lazy)
        self._use_mock_env = os.getenv("USE_MOCK_KG", "false").lower() == "true"
        self._reconnecting = False

        if self._use_mock_env:
            _log.info("USE_MOCK_KG=true — using MockAdapter exclusively.")
        else:
            self._try_connect()

    # ── internal helpers ────────────────────────────────────────────────────

    def _get_mock(self):
        if self._mock is None:
            from backend.core.kg_adapter.mock_adapter import MockAdapter
            self._mock = MockAdapter()
        return self._mock

    def _try_connect(self) -> bool:
        """Attempt to create a Neo4jAdapter. Returns True on success."""
        try:
            from backend.core.kg_adapter.neo4j_adapter import Neo4jAdapter
            adapter = Neo4jAdapter()
            with self._lock:
                self._neo4j = adapter
            _log.info("SmartAdapter: connected to Neo4j.")
            return True
        except Exception as exc:
            _log.warning("SmartAdapter: Neo4j unavailable (%s), using Mock.", exc)
            with self._lock:
                self._neo4j = None
            return False

    def _schedule_reconnect(self):
        """Spin up a daemon thread that retries Neo4j every _RETRY_INTERVAL s."""
        with self._lock:
            if self._reconnecting:
                return
            self._reconnecting = True

        def _worker():
            while True:
                _time.sleep(self._RETRY_INTERVAL)
                _log.info("SmartAdapter: attempting Neo4j reconnect…")
                if self._try_connect():
                    with self._lock:
                        self._reconnecting = False
                    break
            _log.info("SmartAdapter: Neo4j reconnect successful — live data restored.")

        t = _threading.Thread(target=_worker, daemon=True)
        t.start()

    def _active(self):
        with self._lock:
            return self._neo4j

    def _on_fail(self, exc):
        _log.error("SmartAdapter: Neo4j query failed (%s), falling back to Mock.", exc)
        with self._lock:
            self._neo4j = None
        self._schedule_reconnect()

    # ── KGAdapterBase interface ─────────────────────────────────────────────

    def search_nodes(self, *args, **kwargs):
        neo = self._active()
        if neo:
            try:
                return neo.search_nodes(*args, **kwargs)
            except Exception as exc:
                self._on_fail(exc)
        return self._get_mock().search_nodes(*args, **kwargs)

    def get_neighbors(self, *args, **kwargs):
        neo = self._active()
        if neo:
            try:
                return neo.get_neighbors(*args, **kwargs)
            except Exception as exc:
                self._on_fail(exc)
        return self._get_mock().get_neighbors(*args, **kwargs)

    def find_paths(self, *args, **kwargs):
        neo = self._active()
        if neo:
            try:
                return neo.find_paths(*args, **kwargs)
            except Exception as exc:
                self._on_fail(exc)
        return self._get_mock().find_paths(*args, **kwargs)

    def close(self):
        with self._lock:
            if self._neo4j:
                try:
                    self._neo4j.close()
                except Exception:
                    pass


# Shared singletons (created once at import time)
_adapter       = _SmartAdapter()
_lang_router   = LanguageRouter()
_decomposer    = QuestionDecomposer()
_traversal_eng = KGTraversalEngine(_adapter)
_synthesizer   = AnswerSynthesizer()
_explainer     = TemplateExplainer()
_verifier      = FaithfulnessVerifier()
_linker        = EntityLinker(_adapter)

router = APIRouter()

# ── Request / Response models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    lang: str = "auto"   # "en" | "hi" | "auto"
    metadata: Optional[Dict[str, Any]] = None
    debug: bool = False


class EntityResult(BaseModel):
    id: str
    label: str
    name: str
    score: float


class TripleResult(BaseModel):
    subj: str
    rel: str
    obj: str
    time: Optional[str] = None
    source: Optional[str] = None


class ReasoningPathResult(BaseModel):
    path_id: str
    hops: int
    triples: List[TripleResult]
    score: float


class CitationResult(BaseModel):
    triple_index: int
    source_uri: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    answer_lang: str
    entities: List[EntityResult]
    reasoning_paths: List[ReasoningPathResult]
    explanation_steps: List[str]
    citations: List[CitationResult]
    confidence: float
    warnings: List[str]
    debug: Optional[Dict[str, Any]] = None


class ResolveRequest(BaseModel):
    mention: str
    lang: str = "auto"


class ResolveResponse(BaseModel):
    candidates: List[EntityResult]


class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"   # "en"|"hi"|"mr"|"ta"|"te"|"kn"|"ml"
    target_lang: str = "hi"   # "en"|"hi"|"mr"|"ta"|"te"|"kn"|"ml"


class TranslateResponse(BaseModel):
    translated: str
    source_lang: str
    target_lang: str


# ── /chat endpoint ────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.perf_counter()
    warnings: List[str] = []

    # 1 — Language routing
    lang_result = _lang_router.route(req.message, req.lang)
    query_en = lang_result.normalized_query_en

    # 2 — Session memory + coreference
    session = get_session(req.session_id)
    ctx = session.get_context()

    # 3 — Question decomposition
    plan = _decomposer.decompose(query_en, session_context=ctx)

    # 4 — Traversal (with graceful fallback if KG is unavailable)
    try:
        traversal_result = _traversal_eng.execute(plan, query_en)
    except Exception as kg_exc:
        import logging
        logging.getLogger(__name__).error("KG traversal failed: %s", kg_exc)
        # Return a graceful degraded response instead of 500
        from backend.core.traversal import TraversalResult
        traversal_result = TraversalResult(paths=[], answer_candidates=[], query_plan={}, execution_ms=0)
        warnings.append("Knowledge Graph is temporarily unavailable. Answer may be incomplete.")

    # 5 — Entity candidates for response metadata
    try:
        entity_candidates = _linker.link(query_en)
    except Exception as link_exc:
        import logging
        logging.getLogger(__name__).error("Entity linking failed: %s", link_exc)
        entity_candidates = []
        warnings.append("Entity linking failed due to KG unavailability.")

    # 6 — Answer synthesis
    answer_text = _synthesizer.synthesize(
        traversal_result, query_en, plan.question_type
    )

    # 7 — Explanation
    raw_steps: List[str] = []
    verified_steps: List[str] = []
    if traversal_result.paths:
        top_path = traversal_result.paths[0]
        raw_steps = _explainer.explain(top_path)
        ver_result = _verifier.verify(raw_steps, top_path)
        verified_steps = ver_result.verified_steps
        if ver_result.unaligned_sentences:
            warnings.append(
                f"{len(ver_result.unaligned_sentences)} explanation sentence(s) "
                "could not be grounded to KG triples and were removed."
            )
    else:
        verified_steps = []
        warnings.append("No KG path found. Answer may be incomplete.")

    # 7b — Translate answer + explanation back to user's language if Hindi/mixed
    answer_text, verified_steps, answer_lang = translate_answer(
        answer_text, verified_steps, lang_result.detected_lang
    )

    # 8 — Confidence
    confidence = traversal_result.paths[0].score if traversal_result.paths else 0.0
    if confidence < 0.60:
        warnings.append(
            f"Low confidence ({confidence:.2f}). Entity or relation may be ambiguous."
        )

    # 9 — Citations
    citations: List[CitationResult] = []
    if traversal_result.paths:
        for idx, triple in enumerate(traversal_result.paths[0].triples):
            citations.append(CitationResult(
                triple_index=idx,
                source_uri=triple.source or None,
            ))

    # 10 — Entity linking warnings
    if not entity_candidates:
        warnings.append("No entities were matched in the Knowledge Graph.")

    # 11 — Update session memory
    resolved_entity_names = [c.name for c in entity_candidates]
    resolved_entity_ids   = [c.id   for c in entity_candidates]
    event_names: List[str] = []
    if traversal_result.paths:
        tp = traversal_result.paths[0]
        # nodes in the path that are 'Event' type
        event_names = [
            tp.node_names[i]
            for i, label in enumerate(
                getattr(tp, "node_labels", []) or []
            )
            if "event" in (label or "").lower()
        ]
    session.add_turn(
        query=req.message,
        answer=answer_text,
        entity_names=resolved_entity_names,
        entity_ids=resolved_entity_ids,
        event_names=event_names,
    )

    # 12 — Build response
    reasoning_paths = [
        ReasoningPathResult(
            path_id=p.path_id,
            hops=p.hops,
            triples=[
                TripleResult(
                    subj=t.subj, rel=t.rel, obj=t.obj,
                    time=t.time, source=t.source
                )
                for t in p.triples
            ],
            score=round(p.score, 4),
        )
        for p in traversal_result.paths
    ]

    debug_info: Optional[Dict] = None
    if req.debug:
        debug_info = {
            "decomposition": {
                "question_type": plan.question_type,
                "subquestions": [sq.text for sq in plan.subquestions],
            },
            "query_plan": traversal_result.query_plan,
            "execution_stats": {
                "traversal_ms": traversal_result.execution_ms,
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                "paths_found": len(traversal_result.paths),
            },
            "lang_detection": {
                "detected": lang_result.detected_lang,
                "is_mixed": lang_result.is_mixed,
            },
        }

    return ChatResponse(
        answer=answer_text,
        answer_lang=answer_lang,
        entities=[
            EntityResult(
                id=c.id, label=c.label, name=c.name, score=round(c.score, 4)
            )
            for c in entity_candidates
        ],
        reasoning_paths=reasoning_paths,
        explanation_steps=verified_steps,
        citations=citations,
        confidence=round(confidence, 4),
        warnings=warnings,
        debug=debug_info,
    )


# ── /resolve endpoint ─────────────────────────────────────────────────────

@router.post("/resolve", response_model=ResolveResponse)
async def resolve(req: ResolveRequest):
    candidates = _linker.resolve_mention(req.mention)
    return ResolveResponse(
        candidates=[
            EntityResult(
                id=c.id, label=c.label, name=c.name, score=round(c.score, 4)
            )
            for c in candidates
        ]
    )


# ── /translate endpoint ───────────────────────────────────────────────────

@router.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    """
    Translate any text between English and Hindi.
    Used by the frontend to translate UI text, news summaries, etc.

    Body:
        text        — text to translate
        source_lang — "en" or "hi" (default "en")
        target_lang — "en" or "hi" (default "hi")
    """
    result = translate(req.text, source_lang=req.source_lang, target_lang=req.target_lang)
    return TranslateResponse(
        translated=result,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
    )
