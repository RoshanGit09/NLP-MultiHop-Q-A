"""
memory.py — Session memory for multi-turn conversation.

Stores per-session resolved entities, events, and recent turns.
Provides coreference resolution context for the decomposer.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Deque, Dict, List, Optional


@dataclass
class Turn:
    query: str
    answer: str
    entities: List[str]         # entity names resolved in this turn
    entity_ids: List[str]       # corresponding KG node IDs
    event_names: List[str]      # event node names
    timestamp: float = field(default_factory=time.time)


class SessionMemory:
    """
    Lightweight per-session memory.
    Thread-safe for concurrent FastAPI requests.

    Key context exported to decomposer:
      last_entity_name  — most recently resolved company/entity
      last_event_name   — most recently resolved event
      last_entities     — list of entity names from recent turns
    """

    def __init__(self, max_turns: int = 10, entity_ttl_turns: int = 5):
        self._max_turns = max_turns
        self._entity_ttl = entity_ttl_turns
        self._turns: Deque[Turn] = deque(maxlen=max_turns)
        self._lock = Lock()

    def add_turn(
        self,
        query: str,
        answer: str,
        entity_names: List[str],
        entity_ids: List[str],
        event_names: Optional[List[str]] = None,
    ) -> None:
        turn = Turn(
            query=query,
            answer=answer,
            entities=entity_names,
            entity_ids=entity_ids,
            event_names=event_names or [],
        )
        with self._lock:
            self._turns.append(turn)

    def get_context(self) -> Dict[str, Any]:
        """Return context dict for the decomposer's coreference resolution."""
        with self._lock:
            recent = list(self._turns)[-self._entity_ttl:]

        entity_names: List[str] = []
        entity_ids: List[str] = []
        event_names: List[str] = []

        for turn in reversed(recent):
            entity_names.extend(turn.entities)
            entity_ids.extend(turn.entity_ids)
            event_names.extend(turn.event_names)

        # Deduplicate preserving order
        def _dedup(lst):
            seen = set()
            return [x for x in lst if not (x in seen or seen.add(x))]

        entity_names = _dedup(entity_names)
        entity_ids   = _dedup(entity_ids)
        event_names  = _dedup(event_names)

        return {
            "last_entity_name": entity_names[0] if entity_names else None,
            "last_entity_id":   entity_ids[0]   if entity_ids   else None,
            "last_event_name":  event_names[0]  if event_names  else None,
            "last_entities":    entity_names[:3],
            "last_entity_ids":  entity_ids[:3],
            "turn_count":       len(self._turns),
        }

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()


class SessionStore:
    """
    Global in-memory store for all sessions.
    Sessions expire after TTL (seconds).
    """

    def __init__(self, session_ttl: float = 3600.0):
        self._sessions: Dict[str, SessionMemory] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = session_ttl
        self._lock = Lock()

    def get(self, session_id: str) -> SessionMemory:
        with self._lock:
            self._evict_expired()
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionMemory()
            self._timestamps[session_id] = time.time()
            return self._sessions[session_id]

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, ts in self._timestamps.items()
            if now - ts > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            del self._timestamps[sid]


# Singleton
_session_store = SessionStore()


def get_session(session_id: str) -> SessionMemory:
    return _session_store.get(session_id)
