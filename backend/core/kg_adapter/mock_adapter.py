"""
mock_adapter.py — In-memory KG adapter for unit tests.
Contains a tiny toy KG: Infosys, SEBI, RBI, Wipro, GDP.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import KGAdapterBase

# ── Toy KG ────────────────────────────────────────────────────────────────
_NODES: Dict[str, Dict] = {
    "infosys": {"id": "infosys", "name": "Infosys", "label": "Company", "degree": 5},
    "wipro":   {"id": "wipro",   "name": "Wipro",   "label": "Company", "degree": 4},
    "sebi":    {"id": "sebi",    "name": "SEBI",    "label": "Regulator","degree": 6},
    "rbi":     {"id": "rbi",     "name": "RBI",     "label": "Regulator","degree": 7},
    "gdp":     {"id": "gdp",     "name": "GDP",     "label": "Indicator","degree": 3},
    "it_sector": {"id": "it_sector", "name": "IT Sector", "label": "Sector", "degree": 4},
    "narayana_murthy": {"id": "narayana_murthy", "name": "N R Narayana Murthy",
                        "label": "Person", "degree": 2},
    "salil_parekh":    {"id": "salil_parekh",    "name": "Salil Parekh",
                        "label": "Person", "degree": 2},
}

_EDGES: List[Dict] = [
    {"from_id": "infosys", "from_name": "Infosys", "rel": "CEO_OF",
     "to_id": "salil_parekh", "to_name": "Salil Parekh", "to_label": "Person",
     "direction": "outgoing", "t_announce": "2018-01-02", "source": None},
    {"from_id": "infosys", "from_name": "Infosys", "rel": "OPERATES_IN",
     "to_id": "it_sector",   "to_name": "IT Sector",    "to_label": "Sector",
     "direction": "outgoing", "t_announce": None, "source": None},
    {"from_id": "wipro",   "from_name": "Wipro",   "rel": "OPERATES_IN",
     "to_id": "it_sector",   "to_name": "IT Sector",    "to_label": "Sector",
     "direction": "outgoing", "t_announce": None, "source": None},
    {"from_id": "sebi",    "from_name": "SEBI",    "rel": "REGULATES",
     "to_id": "infosys",     "to_name": "Infosys",      "to_label": "Company",
     "direction": "outgoing", "t_announce": None, "source": None},
    {"from_id": "sebi",    "from_name": "SEBI",    "rel": "REGULATES",
     "to_id": "wipro",       "to_name": "Wipro",        "to_label": "Company",
     "direction": "outgoing", "t_announce": None, "source": None},
    {"from_id": "rbi",     "from_name": "RBI",     "rel": "INFLUENCES",
     "to_id": "gdp",         "to_name": "GDP",          "to_label": "Indicator",
     "direction": "outgoing", "t_announce": "2024-04-01","source": None},
    {"from_id": "narayana_murthy", "from_name": "N R Narayana Murthy",
     "rel": "FOUNDED", "to_id": "infosys", "to_name": "Infosys", "to_label": "Company",
     "direction": "outgoing", "t_announce": "1981-07-02", "source": None},
]


class MockAdapter(KGAdapterBase):
    """In-memory adapter for unit tests. No external dependencies."""

    def search_nodes(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = text.lower()
        results = [
            {**n, "match_type": "mock_exact"}
            for n in _NODES.values()
            if q in n["name"].lower()
        ]
        return results[:top_k]

    def get_neighbors(
        self,
        node_id: str,
        rel_types: Optional[List[str]] = None,
        direction: str = "both",
        time_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        out = []
        for e in _EDGES:
            if direction in ("both", "out") and e["from_id"] == node_id:
                out.append({**e})
            elif direction in ("both", "in") and e["to_id"] == node_id:
                flipped = {**e, "direction": "incoming"}
                out.append(flipped)
        if rel_types:
            rts = {r.upper() for r in rel_types}
            out = [e for e in out if e["rel"].upper() in rts]
        return out

    def find_paths(
        self,
        start_node_ids: List[str],
        end_node_ids: Optional[List[str]],
        constraints: Optional[Dict] = None,
        max_hops: int = 3,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """BFS path search on the toy graph."""
        paths: List[Dict] = []
        for start_id in start_node_ids:
            self._bfs(start_id, end_node_ids, max_hops, paths)
        # deduplicate by node sequence
        seen = set()
        unique = []
        for p in paths:
            key = tuple(p["node_ids"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique[:top_k]

    def _bfs(self, start_id: str, targets, max_hops: int, acc: List):
        from collections import deque
        queue = deque()
        queue.append((start_id, [start_id], []))  # (current_id, node_path, edge_path)
        while queue:
            cur, node_path, edge_path = queue.popleft()
            if len(edge_path) > max_hops:
                continue
            if len(edge_path) > 0 and (targets is None or cur in (targets or [])):
                start_node = _NODES.get(node_path[0], {})
                end_node   = _NODES.get(cur, {})
                triples = [
                    {"subj": e["from_name"], "rel": e["rel"],
                     "obj": e["to_name"], "time": e.get("t_announce"),
                     "source": e.get("source")}
                    for e in edge_path
                ]
                acc.append({
                    "path_id": f"mock_{len(acc)}",
                    "hops": len(edge_path),
                    "node_ids": list(node_path),
                    "node_names": [_NODES.get(n, {}).get("name", n) for n in node_path],
                    "node_labels": [_NODES.get(n, {}).get("label", "?") for n in node_path],
                    "triples": triples,
                    "score": 0.0,
                })
            for edge in _EDGES:
                if edge["from_id"] == cur and edge["to_id"] not in node_path:
                    queue.append((edge["to_id"], node_path + [edge["to_id"]], edge_path + [edge]))

    def close(self) -> None:
        pass
