"""
neo4j_adapter.py — Neo4j (Cypher) implementation of KGAdapterBase.
Wraps the existing financial_kg Neo4jStorage + KGQueryEngine.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make financial_kg importable from the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from financial_kg.storage.neo4j_storage import Neo4jStorage  # noqa: E402
from .base import KGAdapterBase


class Neo4jAdapter(KGAdapterBase):
    """
    Thin adapter over Neo4jStorage.
    Translates KGAdapterBase calls into Cypher queries.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._storage = Neo4jStorage(uri=uri, username=username, password=password)

    # ── search_nodes ────────────────────────────────────────────────────────

    def search_nodes(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Case-insensitive substring search on node names."""
        rows = self._storage.execute_query(
            """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($text)
            RETURN n.id AS id, n.name AS name,
                   labels(n)[0] AS label,
                   size([(n)-[]-() | 1]) AS degree
            ORDER BY degree DESC
            LIMIT $top_k
            """,
            {"text": text, "top_k": top_k},
        )
        for r in rows:
            r.setdefault("match_type", "exact_substring")
        return rows

    # ── get_neighbors ────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        node_id: str,
        rel_types: Optional[List[str]] = None,
        direction: str = "both",
        time_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Return all direct neighbors of node_id with relationship metadata."""
        if direction == "out":
            pattern = "(n)-[r]->(m)"
            dir_expr = "'outgoing'"
        elif direction == "in":
            pattern = "(n)<-[r]-(m)"
            dir_expr = "'incoming'"
        else:
            pattern = "(n)-[r]-(m)"
            dir_expr = "CASE WHEN startNode(r).id = $nid THEN 'outgoing' ELSE 'incoming' END"

        # rel_types filter: use WHERE type(r) IN [...] — safe with any rel name
        params: Dict[str, Any] = {"nid": node_id, "limit": 30}
        rel_filter = ""
        if rel_types:
            params["rel_types"] = [rt.upper() for rt in rel_types]
            rel_filter = "AND type(r) IN $rel_types"

        # Temporal filter
        time_clauses = []
        if time_filter:
            if time_filter.get("after"):
                time_clauses.append(
                    "(r.t_announce >= $t_after OR r.t_effective >= $t_after)"
                )
                params["t_after"] = time_filter["after"]
            if time_filter.get("before"):
                time_clauses.append(
                    "(r.t_announce <= $t_before OR r.t_effective <= $t_before)"
                )
                params["t_before"] = time_filter["before"]
        time_filter_str = ("AND " + " AND ".join(time_clauses)) if time_clauses else ""

        cypher = f"""
        MATCH {pattern}
        WHERE n.id = $nid {rel_filter} {time_filter_str}
        RETURN
            n.id          AS from_id,
            n.name        AS from_name,
            type(r)       AS rel,
            r.id          AS rel_id,
            m.id          AS to_id,
            m.name        AS to_name,
            labels(m)[0]  AS to_label,
            {dir_expr}    AS direction,
            r.t_announce  AS t_announce,
            r.t_effective AS t_effective,
            r.sources     AS source,
            r.sentiment_label AS sentiment,
            r.confidence  AS confidence
        LIMIT $limit
        """
        return self._storage.execute_query(cypher, params)

    # ── find_paths ───────────────────────────────────────────────────────────

    def find_paths(
        self,
        start_node_ids: List[str],
        end_node_ids: Optional[List[str]],
        constraints: Optional[Dict] = None,
        max_hops: int = 3,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop path search using explicit hop-by-hop Cypher.

        Strategy (memory-safe for Aura free tier):
          - 1 hop : direct MATCH (a)-[r1]-(b)
          - 2 hops: MATCH (a)-[r1]-(mid)-[r2]-(b)
          - 3 hops: MATCH (a)-[r1]-(m1)-[r2]-(m2)-[r3]-(b)
        Each query is LIMIT-ed tightly so no full-graph scan occurs.
        rel_type filter uses WHERE type(r) IN $rel_types (no backtick syntax).
        """
        constraints = constraints or {}
        rel_types: Optional[List[str]] = constraints.get("rel_types")
        params_base: Dict[str, Any] = {"start_ids": start_node_ids, "top_k": top_k}
        if rel_types:
            params_base["rel_types"] = [rt.upper() for rt in rel_types]

        rel_filter_1 = "AND type(r1) IN $rel_types" if rel_types else ""
        rel_filter_2 = "AND type(r2) IN $rel_types" if rel_types else ""
        rel_filter_3 = "AND type(r3) IN $rel_types" if rel_types else ""

        end_filter = ""
        if end_node_ids:
            params_base["end_ids"] = end_node_ids
            end_filter = "AND b.id IN $end_ids"

        all_rows: List[Dict] = []

        # ── 1-hop ──────────────────────────────────────────────────────────
        q1 = f"""
        MATCH (a)-[r1]-(b)
        WHERE a.id IN $start_ids
          AND a.id <> b.id
          {rel_filter_1}
          {end_filter}
        RETURN
            [a.id, b.id]                        AS node_ids,
            [a.name, b.name]                    AS node_names,
            [labels(a)[0], labels(b)[0]]        AS node_labels,
            1                                   AS hops,
            [{{subj: a.name, rel: type(r1), obj: b.name,
               time: coalesce(r1.t_announce, r1.t_effective, null),
               source: r1.sources}}]            AS triples
        LIMIT $top_k
        """
        all_rows.extend(self._storage.execute_query(q1, dict(params_base)))

        # ── 2-hop ──────────────────────────────────────────────────────────
        if max_hops >= 2 and len(all_rows) < top_k:
            q2 = f"""
            MATCH (a)-[r1]-(mid)-[r2]-(b)
            WHERE a.id IN $start_ids
              AND a.id <> b.id
              AND a.id <> mid.id
              {rel_filter_1}
              {rel_filter_2}
              {end_filter}
            RETURN
                [a.id, mid.id, b.id]                                    AS node_ids,
                [a.name, mid.name, b.name]                              AS node_names,
                [labels(a)[0], labels(mid)[0], labels(b)[0]]           AS node_labels,
                2                                                        AS hops,
                [{{subj: a.name,   rel: type(r1), obj: mid.name,
                   time: coalesce(r1.t_announce, r1.t_effective, null),
                   source: r1.sources}},
                 {{subj: mid.name, rel: type(r2), obj: b.name,
                   time: coalesce(r2.t_announce, r2.t_effective, null),
                   source: r2.sources}}]                                 AS triples
            LIMIT $top_k
            """
            all_rows.extend(self._storage.execute_query(q2, dict(params_base)))

        # ── 3-hop ──────────────────────────────────────────────────────────
        if max_hops >= 3 and len(all_rows) < top_k:
            q3 = f"""
            MATCH (a)-[r1]-(m1)-[r2]-(m2)-[r3]-(b)
            WHERE a.id IN $start_ids
              AND a.id <> b.id
              AND a.id <> m1.id
              AND m1.id <> b.id
              {rel_filter_1}
              {rel_filter_2}
              {rel_filter_3}
              {end_filter}
            RETURN
                [a.id, m1.id, m2.id, b.id]                                       AS node_ids,
                [a.name, m1.name, m2.name, b.name]                               AS node_names,
                [labels(a)[0], labels(m1)[0], labels(m2)[0], labels(b)[0]]      AS node_labels,
                3                                                                  AS hops,
                [{{subj: a.name,  rel: type(r1), obj: m1.name,
                   time: coalesce(r1.t_announce, r1.t_effective, null),
                   source: r1.sources}},
                 {{subj: m1.name, rel: type(r2), obj: m2.name,
                   time: coalesce(r2.t_announce, r2.t_effective, null),
                   source: r2.sources}},
                 {{subj: m2.name, rel: type(r3), obj: b.name,
                   time: coalesce(r3.t_announce, r3.t_effective, null),
                   source: r3.sources}}]                                           AS triples
            LIMIT $top_k
            """
            all_rows.extend(self._storage.execute_query(q3, dict(params_base)))

        # Wrap into standard path dicts
        paths = []
        for i, row in enumerate(all_rows[:top_k]):
            paths.append({
                "path_id": f"p{i}",
                "hops":       row.get("hops", 1),
                "node_ids":   row.get("node_ids", []),
                "node_names": row.get("node_names", []),
                "node_labels":row.get("node_labels", []),
                "triples":    row.get("triples", []),
                "score": 0.0,
            })
        return paths

    def close(self) -> None:
        self._storage.close()
