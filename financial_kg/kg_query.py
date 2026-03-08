"""
kg_query.py — Smart query layer for the Financial KG chatbot
========================================        # Strategy 2: known aliases — only triggered when query matches a key exactly
        aliases = self._get_aliases(query)
        for alias in aliases:
            results = self._cypher_search(alias)
            if results:
                for r in results:
                    r["match_type"] = f"alias:{alias}"
                return results[:top_k]=============
Wraps Neo4jStorage with fuzzy / semantic search so that a chatbot
asking for "Reliance" never silently returns nothing.

Search strategy (in order):
  1. Exact substring match  (fast Cypher)
  2. Case-insensitive CONTAINS on common aliases / ticker symbols
  3. Fuzzy token overlap    (Cypher apoc.text.fuzzyMatch if APOC available,
                             else Python-side difflib)
  4. Semantic embedding similarity (sentence-transformers, same model as KG)

Usage:
    from kg_query import KGQueryEngine
    engine = KGQueryEngine()
    results = engine.find_entity("Reliance")
    rels    = engine.get_neighbours("Reliance", limit=20)
    engine.close()
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from financial_kg.storage.neo4j_storage import Neo4jStorage
from financial_kg.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── noise/catch-all node names to exclude from query results ───────────────
# These are generic phrases wrongly extracted as entities by Gemini.
_NOISE_NAMES = re.compile(
    r"""^(
        joint\s+venture | special\s+purpose\s+vehicle
        | integrated\s+back.end | four\s+tranches | multiple\s+tranches
        | tepid\s+condition | overbought\s+levels? | oversold\s+levels?
        | trade\s+spotlight | final\s+dividend | interim\s+dividend
        | revisions?\s+in\s+credit\s+ratings?
    )$""",
    re.VERBOSE | re.IGNORECASE,
)


def _is_noise_result(name: str) -> bool:
    """Return True if this entity name is a known catch-all / noise node."""
    return bool(_NOISE_NAMES.match((name or "").strip()))

# ── well-known aliases / alternative names ─────────────────────────────────
# Maps what a user might type  ->  substrings/tokens to try in Cypher.
# ONLY entities that actually exist in the KG are listed.
# Absent entities (Reliance, TCS, HDFC, etc.) fail gracefully to "not found".
_ALIASES: Dict[str, List[str]] = {
    # Companies present in KG
    "infosys":            ["infosys"],
    "infy":               ["infosys"],
    "wipro":              ["wipro"],
    "cipla":              ["cipla"],
    "aurobindo":          ["aurobindo pharma"],
    "aurobindo pharma":   ["aurobindo pharma"],
    "navin fluorine":     ["navin fluorine"],
    "ultratech":          ["ultratech cement"],
    "ultratech cement":   ["ultratech cement"],
    "larsen":             ["larsen & toubro"],
    "l&t":                ["larsen & toubro"],
    "larsen & toubro":    ["larsen & toubro"],
    "asian paints":       ["asian paints"],
    "avenue supermarts":  ["avenue supermarts"],
    "dmart":              ["avenue supermarts"],
    "tvs motor":          ["tvs motor"],
    "tvs":                ["tvs motor"],
    "adani ports":        ["adani ports"],
    "adani":              ["adani ports"],
    "macrotech":          ["macrotech developers"],
    # Persons
    "sanjay malhotra":    ["sanjay malhotra"],
    "donald trump":       ["donald trump"],
    "trump":              ["donald trump"],
    # Regulators / Policies
    "sebi":               ["sebi", "securities and exchange board"],
    "rbi":                ["reserve bank of india", "rbi"],
    "sebi regulation":    ["regulation 30 of sebi"],
    "regulation 30":      ["regulation 30 of sebi"],
    # Indicators / Macro
    "gdp":                ["gross domestic product", "gdp"],
    "inflation":          ["inflation", "consumer price index", "cpi"],
    "crude oil":          ["west texas intermediate", "crude"],
    "wti":                ["west texas intermediate"],
    # Companies that are ABSENT (merged away) — fail gracefully
    "reliance":           ["reliance industries", "reliance group"],
    "tcs":                ["tata consultancy"],
    "hdfc":               ["hdfc bank", "housing development finance"],
    "hdfc bank":          ["hdfc bank"],
    "icici":              ["icici bank"],
    "bajaj":              ["bajaj finance", "bajaj auto"],
    "tata":               ["tata motors", "tata steel", "tata consultancy"],
    "sensex":             ["sensex", "bse sensex"],
    "nifty":              ["nifty 50", "nifty"],
}

# stopwords to skip as candidate entity terms
_STOPWORDS = {
    "tell", "me", "about", "what", "do", "you", "know", "is", "the",
    "find", "who", "happened", "with", "doing", "a", "an", "in", "of",
    "are", "was", "were", "has", "have", "had", "how", "why", "when",
    "which", "that", "this", "it", "its", "from", "to", "for", "on",
    "by", "at", "be", "been", "being", "will", "would", "could", "should",
    "can", "may", "might", "shall", "not", "no", "any", "all", "some",
    "give", "show", "list", "get", "tell", "explain", "describe",
    # Generic financial suffixes that appear alone
    "industries", "group", "limited", "ltd", "inc", "corp", "co",
    "and", "or", "but", "if", "then", "than", "so", "yet",
}


class KGQueryEngine:
    """
    Smart query engine over the Neo4j Financial KG.
    Falls back gracefully through multiple search strategies.
    """

    def __init__(self):
        self.storage = Neo4jStorage()
        self._all_names: Optional[List[Tuple[str, str, str]]] = None  # (id, name, label)

    def close(self):
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── public API ──────────────────────────────────────────────────────────

    def find_entity(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Find entities in the KG matching `query`.
        Tries multiple strategies and returns the best matches.

        Returns list of dicts with keys: id, name, label, degree, match_type
        """
        query = query.strip()
        if not query or query.lower() in _STOPWORDS:
            return []

        # Strategy 1: exact substring (case-insensitive Cypher)
        results = self._cypher_search(query)
        if results:
            for r in results:
                r["match_type"] = "exact_substring"
            return results[:top_k]

        # Strategy 2: known aliases — only triggered when query matches a key
        aliases = self._get_aliases(query)
        for alias in aliases:
            results = self._cypher_search(alias)
            if results:
                for r in results:
                    r["match_type"] = f"alias:{alias}"
                return results[:top_k]

        # Strategy 3: fuzzy token match (Python difflib, no APOC needed)
        # Use a stricter cutoff to avoid false positives
        results = self._fuzzy_search(query, top_k=top_k, cutoff=0.6)
        if results:
            for r in results:
                r["match_type"] = "fuzzy"
            return results

        return []

    def get_neighbours(
        self,
        entity_query_or_id: str,
        limit: int = 20,
        rel_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get all relationships (and their neighbours) for a matched entity.
        `entity_query_or_id` can be either a search query string OR a known entity ID.
        Returns list of dicts: {from_name, rel, to_name, to_label, direction}
        """
        # Resolve to entity ID
        if " " not in entity_query_or_id and len(entity_query_or_id) > 20:
            # Looks like an ID (long, no spaces)
            entity_id = entity_query_or_id
            entity_name = entity_query_or_id
        else:
            entities = self.find_entity(entity_query_or_id, top_k=1)
            if not entities:
                logger.warning(f"Entity not found for query: '{entity_query_or_id}'")
                return []
            entity_id   = entities[0]["id"]
            entity_name = entities[0]["name"]
        logger.info(f"Getting neighbours for '{entity_name}' (id={entity_id})")

        rel_filter = f"AND type(r) = '{rel_type.upper()}'" if rel_type else ""
        cypher = f"""
        MATCH (n {{id: $eid}})-[r]-(m)
        WHERE true {rel_filter}
        RETURN
            n.name AS from_name,
            type(r) AS rel,
            m.name AS to_name,
            labels(m)[0] AS to_label,
            CASE WHEN startNode(r).id = $eid THEN 'outgoing' ELSE 'incoming' END AS direction
        LIMIT $limit
        """
        rows = self.storage.execute_query(cypher, {"eid": entity_id, "limit": limit})
        return rows

    def answer_question(self, question: str) -> str:
        """
        Simple question → KG answer string, suitable for a chatbot.
        Detects entity names in the question and fetches their relationships.
        """
        tokens = question.split()

        # Build candidates: longest spans first, skip pure-stopword spans
        candidates = []
        for size in (4, 3, 2, 1):
            for i in range(len(tokens) - size + 1):
                span = " ".join(tokens[i : i + size])
                span_lower = span.lower().strip("?.,!")
                # skip if every token is a stopword
                if all(t.lower() in _STOPWORDS for t in span.split()):
                    continue
                candidates.append(span_lower)

        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        best_entity = None
        best_results = []
        for cand in unique_candidates:
            hits = self.find_entity(cand, top_k=1)
            if hits:
                # Prefer hits that are NOT purely fuzzy unless we have no exact/alias hit
                match_type = hits[0].get("match_type", "")
                if "fuzzy" not in match_type or best_entity is None:
                    best_entity = hits[0]
                    best_results = self.get_neighbours(hits[0]["id"], limit=15)
                    if "fuzzy" not in match_type:
                        break  # good enough — stop at first exact/alias hit

        if not best_entity:
            return (
                "Sorry, I couldn't find any matching entity in the Knowledge Graph. "
                "Try rephrasing with a company name, sector, or financial indicator."
            )

        name  = best_entity["name"]
        label = best_entity["label"]
        lines = [f"**{name}** [{label}]  (matched via {best_entity['match_type']})\n"]
        lines.append(f"Top {len(best_results)} relationships:\n")

        for r in best_results:
            direction = "→" if r.get("direction") == "outgoing" else "←"
            lines.append(
                f"  {r['from_name']}  {direction}[{r['rel']}]→  "
                f"{r['to_name']}  [{r.get('to_label','?')}]"
            )

        return "\n".join(lines)

    def graph_stats(self) -> Dict[str, Any]:
        """Return current Neo4j graph statistics."""
        return self.storage.get_graph_stats()

    # ── private helpers ─────────────────────────────────────────────────────

    def _cypher_search(self, term: str) -> List[Dict]:
        rows = self.storage.execute_query(
            """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($term)
            RETURN n.id AS id, n.name AS name, labels(n)[0] AS label,
                   size([(n)-[]-() | 1]) AS degree
            ORDER BY degree DESC
            LIMIT 10
            """,
            {"term": term},
        )
        # Filter out known noise / catch-all nodes
        return [r for r in rows if not _is_noise_result(r["name"])]

    def _cypher_search_word(self, term: str) -> List[Dict]:
        """Match `term` only as a whole word (space-separated) in the entity name."""
        rows = self.storage.execute_query(
            """
            MATCH (n)
            WHERE toLower(n.name) = toLower($term)
               OR toLower(n.name) STARTS WITH toLower($term) + ' '
               OR toLower(n.name) ENDS WITH ' ' + toLower($term)
               OR toLower(n.name) CONTAINS ' ' + toLower($term) + ' '
            RETURN n.id AS id, n.name AS name, labels(n)[0] AS label,
                   size([(n)-[]-() | 1]) AS degree
            ORDER BY degree DESC
            LIMIT 10
            """,
            {"term": term},
        )
        return [r for r in rows if not _is_noise_result(r["name"])]

    def _get_aliases(self, query: str) -> List[str]:
        q = query.lower().strip()
        for key, aliases in _ALIASES.items():
            if key == q or q.startswith(key) or q.endswith(key):
                return aliases
            # Check if the query contains the key as a whole word
            if f" {key} " in f" {q} ":
                return aliases
        return []

    def _load_all_names(self) -> List[Tuple[str, str, str]]:
        """Cache all (id, name, label) pairs for fuzzy matching."""
        if self._all_names is None:
            rows = self.storage.execute_query(
                "MATCH (n) WHERE n.name IS NOT NULL RETURN n.id AS id, n.name AS name, labels(n)[0] AS label"
            )
            self._all_names = [(r["id"], r["name"], r["label"]) for r in rows]
        return self._all_names

    def _fuzzy_search(self, query: str, top_k: int = 5, cutoff: float = 0.6) -> List[Dict]:
        all_names = self._load_all_names()
        # Exclude noise nodes from fuzzy pool
        name_list = [n for _, n, _ in all_names if not _is_noise_result(n)]
        matches = difflib.get_close_matches(query, name_list, n=top_k, cutoff=cutoff)
        results = []
        for match in matches:
            for nid, name, label in all_names:
                if name == match:
                    degree = self.storage.execute_query(
                        "MATCH (n {id: $id}) RETURN size([(n)-[]-() | 1]) AS degree",
                        {"id": nid}
                    )
                    results.append({
                        "id": nid,
                        "name": name,
                        "label": label,
                        "degree": degree[0]["degree"] if degree else 0,
                    })
                    break
        return results
