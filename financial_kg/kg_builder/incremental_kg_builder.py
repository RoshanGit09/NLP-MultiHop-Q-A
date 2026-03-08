"""
IncrementalKGBuilder
====================
Handles redundancy in iText2KG-style extraction by merging new extractions
into a persistent NetworkX DiGraph using STRING-SIMILARITY deduplication.

Pipeline per build_incremental() call
--------------------------------------
1.  (Optional) Load prev_kg — populate self.global_kg & self.global_entities.
2.  Call iText2KG adapter (GeminiAtomicExtractor) → raw dicts.
3.  String-similarity union-find dedup (SequenceMatcher, same-label only).
     - Threshold: str_threshold = max(0.85, self._eps)
     - Cross-label merges are NEVER allowed (Company ≠ Person etc.)
     - Generic/noisy phrases (< 3 chars, pure numbers, known noise) skipped.
4.  For each cluster pick best canonical entity (highest confidence, prefers
    longer meaningful names over single-word abbreviations unless confidence
    of the short form is clearly higher).
5.  MERGE into self.global_kg:
        node  → add_node if new; always update attrs (union, max confidence).
        edge  → add_edge if both endpoint nodes exist, NO self-loops.
6.  Persist updated self.global_entities list (no embeddings stored).

Key design decisions
---------------------
- NO sentence-transformer embeddings for dedup: all-MiniLM-L6-v2 encodes
  short company names (Reliance, TCS, Aurobindo Pharma) to nearly the same
  cosine-space vector, causing catastrophic over-merging at eps=0.55.
- String similarity (SequenceMatcher) is safe: "Reliance Industries" vs
  "Aurobindo Pharma" ratio ≈ 0.25 → never merged.
- Self-loops are silently dropped (entity pointing at itself).
- Duplicate edges (same src, type, tgt) are deduplicated.
"""

from __future__ import annotations

import collections
import difflib
import hashlib
import io
import asyncio
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# Suppress noisy output from sentence-transformers v5+ / huggingface_hub.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _noisy in (
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "transformers",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# Module-level model cache: kept for backward compat but not used in string dedup.
# SentenceTransformer is lazily imported only if explicitly requested.
_MODEL_CACHE: dict = {}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _entity_rep(entity: dict) -> str:
    """Canonical text representation used for embedding."""
    name = entity.get("name", "").strip()
    label = entity.get("label", "").strip()
    return f"{name} ({label})" if label else name


def _stable_id(name: str, label: str) -> str:
    """
    Generate a stable, collision-resistant ID from name + label.
    Format: ``{slug}_{8-char-sha256}``
    """
    slug = name.strip().upper().replace(" ", "_").replace("-", "_")[:32]
    digest = hashlib.sha256(f"{name.lower().strip()}::{label.lower().strip()}".encode()).hexdigest()[:8]
    return f"{slug}_{digest}"


def _merge_attrs(existing: dict, incoming: dict) -> dict:
    """
    Union two attribute dicts.
    - confidence  → take the maximum
    - other keys  → keep existing unless None/missing, then take incoming
    """
    merged = dict(existing)
    for key, val in incoming.items():
        if key == "confidence":
            merged["confidence"] = max(
                float(merged.get("confidence") or 0.0),
                float(val or 0.0),
            )
        elif merged.get(key) is None:
            merged[key] = val
    return merged


# Generic/noise entity names that Gemini sometimes extracts as entities but
# are not real named entities.  These should never be merged with real names.
# IMPORTANT: patterns must NOT match real acronyms like TCS, NSE, BSE, RBI.
# Rule: uppercase-only short strings (e.g. "TCS") are real; lowercase-only
# short strings (e.g. "buy", "abc") are noise.
_NOISE_PATTERNS = re.compile(
    r"""^(
        # Pure numbers / percentages / currency fragments
        [\d\s%.,/-]+
        | \d+(\.\d+)?\s*(million|billion|crore|lakh|thousand|tons?|kg|usd|inr|year|quarter|fy\d*)?
        # Currency symbol followed by amount (e.g. "Rs.", "rs. 100")
        | rs\.?\s*[\d.,]*
        # Generic business phrases (multi-word, clearly not a named entity)
        | joint\s+venture | special\s+purpose\s+vehicle | spv
        | integrated\s+back[\s\-]?end | back[\s\-]?end | front[\s\-]?end
        | final\s+dividend | interim\s+dividend
        | trade\s+spotlight | market\s+update | press\s+release
        | four\s+tranches | multiple\s+tranches | tranches?
        # Financial jargon without a proper noun (standalone action words)
        | (buy|sell|hold|neutral|overweight|underweight)\s*$
        | (overbought|oversold|tepid)\s*(condition|level|zone)?
        # Single ALL-LOWERCASE words of 1-4 chars (not acronyms like TCS/NSE/BSE)
        # Acronyms are UPPERCASE so they will NOT match this branch
        | [a-z]{1,4}
    )$""",
    re.VERBOSE,  # NOTE: no IGNORECASE — uppercase acronyms must NOT match
)


def _is_noise_entity(name: str) -> bool:
    """Return True if `name` looks like a generic phrase, not a real named entity.

    Uppercase acronyms like TCS, NSE, BSE, RBI are NOT noise.
    Lowercase short words like 'buy', 'abc', 'rs.' ARE noise.
    """
    if not name or len(name.strip()) < 2:
        return True
    return bool(_NOISE_PATTERNS.match(name.strip()))


# ---------------------------------------------------------------------------
# iText2KG adapter (wraps GeminiAtomicExtractor)
# ---------------------------------------------------------------------------

class _IText2KGAdapter:
    """
    Thin synchronous wrapper around GeminiAtomicExtractor that exposes
    the ``build_graph(sections)`` interface expected by IncrementalKGBuilder.

    Returns
    -------
    global_ent : List[dict]  – ``{'name', 'label', 'confidence', 'id'}``
    global_rel : List[dict]  – ``{'name', 'startNode', 'endNode'}``
    """

    def __init__(self):
        # Lazy import to avoid circular deps and keep optional
        try:
            import asyncio
            from ..extractors.gemini_atomic_extractor import GeminiAtomicExtractor
            self._extractor = GeminiAtomicExtractor()
            self._loop = asyncio.new_event_loop()
        except Exception as exc:
            raise RuntimeError(f"Cannot initialise GeminiAtomicExtractor: {exc}") from exc

    def build_graph(
        self, sections: List[str]
    ) -> Tuple[List[dict], List[dict]]:
        """
        Process all sections in PARALLEL using asyncio.gather, then aggregate.
        All sections in the batch are sent to Gemini simultaneously instead of
        one-by-one — gives ~10x speedup when batch_size=10.
        """
        all_entities: List[dict] = []
        all_rels: List[dict] = []

        async def _extract_all():
            tasks = [
                self._extractor.extract_entities_and_relationships(sec)
                for sec in sections
                if sec.strip()
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = self._loop.run_until_complete(_extract_all())

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Section extraction failed: {result}")
                continue
            raw_entities, raw_rels = result
            for e in raw_entities:
                all_entities.append({
                    "name":       e.name,
                    "label":      e.type,
                    "confidence": float(e.confidence),
                    "id":         e.id,
                })
            for r in raw_rels:
                all_rels.append({
                    "name":      r.predicate,
                    "startNode": r.subject.id if hasattr(r.subject, "id") else str(r.subject),
                    "endNode":   r.object.id  if hasattr(r.object,  "id") else str(r.object),
                })

        logger.info(
            f"[iText2KG adapter] extracted {len(all_entities)} entities, "
            f"{len(all_rels)} relationships from {len(sections)} sections"
        )
        return all_entities, all_rels


# ---------------------------------------------------------------------------
# IncrementalKGBuilder
# ---------------------------------------------------------------------------

class IncrementalKGBuilder:
    """
    Incremental Knowledge-Graph builder with embedding-based deduplication.

    Parameters
    ----------
    itext2kg : optional
        Any object with a ``build_graph(sections) -> (ents, rels)`` method.
        Defaults to the GeminiAtomicExtractor-backed ``_IText2KGAdapter``.
    embed_model : str
        SentenceTransformer model name (default: ``all-MiniLM-L6-v2``).
    eps : float
        DBSCAN epsilon (cosine distance threshold for merging, default 0.7).
    min_samples : int
        DBSCAN min_samples (1 = every point gets a cluster, default 1).

    State
    -----
    self.global_kg         : nx.DiGraph  – the persistent knowledge graph
    self.global_entities   : List[str]   – canonical reps ``"name (label)"``
    self._node_embeddings  : np.ndarray  – embeddings for global_entities,
                                          rows correspond 1-to-1 with
                                          self.global_entities
    """

    def __init__(
        self,
        itext2kg=None,
        embed_model: str = "all-MiniLM-L6-v2",
        eps: float = 0.85,
        min_samples: int = 1,
    ):
        # iText2KG interface
        if itext2kg is None:
            itext2kg = _IText2KGAdapter()
        self._itext2kg = itext2kg

        # NOTE: SentenceTransformer is no longer used for deduplication.
        # String-similarity dedup (SequenceMatcher) is used instead to avoid
        # catastrophic over-merging of short company names in cosine space.
        # embed_model param kept for API compatibility but model is NOT loaded.
        self._embed_model = None  # unused
        self._eps = eps           # reused as string-similarity threshold
        self._min_samples = min_samples

        # Persistent state
        self.global_kg: nx.DiGraph = nx.DiGraph()
        self.global_entities: List[str] = []          # rep strings
        self._node_embeddings: np.ndarray = np.empty((0, 384))  # unused placeholder

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build_incremental(
        self,
        sections: List[str],
        prev_kg: Optional[nx.DiGraph] = None,
    ) -> nx.DiGraph:
        """
        Incrementally extend the knowledge graph with the entities and
        relationships extracted from *sections*.

        Parameters
        ----------
        sections : List[str]
            Text sections to extract from.
        prev_kg : nx.DiGraph, optional
            A previously saved graph.  When provided it is loaded into
            ``self.global_kg`` and ``self.global_entities`` is rebuilt
            from its node attributes before processing starts.

        Returns
        -------
        nx.DiGraph
            Updated ``self.global_kg``.
        """
        # ---- Step 1: optionally load a previously persisted KG ----------
        if prev_kg is not None:
            self._load_prev_kg(prev_kg)

        # ---- Step 2: run iText2KG extraction ----------------------------
        logger.info("Running iText2KG extraction …")
        global_ent, global_rel = self._itext2kg.build_graph(sections=sections)

        if not global_ent:
            logger.warning("No entities extracted — skipping merge.")
            return self.global_kg

        # ---- Step 3: string-similarity deduplication --------------------
        # We intentionally do NOT use sentence-transformer embeddings here.
        # all-MiniLM-L6-v2 maps short company names (e.g. "Reliance Industries",
        # "TCS", "Aurobindo Pharma") to nearly identical vectors in cosine space,
        # causing massive over-merging.  String-similarity (SequenceMatcher) only
        # merges when names are genuinely close textually (≥ str_threshold).

        str_threshold = max(0.85, self._eps)   # always at least 85% string sim

        # Build combined list: existing global entities first, then new entities
        combined_ent: List[dict] = []
        rep_to_id: Dict[str, str] = {
            self._global_rep(nid): nid for nid in self.global_kg.nodes
        }
        for rep in self.global_entities:
            nid = rep_to_id.get(rep)
            if nid:
                combined_ent.append(dict(self.global_kg.nodes[nid]))
            else:
                combined_ent.append({"name": rep, "label": "", "confidence": 0.0})
        n_global = len(combined_ent)
        combined_ent.extend(list(global_ent))   # new entities appended after

        # Union-find
        parent = list(range(len(combined_ent)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int):
            parent[_find(x)] = _find(y)

        # Shared suffix pattern used in candidate checks
        _suffix_re = re.compile(r'\s+(limited|ltd\.?|inc\.?|corp\.?|pvt\.?|plc\.?)$')

        def _should_merge(na: str, nb: str) -> bool:
            """Return True if na and nb are close enough to deduplicate."""
            na_l, nb_l = na.lower().strip(), nb.lower().strip()
            # Strategy A: SequenceMatcher ratio (catches spelling variants)
            sim = difflib.SequenceMatcher(None, na_l, nb_l).ratio()
            if sim >= str_threshold:
                return True
            # Strategy B: prefix match — "Infosys" is a prefix of "Infosys Limited"
            short, long = (na_l, nb_l) if len(na_l) < len(nb_l) else (nb_l, na_l)
            base = _suffix_re.sub('', long).strip()
            return len(short) >= 4 and short == base

        # ---------------------------------------------------------------
        # OPTIMISED DEDUP — O(N*K) instead of O(N²)
        # ---------------------------------------------------------------
        # Phase 1: dedup new entities among themselves (small set ~2k)
        # Phase 2: match each new entity against global KG using a prefix
        #          index to prune candidates from O(N_global) to O(K)
        # ---------------------------------------------------------------

        # Group by label for cross-label safety
        new_by_label:    Dict[str, List[int]] = collections.defaultdict(list)
        global_by_label: Dict[str, List[int]] = collections.defaultdict(list)
        for idx in range(n_global):
            lbl = combined_ent[idx].get("label", "")
            global_by_label[lbl].append(idx)
        for idx in range(n_global, len(combined_ent)):
            lbl = combined_ent[idx].get("label", "")
            new_by_label[lbl].append(idx)

        # Build prefix index per label over global entities:
        # key = (label, first-4-chars-of-lowercased-name) → list of global indices
        # This limits each new entity's candidate set to names sharing the same prefix.
        prefix_index: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
        for lbl, idxs in global_by_label.items():
            for idx in idxs:
                name = combined_ent[idx].get("name", "").lower().strip()
                if name and not _is_noise_entity(combined_ent[idx].get("name", "")):
                    prefix_index[(lbl, name[:4])].append(idx)

        # Phase 1: new vs new (O(M²) where M ≈ 2k new entities per batch)
        for lbl, indices in new_by_label.items():
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    ia, ib = indices[a], indices[b]
                    if _find(ia) == _find(ib):
                        continue
                    na = combined_ent[ia].get("name", "")
                    nb = combined_ent[ib].get("name", "")
                    if _is_noise_entity(na) or _is_noise_entity(nb):
                        continue
                    if _should_merge(na, nb):
                        _union(ia, ib)
                        logger.debug(f"String-dedup MERGE (new-new) [{lbl}] '{na}' ≈ '{nb}'")

        # Phase 2: new vs global — use prefix index to limit candidates
        for lbl, new_indices in new_by_label.items():
            for ia in new_indices:
                na = combined_ent[ia].get("name", "")
                if _is_noise_entity(na):
                    continue
                na_l = na.lower().strip()
                # Gather candidates sharing any of the first 1-4 char prefixes
                candidates: List[int] = []
                for plen in (4, 3):
                    if len(na_l) >= plen:
                        candidates.extend(prefix_index.get((lbl, na_l[:plen]), []))
                # Deduplicate candidate list
                seen_cands: set = set()
                for ib in candidates:
                    if ib in seen_cands or _find(ia) == _find(ib):
                        continue
                    seen_cands.add(ib)
                    nb = combined_ent[ib].get("name", "")
                    if _is_noise_entity(nb):
                        continue
                    if _should_merge(na, nb):
                        _union(ia, ib)
                        logger.debug(f"String-dedup MERGE (new-global) [{lbl}] '{na}' ≈ '{nb}'")

        # Build cluster_map from union-find
        cluster_map: Dict[int, List[int]] = collections.defaultdict(list)
        for idx in range(len(combined_ent)):
            cluster_map[_find(idx)].append(idx)

        # ---- Step 4: resolve clusters → canonical entity ----------------
        old_id_to_canonical: Dict[str, str] = {}

        for root_idx, member_indices in cluster_map.items():
            members_with_dicts = [(idx, combined_ent[idx]) for idx in member_indices]

            # Canonical selection: pick best entity by score
            # Score = confidence * word_count_bonus (prefer multi-word real names
            # over single-character abbreviations, but don't over-penalize short names)
            def _canon_score(d: dict) -> float:
                name = (d.get("name") or "").strip()
                conf = float(d.get("confidence") or 0.0)
                words = len(name.split())
                # Bonus for multi-word names (more likely to be the real full name)
                word_bonus = min(words / 2.0, 1.5)
                # Penalty for very short (1-char) names  
                short_penalty = 0.5 if len(name) <= 2 else 1.0
                # Prefer entities from global (already stable) over new ones
                is_global = 1.1 if member_indices.index(root_idx) < n_global else 1.0  # type: ignore[attr-defined]
                return conf * word_bonus * short_penalty

            canonical_dict = max(members_with_dicts, key=lambda t: _canon_score(t[1]))[1]

            # Ensure non-empty label
            if not canonical_dict.get("label"):
                for _, d in members_with_dicts:
                    if d.get("label"):
                        canonical_dict = d
                        break

            canon_name  = canonical_dict.get("name", "").strip()
            canon_label = canonical_dict.get("label", "").strip()
            canon_id    = _stable_id(canon_name, canon_label)

            # Build merged attrs from all cluster members
            merged_attrs: dict = {
                "id":         canon_id,
                "name":       canon_name,
                "label":      canon_label,
                "confidence": float(canonical_dict.get("confidence") or 0.0),
            }
            for _, member_dict in members_with_dicts:
                merged_attrs = _merge_attrs(merged_attrs, member_dict)

            # Register reverse mapping for every member
            for _, member_dict in members_with_dicts:
                old_id = member_dict.get("id", "")
                if old_id:
                    old_id_to_canonical[old_id] = canon_id
                member_stable = _stable_id(
                    member_dict.get("name", ""),
                    member_dict.get("label", ""),
                )
                old_id_to_canonical[member_stable] = canon_id

            # ---- Step 5a: MERGE node into global_kg --------------------
            if self.global_kg.has_node(canon_id):
                existing_attrs = dict(self.global_kg.nodes[canon_id])
                self.global_kg.nodes[canon_id].update(
                    _merge_attrs(existing_attrs, merged_attrs)
                )
            else:
                self.global_kg.add_node(canon_id, **merged_attrs)

        # ---- Step 5b: MERGE edges into global_kg -----------------------
        # Re-map raw relation endpoints through old_id_to_canonical.
        # Self-loops and duplicate edges are dropped.
        seen_edges: set = set()
        for rel in global_rel:
            raw_start = rel.get("startNode", "")
            raw_end   = rel.get("endNode",   "")
            rel_name  = rel.get("name", "RELATED_TO").upper().replace(" ", "_")

            canon_start = old_id_to_canonical.get(
                raw_start,
                old_id_to_canonical.get(_stable_id(raw_start, ""), raw_start),
            )
            canon_end = old_id_to_canonical.get(
                raw_end,
                old_id_to_canonical.get(_stable_id(raw_end, ""), raw_end),
            )

            # Drop self-loops
            if canon_start == canon_end:
                logger.debug(f"Dropping self-loop '{rel_name}' on node {canon_start!r}")
                continue

            # Drop duplicate edges (same src, rel_type, tgt)
            edge_key = (canon_start, rel_name, canon_end)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            if self.global_kg.has_node(canon_start) and self.global_kg.has_node(canon_end):
                self.global_kg.add_edge(
                    canon_start,
                    canon_end,
                    name=rel_name,
                    type=rel_name,          # store in BOTH fields for compatibility
                    startNode=canon_start,
                    endNode=canon_end,
                )
            else:
                logger.debug(
                    f"Skipping edge '{rel_name}': endpoint(s) not in graph "
                    f"(start={canon_start!r}, end={canon_end!r})"
                )

        # ---- Step 8: update self.global_entities & embeddings ----------
        self._rebuild_global_state()

        logger.info(
            f"[IncrementalKGBuilder] global_kg now has "
            f"{self.global_kg.number_of_nodes()} nodes, "
            f"{self.global_kg.number_of_edges()} edges"
        )
        return self.global_kg

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _load_prev_kg(self, prev_kg: nx.DiGraph) -> None:
        """Replace self.global_kg with *prev_kg* and sync self.global_entities."""
        self.global_kg = prev_kg.copy()
        self._rebuild_global_state()
        logger.info(
            f"Loaded prev_kg: {self.global_kg.number_of_nodes()} nodes, "
            f"{self.global_kg.number_of_edges()} edges"
        )

    def _rebuild_global_state(self) -> None:
        """
        Recompute self.global_entities from self.global_kg.
        (No longer re-embeds nodes — dedup is now string-based.)
        """
        node_ids = list(self.global_kg.nodes)
        if not node_ids:
            self.global_entities = []
            self._node_embeddings = np.empty((0, 384))
            return

        self.global_entities = [
            self._global_rep(nid) for nid in node_ids
        ]
        # _node_embeddings kept as empty placeholder — not used in string dedup
        self._node_embeddings = np.empty((0, 384))

    def _global_rep(self, node_id: str) -> str:
        """Return the embedding representation string for a node already in global_kg."""
        attrs = self.global_kg.nodes[node_id]
        return _entity_rep(attrs) if attrs else node_id

    # ------------------------------------------------------------------
    # persistence helpers (NetworkX ↔ JSON / file)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist global_kg to a JSON file (node-link format)."""
        import json
        data = nx.node_link_data(self.global_kg)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info(f"Saved global_kg to {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "IncrementalKGBuilder":
        """
        Load a previously saved graph and return a fresh builder
        with global_kg pre-populated.
        """
        import json
        builder = cls(**kwargs)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        builder.global_kg = nx.node_link_graph(data)
        builder._rebuild_global_state()
        logger.info(
            f"Loaded global_kg from {path}: "
            f"{builder.global_kg.number_of_nodes()} nodes, "
            f"{builder.global_kg.number_of_edges()} edges"
        )
        return builder

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict of the current graph state."""
        return {
            "nodes":          self.global_kg.number_of_nodes(),
            "edges":          self.global_kg.number_of_edges(),
            "global_entities": len(self.global_entities),
            "labels":          list(
                {d.get("label", "") for _, d in self.global_kg.nodes(data=True)}
            ),
        }
