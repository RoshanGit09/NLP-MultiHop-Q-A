"""
kg_entity_merger.py
-------------------
Merges near-duplicate entity nodes within the same label group.

Strategy
~~~~~~~~
1.  Within each label, build a prefix-index (first 4 chars of normalised name).
2.  For each pair of candidates, compute SequenceMatcher ratio.
3.  Apply PROTECTED_SUBSTRINGS guard — if both names share a protected token
    (e.g. "jio", "retail", "power") they are kept separate even at high similarity.
4.  Pairs above threshold are unioned into a merge-group; the longest name wins
    as the canonical node (except where an explicit CANONICAL_OVERRIDES entry exists).
5.  All edges pointing to merged-away nodes are rewired to the canonical node,
    then self-loops and duplicate edges are removed.
6.  In --dry-run mode nothing is written — just prints the merge plan.

Usage
~~~~~
    python financial_kg/kg_entity_merger.py --kg financial_kg/output/financial_kg_v2_work.json [--dry-run]
    python financial_kg/kg_entity_merger.py --kg financial_kg/output/financial_kg_v2_work.json --threshold 0.88

Flags
~~~~~
    --kg          Path to the KG JSON file (default: financial_kg/output/financial_kg_v2_work.json)
    --threshold   Similarity threshold 0-1 (default 0.88)
    --dry-run     Print merge plan without writing
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ── Configuration ────────────────────────────────────────────────────────────

# If BOTH nodes contain any of these substrings (after normalisation), they are
# treated as distinct entities even when similarity >= threshold.
PROTECTED_SUBSTRINGS: List[str] = [
    "jio", "retail", "power", "infrastructure", "capital", "securities",
    "insurance", "finance", "bank", "asset", "petroleum", "energy",
    "pharma", "healthcare", "motors", "auto", "steel", "cement",
    "telecom", "media", "realty", "housing", "ventures", "foundation",
    "mutual fund", "amc", "trustee", "mobility",
]

# Force a specific canonical name for a (label, frozenset_of_names) → canonical
# key = (label_lower, normalised_canonical_name)
CANONICAL_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("company", "reliance industries"): "Reliance Industries Ltd",
    ("company", "hdfc bank"):           "HDFC Bank",
    ("company", "tata consultancy"):    "Tata Consultancy Services",
    ("company", "infosys"):             "Infosys Ltd",
    ("company", "wipro"):               "Wipro Ltd.",
    ("company", "icici bank"):          "ICICI Bank",
    ("company", "axis bank"):           "Axis Bank",
    ("company", "kotak mahindra"):      "Kotak Mahindra Bank",
    ("company", "bharti airtel"):       "Bharti Airtel Limited",
    ("company", "hindustan unilever"):  "Hindustan Unilever Ltd",
    ("company", "tata motors"):         "Tata Motors Ltd",
    ("company", "tata steel"):          "Tata Steel Ltd",
    ("company", "adani ports"):         "Adani Ports and SEZ Ltd",
    ("company", "adani enterprises"):   "Adani Enterprises Ltd",
    ("company", "maruti suzuki"):       "Maruti Suzuki India Ltd",
    ("company", "state bank"):          "State Bank of India",
}

# Labels to skip entirely (merging events/policies is too risky — context matters)
SKIP_LABELS: Set[str] = {"Event", "Policy"}

# Regex patterns: if a node name matches, it is NEVER merged (too specific / numeric)
import re as _re
_SKIP_PATTERNS = [
    _re.compile(r"^\d"),                          # starts with digit (raw numbers, prices, %)
    _re.compile(r"^-?\d"),                        # negative numbers
    _re.compile(r"^\$"),                          # dollar amounts
    _re.compile(r"^rs\b", _re.I),                # Rs amounts
    _re.compile(r"^₹"),                           # rupee symbol
    _re.compile(r"\bfy\d{2}", _re.I),            # FY26E, FY27E, FY28 etc — fiscal year specifics
    _re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),     # dates like 2025-07-05
    _re.compile(r"\d+\s*(crore|lakh|mtpa|million|billion|shares|points|percent|per\s+cent)\b", _re.I),
]

def _should_skip_node(name: str) -> bool:
    """Return True if this node should never be merged (too numeric/specific)."""
    n = name.strip()
    for pat in _SKIP_PATTERNS:
        if pat.search(n):
            return True
    return False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    """Lower-case, collapse whitespace, strip common legal suffixes for comparison."""
    n = name.lower().strip()
    n = re.sub(r"\s+", " ", n)
    # strip trailing legal noise for comparison only
    for suffix in (" ltd.", " ltd", " limited", " inc.", " inc",
                   " corp.", " corp", " llp", " llc", " pvt", " private"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _protected_pair(norm_a: str, norm_b: str) -> bool:
    """Return True if both names share a protected token → keep them separate."""
    for token in PROTECTED_SUBSTRINGS:
        if token in norm_a and token in norm_b:
            return True
    return False


def _pick_canonical(names: List[str], label: str) -> str:
    """Pick the best canonical name for a merge group."""
    norm_label = label.lower()
    for key_fragment, canonical in CANONICAL_OVERRIDES.items():
        if key_fragment[0] == norm_label:
            for n in names:
                if key_fragment[1] in _normalise(n):
                    return canonical
    # Default: longest name (most descriptive)
    return max(names, key=len)


# ── Union-Find ────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self, members: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = defaultdict(list)
        for m in members:
            out[self.find(m)].append(m)
        return dict(out)


# ── Core merge logic ──────────────────────────────────────────────────────────

def merge_kg(kg_path: str, threshold: float = 0.88, dry_run: bool = False) -> dict:
    path = Path(kg_path)
    print(f"\nLoading KG from {path.name} …")
    with open(path, encoding="utf-8") as f:
        kg = json.load(f)

    nodes: List[dict] = kg.get("nodes", kg.get("entities", []))
    edges: List[dict] = kg.get("edges", kg.get("relationships", []))
    print(f"  Before: {len(nodes)} nodes, {len(edges)} edges")
    print(f"  Threshold: {threshold}  |  Dry-run: {dry_run}\n")

    # Build id → node map
    id_to_node: Dict[str, dict] = {n["id"]: n for n in nodes}

    # Group node IDs by label
    by_label: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        lbl = (n.get("label") or n.get("type") or "Unknown")
        by_label[lbl].append(n["id"])

    # ── Find merge pairs per label ────────────────────────────────────────────
    # id → canonical_id  (for nodes that get merged away)
    redirect: Dict[str, str] = {}
    total_groups = 0
    total_merged = 0

    for label, ids in sorted(by_label.items()):
        if label in SKIP_LABELS:
            print(f"  [{label}] SKIPPED ({len(ids)} nodes)")
            continue

        # Use tighter threshold for Indicator — more noise-prone
        label_threshold = 0.92 if label == "Indicator" else threshold

        # Build prefix index:  norm_name[:4] → list of ids
        prefix_idx: Dict[str, List[str]] = defaultdict(list)
        for nid in ids:
            nm = _normalise(id_to_node[nid]["name"])
            prefix_idx[nm[:4]].append(nid)
            if len(nm) >= 3:
                prefix_idx[nm[:3]].append(nid)

        uf = UnionFind()
        pair_count = 0

        for nid_a in ids:
            raw_name_a = id_to_node[nid_a]["name"]
            if _should_skip_node(raw_name_a):
                continue
            na = _normalise(raw_name_a)
            # Gather candidates via prefix index
            candidates: Set[str] = set()
            for plen in (4, 3):
                prefix = na[:plen]
                for cid in prefix_idx.get(prefix, []):
                    if cid != nid_a:
                        candidates.add(cid)

            for nid_b in candidates:
                raw_name_b = id_to_node[nid_b]["name"]
                if _should_skip_node(raw_name_b):
                    continue
                nb = _normalise(raw_name_b)
                if _protected_pair(na, nb):
                    continue
                sim = _similarity(na, nb)
                if sim >= label_threshold:
                    uf.union(nid_a, nid_b)
                    pair_count += 1

        # Resolve groups
        groups = uf.groups(ids)
        merge_groups = {root: members for root, members in groups.items()
                        if len(members) > 1}

        if not merge_groups:
            continue

        label_merged = sum(len(m) - 1 for m in merge_groups.values())
        total_groups += len(merge_groups)
        total_merged += label_merged
        print(f"  [{label}]  {len(merge_groups)} merge groups, {label_merged} nodes to remove:")

        for root_id, members in sorted(merge_groups.items(),
                                       key=lambda x: -len(x[1])):
            names = [id_to_node[mid]["name"] for mid in members]
            canonical_name = _pick_canonical(names, label)

            # Find or create canonical node id
            canonical_id = next(
                (mid for mid in members
                 if id_to_node[mid]["name"] == canonical_name),
                members[0]
            )
            # If canonical_name doesn't match any existing node name exactly,
            # rename the chosen node
            id_to_node[canonical_id]["name"] = canonical_name

            merged_away = [mid for mid in members if mid != canonical_id]
            for mid in merged_away:
                redirect[mid] = canonical_id

            print(f"    ✔ canonical: '{canonical_name}'  ← merges {len(merged_away)} alias(es):")
            for mid in merged_away:
                print(f"        - '{id_to_node[mid]['name']}'")

    print(f"\n  Total: {total_groups} groups, {total_merged} nodes to redirect & remove")

    if dry_run:
        print("\n  [DRY RUN] No changes written.")
        return kg

    # ── Apply redirects to edges ──────────────────────────────────────────────
    rewired = 0
    for e in edges:
        new_src = redirect.get(e["source"])
        new_tgt = redirect.get(e["target"])
        if new_src:
            e["source"] = new_src
            rewired += 1
        if new_tgt:
            e["target"] = new_tgt
            rewired += 1

    # Remove self-loops created by merge
    pre = len(edges)
    edges = [e for e in edges if e["source"] != e["target"]]
    loops_removed = pre - len(edges)

    # Remove duplicate edges (same src, type, tgt)
    seen: Set[Tuple] = set()
    deduped: List[dict] = []
    dup_removed = 0
    for e in edges:
        key = (e["source"], e.get("type", "?"), e["target"])
        if key in seen:
            dup_removed += 1
        else:
            seen.add(key)
            deduped.append(e)
    edges = deduped

    # Remove merged-away nodes
    removed_ids = set(redirect.keys())
    nodes = [n for n in nodes if n["id"] not in removed_ids]

    print(f"\n  Edges rewired  : {rewired}")
    print(f"  Self-loops removed (post-merge): {loops_removed}")
    print(f"  Duplicate edges removed        : {dup_removed}")
    print(f"  After: {len(nodes)} nodes, {len(edges)} edges")

    # ── Write output ──────────────────────────────────────────────────────────
    backup_path = str(path).replace(".json", "_backup_premerge.json")
    shutil.copy2(path, backup_path)
    print(f"\n  Backup saved to: {Path(backup_path).name}")

    out = dict(kg)
    if "nodes" in out:
        out["nodes"] = nodes
    else:
        out["entities"] = nodes
    if "edges" in out:
        out["edges"] = edges
    else:
        out["relationships"] = edges

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Merged KG written to: {path.name}")

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge near-duplicate entity nodes in a Financial KG.")
    parser.add_argument("--kg", default="financial_kg/output/financial_kg_v2_work.json",
                        help="Path to KG JSON file.")
    parser.add_argument("--threshold", type=float, default=0.88,
                        help="SequenceMatcher similarity threshold (default 0.88).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print merge plan without writing changes.")
    args = parser.parse_args()
    merge_kg(args.kg, threshold=args.threshold, dry_run=args.dry_run)
