"""
kg_cleaner.py
=============
Cleans the existing (corrupted) financial_kg.json in-place:

1. Removes self-loop edges (source == target)           → was 43
2. Adds `type` field to edges (copies from `name`)      → was missing (all '?')
3. Deduplicates edges with same (src, type, tgt)        → preventive
4. Normalises edge `name` → UPPER_SNAKE_CASE
5. Removes completely orphaned nodes (no edges at all)
6. Removes known generic/noise Company nodes that are
   not real companies (Joint Venture, etc.)
7. Reports all changes made

Run:
    python kg_cleaner.py [--kg financial_kg/output/financial_kg.json] [--dry-run]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── noise node names to drop (wrongly classified or too generic) ────────────
_NOISE_COMPANY_NAMES = {
    "joint venture",
    "special purpose vehicle",
    "spv",
    "integrated back-end",
    "back-end",
    "four tranches",
    "multiple tranches",
    "final dividend",
    "interim dividend",
}

_NOISE_INDICATOR_NAMES = {
    "tepid condition",
    "manappuram finance is at overbought levels",
    "overbought levels",
    "oversold levels",
}

_NOISE_EVENT_NAMES = {
    "trade spotlight",
    "fy26",          # too generic as an Event; kept if it has useful edges
}

_NOISE_POLICY_NAMES = {
    "integrated back-end",
}


def _to_snake(name: str) -> str:
    """Convert relationship name to UPPER_SNAKE_CASE."""
    name = name.strip()
    # Replace spaces and hyphens with underscore
    name = re.sub(r"[\s\-]+", "_", name)
    # Remove non-alphanumeric (except underscore)
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    return name.upper() if name else "RELATED_TO"


def clean_kg(kg_path: str, dry_run: bool = False) -> dict:
    print(f"Loading KG from {kg_path} …")
    with open(kg_path, encoding="utf-8") as f:
        kg = json.load(f)

    nodes = kg.get("nodes", kg.get("entities", []))
    edges = kg.get("edges", kg.get("relationships", []))

    print(f"  Before: {len(nodes)} nodes, {len(edges)} edges")

    # ── Build lookup structures ─────────────────────────────────────────────
    node_by_id = {n["id"]: n for n in nodes}
    id_to_name = {n["id"]: n["name"] for n in nodes}
    id_to_label = {n["id"]: (n.get("label") or n.get("type", "?")) for n in nodes}

    # ── 1. Normalize edge name/type fields ──────────────────────────────────
    fixed_type = 0
    for e in edges:
        rel_name = e.get("name") or e.get("type") or "RELATED_TO"
        normalized = _to_snake(rel_name)
        if e.get("type") != normalized or e.get("name") != normalized:
            e["name"] = normalized
            e["type"] = normalized
            fixed_type += 1

    print(f"  Normalized edge name/type: {fixed_type} edges updated")

    # ── 2. Remove self-loops ─────────────────────────────────────────────────
    pre_len = len(edges)
    edges = [e for e in edges if e["source"] != e["target"]]
    removed_loops = pre_len - len(edges)
    print(f"  Self-loops removed: {removed_loops}")

    # ── 3. Deduplicate edges (same src, type, tgt) ───────────────────────────
    seen = set()
    deduped_edges = []
    dup_count = 0
    for e in edges:
        key = (e["source"], e.get("type", "?"), e["target"])
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
            deduped_edges.append(e)
    edges = deduped_edges
    print(f"  Duplicate edges removed: {dup_count}")

    # ── 4. Identify noise nodes to remove ────────────────────────────────────
    noise_ids = set()
    for n in nodes:
        nm = n["name"].lower().strip()
        lbl = (n.get("label") or "").lower()
        if lbl == "company" and nm in _NOISE_COMPANY_NAMES:
            noise_ids.add(n["id"])
        elif lbl == "indicator" and nm in _NOISE_INDICATOR_NAMES:
            noise_ids.add(n["id"])
        elif lbl == "policy" and nm in _NOISE_POLICY_NAMES:
            noise_ids.add(n["id"])
    print(f"  Noise nodes identified for removal: {len(noise_ids)}")
    for nid in noise_ids:
        print(f"    - [{id_to_label.get(nid)}] {id_to_name.get(nid)}")

    # Remove noise nodes and their edges
    if noise_ids:
        nodes = [n for n in nodes if n["id"] not in noise_ids]
        edges = [e for e in edges
                 if e["source"] not in noise_ids and e["target"] not in noise_ids]

    # ── 5. Find and remove orphan nodes ──────────────────────────────────────
    connected = set()
    for e in edges:
        connected.add(e["source"])
        connected.add(e["target"])
    orphans = [n for n in nodes if n["id"] not in connected]
    orphan_count = len(orphans)
    print(f"  Orphan nodes (no edges): {orphan_count}")
    for n in orphans[:10]:
        lbl = n.get("label") or n.get("type", "?")
        print(f"    - [{lbl}] {n['name']}")
    if orphan_count > 10:
        print(f"    ... and {orphan_count - 10} more")
    nodes = [n for n in nodes if n["id"] in connected]

    # ── 6. Fix broken edge endpoints ─────────────────────────────────────────
    all_ids = {n["id"] for n in nodes}
    pre_len = len(edges)
    edges = [e for e in edges if e["source"] in all_ids and e["target"] in all_ids]
    broken_removed = pre_len - len(edges)
    print(f"  Broken-endpoint edges removed: {broken_removed}")

    # ── 7. Final stats ────────────────────────────────────────────────────────
    print(f"\n  After:  {len(nodes)} nodes, {len(edges)} edges")

    # Label distribution
    label_dist = Counter((n.get("label") or n.get("type", "?")) for n in nodes)
    print("\n  Label distribution:")
    for lbl, cnt in label_dist.most_common():
        print(f"    {lbl}: {cnt}")

    # Top 10 by degree
    deg = Counter()
    for e in edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    id_to_name_new = {n["id"]: n["name"] for n in nodes}
    print("\n  Top 10 by degree:")
    for nid, d in sorted(deg.items(), key=lambda x: -x[1])[:10]:
        lbl = id_to_label.get(nid, "?")
        print(f"    {d:4d}  [{lbl}]  {id_to_name_new.get(nid, nid)}")

    # Edge type distribution
    print("\n  Top 15 edge types:")
    edge_types = Counter(e.get("type", "?") for e in edges)
    for t, c in edge_types.most_common(15):
        print(f"    {c:4d}  {t}")

    if dry_run:
        print("\n  [DRY RUN] No changes written.")
        return kg

    # ── Write cleaned KG ─────────────────────────────────────────────────────
    # Preserve top-level structure
    out = dict(kg)
    if "nodes" in out:
        out["nodes"] = nodes
    else:
        out["entities"] = nodes
    if "edges" in out:
        out["edges"] = edges
    else:
        out["relationships"] = edges

    # Backup original
    backup_path = kg_path.replace(".json", "_backup_precleaned.json")
    import shutil
    shutil.copy2(kg_path, backup_path)
    print(f"\n  Backup saved to: {backup_path}")

    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Cleaned KG written to: {kg_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the Financial KG JSON file.")
    parser.add_argument(
        "--kg",
        default="financial_kg/output/financial_kg.json",
        help="Path to the KG JSON file to clean.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing changes.",
    )
    args = parser.parse_args()
    clean_kg(args.kg, dry_run=args.dry_run)
