"""
kg_neo4j_upload.py
------------------
Standalone script to upload a cleaned KG JSON file to Neo4j.
Reads the flat JSON (nodes + edges) and uploads using Neo4jStorage.

Usage
~~~~~
    python financial_kg/kg_neo4j_upload.py --kg financial_kg/output/financial_kg_v2_work.json
    python financial_kg/kg_neo4j_upload.py --kg ... --clear
    python financial_kg/kg_neo4j_upload.py --kg ... --skip-nodes --start-edge 16000

Flags
~~~~~
    --kg          Path to KG JSON file (default: financial_kg/output/financial_kg_v2_work.json)
    --clear       Clear the Neo4j database before uploading (default: False)
    --batch       Batch size for uploads (default: 500)
    --skip-nodes  Skip node upload (resume after nodes already done)
    --start-edge  Resume edge upload from this index (default: 0)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── ensure the package is importable ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase
from financial_kg.utils.config import get_config
from financial_kg.utils.logging_config import get_logger

logger = get_logger(__name__)
config = get_config()

# Recognised entity label types (used as Neo4j node labels)
_KNOWN_LABELS = {"Company", "Person", "Sector", "Event", "Policy", "Indicator"}


def _safe_label(label: str) -> str:
    return label if label in _KNOWN_LABELS else "Indicator"


def upload_from_json(
    kg_path: str,
    clear: bool = False,
    batch_size: int = 500,
    skip_nodes: bool = False,
    start_edge: int = 0,
) -> None:
    path = Path(kg_path)
    print(f"\nLoading KG from {path.name} …")
    with open(path, encoding="utf-8") as f:
        kg = json.load(f)

    nodes: List[dict] = kg.get("nodes", kg.get("entities", []))
    edges: List[dict] = kg.get("edges", kg.get("relationships", []))
    print(f"  Nodes : {len(nodes)}")
    print(f"  Edges : {len(edges)}")

    uri      = config.neo4j.uri
    username = config.neo4j.username
    password = config.neo4j.password
    # AuraDB uses the username as the database name
    database = getattr(config.neo4j, "database", None) or config.neo4j.username or "neo4j"

    print(f"\nConnecting to Neo4j: {uri} (db={database}) …")
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # ── test connection ──────────────────────────────────────────────────────
    try:
        with driver.session(database=database) as s:
            s.run("RETURN 1")
        print("  Connected ✔")
    except Exception as e:
        print(f"  Connection FAILED: {e}")
        driver.close()
        sys.exit(1)

    # ── optional clear ───────────────────────────────────────────────────────
    if clear:
        print("\n  Clearing existing database …")
        with driver.session(database=database) as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("  Cleared ✔")

    # ── build id→label map ───────────────────────────────────────────────────
    id_to_label: Dict[str, str] = {
        n["id"]: _safe_label(n.get("label") or n.get("type") or "Indicator")
        for n in nodes
    }

    # ── upload nodes in batches ──────────────────────────────────────────────
    if not skip_nodes:
        print(f"\nUploading {len(nodes)} nodes (batch={batch_size}) …")
        t0 = time.time()
        total_nodes = 0

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            with driver.session(database=database) as s:
                s.execute_write(_upload_node_batch, batch)
            total_nodes += len(batch)
            pct = total_nodes / len(nodes) * 100
            elapsed = time.time() - t0
            print(f"  {total_nodes}/{len(nodes)}  ({pct:.0f}%)  [{elapsed:.0f}s]", end="\r")

        print(f"\n  Nodes uploaded: {total_nodes}  [{time.time()-t0:.1f}s]  ✔")
    else:
        print("\n  Skipping node upload (--skip-nodes) ✔")

    # ── create indexes on :Label(id) so MATCH in edge upload is fast ─────────
    print("\nCreating/ensuring indexes …")
    labels_present = set(id_to_label.values())
    with driver.session(database=database) as s:
        for lbl in labels_present:
            try:
                s.run(f"CREATE INDEX {lbl}_id_idx IF NOT EXISTS FOR (n:{lbl}) ON (n.id)")
            except Exception:
                pass
    print("  Indexes ready ✔")

    # ── upload edges — one transaction per rel-group, short timeout safe ─────
    edges_to_upload = edges[start_edge:]
    print(f"\nUploading {len(edges_to_upload)} edges "
          f"(starting at {start_edge}, batch={batch_size}) …")
    t0 = time.time()
    total_done = 0
    skipped = 0

    for i in range(0, len(edges_to_upload), batch_size):
        chunk = edges_to_upload[i : i + batch_size]
        sk = _upload_edge_chunk(driver, database, chunk, id_to_label)
        skipped += sk
        total_done += len(chunk)
        pct = (start_edge + total_done) / len(edges) * 100
        elapsed = time.time() - t0
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = (len(edges_to_upload) - total_done) / rate if rate > 0 else 0
        print(
            f"  [{elapsed:5.0f}s]  {start_edge + total_done}/{len(edges)}"
            f"  ({pct:.1f}%)  skipped={skipped}"
            f"  ~{remaining/60:.1f}min left"
        )

    print(
        f"\n  Edges uploaded: {total_done - skipped}  "
        f"skipped: {skipped}  [{time.time()-t0:.1f}s]  ✔"
    )

    driver.close()
    print("\nUpload complete ✔")


# ── Transaction functions (run inside execute_write) ─────────────────────────

def _upload_node_batch(tx, batch: List[dict]) -> None:
    """MERGE each node by id using UNWIND — one round-trip per label per batch."""
    from collections import defaultdict

    rows = []
    for n in batch:
        label = _safe_label(n.get("label") or n.get("type") or "Indicator")
        props = {"id": n["id"], "name": n.get("name", n["id"]), "label": label}
        for k, v in n.items():
            if k not in ("id", "name", "label", "type") and v is not None:
                props[k] = v if not isinstance(v, (dict, list)) else str(v)
        rows.append({"label": label, "id": n["id"], "props": props})

    by_label: Dict[str, list] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    for label, label_rows in by_label.items():
        tx.run(
            f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{id: row.id}})
            SET n += row.props
            """,
            rows=label_rows,
        )


def _upload_edge_chunk(
    driver,
    database: str,
    chunk: List[dict],
    id_to_label: Dict[str, str],
) -> int:
    """
    Upload a chunk of edges to Neo4j.
    Groups edges by (src_label, tgt_label, rel_type) and runs groups in parallel
    using a thread pool — each group gets its own short transaction.
    Returns number of skipped edges.
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    groups: Dict[tuple, list] = defaultdict(list)
    skipped = 0

    for e in chunk:
        src_id = e.get("source")
        tgt_id = e.get("target")
        rel_type = (
            (e.get("type") or e.get("name") or "RELATED_TO")
            .upper().replace(" ", "_").replace("-", "_")
        )
        src_label = id_to_label.get(src_id)
        tgt_label = id_to_label.get(tgt_id)
        if not src_label or not tgt_label:
            skipped += 1
            continue
        props: dict = {}
        for k, v in e.items():
            if k not in ("source", "target", "type", "name") and v is not None:
                props[k] = v if not isinstance(v, (dict, list)) else str(v)
        groups[(src_label, tgt_label, rel_type)].append(
            {"src": src_id, "tgt": tgt_id, "props": props}
        )

    def _run_group(src_lbl, tgt_lbl, rel_type, rows):
        cypher = f"""
        UNWIND $rows AS row
        MATCH (a:{src_lbl} {{id: row.src}})
        MATCH (b:{tgt_lbl} {{id: row.tgt}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += row.props
        """
        try:
            with driver.session(database=database) as s:
                s.execute_write(lambda tx, r=rows, q=cypher: tx.run(q, rows=r))
            return 0
        except Exception:
            return len(rows)

    # Run all groups in parallel — 8 concurrent connections to AuraDB
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_run_group, src_lbl, tgt_lbl, rel_type, rows): len(rows)
            for (src_lbl, tgt_lbl, rel_type), rows in groups.items()
        }
        for fut in as_completed(futures):
            skipped += fut.result()

    return skipped


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a KG JSON file to Neo4j.")
    parser.add_argument(
        "--kg",
        default="financial_kg/output/financial_kg_v2_work.json",
        help="Path to KG JSON file.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the Neo4j database before uploading.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=500,
        help="Batch size for uploads (default: 500).",
    )
    parser.add_argument(
        "--skip-nodes",
        action="store_true",
        dest="skip_nodes",
        help="Skip node upload (nodes already in Neo4j).",
    )
    parser.add_argument(
        "--start-edge",
        type=int,
        default=0,
        dest="start_edge",
        help="Resume edge upload from this index (default: 0).",
    )
    args = parser.parse_args()
    upload_from_json(
        args.kg,
        clear=args.clear,
        batch_size=args.batch,
        skip_nodes=args.skip_nodes,
        start_edge=args.start_edge,
    )
