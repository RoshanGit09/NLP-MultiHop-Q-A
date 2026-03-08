"""discover_kg.py — print actual Neo4j schema (labels, rel types, properties)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "backend" / ".env")

from backend.core.kg_adapter.neo4j_adapter import Neo4jAdapter  # noqa: E402

adapter = Neo4jAdapter(
    uri=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

storage = adapter._storage

# --- Node labels + sample properties ---
print("=== NODE LABELS ===")
rows = storage.execute_query(
    "CALL db.labels() YIELD label RETURN label ORDER BY label", {}
)
for r in rows:
    print(" ", r["label"])

print("\n=== RELATION TYPES ===")
rows2 = storage.execute_query(
    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType",
    {},
)
for r in rows2:
    print(" ", r["relationshipType"])

print("\n=== SAMPLE NODE PROPERTIES (first node per label) ===")
labels_rows = storage.execute_query(
    "CALL db.labels() YIELD label RETURN label", {}
)
for lr in labels_rows:
    lbl = lr["label"]
    sample = storage.execute_query(
        f"MATCH (n:{lbl}) RETURN keys(n) AS props LIMIT 1", {}
    )
    if sample:
        print(f"  {lbl}: {sample[0]['props']}")

print("\n=== SAMPLE RELATION PROPERTIES (first rel per type) ===")
rel_rows = storage.execute_query(
    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType", {}
)
for rr in rel_rows:
    rt = rr["relationshipType"]
    sample = storage.execute_query(
        f"MATCH ()-[r:{rt}]->() RETURN keys(r) AS props LIMIT 1", {}
    )
    if sample:
        print(f"  {rt}: {sample[0]['props']}")

adapter.close()
