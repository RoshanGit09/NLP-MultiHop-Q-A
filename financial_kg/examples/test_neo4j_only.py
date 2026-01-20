r"""
Test Neo4j connection with YOUR credentials
Run: python examples\test_neo4j_only.py
"""
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

from models.entity import create_entity
from models.relationship import Relationship, RelationshipProperties
from models.knowledge_graph import KnowledgeGraph

# Import Neo4j but create storage manually to avoid config issues
from neo4j import GraphDatabase

# YOUR NEO4J CREDENTIALS (from Neo4j-77d67632-Created-2026-01-20.txt)
NEO4J_URI = "neo4j+s://77d67632.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "qUWGayrJ5kWFqrOBSw8xT7_srIGqeoQZUqyB01f7xyk"

print("=" * 80)
print("TESTING NEO4J CONNECTION")
print("=" * 80)

# Test 1: Connection
print("\n1. Testing connection to Neo4j Aura...")
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        print(f"  ✅ Connected successfully!")
    driver.close()
except Exception as e:
    print(f"  ✗ Connection failed: {e}")
    exit(1)

# Test 2: Create simple KG
print("\n2. Creating simple Knowledge Graph...")
kg = KnowledgeGraph()

reliance = create_entity(
    entity_type="Company",
    id="RELIANCE",
    name="Reliance Industries",
    properties={"ticker": "RELIANCE"}
)
kg.add_entity(reliance)
print(f"  ✓ Created: {reliance.name}")

# Test 3: Upload to Neo4j
print("\n3. Uploading to Neo4j...")
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Clear existing
        session.run("MATCH (n) DETACH DELETE n")
        print("  ✓ Cleared existing data")
        
        # Upload entity
        session.run("""
            MERGE (n:Company {id: $id})
            SET n.name = $name, n.ticker = $ticker
        """, id=reliance.id, name=reliance.name, ticker="RELIANCE")
        print(f"  ✓ Uploaded: {reliance.name}")
        
        # Check
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()['count']
        print(f"  ✓ Nodes in Neo4j: {count}")
    
    driver.close()
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS! Data uploaded to Neo4j")
    print("=" * 80)
    print("\n🌐 View your data:")
    print("  1. Go to: https://console.neo4j.io")
    print("  2. Click 'Open' on your database")
    print("  3. Run query: MATCH (n) RETURN n")
    print("\n💡 You should see: Reliance Industries node")
    
except Exception as e:
    print(f"  ✗ Upload failed: {e}")
    import traceback
    traceback.print_exc()
