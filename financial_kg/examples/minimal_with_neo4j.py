r"""
Minimal demo WITH Neo4j upload
Run from: g:\projects\NLP\financial_kg\
Command: python examples\minimal_with_neo4j.py
"""
import asyncio
from datetime import datetime
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

# Import modules
from models.entity import create_entity
from models.relationship import Relationship, RelationshipProperties
from models.knowledge_graph import KnowledgeGraph
from storage.neo4j_storage import Neo4jStorage


async def main():
    """Minimal demo WITH Neo4j upload"""
    
    print("=" * 80)
    print("FINANCIAL KG - MINIMAL DEMO + NEO4J")
    print("=" * 80)
    
    # Create KG
    kg = KnowledgeGraph()
    
    print("\n1. Creating Entities...")
    
    # Create companies
    reliance = create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries",
        properties={"ticker": "RELIANCE", "sector": "Energy"}
    )
    kg.add_entity(reliance)
    print(f"  ✓ {reliance.name}")
    
    tcs = create_entity(
        entity_type="Company",
        id="TCS",
        name="Tata Consultancy Services",
        properties={"ticker": "TCS", "sector": "IT"}
    )
    kg.add_entity(tcs)
    print(f"  ✓ {tcs.name}")
    
    # Create sectors
    energy = create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy Sector",
        properties={"index": "NIFTY_ENERGY"}
    )
    kg.add_entity(energy)
    print(f"  ✓ {energy.name}")
    
    it_sector = create_entity(
        entity_type="Sector",
        id="IT",
        name="IT Sector",
        properties={"index": "NIFTY_IT"}
    )
    kg.add_entity(it_sector)
    print(f"  ✓ {it_sector.name}")
    
    print("\n2. Creating Relationships...")
    
    # Reliance → Energy
    rel1 = Relationship(
        id="REL_001",
        name="OPERATES_IN",
        subject=reliance,
        predicate="OPERATES_IN",
        object=energy,
        properties=RelationshipProperties(
            sentiment=0.8,
            sentiment_label="positive",
            confidence=0.95
        )
    )
    kg.add_relationship(rel1)
    print(f"  ✓ {reliance.name} → {energy.name}")
    
    # TCS → IT
    rel2 = Relationship(
        id="REL_002",
        name="OPERATES_IN",
        subject=tcs,
        predicate="OPERATES_IN",
        object=it_sector,
        properties=RelationshipProperties(
            sentiment=0.9,
            sentiment_label="positive",
            confidence=0.95
        )
    )
    kg.add_relationship(rel2)
    print(f"  ✓ {tcs.name} → {it_sector.name}")
    
    print("\n3. KG Statistics:")
    stats = kg.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Upload to Neo4j
    print("\n4. Uploading to Neo4j...")
    try:
        with Neo4jStorage() as storage:
            storage.visualize_graph(kg, clear_existing=True)
            
            # Get Neo4j stats
            neo4j_stats = storage.get_graph_stats()
            print(f"\n  ✅ Successfully uploaded to Neo4j!")
            print(f"     Nodes in Neo4j: {neo4j_stats['total_nodes']}")
            print(f"     Relationships: {neo4j_stats['total_relationships']}")
            
            print("\n  🌐 View your graph:")
            print("     1. Open Neo4j Aura console")
            print("     2. Click 'Open' on your database")
            print("     3. Run query: MATCH (n) RETURN n")
            print("     4. See your Financial KG!")
            
            print("\n  💡 Useful queries:")
            print("     • All companies: MATCH (c:Company) RETURN c")
            print("     • All relationships: MATCH ()-[r]->() RETURN r")
            print("     • Positive sentiment: MATCH ()-[r]->() WHERE r.sentiment > 0.5 RETURN r")
            
    except Exception as e:
        print(f"\n  ✗ Neo4j upload failed: {e}")
        print("\n  ⚠️  Make sure you configured Neo4j in config/.env:")
        print(r"     NEO4J_URI=neo4j+s://your-database.neo4j.io")
        print("     NEO4J_USERNAME=neo4j")
        print("     NEO4J_PASSWORD=your_password")
        print("\n  📖 See NEO4J_SETUP.md for help")
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
    
    return kg


if __name__ == "__main__":
    print("\nThis demo will:")
    print("  • Create 2 companies (Reliance, TCS)")
    print("  • Create 2 sectors (Energy, IT)")
    print("  • Create 2 relationships")
    print("  • Upload to Neo4j for visualization")
    print("\n" + "=" * 80)
    
    kg = asyncio.run(main())
