r"""
Minimal demo - No Gemini, just core functionality
Run from: g:\projects\NLP\financial_kg\
Command: python examples\minimal_demo.py
"""
import asyncio
from datetime import datetime
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

# Import modules directly (NO Gemini to avoid API issues)
from models.entity import create_entity
from models.relationship import Relationship, RelationshipProperties
from models.knowledge_graph import KnowledgeGraph


async def main():
    """Minimal demo without Gemini"""
    
    print("=" * 80)
    print("FINANCIAL KG - MINIMAL DEMO (No Gemini)")
    print("=" * 80)
    
    # Create KG
    kg = KnowledgeGraph()
    
    print("\n1. Creating Entities...")
    
    # Create Reliance
    reliance = create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries",
        properties={
            "ticker": "RELIANCE",
            "sector": "Energy"
        }
    )
    kg.add_entity(reliance)
    print(f"  ✓ {reliance.name}")
    
    # Create TCS
    tcs = create_entity(
        entity_type="Company",
        id="TCS",
        name="Tata Consultancy Services",
        properties={
            "ticker": "TCS",
            "sector": "IT"
        }
    )
    kg.add_entity(tcs)
    print(f"  ✓ {tcs.name}")
    
    # Create Energy sector
    energy = create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy Sector",
        properties={"index": "NIFTY_ENERGY"}
    )
    kg.add_entity(energy)
    print(f"  ✓ {energy.name}")
    
    print("\n2. Creating Relationships...")
    
    # Create relationship
    rel = Relationship(
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
    kg.add_relationship(rel)
    print(f"  ✓ {reliance.name} -> {energy.name}")
    print(f"     Sentiment: {rel.properties.sentiment} ({rel.properties.sentiment_label})")
    
    print("\n3. Statistics:")
    stats = kg.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS! Knowledge Graph Created")
    print("=" * 80)
    
    return kg


if __name__ == "__main__":
    kg = asyncio.run(main())
    
    print("\nNext steps:")
    print(r"  • Configure Gemini API in config\.env")
    print(r"  • Configure Neo4j in config\.env")  
    print(r"  • Test your data: python scripts\test_my_data.py")
    print(r"  • Process data: python scripts\process_my_data.py")
