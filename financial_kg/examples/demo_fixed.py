
import asyncio
from datetime import datetime
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

# Import modules directly
from models.entity import create_entity
from models.relationship import Relationship, RelationshipProperties
from models.knowledge_graph import KnowledgeGraph
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


async def main():
    """Simple working demo"""
    
    logger.info("=" * 80)
    logger.info("SIMPLE FINANCIAL KG DEMO")
    logger.info("=" * 80)
    
    # Create KG
    kg = KnowledgeGraph()
    
    logger.info("\n1. Creating Entities...")
    
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
    logger.info(f"  ✓ {reliance.name}")
    
    # Create Energy sector
    energy = create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy Sector",
        properties={"index": "NIFTY_ENERGY"}
    )
    kg.add_entity(energy)
    logger.info(f"  ✓ {energy.name}")
    
    logger.info("\n2. Creating Relationships...")
    
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
    
    stats = kg.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ SUCCESS! Knowledge Graph Created")
    logger.info("=" * 80)
    
    return kg


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FINANCIAL KG - SIMPLE DEMO")
    print("=" * 80)
    print("\nThis demo creates:")
    print("  • 2 entities (Company, Sector)")
    print("  • 1 relationship with sentiment")
    print("\n" + "=" * 80)
    
    kg = asyncio.run(main())
    
    print("\n✅ Done!")
    print("\nNext steps:")
    print("  • Configure Gemini API in config/.env")
    print(r"  • Run: python scripts\test_my_data.py")
    print(r"  • Then: python scripts\process_my_data.py")
