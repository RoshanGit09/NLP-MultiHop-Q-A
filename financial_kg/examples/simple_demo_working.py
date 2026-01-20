"""
Simple working demo - Fixed imports

Run this from g:\projects\NLP\ directory:
python -m financial_kg.examples.simple_demo_working
"""
import asyncio
from datetime import datetime
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now import
import financial_kg.models.entity as entity_mod
import financial_kg.models.relationship as rel_mod
import financial_kg.models.knowledge_graph as kg_mod
import financial_kg.utils.logging_config as log_mod

# Setup logging
log_mod.setup_logging(level="INFO")
logger = log_mod.get_logger(__name__)


async def simple_example():
    """Simple example without Gemini (to avoid API costs)"""
    
    logger.info("=" * 80)
    logger.info("SIMPLE EXAMPLE: Building Financial KG")
    logger.info("=" * 80)
    
    # Initialize KG
    kg = kg_mod.KnowledgeGraph()
    
    logger.info("\n1. CREATING ENTITIES...")
    
    # Create Reliance entity
    reliance = entity_mod.create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries",
        properties={
            "ticker": "RELIANCE",
            "sector": "Energy",
            "exchange": "NSE"
        }
    )
    kg.add_entity(reliance)
    logger.info(f"✓ Created entity: {reliance.name}")
    
    # Create Energy sector
    energy = entity_mod.create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy Sector",
        properties={"index": "NIFTY_ENERGY"}
    )
    kg.add_entity(energy)
    logger.info(f"✓ Created entity: {energy.name}")
    
    # Create earnings event
    earnings = entity_mod.create_entity(
        entity_type="Event",
        id="REL_Q2_2024",
        name="Reliance Q2 Earnings 2024",
        properties={
            "event_type": "Earnings",
            "amount": "₹18,500 crore"
        }
    )
    kg.add_entity(earnings)
    logger.info(f"✓ Created entity: {earnings.name}")
    
    logger.info("\n2. CREATING RELATIONSHIPS WITH SENTIMENT...")
    
    # Reliance → Energy sector
    rel1 = rel_mod.Relationship(
        id="REL_001",
        name="OPERATES_IN",
        subject=reliance,
        predicate="OPERATES_IN",
        object=energy,
        properties=rel_mod.RelationshipProperties(
            sentiment=0.8,
            sentiment_label="positive",
            confidence=0.95
        )
    )
    kg.add_relationship(rel1)
    logger.info(f"✓ {reliance.name} → {energy.name} (sentiment: 0.8)")
    
    # Reliance → Earnings
    rel2 = rel_mod.Relationship(
        id="REL_002",
        name="ANNOUNCED",
        subject=reliance,
        predicate="ANNOUNCED",
        object=earnings,
        properties=rel_mod.RelationshipProperties(
            sentiment=0.85,
            sentiment_label="positive",
            confidence=0.95,
            price_change_percent=4.2
        )
    )
    kg.add_relationship(rel2)
    logger.info(f"✓ {reliance.name} → {earnings.name} (price change: +4.2%)")
    
    logger.info("\n3. KNOWLEDGE GRAPH STATISTICS:")
    stats = kg.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ SUCCESS!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  • Add Gemini API key to config/.env")
    logger.info("  • Run scripts/process_my_data.py to process your data")
    logger.info("  • Upload to Neo4j for visualization")
    
    return kg


if __name__ == "__main__":
    asyncio.run(simple_example())
