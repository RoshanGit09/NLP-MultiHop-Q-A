"""
Complete example: Build KG and visualize in Neo4j

This example shows the full pipeline:
1. Create financial entities
2. Add relationships with sentiment
3. Build knowledge graph
4. Upload to Neo4j for visualization
"""
import asyncio
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path so we can import financial_kg
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_kg import (
    create_entity,
    Relationship,
    RelationshipProperties,
    TemporalAttributes,
    KnowledgeGraph,
    get_logger,
    setup_logging
)
from financial_kg.storage import Neo4jStorage
from financial_kg.extractors.gemini_atomic_extractor import GeminiAtomicExtractor

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


async def build_and_visualize():
    """Build a financial KG and visualize in Neo4j"""
    
    logger.info("=" * 80)
    logger.info("BUILDING FINANCIAL KG + NEO4J VISUALIZATION")
    logger.info("=" * 80)
    
    # Step 1: Create Knowledge Graph
    logger.info("\n1. CREATING KNOWLEDGE GRAPH...")
    kg = KnowledgeGraph()
    
    # Create companies
    companies = [
        ("RELIANCE", "Reliance Industries", "Energy"),
        ("TCS", "Tata Consultancy Services", "IT"),
        ("HDFCBANK", "HDFC Bank", "Banking"),
        ("INFY", "Infosys", "IT"),
        ("ICICIBANK", "ICICI Bank", "Banking"),
    ]
    
    for ticker, name, sector in companies:
        company = create_entity(
            entity_type="Company",
            id=ticker,
            name=name,
            properties={
                "ticker": ticker,
                "sector": sector,
                "exchange": "NSE"
            }
        )
        kg.add_entity(company)
        logger.info(f"  ✓ Added: {name}")
    
    # Create sectors
    sectors = ["Energy", "IT", "Banking"]
    for sector_name in sectors:
        sector = create_entity(
            entity_type="Sector",
            id=sector_name.upper(),
            name=f"{sector_name} Sector",
            properties={"index": f"NIFTY_{sector_name.upper()}"}
        )
        kg.add_entity(sector)
    
    # Step 2: Create relationships
    logger.info("\n2. CREATING RELATIONSHIPS...")
    
    # Company → Sector relationships
    company_sectors = [
        ("RELIANCE", "ENERGY", 0.8),
        ("TCS", "IT", 0.9),
        ("INFY", "IT", 0.85),
        ("HDFCBANK", "BANKING", 0.95),
        ("ICICIBANK", "BANKING", 0.90),
    ]
    
    for idx, (company_id, sector_id, sentiment) in enumerate(company_sectors):
        company = kg.get_entity_by_id(company_id)
        sector = kg.get_entity_by_id(sector_id)
        
        if company and sector:
            rel = Relationship(
                id=f"REL_{idx:03d}",
                name="OPERATES_IN",
                subject=company,
                predicate="OPERATES_IN",
                object=sector,
                properties=RelationshipProperties(
                    sentiment=sentiment,
                    sentiment_label="positive" if sentiment > 0.5 else "neutral",
                    confidence=0.95,
                    sources=["company_profile"]
                ),
                temporal=TemporalAttributes(
                    t_observe=datetime.now()
                )
            )
            kg.add_relationship(rel)
            logger.info(f"  ✓ {company.name} → {sector.name} (sentiment: {sentiment})")
    
    # Add some events
    logger.info("\n3. ADDING EVENTS...")
    
    earnings_event = create_entity(
        entity_type="Event",
        id="RELIANCE_Q2_2024",
        name="Reliance Q2 Earnings 2024",
        properties={
            "event_type": "Earnings",
            "amount": "₹18,500 crore",
            "date": "2024-01-15"
        }
    )
    kg.add_entity(earnings_event)
    
    # Reliance → Earnings relationship
    reliance = kg.get_entity_by_id("RELIANCE")
    if reliance:
        rel = Relationship(
            id="REL_EARNINGS",
            name="ANNOUNCED",
            subject=reliance,
            predicate="ANNOUNCED",
            object=earnings_event,
            properties=RelationshipProperties(
                sentiment=0.85,
                sentiment_label="positive",
                confidence=0.95,
                price_change_percent=4.2,
                sources=["news_2024_01_15"]
            ),
            temporal=TemporalAttributes(
                t_announce=datetime(2024, 1, 15, 10, 0),
                t_observe=datetime.now()
            )
        )
        kg.add_relationship(rel)
        logger.info(f"  ✓ {reliance.name} → Earnings Event (price change: +4.2%)")
    
    # Step 4: Show stats
    logger.info("\n4. KNOWLEDGE GRAPH STATISTICS:")
    stats = kg.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Step 5: Upload to Neo4j
    logger.info("\n5. UPLOADING TO NEO4J...")
    
    try:
        with Neo4jStorage() as storage:
            # Clear existing and upload
            storage.visualize_graph(kg, clear_existing=True)
            
            # Get Neo4j stats
            neo4j_stats = storage.get_graph_stats()
            logger.info(f"\n  Neo4j Database Stats:")
            logger.info(f"    Total nodes: {neo4j_stats['total_nodes']}")
            logger.info(f"    Total relationships: {neo4j_stats['total_relationships']}")
            logger.info(f"    Node types: {neo4j_stats['node_types']}")
            
            # Test query
            logger.info(f"\n  Testing query...")
            companies_in_it = storage.get_entities_by_type("Company")
            logger.info(f"    Found {len(companies_in_it)} companies in Neo4j")
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ SUCCESS! Knowledge Graph uploaded to Neo4j")
            logger.info("=" * 80)
            logger.info("\n📊 NEXT STEPS:")
            logger.info("  1. Open Neo4j Browser: http://localhost:7474")
            logger.info("  2. Login with your credentials")
            logger.info("  3. Run query: MATCH (n) RETURN n LIMIT 100")
            logger.info("  4. Explore your Financial Knowledge Graph!")
            logger.info("\n💡 USEFUL QUERIES:")
            logger.info("  • All companies: MATCH (c:Company) RETURN c")
            logger.info("  • All positive relationships: MATCH ()-[r]->() WHERE r.sentiment > 0.5 RETURN r")
            logger.info("  • Reliance connections: MATCH (n {id: 'RELIANCE'})-[r]-(m) RETURN n,r,m")
            logger.info("=" * 80)
            
    except Exception as e:
        logger.error(f"\n✗ Neo4j upload failed: {e}")
        logger.info("\n⚠️  TROUBLESHOOTING:")
        logger.info("  1. Make sure Neo4j is installed and running")
        logger.info("  2. Check config/.env has correct NEO4J credentials")
        logger.info("  3. Default: bolt://localhost:7687, neo4j/password")
        logger.info("\n📥 INSTALL NEO4J:")
        logger.info("  • Desktop: https://neo4j.com/download/")
        logger.info("  • Docker: docker run -p 7474:7474 -p 7687:7687 neo4j")
        logger.info("  • Cloud: https://neo4j.com/cloud/aura/ (free tier)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FINANCIAL KG → NEO4J VISUALIZATION DEMO")
    print("=" * 80)
    print("\nThis demo will:")
    print("  ✓ Create 5 companies (RELIANCE, TCS, HDFC, INFY, ICICI)")
    print("  ✓ Create 3 sectors (Energy, IT, Banking)")
    print("  ✓ Add 1 earnings event")
    print("  ✓ Create relationships with sentiment")
    print("  ✓ Upload to Neo4j for visualization")
    print("\n" + "=" * 80)
    
    asyncio.run(build_and_visualize())
