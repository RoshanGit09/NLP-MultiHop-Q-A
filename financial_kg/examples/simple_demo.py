"""
Simple end-to-end example: Building a financial KG from news and stock data

This script demonstrates the complete pipeline:
1. Load news data
2. Extract atomic facts using Gemini
3. Load corresponding stock price data  
4. Build knowledge graph with entities and relationships
5. Add sentiment and temporal attributes
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path so we can import financial_kg
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import FinancialKG components

from financial_kg import (
    create_entity,
    Relationship,
    RelationshipProperties,
    TemporalAttributes,
    KnowledgeGraph,
    get_gemini_client,
    get_logger,
    setup_logging
)

from financial_kg.data_loaders.news_loader import NewsDataLoader
from financial_kg.data_loaders.stock_loader import StockDataLoader
from financial_kg.extractors.gemini_atomic_extractor import GeminiAtomicExtractor

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


async def simple_example():
    """
    Simple example: Build KG from a single news article + stock data
    """
    logger.info("=" * 80)
    logger.info("SIMPLE EXAMPLE: Building Financial KG from News + Stock Data")
    logger.info("=" * 80)
    
    # Initialize KG
    kg = KnowledgeGraph()
    
    # Sample news article
    news_article = """
    Mumbai, January 15, 2024 - Reliance Industries announced record quarterly 
    earnings of ₹18,500 crore, surpassing analyst expectations by 15%. The 
    energy and petrochemicals giant saw a 23% year-over-year growth driven by 
    strong performance in its retail and digital divisions. The stock price 
    surged 4.2% following the announcement.
    """
    
    logger.info("\n1. EXTRACTING ATOMIC FACTS...")
    logger.info(f"News: {news_article[:100]}...")
    
    # Extract atomic facts using Gemini
    extractor = GeminiAtomicExtractor(use_pro=False)
    atomic_facts = await extractor.extract_atomic_facts(
        news_article,
        observation_date="2024-01-15"
    )
    
    logger.info(f"✓ Extracted {len(atomic_facts)} atomic facts:")
    for i, fact in enumerate(atomic_facts, 1):
        logger.info(f"  {i}. {fact}")
    
    # Step 2: Create entities
    logger.info("\n2. CREATING ENTITIES...")
    
    # Create Reliance entity
    reliance = create_entity(
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
    
    # Create earnings event
    earnings_event = create_entity(
        entity_type="Event",
        id="REL_Q4_EARNINGS_2024",
        name="Reliance Q4 Earnings 2024",
        properties={
            "event_type": "Earnings",
            "amount": "₹18,500 crore",
            "beat_estimate": "15%"
        }
    )
    kg.add_entity(earnings_event)
    logger.info(f"✓ Created entity: {earnings_event.name}")
    
    # Step 3: Create relationships with sentiment
    logger.info("\n3. CREATING RELATIONSHIPS WITH SENTIMENT...")
    
    rel_earnings = Relationship(
        id="REL_001",
        name="ANNOUNCED",
        subject=reliance,
        predicate="ANNOUNCED",
        object=earnings_event,
        properties=RelationshipProperties(
            sentiment=0.85,  # Very positive
            sentiment_label="positive",
            confidence=0.95,
            price_change_percent=4.2,
            sources=["news_article_2024_01_15"]
        ),
        temporal=TemporalAttributes(
            t_announce=datetime(2024, 1, 15, 10, 0),
            t_observe=datetime(2024, 1, 15, 11, 0),
            observation_dates=[datetime(2024, 1, 15)]
        )
    )
    
    kg.add_relationship(rel_earnings)
    logger.info(f"✓ Created relationship: {rel_earnings}")
    logger.info(f"  Sentiment: {rel_earnings.properties.sentiment:.2f} ({rel_earnings.properties.sentiment_label})")
    logger.info(f"  Price change: +{rel_earnings.properties.price_change_percent}%")
    
    # Step 4: Show KG statistics
    logger.info("\n4. KNOWLEDGE GRAPH STATISTICS:")
    stats = kg.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ SIMPLE EXAMPLE COMPLETE!")
    logger.info("=" * 80)
    
    return kg


async def batch_example():
    """
    Batch example: Build KG from multiple news articles
    """
    logger.info("=" * 80)
    logger.info("BATCH EXAMPLE: Processing Multiple News Articles")
    logger.info("=" * 80)
    
    # Sample news articles
    news_articles = [
        {
            "text": "TCS reports 12% growth in Q3 revenue, reaching ₹58,000 crore.",
            "date": "2024-01-10",
            "ticker": "TCS"
        },
        {
            "text": "HDFC Bank's profit rises 20% to ₹14,500 crore amid strong loan growth.",
            "date": "2024-01-12",
            "ticker": "HDFCBANK"
        },
        {
            "text": "Infosys wins $1.5 billion deal with global financial services firm.",
            "date": "2024-01-14",
            "ticker": "INFY"
        }
    ]
    
    kg = KnowledgeGraph()
    extractor = GeminiAtomicExtractor(use_pro=False)
    
    logger.info(f"\nProcessing {len(news_articles)} news articles...")
    
    for article in news_articles:
        logger.info(f"\n• Processing: {article['text'][:60]}...")
        
        # Extract facts
        facts = await extractor.extract_atomic_facts(
            article['text'],
            observation_date=article['date']
        )
        logger.info(f"  → Extracted {len(facts)} facts")
        
        # Create company entity (simplified)
        company = create_entity(
            entity_type="Company",
            id=article['ticker'],
            name=article['ticker'],
            properties={"ticker": article['ticker']}
        )
        kg.add_entity(company)
    
    logger.info(f"\n✓ Batch processing complete!")
    logger.info(f"✓ Total entities: {len(kg.entities)}")
    logger.info(f"✓ Total relationships: {len(kg.relationships)}")
    
    return kg


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FINANCIAL KG - END-TO-END EXAMPLE")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  1. Atomic fact extraction with Gemini")
    print("  2. Entity creation (Company, Event)")
    print("  3. Sentiment-aware relationships")
    print("  4. Temporal modeling")
    print("  5. Knowledge graph construction")
    print("\n" + "=" * 80)
    
    # Run simple example
    kg_simple = asyncio.run(simple_example())
    
    print("\n")
    input("Press Enter to continue to batch example...")
    print("\n")
    
    # Run batch example
    kg_batch = asyncio.run(batch_example())
    
    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print(f"\nNext steps:")
    print("  • Download real datasets (see DATASET_ACCESS_GUIDE.md)")
    print("  • Process NIFTY-50 stocks")
    print("  • Build larger knowledge graphs")
    print("  • Integrate with Neo4j for visualization")
    print("  • Connect to your MultiHop-Q-A project")
