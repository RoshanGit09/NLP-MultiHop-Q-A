"""
Process YOUR actual data from D/D folder!

This script:
1. Loads NIFTY-50 stock data from your Kaggle folder
2. Loads IN-FINews articles from your Zenodo folder
3. Builds knowledge graph
4. Uploads to Neo4j for visualization
"""
import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_kg import (
    create_entity,
    Relationship,
    RelationshipProperties,
    TemporalAttributes,
    KnowledgeGraph,
    Neo4jStorage,
    get_logger,
    setup_logging
)
from financial_kg.data_loaders.stock_loader import StockDataLoader
from financial_kg.extractors.gemini_atomic_extractor import GeminiAtomicExtractor
import pandas as pd

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


async def main():
    """Process your actual data"""
    
    logger.info("=" * 80)
    logger.info("PROCESSING YOUR ACTUAL DATA FROM D/D FOLDER")
    logger.info("=" * 80)
    
    # Initialize
    kg = KnowledgeGraph()
    
    # Your data paths
    NIFTY50_DIR = r"g:\projects\NLP\D\D\kaggle\NIFTY-50 Stock Market Data (2000 - 2021)"
    INFINEWS_FILE = r"g:\projects\NLP\D\D\zenodo\IN-FINews Dataset.csv"
    
    # ========== STEP 1: LOAD STOCK DATA ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING NIFTY-50 STOCK DATA")
    logger.info("=" * 80)
    
    stock_loader = StockDataLoader(data_dir=NIFTY50_DIR)
    
    # Top 10 companies to start with
    companies = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
    ]
    
    logger.info(f"Processing {len(companies)} companies...")
    
    for ticker in companies:
        try:
            logger.info(f"\n  📊 Processing {ticker}...")
            
            # Load stock data
            df = stock_loader.load_stock_csv(ticker)
            
            # Get latest price
            latest = df.iloc[-1]
            
            # Create company entity
            company = create_entity(
                entity_type="Company",
                id=ticker,
                name=ticker,
                properties={
                    "ticker": ticker,
                    "total_days": len(df),
                    "date_range": f"{df['Date'].min()} to {df['Date'].max()}",
                    "latest_close": float(latest['Close']) if 'Close' in latest else None,
                    "latest_volume": float(latest['Volume']) if 'Volume' in latest else None
                }
            )
            kg.add_entity(company)
            
            logger.info(f"    ✓ Added {ticker}")
            logger.info(f"      - Days of data: {len(df)}")
            logger.info(f"      - Latest close: ₹{latest['Close'] if 'Close' in latest else 'N/A'}")
            
        except Exception as e:
            logger.error(f"    ✗ Failed to load {ticker}: {e}")
    
    # ========== STEP 2: ADD SECTORS ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CREATING SECTOR ENTITIES")
    logger.info("=" * 80)
    
    sectors_map = {
        'Energy': ['RELIANCE'],
        'IT': ['TCS', 'INFY'],
        'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK'],
        'FMCG': ['HINDUNILVR', 'ITC'],
        'Telecom': ['BHARTIARTL']
    }
    
    # Create sector entities
    for sector_name in sectors_map.keys():
        sector = create_entity(
            entity_type="Sector",
            id=sector_name.upper(),
            name=f"{sector_name} Sector",
            properties={"index": f"NIFTY_{sector_name.upper()}"}
        )
        kg.add_entity(sector)
        logger.info(f"  ✓ Created {sector_name} sector")
    
    # Create Company → Sector relationships
    logger.info("\n  Creating Company → Sector relationships...")
    rel_count = 0
    for sector_name, company_tickers in sectors_map.items():
        for ticker in company_tickers:
            company = kg.get_entity_by_id(ticker)
            sector = kg.get_entity_by_id(sector_name.upper())
            
            if company and sector:
                rel = Relationship(
                    id=f"REL_{ticker}_{sector_name}",
                    name="OPERATES_IN",
                    subject=company,
                    predicate="OPERATES_IN",
                    object=sector,
                    properties=RelationshipProperties(
                        sentiment=0.8,
                        sentiment_label="positive",
                        confidence=1.0,
                        sources=["nifty50_classification"]
                    )
                )
                kg.add_relationship(rel)
                rel_count += 1
    
    logger.info(f"  ✓ Created {rel_count} relationships")
    
    # ========== STEP 3: LOAD NEWS DATA ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: LOADING IN-FINews DATA")
    logger.info("=" * 80)
    
    try:
        # Load first 20 articles for testing
        logger.info(f"  Loading from: {INFINEWS_FILE}")
        news_df = pd.read_csv(INFINEWS_FILE, nrows=20)
        logger.info(f"  ✓ Loaded {len(news_df)} news articles")
        
        # Process news articles
        extractor = GeminiAtomicExtractor(use_pro=False)
        
        for idx, row in news_df.iterrows():
            if pd.notna(row.get('Content')) and str(row['Content']).strip():
                logger.info(f"\n  📰 Processing article {idx+1}/{len(news_df)}...")
                logger.info(f"     Title: {str(row['Title'])[:60]}...")
                
                # Extract facts (first 500 chars for speed)
                content = str(row['Content'])[:500]
                facts = await extractor.extract_atomic_facts(
                    content,
                    observation_date=str(row.get('Date', ''))
                )
                
                logger.info(f"     ✓ Extracted {len(facts)} atomic facts")
                
                # Create event entity
                event = create_entity(
                    entity_type="Event",
                    id=f"NEWS_{idx}",
                    name=str(row['Title'])[:100],
                    properties={
                        "date": str(row.get('Date', '')),
                        "author": str(row.get('Author', '')),
                        "keywords": str(row.get('Keywords', '')),
                        "facts_count": len(facts),
                        "content_preview": content[:200]
                    }
                )
                kg.add_entity(event)
                
                if idx >= 4:  # Process just 5 for demo
                    logger.info(f"\n  ℹ️  Processed {idx+1} articles (demo mode, set higher for full processing)")
                    break
        
    except Exception as e:
        logger.error(f"  ✗ Failed to load news: {e}")
        logger.info("  ℹ️  Continuing without news data...")
    
    # ========== STEP 4: SHOW STATISTICS ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: KNOWLEDGE GRAPH STATISTICS")
    logger.info("=" * 80)
    
    stats = kg.get_stats()
    logger.info(f"\n  📊 Graph Stats:")
    logger.info(f"     Total Entities: {stats['total_entities']}")
    logger.info(f"     Total Relationships: {stats['total_relationships']}")
    logger.info(f"     Entity Types: {stats['entity_types']}")
    logger.info(f"     Relationship Types: {stats['relationship_types']}")
    logger.info(f"     Total Sources: {stats['total_sources']}")
    
    # ========== STEP 5: UPLOAD TO NEO4J ==========
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: UPLOADING TO NEO4J")
    logger.info("=" * 80)
    
    try:
        with Neo4jStorage() as storage:
            logger.info("  📤 Uploading to Neo4j...")
            storage.visualize_graph(kg, clear_existing=True)
            
            # Get Neo4j stats
            neo4j_stats = storage.get_graph_stats()
            logger.info(f"\n  ✓ Successfully uploaded to Neo4j!")
            logger.info(f"     Nodes: {neo4j_stats['total_nodes']}")
            logger.info(f"     Relationships: {neo4j_stats['total_relationships']}")
            
            logger.info("\n  🌐 View your graph:")
            logger.info("     • Open Neo4j Browser (Aura console or http://localhost:7474)")
            logger.info("     • Run: MATCH (n) RETURN n LIMIT 100")
            logger.info("     • Explore your Financial Knowledge Graph!")
            
    except Exception as e:
        logger.error(f"  ✗ Neo4j upload failed: {e}")
        logger.info("  ℹ️  Make sure Neo4j is configured in config/.env")
    
    # ========== DONE ==========
    logger.info("\n" + "=" * 80)
    logger.info("✅ PROCESSING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\n  📊 Built Knowledge Graph:")
    logger.info(f"     • {stats['total_entities']} entities from YOUR data")
    logger.info(f"     • {stats['total_relationships']} relationships")
    logger.info(f"     • NIFTY-50 stocks: {len(companies)} companies")
    logger.info(f"     • News articles: Processed from IN-FINews")
    logger.info("\n  🎯 Next steps:")
    logger.info("     • Increase article count (change nrows=20 to higher)")
    logger.info("     • Process all 50 NIFTY companies")
    logger.info("     • Add more datasets from your D/D folder")
    logger.info("     • Explore in Neo4j Browser")
    logger.info("=" * 80)
    
    return kg


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FINANCIAL KG - PROCESSING YOUR ACTUAL DATA")
    print("=" * 80)
    print("\nThis will process:")
    print("  ✓ NIFTY-50 stocks from your Kaggle folder")
    print("  ✓ IN-FINews articles from your Zenodo folder")
    print("  ✓ Build knowledge graph")
    print("  ✓ Upload to Neo4j")
    print("\n" + "=" * 80)
    
    kg = asyncio.run(main())
    
    print("\n✅ Done! Check Neo4j Browser to see your graph!")
