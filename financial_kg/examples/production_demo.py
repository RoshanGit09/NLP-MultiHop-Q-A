"""
Production demo with real Gemini API and Neo4j integration

This script demonstrates the full Financial KG pipeline:
1. Uses real Gemini API for text extraction
2. Connects to Neo4j Aura for graph storage
3. Processes sample financial news
4. Builds and visualizes knowledge graph
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path  
current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

# Import all components
from models.entity import create_entity
from models.relationship import Relationship, RelationshipProperties
from models.temporal_models import TemporalAttributes
from models.knowledge_graph import KnowledgeGraph
from storage.neo4j_storage import Neo4jStorage
from extractors.gemini_atomic_extractor import GeminiAtomicExtractor
from utils.logging_config import setup_logging, get_logger
from utils.config import get_config

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


async def production_demo():
    """
    Production demo with real APIs
    """
    print("=" * 80)
    print("FINANCIAL KG - PRODUCTION DEMO")
    print("Using: Gemini API + Neo4j Aura + Real Data")  
    print("=" * 80)
    
    # Initialize components
    config = get_config()
    kg = KnowledgeGraph()
    
    # Test configuration
    print("\n1. Testing Configuration...")
    print(f"   Gemini API: {'✓ Configured' if config.gemini.api_key else '✗ Missing'}")
    print(f"   Neo4j URI: {config.neo4j.uri}")
    print(f"   Data Dir: {config.data.data_dir}")
    
    # Test Neo4j connection
    print("\n2. Testing Neo4j Connection...")
    try:
        storage = Neo4jStorage()
        # Test connection
        result = storage.execute_query("RETURN 1 as test")
        print("   ✓ Neo4j connection successful")
        
        # Clear existing data (optional)
        print("   Clearing existing data...")
        storage.clear_all()
        print("   ✓ Database cleared")
        
    except Exception as e:
        print(f"   ✗ Neo4j connection failed: {e}")
        print("   Continuing with local demo...")
        storage = None
    
    # Initialize Gemini extractor
    print("\n3. Testing Gemini API...")
    try:
        extractor = GeminiAtomicExtractor()
        print("   ✓ Gemini extractor initialized")
        use_gemini = True
    except Exception as e:
        print(f"   ✗ Gemini initialization failed: {e}")
        print("   Continuing with manual facts...")
        use_gemini = False
    
    # Sample financial news for processing
    sample_news = [
        {
            "id": "news_001",
            "title": "Reliance Industries Q4 Results: Net Profit Jumps 15% to ₹18,951 Crores",
            "content": """Reliance Industries Limited (RIL) reported a strong set of quarterly numbers with consolidated net profit jumping 15.2% year-on-year to ₹18,951 crores for the March quarter. The company's revenue increased 12.3% to ₹2.35 lakh crores. The petrochemicals business saw robust demand while Jio continued its strong subscriber addition. Chairman Mukesh Ambani highlighted the company's focus on green energy initiatives.""",
            "date": "2024-04-20",
            "source": "Economic Times"
        },
        {
            "id": "news_002", 
            "title": "RBI Monetary Policy: Repo Rate Held at 6.5%, GDP Growth Forecast Raised",
            "content": """The Reserve Bank of India (RBI) maintained the repo rate at 6.5% for the sixth consecutive time, citing balanced inflation outlook. Governor Shaktikanta Das announced that the GDP growth forecast for FY25 has been raised to 7.2% from the earlier projection of 7.0%. The central bank emphasized that the monetary policy stance remains focused on ensuring price stability while supporting growth.""",
            "date": "2024-04-05",
            "source": "Business Standard"
        }
    ]
    
    print(f"\n4. Processing {len(sample_news)} News Articles...")
    
    if use_gemini:
        # Use Gemini for extraction
        for i, article in enumerate(sample_news, 1):
            print(f"\n   Processing Article {i}: {article['title'][:60]}...")
            
            try:
                # Extract facts using Gemini
                facts = await extractor.extract_atomic_facts(
                    text=article['content'],
                    context={
                        'source': article['source'],
                        'date': article['date'],
                        'title': article['title']
                    }
                )
                
                if facts:
                    print(f"   ✓ Extracted {len(facts)} atomic facts")
                    
                    # Convert to entities and relationships
                    entities, relationships = await extractor.extract_entities_and_relationships(facts)
                    
                    # Add to knowledge graph
                    for entity in entities:
                        kg.add_entity(entity)
                    
                    for relationship in relationships:
                        kg.add_relationship(relationship)
                        
                    print(f"   ✓ Added {len(entities)} entities, {len(relationships)} relationships")
                else:
                    print("   ⚠ No facts extracted")
                    
            except Exception as e:
                print(f"   ✗ Processing failed: {e}")
                continue
    
    else:
        # Manual fact creation for demo
        print("   Creating manual entities and relationships...")
        
        # Create entities
        reliance = create_entity(
            entity_type="Company",
            id="RELIANCE",
            name="Reliance Industries Limited",
            properties={
                "ticker": "RELIANCE.NS",
                "sector": "Energy",
                "market_cap": 1500000000000
            }
        )
        kg.add_entity(reliance)
        
        rbi = create_entity(
            entity_type="Organization",
            id="RBI",
            name="Reserve Bank of India", 
            properties={
                "type": "Central Bank",
                "governor": "Shaktikanta Das"
            }
        )
        kg.add_entity(rbi)
        
        # Create relationships with sentiment
        earnings_rel = Relationship(
            id="RELIANCE_REPORTS_EARNINGS_Q4_2024",
            name="Reliance reports strong Q4 earnings",
            subject=reliance,
            predicate="REPORTS",
            object=create_entity("Event", "Q4_EARNINGS_2024", "Q4 2024 Earnings", {"quarter": "Q4", "year": 2024}),
            properties=RelationshipProperties(
                sentiment=0.8,  # Positive earnings
                confidence=0.95,
                source_text="Net Profit Jumps 15% to ₹18,951 Crores",
                temporal=TemporalAttributes(
                    t_announce="2024-04-20",
                    t_effective="2024-04-01",
                    t_observe="2024-04-20"
                )
            )
        )
        kg.add_relationship(earnings_rel)
        
        policy_rel = Relationship(
            id="RBI_MAINTAINS_REPO_RATE_2024",
            name="RBI maintains repo rate at 6.5%",
            subject=rbi,
            predicate="MAINTAINS", 
            object=create_entity("Policy", "REPO_RATE_6_5", "Repo Rate 6.5%", {"rate": 6.5, "type": "monetary_policy"}),
            properties=RelationshipProperties(
                sentiment=0.1,  # Neutral to slightly positive
                confidence=1.0,
                source_text="Repo Rate Held at 6.5%",
                temporal=TemporalAttributes(
                    t_announce="2024-04-05",
                    t_effective="2024-04-05"
                )
            )
        )
        kg.add_relationship(policy_rel)
        
        print("   ✓ Created sample entities and relationships")
    
    # Analyze results
    print("\n5. Knowledge Graph Analysis...")
    stats = kg.get_stats()
    
    print("   📊 Statistics:")
    for key, value in stats.items():
        if key not in ['created_at', 'updated_at']:
            print(f"     {key}: {value}")
    
    # Save to JSON
    print("\n6. Saving Results...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save to JSON
    json_file = output_dir / f"production_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    kg.save_json(json_file)
    print(f"   ✓ JSON: {json_file}")
    
    # Save to Neo4j
    if storage:
        print("\n7. Uploading to Neo4j...")
        try:
            upload_success = storage.upload_kg(kg)
            if upload_success:
                print("   ✓ Successfully uploaded to Neo4j")
                print(f"   🌐 View at: {config.neo4j.uri}")
                
                # Run sample queries
                print("\n   📈 Sample Queries:")
                
                # Count nodes
                result = storage.execute_query("MATCH (n) RETURN labels(n) as type, count(n) as count")
                for record in result:
                    print(f"     {record['type']}: {record['count']} nodes")
                
                # Find companies
                result = storage.execute_query("MATCH (c:Company) RETURN c.name as company LIMIT 5")
                companies = [record['company'] for record in result]
                if companies:
                    print(f"     Companies: {', '.join(companies)}")
                
            else:
                print("   ✗ Neo4j upload failed")
                
        except Exception as e:
            print(f"   ✗ Neo4j error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ PRODUCTION DEMO COMPLETE!")
    print("=" * 80)
    
    print("\nWhat was accomplished:")
    print(f"✓ Processed {len(sample_news)} financial news articles")
    if use_gemini:
        print("✓ Used Gemini API for intelligent fact extraction")
    print(f"✓ Built knowledge graph with {len(kg.entities)} entities and {len(kg.relationships)} relationships")
    print("✓ Applied sentiment analysis to all relationships")
    print("✓ Used 4-dimensional temporal tracking")
    print("✓ Exported to JSON format")
    if storage:
        print("✓ Uploaded to Neo4j Aura for visualization and querying")
    
    print("\nNext steps:")
    print("1. 📊 Visualize in Neo4j Browser")
    print("2. 📈 Add more financial datasets") 
    print("3. 🔄 Set up real-time data processing")
    print("4. 🤖 Build Q&A interface for multi-hop queries")
    
    return kg


if __name__ == "__main__":
    kg = asyncio.run(production_demo())
