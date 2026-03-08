"""
Fixed working demo - No external dependencies needed

This script demonstrates the Financial KG framework with sample data.
Run this from the financial_kg directory:

cd financial_kg
python examples/working_demo.py
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path  
current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

# Direct imports to avoid module issues
from models.entity import create_entity
from models.relationship import Relationship, RelationshipProperties
from models.temporal_models import TemporalAttributes
from models.knowledge_graph import KnowledgeGraph
from utils.logging_config import setup_logging, get_logger

# Setup logging with safe encoding
setup_logging(level="INFO")
logger = get_logger(__name__)


async def working_demo():
    """
    Complete working demo of Financial KG construction
    """
    print("=" * 80)
    print("FINANCIAL KNOWLEDGE GRAPH - WORKING DEMO")  
    print("=" * 80)
    print("Building a sample Financial KG from scratch...")
    print()
    
    # Initialize Knowledge Graph
    kg = KnowledgeGraph()
    
    print("1. Creating Financial Entities...")
    
    # Create companies
    reliance = create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries Ltd",
        properties={
            "ticker": "RELIANCE.NS",
            "sector": "Energy", 
            "exchange": "NSE",
            "market_cap": 1500000000000  # 15 lakh crores
        }
    )
    kg.add_entity(reliance)
    print(f"   + {reliance.name}")
    
    tcs = create_entity(
        entity_type="Company", 
        id="TCS",
        name="Tata Consultancy Services",
        properties={
            "ticker": "TCS.NS",
            "sector": "Information Technology",
            "exchange": "NSE"
        }
    )
    kg.add_entity(tcs)
    print(f"   + {tcs.name}")
    
    # Create sectors
    energy_sector = create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy Sector",
        properties={
            "industry_group": "Oil & Gas",
            "index": "NIFTY Energy"
        }
    )
    kg.add_entity(energy_sector)
    print(f"   + {energy_sector.name}")
    
    it_sector = create_entity(
        entity_type="Sector",
        id="IT",
        name="Information Technology Sector", 
        properties={
            "industry_group": "Software",
            "index": "NIFTY IT"
        }
    )
    kg.add_entity(it_sector)
    print(f"   + {it_sector.name}")
    
    # Create key person
    mukesh_ambani = create_entity(
        entity_type="Person",
        id="MUKESH_AMBANI",
        name="Mukesh Ambani",
        properties={
            "role": "Chairman & Managing Director",
            "affiliation": "Reliance Industries"
        }
    )
    kg.add_entity(mukesh_ambani)
    print(f"   + {mukesh_ambani.name}")
    
    # Create market event
    earnings_event = create_entity(
        entity_type="Event",
        id="RIL_Q4_EARNINGS",
        name="Reliance Q4 Earnings Announcement",
        properties={
            "event_type": "Quarterly Results",
            "severity": "High",
            "date": "2024-01-15"
        }
    )
    kg.add_entity(earnings_event)
    print(f"   + {earnings_event.name}")
    
    print(f"\\n   Created {len(kg.entities)} entities")
    
    print("\\n2. Creating Relationships...")
    
    # Company-Sector relationships
    rel1_props = RelationshipProperties(
        confidence=0.95,
        sentiment=0.1,  # Neutral to slightly positive
        source_text="Reliance operates in the energy sector",
        temporal=TemporalAttributes(
            t_announce="2024-01-01",
            t_effective="2024-01-01",
            t_observe="2024-01-01",
            t_impact="2024-01-01"
        )
    )
    
    rel1 = Relationship(
        id="REL_OPERATES_IN_ENERGY_1",
        name="Reliance operates in Energy sector",
        subject=reliance,
        predicate="OPERATES_IN", 
        object=energy_sector,
        properties=rel1_props
    )
    kg.add_relationship(rel1)
    print("   + Reliance OPERATES_IN Energy Sector")
    
    # TCS-IT Sector relationship
    rel2_props = RelationshipProperties(
        confidence=0.98,
        sentiment=0.2,
        source_text="TCS is a leading IT services company",
        temporal=TemporalAttributes()
    )
    
    rel2 = Relationship(
        id="TCS_OPERATES_IN_IT_1", 
        name="TCS operates in IT sector",
        subject=tcs,
        predicate="OPERATES_IN",
        object=it_sector, 
        properties=rel2_props
    )
    kg.add_relationship(rel2)
    print("   + TCS OPERATES_IN IT Sector")
    
    # Leadership relationship
    rel3_props = RelationshipProperties(
        confidence=1.0,
        sentiment=0.3,
        source_text="Mukesh Ambani leads Reliance Industries"
    )
    
    rel3 = Relationship(
        id="MUKESH_LEADS_RELIANCE_1",
        name="Mukesh Ambani leads Reliance", 
        subject=mukesh_ambani,
        predicate="LEADS",
        object=reliance,
        properties=rel3_props  
    )
    kg.add_relationship(rel3)
    print("   + Mukesh Ambani LEADS Reliance")
    
    # Event relationship
    rel4_props = RelationshipProperties(
        confidence=0.9,
        sentiment=0.6,  # Positive earnings
        source_text="Reliance announces strong Q4 earnings"
    )
    
    rel4 = Relationship(
        id="RELIANCE_ANNOUNCES_EARNINGS_1",
        name="Reliance announces Q4 earnings",
        subject=reliance,
        predicate="ANNOUNCES",
        object=earnings_event,
        properties=rel4_props
    )
    kg.add_relationship(rel4)
    print("   + Reliance ANNOUNCES Q4 Earnings")
    
    # Competitive relationship
    rel5_props = RelationshipProperties(
        confidence=0.7,
        sentiment=0.0,  # Neutral competition
        source_text="Both are leading companies in their sectors"
    )
    
    rel5 = Relationship(
        id="RELIANCE_COMPETES_TCS_1",
        name="Reliance competes with TCS in market cap",
        subject=reliance,
        predicate="COMPETES_WITH",
        object=tcs,
        properties=rel5_props
    )
    kg.add_relationship(rel5)
    print("   + Reliance COMPETES_WITH TCS (market cap)")
    
    print(f"\\n   Created {len(kg.relationships)} relationships")
    
    print("\\n3. Analyzing Knowledge Graph...")
    
    # Get statistics
    stats = kg.get_stats()
    print("\\n   KG Statistics:")
    for key, value in stats.items():
        if key != 'created_at' and key != 'updated_at':
            print(f"     {key}: {value}")
    
    print("\\n4. Saving Knowledge Graph...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save to JSON
    output_file = output_dir / "demo_financial_kg.json"
    try:
        kg.save_json(output_file)
        print(f"   Saved to: {output_file}")
    except Exception as e:
        print(f"   Save failed: {e}")
    
    print("\\n" + "=" * 80)
    print("SUCCESS! Financial Knowledge Graph Demo Complete")
    print("=" * 80)
    
    print("\\nWhat was created:")
    print(f"- {len(kg.entities)} entities (companies, sectors, people, events)")
    print(f"- {len(kg.relationships)} relationships with sentiment scores")
    print(f"- Temporal attributes for time-aware analysis")
    print(f"- JSON export for further processing")
    
    print("\\nNext Steps:")
    print("1. Configure Gemini API key for real text extraction")
    print("2. Set up Neo4j for graph database storage") 
    print("3. Add your own financial datasets")
    print("4. Scale to thousands of entities and relationships")
    
    return kg


if __name__ == "__main__":
    kg = asyncio.run(working_demo())
