# Getting Started with FinancialKG

This guide will help you set up and start using FinancialKG for building multi-modal financial knowledge graphs.

## 📋 Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))
- Neo4j database (optional, for visualization)
- Git

## 🚀 Installation

### 1. Clone/Navigate to the Project

```bash
cd g:\projects\NLP\financial_kg
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Configuration

```bash
# Copy the example environment file
cp config\.env.example config\.env

# Edit config\.env with your credentials
notepad config\.env  # or your preferred editor
```

**Required configurations:**
- `GEMINI_API_KEY`: Your Google Gemini API key
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: If using Neo4j for visualization

## 🧪 Quick Test

Test if everything is set up correctly:

```python
import asyncio
from financial_kg import get_gemini_client, create_entity, KnowledgeGraph

async def test_setup():
    # Test Gemini client
    client = get_gemini_client()
    response = await client.generate_async("Hello, Gemini!")
    print(f"✓ Gemini API working: {response[:50]}...")
    
    # Test entity creation
    company = create_entity(
        entity_type="Company",
        id="TEST",
        name="Test Company",
        properties={"ticker": "TEST"}
    )
    print(f"✓ Entity created: {company.name}")
    
    # Test KG
    kg = KnowledgeGraph()
    kg.add_entity(company)
    print(f"✓ Knowledge Graph created with {len(kg.entities)} entities")
    
    print("\n✅ All tests passed! You're ready to go.")

asyncio.run(test_setup())
```

Save this as `test_setup.py` and run:
```bash
python test_setup.py
```

## 📊 Download Datasets

### Option 1: Using Scripts (Recommended)

We'll create download scripts for each dataset:

```bash
python scripts/download_datasets.py --dataset nifty50
python scripts/download_datasets.py --dataset infin ews
python scripts/download_datasets.py --all  # Download all
```

### Option 2: Manual Download

1. **NIFTY-50 Data**: Visit [Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)
2. **IN-FINews**: Visit [Zenodo](https://zenodo.org/records/16991843)
3. **HuggingFace Datasets**: Use `datasets` library (automated in our scripts)

Place downloaded data in:
```
financial_kg/data/
├── nifty50/
├── stock_prices/
├── financial_metrics/
├── news/
│   ├── infinews/
│   ├── indian_financial_news/
│   └── headlines/
└── macro/
```

## 🏗️ Building Your First Financial KG

### Example 1: Simple Stock-Company KG

```python
import asyncio
import pandas as pd
from datetime import datetime
from financial_kg import (
    create_entity, 
    Relationship, 
    RelationshipProperties,
    TemporalAttributes,
    KnowledgeGraph
)

async def build_simple_kg():
    # Create KG
    kg = KnowledgeGraph()
    
    # Create some companies
    reliance = create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries Ltd.",
        properties={
            "ticker": "RELIANCE",
            "sector": "Energy",
            "exchange": "NSE",
            "market_cap": 1750000000000
        }
    )
    
    tcs = create_entity(
        entity_type="Company",
        id="TCS",
        name="Tata Consultancy Services",
        properties={
            "ticker": "TCS",
            "sector": "IT",
            "exchange": "NSE",
            "market_cap": 1350000000000
        }
    )
    
    # Create sectors
    energy = create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy",
        properties={"index": "NIFTY_ENERGY"}
    )
    
    it_sector = create_entity(
        entity_type="Sector",
        id="IT",
        name="Information Technology",
        properties={"index": "NIFTY_IT"}
    )
    
    # Add entities
    kg.add_entity(reliance)
    kg.add_entity(tcs)
    kg.add_entity(energy)
    kg.add_entity(it_sector)
    
    # Create relationships
    rel1 = Relationship(
        id="REL_ENERGY",
        name="OPERATES_IN",
        subject=reliance,
        predicate="OPERATES_IN",
        object=energy,
        properties=RelationshipProperties(
            confidence=1.0,
            sources=["company_profile"]
        )
    )
    
    rel2 = Relationship(
        id="TCS_IT",
        name="OPERATES_IN",
        subject=tcs,
        predicate="OPERATES_IN",
        object=it_sector,
        properties=RelationshipProperties(
            confidence=1.0,
            sources=["company_profile"]
        )
    )
    
    kg.add_relationship(rel1)
    kg.add_relationship(rel2)
    
    # Print stats
    stats = kg.get_stats()
    print("Knowledge Graph Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return kg

# Run
kg = asyncio.run(build_simple_kg())
```

### Example 2: News-Based KG with Sentiment

```python
import asyncio
from financial_kg import (
    get_gemini_client,
    create_entity,
    Relationship,
    RelationshipProperties,
    TemporalAttributes,
    KnowledgeGraph,
    get_market_session,
    get_trading_day
)
from datetime import datetime

async def extract_from_news():
    client = get_gemini_client()
    kg = KnowledgeGraph()
    
    # Example news article
    news_text = """
    Mumbai, Jan 15, 2024 - Reliance Industries announced record quarterly 
    earnings today, surpassing analyst expectations. The energy giant reported 
    a 20% increase in profits, driven by strong performance in its 
    petrochemicals division.
    """
    
    # Use Gemini to extract entities and sentiment
    prompt = f"""
    Extract financial entities and relationships from this news:
    
    {news_text}
    
    Return in JSON format:
    {{
        "entities": [
            {{"name": "...", "type": "Company|Person|Event", "properties": {{}}}}
        ],
        "relationships": [
            {{"subject": "...", "predicate": "...", "object": "...", "sentiment": -1 to 1}}
        ]
    }}
    """
    
    response = await client.generate_async(prompt, use_pro=False)
    print(f"Extracted: {response}")
    
    # Parse and build KG (simplified - in practice, use proper JSON parsing)
    # ... (entity and relationship creation based on extracted data)
    
    return kg

# Run
kg = asyncio.run(extract_from_news())
```

## 📚 Next Steps

1. **Explore Notebooks**: Check `notebooks/` for detailed examples
2. **Process Real Data**: Use scripts in `scripts/` to process downloaded datasets
3. **Visualize in Neo4j**: Use Neo4j Browser to explore your KG
4. **Build Multi-Hop Q&A**: Integrate with your NLP-MultiHop-Q-A project

## 🔧 Common Issues

### Issue: "Gemini API key not found"
**Solution**: Ensure you've set `GEMINI_API_KEY` in `config/.env`

### Issue: "Module not found"
**Solution**: Make sure you're in the project root and virtual environment is activated

### Issue: "Neo4j connection failed"
**Solution**: Install and start Neo4j, or skip visualization for now

## 📖 More Examples

- **Building temporal KG from time-series news**: See `notebooks/temporal_kg_example.ipynb`
- **Processing NIFTY-50 data**: See `scripts/process_nifty50.py`
- **Sentiment analysis integration**: See `extractors/sentiment_analyzer.py`

## 🆘 Getting Help

- Check the [README](README.md) for architecture overview
- Read [ANALYSIS_AND_PLAN](ANALYSIS_AND_PLAN.md) for methodology details
- Review itext2kg documentation: https://github.com/auvalab/itext2kg

---

**Ready to build financial knowledge graphs! 🚀📈**
