# FinancialKG - Quick Command Reference

**Essential commands for working with the FinancialKG project**

---

## 🚀 Initial Setup

```bash
# Navigate to project
cd g:\projects\NLP\financial_kg

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Copy environment template
copy config\.env.example config\.env

# Edit .env file with your API keys
notepad config\.env
# Add: GEMINI_API_KEY=your_key_here
```

---

## 🧪 Quick Tests

### Test Configuration
```python
python -c "from financial_kg import get_config; print('Config loaded:', get_config().gemini.model_fast)"
```

### Test Gemini Client
```python
python -c "import asyncio; from financial_kg import get_gemini_client; print(asyncio.run(get_gemini_client().generate_async('Hello')))"
```

### Test Entity Creation
```python
python -c "from financial_kg import create_entity; c = create_entity('Company', id='TEST', name='Test Co'); print(f'Created: {c.name}')"
```

### Run Full Demo
```bash
python examples\simple_demo.py
```

---

## 📥 Download Datasets

### Setup Kaggle
```bash
# Install kaggle CLI
pip install kaggle

# Configure credentials (get from kaggle.com/account)
# Place kaggle.json in: C:\Users\<username>\.kaggle\
# Or set environment variables
set KAGGLE_USERNAME=your_username
set KAGGLE_KEY=your_api_key
```

### Download NIFTY-50
```bash
mkdir data\nifty50
kaggle datasets download -d rohanrao/nifty50-stock-market-data
tar -xf nifty50-stock-market-data.zip -C data\nifty50\
```

### Download Financial Metrics
```bash
mkdir data\financial_metrics
kaggle datasets download -d sameerprogrammer/detailed-financial-data-of-4456-nse-and-bse-company
tar -xf detailed-financial-data-of-4456-nse-and-bse-company.zip -C data\financial_metrics\
```

---

## 💻 Common Development Commands

### Load Stock Data
```python
from financial_kg.data_loaders.stock_loader import StockDataLoader

loader = StockDataLoader()
reliance_df = loader.load_stock_csv('RELIANCE')
print(f"Loaded {len(reliance_df)} rows")

# Get price on specific date
price = loader.get_price_on_date('RELIANCE', '2024-01-15')
print(f"Close: ₹{price['close']}")

# Calculate price change
change = loader.calculate_price_change('RELIANCE', '2024-01-01', '2024-01-31')
print(f"Change: {change:.2f}%")
```

### Load News Data
```python
from financial_kg.data_loaders.news_loader import NewsDataLoader

loader = NewsDataLoader()

# Load Indian Financial News (requires internet)
news = loader.load_indian_financial_news(max_items=100)
print(f"Loaded {len(news)} articles")

# Search by keyword
reliance_news = loader.search_by_keyword(news, ['Reliance', 'RIL'])
print(f"Found {len(reliance_news)} Reliance articles")
```

### Extract Atomic Facts
```python
import asyncio
from financial_kg.extractors.gemini_atomic_extractor import GeminiAtomicExtractor

async def extract():
    extractor = GeminiAtomicExtractor(use_pro=False)
    
    text = "Reliance Industries reported Q2 earnings of ₹18,500 crore on Jan 15, 2024."
    facts = await extractor.extract_atomic_facts(text, observation_date="2024-01-15")
    
    for fact in facts:
        print(f"• {fact}")

asyncio.run(extract())
```

### Build Knowledge Graph
```python
from financial_kg import create_entity, Relationship, KnowledgeGraph

# Create KG
kg = KnowledgeGraph()

# Add entities
company = create_entity(
    entity_type="Company",
    id="RELIANCE",
    name="Reliance Industries",
    properties={"ticker": "RELIANCE", "sector": "Energy"}
)
kg.add_entity(company)

# Get stats
stats = kg.get_stats()
print(f"Entities: {stats['total_entities']}")
```

---

## 📊 Data Processing Workflows

### Process Single Company
```python
from financial_kg.data_loaders.stock_loader import StockDataLoader
from financial_kg import create_entity, KnowledgeGraph

# Load stock data
loader = StockDataLoader()
df = loader.load_stock_csv('RELIANCE')

# Detect significant movements
significant = loader.detect_significant_movements('RELIANCE', threshold=5.0)
print(f"Found {len(significant)} days with >5% movement")

# Create entity
company = create_entity(
    entity_type="Company",
    id="RELIANCE",
    name="Reliance Industries",
    properties={
        "ticker": "RELIANCE",
        "sector": "Energy",
        "total_days": len(df),
        "significant_movements": len(significant)
    }
)
```

### Batch Process News
```python
import asyncio
from financial_kg.data_loaders.news_loader import NewsDataLoader
from financial_kg.extractors.gemini_atomic_extractor import GeminiAtomicExtractor

async def process_news():
    news_loader = NewsDataLoader()
    extractor = GeminiAtomicExtractor()
    
    # Load news
    articles = news_loader.load_indian_financial_news(max_items=10)
    
    # Extract facts from all
    texts = [a['content'] for a in articles]
    dates = [a['date'] for a in articles]
    
    all_facts = await extractor.extract_batch(texts, observation_dates=dates)
    
    for i, facts in enumerate(all_facts):
        print(f"\nArticle {i+1}: {len(facts)} facts")

asyncio.run(process_news())
```

---

## 🔧 Debugging Commands

### Check Installation
```bash
python -c "import financial_kg; print(f'FinancialKG v{financial_kg.__version__}')"
```

### Verify Data Directories
```bash
python -c "from financial_kg import get_config; c = get_config(); print(f'Data: {c.data.data_dir}')"
```

### Test Gemini Connection
```bash
python -c "import asyncio; from financial_kg.utils.gemini_client import GeminiClient; print(asyncio.run(GeminiClient().generate_async('test')))"
```

### View Logs
```bash
# Logs are in logs/ directory
type logs\financial_kg_*.log
```

---

## 📖 Documentation Quick Access

```bash
# View README
type README.md | more

# View getting started
type GETTING_STARTED.md | more

# View dataset guide
type DATASET_ACCESS_GUIDE.md | more

# View all markdown files
dir *.md
```

---

## 🎯 Next Steps Checklist

```bash
# 1. Setup environment
[ ] python -m venv venv && venv\Scripts\activate
[ ] pip install -r requirements.txt
[ ] copy config\.env.example config\.env (add API key)

# 2. Test installation
[ ] python -c "from financial_kg import get_config; print('OK')"
[ ] python examples\simple_demo.py

# 3. Download data
[ ] Configure Kaggle credentials
[ ] Download NIFTY-50 dataset
[ ] Download news datasets

# 4. Start building
[ ] Load your first stock data
[ ] Extract atomic facts from news
[ ] Build your first KG
[ ] Visualize in Neo4j (optional)
```

---

## 💡 Useful Aliases (PowerShell)

Add to your PowerShell profile:

```powershell
function fkg-activate { Set-Location "g:\projects\NLP\financial_kg"; .\venv\Scripts\Activate.ps1 }
function fkg-demo { python examples\simple_demo.py }
function fkg-test { python -c "from financial_kg import get_config; print('✓ FinancialKG ready!')" }
```

Then use:
```bash
fkg-activate  # Navigate and activate venv
fkg-demo      # Run demo
fkg-test      # Quick test
```

---

## 🆘 Common Issues & Solutions

### "Module not found"
```bash
# Make sure you're in venv
venv\Scripts\activate
pip install -r requirements.txt
```

### "Gemini API key not found"
```bash
# Check .env file exists
type config\.env
# Should show GEMINI_API_KEY=...
```

### "Stock data file not found"
```bash
# Download and extract NIFTY-50 data first
# See DATASET_ACCESS_GUIDE.md
```

### "HuggingFace dataset error"
```bash
# Install datasets library
pip install datasets

# Set HF token if needed (for private datasets)
set HF_TOKEN=your_token
```

---

**For more detailed information, see:**
- [GETTING_STARTED.md](GETTING_STARTED.md) - Complete setup guide
- [PROJECT_INDEX.md](PROJECT_INDEX.md) - Navigation reference
- [ACCOMPLISHMENTS.md](ACCOMPLISHMENTS.md) - What's been built

**Ready to build financial knowledge graphs! 🚀📊**
