# FinancialKG: Multi-Modal Financial Knowledge Graph with Gemini

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

> A novel implementation of financial knowledge graph construction using Google's Gemini models, combining structured market data with unstructured financial news for comprehensive market intelligence.

## 🎯 Project Overview

**FinancialKG** extends the methodology of [itext2kg](https://github.com/auvalab/itext2kg) and ATOM for the financial domain, with several key innovations:

- ✨ **Multi-Modal Construction**: Combines structured data (OHLCV, financial metrics) with unstructured text (news, earnings calls)
- ✨ **Gemini-Powered**: Leverages Google Gemini 2.0 Flash & 1.5 Pro for extraction and reasoning
- ✨ **4-Dimensional Temporal Modeling**: t_announce, t_effective, t_observe, t_impact
- ✨ **Sentiment-Aware Relationships**: Financial sentiment scoring on all relationships
- ✨ **Real-Time Updates**: Streaming pipeline for live market data integration
- ✨ **Multi-Hop Reasoning**: Integrated query layer for complex financial questions

## 📊 Datasets Covered

### Stock Market Data (Structured)
- ☑ NIFTY-50 Stock Market Data (Kaggle)
- ☑ Stock Price History Dataset (Mendeley)
- ☑ Asian Stock Market Data (Mendeley)
- ☑ Detailed Financial Data - 4456 Companies (Kaggle)
- ☑ Macro Market Data 1980-2024 (Mendeley)

### Financial News (Unstructured)
- ☑ IN-FINews News Corpus (Zenodo)
- ☑ Indian Financial News - 26k articles (HuggingFace)
- ☑ Financial News Headlines (HuggingFace)
- ☑ NIFTY News Headlines (HuggingFace)

### Specialized Data
- 🔍 MiMIC Earnings Calls (In Progress)
- 🔍 BASIR Budget Impact Analysis (In Progress)

## 🏗️ Architecture

```
Data Sources → Atomic Extractors → Entity Resolution → Temporal Merge → Neo4j KG
     ↓              ↓                    ↓                  ↓              ↓
 Structured    Gemini 2.0          Ticker Mapping    4-Time Model    Multi-Hop Q&A
    +          Flash/1.5 Pro       + Embeddings      + Sentiment     Reasoning
Unstructured
```

## 📁 Project Structure

```
financial_kg/
├── data_loaders/          # Dataset loaders for all data sources
├── extractors/            # Gemini-based atomic fact extractors
├── kg_builder/            # Entity resolution & relationship building
├── models/                # Data models & schemas
├── storage/               # Neo4j & vector storage
├── utils/                 # Utilities (Gemini client, config, logging)
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Processing scripts
├── tests/                 # Unit tests
├── config/                # Configuration files
└── data/                  # Data directory (gitignored)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd g:\projects\NLP\financial_kg

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Set up environment variables
cp config/.env.example config/.env

# Edit config/.env with your API keys:
# - GEMINI_API_KEY=your_gemini_api_key
# - NEO4J_URI=bolt://localhost:7687
# - NEO4J_USERNAME=neo4j
# - NEO4J_PASSWORD=your_password
```

### 3. Run Sample Pipeline

```python
import asyncio
from financial_kg import FinancialKG

async def main():
    # Initialize the FinancialKG
    fkg = FinancialKG()
    
    # Load NIFTY-50 data
    stock_data = await fkg.load_stock_data("NIFTY-50")
    
    # Load financial news
    news_data = await fkg.load_news_data("IN-FINews", limit=1000)
    
    # Build knowledge graph
    kg = await fkg.build_graph(
        stock_data=stock_data,
        news_data=news_data,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Visualize in Neo4j
    await fkg.visualize(kg)
    
    # Run multi-hop query
    result = await fkg.query(
        "What caused RELIANCE stock price to drop in January 2024?"
    )
    print(result)

asyncio.run(main())
```

## 🔬 Key Innovations

### 1. Multi-Modal Knowledge Graph
Unlike text-only approaches (ATOM, iText2KG), we integrate:
- **Structured**: OHLCV prices, financial ratios, macro indicators
- **Unstructured**: News articles, earnings calls, budget documents
- **Cross-Validation**: News events ↔ Price movements correlation

### 2. 4-Dimensional Temporal Model
```python
class FinancialRelationship:
    t_announce: datetime     # When information announced (news timestamp)
    t_effective: datetime    # When it takes legal/business effect
    t_observe: datetime      # When we collected the data
    t_impact: TimeRange      # Price impact window
    market_session: str      # pre_market, regular, post_market
```

### 3. Sentiment-Aware Relationships
```python
# Example: "RBI announces interest rate hike"
Relationship(
    subject=Entity("RBI", type="Organization"),
    predicate="ANNOUNCES",
    object=Entity("Interest Rate Hike", type="Policy"),
    sentiment=-0.65,  # Negative for market
    confidence=0.95,
    sources=["news_article_123"]
)
```

### 4. Gemini Model Selection
- **Gemini 2.0 Flash**: Fast atomic fact extraction (high volume)
- **Gemini 1.5 Pro**: Complex financial reasoning (earnings calls, 2M context)
- **Gemini Embeddings**: Entity/relationship similarity

### 5. Real-Time Pipeline
```
News Feed → Every 15min → Atomic Facts → KG Update → Alert System
```

## 📊 Entity & Relationship Schema

### Entity Types
- **Company**: ticker, sector, exchange, market_cap, fundamentals
- **Person**: role (CEO, CFO, Analyst), affiliation
- **Sector**: name, index, industry_group
- **Event**: type (earnings, merger, scandal), severity
- **Policy**: type (monetary, fiscal), effective_date
- **Indicator**: name (GDP, CPI, IIP), value, unit

### Relationship Types
- **OPERATES_IN** (Company → Sector)
- **LISTED_ON** (Company → Exchange)
- **ANNOUNCED** (Company → Event) [sentiment]
- **AFFECTS** (Event → Company) [sentiment, magnitude]
- **LED_BY** (Company → Person)
- **IMPACTS** (Policy → Sector) [sentiment]
- **CORRELATES_WITH** (Company ↔ Company) [correlation]
- **MENTIONED_IN** (Company → News) [sentiment, prominence]

## 🎯 Use Cases

1. **Market Analysis**: "Which sectors were most affected by the 2024 budget?"
2. **Event Impact**: "How did Reliance's Q2 earnings affect its suppliers?"
3. **Sentiment Tracking**: "What's the overall sentiment toward IT sector this week?"
4. **Prediction Features**: "Companies likely to benefit from new EV policy?"
5. **Risk Monitoring**: "Detect companies with increasing negative sentiment"

## 📈 Comparison with ATOM

| Feature | ATOM | FinancialKG |
|---------|------|-------------|
| Data Type | Text-only | Multi-modal |
| LLM | GPT-4 | Gemini 2.0/1.5 |
| Temporal Model | 2-time | 4-time |
| Domain | General | Financial |
| Sentiment | No | Yes |
| Real-time | Batch | Streaming |
| Context Window | 128K | 2M tokens |

## 🧪 Evaluation Metrics

- **Coverage**: % of NIFTY-50 companies represented
- **Temporal Accuracy**: Correct event-price correlation
- **Sentiment Accuracy**: Agreement with human analysts
- **Completeness**: Entity/relationship extraction rate
- **Query Performance**: Multi-hop question answering accuracy

## 🛣️ Roadmap

- [x] Phase 1: Foundation & data loaders
- [x] Phase 2: Core KG construction
- [ ] Phase 3: Data processing (In Progress)
- [ ] Phase 4: Advanced features (real-time, dashboard)
- [ ] Phase 5: Evaluation & optimization

## 📚 References

- **ATOM**: [AdapTive and OptiMized DTKG](https://arxiv.org/abs/2510.22590)
- **iText2KG**: [Incremental KG Construction](https://arxiv.org/abs/2409.03284)
- **Gemini**: [Google's Multimodal AI](https://deepmind.google/technologies/gemini/)

## 🤝 Contributing

This is a research project. Contributions, suggestions, and feedback are welcome!

## 📄 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

Built as part of NLP Multi-Hop Q&A research project

---

**Status**: Active Development 🚧
**Last Updated**: January 2026
