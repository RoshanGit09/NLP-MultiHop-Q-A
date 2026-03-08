# FinTraceQA — Multilingual Multi-Hop Financial Q&A System

A full-stack AI system for answering complex, multi-hop financial questions over an Indian Financial Events Knowledge Graph. Supports **7 languages** (English + 6 Indian languages) with a locally trained translation model, a FastAPI backend, and a React Native mobile frontend.

---

## 📌 Project Summary

FinTraceQA combines three systems into one end-to-end pipeline:

| Component | What it does |
|-----------|-------------|
| **Financial KG** (`financial_kg/`) | Builds an Indian financial knowledge graph from structured stock data and unstructured financial news using Google Gemini, stored in Neo4j |
| **Backend** (`backend/`) | FastAPI server that answers multi-hop financial questions by decomposing queries, traversing the KG, and synthesising natural-language answers via DeepSeek-V3 |
| **Translation Model** (`translation_model/`) | Custom 176M-parameter Encoder-Decoder Transformer trained on the Samanantar corpus for English ↔ Hindi/Tamil/Telugu/Marathi/Kannada/Malayalam |
| **Mobile App** (`Finterest/`) | React Native (Expo) app with multilingual UI, live financial news feed, and AI chatbot powered by the backend |

### System Architecture

```
User (Mobile App)
       │
       ▼
  Finterest (React Native / Expo)
       │  REST API
       ▼
  FastAPI Backend  ──────────────────────────────────────────┐
       │                                                      │
  ┌────┴────────────────────────────────┐                    │
  │  Language Router (Unicode detect)  │                    │
  │  → local Transformer translation   │                    │
  │  → DeepSeek-V3 fallback            │                    │
  └────┬────────────────────────────────┘                    │
       │                                                      │
  Question Decomposer (DeepSeek-V3)                          │
       │                                                      │
  Entity + Relation Linker                                   │
       │                                                      │
  KG Traversal Engine (multi-hop, up to 4 hops)             │
       │                                                      │
  Neo4j Aura (Financial KG) ◄────── financial_kg/ builder ──┘
       │
  Answer Synthesiser (DeepSeek-V3)
       │
  Multilingual Response → User
```

### Supported Languages

| Code | Language | Script |
|------|----------|--------|
| `en` | English | Latin |
| `hi` | Hindi | Devanagari |
| `mr` | Marathi | Devanagari |
| `ta` | Tamil | Tamil |
| `te` | Telugu | Telugu |
| `kn` | Kannada | Kannada |
| `ml` | Malayalam | Malayalam |

---

## 📁 Repository Structure

```
NLP-MultiHop-Q-A/
├── backend/                   ← FastAPI server (main application)
│   ├── app/
│   │   ├── main.py            ← Entry point, CORS config
│   │   └── routers/chat.py    ← POST /chat, POST /resolve, GET /health, GET /news
│   ├── core/
│   │   ├── lang_router.py     ← Language detection (Unicode + langdetect)
│   │   ├── translator.py      ← Local model → DeepSeek translation pipeline
│   │   ├── local_translator.py← Wrapper for trained Transformer model
│   │   ├── decomposer.py      ← Multi-hop question decomposer
│   │   ├── entity_linker.py   ← Entity mention → KG node matching
│   │   ├── relation_linker.py ← Predicate → KG relation type
│   │   ├── traversal.py       ← Multi-hop KG traversal + scoring
│   │   ├── synthesizer.py     ← KG path → natural-language answer
│   │   ├── memory.py          ← Session memory + coreference
│   │   └── kg_adapter/        ← Neo4j and mock KG adapters
│   ├── configs/weights.yaml   ← Scoring weights (tunable)
│   ├── scripts/eval_benchmark.py
│   ├── .env.example
│   └── README.md
│
├── financial_kg/              ← Knowledge Graph builder
│   ├── build_kg.py            ← Main KG build pipeline
│   ├── build_production_kg.py ← Production-grade builder
│   ├── kg_query.py            ← KG query utilities
│   ├── kg_cleaner.py          ← Deduplication + normalisation
│   ├── kg_entity_merger.py    ← Entity resolution
│   ├── kg_neo4j_upload.py     ← Upload to Neo4j
│   ├── extractors/            ← Gemini-powered entity/relation extractors
│   ├── data_loaders/          ← Dataset-specific loaders
│   ├── storage/               ← Neo4j storage adapter
│   ├── models/                ← KG data models (Entity, Relationship)
│   └── utils/                 ← Logging, config helpers
│
├── translation_model/         ← Indic translation model (training + inference)
│   ├── 1_train_tokenizer.py   ← SentencePiece tokenizer training
│   ├── 2_prepare_data.py      ← Samanantar dataset preparation
│   ├── 3_train_model.py       ← Model training script
│   ├── inference.py           ← MultilingualInference class
│   ├── models/transformer.py  ← Encoder-Decoder Transformer architecture
│   ├── tokenizer/             ← Trained SentencePiece model (100K vocab)
│   ├── checkpoints/           ← Trained model weights (final_model.pt)
│   └── docs/                  ← Training documentation
│
├── Finterest/                 ← React Native mobile app
│   ├── src/
│   │   ├── screens/           ← Login, Signup, News, Chatbot, Profile
│   │   ├── navigation/        ← Stack + Tab navigators
│   │   ├── services/apiService.ts ← Backend REST client
│   │   ├── firebase/          ← Supabase auth service
│   │   └── context/           ← Auth context
│   ├── app/                   ← Expo Router pages
│   └── package.json
│
├── data/                      ← KG source data (not tracked in git)
├── .gitignore
├── requirements.txt           ← All Python dependencies
├── PROJECT_REPORT.md          ← Full technical report
└── README.md                  ← This file
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.12+ |
| Node.js | 18+ |
| npm / npx | 9+ |
| Neo4j Aura (or local) | 5.x |
| DeepSeek API key | — |
| Google Gemini API key | (KG build only) |
| NewsAPI key | (optional, for live news) |

---

## 1. Python Environment Setup

All Python components share one virtual environment at the project root.

```powershell
# From NLP-MultiHop-Q-A\
python -m venv .venv
.\.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

## 2. Backend (FastAPI Server)

### Environment Variables

```powershell
Copy-Item backend\.env.example backend\.env
```

Edit `backend\.env`:

```env
# Neo4j connection
NEO4J_URI=neo4j+s://<your-aura-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here

# DeepSeek-V3 (used for decomposition, synthesis, and translation fallback)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# NewsAPI (optional — stubs are returned if not set)
NEWSAPI_KEY=your_newsapi_key_here

# Use in-memory mock KG instead of Neo4j (for development)
USE_MOCK_KG=false

PORT=8000
LOG_LEVEL=INFO
```

### Run the Server

```powershell
# Activate venv first
.\.venv\Scripts\activate

# Run with Neo4j
$env:PYTHONPATH = "."
.\.venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Run with mock KG (no Neo4j needed)
$env:USE_MOCK_KG = "true"
.\.venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Ask a financial question (multilingual) |
| `POST` | `/resolve` | Resolve an entity mention to KG nodes |
| `GET` | `/news` | Fetch latest Indian financial news |

**Example — chat request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"Who is the CEO of Infosys?","lang":"en"}'

# Hindi query (auto-detected)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"Infosys के CEO कौन हैं?","lang":"auto"}'
```

---

## 3. Financial Knowledge Graph Builder

### Environment Variables

Create `financial_kg/config/.env` (or set in shell):

```env
GEMINI_API_KEY=your_gemini_api_key_here
NEO4J_URI=neo4j+s://<your-aura-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

### Install KG-Specific Dependencies

```powershell
.\.venv\Scripts\activate
pip install -r financial_kg/requirements.txt
```

### Build the Knowledge Graph

```powershell
# Step 1 — Build KG from data sources
.\.venv\Scripts\python financial_kg/build_production_kg.py

# Step 2 — Clean and deduplicate
.\.venv\Scripts\python financial_kg/kg_cleaner.py

# Step 3 — Merge duplicate entities
.\.venv\Scripts\python financial_kg/kg_entity_merger.py

# Step 4 — Upload to Neo4j
.\.venv\Scripts\python financial_kg/kg_neo4j_upload.py

# Query the KG
.\.venv\Scripts\python financial_kg/kg_query.py
```

---

## 4. Translation Model

The trained model (`translation_model/checkpoints/final_model.pt`) and tokenizer (`translation_model/tokenizer/multilingual_indic-3.model`) are **not tracked in git** (too large). They must be trained or obtained separately.

### Train from Scratch

```powershell
.\.venv\Scripts\activate

# Step 1 — Train SentencePiece tokenizer (100K vocab, EN + 6 Indic languages)
.\.venv\Scripts\python translation_model/1_train_tokenizer.py

# Step 2 — Download & prepare Samanantar parallel corpus
.\.venv\Scripts\python translation_model/2_prepare_data.py

# Step 3 — Train the Encoder-Decoder Transformer
.\.venv\Scripts\python translation_model/3_train_model.py
# Default: 3 epochs, batch_size=16, lr=5e-4
# Outputs: translation_model/checkpoints/final_model.pt

# Step 4 — Run inference
.\.venv\Scripts\python translation_model/inference.py
```

### Training Requirements

- GPU strongly recommended (model is 176M parameters)
- ~50 GB disk space for Samanantar corpus
- ~8 GB GPU VRAM for batch_size=16

> **Note:** The backend automatically falls back to DeepSeek-V3 for translation if `final_model.pt` is not present.

---

## 5. Mobile App (Finterest)

### Prerequisites

- Node.js 18+
- Expo CLI: `npm install -g expo-cli`
- Expo Go app on your phone (or Android/iOS emulator)

### Supabase Setup

1. Create a project at [supabase.com](https://supabase.com)
2. Run the SQL in `Finterest/supabase_setup.sql` in your Supabase SQL editor
3. Update `Finterest/src/firebase/config.ts` with your Supabase URL and anon key

### Install & Run

```powershell
cd Finterest

# Install dependencies
npm install

# Start Expo development server
npm start
# or: npx expo start

# Run on specific platform
npm run android    # Android emulator / device
npm run ios        # iOS simulator (macOS only)
npm run web        # Browser (limited features)
```

### Connect to Backend

In `Finterest/src/services/apiService.ts`, set the backend URL:

```typescript
// For physical device (replace with your machine's local IP)
const BASE_URL = "http://192.168.x.x:8000";

// For emulator
const BASE_URL = "http://10.0.2.2:8000";   // Android
const BASE_URL = "http://localhost:8000";   // iOS simulator
```

---

## 🔑 Required API Keys Summary

| Key | Used By | Where to get |
|-----|---------|-------------|
| `DEEPSEEK_API_KEY` | Backend (LLM) | [platform.deepseek.com](https://platform.deepseek.com) |
| `NEO4J_URI / PASSWORD` | Backend + KG builder | [console.neo4j.io](https://console.neo4j.io) (Aura free tier) |
| `GEMINI_API_KEY` | KG builder only | [aistudio.google.com](https://aistudio.google.com) |
| `NEWSAPI_KEY` | Backend (news feed) | [newsapi.org](https://newsapi.org) (free tier) |
| Supabase URL + anon key | Mobile app | [supabase.com](https://supabase.com) |

---

## 📄 License

- Code: [MIT License](https://opensource.org/licenses/MIT)
- Trained model: CC-BY-4.0 (Samanantar / Sangraha data)
- Tokenizer: CC-BY-4.0

