# FinTraceQA: A Multilingual Multi-Hop Question Answering System for Indian Financial Markets

**Course**: Natural Language Processing (Semester 6)  
**Repository**: NLP-MultiHop-Q-A  
**Stack**: Python · FastAPI · Neo4j Aura · DeepSeek-V3 · React Native (Expo) · PyTorch  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Financial Knowledge Graph](#3-financial-knowledge-graph)
4. [Multi-Hop QA Backend](#4-multi-hop-qa-backend)
5. [Multilingual Translation Model](#5-multilingual-translation-model)
6. [Multilingual Pipeline](#6-multilingual-pipeline)
7. [Frontend Application — Finterest](#7-frontend-application--finterest)
8. [APIs and Endpoints](#8-apis-and-endpoints)
9. [Evaluation and Results](#9-evaluation-and-results)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [References](#11-references)

---

## 1. Project Overview

**FinTraceQA** is an end-to-end multilingual question answering system that answers complex, multi-hop financial questions about the Indian stock market. Unlike single-hop QA systems that retrieve one fact, FinTraceQA follows chains of relationships in a Knowledge Graph — for example, to answer *"Which company that Mukesh Ambani controls had the highest revenue growth after the 2020 oil price crash?"*, the system must:

1. Find all companies linked to Mukesh Ambani
2. Filter those that are oil-related
3. Retrieve post-2020 revenue data for each
4. Rank by growth rate

The system supports **7 languages**: English, Hindi, Tamil, Telugu, Marathi, Kannada, and Malayalam — accepting queries in any of these and returning answers in the same language.

### Key Contributions

| Component | What was built |
|---|---|
| Financial KG | Multi-source Indian financial knowledge graph (~3,300+ entities, 10,000+ relationships) stored in Neo4j |
| Multi-Hop Engine | 4-type question decomposition → graph traversal → faithful answer synthesis |
| Translation Model | Custom Encoder-Decoder Transformer (176M params) trained on ~3M Samanantar EN↔Indic pairs |
| Backend API | FastAPI server with smart Neo4j failover, caching, session memory |
| Mobile App | React Native (Expo) app with chat UI, live financial news, language switching |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Finterest (React Native App)                 │
│   Chat UI  │  News Feed  │  Language Picker  │  Explain Panel  │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP REST  (port 8000)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                              │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ LangRouter  │   │  Decomposer  │   │  AnswerSynthesizer   │ │
│  │ (Unicode    │   │  (DeepSeek)  │   │  (DeepSeek)          │ │
│  │  detection) │   └──────┬───────┘   └──────────────────────┘ │
│  └──────┬──────┘          │                      ▲             │
│         │           ┌─────▼──────┐               │             │
│         │           │  Entity    │   ┌────────────┴──────────┐ │
│         │           │  Linker    │   │  KG Traversal Engine  │ │
│         │           └─────┬──────┘   │  (multi-hop scoring)  │ │
│         │                 └──────────►                       │ │
│  ┌──────▼──────┐                     └────────────┬──────────┘ │
│  │ Translator  │◄────────────────────────────────┘            │
│  │ (Local +    │                                               │
│  │  DeepSeek)  │                                               │
│  └─────────────┘                                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │  Bolt (neo4j+s://)
                                 ▼
                    ┌────────────────────────┐
                    │    Neo4j Aura (Cloud)  │
                    │   Financial KG         │
                    │   ~3300 entities       │
                    │   ~10000 relationships │
                    └────────────────────────┘
```

### Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Backend language | Python | 3.12.1 |
| Web framework | FastAPI + Uvicorn | 0.111 |
| LLM (QA synthesis) | DeepSeek-V3 (`deepseek-chat`) | via OpenAI-compatible API |
| Graph database | Neo4j Aura | Cloud |
| Graph driver | `neo4j` Python driver | 5.x |
| Translation model | Custom Transformer (PyTorch) | 176M params |
| Tokenizer | SentencePiece (Unigram) | 100K vocab |
| Frontend | React Native + Expo | SDK 52 |
| Language detection | Unicode block counting + `langdetect` | — |

---

## 3. Financial Knowledge Graph

### 3.1 Data Sources

The KG was built by ingesting **6 heterogeneous data sources**, each contributing different kinds of financial knowledge:

| # | Source | Format | Entities / Facts |
|---|---|---|---|
| 1 | IN-FINews Dataset (Zenodo) | JSON — 3,348 Indian financial news articles | Events, company sentiments, market relations |
| 2 | NIFTY-50 Stock Market Data (Kaggle) | 49 CSV files — OHLCV from 2000–2021 | Price, volume, trading facts per ticker |
| 3 | `stock_metadata.csv` | Structured CSV | Company ↔ industry ↔ exchange mappings |
| 4 | India Macro Dataset (Mendeley) | XLSX — 1980–2024 | GDP, SENSEX levels, inflation rates |
| 5 | Asian Stock Market Data (Mendeley) | XLSX — multi-index | Cross-market index correlations |
| 6 | Stock Price History (Mendeley) | 6 ticker CSVs | Additional historical price series |

### 3.2 KG Construction Pipeline

```
Raw Data Sources
     │
     ▼
Structured → Text conversion
  OHLCV rows → natural language sentences
  "TCS traded at ₹3200 with volume 1.2M on 2021-03-15"
     │
     ▼
GeminiAtomicExtractor (Gemini 2.0 Flash)
  Extracts (entity, relation, entity) triples + metadata
  Entity types: Company, Person, Index, Sector, Country, Event
  Relation types: TRADED_AT, ACQUIRED, LEADS, PART_OF, IMPACTED_BY, ...
     │
     ▼
IncrementalKGBuilder — String-Similarity Deduplication
  - SequenceMatcher threshold: 0.85 (same label class only)
  - Cross-label merges NEVER allowed (Company ≠ Person)
  - Canonical entity selection: highest confidence, longer meaningful names
  - Self-loops silently dropped
  - Stable node IDs: {SLUG}_{8-char-SHA256}
     │
     ▼
NetworkX DiGraph (in-memory)
     │
     ▼
Neo4j Aura Upload (neo4j+s://745272d9.databases.neo4j.io)
  Nodes with: name, label, confidence, aliases
  Relationships with: type, time, source, sentiment_score
```

### 3.3 Entity and Relation Schema

**Node labels**:
- `Company` — e.g. Reliance Industries, TCS, Infosys
- `Person` — e.g. Mukesh Ambani, Ratan Tata
- `Index` — e.g. NIFTY-50, SENSEX, BSE 500
- `Sector` — e.g. IT, Energy, Banking, FMCG
- `Country` — e.g. India, China, USA
- `Event` — e.g. rate cut, earnings announcement, acquisition
- `Metric` — e.g. revenue, EPS, P/E ratio

**Key relationship types**:
- `TRADED_AT` — company traded at price on date
- `ACQUIRED` — M&A events with date
- `LEADS` / `FOUNDED_BY` — person–company links
- `PART_OF` — company–sector, index–market
- `IMPACTED_BY` — event causality chains
- `CORRELATED_WITH` — cross-market index correlations
- `REPORTED` — earnings/metric reporting events

### 3.4 Temporal Modeling

Every relationship carries up to **4 time dimensions**:

| Dimension | Meaning |
|---|---|
| `t_announce` | When the event was announced |
| `t_effective` | When the event took effect |
| `t_observe` | When the market price reflected the event |
| `t_impact` | Duration over which the impact was measured |

This enables temporal multi-hop queries like *"What happened to Reliance's stock price in the 3 months after the RIL-Jio announcement?"*

### 3.5 KG Scale

After deduplication and upload:
- **~3,300+ unique nodes** across all label types
- **~10,000+ relationships** with temporal and sentiment metadata
- **Neo4j Aura** cloud instance with TLS encryption (`neo4j+s://`)

---

## 4. Multi-Hop QA Backend

### 4.1 Backend Structure

```
backend/
├── app/
│   ├── main.py          — FastAPI app, CORS, news endpoint
│   └── routers/
│       └── chat.py      — /chat, /resolve, /translate, /entities
├── core/
│   ├── lang_router.py   — Language detection + query normalisation
│   ├── decomposer.py    — Question type classification + sub-question generation
│   ├── entity_linker.py — Entity fuzzy-matching against KG
│   ├── traversal.py     — Multi-hop path scoring and ranking
│   ├── synthesizer.py   — Answer generation from triples
│   ├── explanations.py  — Faithfulness verification + step generation
│   ├── memory.py        — Session-based conversation memory
│   ├── gemini_client.py — DeepSeek-V3 wrapper (OpenAI-compatible)
│   ├── translator.py    — Translation orchestration (local → DeepSeek fallback)
│   └── local_translator.py — Trained Transformer inference wrapper
└── .env                 — API keys, Neo4j credentials
```

### 4.2 Smart Neo4j Adapter

The backend uses a **transparent failover adapter** that never crashes:

```
Request arrives
    │
    ▼
SmartAdapter tries Neo4j (live data)
    │ success → return real KG results
    │ failure (ServiceUnavailable / SessionExpired)
    ▼
Fall back to MockAdapter (stub data)
    │
Background thread retries Neo4j every 30s
    │ reconnects silently → live data restored automatically
```

This means the server stays responsive even if the cloud Neo4j instance is temporarily unreachable.

### 4.3 Question Decomposition

The first step after language detection is **classifying and decomposing** the user's English query using DeepSeek-V3.

**4 question types supported**:

| Type | Description | Example |
|---|---|---|
| `single_hop` | One fact from KG | "What sector is TCS in?" |
| `multi_hop_chain` | Follow A→B→C chain | "What company did the person who founded Infosys previously work at?" |
| `multi_hop_intersection` | Two conditions met simultaneously | "Which NIFTY-50 company in the IT sector has P/E > 30?" |
| `temporal` | Time-bounded filtering | "What was Reliance's revenue after the 2020 oil crash?" |

DeepSeek returns structured JSON:
```json
{
  "question_type": "multi_hop_chain",
  "subquestions": [
    "What companies does Mukesh Ambani control?",
    "What was the revenue of each of those companies in FY2024?"
  ],
  "anchor_entities": ["Mukesh Ambani", "Reliance Industries"],
  "constraints": {"time_after": "2024-04-01"},
  "reasoning": "Requires entity traversal then metric lookup"
}
```

### 4.4 Entity Linking

The **EntityLinker** maps extracted entity strings to actual KG nodes:

- Uses **fuzzy string matching** (SequenceMatcher) with a configurable threshold
- Also checks **aliases** stored on each node (e.g. "RIL" → "Reliance Industries Limited")
- Scores each candidate: entity confidence × string similarity
- Returns ranked candidates — top candidate used for traversal start

### 4.5 Multi-Hop Traversal Engine

The `KGTraversalEngine` executes the decomposition plan against Neo4j:

```
For each sub-question:
  1. Start at anchor entity node
  2. Expand neighbors (configurable depth: 1–3 hops)
  3. Filter by relation type constraints from decomposer
  4. Apply temporal constraints (time_after / time_before)
  5. Score each path:
       path_score = entity_link_score × 0.4
                  + relation_link_score × 0.3
                  + temporal_match_score × 0.2
                  + hop_penalty × 0.1
  6. Rank and keep top-k paths

For multi_hop_chain:
  Chain paths across sub-questions — terminal node of path 1
  becomes start node of path 2.
```

**Scoring weights** are configurable via `backend/configs/weights.yaml`.

### 4.6 Answer Synthesis

The top-ranked reasoning paths (expressed as triples) are passed to DeepSeek-V3 with a carefully engineered prompt:

```
System: You are a financial analyst assistant. Given KG triples and
        a user question, synthesize a clear, accurate, fluent answer.
        Cite only facts from the provided triples. Be concise.

User:   Question: [user's question in English]
        Triples: [(Reliance, revenue, ₹9.7L cr, FY2024), ...]
```

DeepSeek produces a **fluent English paragraph** — not a template like "Answer: Reliance". This was the key improvement over initial rule-based synthesis.

### 4.7 Faithfulness Verification

After synthesis, every sentence in the answer is verified against the supporting triples using `FaithfulnessVerifier`:

- Each sentence is checked: "Is this claim supported by at least one triple?"
- Unsupported sentences are flagged or removed
- Produces `explanation_steps` — human-readable reasoning chain shown to the user

### 4.8 Session Memory

`memory.py` maintains per-session conversation history:
- Stores last N turns (question + answer) per `session_id`
- Passed as context to the decomposer so follow-up questions work naturally
- e.g. "What about TCS?" after asking about Infosys resolves correctly

### 4.9 Live Financial News

The `/financial-news` endpoint fetches **real Indian financial news** via NewsAPI:

- Queries: `site:economictimes.indiatimes.com OR site:moneycontrol.com OR site:livemint.com` + keywords `NIFTY NSE BSE Sensex stock`
- Returns 15 most recent articles
- **5-minute cache** — avoids hitting NewsAPI rate limits on every app refresh
- Falls back to stub articles if NewsAPI is unavailable

---

## 5. Multilingual Translation Model

### 5.1 Architecture

A custom **Encoder-Decoder Transformer** was designed and trained from scratch, based on *"Attention Is All You Need"* (Vaswani et al., 2017) with modern improvements:

```
Source Text (any language)
        │
        ▼  SentencePiece tokenization
        │
   ┌────▼──────────────────────┐
   │   Encoder (6 layers)      │
   │   d_model = 768           │
   │   heads = 12              │
   │   d_ff = 3072             │
   │   Sinusoidal pos. enc.    │
   │   Pre-LayerNorm           │
   │   GELU activation         │
   └────────────┬──────────────┘
                │  Cross-Attention
   ┌────────────▼──────────────┐
   │   Decoder (6 layers)      │
   │   Same config as encoder  │
   │   Starts with <2xx> token │
   │   (language direction)    │
   └────────────┬──────────────┘
                │
                ▼  Linear projection (weight-tied with embedding)
         Output token distribution
                │
                ▼  Greedy / Beam search decoding
         Translated text
```

**Model statistics**:

| Parameter | Value |
|---|---|
| Encoder layers | 6 |
| Decoder layers | 6 |
| Model dimension (`d_model`) | 768 |
| Attention heads | 12 |
| FFN dimension (`d_ff`) | 3072 |
| Max sequence length | 512 tokens |
| Dropout | 0.1 |
| Total parameters | **176,040,960 (~176M)** |
| Model size on disk | **671 MB** |

### 5.2 Key Design Choices

**Pre-Layer Normalization**: Applied before each sub-layer (not after), which stabilises training with large batch sizes and high learning rates.

**Weight Tying**: The output projection matrix is shared with the embedding matrix, reducing the parameter count and improving generalisation on the large vocabulary.

**Language Direction Tokens**: Instead of using BOS (`<s>`) to start decoding, the decoder is initialised with a language token (`<2hi>`, `<2ta>`, `<2en>`, etc.). This single token steers the decoder to output in the target language without any additional architecture changes.

### 5.3 Tokenizer

A **SentencePiece Unigram** tokenizer was trained on the combined multilingual corpus:

| Setting | Value |
|---|---|
| Algorithm | Unigram (better for morphologically-rich languages) |
| Vocabulary size | **100,000 tokens** |
| Character coverage | 99.95% (covers rare Brahmic script characters) |
| Normalisation | NFKC |
| Special tokens | `<pad>` (0), `<unk>` (1), `<s>` (2), `</s>` (3) |
| Language tokens | `<2en>` (9), `<2hi>` (10), `<2ta>` (11), `<2te>` (12), `<2mr>` (13), `<2kn>` (14), `<2ml>` (15) |

The unigram algorithm was chosen over BPE because Indian scripts (Devanagari, Tamil, Telugu, etc.) are morphologically rich — unigram produces lower token fertility (fewer tokens per word), which means the model needs fewer decoding steps.

### 5.4 Training Data — Samanantar

The model was trained on the **Samanantar** dataset — the largest publicly available parallel corpus for Indian languages:

| Language pair | Training pairs |
|---|---|
| English ↔ Hindi | 500K+ pairs |
| English ↔ Tamil | 500K+ pairs |
| English ↔ Telugu | 500K+ pairs |
| English ↔ Marathi | 500K+ pairs |
| English ↔ Kannada | 500K+ pairs |
| English ↔ Malayalam | 500K+ pairs |
| **Total** | **~3 million pairs** |

Data format (after tokenization):
```
Encoder input:  [156, 892, 34, 52, ...]   ← English token IDs
Decoder input:  [11, 1204, 567, 89, ...]  ← <2ta> + Tamil tokens (excluding last)
Labels:         [1204, 567, 89, ..., 3]   ← Tamil tokens + </s> (shifted by 1)
```

### 5.5 Training Setup

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 5×10⁻⁴ |
| Warmup steps | 4,000 (as per original Transformer paper) |
| LR schedule | Linear decay after warmup |
| Batch size | 16 per device |
| Gradient accumulation | 4 steps (effective batch = 64) |
| Epochs | 3 |
| β₁, β₂ | 0.9, 0.98 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Precision | bfloat16 (if supported) |
| Framework | HuggingFace `Trainer` with custom `TranslationTrainer` |

### 5.6 Inference

Two decoding strategies are supported:

**Greedy decoding** (default — fast):
- Temperature scaling applied (T = 0.8)
- Repetition penalty (1.2×) on already-generated tokens
- N-gram blocking (n=3) to prevent repeated phrases
- Used for query→English translation (speed matters)

**Beam search** (higher quality):
- Beam size = 5
- Length penalty = 1.0
- Same repetition penalties as greedy
- Available for answer→native translation when quality is prioritised

### 5.7 Observed Behaviour

The model successfully:
- Produces output in the correct target script
- Preserves proper nouns (company names, tickers) without translating them
- Captures the general topic of financial queries

Current limitations (due to general-domain training data):
- Occasional dropped words on long sentences
- Some UNK tokens (`⁇`) on rare financial terminology
- Grammar errors on complex sentence structures

These limitations are handled by the **DeepSeek fallback** — if the local model output is empty or malformed, the system automatically uses DeepSeek for translation.

---

## 6. Multilingual Pipeline

### 6.1 Language Detection

Language detection is performed **without any ML model** — it uses Unicode block counting:

```python
_TAMIL_RE    = re.compile(r"[\u0B80-\u0BFF]")   # Tamil: exclusively these codepoints
_TELUGU_RE   = re.compile(r"[\u0C00-\u0C7F]")
_KANNADA_RE  = re.compile(r"[\u0C80-\u0CFF]")
_MALAYALAM_RE= re.compile(r"[\u0D00-\u0D7F]")
_DEVANAGARI_RE=re.compile(r"[\u0900-\u097F]")   # Hindi + Marathi (same block)
```

For a Tamil query, every Tamil character falls in `U+0B80–U+0BFF` exclusively — no other script shares this range. Detection is therefore **100% accurate** for script-typed input.

The only ambiguous case is **Hindi vs Marathi** (both use Devanagari). For this, `langdetect` is called as a secondary check — if it returns `"mr"`, the language is Marathi, otherwise Hindi.

For Latin-script text (English or romanised Indian languages), `langdetect` is used to detect the language.

### 6.2 End-to-End Multilingual Flow

```
User query in language L
         │
         ▼ 1. DETECT
    LangRouter._detect()
    Unicode block → L ∈ {hi, mr, ta, te, kn, ml, en}
         │
         ▼ 2. TRANSLATE TO ENGLISH (if L ≠ en)
    translator.translate(query, L → en)
      → local Transformer first
      → DeepSeek fallback if local fails
    Result: English query string
         │
         ▼ 3. DECOMPOSE
    DeepSeek classifies question type
    Extracts sub-questions + anchor entities
         │
         ▼ 4. KG SEARCH
    EntityLinker + KGTraversalEngine on Neo4j
    Returns ranked reasoning paths (triples)
         │
         ▼ 5. SYNTHESIZE (always in English)
    DeepSeek: triples → fluent English answer
         │
         ▼ 6. VERIFY
    FaithfulnessVerifier: check each sentence
    Generate explanation steps
         │
         ▼ 7. TRANSLATE BACK (if L ≠ en)
    translator.translate_answer(answer, steps, L)
      → local Transformer first
      → DeepSeek fallback if local fails
         │
         ▼
    Response to user in language L
    { answer, answer_lang: L, explanation_steps, confidence }
```

### 6.3 Translation Priority

```
translate(text, source, target)
    │
    ├─► local_translator.translate_local()
    │       Load MultilingualInference (singleton, loaded once)
    │       SentencePiece tokenize → Tensor
    │       Encoder-Decoder forward pass
    │       Greedy decode → detokenize
    │       Strip <2xx> language token from output
    │       ↓ returns result if non-empty
    │
    └─► DeepSeek fallback (if local returns None)
            _generate(system=translate_prompt, user=text)
```

### 6.4 Supported Languages

| ISO 639-1 | Language | Script | Unicode Block |
|---|---|---|---|
| `en` | English | Latin | — |
| `hi` | Hindi | Devanagari | U+0900–U+097F |
| `mr` | Marathi | Devanagari | U+0900–U+097F |
| `ta` | Tamil | Tamil | U+0B80–U+0BFF |
| `te` | Telugu | Telugu | U+0C00–U+0C7F |
| `kn` | Kannada | Kannada | U+0C80–U+0CFF |
| `ml` | Malayalam | Malayalam | U+0D00–U+0D7F |

---

## 7. Frontend Application — Finterest

### 7.1 Technology

The frontend is a **React Native** mobile application built with **Expo SDK 52**, supporting iOS, Android, and web via a single codebase.

| Technology | Purpose |
|---|---|
| React Native + Expo | Cross-platform mobile app |
| TypeScript | Type-safe API contracts |
| Expo Router (file-based) | Navigation between screens |
| Supabase | User authentication + profile storage |
| i18n (react-i18next) | UI string localisation |
| Axios | HTTP client for backend API |

### 7.2 App Structure

```
Finterest/
├── app/
│   ├── (tabs)/          — Main tab screens
│   └── modal.tsx        — Explanation/detail modal
├── src/
│   ├── screens/         — Chat, News, Profile screens
│   ├── components/      — Reusable UI components
│   ├── services/
│   │   └── apiService.ts — All backend API calls
│   ├── context/         — App-wide state (auth, language)
│   └── locales/         — Translation strings for 7 languages
├── constants/           — Colors, theme tokens
└── hooks/               — Custom React hooks
```

### 7.3 Key Screens

**Chat Screen**:
- Text input with language auto-detection indicator
- Sends query with `lang` hint to `/chat` endpoint
- Renders answer in the user's language with confidence score
- Collapsible "Reasoning Steps" panel showing the KG traversal path
- Handles `answer_lang` — if answer comes back in a different language, shows it correctly

**News Feed Screen**:
- Pulls live Indian financial news from `/financial-news`
- Shows article title, summary, source, and link
- Auto-refreshes every 5 minutes (matching backend cache TTL)
- Falls back gracefully to stub articles when offline

**Language Picker**:
- Allows user to select from 7 supported languages
- Sets the `lang` field on all subsequent API requests
- UI labels switch language via i18n

### 7.4 API Service (`apiService.ts`)

All backend calls are centralised in one typed service file:

```typescript
export type SupportedLangCode = 'en' | 'hi' | 'mr' | 'ta' | 'te' | 'kn' | 'ml';

export const LANG_NAMES: Record<SupportedLangCode, string> = {
  en: 'English', hi: 'Hindi', mr: 'Marathi',
  ta: 'Tamil',  te: 'Telugu', kn: 'Kannada', ml: 'Malayalam',
};

// Main chat call
sendChatMessage(payload: ChatRequest): Promise<ChatResponse>

// Standalone translation
translateText(payload: TranslateRequest): Promise<string>

// Entity search
searchEntities(query: string): Promise<EntitySearchResult[]>

// News feed
getFinancialNews(): Promise<NewsArticle[]>
```

---

## 8. APIs and Endpoints

Base URL: `http://<server>:8000`

### `POST /chat`

Main QA endpoint.

**Request**:
```json
{
  "session_id": "uuid",
  "message": "ரிலையன்ஸ் வருவாய் என்ன?",
  "lang": "ta",
  "debug": false
}
```

**Response**:
```json
{
  "answer": "ரிலையன்ஸ் இண்டஸ்ட்ரீஸ் FY2024ல் ₹9.7 லட்சம் கோடி வருவாய்...",
  "answer_lang": "ta",
  "entities": [{"id": "RELIANCE_abc1", "label": "Company", "name": "Reliance Industries", "score": 0.97}],
  "reasoning_paths": [{"path_id": "p0", "hops": 2, "triples": [...], "score": 0.84}],
  "explanation_steps": ["Found Reliance Industries in KG", "Retrieved FY2024 revenue triple", ...],
  "citations": [{"triple_index": 0, "source_uri": "economictimes.com/..."}],
  "confidence": 0.84,
  "warnings": []
}
```

### `POST /translate`

Standalone translation endpoint.

**Request**:
```json
{"text": "What is Reliance revenue?", "source_lang": "en", "target_lang": "hi"}
```

**Response**:
```json
{"translated": "रिलायंस की आय क्या है?", "source_lang": "en", "target_lang": "hi"}
```

### `GET /resolve`

Entity search against KG.

### `GET /financial-news`

Returns 15 live Indian financial news articles (5-minute cache).

### `GET /health`

```json
{"status": "ok", "service": "FinTraceQA", "kg": "neo4j", "neo4j_connected": true}
```

---

## 9. Evaluation and Results

### 9.1 Multi-Hop QA

The system was manually evaluated on a set of 50 questions across all 4 question types:

| Question Type | Accuracy (correct entity chain) | Fluency |
|---|---|---|
| Single-hop | ~85% | High (DeepSeek synthesis) |
| Multi-hop chain | ~72% | High |
| Multi-hop intersection | ~65% | High |
| Temporal | ~70% | High |

The main source of errors is **entity linking** — when a company name in the query does not fuzzy-match well to a KG node (e.g. abbreviations, alternate spellings).

### 9.2 Translation Model (BLEU on general domain)

The model was trained on general Samanantar data, not financial domain:

| Direction | Quality (qualitative) |
|---|---|
| EN → HI | Captures topic, some dropped words |
| EN → TA | Correct script, occasional UNK tokens |
| HI → EN | Grammatically approximate |
| TA → EN | Topic preserved, some paraphrasing |

**Note**: For production accuracy on the KG search step, DeepSeek is used as a fallback when local model output is insufficient. The local model demonstrates the feasibility of a fully offline multilingual translation system.

### 9.3 Language Detection Accuracy

Unicode block counting achieves **100% accuracy** on script-typed input (Tamil, Telugu, Kannada, Malayalam). The only edge case — Hindi vs Marathi — is resolved with `langdetect`.

---

## 10. Limitations and Future Work

### Current Limitations

| Area | Limitation |
|---|---|
| KG coverage | ~3,300 entities covers NIFTY-50 well but misses mid/small-cap stocks |
| Translation model | Undertrained on financial domain — general Samanantar corpus only |
| Entity linking | Abbreviations and alternate company names cause missed links |
| Temporal resolution | 4-time model not always populated — depends on source data quality |
| Inference speed | 176M param model on CPU is ~2–5 seconds per translation |

### Future Work

1. **Financial domain fine-tuning**: Fine-tune the translation model on parallel financial news data (EN↔Indic financial corpora) to improve translation accuracy on domain vocabulary.

2. **KG expansion**: Add mid-cap and small-cap companies, mutual funds, and commodity markets.

3. **Beam search by default**: Switch the answer-translation step to beam search for better fluency.

4. **Vector retrieval**: Augment graph traversal with dense vector search (FAISS/Pinecone) for entities not directly connected in the KG.

5. **Streaming responses**: Stream the DeepSeek answer token-by-token to the frontend for a better UX on slow connections.

6. **Evaluation dataset**: Build a formal annotated evaluation set of 500+ financial multi-hop questions with gold answers for rigorous BLEU/EM/F1 evaluation.

---

## 11. References

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
2. Ramesh, G. et al. (2022). *Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages*. TACL.
3. iText2KG: [https://github.com/auvalab/itext2kg](https://github.com/auvalab/itext2kg)
4. ATOM: Atomic Fact Extraction for Knowledge Graph Construction.
5. DeepSeek-V3: [https://api.deepseek.com](https://api.deepseek.com) — `deepseek-chat` model, OpenAI-compatible API.
6. Neo4j Aura: [https://neo4j.com/cloud/platform/aura-graph-database/](https://neo4j.com/cloud/platform/aura-graph-database/)
7. Kunchukuttan, A. (2020). *The IndicNLP Library*. [https://github.com/anoopkunchukuttan/indic_nlp_library](https://github.com/anoopkunchukuttan/indic_nlp_library)
8. IN-FINews Dataset (Zenodo): Indian Financial News Corpus — 3,348 articles.
9. NIFTY-50 Stock Market Data 2000–2021 (Kaggle).
10. Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units* (BPE). ACL 2016.
11. Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates* (Unigram). ACL 2018.

---

*Report generated: March 2026*  
*GitHub: [RoshanGit09/NLP-MultiHop-Q-A](https://github.com/RoshanGit09/NLP-MultiHop-Q-A)*
