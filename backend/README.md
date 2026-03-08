# FinTraceQA — Backend

Multi-hop financial QA chatbot backend over an Indian financial events Knowledge Graph.

---

## Architecture

```
backend/
├── app/
│   ├── main.py                  ← FastAPI app + CORS
│   └── routers/
│       └── chat.py              ← POST /chat, POST /resolve, GET /health
├── core/
│   ├── lang_router.py           ← Language detection & normalisation (EN/HI/mixed)
│   ├── decomposer.py            ← Multi-hop question decomposer / planner
│   ├── entity_linker.py         ← Mention → KG node ID (exact, alias, fuzzy)
│   ├── relation_linker.py       ← Predicate → KG relation type
│   ├── traversal.py             ← Multi-hop KG traversal engine + scorer
│   ├── synthesizer.py           ← KG path → natural-language answer
│   ├── memory.py                ← Session memory + coreference
│   ├── kg_adapter/
│   │   ├── base.py              ← Abstract KGAdapterBase interface
│   │   ├── neo4j_adapter.py     ← Neo4j / Cypher implementation
│   │   └── mock_adapter.py      ← In-memory toy KG for tests
│   └── explanations/
│       ├── templates.py         ← Template-based faithful explanation generator
│       └── faithfulness_verifier.py  ← Drops sentences not grounded in path
├── configs/
│   └── weights.yaml             ← Scoring weights (configurable)
├── tests/
│   ├── test_entity_linker.py
│   ├── test_relation_linker.py
│   ├── test_traversal.py
│   └── test_faithfulness.py
├── scripts/
│   └── eval_benchmark.py        ← Offline benchmark harness
└── .env.example
```

---

## Setup

### 1. Use the project root `.venv`

```powershell
# From NLP-MultiHop-Q-A\
.\.venv\Scripts\activate
pip install fastapi "uvicorn[standard]" langdetect python-multipart PyYAML
```

### 2. Configure environment

```powershell
Copy-Item backend\.env.example backend\.env
# Edit backend\.env with your Neo4j credentials
```

### 3. Run the server

```powershell
# With Neo4j (default)
$env:PYTHONPATH = "."
.\.venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# With Mock KG (no Neo4j needed)
$env:USE_MOCK_KG = "true"
.\.venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "service": "FinTraceQA" }
```

### `POST /chat`

**Request**
```json
{
  "session_id": "user123",
  "message": "Who is the CEO of Infosys?",
  "lang": "auto",
  "debug": false
}
```

**Response**
```json
{
  "answer": "Infosys is the CEO of Salil Parekh (as of 2018-01-02).",
  "answer_lang": "en",
  "entities": [{ "id": "infosys", "label": "Company", "name": "Infosys", "score": 1.0 }],
  "reasoning_paths": [{
    "path_id": "p0", "hops": 1,
    "triples": [{ "subj": "Infosys", "rel": "CEO_OF", "obj": "Salil Parekh", "time": "2018-01-02" }],
    "score": 0.8
  }],
  "explanation_steps": ["Step 1: **Infosys** serves as CEO of **Salil Parekh** (recorded: 2018-01-02)."],
  "citations": [{ "triple_index": 0, "source_uri": null }],
  "confidence": 0.8,
  "warnings": []
}
```

### `POST /resolve`

**Request**
```json
{ "mention": "Infosys", "lang": "en" }
```

**Response**
```json
{ "candidates": [{ "id": "infosys", "label": "Company", "name": "Infosys", "score": 1.0 }] }
```

---

## Example cURL Requests

```bash
# Health check
curl http://localhost:8000/health

# Single-hop question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"Who is the CEO of Infosys?","lang":"en"}'

# Multi-hop question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"Which companies benefited from the RBI rate cut in 2024?","lang":"en"}'

# Hindi query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"Infosys के CEO कौन हैं?","lang":"hi"}'

# Entity disambiguation
curl -X POST http://localhost:8000/resolve \
  -H "Content-Type: application/json" \
  -d '{"mention":"RBI","lang":"en"}'

# Debug mode (includes decomposition + execution stats)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"What changed after RBI repo rate cut in April 2024?","lang":"en","debug":true}'
```

---

## Running Tests

```powershell
# From NLP-MultiHop-Q-A\
.\.venv\Scripts\python.exe -m pytest backend/tests/ -v
```

---

## Offline Benchmark

```powershell
# Create a sample FinEventQA.jsonl first
# Format: {"question": "...", "gold_answer": "...", "gold_path": [...]}

.\.venv\Scripts\python.exe backend/scripts/eval_benchmark.py \
    --input data/FinEventQA.jsonl \
    --use-mock   # omit this flag to use Neo4j
```

Output:
```
=== FinTraceQA Benchmark Results ===
Total questions    : 100
Exact Match (EM)   : 42.00%
Path Match         : 68.00%
Faithfulness Rate  : 94.00%
Latency P50        : 85.3 ms
Latency P95        : 210.7 ms
```

---

## Scoring Weights

Edit `configs/weights.yaml` to tune path scoring:

```yaml
scoring:
  entity_link_weight: 0.35
  relation_link_weight: 0.25
  temporal_match_weight: 0.20
  path_length_penalty_weight: 0.20
path_length_penalty_per_hop: 0.08
max_hops: 4
top_k_paths: 3
confidence_threshold: 0.60
```

---

## Environment Variables

| Variable        | Default               | Description                          |
|-----------------|-----------------------|--------------------------------------|
| `NEO4J_URI`     | `bolt://localhost:7687` | Neo4j connection URI               |
| `NEO4J_USERNAME`| `neo4j`               | Neo4j username                       |
| `NEO4J_PASSWORD`| *(required)*          | Neo4j password                       |
| `USE_MOCK_KG`   | `false`               | Use in-memory mock KG (no Neo4j)     |
| `PORT`          | `8000`                | Server port                          |
