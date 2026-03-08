"""
main.py — FastAPI entry point for the FinTraceQA chatbot backend.
"""
from __future__ import annotations

import os
import sys
import logging
import httpx
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on the Python path so financial_kg is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load backend .env (NEO4J_URI, NEO4J_PASSWORD, USE_MOCK_KG, etc.)
load_dotenv(dotenv_path=_ROOT / "backend" / ".env")

from backend.app.routers.chat import router as chat_router  # noqa: E402

app = FastAPI(
    title="FinTraceQA — Financial Multi-Hop QA Backend",
    description=(
        "Answers multi-hop questions over an Indian financial events "
        "Knowledge Graph with faithful, step-by-step explanations."
    ),
    version="1.0.0",
)

# CORS — allow the Expo / React Native frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(chat_router, tags=["Chatbot"])


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    from backend.app.routers.chat import _adapter
    kg_live = _adapter._active() is not None
    return {
        "status": "ok",
        "service": "FinTraceQA",
        "kg": "neo4j" if kg_live else "mock",
        "neo4j_connected": kg_live,
    }


# ── Financial news — live via NewsAPI, fallback to stubs ─────────────────

_news_logger = logging.getLogger("news")

_NEWS_CACHE_TTL = 300   # seconds — refresh every 5 minutes
_news_cache: Tuple[float, List[Dict[str, Any]]] = (0.0, [])   # (fetched_at, articles)

_NEWS_STUB = [
    {"id": "1", "title": "Sensex Hits Record High Amid Global Rally",
     "summary": "The BSE Sensex crossed the 80,000 mark for the first time as global equities rallied.",
     "source": "Economic Times", "url": ""},
    {"id": "2", "title": "RBI Keeps Repo Rate Unchanged at 6.5%",
     "summary": "The Reserve Bank of India maintained its benchmark lending rate citing persistent inflation concerns.",
     "source": "Mint", "url": ""},
    {"id": "3", "title": "IT Sector Leads Market Gains on Strong Q4 Earnings",
     "summary": "Top IT companies including TCS and Infosys reported better-than-expected quarterly results.",
     "source": "Business Standard", "url": ""},
    {"id": "4", "title": "Gold Prices Rise as Dollar Weakens",
     "summary": "Gold futures climbed to a three-month high as the US dollar index fell.",
     "source": "Financial Express", "url": ""},
    {"id": "5", "title": "Mutual Fund SIP Inflows Hit All-Time High",
     "summary": "Monthly SIP contributions to Indian mutual funds crossed ₹20,000 crore for the first time.",
     "source": "Moneycontrol", "url": ""},
]


@app.get("/financial-news", tags=["News"])
async def financial_news():
    """
    Returns the latest Indian financial news headlines from NewsAPI.
    Results are cached for 5 minutes to protect the free-tier quota (100 req/day).
    Falls back to stub data if the API key is missing or the call fails.
    """
    import time as _time
    global _news_cache

    # ── Serve from cache if still fresh ──────────────────────────────────
    cached_at, cached_articles = _news_cache
    if cached_articles and (_time.monotonic() - cached_at) < _NEWS_CACHE_TTL:
        _news_logger.debug("NewsAPI: serving from cache (%ds old).",
                           int(_time.monotonic() - cached_at))
        return cached_articles

    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        _news_logger.warning("NEWSAPI_KEY not set — returning stub news.")
        return _stub_news()

    # Targeted Indian financial news query — sources + keywords to avoid global noise
    url = (
        "https://newsapi.org/v2/everything"
        "?q=(Sensex+OR+Nifty+OR+BSE+OR+NSE+OR+RBI+OR+SEBI+OR+%22Indian+stock%22+OR+%22Dalal+Street%22)"
        "+AND+(India+OR+Indian+OR+Mumbai+OR+rupee)"
        "&language=en"
        "&sortBy=publishedAt"
        "&pageSize=20"
        "&domains=economictimes.indiatimes.com,livemint.com,business-standard.com,"
        "financialexpress.com,moneycontrol.com,ndtvprofit.com,thehindu.com,"
        "businesstoday.in,bloombergquint.com,zeebiz.com"
        f"&apiKey={api_key}"
    )

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles") or []
        result = []
        for i, art in enumerate(articles[:15], 1):
            title   = art.get("title")   or ""
            summary = art.get("description") or art.get("content") or ""
            source  = (art.get("source") or {}).get("name") or "NewsAPI"
            pub_at  = art.get("publishedAt") or datetime.now(timezone.utc).isoformat()
            article_url = art.get("url") or ""

            # Skip "[Removed]" articles that NewsAPI sometimes returns
            if "[Removed]" in title or not title.strip():
                continue

            result.append({
                "id":        str(i),
                "title":     title.strip(),
                "summary":   summary.strip(),
                "source":    source,
                "timestamp": pub_at,
                "url":       article_url,
            })

        if not result:
            _news_logger.warning("NewsAPI returned 0 usable articles — using stubs.")
            return _stub_news()

        # ── Store in cache ────────────────────────────────────────────────
        _news_cache = (_time.monotonic(), result)
        _news_logger.info("NewsAPI: fetched %d articles (cached for %ds).",
                          len(result), _NEWS_CACHE_TTL)
        return result

    except Exception as exc:
        _news_logger.warning("NewsAPI call failed (%s) — returning stub news.", exc)
        # Return stale cache if available, else stubs
        if cached_articles:
            _news_logger.info("NewsAPI: returning stale cache as fallback.")
            return cached_articles
        return _stub_news()


def _stub_news() -> List[Dict[str, Any]]:
    """Return the static stub news list with fresh timestamps."""
    now = datetime.now(timezone.utc)
    return [
        {**item, "timestamp": (now - timedelta(hours=i)).isoformat()}
        for i, item in enumerate(_NEWS_STUB)
    ]


# ── Dev runner ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
