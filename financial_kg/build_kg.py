"""
build_kg.py — Production Knowledge Graph builder
=================================================
Processes ALL available data sources in the correct order and builds
one unified, deduplicated Knowledge Graph, then uploads it to Neo4j.

Data sources (processed in order)
-----------------------------------
1. zenodo/IN-FINews Dataset.json  — 3 348 Indian financial news articles
2. NIFTY-50 stock CSVs            — 49 stock price series (2000-2021)
3. stock_metadata.csv             — company ↔ industry ↔ exchange facts
4. India macro xlsx               — yearly GDP / SENSEX / inflation
5. Asian stock indices xlsx       — multi-market index correlations
6. Stock price history CSVs       — 6 additional ticker histories

Usage
-----
    # full build (all sources)
    python -m financial_kg.build_kg

    # limit articles processed (fast test)
    python -m financial_kg.build_kg --max-articles 50

    # skip Neo4j upload (local KG only)
    python -m financial_kg.build_kg --no-neo4j

    # resume from a previous checkpoint
    python -m financial_kg.build_kg --resume output/kg_checkpoint.json

    # only specific sources
    python -m financial_kg.build_kg --sources news nifty macro
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── package root on sys.path so we can run as `python financial_kg/build_kg.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from financial_kg.kg_builder.incremental_kg_builder import (
    IncrementalKGBuilder,
    _IText2KGAdapter,
)
from financial_kg.utils.config import get_config
from financial_kg.utils.logging_config import get_logger

logger = get_logger(__name__)
cfg    = get_config()

# ── paths ------------------------------------------------------------------──
_HERE       = Path(__file__).resolve().parent
_DATA       = _HERE / "data"
_OUTPUT     = _HERE / "output"
_OUTPUT.mkdir(parents=True, exist_ok=True)

_NIFTY_DIR  = _DATA / "kaggle" / "NIFTY-50 Stock Market Data (2000 - 2021)"
_META_CSV   = _NIFTY_DIR / "stock_metadata.csv"
_ZENODO_JSON= _DATA / "zenodo" / "IN-FINews  Dataset.json"
_INDIA_XLSX = (_DATA / "MEndely"
               / "India Stock Market Dataset (1980\u20132024)"
               / "India Stock Market Dataset (1980\u20132024)"
               / "India_Stock_Market_Data.xlsx")
_ASIAN_XLSX = (_DATA / "MEndely"
               / "Asian stock market data" / "gd9jkdbnby-1"
               / "asian stock market data.xlsx")
_HIST_DIR   = (_DATA / "MEndely"
               / "Stock_Price_History_Datset"
               / "Stock_Price_History_Datset"
               / "New folder"
               / "Stock_Price_History_Datset")

# INFRATEL.csv is known-empty
_NIFTY_SKIP = {"INFRATEL", "NIFTY50_all", "stock_metadata"}


# ===========================================================================
# Structured-data -> text conversion helpers
# These turn rows/records into natural-language sentences so that
# IncrementalKGBuilder (which calls Gemini) can extract entities from them.
# ===========================================================================

def _stock_row_to_text(ticker: str, company: str, row: pd.Series) -> str:
    """Convert one OHLCV row into a sentence Gemini can extract facts from."""
    close_col = "Close" if "Close" in row.index else "Close Price"
    close = row.get(close_col, "N/A")
    vol   = row.get("Volume", "N/A")
    date  = row.get("Date", "unknown date")
    return (
        f"{company} (NSE: {ticker}) traded at a closing price of ₹{close} "
        f"with volume {vol} on {date}."
    )


def _meta_row_to_text(row: pd.Series) -> str:
    """Convert stock_metadata row -> sentence."""
    return (
        f"{row['Company Name']} (Symbol: {row['Symbol']}, "
        f"ISIN: {row.get('ISIN Code', 'N/A')}) is listed on NSE/BSE "
        f"in the {row['Industry']} industry."
    )


def _macro_row_to_text(row: pd.Series) -> str:
    """Convert India macro xlsx row -> sentence.
    Dates are written as full ISO strings (YYYY-01-01) so Pydantic TemporalAttributes parses them."""
    year = int(row['Year'])
    return (
        f"As of {year}-01-01, India's GDP growth was {row['GDP Growth (%)']:.2f}%, "
        f"inflation was {row['Inflation (%)']:.2f}%, "
        f"the SENSEX closed at {row['SENSEX']:.2f}, "
        f"and the USD/INR exchange rate was {row['Exch Rate (INR/USD)']:.2f}."
    )


def _asian_row_to_text(idx: int, row: pd.Series) -> str:
    """Convert Asian market indices row -> sentence."""
    return (
        f"At observation {idx}: SENSEX={row.get('sensex', 'N/A')}, "
        f"Straits Times Index={row.get('strait', 'N/A')}, "
        f"KOSPI={row.get('kospi', 'N/A')}, "
        f"Shanghai Composite={row.get('shanghai', 'N/A')}, "
        f"JASDAQ={row.get('jasdaq', 'N/A')}. "
        f"These Asian indices show co-movement across markets."
    )


def _hist_row_to_text(ticker: str, row: pd.Series) -> str:
    """Convert price-history CSV row -> sentence."""
    close = row.get("Close Price", row.get("Close", "N/A"))
    date  = row.get("Date", "unknown date")
    return (
        f"{ticker} had a closing price of ₹{close} on {date} "
        f"according to historical records."
    )


# ===========================================================================
# Batch helpers — aggregate multiple rows into one Gemini call
# ===========================================================================

def _chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ===========================================================================
# Source processors
# Each returns nothing — they call builder.build_incremental() in-place.
# ===========================================================================

def process_news(
    builder: IncrementalKGBuilder,
    max_articles: Optional[int],
    batch_size: int,
    checkpoint_every: int = 0,
    output_path: str = "",
    skip_articles: int = 0,
) -> int:
    """Process zenodo IN-FINews articles."""
    if not _ZENODO_JSON.exists():
        logger.warning(f"zenodo JSON not found: {_ZENODO_JSON}")
        return 0

    logger.info("--- SOURCE 1: zenodo IN-FINews articles ---")
    with open(_ZENODO_JSON, encoding="utf-8-sig") as f:
        articles = json.load(f)

    if skip_articles:
        logger.info(f"  Skipping first {skip_articles} articles (already in checkpoint)")
        articles = articles[skip_articles:]

    if max_articles:
        articles = articles[:max_articles]

    total = 0
    last_checkpoint = 0
    for i, batch in enumerate(_chunk(articles, batch_size)):
        sections = []
        for art in batch:
            content = (art.get("Content") or "").strip()
            title   = (art.get("Title")   or "").strip()
            date    = art.get("Date", "")
            if not content:
                continue
            # Prepend title + date for better entity context
            sections.append(f"[{date}] {title}\n{content[:1200]}")

        if not sections:
            continue

        builder.build_incremental(sections)
        total += len(sections)
        stats = builder.stats()
        logger.info(
            f"  news batch {i+1}/{-(-len(articles)//batch_size)}  "
            f"| processed={total}  "
            f"| KG: {stats['nodes']} nodes, {stats['edges']} edges"
        )

        # Mid-stream checkpoint — use absolute article count (skip_articles + total)
        # so checkpoint_800.json always means "first 800 articles processed"
        # regardless of how many times the build was resumed.
        if checkpoint_every and output_path and (total - last_checkpoint) >= checkpoint_every:
            _save_checkpoint(builder, output_path, skip_articles + total)
            last_checkpoint = total

    logger.info(f"  OK news done — {total} articles processed")
    return total


def process_nifty_metadata(builder: IncrementalKGBuilder) -> int:
    """Process stock_metadata.csv — company / industry / exchange facts."""
    if not _META_CSV.exists():
        logger.warning(f"metadata CSV not found: {_META_CSV}")
        return 0

    logger.info("--- SOURCE 2: NIFTY-50 stock metadata ---")
    meta = pd.read_csv(_META_CSV)
    sections = [_meta_row_to_text(row) for _, row in meta.iterrows()]

    # All 50 rows fit in one call
    builder.build_incremental(sections)
    stats = builder.stats()
    logger.info(
        f"  OK metadata done — {len(sections)} companies  "
        f"| KG: {stats['nodes']} nodes, {stats['edges']} edges"
    )
    return len(sections)


def process_nifty_prices(
    builder: IncrementalKGBuilder,
    sample_rows: int,
    batch_size: int,
    checkpoint_every: int = 0,
    output_path: str = "",
) -> int:
    """
    Process NIFTY-50 price CSVs — one ticker at a time with progress logging.
    We take `sample_rows` evenly-spaced rows per ticker (not every day —
    that would be 5000+ Gemini calls per ticker).
    Saves a checkpoint after every ticker when checkpoint_every > 0.
    """
    logger.info(f"--- SOURCE 3: NIFTY-50 price CSVs (sample={sample_rows}/ticker) ---")

    meta_map: Dict[str, str] = {}
    if _META_CSV.exists():
        meta = pd.read_csv(_META_CSV)
        meta_map = dict(zip(meta["Symbol"], meta["Company Name"]))

    csv_files = sorted(
        f for f in _NIFTY_DIR.glob("*.csv")
        if f.stem not in _NIFTY_SKIP
    )
    total_tickers = len(csv_files)

    total = 0
    for ticker_idx, csv_path in enumerate(csv_files, 1):
        ticker  = csv_path.stem
        company = meta_map.get(ticker, ticker)
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                logger.warning(f"  [{ticker_idx}/{total_tickers}] {ticker}: empty CSV, skipping")
                continue
            # Standardise close column
            if "Close" not in df.columns and "Close Price" in df.columns:
                df["Close"] = df["Close Price"]
            # Evenly sample rows
            step = max(1, len(df) // sample_rows)
            sampled = df.iloc[::step].head(sample_rows)

            sections = [
                _stock_row_to_text(ticker, company, row)
                for _, row in sampled.iterrows()
            ]
            for batch in _chunk(sections, batch_size):
                builder.build_incremental(batch)
                total += len(batch)

            stats = builder.stats()
            logger.info(
                f"  [{ticker_idx:2d}/{total_tickers}] {ticker:<12} ({company[:30]:<30}) "
                f"| rows={len(sampled)}  KG: {stats['nodes']} nodes, {stats['edges']} edges"
            )

            # Checkpoint after each ticker when enabled
            if checkpoint_every > 0 and output_path:
                _save_checkpoint(builder, output_path, f"nifty_{ticker_idx:02d}_{ticker}")

        except Exception as exc:
            logger.warning(f"  [{ticker_idx}/{total_tickers}] Skipped {ticker}: {exc}")

    stats = builder.stats()
    logger.info(
        f"  OK NIFTY prices done — {total} sentences across {total_tickers} tickers "
        f"| KG: {stats['nodes']} nodes, {stats['edges']} edges"
    )
    return total


def process_price_history(
    builder: IncrementalKGBuilder,
    sample_rows: int,
    batch_size: int,
    checkpoint_every: int = 0,
    output_path: str = "",
) -> int:
    """Process Mendeley stock price history CSVs with per-file progress."""
    if not _HIST_DIR.exists():
        logger.warning(f"price history dir not found: {_HIST_DIR}")
        return 0

    logger.info("--- SOURCE 4: Mendeley price history CSVs ---")
    hist_files = sorted(_HIST_DIR.glob("*.csv"))
    total_files = len(hist_files)
    total = 0
    for file_idx, csv_path in enumerate(hist_files, 1):
        ticker = csv_path.stem
        try:
            df   = pd.read_csv(csv_path)
            step = max(1, len(df) // sample_rows)
            sampled = df.iloc[::step].head(sample_rows)
            sections = [
                _hist_row_to_text(ticker, row)
                for _, row in sampled.iterrows()
            ]
            for batch in _chunk(sections, batch_size):
                builder.build_incremental(batch)
                total += len(batch)
            stats = builder.stats()
            logger.info(
                f"  [{file_idx:2d}/{total_files}] {ticker:<15} | rows={len(sampled)} "
                f"KG: {stats['nodes']} nodes, {stats['edges']} edges"
            )
            if checkpoint_every > 0 and output_path:
                _save_checkpoint(builder, output_path, f"hist_{file_idx:02d}_{ticker}")
        except Exception as exc:
            logger.warning(f"  Skipped {ticker}: {exc}")

    stats = builder.stats()
    logger.info(
        f"  OK price history done — {total} sentences  "
        f"| KG: {stats['nodes']} nodes, {stats['edges']} edges"
    )
    return total


def process_india_macro(
    builder: IncrementalKGBuilder,
    batch_size: int,
    checkpoint_every: int = 0,
    output_path: str = "",
) -> int:
    """Process India macro xlsx — GDP / SENSEX / inflation per year."""
    if not _INDIA_XLSX.exists():
        logger.warning(f"India macro xlsx not found: {_INDIA_XLSX}")
        return 0

    logger.info("--- SOURCE 5: India macro-economic data ---")
    df = pd.read_excel(_INDIA_XLSX)
    df = df.dropna(subset=["Year", "SENSEX"])

    sections = [_macro_row_to_text(row) for _, row in df.iterrows()]
    for batch in _chunk(sections, batch_size):
        builder.build_incremental(batch)

    stats = builder.stats()
    logger.info(
        f"  OK macro done — {len(sections)} years  "
        f"| KG: {stats['nodes']} nodes, {stats['edges']} edges"
    )
    if checkpoint_every > 0 and output_path:
        _save_checkpoint(builder, output_path, "post_macro")
    return len(sections)


def process_asian_indices(
    builder: IncrementalKGBuilder,
    batch_size: int,
    checkpoint_every: int = 0,
    output_path: str = "",
) -> int:
    """Process Asian stock market indices xlsx."""
    if not _ASIAN_XLSX.exists():
        logger.warning(f"Asian indices xlsx not found: {_ASIAN_XLSX}")
        return 0

    logger.info("--- SOURCE 6: Asian stock market indices ---")
    df = pd.read_excel(_ASIAN_XLSX).dropna()

    # Sample every 10th row (240 rows -> 24 sentences)
    sampled  = df.iloc[::10]
    sections = [_asian_row_to_text(i, row) for i, (_, row) in enumerate(sampled.iterrows())]

    for batch in _chunk(sections, batch_size):
        builder.build_incremental(batch)

    stats = builder.stats()
    logger.info(
        f"  OK Asian indices done — {len(sections)} observations  "
        f"| KG: {stats['nodes']} nodes, {stats['edges']} edges"
    )
    if checkpoint_every > 0 and output_path:
        _save_checkpoint(builder, output_path, "post_asian")
    return len(sections)


# ===========================================================================
# Neo4j upload helper
# ===========================================================================

def upload_to_neo4j(builder: IncrementalKGBuilder, clear: bool = False) -> None:
    """
    Convert the NetworkX DiGraph stored in ``builder.global_kg`` into the
    Pydantic KnowledgeGraph / Entity / Relationship model that Neo4jStorage
    expects, then upload.

    Node attribute mapping (DiGraph -> Pydantic Entity):
      attrs['name']       -> Entity.name
      attrs['label']      -> Entity.type  (Gemini extraction uses 'label' for type)
      attrs['confidence'] -> Entity.confidence
    """
    from financial_kg.storage.neo4j_storage import Neo4jStorage
    from financial_kg.models.knowledge_graph import KnowledgeGraph
    from financial_kg.models.entity import Entity, create_entity
    from financial_kg.models.relationship import Relationship, RelationshipProperties

    # Recognised entity types that map to specific sub-classes
    _KNOWN_TYPES = {"Company", "Person", "Sector", "Event", "Policy", "Indicator"}

    logger.info("--- Uploading KG to Neo4j ---")
    kg  = KnowledgeGraph()
    g   = builder.global_kg
    node_map: Dict[str, Entity] = {}   # node_id -> Entity

    # ── nodes ------------------------------------------------------------─
    for node_id, attrs in g.nodes(data=True):
        raw_label  = (attrs.get("label") or "").strip()
        # Map to a recognised type, fall back to "Entity"
        ent_type   = raw_label if raw_label in _KNOWN_TYPES else "Indicator"
        name       = (attrs.get("name") or node_id).strip()
        confidence = float(attrs.get("confidence") or 0.8)
        # Collect any extra attrs as properties
        extra_props = {
            k: v for k, v in attrs.items()
            if k not in ("id", "name", "label", "confidence")
        }
        try:
            ent = create_entity(
                entity_type=ent_type,
                id=node_id,
                name=name,
                confidence=confidence,
                properties=extra_props,
            )
            kg.add_entity(ent)
            node_map[node_id] = ent
        except Exception as exc:
            logger.warning(f"Skipping node '{node_id}': {exc}")

    # ── edges ------------------------------------------------------------─
    for src, dst, edata in g.edges(data=True):
        if src not in node_map or dst not in node_map:
            continue
        rel_name = (edata.get("name") or "RELATED_TO").strip()
        conf     = float(edata.get("confidence") or 0.8)
        sent     = edata.get("sentiment")
        src_text = str(edata.get("source_text") or "")
        try:
            props = RelationshipProperties(
                confidence=conf,
                sentiment=float(sent) if sent is not None else None,
                source_text=src_text if src_text else None,
            )
            rel = Relationship(
                id=f"{src}__{rel_name}__{dst}",
                name=rel_name,
                subject=node_map[src],
                predicate=rel_name,
                object=node_map[dst],
                properties=props,
            )
            kg.add_relationship(rel)
        except Exception as exc:
            logger.warning(f"Skipping edge {src}->{dst}: {exc}")

    with Neo4jStorage() as storage:
        storage.upload_kg(kg, clear_existing=clear)

    logger.info(
        f"  OK Neo4j upload complete — "
        f"{len(kg.entities)} nodes, {len(kg.relationships)} edges"
    )


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Financial Knowledge Graph from all data sources."
    )
    p.add_argument(
        "--max-articles", type=int, default=None,
        help="Max zenodo news articles to process (default: all 3 348)",
    )
    p.add_argument(
        "--nifty-sample", type=int, default=20,
        help="Evenly-spaced rows to sample per NIFTY-50 ticker (default: 20)",
    )
    p.add_argument(
        "--hist-sample", type=int, default=20,
        help="Evenly-spaced rows to sample per price-history ticker (default: 20)",
    )
    p.add_argument(
        "--batch-size", type=int, default=int(cfg.processing.batch_size),
        help="Sections per IncrementalKGBuilder call (default from .env)",
    )
    p.add_argument(
        "--eps", type=float, default=0.85,
        help=(
            "String-similarity threshold for entity deduplication (default: 0.85). "
            "Values below 0.85 risk over-merging — DO NOT lower below 0.85."
        ),
    )
    p.add_argument(
        "--sources", nargs="+",
        choices=["news", "metadata", "nifty", "history", "macro", "asian"],
        default=["news", "metadata", "nifty", "history", "macro", "asian"],
        help="Which sources to process (default: all)",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        metavar="CHECKPOINT",
        help="Path to a previously saved KG JSON checkpoint to resume from",
    )
    p.add_argument(
        "--output", type=str,
        default=str(_OUTPUT / "financial_kg.json"),
        help="Output path for the saved KG JSON",
    )
    p.add_argument(
        "--no-neo4j", action="store_true",
        help="Skip Neo4j upload (save locally only)",
    )
    p.add_argument(
        "--clear-neo4j", action="store_true",
        help="Clear Neo4j database before uploading",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=200,
        help="Save a news checkpoint every N articles (default: 200; 0 = disabled)",
    )
    p.add_argument(
        "--skip-articles", type=int, default=0,
        help="Skip the first N news articles (use when resuming from a checkpoint)",
    )
    return p.parse_args()


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    args   = parse_args()
    t_start = time.perf_counter()

    logger.info("==========================================")
    logger.info("  Financial KG — Production Build")
    logger.info(f"  sources        : {args.sources}")
    logger.info(f"  max-articles   : {args.max_articles or 'all'}")
    logger.info(f"  nifty-sample   : {args.nifty_sample} rows/ticker")
    logger.info(f"  batch-size     : {args.batch_size}")
    logger.info(f"  eps (str-dedup): {args.eps}  [SAFE minimum: 0.85]")
    logger.info(f"  checkpoint-every: {args.checkpoint_every} articles (0=off)")
    logger.info(f"  output         : {args.output}")
    logger.info(f"  neo4j upload   : {'NO' if args.no_neo4j else 'YES'}")
    logger.info("==========================================")

    # Guard against accidental eps < 0.85 which caused the DBSCAN disaster
    if args.eps < 0.85:
        logger.error(
            f"FATAL: --eps={args.eps} is below the safe minimum of 0.85. "
            "Values this low will cause catastrophic entity over-merging "
            "(all companies collapse into one node). Aborting. "
            "Use --eps 0.85 or higher."
        )
        sys.exit(1)

    # ── build or resume builder ------------------------------------------─
    adapter = _IText2KGAdapter()

    if args.resume and Path(args.resume).exists():
        logger.info(f"Resuming from checkpoint: {args.resume}")
        builder = IncrementalKGBuilder.load(
            args.resume, itext2kg=adapter, eps=args.eps
        )
    else:
        builder = IncrementalKGBuilder(itext2kg=adapter, eps=args.eps)

    sources = set(args.sources)
    processed = 0

    # ── source 1: news ---------------------------------------------------─
    if "news" in sources:
        processed += process_news(
            builder, args.max_articles, args.batch_size,
            checkpoint_every=args.checkpoint_every,
            output_path=args.output,
            skip_articles=args.skip_articles,
        )

    # ── source 2: metadata ------------------------------------------------
    if "metadata" in sources:
        processed += process_nifty_metadata(builder)

    # ── source 3: NIFTY-50 prices ---------------------------------------──
    if "nifty" in sources:
        processed += process_nifty_prices(
            builder, args.nifty_sample, args.batch_size,
            checkpoint_every=args.checkpoint_every,
            output_path=args.output,
        )

    # ── source 4: price history ------------------------------------------─
    if "history" in sources:
        processed += process_price_history(
            builder, args.hist_sample, args.batch_size,
            checkpoint_every=args.checkpoint_every,
            output_path=args.output,
        )

    # ── source 5: India macro ---------------------------------------------
    if "macro" in sources:
        processed += process_india_macro(
            builder, args.batch_size,
            checkpoint_every=args.checkpoint_every,
            output_path=args.output,
        )

    # ── source 6: Asian indices ------------------------------------------─
    if "asian" in sources:
        processed += process_asian_indices(
            builder, args.batch_size,
            checkpoint_every=args.checkpoint_every,
            output_path=args.output,
        )

    # ── final save ------------------------------------------------------──
    builder.save(args.output)
    stats = builder.stats()
    elapsed = time.perf_counter() - t_start

    logger.info("==========================================")
    logger.info("  BUILD COMPLETE")
    logger.info(f"  nodes   : {stats['nodes']}")
    logger.info(f"  edges   : {stats['edges']}")
    logger.info(f"  labels  : {sorted(stats['labels'])}")
    logger.info(f"  total inputs processed : {processed}")
    logger.info(f"  elapsed : {elapsed/60:.1f} min")
    logger.info(f"  saved   : {args.output}")
    logger.info("==========================================")

    # ── Neo4j upload ------------------------------------------------------
    if not args.no_neo4j:
        upload_to_neo4j(builder, clear=args.clear_neo4j)

    # ── print top entities ------------------------------------------------
    _print_top_entities(builder)


def _save_checkpoint(
    builder: IncrementalKGBuilder,
    base_path: str,
    label,          # int (article count) or str (e.g. "nifty_05_TCS")
) -> None:
    p = Path(base_path)
    cp = p.parent / f"{p.stem}_checkpoint_{label}{p.suffix}"
    builder.save(str(cp))
    logger.info(f"  [OK] checkpoint saved -> {cp.name}")


def _print_top_entities(builder: IncrementalKGBuilder, top_n: int = 20) -> None:
    """Print the most-connected nodes as a quick sanity check."""
    g = builder.global_kg
    if g.number_of_nodes() == 0:
        return
    by_degree = sorted(
        g.nodes(data=True),
        key=lambda t: g.degree(t[0]),
        reverse=True,
    )[:top_n]
    logger.info(f"\n  Top {top_n} entities by degree:")
    for node_id, attrs in by_degree:
        deg   = g.degree(node_id)
        name  = attrs.get("name", node_id)
        label = attrs.get("label", "?")
        conf  = attrs.get("confidence", 0.0)
        logger.info(f"    [{deg:3d}]  {name:<40}  {label:<15}  conf={conf:.2f}")


if __name__ == "__main__":
    main()
