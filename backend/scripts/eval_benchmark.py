"""
eval_benchmark.py — Offline benchmark harness for FinTraceQA.

Input:  FinEventQA.jsonl  (one JSON per line)
        Each line: {"question": "...", "gold_answer": "...", "gold_path": [...] (optional)}

Output: Exact Match (EM), Path Match, Faithfulness Rate, P50/P95 latency.

Usage:
    python scripts/eval_benchmark.py --input data/FinEventQA.jsonl [--use-mock]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("USE_MOCK_KG", "false")

from backend.core.lang_router import LanguageRouter
from backend.core.decomposer import QuestionDecomposer
from backend.core.traversal import KGTraversalEngine
from backend.core.synthesizer import AnswerSynthesizer
from backend.core.explanations import TemplateExplainer, FaithfulnessVerifier
from backend.core.memory import SessionMemory


def _build_adapter(use_mock: bool):
    if use_mock:
        from backend.core.kg_adapter.mock_adapter import MockAdapter
        return MockAdapter()
    try:
        from backend.core.kg_adapter.neo4j_adapter import Neo4jAdapter
        return Neo4jAdapter()
    except Exception as e:
        print(f"[warn] Neo4j unavailable ({e}), using MockAdapter")
        from backend.core.kg_adapter.mock_adapter import MockAdapter
        return MockAdapter()


def exact_match(pred: str, gold: str) -> bool:
    def norm(s):
        return " ".join(s.lower().strip().split())
    return norm(pred) == norm(gold)


def path_match(pred_triples: List[Dict], gold_path: List[Dict]) -> bool:
    """Check if any predicted triple matches the gold path triple."""
    if not gold_path:
        return True  # no gold path provided, skip
    for gt in gold_path:
        gold_rel = gt.get("rel", "").upper()
        gold_obj = gt.get("obj", "").lower()
        for pt in pred_triples:
            if (pt.get("rel", "").upper() == gold_rel and
                    pt.get("obj", "").lower() == gold_obj):
                return True
    return False


def run_benchmark(input_file: str, use_mock: bool = False):
    adapter   = _build_adapter(use_mock)
    lang_rt   = LanguageRouter()
    decomp    = QuestionDecomposer()
    engine    = KGTraversalEngine(adapter)
    synth     = AnswerSynthesizer()
    explainer = TemplateExplainer()
    verifier  = FaithfulnessVerifier()
    session   = SessionMemory()

    records: List[Dict] = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("[warn] No records found in input file.")
        return

    em_hits = 0
    pm_hits = 0
    pm_total = 0
    faith_ratios: List[float] = []
    latencies: List[float] = []

    for rec in records:
        question    = rec.get("question", "")
        gold_answer = rec.get("gold_answer", "")
        gold_path   = rec.get("gold_path", [])

        t0 = time.perf_counter()

        lang_res = lang_rt.route(question)
        ctx = session.get_context()
        plan = decomp.decompose(lang_res.normalized_query_en, session_context=ctx)
        result = engine.execute(plan, question)
        answer = synth.synthesize(result, question, plan.question_type)

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        # EM
        if exact_match(answer, gold_answer):
            em_hits += 1

        # Path Match
        if gold_path:
            pm_total += 1
            pred_triples = []
            if result.paths:
                pred_triples = [t.__dict__ for t in result.paths[0].triples]
            if path_match(pred_triples, gold_path):
                pm_hits += 1

        # Faithfulness
        if result.paths:
            steps = explainer.explain(result.paths[0])
            vr = verifier.verify(steps, result.paths[0])
            faith_ratios.append(vr.alignment_ratio)
        else:
            faith_ratios.append(0.0)

        # Update session memory
        session.add_turn(
            query=question,
            answer=answer,
            entity_names=[],
            entity_ids=[],
        )

    n = len(records)
    latencies.sort()
    p50 = latencies[int(n * 0.50)] if n else 0
    p95 = latencies[int(n * 0.95)] if n else 0
    em_rate = em_hits / n if n else 0
    pm_rate = pm_hits / pm_total if pm_total else None
    faith_rate = sum(faith_ratios) / len(faith_ratios) if faith_ratios else 0

    print("\n=== FinTraceQA Benchmark Results ===")
    print(f"Total questions    : {n}")
    print(f"Exact Match (EM)   : {em_rate:.2%} ({em_hits}/{n})")
    if pm_rate is not None:
        print(f"Path Match         : {pm_rate:.2%} ({pm_hits}/{pm_total})")
    else:
        print("Path Match         : N/A (no gold paths provided)")
    print(f"Faithfulness Rate  : {faith_rate:.2%}")
    print(f"Latency P50        : {p50:.1f} ms")
    print(f"Latency P95        : {p95:.1f} ms")
    print("====================================\n")

    adapter.close()


def main():
    parser = argparse.ArgumentParser(description="FinTraceQA Benchmark")
    parser.add_argument("--input", required=True, help="Path to FinEventQA.jsonl")
    parser.add_argument("--use-mock", action="store_true", help="Use mock KG instead of Neo4j")
    args = parser.parse_args()
    run_benchmark(args.input, use_mock=args.use_mock)


if __name__ == "__main__":
    main()
