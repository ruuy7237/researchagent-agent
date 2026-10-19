"""ResearchAgent 评测：事实准确率、引用可溯源、存疑与拒答、信息增益停止效果。

    python eval/run_eval.py [--no-llm]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app.agent import ResearchAgent  # noqa: E402
from app.ingest import load  # noqa: E402
from core.llm import LLM  # noqa: E402
from metrics import dump, pct  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    store = load()
    llm = LLM()
    if args.no_llm:
        llm.api_key = None
    agent = ResearchAgent(store, llm)

    golden = [json.loads(l) for l in (PROJECT / "data" / "golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pos = [g for g in golden if g["answerable"]]
    neg = [g for g in golden if not g["answerable"]]

    fact_ok = 0
    cite_ok = 0
    rounds, chunks, lat, covs = [], [], [], []

    for g in pos:
        res = agent.research(g["q"])
        body = res["answer"] + " " + " ".join(e["snippet"] for e in res["evidence"])
        cov = sum(1 for k in g["must"] if k in body) / len(g["must"])
        covs.append(cov)
        if not res["refused"] and cov >= 0.6:
            fact_ok += 1
        if res["citation_valid"]:
            cite_ok += 1
        rounds.append(res["rounds"])
        chunks.append(res["evidence_chunks"])
        lat.append(res["latency_ms"])

    refused_ok = sum(1 for g in neg if agent.research(g["q"])["refused"])

    n = len(pos) or 1
    result = {
        "dataset": {"propositions": len(golden), "answerable": len(pos), "unanswerable": len(neg),
                    "chunks": len(store.chunks),
                    "mode": "extractive" if args.no_llm or not llm.is_live else "llm"},
        "quality": {
            "fact_accuracy": round(fact_ok / n, 4),
            "keyword_coverage_avg": round(sum(covs) / n, 4),
            "citation_valid_rate": round(cite_ok / n, 4),
        },
        "safety": {"refusal_accuracy": round(refused_ok / (len(neg) or 1), 4)},
        "efficiency": {
            "avg_rounds": round(statistics.mean(rounds), 2),
            "avg_evidence_chunks": round(statistics.mean(chunks), 2),
            "p50_latency_ms": round(statistics.median(lat), 1),
        },
        "llm_usage": llm.usage.as_dict(),
    }
    dump(PROJECT / "data" / "metrics.json", result)

    print(f"\n=== ResearchAgent 评测（{result['dataset']['mode']}）===")
    print(f"命题 {len(golden)} 条（可答 {len(pos)} / 不可答 {len(neg)}），索引 {len(store.chunks)} 块")
    print(f"事实准确率 {pct(result['quality']['fact_accuracy'])} | 要点覆盖 {pct(result['quality']['keyword_coverage_avg'])} | 引用有效 {pct(result['quality']['citation_valid_rate'])}")
    print(f"拒答准确 {pct(result['safety']['refusal_accuracy'])} | 平均轮次 {result['efficiency']['avg_rounds']} | 证据块 {result['efficiency']['avg_evidence_chunks']} | p50 {result['efficiency']['p50_latency_ms']}ms")


if __name__ == "__main__":
    main()
