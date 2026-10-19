"""Plan → Execute → Reflect 循环，输出带引用，资料说法冲突时标存疑。

停止条件是新增证据占比，低于阈值就退出，不按固定轮数跑。
固定轮数的毛病是后面几轮都在查同一批文档，多花的 token 换不来新结论。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import LLM  # noqa: E402
from core.store import HybridStore  # noqa: E402
from core.text import split_sentences, tokenize  # noqa: E402

STOPWORDS = {"什么", "怎么", "为什么", "哪几", "哪些", "多少", "是否", "如何", "以及", "分别"}
GATE = 0.30  # 证据闸门：问题词在最佳证据中的覆盖率下限

SYSTEM = """你是行业研究助手。规则：
1. 只能使用【证据】中的内容作答，不得补充外部知识；
2. 每条结论后用 [编号] 标注来源；
3. 证据不足或互相冲突时，明确写出"存疑：…"，不要下结论；
4. 中文作答，先给结论，再分点给依据，不超过 260 字。"""


class ResearchAgent:
    def __init__(self, store: HybridStore, llm: LLM | None = None,
                 max_rounds: int = 2, gain_threshold: float = 0.12) -> None:
        self.store = store
        self.llm = llm or LLM()
        self.max_rounds = max_rounds
        self.gain = gain_threshold

    # ---------- 计划 ----------
    def plan(self, question: str) -> list[str]:
        subs = [question]
        keys = [t for t in tokenize(question) if len(t) == 2 and t not in STOPWORDS]
        # 取前 3 个不同关键词做检索扩展，避免子问题膨胀
        seen, extra = set(), []
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            extra.append(k)
        subs += [f"{k}" for k in extra[:3]]
        return subs

    # ---------- 执行 + 反思 ----------
    def research(self, question: str) -> dict:
        started = time.perf_counter()
        keys = [t for t in tokenize(question) if len(t) == 2 and t not in STOPWORDS]
        queries = self.plan(question)
        evidence: dict[int, dict] = {}
        rounds = 0

        while rounds < self.max_rounds and queries:
            rounds += 1
            before = len(evidence)
            for sq in queries:
                for h in self.store.search(sq, top_k=3):
                    evidence.setdefault(h["cid"], h)

            gain = (len(evidence) - before) / max(len(evidence), 1)
            if gain < self.gain:
                break  # 信息增益不足，停止探索

            # 反思：逐个子问题检查证据覆盖率，只对没覆盖的换说法重查
            chunks = list(evidence.values())
            uncovered = []
            for sq in queries:
                qs = set(tokenize(sq))
                best = 0.0
                for h in chunks:
                    hs = set(tokenize(h["text"] + h["title"]))
                    best = max(best, len(qs & hs) / (len(qs) or 1))
                if best < 0.4:
                    uncovered.append(sq)
            queries = [f"{sq} {keys[0]}" for sq in uncovered[:2]] if uncovered and len(keys) > 1 else []

        hits = sorted(evidence.values(), key=lambda h: -h["score"])[:6]
        if not hits:
            return self._refuse(started, rounds, 0)

        # 证据闸门：问题词在最佳证据中的覆盖率过低，说明语料里根本没这块知识
        q_strict = {t for t in tokenize(question) if len(t) >= 2}
        best_cov = max((len(q_strict & {t for t in tokenize(h["text"] + h["title"])})
                        / (len(q_strict) or 1)) for h in hits)
        if best_cov < GATE:
            return self._refuse(started, rounds, len(evidence))

        context = "\n\n".join(f"[{i}]《{h['title']}》\n{h['text']}" for i, h in enumerate(hits, 1))
        answer, doubts = self._generate(question, context, hits)
        if answer is None:
            return self._refuse(started, rounds, len(evidence))

        cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
        return {
            "answer": answer,
            "refused": False,
            "evidence": [
                {"n": i, "doc": h["doc_id"], "title": h["title"],
                 "chunk": h["cid"], "snippet": h["text"][:100]}
                for i, h in enumerate(hits, 1) if i in cited
            ],
            "doubts": doubts,
            "rounds": rounds,
            "evidence_chunks": len(evidence),
            "citation_valid": all(1 <= c <= len(hits) for c in cited) and bool(cited),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": self.llm.usage.as_dict(),
        }

    # ---------- 生成 ----------
    def _generate(self, question: str, context: str, hits: list[dict]):
        reply = self.llm.chat(SYSTEM, f"【证据】\n{context}\n\n【问题】{question}",
                              temperature=0.1, max_tokens=600)
        if reply:
            if reply.strip().startswith("无依据"):
                return None, []
            doubts = [s for s in split_sentences(reply) if s.startswith("存疑")]
            return reply, doubts
        return self._extractive(question, context), []

    def _extractive(self, question: str, context: str) -> str:
        q = set(tokenize(question))
        pool: list[tuple[float, int, str]] = []
        cur = 1
        for line in context.splitlines():
            m = re.match(r"\[(\d+)\]", line)
            if m:
                cur = int(m.group(1))
                continue
            if not line.strip():
                continue
            for sent in split_sentences(line):
                if len(sent) < 10:
                    continue
                score = len(q & set(tokenize(sent))) / (len(q) or 1)
                if score >= 0.1:
                    pool.append((score, cur, sent))
        if not pool:
            return None
        out, seen = [], set()
        for _, idx, sent in sorted(pool, reverse=True)[:3]:
            if sent in seen:
                continue
            seen.add(sent)
            out.append(f"{sent.rstrip('。')}[{idx}]")
        return "；".join(out) + "。"

    def _refuse(self, started: float, rounds: int, n_ev: int) -> dict:
        return {
            "answer": "现有证据不足以回答该问题，建议补充资料或缩小问题范围。",
            "refused": True, "evidence": [], "doubts": [],
            "rounds": rounds, "evidence_chunks": n_ev, "citation_valid": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": self.llm.usage.as_dict(),
        }
