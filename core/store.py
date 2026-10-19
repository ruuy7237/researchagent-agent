"""混合检索：BM25 稀疏召回 + 向量稠密召回 + RRF 融合 + 词面重排。

语料规模在万级以内，索引全部放内存，落盘为 jsonl + npz，便于提交与复现。
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .embed import get_embedder
from .text import chunk_text, tokenize


@dataclass
class Chunk:
    cid: int
    doc_id: str
    title: str
    source: str
    ord: int
    text: str
    tokens: list[str] = field(default_factory=list)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.df: Counter = Counter()
        self.tf: list[Counter] = []
        self.dl: list[int] = []
        self.avgdl = 0.0
        self.n = 0

    def fit(self, docs_tokens: Sequence[Sequence[str]]) -> None:
        self.tf = [Counter(t) for t in docs_tokens]
        self.dl = [len(t) for t in docs_tokens]
        self.n = len(docs_tokens)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        for c in self.tf:
            for tok in c:
                self.df[tok] += 1

    def score(self, query_tokens: Iterable[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for tok in query_tokens:
            df = self.df.get(tok, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(tok, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


class HybridStore:
    def __init__(self, embedder=None) -> None:
        self.embedder = embedder or get_embedder()
        self.chunks: list[Chunk] = []
        self.bm25 = BM25()
        self.mat: np.ndarray | None = None

    # ---------- 建库 ----------
    def add_document(self, doc_id: str, title: str, source: str, text: str,
                     chunk_size: int = 400, overlap: int = 80) -> int:
        added = 0
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            self.chunks.append(
                Chunk(len(self.chunks), doc_id, title, source, i, piece))
            added += 1
        return added

    def build(self) -> None:
        if not self.chunks:
            raise RuntimeError("语料为空，先 add_document")
        for c in self.chunks:
            c.tokens = tokenize(c.title + " " + c.text)
        self.bm25.fit([c.tokens for c in self.chunks])
        self.mat = self.embedder.encode([c.title + "\n" + c.text for c in self.chunks])

    # ---------- 落盘 ----------
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(
                    {"cid": c.cid, "doc_id": c.doc_id, "title": c.title,
                     "source": c.source, "ord": c.ord, "text": c.text},
                    ensure_ascii=False) + "\n")
        np.save(out / "vectors.npy", self.mat)

    @classmethod
    def load(cls, in_dir: str | Path, embedder=None) -> "HybridStore":
        d = Path(in_dir)
        store = cls(embedder)
        for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            o = json.loads(line)
            store.chunks.append(Chunk(o["cid"], o["doc_id"], o["title"],
                                      o["source"], o["ord"], o["text"]))
        store.mat = np.load(d / "vectors.npy")
        for c in store.chunks:
            c.tokens = tokenize(c.title + " " + c.text)
        store.bm25.fit([c.tokens for c in store.chunks])
        return store

    # ---------- 检索 ----------
    def _dense(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query])[0]
        return self.mat @ q

    def _rrf(self, ranks: Sequence[np.ndarray], k: int = 60) -> np.ndarray:
        fused = np.zeros(len(self.chunks), dtype=np.float32)
        for score in ranks:
            order = np.argsort(-score)
            for pos, idx in enumerate(order[: max(k * 4, 20)]):
                fused[idx] += 1.0 / (k + pos + 1)
        return fused

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
        bm = self.bm25.score(tokenize(query))
        dn = self._dense(query)
        if mode == "bm25":
            fused = bm
        elif mode == "dense":
            fused = dn
        else:
            fused = self._rrf([bm, dn]) + 1e-6 * dn  # 稠密分作细粒度 tie-break

        order = np.argsort(-fused)[: max(top_k * 3, top_k)]
        picked = []
        for idx in order:
            c = self.chunks[int(idx)]
            picked.append({
                "cid": c.cid, "doc_id": c.doc_id, "title": c.title,
                "source": c.source, "text": c.text,
                "score": float(fused[idx]), "bm25": float(bm[idx]),
                "dense": float(dn[idx]),
                "rerank": self._lexical_overlap(query, c),
            })
        if mode == "hybrid":
            picked.sort(key=lambda x: (x["rerank"], x["score"]), reverse=True)
        return picked[:top_k]

    @staticmethod
    def _lexical_overlap(query: str, chunk: Chunk) -> float:
        """轻量重排：查询词在标题中的命中权重更高。语料小、无 GPU 时的性价比选择。"""
        q = set(tokenize(query))
        if not q:
            return 0.0
        t = set(tokenize(chunk.title))
        c = set(chunk.tokens)
        head = len(q & t) / len(q)
        body = len(q & c) / len(q)
        return 0.6 * head + 0.4 * body
