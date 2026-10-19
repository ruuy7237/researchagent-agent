"""ResearchAgent 语料入库。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.store import HybridStore  # noqa: E402
from core.text import iter_docs  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT / "data" / "corpus"
INDEX = PROJECT / "data" / "index"


def build() -> HybridStore:
    files = sorted(CORPUS.glob("*.md"))
    store = HybridStore()
    total = 0
    for source, title, text in iter_docs(files):
        n = store.add_document(Path(source).stem, title, source, text, 220, 40)
        total += n
        print(f"  {source:<36} {n:>3} 块")
    store.build()
    store.save(INDEX)
    print(f"[ingest] {len(files)} 篇 / {total} 块 / {store.embedder.backend}")
    return store


def load() -> HybridStore:
    if not (INDEX / "chunks.jsonl").exists():
        return build()
    return HybridStore.load(INDEX)


if __name__ == "__main__":
    build()
