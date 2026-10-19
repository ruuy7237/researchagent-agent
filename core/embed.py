"""向量化。默认用 md5 哈希把 token 映射到 512 维，装了 sentence-transformers 可换 bge-m3。

哈希这个做法是为了不依赖 GPU 和外网。md5 前 4 字节取模决定落在哪一维，第 5 字节的最低位决定正负号，
本质是一个 random projection，中文二元组上用起来够。效果肯定不如真 embedding，但整套链路能在任何机器上跑通。

要换真向量就设 EMBED_BACKEND=bge-m3，两个后端接口一样，可以直接对比召回差异。
"""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np

from .text import tokenize


def _hash_index(token: str, dim: int) -> tuple[int, float]:
    h = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1.0 if h[4] & 1 else -1.0
    return idx, sign


class HashingEmbedder:
    backend = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                i, s = _hash_index(tok, self.dim)
                mat[row, i] += s
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class STEmbedder:
    """需要 pip install sentence-transformers，且首次运行会下载模型。"""

    backend = "bge-m3"

    def __init__(self, name: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self.model = SentenceTransformer(name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder():
    backend = os.getenv("EMBED_BACKEND", "hashing").lower()
    if backend in {"bge-m3", "bge", "st"}:
        try:
            return STEmbedder()
        except Exception as exc:  # 模型或依赖缺失时自动回退，保证可运行
            print(f"[embed] bge-m3 不可用（{exc.__class__.__name__}），回退 hashing")
    return HashingEmbedder()
