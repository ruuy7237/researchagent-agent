"""中文/英文混合文本处理：分词、分块、句子切分。

刻意不依赖 jieba，用「英文单词 + 中文字符二元组」的混合切分，
在中文检索上接近粗粒度分词效果，且零额外依赖、可离线。
"""
from __future__ import annotations

import re
from typing import Iterable

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_]+")
_SENT_END = re.compile(r"(?<=[。！？；!?;])|(?<=[.])\s+")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")


def cjk_chars(text: str) -> str:
    return "".join(_CJK.findall(text))


def tokenize(text: str) -> list[str]:
    """英文按词、中文按字 + 二元组。返回去停用词后的小写 token。"""
    low = text.lower()
    tokens = _WORD.findall(low)
    chars = cjk_chars(low)
    if chars:
        tokens.extend(chars)  # 单字
        tokens.extend(chars[i : i + 2] for i in range(len(chars) - 1))  # 二元组
    return [t for t in tokens if len(t) <= 12 or _WORD.match(t)]


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    return parts


def parse_markdown(text: str) -> list[tuple[str, str]]:
    """把 markdown 拆成 [(当前标题, 段落文本)]，标题用于给分块打标签与提升召回。"""
    blocks: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            blocks.append((heading, body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            heading = m.group(2).strip()
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush()
    return blocks


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """按句子边界聚合到 size 字符左右，相邻块重叠 overlap 字符。"""
    sents = split_sentences(text) or [text]
    chunks: list[str] = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) <= size:
            cur += s
            continue
        if cur:
            chunks.append(cur)
        cur = s
    if cur:
        chunks.append(cur)

    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, nxt in zip(chunks, chunks[1:]):
        out.append(prev[-overlap:] + nxt)
    return out


def iter_docs(paths: Iterable) -> Iterable[tuple[str, str, str]]:
    """读取一批 .md/.txt，产出 (source, title, text)。"""
    from pathlib import Path

    for p in paths:
        p = Path(p)
        raw = p.read_text(encoding="utf-8", errors="ignore")
        title = p.stem
        for line in raw.splitlines():
            m = _HEADING.match(line)
            if m and m.group(1) == "#":
                title = m.group(2).strip()
                break
        yield p.name, title, raw
