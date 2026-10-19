"""LLM 客户端，走 OpenAI 兼容协议，DeepSeek / 通义 / 本地 vLLM 都能用。

没配 key 的时候 chat() 返回 None，不抛异常。调用方拿到 None 就走确定性分支，
这样没网的机器也能把评测跑完。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_env() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent.parent, *here.parents]:
        env = parent / ".env"
        if env.exists():
            load_dotenv(env, override=False)
            return


_load_env()


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "errors": self.errors,
        }


@dataclass
class LLM:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    timeout: float = 60.0
    max_retry: int = 2
    usage: Usage = field(default_factory=Usage)

    @property
    def is_live(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, temperature: float = 0.2,
             json_mode: bool = False, max_tokens: int = 1200) -> str | None:
        """返回文本；未配置 key 或连续失败时返回 None。"""
        if not self.is_live:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retry + 1):
            started = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self.usage.calls += 1
                self.usage.latency_ms += (time.perf_counter() - started) * 1000
                if resp.usage:
                    self.usage.prompt_tokens += resp.usage.prompt_tokens
                    self.usage.completion_tokens += resp.usage.completion_tokens
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                self.usage.errors += 1
                if attempt == self.max_retry:
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> dict | None:
        raw = self.chat(system, user, temperature=temperature, json_mode=True)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    return None
        return None
