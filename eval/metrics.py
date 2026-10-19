"""评测用的指标函数，各项目的 eval 脚本共用。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def dump(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


