"""ResearchAgent HTTP 服务。 uvicorn app.server:app --port 8013"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app.agent import ResearchAgent  # noqa: E402
from app.ingest import load  # noqa: E402

app = FastAPI(title="ResearchAgent", version="1.0.0")
_agent = None


def agent() -> ResearchAgent:
    global _agent
    if _agent is None:
        _agent = ResearchAgent(load())
    return _agent


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=300)


@app.post("/research")
def research(body: AskIn) -> dict:
    return agent().research(body.question)


@app.get("/health")
def health() -> dict:
    a = agent()
    return {"ok": True, "live_llm": a.llm.is_live, "chunks": len(a.store.chunks)}
