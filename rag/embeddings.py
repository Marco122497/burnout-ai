"""Embeddings for RAG chunks and retrieval queries.

Default: local ONNX model (no OpenAI credits).
Optional: OpenAI when EMBEDDING_PROVIDER=openai.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv()

LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
LOCAL_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LOCAL_EMBEDDING_DIMENSIONS = 384
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))

EMBEDDING_MODEL = (
    OPENAI_EMBEDDING_MODEL
    if EMBEDDING_PROVIDER == "openai"
    else LOCAL_EMBEDDING_MODEL
)
EMBEDDING_DIMENSIONS = (
    OPENAI_EMBEDDING_DIMENSIONS
    if EMBEDDING_PROVIDER == "openai"
    else LOCAL_EMBEDDING_DIMENSIONS
)


def openai_configured() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and "your_openai" not in key.lower()


def llm_client():
    return _openai_client()


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    if not openai_configured():
        raise RuntimeError(
            "Set OPENAI_API_KEY in burnout-ai/.env (or the process environment)."
        )
    return OpenAI()


@lru_cache(maxsize=1)
def _local_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=LOCAL_EMBEDDING_MODEL)


def _embed_openai(texts: list[str]) -> list[list[float]]:
    kwargs: dict = {"model": OPENAI_EMBEDDING_MODEL, "input": texts}
    if OPENAI_EMBEDDING_MODEL.startswith("text-embedding-3"):
        kwargs["dimensions"] = OPENAI_EMBEDDING_DIMENSIONS
    response = _openai_client().embeddings.create(**kwargs)
    by_index = {item.index: item.embedding for item in response.data}
    return [by_index[i] for i in range(len(texts))]


def _embed_local(texts: list[str]) -> list[list[float]]:
    vectors = list(_local_model().embed(texts))
    return [vector.tolist() for vector in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
