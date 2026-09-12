"""
Read knowledge .txt files, split into chunks, embed, and store in Supabase.

Run from the burnout-ai directory:

    python -m rag.ingestion
"""

from __future__ import annotations

import re
from pathlib import Path

from app.database import supabase
from rag.embeddings import embed_texts

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"

# Fallback metadata when a file has no TITLE/CATEGORY header.
DOCUMENT_META = {
    "01_stress_management.txt": {
        "title": "Stress Management for College Students",
        "category": "Stress",
        "source": "Amanvermez et al. (2022); Worsley, Pennington, & Corcoran (2022)",
    },
    "02_academic_workload.txt": {
        "title": "Academic Workload Management",
        "category": "Academic Workload",
        "source": "Liu, Ma, & Jia (2026); Wei et al. (2026)",
    },
    "03_sleep_management.txt": {
        "title": "Sleep and Study-Rest Balance",
        "category": "Sleep",
        "source": "CDC (2024); Gardani et al. (2022)",
    },
    "04_study_time_and_learning.txt": {
        "title": "Study Time and Effective Learning",
        "category": "Study Time",
        "source": "McDermott (2021); Toppino & Pagano (2021); Liu, Ma, & Jia (2026)",
    },
    "05_student_support_and_referral.txt": {
        "title": "Student Support, Guidance, and Referral",
        "category": "Student Support",
        "source": "Worsley, Pennington, & Corcoran (2022)",
    },
    "06_system_rag_rules.txt": {
        "title": "RAG System Rules for Academic Burnout Early Warning",
        "category": "System Rules",
        "source": "System design rules for MFBI + ML + RAG",
    },
}

TARGET_CHARS = 900
MIN_CHARS = 220
HEADER_KEYS = ("TITLE", "CATEGORY", "SOURCE", "PURPOSE")


def parse_document(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8").strip()
    meta = dict(DOCUMENT_META.get(path.name, {}))
    lines = raw.splitlines()
    consumed = 0
    for line in lines[:6]:
        if ":" not in line:
            break
        key, value = line.split(":", 1)
        key_up = key.strip().upper()
        if key_up not in HEADER_KEYS:
            break
        meta[key_up.lower()] = value.strip()
        consumed += 1
    body = "\n".join(lines[consumed:]).strip()
    meta.setdefault("title", path.stem.replace("_", " ").title())
    meta.setdefault("category", "General")
    meta.setdefault("source", path.name)
    return meta, body


def split_into_chunks(text: str) -> list[str]:
    parts = re.split(r"\n(?=(?:SECTION\s+\d+|\d+\.\s+[A-Z]))", text.strip())
    units: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= TARGET_CHARS * 1.4:
            units.append(part)
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", part) if p.strip()]
        units.extend(paragraphs or [part])

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        if len(current) + 2 + len(unit) <= TARGET_CHARS:
            current = f"{current}\n\n{unit}"
        else:
            chunks.append(current.strip())
            overlap = current[-120:].strip() if len(current) > 240 else ""
            current = f"{overlap}\n\n{unit}" if overlap else unit
    if current.strip():
        chunks.append(current.strip())

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)
    return [chunk for chunk in merged if chunk.strip()]


def ingest_documents(documents_dir: Path = DOCUMENTS_DIR) -> int:
    files = sorted(documents_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt knowledge files in {documents_dir}")

    total = 0
    for path in files:
        meta, body = parse_document(path)
        chunks = split_into_chunks(body)
        if not chunks:
            print(f"skip empty: {path.name}")
            continue

        print(f"embedding {path.name} ({len(chunks)} chunks)…")
        embeddings = embed_texts(chunks)

        supabase.table("rag_documents").delete().eq("source", path.name).execute()

        rows = []
        for index, (content, embedding) in enumerate(zip(chunks, embeddings), start=1):
            rows.append(
                {
                    "title": meta["title"],
                    "category": meta["category"],
                    "content": content,
                    "source": path.name,
                    "chunk_number": index,
                    "embedding": embedding,
                }
            )

        supabase.table("rag_documents").insert(rows).execute()
        total += len(rows)
        print(f"  stored {len(rows)} rows [{meta['category']}]")

    return total


def main() -> None:
    count = ingest_documents()
    print(f"Ingestion complete. {count} chunks stored in rag_documents.")


if __name__ == "__main__":
    main()
