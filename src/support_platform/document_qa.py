"""Document ingestion, SQLite indexing, retrieval, and grounded QA."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import OpenAILLM


@dataclass(frozen=True)
class DocumentChunk:
    document_name: str
    chunk_index: int
    content: str


DOCUMENT_QA_PROMPT = """You answer questions using only the supplied document excerpts.
If the excerpts do not contain the answer, say that the answer was not found in the uploaded documents.
Cite the document name and chunk number for each factual claim.
Be concise and do not invent facts.
"""


class DocumentStore:
    def __init__(self, database_path: str = ":memory:") -> None:
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                document_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                PRIMARY KEY (document_name, chunk_index)
            )
            """
        )
        self.connection.commit()

    def replace_document(self, document_name: str, chunks: list[DocumentChunk]) -> None:
        self.connection.execute("DELETE FROM document_chunks WHERE document_name = ?", (document_name,))
        self.connection.executemany(
            "INSERT INTO document_chunks (document_name, chunk_index, content) VALUES (?, ?, ?)",
            [(chunk.document_name, chunk.chunk_index, chunk.content) for chunk in chunks],
        )
        self.connection.commit()

    def search(self, question: str, limit: int = 5) -> list[DocumentChunk]:
        terms = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
        rows = self.connection.execute(
            "SELECT document_name, chunk_index, content FROM document_chunks"
        ).fetchall()
        ranked: list[tuple[int, DocumentChunk]] = []
        for row in rows:
            content_terms = set(re.findall(r"[a-z0-9]{3,}", row["content"].lower()))
            score = len(terms & content_terms)
            if score:
                ranked.append((score, DocumentChunk(row["document_name"], row["chunk_index"], row["content"])))
        ranked.sort(key=lambda item: (-item[0], item[1].document_name, item[1].chunk_index))
        return [chunk for _, chunk in ranked[:limit]]

    def close(self) -> None:
        self.connection.close()


def extract_text(file_name: str, file_bytes: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".csv", ".json"}:
        return file_bytes.decode("utf-8", errors="replace")
    if suffix == ".docx":
        from docx import Document

        document = Document(__import__("io").BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(__import__("io").BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("Supported file types are PDF, DOCX, TXT, Markdown, CSV, and JSON.")


def chunk_text(document_name: str, text: str, chunk_size: int = 1200) -> list[DocumentChunk]:
    words = text.split()
    chunks: list[DocumentChunk] = []
    for start in range(0, len(words), chunk_size):
        content = " ".join(words[start : start + chunk_size]).strip()
        if content:
            chunks.append(DocumentChunk(document_name, len(chunks), content))
    return chunks


class DocumentQA:
    def __init__(self, store: DocumentStore, llm: OpenAILLM | None = None) -> None:
        self.store = store
        self.llm = llm or OpenAILLM.from_environment()

    def ingest(self, file_name: str, file_bytes: bytes) -> int:
        text = extract_text(file_name, file_bytes)
        chunks = chunk_text(file_name, text)
        if not chunks:
            raise ValueError("The uploaded document contains no extractable text.")
        self.store.replace_document(file_name, chunks)
        return len(chunks)

    def answer(self, question: str, limit: int = 5) -> tuple[str, list[DocumentChunk]]:
        chunks = self.store.search(question, limit=limit)
        if not chunks:
            return "I could not find relevant information in the uploaded documents.", []

        context = "\n\n".join(
            f"[{chunk.document_name} | chunk {chunk.chunk_index}]\n{chunk.content}"
            for chunk in chunks
        )
        if self.llm is not None:
            return self.llm.complete(DOCUMENT_QA_PROMPT, f"Question: {question}\n\nExcerpts:\n{context}"), chunks

        sentences = re.split(r"(?<=[.!?])\s+", " ".join(chunk.content for chunk in chunks))
        terms = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
        selected = [sentence for sentence in sentences if terms & set(re.findall(r"[a-z0-9]{3,}", sentence.lower()))]
        answer = " ".join(selected[:3]) or chunks[0].content[:600]
        return f"Extractive answer: {answer}\n\nSources: " + ", ".join(
            f"{chunk.document_name} chunk {chunk.chunk_index}" for chunk in chunks
        ), chunks
