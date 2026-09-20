"""Tools exposed to the main support orchestrator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .corpus import CorpusQA, CorpusRecord, CorpusRepository
from .document_qa import DocumentChunk, DocumentStore
from .llm import OpenAILLM


@dataclass(frozen=True)
class ToolAnswer:
    answer: str
    tool: str
    sources: list[str]
    corpus_records: list[CorpusRecord]
    document_chunks: list[DocumentChunk]


class MetadataCorpusTool:
    name = "metadata_corpus"

    def __init__(self, repository: CorpusRepository) -> None:
        self.qa = CorpusQA(repository)

    def run(self, question: str) -> ToolAnswer:
        answer, records = self.qa.answer(question)
        return ToolAnswer(answer, self.name, [record.source for record in records], records, [])


class FAQTool:
    """Answers simple operational questions from maintained FAQ records only."""

    name = "faq"

    def __init__(self, repository: CorpusRepository) -> None:
        self.qa = CorpusQA(repository)
        self.repository = repository

    def run(self, question: str, limit: int = 3) -> ToolAnswer:
        candidates = [
            record
            for record in self.repository.search(question, limit=10)
            if record.record_id.startswith("FAQ-") or record.source.startswith("faq://")
        ]
        records = candidates[:1] if candidates else []
        if not records:
            return ToolAnswer("No sufficiently relevant FAQ answer was found.", self.name, [], [], [])

        best = records[0]
        answer = (
            f"{best.answer}\n\nSteps:\n"
            + "\n".join(f"{index}. {step}" for index, step in enumerate(best.procedure, 1))
            + f"\n\nPOC: {best.poc}\nApproval: {best.approval}"
            + f"\nSource: {best.record_id} ({best.source})"
        )
        return ToolAnswer(answer, self.name, [record.source for record in records], records, [])


class AgenticRAGTool:
    """Iterative retrieval tool for indexed PDF, DOCX, and text chunks."""

    name = "agentic_rag"

    def __init__(self, store: DocumentStore, llm: OpenAILLM | None = None) -> None:
        self.store = store
        self.llm = llm or OpenAILLM.from_environment()

    def run(self, question: str, limit: int = 5) -> ToolAnswer:
        chunks: list[DocumentChunk] = []
        seen: set[tuple[str, int]] = set()
        for query in (question, self._focused_query(question)):
            for chunk in self.store.search(query, limit=limit):
                key = (chunk.document_name, chunk.chunk_index)
                if key not in seen:
                    chunks.append(chunk)
                    seen.add(key)
            if chunks:
                break
        if not chunks:
            return ToolAnswer("I could not find relevant information in the indexed documents.", self.name, [], [], [])

        context = "\n\n".join(
            f"[{chunk.document_name} | chunk {chunk.chunk_index}]\n{chunk.content}" for chunk in chunks
        )
        if self.llm is not None:
            answer = self.llm.complete(
                "Answer only from the retrieved document excerpts. Cite document and chunk. Say when the answer is not present.",
                f"Question: {question}\n\nExcerpts:\n{context}",
            )
        else:
            answer = "\n\n".join(chunk.content[:600] for chunk in chunks[:2])
        return ToolAnswer(
            answer,
            self.name,
            [f"{chunk.document_name}#chunk-{chunk.chunk_index}" for chunk in chunks],
            [],
            chunks,
        )

    @staticmethod
    def _focused_query(question: str) -> str:
        return " ".join(dict.fromkeys(re.findall(r"[a-z0-9]{4,}", question.lower())))


class TextToSQLTool:
    """Guarded text-to-SQL tool over the read-only support corpus."""

    name = "text_to_sql"

    def __init__(self, repository: CorpusRepository) -> None:
        self.repository = repository

    def run(self, question: str, limit: int = 10) -> ToolAnswer:
        sql, parameters = self.to_sql(question, limit)
        rows = self._execute_read_only(sql, parameters)
        records = [self._row_to_record(row) for row in rows]
        answer = "\n".join(
            f"{record.record_id}: {record.title} | POC: {record.poc} | Approval: {record.approval}"
            for record in records
        ) or "No matching metadata records were found."
        return ToolAnswer(answer, self.name, [record.source for record in records], records, [])

    def to_sql(self, question: str, limit: int = 10) -> tuple[str, tuple[str, ...]]:
        ignored = {"what", "which", "where", "does", "have", "with", "from", "show", "list"}
        terms = [term for term in re.findall(r"[a-z0-9]{3,}", question.lower()) if term not in ignored]
        clauses: list[str] = []
        parameters: list[str] = []
        for term in dict.fromkeys(terms[:6]):
            clauses.append("(title LIKE ? OR category LIKE ? OR keywords LIKE ? OR answer LIKE ?)")
            parameters.extend([f"%{term}%"] * 4)
        where = " OR ".join(clauses) or "1 = 1"
        sql = (
            "SELECT record_id, category, title, keywords, answer, procedure, poc, approval, source "
            f"FROM support_corpus WHERE {where} ORDER BY record_id LIMIT ?"
        )
        return sql, tuple(parameters + [str(limit)])

    def _execute_read_only(self, sql: str, parameters: tuple[str, ...]) -> list[Any]:
        normalized = sql.strip().lower()
        blocked = ("insert ", "update ", "delete ", "drop ", "alter ", "pragma ")
        if not normalized.startswith("select") or ";" in normalized or any(word in normalized for word in blocked):
            raise ValueError("Text-to-SQL only permits one read-only SELECT statement.")
        if "from support_corpus" not in normalized:
            raise ValueError("Query references a table outside the allowlist.")
        return self.repository.connection.execute(sql, parameters).fetchall()

    @staticmethod
    def _row_to_record(row: Any) -> CorpusRecord:
        return CorpusRecord(
            row["record_id"], row["category"], row["title"], tuple(row["keywords"].split("|")),
            row["answer"], tuple(row["procedure"].split("|")), row["poc"], row["approval"], row["source"],
        )
