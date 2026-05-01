from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..memory.vector_memory import InMemoryVectorStore


@dataclass(slots=True)
class TextDocumentChunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text_length": len(self.text),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TextDocumentIngestor:
    """Local text document ingestion contract for clinical knowledge sources."""

    vector_store: InMemoryVectorStore = field(default_factory=InMemoryVectorStore)
    chunk_size: int = 900
    chunk_overlap: int = 120
    parser_mode: str = "local_text_ingestion_contract"

    def ingest_path(self, path: str | Path, source: str = "clinical_document", document_type: str = "text_document") -> dict[str, Any]:
        doc_path = Path(path)
        text = doc_path.read_text(encoding="utf-8")
        chunks = self.chunk_text(
            text=text,
            base_id=doc_path.stem,
            metadata={
                "source": source,
                "path": str(doc_path),
                "document_type": document_type,
                "parser_mode": self.parser_mode,
            },
        )
        stored = []
        for chunk in chunks:
            stored.append(
                self.vector_store.upsert_text(
                    record_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    source=source,
                    learning_status="candidate",
                )
            )
        return {
            "status": "ingested",
            "parser_mode": self.parser_mode,
            "source": source,
            "document_type": document_type,
            "path": str(doc_path),
            "chunk_count": len(chunks),
            "chunks": [chunk.describe() for chunk in chunks],
            "stored_records": stored,
            "vector_memory": self.vector_store.describe(),
            "future_structured_parser_ready": True,
        }

    def chunk_text(self, text: str, base_id: str, metadata: dict | None = None) -> list[TextDocumentChunk]:
        metadata = metadata or {}
        clean = " ".join(text.split())
        if not clean:
            return [TextDocumentChunk(chunk_id=f"{base_id}-chunk-0", text="", metadata={**metadata, "chunk_index": 0})]
        chunks = []
        start = 0
        index = 0
        step = max(self.chunk_size - self.chunk_overlap, 1)
        while start < len(clean):
            end = min(start + self.chunk_size, len(clean))
            chunks.append(
                TextDocumentChunk(
                    chunk_id=f"{base_id}-chunk-{index}",
                    text=clean[start:end],
                    metadata={**metadata, "chunk_index": index, "start_char": start, "end_char": end},
                )
            )
            if end == len(clean):
                break
            start += step
            index += 1
        return chunks
