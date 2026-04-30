from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_path: str
    section_title: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "record_id": self.chunk_id,
            "text": self.text,
            "metadata": {
                "source_path": self.source_path,
                "section_title": self.section_title,
                **self.metadata,
            },
            "source": "document_ingestion_contract",
        }


@dataclass(slots=True)
class DocumentIngestionContract:
    """Offline document ingestion contract for already-converted clinical text."""

    max_chunk_chars: int = 1200
    overlap_chars: int = 120
    converter_name: str = "structured_document_contract"

    def chunk_text(self, text: str, source_path: str, metadata: dict | None = None) -> list[DocumentChunk]:
        text = text or ""
        metadata = metadata or {}
        if not text.strip():
            return []
        chunks = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.max_chunk_chars, len(text))
            value = text[start:end].strip()
            if value:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{Path(source_path).stem}-chunk-{index + 1}",
                        text=value,
                        source_path=source_path,
                        section_title=metadata.get("section_title"),
                        metadata={
                            **metadata,
                            "chunk_index": index,
                            "converter": self.converter_name,
                            "chunk_start": start,
                            "chunk_end": end,
                        },
                    )
                )
            if end == len(text):
                break
            start = max(end - self.overlap_chars, start + 1)
            index += 1
        return chunks

    def ingest_text(self, text: str, source_path: str, metadata: dict | None = None) -> dict:
        chunks = self.chunk_text(text=text, source_path=source_path, metadata=metadata)
        return {
            "status": "text_chunked",
            "source_path": source_path,
            "chunk_count": len(chunks),
            "chunks": [chunk.to_record() for chunk in chunks],
            "provenance": {
                "converter": self.converter_name,
                "input_type": "already_converted_text",
                "runtime_dependency_required_for_tests": False,
            },
        }

    def describe(self) -> dict:
        return {
            "contract": "document_ingestion_contract",
            "converter_name": self.converter_name,
            "max_chunk_chars": self.max_chunk_chars,
            "overlap_chars": self.overlap_chars,
            "runtime_dependency_required_for_tests": False,
            "future_target": "structured clinical document conversion pipeline",
        }
