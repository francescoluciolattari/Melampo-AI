from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ClinicalDocumentChunk:
    chunk_id: str
    text: str
    source_path: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_document(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "modality": self.metadata.get("modality", "clinical_document_text"),
            "source": "clinical_document_processor",
            "learning_status": "candidate",
            "metadata": {
                "record_id": self.chunk_id,
                "source_path": self.source_path,
                "page": self.page,
                "section": self.section,
                "focus": self.metadata.get("focus", "document_rag"),
                **self.metadata,
            },
        }


@dataclass(slots=True)
class ClinicalDocumentProcessor:
    """Document ingestion contract for literature, guidelines, PDFs and reports.

    Production recommendation: Docling as the first parser because it converts
    PDF, DOCX, PPTX, images, tables, formulas and layout-rich documents into
    structured outputs suitable for RAG. This local fallback keeps Melampo
    executable without optional dependencies and should be replaced by Docling
    whenever clinical document ingestion is enabled.
    """

    parser_backend: str = "docling_recommended_with_plain_text_fallback"
    chunk_size: int = 1200
    chunk_overlap: int = 160

    def describe(self) -> dict[str, Any]:
        return {
            "parser_backend": self.parser_backend,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "recommended_parser": "Docling",
            "supported_target_inputs": ["pdf", "docx", "pptx", "html", "markdown", "images", "clinical_reports"],
            "fallback_mode": "plain_text_file_reader",
        }

    def load_text_fallback(self, path: str | Path) -> str:
        path = Path(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def chunk_text(self, text: str, source_path: str, metadata: dict[str, Any] | None = None) -> list[ClinicalDocumentChunk]:
        metadata = metadata or {}
        chunks: list[ClinicalDocumentChunk] = []
        if not text:
            return chunks
        step = max(self.chunk_size - self.chunk_overlap, 1)
        for index, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start : start + self.chunk_size].strip()
            if not chunk_text:
                continue
            chunks.append(
                ClinicalDocumentChunk(
                    chunk_id=f"doc:{Path(source_path).name}:{index}",
                    text=chunk_text,
                    source_path=str(source_path),
                    section=metadata.get("section"),
                    metadata={**metadata, "chunk_index": index},
                )
            )
        return chunks

    def process_plain_text_file(self, path: str | Path, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        text = self.load_text_fallback(path)
        return [chunk.to_memory_document() for chunk in self.chunk_text(text=text, source_path=str(path), metadata=metadata)]

    def docling_integration_plan(self) -> dict[str, Any]:
        return {
            "status": "optional_dependency_contract",
            "package": "docling",
            "intended_flow": [
                "DocumentConverter().convert(source).document",
                "export structured markdown/json",
                "preserve tables, formulas, reading order and page metadata",
                "chunk by clinical section and semantic boundaries",
                "upsert chunks into VectorMemoryStore with source governance metadata",
            ],
            "governance_requirements": [
                "track source license and publication date",
                "separate peer-reviewed literature, guidelines, local protocols and synthetic traces",
                "retain page/section provenance for every RAG answer",
                "never promote dream-generated traces without rational-control validation",
            ],
        }
