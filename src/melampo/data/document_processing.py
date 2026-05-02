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
    executable without optional dependencies.
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
            "docling_available": self._docling_available()["available"],
        }

    def _docling_available(self) -> dict[str, Any]:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            return {"available": False, "converter": None, "error": str(exc)}
        return {"available": True, "converter": DocumentConverter, "error": None}

    def load_text_fallback(self, path: str | Path) -> str:
        path = Path(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def load_with_docling(self, path: str | Path) -> dict[str, Any]:
        """Convert a document with Docling when installed.

        Returns a structured result instead of raising when Docling is missing.
        """
        availability = self._docling_available()
        if not availability["available"]:
            return {
                "status": "not_executed",
                "reason": "docling_unavailable",
                "error": availability["error"],
                "source_path": str(path),
            }
        converter_cls = availability["converter"]
        converter = converter_cls()
        try:
            result = converter.convert(str(path))
            document = result.document
            text = document.export_to_markdown()
        except Exception as exc:  # pragma: no cover - depends on optional parser/files
            return {
                "status": "failed",
                "reason": "docling_conversion_failed",
                "error": str(exc),
                "source_path": str(path),
            }
        return {
            "status": "completed",
            "source_path": str(path),
            "text": text,
            "parser": "docling",
            "metadata": {
                "parser": "docling",
                "source_path": str(path),
                "layout_preserved": True,
            },
        }

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

    def process_document(self, path: str | Path, metadata: dict[str, Any] | None = None, prefer_docling: bool = True) -> dict[str, Any]:
        metadata = metadata or {}
        docling_result = self.load_with_docling(path) if prefer_docling else {"status": "not_requested"}
        if docling_result.get("status") == "completed":
            text = str(docling_result.get("text", ""))
            parser_metadata = dict(docling_result.get("metadata", {}))
            chunks = self.chunk_text(text=text, source_path=str(path), metadata={**metadata, **parser_metadata})
            return {
                "status": "completed",
                "parser": "docling",
                "source_path": str(path),
                "chunk_count": len(chunks),
                "documents": [chunk.to_memory_document() for chunk in chunks],
                "governance": self.docling_integration_plan()["governance_requirements"],
            }
        try:
            documents = self.process_plain_text_file(path, metadata={**metadata, "parser": "plain_text_fallback"})
        except Exception as exc:
            return {
                "status": "failed",
                "parser": "plain_text_fallback",
                "source_path": str(path),
                "error": str(exc),
                "docling_result": docling_result,
            }
        return {
            "status": "completed",
            "parser": "plain_text_fallback",
            "source_path": str(path),
            "chunk_count": len(documents),
            "documents": documents,
            "docling_result": docling_result,
            "governance": self.docling_integration_plan()["governance_requirements"],
        }

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
