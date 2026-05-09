from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 /,:;()\-]{5,})$", re.MULTILINE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_CLINICAL_TERMS = {
    "cough": "Symptom:Cough",
    "fever": "Symptom:Fever",
    "pain": "Symptom:Pain",
    "dyspnea": "Symptom:Dyspnea",
    "opacity": "ImagingFinding:Opacity",
    "nodule": "ImagingFinding:Nodule",
    "lesion": "ImagingFinding:Lesion",
    "pneumonia": "Pathology:Pneumonia",
    "infection": "Pathology:Infection",
    "smoking": "EpidemiologicalFactor:Smoking",
    "travel": "EpidemiologicalFactor:TravelExposure",
    "exposure": "EpidemiologicalFactor:Exposure",
    "ct": "ImagingStudy:CT",
    "mri": "ImagingStudy:MRI",
    "xray": "ImagingStudy:XR",
    "xr": "ImagingStudy:XR",
}


def _stable_id(*parts: Any) -> str:
    payload = ":".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class ClinicalDocumentChunk:
    chunk_id: str
    text: str
    source_path: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_document(self) -> dict[str, Any]:
        metadata = {
            "record_id": self.chunk_id,
            "source_path": self.source_path,
            "page": self.page,
            "section": self.section,
            "focus": self.metadata.get("focus", "document_rag"),
            **self.metadata,
        }
        return {
            "text": self.text,
            "modality": self.metadata.get("modality", "clinical_document_text"),
            "source": "clinical_document_processor",
            "learning_status": self.metadata.get("learning_status", "candidate"),
            "metadata": metadata,
        }


@dataclass(slots=True)
class ClinicalDocumentProcessor:
    """Document ingestion contract for literature, guidelines, PDFs and reports.

    Phase-2 adds enterprise RAG metadata: deterministic document ids, semantic
    section chunking, simple clinical entity/ontology extraction, source
    governance, license/publication metadata, and optional PHI-like redaction.
    The implementation remains dependency-free and Docling-aware for local tests
    and air-gapped execution.
    """

    parser_backend: str = "docling_recommended_with_plain_text_fallback"
    chunk_size: int = 1200
    chunk_overlap: int = 160
    redact_phi: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "parser_backend": self.parser_backend,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "recommended_parser": "Docling",
            "supported_target_inputs": ["pdf", "docx", "pptx", "html", "markdown", "images", "clinical_reports"],
            "fallback_mode": "plain_text_file_reader",
            "docling_available": self._docling_available()["available"],
            "phase2_enterprise_features": [
                "section_aware_chunking",
                "clinical_entity_extraction",
                "ontology_reference_metadata",
                "source_license_publication_tracking",
                "phi_like_redaction",
                "weaviate_ready_memory_documents",
            ],
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

    def document_id(self, source_path: str, text: str, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        stable_source = metadata.get("source_uri") or source_path
        return f"doc:{_stable_id(stable_source, text[:2048])}"

    def redact_text(self, text: str) -> tuple[str, list[str]]:
        redactions: list[str] = []
        if not self.redact_phi:
            return text, redactions
        redacted = _EMAIL_RE.sub(lambda _: redactions.append("email") or "[REDACTED_EMAIL]", text)
        redacted = _PHONE_RE.sub(lambda _: redactions.append("phone") or "[REDACTED_PHONE]", redacted)
        redacted = _DATE_RE.sub(lambda _: redactions.append("date") or "[REDACTED_DATE]", redacted)
        return redacted, redactions

    def extract_clinical_entities(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        ontology_refs = []
        entities = []
        for token, ref in sorted(_CLINICAL_TERMS.items()):
            if re.search(rf"\b{re.escape(token)}\b", lowered):
                ontology_refs.append(ref)
                category, _, label = ref.partition(":")
                entities.append({"text": token, "category": category, "normalized": label, "ontology_ref": ref})
        return {
            "clinical_entities": entities,
            "ontology_refs": sorted(set(ontology_refs)),
        }

    def infer_source_governance(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        source_type = metadata.get("source_type") or metadata.get("document_type") or "unknown"
        license_class = metadata.get("license") or metadata.get("license_class") or "unknown"
        publication_date = metadata.get("publication_date")
        provenance_quality = 0.2
        if metadata.get("source_uri") or metadata.get("source_path"):
            provenance_quality += 0.2
        if source_type != "unknown":
            provenance_quality += 0.2
        if license_class != "unknown":
            provenance_quality += 0.2
        if publication_date:
            provenance_quality += 0.2
        return {
            "source_type": source_type,
            "license": license_class,
            "publication_date": publication_date,
            "provenance_quality": round(_clamp(provenance_quality), 3),
            "governance_status": "complete" if provenance_quality >= 0.8 else "needs_review",
            "synthetic_source": source_type in {"synthetic", "dream_trace", "counterfactual"},
        }

    def split_sections(self, text: str) -> list[dict[str, Any]]:
        if not text.strip():
            return []
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [{"title": "body", "start": 0, "end": len(text), "text": text}]
        sections = []
        if matches[0].start() > 0:
            sections.append({"title": "front_matter", "start": 0, "end": matches[0].start(), "text": text[: matches[0].start()]})
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(0).lstrip("#").strip()
            body = text[start:end].strip()
            if body:
                sections.append({"title": title, "start": start, "end": end, "text": body})
        return sections or [{"title": "body", "start": 0, "end": len(text), "text": text}]

    def _chunk_section(self, section_text: str) -> Iterable[tuple[int, str]]:
        step = max(self.chunk_size - self.chunk_overlap, 1)
        for start in range(0, len(section_text), step):
            chunk_text = section_text[start : start + self.chunk_size].strip()
            if chunk_text:
                yield start, chunk_text

    def chunk_text(self, text: str, source_path: str, metadata: dict[str, Any] | None = None) -> list[ClinicalDocumentChunk]:
        metadata = metadata or {}
        chunks: list[ClinicalDocumentChunk] = []
        if not text:
            return chunks
        redacted_text, redactions = self.redact_text(text)
        doc_id = metadata.get("document_id") or self.document_id(source_path=source_path, text=redacted_text, metadata=metadata)
        governance = self.infer_source_governance({**metadata, "source_path": source_path})
        for section_index, section in enumerate(self.split_sections(redacted_text)):
            for chunk_index, (offset, chunk_text) in enumerate(self._chunk_section(section["text"])):
                extracted = self.extract_clinical_entities(chunk_text)
                chunk_id = f"{doc_id}:chunk:{section_index}:{chunk_index}"
                chunk_metadata = {
                    **metadata,
                    **governance,
                    **extracted,
                    "document_id": doc_id,
                    "chunk_index": len(chunks),
                    "section_index": section_index,
                    "section": section["title"],
                    "char_start": section["start"] + offset,
                    "char_end": section["start"] + offset + len(chunk_text),
                    "redacted": bool(redactions),
                    "redaction_types": sorted(set(redactions)),
                    "source_path": str(source_path),
                    "relations": [
                        {"from": chunk_id, "predicate": "mentions", "to": ref}
                        for ref in extracted["ontology_refs"]
                    ],
                }
                chunks.append(
                    ClinicalDocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source_path=str(source_path),
                        page=metadata.get("page"),
                        section=section["title"],
                        metadata=chunk_metadata,
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
                "document_id": chunks[0].metadata.get("document_id") if chunks else self.document_id(str(path), text, metadata),
                "chunk_count": len(chunks),
                "documents": [chunk.to_memory_document() for chunk in chunks],
                "governance": self.docling_integration_plan()["governance_requirements"],
                "enterprise_metadata": self.infer_source_governance({**metadata, **parser_metadata, "source_path": str(path)}),
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
        document_id = documents[0]["metadata"].get("document_id") if documents else self.document_id(str(path), "", metadata)
        return {
            "status": "completed",
            "parser": "plain_text_fallback",
            "source_path": str(path),
            "document_id": document_id,
            "chunk_count": len(documents),
            "documents": documents,
            "docling_result": docling_result,
            "governance": self.docling_integration_plan()["governance_requirements"],
            "enterprise_metadata": self.infer_source_governance({**metadata, "source_path": str(path)}),
        }

    def upsert_processed_document(self, processed: dict[str, Any], memory_adapter: Any) -> dict[str, Any]:
        documents = list(processed.get("documents", []))
        results = []
        for document in documents:
            if hasattr(memory_adapter, "upsert_clinical_document_chunk"):
                results.append(memory_adapter.upsert_clinical_document_chunk(document))
            elif hasattr(memory_adapter, "upsert_many"):
                memory_adapter.upsert_many([document])
                results.append({"status": "upserted_via_upsert_many", "record_id": document.get("metadata", {}).get("record_id")})
            elif hasattr(memory_adapter, "add_document"):
                memory_adapter.add_document(document)
                results.append({"status": "upserted_via_add_document", "record_id": document.get("metadata", {}).get("record_id")})
            else:
                results.append({"status": "not_upserted", "reason": "adapter_has_no_supported_upsert_method"})
        return {
            "status": "completed",
            "document_id": processed.get("document_id"),
            "attempted": len(documents),
            "stored": sum(1 for item in results if str(item.get("status", "")).startswith(("stored", "upserted"))),
            "results": results,
        }

    def docling_integration_plan(self) -> dict[str, Any]:
        return {
            "status": "phase2_enterprise_contract",
            "package": "docling",
            "intended_flow": [
                "DocumentConverter().convert(source).document",
                "export structured markdown/json",
                "preserve tables, formulas, reading order and page metadata",
                "chunk by clinical section and semantic boundaries",
                "extract clinical entities and ontology references",
                "track license, publication date, source type and provenance quality",
                "redact PHI-like patterns before vector upsert when configured",
                "upsert chunks into WeaviateEnterpriseMemoryAdapter or VectorMemoryStore",
            ],
            "governance_requirements": [
                "track source license and publication date",
                "separate peer-reviewed literature, guidelines, local protocols and synthetic traces",
                "retain page/section provenance for every RAG answer",
                "never promote dream-generated traces without rational-control validation",
                "mark incomplete provenance as needs_review",
                "preserve ontology_refs and relations for graph expansion",
            ],
        }
