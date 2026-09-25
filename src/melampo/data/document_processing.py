from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 /,:;()\-]{5,})$", re.MULTILINE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# The same separator on both sides ("12/09/2026", "12.09.2026", "12-09-2026"),
# never mixed: a mixed form would also catch a laboratory range written
# "4.0-10.0" ("4.0-10") and redact it as a date.
_DATE_RE = re.compile(r"\b\d{1,2}([/.\-])\d{1,2}\1\d{2,4}\b")

# Phone numbers: see _is_phone_number() for why a long run of digits alone is
# not enough. A candidate is digit groups joined by exactly ONE separator
# (space, dot, hyphen), optionally led by "+" and/or a parenthesised prefix:
# a laboratory table's column alignment (two or more spaces) and a spaced
# range (" - ") both break a candidate apart instead of gluing two values
# into one "number".
_PHONE_CANDIDATE_RE = re.compile(r"(?<![\w.,+])(?:\+ ?)?(?:\(\d{1,5}\) ?)?\d+(?:[ .\-]\d+)*(?!\w)")
_PHONE_KEYWORD_BEFORE_RE = re.compile(
    r"(?i)\b(?:tel|telefono|cell|cellulare|fax|phone|mobile|recapito)\b\.?:?\s*$"
)
# A dot followed by 1-3 digits that end the group ("13.5", "0.50", "11.8 10.5"):
# a decimal value, not a dotted phone group ("06.1234.5678" keeps 4-digit groups).
_DECIMAL_GROUP_RE = re.compile(r"\.\d{1,3}(?=[ \-]|$)")
# (3, 3, 3) deliberately absent: a legacy 9-digit mobile written that way is
# rare, while three single-spaced 3-digit values ("312 250 198") are common.
_MOBILE_GROUPINGS = {(9,), (10,), (3, 6), (3, 7), (3, 3, 4), (3, 4, 3), (3, 3, 2, 2)}


def _is_phone_number(candidate: str, preceding_text: str) -> bool:
    """Whether a digit-group candidate is a phone number, by Italian numbering-plan shape or explicit context.

    Replaces a rule that redacted ANY run of 9+ digit-or-separator
    characters as a phone. Verified defect before this fix: in a laboratory
    report that rule erased reference ranges ("12.0 - 16.0" ->
    "[REDACTED_PHONE]") and even values, gluing a result to the next
    column ("11.8   10^3/uL" -> "[REDACTED_PHONE]^3/uL") -- clinical data
    destroyed silently in the text that becomes the case's report_text.

    A candidate is a phone only when one of these holds:
    - it starts with "+" (international form) and has 8-15 digits;
    - it follows an explicit phone word ("Tel.", "Cell:", "Fax", ...) and
      has 6-15 digits;
    - it starts with 0 (every Italian geographic number, and the "00"
      international prefix) and has 8-15 digits;
    - it starts with 3 and has 9-10 digits (Italian mobile shape).
    And it never looks like a decimal value (see _DECIMAL_GROUP_RE).

    Accepted, stated trade-off: a bare number with no "+", no phone word
    and neither a 0- nor 3-prefix (e.g. a toll-free "800 123456" printed
    without "Tel.") is not redacted. Laboratory values start with any digit
    and are common in exactly the documents this processes; a phone number
    printed without any of those markers is rare in Italian clinical
    documents. The previous rule's opposite trade-off destroyed data.
    """
    if _DECIMAL_GROUP_RE.search(candidate):
        return False
    digits = re.sub(r"\D", "", candidate)
    if not 6 <= len(digits) <= 15:
        return False
    if candidate.startswith("+"):
        return len(digits) >= 8
    if _PHONE_KEYWORD_BEFORE_RE.search(preceding_text):
        return True
    groups = tuple(len(group) for group in re.findall(r"\d+", candidate))
    if digits.startswith("0"):
        # An Italian prefix is "0" plus 1-3 digits (06, 02, 0571) or the "00"
        # international prefix -- never a lone "0", which is how a value
        # series would start ("0 12 34 56").
        return len(digits) >= 8 and groups[0] >= 2
    if digits.startswith("3"):
        # Only the ways an Italian mobile is actually written: contiguous, or
        # a 3-digit operator code then the rest. "312 250 198" (a series of
        # values) has the right digit count but not the shape.
        return groups in _MOBILE_GROUPINGS
    return False
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


FORMAT_PDF = "pdf"
FORMAT_PNG = "png"
FORMAT_JPEG = "jpeg"
FORMAT_DICOM = "dicom"
FORMAT_TEXT = "text"
FORMAT_UNKNOWN = "unknown"

_MIME_BY_FORMAT = {FORMAT_PNG: "image/png", FORMAT_JPEG: "image/jpeg"}


def detect_document_format(data: bytes) -> str:
    """The document's real format, from its own leading bytes -- never from a filename.

    A filename can lie (a JPEG saved as "referto.pdf", a DICOM file with
    no extension at all, which is common for files exported from a PACS);
    the file's own signature cannot. Standard, published signatures only:
    PDF begins with "%PDF"; PNG with the fixed 8-byte signature; JPEG with
    FF D8 FF; DICOM Part 10 files carry a 128-byte preamble followed by the
    literal "DICM" at offset 128. Anything else that decodes cleanly as
    UTF-8 is text; everything remaining is unknown -- and is never decoded
    as text anyway, which is exactly the failure this function exists to
    prevent (see process_document_bytes).
    """
    if data.startswith(b"%PDF"):
        return FORMAT_PDF
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return FORMAT_PNG
    if data.startswith(b"\xff\xd8\xff"):
        return FORMAT_JPEG
    if len(data) >= 132 and data[128:132] == b"DICM":
        return FORMAT_DICOM
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return FORMAT_UNKNOWN
    return FORMAT_TEXT


def _extract_pdf_text_layer(data: bytes) -> str | None:
    """A digital PDF's own embedded text layer, via poppler's pdftotext, entirely in memory -- None if the tool is unavailable.

    Many clinical documents (laboratory reports especially) are generated
    digitally and carry a real, selectable text layer -- readable without
    any OCR or vision model at all. pdftotext reads the PDF from stdin and
    writes text to stdout ("-" for both), so no temporary file is ever
    written, verified directly before relying on it. A scanned PDF has no
    text layer and returns an empty string here -- honestly nothing, not
    something invented.
    """
    import subprocess

    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", "-", "-"], input=data, capture_output=True, timeout=60, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _parse_failure_types() -> tuple[type[BaseException], ...]:
    """The failures a document parser is genuinely expected to hit -- caught and reported as "failed", never anything wider.

    Network errors (requests), an input the parser cannot take or a
    malformed response (ValueError), unreadable bytes/files (OSError), and
    poppler/pdf2image's own errors (missing binaries, unparseable PDF,
    timeout). Deliberately not bare `Exception`: that would also swallow a
    genuine programming error -- a TypeError, a KeyError from a refactor --
    and report it as "parser unavailable", hiding exactly the kind of
    defect this project wants surfaced, not smoothed over.
    """
    import requests
    from pdf2image import exceptions as pdf2image_exceptions

    return (
        requests.RequestException,
        ValueError,
        OSError,
        pdf2image_exceptions.PDFInfoNotInstalledError,
        pdf2image_exceptions.PDFPageCountError,
        pdf2image_exceptions.PDFSyntaxError,
        pdf2image_exceptions.PDFPopplerTimeoutError,
    )


def _parse_nemotron_parse_response(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The model's raw output, from the documented response envelope.

    Verified against NVIDIA's own published API docs: a Nemotron-Parse
    response is an OpenAI-style chat completion, with the parsed page
    content in `response["choices"][0]["message"]["content"]`.

    **What is not verified, stated plainly rather than guessed at**:
    NVIDIA's own worked examples post-process that content through a
    `postprocessing` helper module (`extract_classes_bboxes`,
    `transform_bbox_to_original`, `postprocess_text`) to recover
    structured classes, bounding boxes and clean text from the model's
    layout markup -- that helper's exact grammar was not published
    alongside the API reference this was built from, so it is not
    replicated here. This returns the raw message content as the page's
    text (usable as-is for chunking, entity extraction and ontology
    matching, which is everything downstream of this function actually
    needs) and a minimal metadata dict noting that bounding-box
    structure was not decoded. A deployment that needs page-element
    bounding boxes specifically should verify the live endpoint's exact
    output grammar and extend this function accordingly -- guessing at
    a proprietary format here would risk silently corrupting text no
    differently than the pdfplumber/Pydantic issue a comparable project
    reported hitting with a different parser under the same pressure to
    guess rather than verify.
    """
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"unexpected Nemotron-Parse response shape: {error}") from error
    return str(content), {"bounding_boxes_decoded": False}


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
    The implementation remains dependency-free when no parser is configured,
    falling back to plain text -- the same posture Docling held before it was
    replaced.

    Docling was removed rather than kept as a fallback option. A direct
    comparison found no evidence it was ever the strongest choice for clinical
    documents specifically: one 2026 assessment calls it "weaker on complex
    layouts" against current leaders, while a clinical-document-specific
    comparison names a different tool "the benchmark solution" for exactly
    this domain. Keeping Docling wired in as an unused option would have left
    a reference to a decision this project no longer holds.

    Two parsers are supported instead, matching the deployment-mode decision
    already made for the vetting engine (see docs/recursive_engine_decision_record.md):
    Nemotron-Parse is the default because it is the one genuinely on-premise
    option -- open weights, no vendor API dependency -- while a hosted
    alternative (LlamaParse) is checked separately, since its own vendor
    documentation states it does not offer true on-premise deployment (VPC is
    the closest equivalent), and it is used only where cloud deployment is
    already acceptable and its documented strength on clinical tables and
    mixed formatting is worth the trade. Running both and comparing their
    output is the double-reading-at-ingestion principle already agreed for
    this project: an ingestion error is more costly than a navigation error,
    because it propagates silently into everything read afterwards.
    """

    parser_backend: str = "nemotron_parse_recommended_with_plain_text_fallback"
    chunk_size: int = 1200
    chunk_overlap: int = 160
    redact_phi: bool = True
    concept_resolver: Any = None
    language: str = "en"
    nemotron_parse_endpoint: str | None = None
    nemotron_parse_api_key: str | None = None
    llamaparse_api_key: str | None = None

    @classmethod
    def from_env(cls) -> ClinicalDocumentProcessor:
        """Configured from the environment -- the only way a deployment turns Nemotron-Parse on.

        Found missing while wiring attachments into ingestion: nothing in the
        project ever read an endpoint or key from anywhere, so even a
        deployment with a live Nemotron-Parse NIM would never have used it.
        Unset variables leave the parser unconfigured, which degrades exactly
        as before (digital-PDF text layer, honest "no_text_extracted").
        """
        import os

        return cls(
            nemotron_parse_endpoint=os.environ.get("NEMOTRON_PARSE_ENDPOINT") or None,
            nemotron_parse_api_key=os.environ.get("NEMOTRON_PARSE_API_KEY") or None,
            llamaparse_api_key=os.environ.get("LLAMAPARSE_API_KEY") or None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "parser_backend": self.parser_backend,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "extraction_mode": "lexicon" if self.concept_resolver is None else "ontology_index",
            "extraction_language": self.language,
            "recommended_parser": "Nemotron-Parse",
            "cross_check_parser": "LlamaParse (cloud deployments only -- no true on-premise mode)",
            "supported_target_inputs": ["pdf", "docx", "pptx", "html", "markdown", "images", "clinical_reports"],
            "fallback_mode": "plain_text_file_reader",
            "nemotron_parse_available": self._nemotron_parse_available()["available"],
            "llamaparse_available": self._llamaparse_available()["available"],
            "phase2_enterprise_features": [
                "section_aware_chunking",
                "clinical_entity_extraction",
                "ontology_reference_metadata",
                "source_license_publication_tracking",
                "phi_like_redaction",
                "weaviate_ready_memory_documents",
            ],
        }

    def _nemotron_parse_available(self) -> dict[str, Any]:
        """Whether a Nemotron-Parse endpoint is configured.

        Called via HTTP against a NIM endpoint or an OpenRouter-style
        provider, the same pattern the model-comparison bench scripts use for
        every other candidate in this project -- not a local pip package, so
        availability is a matter of configuration (endpoint and key), not of
        whether a library happens to be installed.
        """
        endpoint = self.nemotron_parse_endpoint
        key = self.nemotron_parse_api_key
        if not endpoint or not key:
            return {"available": False, "error": "nemotron_parse_endpoint_or_key_not_configured"}
        return {"available": True, "error": None}

    def _llamaparse_available(self) -> dict[str, Any]:
        """Whether a LlamaParse API key is configured.

        Cloud-only by the vendor's own documentation (VPC is the closest
        on-premise equivalent) -- used as an optional cross-check parser
        where cloud deployment is already accepted, never as the sole or
        default parser for a deployment that requires genuine on-premise
        operation.
        """
        key = self.llamaparse_api_key
        if not key:
            return {"available": False, "error": "llamaparse_api_key_not_configured"}
        return {"available": True, "error": None}

    def load_text_fallback(self, path: str | Path) -> str:
        path = Path(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def load_with_nemotron_parse(self, path: str | Path) -> dict[str, Any]:
        """Convert a document with Nemotron-Parse when an endpoint is configured.

        Returns a structured result instead of raising when unavailable --
        the same graceful-degradation contract the removed Docling path held,
        so callers that already handle a "not_executed" status need no
        change.
        """
        availability = self._nemotron_parse_available()
        if not availability["available"]:
            return {
                "status": "not_executed",
                "reason": "nemotron_parse_unavailable",
                "error": availability["error"],
                "source_path": str(path),
            }
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            return {"status": "failed", "reason": "nemotron_parse_conversion_failed", "error": str(exc), "source_path": str(path)}
        return self.load_with_nemotron_parse_bytes(data, source_name=str(path))

    def load_with_nemotron_parse_bytes(self, data: bytes, source_name: str = "<memory>") -> dict[str, Any]:
        """load_with_nemotron_parse(), from bytes already in memory -- same result contract, same graceful degradation."""
        availability = self._nemotron_parse_available()
        if not availability["available"]:
            return {
                "status": "not_executed",
                "reason": "nemotron_parse_unavailable",
                "error": availability["error"],
                "source_path": source_name,
            }
        try:
            text, layout_metadata = self._call_nemotron_parse_bytes(data, source_name=source_name)
        except _parse_failure_types() as exc:
            return {
                "status": "failed",
                "reason": "nemotron_parse_conversion_failed",
                "error": str(exc),
                "source_path": source_name,
            }
        return {
            "status": "completed",
            "source_path": source_name,
            "text": text,
            "parser": "nemotron_parse",
            "metadata": {
                "parser": "nemotron_parse",
                "source_path": source_name,
                "layout_preserved": True,
                **layout_metadata,
            },
        }

    def load_with_llamaparse(self, path: str | Path) -> dict[str, Any]:
        """Convert a document with LlamaParse when an API key is configured.

        For the cross-check path only (see the class docstring) -- never the
        sole parser for a deployment requiring genuine on-premise operation,
        since LlamaParse's own documentation offers VPC as its closest
        equivalent, not true on-premise.
        """
        availability = self._llamaparse_available()
        if not availability["available"]:
            return {
                "status": "not_executed",
                "reason": "llamaparse_unavailable",
                "error": availability["error"],
                "source_path": str(path),
            }
        try:
            text, layout_metadata = self._call_llamaparse(path)
        except (NotImplementedError, *_parse_failure_types()) as exc:  # pragma: no cover - depends on the live endpoint/files
            return {
                "status": "failed",
                "reason": "llamaparse_conversion_failed",
                "error": str(exc),
                "source_path": str(path),
            }
        return {
            "status": "completed",
            "source_path": str(path),
            "text": text,
            "parser": "llamaparse",
            "metadata": {"parser": "llamaparse", "source_path": str(path), "layout_preserved": True, **layout_metadata},
        }

    def _call_nemotron_parse(self, path: str | Path) -> tuple[str, dict[str, Any]]:
        """The real HTTP call to a Nemotron-Parse-v2.0 NIM endpoint.

        Nemotron-Parse is a vision-language model, not a text-in parser: it
        takes a page rendered as an image through the standard OpenAI-style
        `/v1/chat/completions` contract (`image_url` content block, base64
        data URL) and returns text with embedded layout markup in
        `response["choices"][0]["message"]["content"]`. Verified against
        NVIDIA's own published Nemotron-Parse-v2.0 API documentation before
        writing this, not assumed from the model's name.

        v2.0 upgrade, requested and verified directly, not assumed from the
        version number alone: a real NVIDIA release (build.nvidia.com/nvidia/nemotron-parse-2.0),
        not v1.2 renamed. Compared with v1.2 it adds a ~20k-token vocabulary
        expansion for multilingual OCR -- directly relevant here, an
        Italian-language deployment -- plus chart-aware parsing. It is also a
        genuinely different request contract, not a drop-in model-string
        swap: v2.0's release notes state plainly that free-text prompts are
        not supported at all -- every request needs an image AND a task
        prompt built from its own control tokens
        (`<predict_bbox><predict_classes><output_markdown>...`), confirmed
        against the self-hosted NIM's own published curl example. v1.2's
        prior prompt here ("Parse this document page: extract all text...")
        would not have been a valid v2.0 request.

        A real, documented divergence exists between the hosted NVIDIA
        Build endpoint (model `nvidia/nemotron-parse`) and the self-hosted
        NIM (`nemotron-parse-v2.0`): they expect different request
        contracts, and sending a self-hosted-style request to the hosted
        endpoint can return HTTP 400 ("model does not support text input").
        This targets the self-hosted NIM contract specifically -- consistent
        with why Nemotron-Parse was chosen over LlamaParse in the first
        place (open weights, no vendor API dependency, genuine on-premise
        operation) -- so a deployment pointed at the hosted Build endpoint
        instead should expect to adjust the payload shape.

        PDF pages are rendered to images via pdf2image (poppler) before the
        call, since the model has no raw-PDF input path. Non-PDF image
        inputs are sent directly.
        """
        return self._call_nemotron_parse_bytes(Path(path).read_bytes(), source_name=str(path))

    def _call_nemotron_parse_bytes(self, data: bytes, source_name: str = "<memory>") -> tuple[str, dict[str, Any]]:
        """The same Nemotron-Parse call, from bytes already in memory -- the path version above now just reads and delegates here."""
        images = self._render_bytes_as_images(data, source_name=source_name)
        if not images:
            raise ValueError(f"no renderable pages found for {source_name}")

        page_texts: list[str] = []
        page_metadata: list[dict[str, Any]] = []
        for page_number, (image_bytes, mime_type) in enumerate(images, start=1):
            response = self._post_nemotron_parse_page(image_bytes, mime_type=mime_type)
            text, layout = _parse_nemotron_parse_response(response)
            page_texts.append(text)
            page_metadata.append({"page": page_number, **layout})
        return "\n\n".join(page_texts), {"page_count": len(images), "pages": page_metadata}

    def _render_pages_as_images(self, path: Path) -> list[tuple[bytes, str]]:
        """Kept for callers that still hold a path -- reads once, delegates to the in-memory version."""
        return self._render_bytes_as_images(Path(path).read_bytes(), source_name=str(path))

    def _render_bytes_as_images(self, data: bytes, source_name: str = "<memory>") -> list[tuple[bytes, str]]:
        """Each page as (image bytes, MIME type) -- pdf2image for PDFs, the bytes themselves for an image, all in memory.

        Format comes from detect_document_format() (the file's own
        signature), not a filename suffix. Returns the real MIME type with
        each page: an earlier version labelled every image
        "data:image/png" regardless of content, so a JPEG was sent to
        Nemotron-Parse mislabelled as PNG -- never surfaced only because
        nothing had ever called this module with a real JPEG.
        """
        document_format = detect_document_format(data)
        if document_format in _MIME_BY_FORMAT:
            return [(data, _MIME_BY_FORMAT[document_format])]
        if document_format != FORMAT_PDF:
            raise ValueError(f"Nemotron-Parse needs an image or PDF input, got {document_format!r} for {source_name}")
        from io import BytesIO

        from pdf2image import convert_from_bytes

        rendered = []
        for page_image in convert_from_bytes(data, dpi=200):
            buffer = BytesIO()
            page_image.save(buffer, format="PNG")
            rendered.append((buffer.getvalue(), "image/png"))
        return rendered

    def _post_nemotron_parse_page(self, image_bytes: bytes, mime_type: str = "image/png") -> dict[str, Any]:
        """One page, one request -- the isolated network call, mocked directly in tests.

        The task prompt is v2.0's own required control-token string, copied
        verbatim from NVIDIA's self-hosted-NIM curl example (build.nvidia.com/nvidia/nemotron-parse-2.0/deploy),
        not a free-text instruction: v2.0's release notes state that
        text-only or freely-worded prompts are not a supported input shape.
        <predict_text_in_pic> is included deliberately -- extraction should
        not silently skip embedded text NVIDIA's default example leaves out
        (<predict_no_text_in_pic>), the opposite of what a clinical document
        parser needs.
        """
        import base64

        import requests

        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": "nvidia/nemotron-parse-v2.0",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                    ],
                }
            ],
        }
        response = requests.post(
            f"{self.nemotron_parse_endpoint.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.nemotron_parse_api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def _call_llamaparse(self, path: str | Path) -> tuple[str, dict[str, Any]]:  # pragma: no cover - network call
        raise NotImplementedError("configure llamaparse_api_key and implement the HTTP call for this deployment")

    def document_id(self, source_path: str, text: str, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        stable_source = metadata.get("source_uri") or source_path
        return f"doc:{_stable_id(stable_source, text[:2048])}"

    def redact_text(self, text: str) -> tuple[str, list[str]]:
        redactions: list[str] = []
        if not self.redact_phi:
            return text, redactions

        def mark(label: str, replacement: str):
            def replace(_: re.Match[str]) -> str:
                redactions.append(label)
                return replacement

            return replace

        def phone(match: re.Match[str]) -> str:
            preceding = match.string[max(0, match.start() - 20) : match.start()]
            if _is_phone_number(match.group(0), preceding):
                redactions.append("phone")
                return "[REDACTED_PHONE]"
            return match.group(0)

        # Dates before phones: a dotted date starting with 0 ("03.10.2026")
        # would otherwise match the phone shape and be labelled as one.
        redacted = _EMAIL_RE.sub(mark("email", "[REDACTED_EMAIL]"), text)
        redacted = _DATE_RE.sub(mark("date", "[REDACTED_DATE]"), redacted)
        redacted = _PHONE_CANDIDATE_RE.sub(phone, redacted)
        return redacted, redactions

    def extract_clinical_entities(self, text: str) -> dict[str, Any]:
        """Extract clinical concepts from a chunk.

        With a ``concept_resolver`` configured, extraction is driven by the
        ontology index: tens of thousands of surface forms rather than sixteen
        hand-written tokens, with modifiers attached to their findings and each
        mention carrying how it is asserted.

        Two lists are returned because they answer different questions.
        ``clinical_entities`` holds every mention, including negated ones — a
        chunk stating "denies fever" *should* be retrievable when searching for
        fever, since the negation is what a reader needs to find.
        ``patient_findings`` holds only what passed the findings boundary, and
        it is that list, never the first, that supplies graph entry points.

        Without a resolver the original lexicon is used unchanged, so existing
        ingestion behaviour is preserved.
        """
        if self.concept_resolver is None:
            return self._extract_with_lexicon(text)
        return self._extract_with_index(text)

    def _extract_with_lexicon(self, text: str) -> dict[str, Any]:
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
            "extraction_mode": "lexicon",
        }

    def _extract_with_index(self, text: str) -> dict[str, Any]:
        from ..memory.assertion import AssertionDetector, select_cues
        from ..memory.concept_resolution import attach_modifiers
        from ..reasoning.findings_boundary import assemble

        resolved = self.concept_resolver.resolve_text(text)
        extraction = attach_modifiers(resolved)
        detector = AssertionDetector(cues=select_cues(self.language))

        entities: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for finding in extraction.findings:
            concept = finding.concept
            assertion = detector.detect(text, concept.char_start or 0, concept.char_end or 0)
            modifiers = [item.label for item in finding.modifiers]
            entities.append(
                {
                    "text": concept.surface,
                    "category": "Phenotype",
                    "normalized": concept.label,
                    "ontology_ref": concept.term_id,
                    "match_kind": concept.match_kind,
                    "verified_match": concept.is_verified_match,
                    "modifiers": modifiers,
                    "char_start": concept.char_start,
                    "char_end": concept.char_end,
                    "assertion": assertion.as_dict(),
                }
            )
            candidates.append(
                {
                    "label": concept.label,
                    "term_id": concept.term_id,
                    "assertion": assertion,
                    "modifiers": modifiers,
                    "char_start": concept.char_start,
                    "char_end": concept.char_end,
                }
            )

        findings = assemble(candidates)
        return {
            "clinical_entities": entities,
            "ontology_refs": sorted({item["ontology_ref"] for item in entities}),
            "patient_findings": [item.as_dict() for item in findings.admitted],
            "excluded_mentions": [item.as_dict() for item in findings.rejected],
            "collapsed_modifiers": [item.label for item in extraction.collapsed_modifiers],
            "inheritance_statements": [item.label for item in extraction.inheritance_statements],
            "extraction_mode": "ontology_index",
            "extraction_language": self.language,
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
            "synthetic_source": source_type in {"synthetic", "nexus_trace", "counterfactual"},
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

    def process_document(
        self, path: str | Path, metadata: dict[str, Any] | None = None,
        prefer_structured_parser: bool = True, also_cross_check_with_llamaparse: bool = False,
    ) -> dict[str, Any]:
        """Parse a document from a path on disk -- reads it once, then delegates to process_document_bytes().

        ``also_cross_check_with_llamaparse`` runs the second reading agreed
        for ingestion specifically -- an ingestion error is more costly than
        a navigation error, since it propagates silently into everything
        read afterwards -- and is off by default because it requires a
        cloud-acceptable deployment; turning it on where only on-premise
        Nemotron-Parse is available would just report the second parser
        unavailable on every call.
        """
        metadata = metadata or {}
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            return {
                "status": "failed",
                "parser": "plain_text_fallback",
                "source_path": str(path),
                "error": str(exc),
                "parser_result": {"status": "not_executed", "reason": "file_unreadable"},
            }
        result = self.process_document_bytes(
            data, source_name=str(path), metadata=metadata, prefer_structured_parser=prefer_structured_parser
        )
        if also_cross_check_with_llamaparse:
            result["cross_check"] = self.load_with_llamaparse(path)
        return result

    def process_document_bytes(
        self, data: bytes, source_name: str = "<memory>", metadata: dict[str, Any] | None = None,
        prefer_structured_parser: bool = True,
    ) -> dict[str, Any]:
        """Parse a document held in memory -- never written to disk, per the decision to keep a case's files in memory during processing.

        Order, per document format (detected from the bytes themselves):
        1. Nemotron-Parse, when preferred and configured -- PDF, PNG, JPEG.
        2. Otherwise an honest format-specific fallback: UTF-8 text decoded
           as text; a digital PDF's own embedded text layer via pdftotext
           (_extract_pdf_text_layer); nothing else.
        3. Everything that yields no genuine text -- an image or scanned
           PDF with no OCR available, a DICOM file (its own handler is a
           separate component), an unrecognised binary -- returns status
           "no_text_extracted" with the reason, never "completed".

        Step 3 corrects a real defect, verified before fixing: the previous
        plain-text fallback decoded ANY file as UTF-8 with errors ignored,
        so a photographed blood test with Nemotron-Parse unconfigured came
        back as status "completed" with the JPEG's own binary header
        ("JFIF...") presented as clinical text -- silent garbage flowing
        into a case's report_text, the worst kind of ingestion error.
        """
        metadata = metadata or {}
        document_format = detect_document_format(data)
        if document_format == FORMAT_DICOM:
            return self._process_dicom_bytes(data, source_name, metadata)
        parser_result = (
            self.load_with_nemotron_parse_bytes(data, source_name=source_name)
            if prefer_structured_parser
            else {"status": "not_requested"}
        )

        if parser_result.get("status") == "completed":
            raw_metadata: Any = parser_result.get("metadata", {})
            parser_metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            return self._completed_document_result(
                str(parser_result.get("text", "")), source_name, metadata, "nemotron_parse", document_format,
                parser_metadata=parser_metadata,
            )

        if document_format == FORMAT_TEXT:
            return self._completed_document_result(
                data.decode("utf-8"), source_name, metadata, "plain_text_fallback", document_format,
                parser_result=parser_result,
            )

        if document_format == FORMAT_PDF:
            text_layer = _extract_pdf_text_layer(data)
            if text_layer:
                return self._completed_document_result(
                    text_layer, source_name, metadata, "pdf_text_layer", document_format, parser_result=parser_result,
                )
            reason = "pdftotext_unavailable" if text_layer is None else "pdf_has_no_text_layer_and_no_ocr_available"
        elif document_format in _MIME_BY_FORMAT:
            reason = "image_requires_nemotron_parse_for_text"
        else:
            reason = "unrecognised_binary_format"

        return {
            "status": "no_text_extracted",
            "parser": None,
            "source_path": source_name,
            "document_format": document_format,
            "reason": reason,
            "chunk_count": 0,
            "documents": [],
            "parser_result": parser_result,
            "governance": self.ingestion_integration_plan()["governance_requirements"],
            "enterprise_metadata": self.infer_source_governance({**metadata, "source_path": source_name}),
        }

    def _process_dicom_bytes(self, data: bytes, source_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """A DICOM file: report text (SR / encapsulated PDF / CDA) as the document text, rendered images alongside.

        Images are returned as `dicom_images_png` for the imaging side of
        the case (ImagingStudy), never folded into text. A DICOM with images
        but no report is "no_text_extracted" with reason "dicom_image_only"
        -- honest about there being no text, while still carrying the
        images; one that cannot be decoded at all says so, with pydicom's
        own reason in `dicom.notes`.
        """
        from .dicom_handler import extract_dicom

        extraction = extract_dicom(data, processor=self)
        dicom_summary = extraction.as_dict()
        dicom_metadata = {**metadata, "dicom_modality": extraction.modality}
        if extraction.report_text:
            result = self._completed_document_result(
                extraction.report_text, source_name, dicom_metadata, f"dicom_{extraction.report_source}", FORMAT_DICOM,
            )
        else:
            result = {
                "status": "no_text_extracted",
                "parser": None,
                "source_path": source_name,
                "document_format": FORMAT_DICOM,
                "reason": "dicom_image_only" if extraction.images_png else "dicom_not_decodable",
                "chunk_count": 0,
                "documents": [],
                "governance": self.ingestion_integration_plan()["governance_requirements"],
                "enterprise_metadata": self.infer_source_governance({**dicom_metadata, "source_path": source_name}),
            }
        result["dicom"] = dicom_summary
        result["dicom_images_png"] = list(extraction.images_png)
        return result

    def _completed_document_result(
        self, text: str, source_name: str, metadata: dict[str, Any], parser: str, document_format: str,
        *, parser_metadata: dict[str, Any] | None = None, parser_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The one "completed" result shape every successful parser path returns.

        `text` is the whole document, redacted exactly as its chunks are,
        read once -- added because callers that needed the document's text
        (case attachments, a DICOM's encapsulated PDF report) rebuilt it by
        joining the chunks, and chunks overlap by `chunk_overlap` characters
        by design (for retrieval). Verified defect: every overlap came back
        twice -- 12 duplicated lines in a ~3,900-character document -- and
        in a laboratory report a duplicated line reads as a repeated
        measurement. Chunks stay as they were, for memory/RAG only.
        """
        combined_metadata = {**metadata, **(parser_metadata or {}), "parser": parser}
        chunks = self.chunk_text(text=text, source_path=source_name, metadata=combined_metadata)
        redacted_text, _ = self.redact_text(text)
        result = {
            "status": "completed",
            "parser": parser,
            "source_path": source_name,
            "document_format": document_format,
            "text": redacted_text,
            "document_id": chunks[0].metadata.get("document_id") if chunks else self.document_id(source_name, text, metadata),
            "chunk_count": len(chunks),
            "documents": [chunk.to_memory_document() for chunk in chunks],
            "governance": self.ingestion_integration_plan()["governance_requirements"],
            "enterprise_metadata": self.infer_source_governance({**metadata, **(parser_metadata or {}), "source_path": source_name}),
        }
        if parser_result is not None:
            result["parser_result"] = parser_result
        return result

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

    def ingestion_integration_plan(self) -> dict[str, Any]:
        return {
            "status": "phase2_enterprise_contract",
            "package": "nemotron_parse",
            "cross_check_package": "llamaparse",
            "intended_flow": [
                "call the configured Nemotron-Parse endpoint on the source document",
                "export structured markdown/json with bounding boxes and semantic classes",
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
                "never promote nexus-generated traces without rational-control validation",
                "mark incomplete provenance as needs_review",
                "preserve ontology_refs and relations for graph expansion",
            ],
        }
