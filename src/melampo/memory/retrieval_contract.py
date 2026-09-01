"""Shared retrieval contract for one-shot and recursive retrieval strategies.

``MemoryRetriever.retrieve`` already returns a stable payload shape across all
of its branches. Formalising that shape here lets a recursive strategy join the
pipeline as an additional ``retrieval_mode`` rather than as a rewrite of the
reasoning layer, and gives both strategies a single validator.

The validator exists because the failure modes in this contract are silent. A
strategy that forgets to declare ``memory_backed`` is charged a fallback penalty
in the downstream grounding computation without raising anything; a strategy
that omits provenance offsets is blocked by the safety rails at the far end of
the pipeline, far from the cause. Both are cheap to catch here.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

RETRIEVAL_MODE_SEMANTIC = "semantic_vector_memory"
RETRIEVAL_MODE_EMPTY = "empty_memory_no_fallback"
RETRIEVAL_MODE_FALLBACK = "fallback_contract_only"
RETRIEVAL_MODE_RLM = "rlm_environment"
RETRIEVAL_MODE_DUAL = "dual_path_reconciled"

GROUNDED_MODES = frozenset({RETRIEVAL_MODE_SEMANTIC, RETRIEVAL_MODE_RLM, RETRIEVAL_MODE_DUAL})

COVERAGE_BASIS_TOPK = "topk_ratio"
COVERAGE_BASIS_CORPUS = "corpus_characters"
COVERAGE_BASIS_NONE = "not_applicable"

DEFAULT_COVERAGE_BASIS_BY_MODE = {
    RETRIEVAL_MODE_SEMANTIC: COVERAGE_BASIS_TOPK,
    RETRIEVAL_MODE_EMPTY: COVERAGE_BASIS_NONE,
    RETRIEVAL_MODE_FALLBACK: COVERAGE_BASIS_NONE,
    RETRIEVAL_MODE_RLM: COVERAGE_BASIS_CORPUS,
    RETRIEVAL_MODE_DUAL: COVERAGE_BASIS_CORPUS,
}

REQUIRED_KEYS = (
    "query",
    "focus",
    "target_areas",
    "status",
    "retrieval_mode",
    "evidence",
    "evidence_count",
    "retrieval_quality",
)

REQUIRED_QUALITY_KEYS = ("memory_backed", "coverage", "mean_grounding_score", "fallback_used")


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Structural contract every retrieval strategy must satisfy.

    Mirrors the existing ``MemoryRetriever.retrieve`` signature so one-shot and
    recursive strategies remain interchangeable at the call site.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        required_status: Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
        case_context: dict[str, Any] | None = None,
        target_areas: list[str] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ContractViolation:
    code: str
    detail: str


def validate_retrieval_payload(payload: dict[str, Any], *, require_provenance: bool = True) -> list[ContractViolation]:
    """Return contract violations for a retrieval payload. Empty list means valid."""
    violations: list[ContractViolation] = []

    for key in REQUIRED_KEYS:
        if key not in payload:
            violations.append(ContractViolation("missing_key", f"payload is missing '{key}'"))

    evidence = payload.get("evidence")
    if evidence is not None and not isinstance(evidence, list):
        violations.append(ContractViolation("invalid_evidence", "'evidence' must be a list"))
        evidence = None

    if isinstance(evidence, list) and payload.get("evidence_count") != len(evidence):
        violations.append(
            ContractViolation("evidence_count_mismatch", "'evidence_count' does not match len(evidence)")
        )

    quality = payload.get("retrieval_quality")
    if not isinstance(quality, dict):
        violations.append(ContractViolation("missing_quality", "'retrieval_quality' must be a dict"))
        quality = {}

    for key in REQUIRED_QUALITY_KEYS:
        if key not in quality:
            violations.append(ContractViolation("missing_quality_key", f"retrieval_quality is missing '{key}'"))

    mode = payload.get("retrieval_mode")
    if mode in GROUNDED_MODES and quality.get("memory_backed") is not True:
        violations.append(
            ContractViolation(
                "undeclared_memory_backing",
                f"mode '{mode}' must declare memory_backed=True or it silently incurs the fallback penalty",
            )
        )

    if require_provenance and isinstance(evidence, list):
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                violations.append(ContractViolation("invalid_evidence_item", f"evidence[{index}] is not a dict"))
                continue
            if not _has_trace(item):
                violations.append(
                    ContractViolation(
                        "untraceable_evidence",
                        f"evidence[{index}] has no record_id, page, section or character offsets",
                    )
                )

    return violations


def assert_retrieval_contract(payload: dict[str, Any], *, require_provenance: bool = True) -> None:
    """Raise ``ValueError`` when the payload violates the retrieval contract."""
    violations = validate_retrieval_payload(payload, require_provenance=require_provenance)
    if violations:
        detail = "; ".join(f"{item.code}: {item.detail}" for item in violations)
        raise ValueError(f"retrieval contract violation -> {detail}")


def coverage_basis(payload: dict[str, Any]) -> str:
    """Return the semantics under which this payload's coverage was computed.

    ``coverage`` means different things per strategy. One-shot retrieval reports
    ``len(evidence) / top_k``: a selection ratio. Recursive retrieval reports the
    fraction of the case corpus actually inspected. The two are not the same
    quantity, and the one-shot definition actively penalises a strategy that
    explores widely and reports selectively, which is the behaviour recursive
    retrieval is expected to exhibit.
    """
    quality = payload.get("retrieval_quality") if isinstance(payload, dict) else None
    if isinstance(quality, dict) and quality.get("coverage_basis"):
        return str(quality["coverage_basis"])
    return DEFAULT_COVERAGE_BASIS_BY_MODE.get(str(payload.get("retrieval_mode")), COVERAGE_BASIS_NONE)


def assert_coverage_comparable(*payloads: dict[str, Any]) -> str:
    """Raise when coverage figures from different bases would be compared.

    A/B evaluation across retrieval strategies is only meaningful when the
    compared quantities share a definition. Without this guard the comparison
    silently produces a number, which is worse than failing.
    """
    bases = {coverage_basis(payload) for payload in payloads}
    bases.discard(COVERAGE_BASIS_NONE)
    if len(bases) > 1:
        raise ValueError(
            "coverage figures use incompatible bases and cannot be compared directly: " + ", ".join(sorted(bases))
        )
    return bases.pop() if bases else COVERAGE_BASIS_NONE


def _has_trace(item: dict[str, Any]) -> bool:
    if item.get("record_id"):
        return True
    provenance = item.get("provenance")
    if isinstance(provenance, dict):
        if provenance.get("page") or provenance.get("section"):
            return True
        start = provenance.get("char_start")
        end = provenance.get("char_end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            return True
    metadata = item.get("metadata")
    return isinstance(metadata, dict) and bool(metadata.get("page") or metadata.get("section"))
