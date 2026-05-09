from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def _tokenize(text: str) -> set[str]:
    return {token.strip(".,;:()[]{}!?\"'`)._").lower() for token in str(text).split() if token.strip(".,;:()[]{}!?\"'`)._")}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class RAGEvaluationRecord:
    query: str
    evidence: list[dict[str, Any]]
    answer: str = ""
    expected_terms: list[str] = field(default_factory=list)
    required_sources: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RAGEvaluationReport:
    sample_count: int
    context_precision: float
    context_recall: float
    faithfulness: float
    groundedness: float
    provenance_completeness: float
    citation_coverage: float
    mean_grounding_score: float
    noise_sensitivity: float
    records: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "groundedness": self.groundedness,
            "provenance_completeness": self.provenance_completeness,
            "citation_coverage": self.citation_coverage,
            "mean_grounding_score": self.mean_grounding_score,
            "noise_sensitivity": self.noise_sensitivity,
            "records": self.records,
        }


@dataclass(slots=True)
class RAGEvaluator:
    """Dependency-free RAG quality evaluator for Melampo Phase 2.

    The evaluator intentionally uses transparent lexical and metadata-based
    metrics so it can run in CI without external LLM judges. Production setups
    can add model-based judges behind the same report contract.
    """

    min_relevant_overlap: float = 0.15

    def evaluate_record(self, record: RAGEvaluationRecord | dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, dict):
            record = RAGEvaluationRecord(
                query=str(record.get("query", "")),
                evidence=list(record.get("evidence", [])),
                answer=str(record.get("answer", "")),
                expected_terms=list(record.get("expected_terms", [])),
                required_sources=list(record.get("required_sources", [])),
            )
        query_terms = _tokenize(record.query)
        answer_terms = _tokenize(record.answer)
        expected_terms = {term.lower() for term in record.expected_terms}
        evidence = [item for item in record.evidence if isinstance(item, dict)]
        overlaps = []
        grounding_scores = []
        provenance_complete = 0
        citations_present = 0
        relevant_count = 0
        noise_count = 0
        evidence_terms_all: set[str] = set()

        for item in evidence:
            text = str(item.get("text") or item.get("value") or item.get("summary") or "")
            terms = _tokenize(text)
            evidence_terms_all.update(terms)
            overlap = len(query_terms.intersection(terms)) / max(len(query_terms), 1) if query_terms else 0.0
            overlaps.append(overlap)
            if overlap >= self.min_relevant_overlap or expected_terms.intersection(terms):
                relevant_count += 1
            else:
                noise_count += 1
            grounding_scores.append(_safe_float(item.get("score_final", item.get("grounding_score", 0.0))))
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            provenance = item.get("provenance", {}) if isinstance(item.get("provenance", {}), dict) else metadata
            if provenance.get("source_path") or provenance.get("source_uri") or item.get("source"):
                provenance_complete += 1
            if metadata.get("page") is not None or metadata.get("section") or item.get("record_id"):
                citations_present += 1

        context_precision = relevant_count / max(len(evidence), 1)
        context_recall = len(expected_terms.intersection(evidence_terms_all)) / max(len(expected_terms), 1) if expected_terms else (max(overlaps) if overlaps else 0.0)
        answer_supported_terms = answer_terms.intersection(evidence_terms_all)
        faithfulness = len(answer_supported_terms) / max(len(answer_terms), 1) if answer_terms else 0.0
        groundedness = sum(grounding_scores) / max(len(grounding_scores), 1)
        source_hits = 0
        if record.required_sources:
            for required in record.required_sources:
                if any(required in str(item.get("source", "")) or required in str(item.get("metadata", {})) for item in evidence):
                    source_hits += 1
            citation_coverage = source_hits / max(len(record.required_sources), 1)
        else:
            citation_coverage = citations_present / max(len(evidence), 1)
        return {
            "query": record.query,
            "evidence_count": len(evidence),
            "context_precision": round(_clamp(context_precision), 3),
            "context_recall": round(_clamp(context_recall), 3),
            "faithfulness": round(_clamp(faithfulness), 3),
            "groundedness": round(_clamp(groundedness), 3),
            "provenance_completeness": round(provenance_complete / max(len(evidence), 1), 3),
            "citation_coverage": round(_clamp(citation_coverage), 3),
            "mean_grounding_score": round(_clamp(groundedness), 3),
            "noise_sensitivity": round(_clamp(noise_count / max(len(evidence), 1)), 3),
        }

    def evaluate(self, records: Iterable[RAGEvaluationRecord | dict[str, Any]]) -> RAGEvaluationReport:
        evaluated = [self.evaluate_record(record) for record in records]
        sample_count = len(evaluated)

        def mean(key: str) -> float:
            return round(sum(item[key] for item in evaluated) / max(sample_count, 1), 3)

        return RAGEvaluationReport(
            sample_count=sample_count,
            context_precision=mean("context_precision"),
            context_recall=mean("context_recall"),
            faithfulness=mean("faithfulness"),
            groundedness=mean("groundedness"),
            provenance_completeness=mean("provenance_completeness"),
            citation_coverage=mean("citation_coverage"),
            mean_grounding_score=mean("mean_grounding_score"),
            noise_sensitivity=mean("noise_sensitivity"),
            records=evaluated,
        )

    @staticmethod
    def enterprise_thresholds(report: RAGEvaluationReport, min_context_precision: float = 0.6, min_provenance: float = 0.8) -> dict[str, Any]:
        failures = []
        if report.context_precision < min_context_precision:
            failures.append("context_precision_below_threshold")
        if report.provenance_completeness < min_provenance:
            failures.append("provenance_completeness_below_threshold")
        if report.faithfulness < 0.5:
            failures.append("faithfulness_below_threshold")
        if report.noise_sensitivity > 0.4:
            failures.append("noise_sensitivity_above_threshold")
        return {
            "status": "pass" if not failures else "needs_review",
            "failures": failures,
            "thresholds": {
                "min_context_precision": min_context_precision,
                "min_provenance": min_provenance,
                "min_faithfulness": 0.5,
                "max_noise_sensitivity": 0.4,
            },
            "observed": report.as_dict(),
        }
