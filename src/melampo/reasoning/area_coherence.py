from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .neuro_dynamics import NeuroDynamicMetrics


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_:-]{3,}")
_POSITIVE_POLARITIES = {"present", "positive", "observed", "detected", "suspected", "supports", "increased"}
_NEGATIVE_POLARITIES = {"absent", "negative", "negated", "ruled_out", "excluded", "decreased", "denies"}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def _tokens_from_text(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text or "")}


def _claim_entity(claim: dict[str, Any]) -> str | None:
    refs = claim.get("ontology_refs") or claim.get("ontology_ref") or claim.get("code")
    if isinstance(refs, str) and refs:
        return refs.casefold()
    if isinstance(refs, list) and refs:
        return str(refs[0]).casefold()
    entity = claim.get("normalized_entity") or claim.get("entity") or claim.get("label") or claim.get("name")
    return str(entity).casefold() if entity else None


def _claim_polarity(claim: dict[str, Any]) -> str:
    raw = str(claim.get("polarity") or claim.get("status") or claim.get("assertion") or "present").casefold()
    if raw in _NEGATIVE_POLARITIES:
        return "absent"
    if raw in _POSITIVE_POLARITIES:
        return "present"
    return "uncertain"


@dataclass
class AreaCoherenceAnalyzer:
    """Compute dynamic coherence, mismatch and neuro-inspired interaction metrics.

    Phase-1/P0 upgrade: pair status is no longer a static lookup table. The
    canonical area-pair priors still encode expected Melampo interactions, but
    the final status is driven by structured claims, ontology refs, semantic
    overlap, evidence sufficiency, precision and explicit contradictions.
    """

    metrics: NeuroDynamicMetrics = field(default_factory=NeuroDynamicMetrics)

    COHERENT_PAIRS = {
        _canonical_pair("language_listening", "visual_diagnostic"),
        _canonical_pair("epidemiology", "visual_diagnostic"),
        _canonical_pair("case_context", "language_listening"),
        _canonical_pair("case_context", "epidemiology"),
        _canonical_pair("language_listening", "epidemiology"),
        _canonical_pair("case_context", "visual_diagnostic"),
    }

    def _area_state(self, payload: dict[str, Any]) -> dict[str, float]:
        salience = _clamp(_safe_float(payload.get("salience_score", 0.0)))
        uncertainty = _clamp(_safe_float(payload.get("uncertainty_score", 1.0 - salience)))
        signal_count = max(_safe_int(payload.get("signal_count", len(payload))), 0)
        precision = _clamp(salience / max(salience + uncertainty + 0.001, 0.001))
        return {
            "salience": salience,
            "uncertainty": uncertainty,
            "signal_count": float(signal_count),
            "precision": precision,
        }

    def _claims(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        claims = payload.get("claims", [])
        if isinstance(claims, list):
            return [claim for claim in claims if isinstance(claim, dict)]
        return []

    def _collect_terms(self, value: Any, depth: int = 0) -> set[str]:
        if depth > 3:
            return set()
        if value is None:
            return set()
        if isinstance(value, str):
            return _tokens_from_text(value)
        if isinstance(value, (int, float, bool)):
            return {str(value).casefold()}
        if isinstance(value, list | tuple | set):
            terms: set[str] = set()
            for item in value:
                terms.update(self._collect_terms(item, depth + 1))
            return terms
        if isinstance(value, dict):
            terms: set[str] = set()
            for key, item in value.items():
                key_text = str(key).casefold()
                if key_text not in {"metadata", "provenance", "source_path", "series_paths"}:
                    terms.update(_tokens_from_text(key_text))
                if key_text in {
                    "area",
                    "focus",
                    "salient_streams",
                    "ontology_refs",
                    "normalized_entity",
                    "entity",
                    "name",
                    "label",
                    "description",
                    "merged_text",
                    "text",
                    "report_text",
                    "ehr_text",
                    "patient_complaints",
                    "signals",
                    "exposures",
                    "findings",
                    "risk_factors",
                    "demographics",
                    "pathology",
                    "volume",
                    "patient_visual",
                    "labs_snapshot",
                }:
                    terms.update(self._collect_terms(item, depth + 1))
            return terms
        return _tokens_from_text(str(value))

    def _claim_terms(self, payload: dict[str, Any]) -> set[str]:
        terms: set[str] = set()
        for claim in self._claims(payload):
            entity = _claim_entity(claim)
            if entity:
                terms.update(_tokens_from_text(entity))
            refs = claim.get("ontology_refs", [])
            if isinstance(refs, list):
                for ref in refs:
                    terms.update(_tokens_from_text(str(ref)))
            for key in ("type", "polarity", "status", "evidence"):
                if key in claim:
                    terms.update(_tokens_from_text(str(claim[key])))
        return terms

    def _terms_for_area(self, payload: dict[str, Any]) -> set[str]:
        terms = self._collect_terms(payload)
        terms.update(self._claim_terms(payload))
        return {term for term in terms if len(term) >= 3}

    def _explicit_claim_contradiction(self, first: dict[str, Any], second: dict[str, Any]) -> float:
        first_claims: dict[str, set[str]] = {}
        second_claims: dict[str, set[str]] = {}
        for claim in self._claims(first):
            entity = _claim_entity(claim)
            if entity:
                first_claims.setdefault(entity, set()).add(_claim_polarity(claim))
        for claim in self._claims(second):
            entity = _claim_entity(claim)
            if entity:
                second_claims.setdefault(entity, set()).add(_claim_polarity(claim))

        contradiction = 0.0
        for entity in set(first_claims).intersection(second_claims):
            polarities = first_claims[entity].union(second_claims[entity])
            if "present" in polarities and "absent" in polarities:
                contradiction = max(contradiction, 1.0)
            elif "uncertain" in polarities and len(polarities) > 1:
                contradiction = max(contradiction, 0.45)
        return contradiction

    def _pair_profile(self, name: str, other: str, first: dict[str, Any], second: dict[str, Any], total_salience: float) -> dict[str, Any]:
        pair = _canonical_pair(name, other)
        first_state = self._area_state(first)
        second_state = self._area_state(second)
        first_terms = self._terms_for_area(first)
        second_terms = self._terms_for_area(second)
        intersection = first_terms.intersection(second_terms)
        semantic_overlap = _clamp(len(intersection) / math.sqrt(max(len(first_terms) * len(second_terms), 1)))
        prior_expected = 1.0 if pair in self.COHERENT_PAIRS else 0.0
        focus_alignment = bool(first.get("focus") == second.get("focus"))
        confidence_alignment = _clamp(1.0 - abs(first_state["precision"] - second_state["precision"]))
        min_precision = min(first_state["precision"], second_state["precision"])
        pair_signal_count = int(first_state["signal_count"] + second_state["signal_count"])
        pair_salience = round(first_state["salience"] + second_state["salience"], 3)
        if pair_salience <= 0.0 and pair_signal_count > 0:
            pair_salience = round(min(1.0, pair_signal_count * 0.05), 3)

        explicit_contradiction = self._explicit_claim_contradiction(first, second)
        uncertainty_conflict = _clamp(((first_state["uncertainty"] + second_state["uncertainty"]) / 2.0) * (1.0 - semantic_overlap) * 0.35)
        missing_evidence_score = _clamp(max(0.0, 1.0 - pair_signal_count / 3.0) * 0.55 + max(0.0, 0.42 - min_precision) * 0.45)
        focus_disagreement_penalty = 0.12 if not focus_alignment and semantic_overlap < 0.12 and prior_expected < 1.0 else 0.0

        agreement_score = _clamp(
            semantic_overlap * 0.38
            + (0.18 if focus_alignment else 0.0)
            + prior_expected * 0.18
            + confidence_alignment * 0.16
            + min_precision * 0.10
            - explicit_contradiction * 0.45
            - missing_evidence_score * 0.08
        )
        contradiction_score = _clamp(explicit_contradiction + uncertainty_conflict + focus_disagreement_penalty)
        dynamic_mismatch_score = _clamp(
            contradiction_score * 0.55
            + missing_evidence_score * 0.25
            + max(0.0, 0.45 - agreement_score) * 0.20
        )
        status = "coherent" if agreement_score >= 0.34 and contradiction_score < 0.50 and missing_evidence_score < 0.85 else "mismatch"

        return {
            "pair": pair,
            "status": status,
            "status_detail": "dynamic_content_alignment" if status == "coherent" else "dynamic_mismatch_or_insufficient_evidence",
            "pair_salience": pair_salience,
            "pair_signal_count": pair_signal_count,
            "focus_alignment": focus_alignment,
            "first_focus": first.get("focus", name),
            "second_focus": second.get("focus", other),
            "semantic_overlap": round(semantic_overlap, 3),
            "agreement_score": round(agreement_score, 3),
            "contradiction_score": round(contradiction_score, 3),
            "missing_evidence_score": round(missing_evidence_score, 3),
            "dynamic_mismatch_score": round(dynamic_mismatch_score, 3),
            "prior_expected_interaction": bool(prior_expected),
            "shared_terms": sorted(intersection)[:12],
            "first_precision": round(first_state["precision"], 3),
            "second_precision": round(second_state["precision"], 3),
            "confidence_alignment": round(confidence_alignment, 3),
            "normalized_pair_salience": round(_clamp(pair_salience / max(total_salience + 1.0, 1.0)), 3),
        }

    def analyze(self, area_signals: dict, dream_pressure: float = 0.0) -> dict:
        names = sorted(area_signals.keys())
        coherence_pairs: list[tuple[str, str]] = []
        mismatch_pairs: list[tuple[str, str]] = []
        pair_profiles: list[dict[str, Any]] = []
        total_salience = 0.0
        for payload in area_signals.values():
            if isinstance(payload, dict):
                total_salience += _safe_float(payload.get("salience_score", 0.0))

        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                first = area_signals.get(name, {}) if isinstance(area_signals.get(name, {}), dict) else {}
                second = area_signals.get(other, {}) if isinstance(area_signals.get(other, {}), dict) else {}
                profile = self._pair_profile(name=name, other=other, first=first, second=second, total_salience=total_salience)
                pair_profiles.append(profile)
                if profile["status"] == "coherent":
                    coherence_pairs.append(profile["pair"])
                else:
                    mismatch_pairs.append(profile["pair"])

        weighted_total = sum(max(_safe_float(profile.get("pair_salience", 0.0)), 0.1) for profile in pair_profiles) or 1.0
        weighted_agreement = sum(_safe_float(profile.get("agreement_score", 0.0)) * max(_safe_float(profile.get("pair_salience", 0.0)), 0.1) for profile in pair_profiles) / weighted_total
        weighted_mismatch = sum(_safe_float(profile.get("dynamic_mismatch_score", 0.0)) * max(_safe_float(profile.get("pair_salience", 0.0)), 0.1) for profile in pair_profiles) / weighted_total
        coherence_score = round(_clamp(weighted_agreement + min(total_salience * 0.04, 0.15)), 3)
        mismatch_score = round(_clamp(weighted_mismatch), 3)
        neuro_dynamic_metrics = self.metrics.compute(
            pair_profiles=pair_profiles,
            coherence_score=coherence_score,
            mismatch_score=mismatch_score,
            total_salience=total_salience,
            dream_pressure=dream_pressure,
            area_signals=area_signals,
        )
        return {
            "coherence_pairs": coherence_pairs,
            "mismatch_pairs": mismatch_pairs,
            "pair_profiles": pair_profiles,
            "coherence_score": coherence_score,
            "mismatch_score": mismatch_score,
            "total_salience": round(total_salience, 3),
            "dynamic_pair_status": True,
            "neuro_dynamic_metrics": neuro_dynamic_metrics,
            "pi_score": neuro_dynamic_metrics["pi_score"],
            "prediction_error": neuro_dynamic_metrics["prediction_error"],
            "precision_weighted_coherence": neuro_dynamic_metrics["precision_weighted_coherence"],
            "convergence_index": neuro_dynamic_metrics["convergence_index"],
            "mismatch_index": neuro_dynamic_metrics["mismatch_index"],
            "deductive_gate": neuro_dynamic_metrics["deductive_gate"],
            "revision_pressure": neuro_dynamic_metrics["revision_pressure"],
            "dream_plasticity": neuro_dynamic_metrics["dream_plasticity"],
            "interdependence_index": neuro_dynamic_metrics["interdependence_index"],
            "evidence_integration_score": neuro_dynamic_metrics["evidence_integration_score"],
            "noise_suppression_score": neuro_dynamic_metrics["noise_suppression_score"],
            "action_potential_gate": neuro_dynamic_metrics["action_potential_gate"],
            "deep_inference_score": neuro_dynamic_metrics["deep_inference_score"],
            "deductive_stability": neuro_dynamic_metrics["deductive_stability"],
        }
