from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    left_norm = math.sqrt(sum(value * value for value in left[:length])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:length])) or 1.0
    return _clamp(sum(left[index] * right[index] for index in range(length)) / (left_norm * right_norm))


def _stable_vector(seed: str, dimensions: int = 64) -> list[float]:
    buckets = [0.0 for _ in range(dimensions)]
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).digest()
    for index in range(dimensions * 2):
        byte = digest[index % len(digest)]
        buckets[(byte + index) % dimensions] += ((byte % 29) + 1) / 29.0
    return _normalize(buckets)


def _matrix_to_vector(matrix: Any, fallback_seed: str, dimensions: int = 64) -> list[float]:
    values: list[float] = []

    def collect(value: Any, depth: int = 0) -> None:
        if depth > 4 or len(values) >= dimensions * 4:
            return
        if isinstance(value, int | float):
            values.append(float(value))
            return
        if isinstance(value, str):
            values.extend(_stable_vector(value, dimensions=min(8, dimensions)))
            return
        if isinstance(value, dict):
            for key in sorted(value):
                collect(key, depth + 1)
                collect(value[key], depth + 1)
            return
        if isinstance(value, list | tuple | set):
            for item in value:
                collect(item, depth + 1)

    collect(matrix)
    if not values:
        return _stable_vector(fallback_seed, dimensions=dimensions)
    buckets = [0.0 for _ in range(dimensions)]
    for index, value in enumerate(values):
        buckets[index % dimensions] += math.tanh(value)
    return _normalize(buckets)


def _hash_vector(vector: list[float]) -> str:
    payload = ",".join(f"{value:.6f}" for value in vector)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_STOP_TERMS = {"and", "or", "of", "the", "a", "an", "with", "without", "di", "del", "della", "dei", "e", "o"}


def _concept_terms(concept: str) -> set[str]:
    return {term for term in _TOKEN_RE.findall(concept.casefold()) if len(term) >= 3 and term not in _STOP_TERMS}


def _semantic_relation(left: "VisualRecognitionImprint", right: "VisualRecognitionImprint") -> dict[str, Any]:
    if left.semantic_concept == right.semantic_concept:
        return {
            "score": 1.0,
            "match_type": "total_semantic_concept",
            "shared_terms": sorted(_concept_terms(left.semantic_concept)),
            "shared_ontology_refs": sorted(set(left.ontology_refs).intersection(right.ontology_refs)),
        }
    left_terms = _concept_terms(left.semantic_concept)
    right_terms = _concept_terms(right.semantic_concept)
    shared_terms = left_terms.intersection(right_terms)
    ontology_overlap = set(left.ontology_refs).intersection(right.ontology_refs)
    containment = len(shared_terms) / max(min(len(left_terms), len(right_terms)), 1) if shared_terms else 0.0
    jaccard = len(shared_terms) / max(len(left_terms.union(right_terms)), 1) if shared_terms else 0.0
    ontology_score = 1.0 if ontology_overlap else 0.0
    score = _clamp(max(containment * 0.78 + jaccard * 0.22, ontology_score))
    if ontology_overlap and shared_terms:
        match_type = "partial_semantic_and_ontology_overlap"
    elif ontology_overlap:
        match_type = "ontology_overlap"
    elif shared_terms:
        match_type = "partial_semantic_concept"
    else:
        match_type = "unrelated_semantic_concept"
    return {
        "score": round(score, 3),
        "match_type": match_type,
        "shared_terms": sorted(shared_terms),
        "shared_ontology_refs": sorted(ontology_overlap),
    }


@dataclass(frozen=True, slots=True)
class VisualRecognitionImprint:
    """Semantic visual footprint for an image-recognition matrix or embedding.

    The imprint is a governed vector/matrix signature associated with a clinical
    semantic concept. It is not an image and is not a diagnosis; it is a
    searchable, auditable memory object for multimodal RAG and dream replay.
    """

    imprint_id: str
    semantic_concept: str
    variant_label: str
    vector: list[float]
    source_object_id: str = "unknown"
    modality: str = "imaging"
    salience: float = 0.0
    uncertainty: float = 1.0
    ontology_refs: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    learning_status: str = "candidate"
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, default_concept: str = "visual_pattern") -> "VisualRecognitionImprint":
        payload = payload if isinstance(payload, dict) else {}
        concept = str(payload.get("semantic_concept") or payload.get("concept") or payload.get("normalized_entity") or default_concept)
        variant_label = str(payload.get("variant_label") or payload.get("label") or payload.get("name") or "observed_variant")
        source_object_id = str(payload.get("source_object_id") or payload.get("study_id") or payload.get("object_id") or "unknown")
        vector = payload.get("vector") or payload.get("embedding") or payload.get("matrix_signature") or payload.get("recognition_matrix")
        dense_vector = _matrix_to_vector(vector, fallback_seed=f"{concept}:{variant_label}:{source_object_id}")
        imprint_id = str(payload.get("imprint_id") or f"vimprint:{_hash_vector(dense_vector)}")
        return cls(
            imprint_id=imprint_id,
            semantic_concept=concept.casefold().strip() or default_concept,
            variant_label=variant_label,
            vector=dense_vector,
            source_object_id=source_object_id,
            modality=str(payload.get("modality", "imaging")),
            salience=_clamp(_safe_float(payload.get("salience", payload.get("salience_score", 0.0)))),
            uncertainty=_clamp(_safe_float(payload.get("uncertainty", payload.get("uncertainty_score", 1.0)))),
            ontology_refs=[str(ref) for ref in payload.get("ontology_refs", [])] if isinstance(payload.get("ontology_refs", []), list) else [],
            provenance=dict(payload.get("provenance", {})) if isinstance(payload.get("provenance", {}), dict) else {},
            learning_status=str(payload.get("learning_status", "candidate")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "imprint_id": self.imprint_id,
            "semantic_concept": self.semantic_concept,
            "variant_label": self.variant_label,
            "source_object_id": self.source_object_id,
            "modality": self.modality,
            "matrix_signature_hash": _hash_vector(self.vector),
            "vector": list(self.vector),
            "salience": round(_clamp(self.salience), 3),
            "uncertainty": round(_clamp(self.uncertainty), 3),
            "ontology_refs": list(self.ontology_refs),
            "provenance": dict(self.provenance),
            "learning_status": self.learning_status,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class VisualImprintBuilder:
    """Build governed visual imprints from Melampo visual-area signals."""

    def from_visual_area(self, signal: dict[str, Any], volume_features: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        signal = signal if isinstance(signal, dict) else {}
        volume_features = volume_features if isinstance(volume_features, dict) else {}
        claims = signal.get("claims", []) if isinstance(signal.get("claims", []), list) else []
        concept_payloads = [
            (
                str(claim.get("normalized_entity") or claim.get("label")),
                claim.get("ontology_refs", []) if isinstance(claim.get("ontology_refs", []), list) else [],
            )
            for claim in claims
            if isinstance(claim, dict) and (claim.get("normalized_entity") or claim.get("label"))
        ]
        if not concept_payloads:
            concept_payloads = [(str(volume_features.get("input_kind") or signal.get("focus") or "diagnostic_visual_pattern"), [])]
        imprints = []
        for index, (concept, ontology_refs) in enumerate(concept_payloads[:4], start=1):
            imprint = VisualRecognitionImprint.from_payload(
                {
                    "semantic_concept": concept,
                    "variant_label": f"observed_visual_variant_{index}",
                    "source_object_id": volume_features.get("study_id", "unknown_study"),
                    "modality": volume_features.get("metadata", {}).get("modality", volume_features.get("input_kind", "imaging")) if isinstance(volume_features.get("metadata", {}), dict) else volume_features.get("input_kind", "imaging"),
                    "recognition_matrix": {
                        "volume": volume_features,
                        "visual_signal": signal,
                        "concept": concept,
                    },
                    "salience": signal.get("salience_score", 0.0),
                    "uncertainty": signal.get("uncertainty_score", 1.0),
                    "ontology_refs": ontology_refs,
                    "provenance": {
                        "source": "visual_diagnostic_area",
                        "hidden_network_call": False,
                        "research_use_only": True,
                    },
                }
            )
            imprints.append(imprint.as_dict())
        return imprints


@dataclass(slots=True)
class VisualImprintMorpher:
    """Dream-safe morphing of visual recognition imprints in vector space.

    Morphing is deterministic interpolation of matrix footprints that share a
    semantic concept. It does not synthesize clinical images and never promotes
    generated associations to clinical truth.
    """

    interpolation_alpha: float = 0.5
    min_similarity: float = 0.35
    min_semantic_overlap: float = 0.34

    def _as_imprints(self, payloads: list[dict[str, Any]] | None) -> list[VisualRecognitionImprint]:
        return [VisualRecognitionImprint.from_payload(payload) for payload in payloads or [] if isinstance(payload, dict)]

    def dream_morph(
        self,
        concept_imprints: list[dict[str, Any]] | None,
        diagnostic_imprints: list[dict[str, Any]] | None = None,
        area_dynamics: dict[str, Any] | None = None,
        limit: int = 6,
    ) -> dict[str, Any]:
        source_imprints = self._as_imprints(concept_imprints)
        diagnostic = self._as_imprints(diagnostic_imprints) or source_imprints
        area_dynamics = area_dynamics if isinstance(area_dynamics, dict) else {}
        neuro = area_dynamics.get("neuro_dynamic_metrics", {}) if isinstance(area_dynamics.get("neuro_dynamic_metrics", {}), dict) else {}
        pi_score = _safe_float(neuro.get("pi_score", area_dynamics.get("pi_score", 0.0)))
        prediction_error = _safe_float(neuro.get("prediction_error", area_dynamics.get("prediction_error", 0.0)))
        mismatch_index = _safe_float(neuro.get("mismatch_index", area_dynamics.get("mismatch_index", 0.0)))
        dream_plasticity = _safe_float(neuro.get("dream_plasticity", 0.0))
        action_gate = _safe_float(neuro.get("action_potential_gate", 0.0))

        morphs: list[dict[str, Any]] = []
        semantic_links: list[dict[str, Any]] = []
        alpha_base = _clamp(self.interpolation_alpha + (dream_plasticity - mismatch_index) * 0.1, 0.2, 0.8)
        for left_index, left in enumerate(source_imprints):
            for right in source_imprints[left_index + 1 :]:
                relation = _semantic_relation(left, right)
                semantic_relation_score = float(relation["score"])
                if semantic_relation_score < self.min_semantic_overlap:
                    continue
                alpha = _clamp(
                    alpha_base
                    + (right.salience - left.salience) * 0.05
                    + (left.uncertainty - right.uncertainty) * 0.05,
                    0.2,
                    0.8,
                )
                bridge_seed = " ".join(
                    [left.semantic_concept, right.semantic_concept, *relation["shared_terms"], *relation["shared_ontology_refs"]]
                )
                bridge_vector = _stable_vector(bridge_seed or f"{left.semantic_concept}:{right.semantic_concept}", dimensions=min(len(left.vector), len(right.vector)))
                bridge_gain = _clamp(semantic_relation_score * 0.18 + action_gate * 0.05 - prediction_error * 0.04, 0.0, 0.28)
                morphed_vector = _normalize([
                    (1.0 - alpha) * left.vector[index]
                    + alpha * right.vector[index]
                    + bridge_gain * bridge_vector[index]
                    for index in range(min(len(left.vector), len(right.vector), len(bridge_vector)))
                ])
                source_similarity = _cosine(left.vector, right.vector)
                related_targets = [
                    item
                    for item in diagnostic
                    if _semantic_relation(left, item)["score"] >= self.min_semantic_overlap
                    or _semantic_relation(right, item)["score"] >= self.min_semantic_overlap
                ] or diagnostic
                best_target = max(related_targets, key=lambda item: _cosine(morphed_vector, item.vector), default=None)
                target_similarity = _cosine(morphed_vector, best_target.vector) if best_target else 0.0
                target_relation = _semantic_relation(left, best_target) if best_target else {"score": 0.0, "match_type": "none", "shared_terms": [], "shared_ontology_refs": []}
                right_target_relation = _semantic_relation(right, best_target) if best_target else {"score": 0.0}
                target_semantic_score = max(float(target_relation.get("score", 0.0)), float(right_target_relation.get("score", 0.0)))
                inference_weight = _clamp(
                    semantic_relation_score * 0.30
                    + target_semantic_score * 0.22
                    + source_similarity * 0.16
                    + target_similarity * 0.14
                    + pi_score * 0.08
                    + action_gate * 0.06
                    + dream_plasticity * 0.04
                    - prediction_error * 0.10
                    - mismatch_index * 0.08
                )
                interference = _clamp(
                    (source_similarity * 0.45 + semantic_relation_score * 0.35 + target_semantic_score * 0.20)
                    * (0.55 + action_gate * 0.25 + dream_plasticity * 0.20)
                    - prediction_error * 0.12
                )
                intuitive_link_score = _clamp(
                    target_similarity * 0.28
                    + semantic_relation_score * 0.22
                    + target_semantic_score * 0.14
                    + inference_weight * 0.14
                    + source_similarity * 0.10
                    + interference * 0.08
                    + pi_score * 0.08
                    + action_gate * 0.06
                    - mismatch_index * 0.10
                )
                if relation["match_type"] == "total_semantic_concept":
                    semantic_concept = left.semantic_concept
                else:
                    shared = " ".join(relation["shared_terms"]) or " / ".join([left.semantic_concept, right.semantic_concept])
                    semantic_concept = f"partial:{shared}"
                morph = {
                    "morph_id": f"vmorph:{_hash_vector(morphed_vector)}",
                    "semantic_concept": semantic_concept,
                    "left_semantic_concept": left.semantic_concept,
                    "right_semantic_concept": right.semantic_concept,
                    "left_imprint_id": left.imprint_id,
                    "right_imprint_id": right.imprint_id,
                    "target_imprint_id": best_target.imprint_id if best_target else "none",
                    "morphing_mode": "inferential_semantic_matrix_morphing",
                    "semantic_match_type": relation["match_type"],
                    "semantic_relation_score": round(semantic_relation_score, 3),
                    "target_semantic_relation_score": round(target_semantic_score, 3),
                    "shared_semantic_terms": relation["shared_terms"],
                    "shared_ontology_refs": relation["shared_ontology_refs"],
                    "interpolation_alpha": round(alpha, 3),
                    "bridge_gain": round(bridge_gain, 3),
                    "source_similarity": round(source_similarity, 3),
                    "target_similarity": round(target_similarity, 3),
                    "inference_weight": round(inference_weight, 3),
                    "interference_score": round(interference, 3),
                    "intuitive_link_score": round(intuitive_link_score, 3),
                    "matrix_signature_hash": _hash_vector(morphed_vector),
                    "concept_bridge_hash": _hash_vector(bridge_vector),
                    "vector": morphed_vector,
                    "learning_status": "candidate",
                    "clinical_status": "research_hypothesis_only",
                }
                morphs.append(morph)
                if intuitive_link_score >= self.min_similarity:
                    semantic_links.append({
                        "semantic_concept": semantic_concept,
                        "morph_id": morph["morph_id"],
                        "target_imprint_id": morph["target_imprint_id"],
                        "relationship": "dream_morphed_visual_imprint_suggests_diagnostic_correlation",
                        "semantic_match_type": relation["match_type"],
                        "semantic_relation_score": round(semantic_relation_score, 3),
                        "score": round(intuitive_link_score, 3),
                        "requires_review": True,
                    })

        morphs.sort(key=lambda item: item["intuitive_link_score"], reverse=True)
        semantic_links.sort(key=lambda item: item["score"], reverse=True)
        top_score = morphs[0]["intuitive_link_score"] if morphs else 0.0
        return {
            "status": "completed",
            "operation": "visual_imprint_dream_morphing",
            "source_imprint_count": len(source_imprints),
            "diagnostic_imprint_count": len(diagnostic),
            "morph_count": len(morphs),
            "visual_morph_candidates": morphs[:limit],
            "semantic_links": semantic_links[:limit],
            "visual_morph_coherence": round(top_score, 3),
            "visual_prediction_link_score": round(top_score, 3),
            "visual_morph_intuition_gain": round(_clamp(top_score * 0.55 + dream_plasticity * 0.25 + action_gate * 0.20 - prediction_error * 0.15), 3),
            "neuroquantum_trace": {
                "formalism": "quantum_like_latent_interference_not_physical_quantum_claim",
                "morphing_mode": "inferential_semantic_matrix_morphing",
                "supports_total_or_partial_semantic_concepts": True,
                "alpha_base": round(alpha_base, 3),
                "pi_score": round(pi_score, 3),
                "prediction_error": round(prediction_error, 3),
                "mismatch_index": round(mismatch_index, 3),
                "dream_plasticity": round(dream_plasticity, 3),
                "action_potential_gate": round(action_gate, 3),
            },
            "governance": {
                "hidden_network_call": False,
                "does_not_generate_clinical_images": True,
                "morphs_total_or_partial_semantic_concepts": True,
                "synthetic_morphs_are_candidate_only": True,
                "automatic_clinical_promotion_allowed": False,
                "human_review_before_clinical_use": True,
            },
        }
