from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass(slots=True)
class DatasetManifest:
    """Governed dataset manifest for Phase 5A validation workflows.

    This manifest is intentionally metadata-only. It never loads clinical data,
    and it does not claim that a dataset is fit for clinical use. It records the
    minimum evidence needed before a benchmark or prospective protocol can be
    interpreted as research governance rather than an ad-hoc test run.
    """

    dataset_id: str
    name: str
    source: str
    license: str
    intended_use: str = "research_only"
    modalities: list[str] = field(default_factory=list)
    population: dict[str, Any] = field(default_factory=dict)
    label_schema: dict[str, Any] = field(default_factory=dict)
    gold_standard: str = "unknown"
    deidentified: bool = False
    splits: dict[str, int] = field(default_factory=dict)
    required_slices: list[str] = field(default_factory=list)
    bias_notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    governance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_id=_clean_string(payload.get("dataset_id", payload.get("id", "unknown_dataset"))),
            name=_clean_string(payload.get("name", "unknown_dataset")),
            source=_clean_string(payload.get("source", "")),
            license=_clean_string(payload.get("license", "")),
            intended_use=_clean_string(payload.get("intended_use", "research_only")) or "research_only",
            modalities=[_clean_string(item) for item in _as_list(payload.get("modalities")) if _clean_string(item)],
            population=dict(payload.get("population", {})),
            label_schema=dict(payload.get("label_schema", {})),
            gold_standard=_clean_string(payload.get("gold_standard", "unknown")) or "unknown",
            deidentified=bool(payload.get("deidentified", False)),
            splits=dict(payload.get("splits", {})),
            required_slices=[_clean_string(item) for item in _as_list(payload.get("required_slices")) if _clean_string(item)],
            bias_notes=[_clean_string(item) for item in _as_list(payload.get("bias_notes")) if _clean_string(item)],
            limitations=[_clean_string(item) for item in _as_list(payload.get("limitations")) if _clean_string(item)],
            governance=dict(payload.get("governance", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "license": self.license,
            "intended_use": self.intended_use,
            "modalities": list(self.modalities),
            "population": self.population,
            "label_schema": self.label_schema,
            "gold_standard": self.gold_standard,
            "deidentified": self.deidentified,
            "splits": self.splits,
            "required_slices": list(self.required_slices),
            "bias_notes": list(self.bias_notes),
            "limitations": list(self.limitations),
            "governance": {
                "clinical_warning": "Dataset manifest metadata does not imply clinical validation.",
                **self.governance,
            },
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    def validate(self) -> dict[str, Any]:
        failures: list[str] = []
        warnings: list[str] = []
        if not self.dataset_id or self.dataset_id == "unknown_dataset":
            failures.append("dataset_id_missing")
        if not self.source:
            failures.append("source_missing")
        if not self.license:
            failures.append("license_missing")
        if not self.deidentified:
            failures.append("dataset_not_marked_deidentified")
        if not self.modalities:
            warnings.append("modalities_missing")
        if not self.label_schema:
            failures.append("label_schema_missing")
        if self.gold_standard == "unknown":
            warnings.append("gold_standard_unknown")
        if self.intended_use not in {"research_only", "retrospective_validation", "prospective_validation"}:
            failures.append("intended_use_not_supported_for_phase5a")
        if not self.required_slices:
            warnings.append("required_slices_missing")
        return {
            "status": "pass" if not failures else "blocked",
            "failures": failures,
            "warnings": warnings,
            "fingerprint": self.fingerprint(),
            "manifest": self.as_dict(),
        }


@dataclass(slots=True)
class DatasetManifestRegistry:
    manifests: dict[str, DatasetManifest] = field(default_factory=dict)

    def register(self, manifest: DatasetManifest | dict[str, Any]) -> dict[str, Any]:
        manifest_obj = manifest if isinstance(manifest, DatasetManifest) else DatasetManifest.from_dict(manifest)
        self.manifests[manifest_obj.dataset_id] = manifest_obj
        validation = manifest_obj.validate()
        return {"status": "registered", "dataset_id": manifest_obj.dataset_id, "validation": validation}

    def get(self, dataset_id: str) -> DatasetManifest:
        return self.manifests[dataset_id]

    def summarize(self) -> dict[str, Any]:
        validations = {dataset_id: manifest.validate()["status"] for dataset_id, manifest in self.manifests.items()}
        return {
            "dataset_count": len(self.manifests),
            "statuses": validations,
            "clinical_warning": "Registry is for research governance, not regulatory approval.",
        }
