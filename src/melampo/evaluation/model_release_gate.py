from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dataset_manifest import DatasetManifest
from .slice_analysis import SliceAnalysisReport
from .validation_protocol import ValidationProtocol


def _safe_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return dict(value)


@dataclass(slots=True)
class ReleaseGateDecision:
    status: str
    allowed_use: str
    clinical_use_allowed: bool
    promotion_allowed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    governance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "allowed_use": self.allowed_use,
            "clinical_use_allowed": self.clinical_use_allowed,
            "promotion_allowed": self.promotion_allowed,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "observed": self.observed,
            "governance": self.governance,
        }


@dataclass(slots=True)
class ModelReleaseGate:
    """Research release gate for model/memory/retriever changes.

    Passing this gate never allows clinical deployment. It only means that the
    current research artifact has enough governance evidence to be promoted to a
    stricter research review stage.
    """

    min_sample_count: int = 2
    min_coverage: float = 0.5
    min_selective_accuracy: float = 0.6
    max_expected_calibration_error: float = 0.25
    min_rag_provenance: float = 0.8
    require_locked_protocol: bool = True

    def evaluate(
        self,
        benchmark_report: Any | None = None,
        calibration_report: Any | None = None,
        rag_report: Any | None = None,
        protocol: ValidationProtocol | None = None,
        dataset_manifest: DatasetManifest | None = None,
        slice_report: SliceAnalysisReport | dict[str, Any] | None = None,
        change_control: dict[str, Any] | None = None,
    ) -> ReleaseGateDecision:
        failures: list[str] = []
        warnings: list[str] = []
        observed: dict[str, Any] = {}

        benchmark = _as_dict(benchmark_report)
        calibration = _as_dict(calibration_report)
        rag = _as_dict(rag_report)
        slices = _as_dict(slice_report)
        change_control = change_control or {}

        if benchmark:
            observed["benchmark"] = benchmark
            if int(benchmark.get("sample_count", 0)) < self.min_sample_count:
                failures.append("benchmark_sample_count_below_threshold")
            if _safe_float(benchmark, "coverage") < self.min_coverage:
                failures.append("coverage_below_threshold")
            if _safe_float(benchmark, "selective_accuracy") < self.min_selective_accuracy:
                failures.append("selective_accuracy_below_threshold")
        else:
            failures.append("benchmark_report_missing")

        if calibration:
            observed["calibration"] = calibration
            if _safe_float(calibration, "expected_calibration_error") > self.max_expected_calibration_error:
                failures.append("calibration_ece_above_threshold")
        else:
            warnings.append("calibration_report_missing")

        if rag:
            observed["rag"] = rag
            rag_status = rag.get("status")
            observed_rag = rag.get("observed", rag)
            if rag_status and rag_status not in {"pass", "research_pass"}:
                failures.append("rag_thresholds_not_passing")
            provenance = _safe_float(observed_rag, "provenance_completeness", _safe_float(rag, "provenance_completeness"))
            if provenance < self.min_rag_provenance:
                failures.append("rag_provenance_below_threshold")
        else:
            warnings.append("rag_report_missing")

        if protocol is None:
            failures.append("validation_protocol_missing")
        else:
            readiness = protocol.readiness(dataset_manifest=dataset_manifest)
            observed["protocol_readiness"] = readiness
            if self.require_locked_protocol and readiness["status"] != "ready":
                failures.extend(readiness["failures"])
            warnings.extend(readiness.get("warnings", []))
            if benchmark:
                protocol_metrics = {
                    "coverage": benchmark.get("coverage"),
                    "selective_accuracy": benchmark.get("selective_accuracy"),
                    "expected_calibration_error": calibration.get("expected_calibration_error", 0.0) if calibration else 0.0,
                    "provenance_completeness": rag.get("provenance_completeness", rag.get("observed", {}).get("provenance_completeness", 1.0)) if rag else 0.0,
                }
                protocol_result = protocol.evaluate_observed_metrics(protocol_metrics)
                observed["protocol_endpoint_evaluation"] = protocol_result
                if protocol_result["status"] != "pass":
                    failures.extend(f"protocol_endpoint:{failure}" for failure in protocol_result["failures"])

        if dataset_manifest is None:
            failures.append("dataset_manifest_missing")
        else:
            manifest_validation = dataset_manifest.validate()
            observed["dataset_manifest"] = manifest_validation
            if manifest_validation["status"] != "pass":
                failures.extend(f"dataset:{failure}" for failure in manifest_validation["failures"])
            warnings.extend(manifest_validation.get("warnings", []))

        if slices:
            observed["slice_analysis"] = slices
            if slices.get("underperforming_slices"):
                failures.append("slice_underperformance_detected")
            if slices.get("missing_required_slices"):
                warnings.append("required_slice_coverage_missing")
        else:
            warnings.append("slice_analysis_missing")

        if change_control:
            observed["change_control"] = change_control
            if change_control.get("approval_status") in {"rejected", "blocked"}:
                failures.append("change_control_rejected")
            if change_control.get("risk_level") == "high" and change_control.get("approval_status") != "approved":
                failures.append("high_risk_change_not_approved")
        else:
            warnings.append("change_control_not_attached")

        deduped_failures = list(dict.fromkeys(failures))
        deduped_warnings = list(dict.fromkeys(warnings))
        status = "research_pass" if not deduped_failures else "blocked"
        return ReleaseGateDecision(
            status=status,
            allowed_use="research_only" if status == "research_pass" else "blocked_pending_review",
            clinical_use_allowed=False,
            promotion_allowed=status == "research_pass",
            failures=deduped_failures,
            warnings=deduped_warnings,
            observed=observed,
            governance={
                "gate": "phase5a_model_release_gate",
                "clinical_warning": "Passing this gate does not authorize clinical use or regulatory claims.",
                "requires_human_review_for_any_clinical_translation": True,
            },
        )
