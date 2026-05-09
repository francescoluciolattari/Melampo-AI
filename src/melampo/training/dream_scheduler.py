from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..memory.vector_memory import InMemoryVectorStore
from .dream_candidate_store import DreamCandidateStore
from .promotion_policy import PromotionPolicy
from .rational_control_validator import RationalControlValidator
from .self_evolution import DreamSelfEvolutionLoop


def _job_id(case_context: dict[str, Any], scheduled_at: float) -> str:
    case_id = str(case_context.get("case_id", "unknown_case"))
    digest = hashlib.sha256(f"job:{case_id}:{json.dumps(case_context, sort_keys=True, default=str)}:{scheduled_at}".encode("utf-8")).hexdigest()
    return f"dream_job:{case_id}:{digest[:12]}"


@dataclass(slots=True)
class LowActivityPolicy:
    min_idle_seconds: float = 0.0
    max_active_requests: int = 0
    max_jobs_per_window: int = 5

    def should_run(self, activity: dict[str, Any] | None = None) -> dict[str, Any]:
        activity = activity or {}
        active_requests = int(activity.get("active_requests", 0) or 0)
        idle_seconds = float(activity.get("idle_seconds", 0.0) or 0.0)
        allowed = active_requests <= self.max_active_requests and idle_seconds >= self.min_idle_seconds
        reasons = []
        if active_requests > self.max_active_requests:
            reasons.append("active_requests_above_low_activity_threshold")
        if idle_seconds < self.min_idle_seconds:
            reasons.append("idle_seconds_below_threshold")
        return {
            "allowed": allowed,
            "reasons": reasons,
            "observed": {"active_requests": active_requests, "idle_seconds": idle_seconds},
            "thresholds": {
                "max_active_requests": self.max_active_requests,
                "min_idle_seconds": self.min_idle_seconds,
                "max_jobs_per_window": self.max_jobs_per_window,
            },
        }


@dataclass(slots=True)
class DreamReplayJob:
    job_id: str
    case_context: dict[str, Any]
    area_dynamics: dict[str, Any]
    dream: dict[str, Any] = field(default_factory=dict)
    retrieval_context: dict[str, Any] = field(default_factory=dict)
    governance_scores: dict[str, Any] = field(default_factory=dict)
    scheduled_at: float = field(default_factory=time.time)
    status: str = "queued"

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "case_context": self.case_context,
            "area_dynamics": self.area_dynamics,
            "dream": self.dream,
            "retrieval_context": self.retrieval_context,
            "governance_scores": self.governance_scores,
            "scheduled_at": self.scheduled_at,
            "status": self.status,
        }


@dataclass(slots=True)
class DreamScheduler:
    """Synchronous low-activity dream replay scheduler with promotion guardrails."""

    candidate_store: DreamCandidateStore = field(default_factory=DreamCandidateStore)
    vector_store: InMemoryVectorStore = field(default_factory=InMemoryVectorStore.enterprise_default)
    self_evolution_loop: DreamSelfEvolutionLoop = field(default_factory=DreamSelfEvolutionLoop)
    validator: RationalControlValidator = field(default_factory=RationalControlValidator)
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    low_activity_policy: LowActivityPolicy = field(default_factory=LowActivityPolicy)
    queue: list[DreamReplayJob] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)

    def enqueue(
        self,
        case_context: dict[str, Any],
        area_dynamics: dict[str, Any],
        dream: dict[str, Any] | None = None,
        retrieval_context: dict[str, Any] | None = None,
        governance_scores: dict[str, Any] | None = None,
    ) -> DreamReplayJob:
        scheduled_at = time.time()
        job = DreamReplayJob(
            job_id=_job_id(case_context=case_context or {}, scheduled_at=scheduled_at),
            case_context=case_context or {},
            area_dynamics=area_dynamics or {},
            dream=dream or {},
            retrieval_context=retrieval_context or {},
            governance_scores=governance_scores or {},
            scheduled_at=scheduled_at,
        )
        self.queue.append(job)
        self.execution_log.append({"event": "job_enqueued", "job_id": job.job_id, "timestamp": scheduled_at})
        return job

    def enqueue_many(self, cases: Iterable[dict[str, Any]]) -> list[DreamReplayJob]:
        jobs = []
        for item in cases:
            jobs.append(
                self.enqueue(
                    case_context=dict(item.get("case_context", item)),
                    area_dynamics=dict(item.get("area_dynamics", {})),
                    dream=dict(item.get("dream", {})),
                    retrieval_context=dict(item.get("retrieval_context", {})),
                    governance_scores=dict(item.get("governance_scores", {})),
                )
            )
        return jobs

    def _execute_job(self, job: DreamReplayJob) -> dict[str, Any]:
        candidate_payload = self.self_evolution_loop.generate_candidate(
            case_context=job.case_context,
            area_dynamics=job.area_dynamics,
            dream=job.dream,
        )
        metadata = dict(candidate_payload.get("metadata", {}))
        auto_plan = job.dream.get("auto_evolution_plan", {}) if isinstance(job.dream, dict) else {}
        candidate_payload = {
            **candidate_payload,
            "case_id": job.case_context.get("case_id", metadata.get("case_id", "unknown_case")),
            "area_dynamics": job.area_dynamics,
            "retrieval_context": job.retrieval_context,
            "governance_scores": job.governance_scores,
            "auto_evolution_plan": auto_plan,
            "metadata": {
                **metadata,
                "risk": job.governance_scores.get("risk", metadata.get("risk", 0.0)),
                "retrieval_coverage": job.governance_scores.get("retrieval_coverage", job.retrieval_context.get("retrieval_coverage", 0.0)),
                "provenance_quality": job.governance_scores.get("provenance_quality", metadata.get("provenance_quality", 0.0)),
                "auto_evolution_plan": auto_plan,
                "source": "dream_scheduler",
            },
        }
        record = self.candidate_store.create_candidate(
            payload=candidate_payload,
            case_id=str(candidate_payload.get("case_id", "unknown_case")),
            source="dream_scheduler",
            learning_status="candidate",
        )
        validation = self.validator.evaluate(
            candidate=candidate_payload,
            area_dynamics=job.area_dynamics,
            retrieval_context=job.retrieval_context,
            governance_scores=job.governance_scores,
        )
        record_payload = self.candidate_store.attach_validation(record.candidate_id, validation)
        decision = self.promotion_policy.decide(candidate=record_payload, validation=validation)
        record_payload = self.candidate_store.attach_promotion_decision(record.candidate_id, decision)
        memory_doc = self.candidate_store.get(record.candidate_id).to_memory_document()
        memory_record = self.vector_store.upsert(
            text=memory_doc["text"],
            metadata=memory_doc["metadata"],
            modality=memory_doc["modality"],
            source=memory_doc["source"],
            learning_status=memory_doc["learning_status"],
        )
        job.status = "completed"
        result = {
            "job": job.as_dict(),
            "candidate": record_payload,
            "validation": validation,
            "promotion_decision": decision,
            "memory_record": memory_record.describe(),
        }
        self.execution_log.append({"event": "job_completed", "job_id": job.job_id, "candidate_id": record.candidate_id, "timestamp": time.time()})
        return result

    def run_once(self, activity: dict[str, Any] | None = None, max_jobs: int | None = None) -> dict[str, Any]:
        permission = self.low_activity_policy.should_run(activity=activity)
        if not permission["allowed"]:
            return {
                "status": "skipped",
                "reason": permission["reasons"],
                "low_activity": permission,
                "queued_jobs": len([job for job in self.queue if job.status == "queued"]),
                "results": [],
            }
        limit = max_jobs or self.low_activity_policy.max_jobs_per_window
        queued = [job for job in self.queue if job.status == "queued"][:limit]
        results = [self._execute_job(job) for job in queued]
        return {
            "status": "completed",
            "low_activity": permission,
            "processed_jobs": len(results),
            "queued_jobs_remaining": len([job for job in self.queue if job.status == "queued"]),
            "results": results,
            "candidate_store": {"record_count": len(self.candidate_store.records)},
            "vector_memory": self.vector_store.describe(),
        }
