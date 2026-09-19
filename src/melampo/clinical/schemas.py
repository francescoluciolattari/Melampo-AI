from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FHIRResourceEnvelope:
    resource_type: str
    payload: dict[str, Any]
    profile_urls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiagnosticReportPayload:
    status: str
    code: str
    conclusion: str
    supporting_info: list[str] = field(default_factory=list)
    presented_form: list[dict[str, Any]] = field(default_factory=list)
