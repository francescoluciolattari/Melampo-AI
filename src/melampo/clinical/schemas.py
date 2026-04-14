from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class FHIRResourceEnvelope:
    resource_type: str
    payload: Dict[str, Any]
    profile_urls: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DiagnosticReportPayload:
    status: str
    code: str
    conclusion: str
    supporting_info: List[str] = field(default_factory=list)
    presented_form: List[Dict[str, Any]] = field(default_factory=list)
