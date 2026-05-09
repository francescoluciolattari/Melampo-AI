from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..orchestration.model_execution_trace import ModelExecutionTrace


def _request_id(provider: str, model_name: str, role: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"provider": provider, "model_name": model_name, "role": role, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class ModelClientConfig:
    """Configuration for safe specialist-model execution.

    Modes:
    - disabled: no model call; returns not_called.
    - dry_run: validates and records the request; no model call.
    - mock: returns a deterministic mock payload for tests and demos.
    - http_json: POSTs JSON only when enabled=True and allow_remote=True.
    - local_subprocess: invokes a configured local command only when enabled=True.
    """

    mode: str = "disabled"
    enabled: bool = False
    endpoint: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 30
    allow_remote: bool = False
    local_command: list[str] = field(default_factory=list)
    mock_payload: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "endpoint_configured": self.endpoint is not None,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "allow_remote": self.allow_remote,
            "local_command_configured": bool(self.local_command),
            "mock_configured": bool(self.mock_payload),
        }


@dataclass(slots=True)
class SafeModelClient:
    provider: str
    model_name: str
    role: str
    config: ModelClientConfig = field(default_factory=ModelClientConfig)
    trace: ModelExecutionTrace = field(default_factory=ModelExecutionTrace)

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = _request_id(self.provider, self.model_name, self.role, payload)
        hidden_network_call = False
        record = self.trace.start(
            provider=self.provider,
            model_name=self.model_name,
            role=self.role,
            mode=self.config.mode,
            request_id=request_id,
            hidden_network_call=False,
            metadata={"config": self.config.describe()},
        )

        if not self.config.enabled or self.config.mode == "disabled":
            record.finish("not_called")
            return {
                "status": "not_called",
                "request_id": request_id,
                "mode": self.config.mode,
                "payload": payload,
                "trace": record.as_dict(),
                "reason": "model_disabled_or_mode_disabled",
            }

        if self.config.mode == "dry_run":
            record.finish("request_prepared")
            return {
                "status": "request_prepared",
                "request_id": request_id,
                "mode": "dry_run",
                "payload": payload,
                "trace": record.as_dict(),
                "hidden_network_call": False,
            }

        if self.config.mode == "mock":
            response = dict(self.config.mock_payload) if self.config.mock_payload else self._default_mock_payload(payload)
            record.finish(str(response.get("status", "completed")))
            return {
                "status": str(response.get("status", "completed")),
                "request_id": request_id,
                "mode": "mock",
                "payload": payload,
                "response": response,
                "trace": record.as_dict(),
                "hidden_network_call": False,
            }

        if self.config.mode == "http_json":
            if not self.config.allow_remote or not self.config.endpoint:
                record.finish("blocked", error="remote_execution_not_allowed_or_endpoint_missing")
                return {
                    "status": "blocked",
                    "request_id": request_id,
                    "mode": "http_json",
                    "payload": payload,
                    "trace": record.as_dict(),
                    "reason": "remote_execution_not_allowed_or_endpoint_missing",
                    "hidden_network_call": False,
                }
            hidden_network_call = False  # explicit, not hidden: allowed by config below
            record.hidden_network_call = hidden_network_call
            return self._execute_http_json(payload=payload, request_id=request_id, record=record)

        if self.config.mode == "local_subprocess":
            if not self.config.local_command:
                record.finish("blocked", error="local_command_missing")
                return {
                    "status": "blocked",
                    "request_id": request_id,
                    "mode": "local_subprocess",
                    "payload": payload,
                    "trace": record.as_dict(),
                    "reason": "local_command_missing",
                }
            return self._execute_local_subprocess(payload=payload, request_id=request_id, record=record)

        record.finish("blocked", error="unknown_model_client_mode")
        return {
            "status": "blocked",
            "request_id": request_id,
            "mode": self.config.mode,
            "payload": payload,
            "trace": record.as_dict(),
            "reason": "unknown_model_client_mode",
        }

    def _default_mock_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "completed",
            "signals": {
                "mock_backend": self.provider,
                "request_keys": sorted(payload.keys()),
            },
            "confidence": 0.62,
            "uncertainty": 0.38,
            "claims": [
                {
                    "claim_id": f"mock:{self.role}:1",
                    "type": "finding",
                    "normalized_entity": payload.get("case_id") or payload.get("study_id") or "mock_entity",
                    "polarity": "present",
                    "confidence": 0.62,
                    "uncertainty": 0.38,
                    "ontology_refs": [],
                    "evidence_refs": [],
                }
            ],
        }

    def _execute_http_json(self, payload: dict[str, Any], request_id: str, record) -> dict[str, Any]:
        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                self.config.endpoint or "",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:  # nosec B310 - explicitly configured endpoint
                response_payload = json.loads(response.read().decode("utf-8"))
            record.finish(str(response_payload.get("status", "completed")))
            return {
                "status": str(response_payload.get("status", "completed")),
                "request_id": request_id,
                "mode": "http_json",
                "payload": payload,
                "response": response_payload,
                "trace": record.as_dict(),
                "hidden_network_call": False,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            record.finish("failed", error=str(exc))
            return {
                "status": "failed",
                "request_id": request_id,
                "mode": "http_json",
                "payload": payload,
                "trace": record.as_dict(),
                "error": str(exc),
                "hidden_network_call": False,
            }

    def _execute_local_subprocess(self, payload: dict[str, Any], request_id: str, record) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                self.config.local_command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                record.finish("failed", error=completed.stderr[:500])
                return {
                    "status": "failed",
                    "request_id": request_id,
                    "mode": "local_subprocess",
                    "payload": payload,
                    "trace": record.as_dict(),
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                }
            response_payload = json.loads(completed.stdout or "{}")
            record.finish(str(response_payload.get("status", "completed")))
            return {
                "status": str(response_payload.get("status", "completed")),
                "request_id": request_id,
                "mode": "local_subprocess",
                "payload": payload,
                "response": response_payload,
                "trace": record.as_dict(),
                "hidden_network_call": False,
            }
        except (TimeoutError, OSError, json.JSONDecodeError) as exc:
            record.finish("failed", error=str(exc))
            return {
                "status": "failed",
                "request_id": request_id,
                "mode": "local_subprocess",
                "payload": payload,
                "trace": record.as_dict(),
                "error": str(exc),
            }
