"""What the case asks of a person, by explicit rules (28 November 2026): no weights, no thresholds to tune.

Until 28 November 2026 ``_derive_governance_scores`` summed a "risk" and an "uncertainty" from the case's signals with
weights chosen by hand (0.30, 0.25, 0.20…), and three gates compared them with thresholds chosen by hand (risk 0.35,
uncertainty 0.65 and 0.75). None was ever measured. Each rule here is one sentence, and the result keeps it.

Escalation to a person (the result is never withheld for it: principle of doubt, 7 November 2026):

* ``discordances_in_the_reading``: the reading sent something to a person (``reasoning.case_discordances``);
* ``document_not_read``: a document of the case no interpreter read;
* ``declared_severity_high``: whoever sent the case declared it severe (``clinical_severity``: ``high``).

Flags, written on the result and not escalated:

* ``no_evidence``: the retrieval found nothing in the memory (only the fallback, or nothing at all);
* ``no_provenance``: the case declares no source.

A number comes back only when it is measured on published cases.
"""

from __future__ import annotations

from typing import Any

ESCALATE = ("discordances_in_the_reading", "document_not_read", "declared_severity_high")
FLAGS = ("no_evidence", "no_provenance")
HIGH = {"high", "alta", "grave", "severe"}

SENTENCES = {
    "discordances_in_the_reading": "the reading of the case sent {n} discordances to a person",
    "document_not_read": "{n} documents of the case were read by no interpreter",
    "declared_severity_high": "the case was declared severe",
    "no_evidence": "the retrieval found nothing in the memory",
    "no_provenance": "the case declares no source",
}


def declared_severity(payload: dict[str, Any] | None) -> str:
    """``high`` when the case declares it (a word, not a number); otherwise ``not_declared``."""
    value = (payload or {}).get("clinical_severity")
    return "high" if isinstance(value, str) and value.strip().lower() in HIGH else "not_declared"


def case_signals(
    payload: dict[str, Any] | None,
    discordances: dict[str, Any] | None,
    retrieval: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = payload or {}
    discordances = discordances or {}
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    quality = retrieval.get("retrieval_quality") or {}
    unread = int((discordances.get("kinds") or {}).get("unread", 0) or 0)
    others = int(discordances.get("count", 0) or 0) - unread
    evidence_count = int(retrieval.get("evidence_count", 0) or 0)
    memory_backed = bool(quality.get("memory_backed", False)) and not quality.get("fallback_used", False)

    found: list[tuple[str, int]] = []
    if others > 0:
        found.append(("discordances_in_the_reading", others))
    if unread > 0:
        found.append(("document_not_read", unread))
    if declared_severity(payload) == "high":
        found.append(("declared_severity_high", 1))
    if not memory_backed or evidence_count == 0:
        found.append(("no_evidence", evidence_count))
    provenance = payload.get("provenance")
    if not (isinstance(provenance, dict) and provenance):
        found.append(("no_provenance", 0))

    reasons = [{"rule": rule, "sentence": SENTENCES[rule].format(n=n)} for rule, n in found if rule in ESCALATE]
    flags = [{"rule": rule, "sentence": SENTENCES[rule].format(n=n)} for rule, n in found if rule in FLAGS]
    return {
        "escalate": bool(reasons),
        "reasons": reasons,
        "flags": flags,
        "declared_severity": declared_severity(payload),
        "evidence": {"count": evidence_count, "memory_backed": memory_backed},
        "derivation": "explicit_rules_without_weights",
    }


def rules_of(signals: dict[str, Any] | None, key: str = "reasons") -> list[str]:
    return [str(item.get("rule")) for item in (signals or {}).get(key, []) or [] if isinstance(item, dict)]


__all__ = ["ESCALATE", "FLAGS", "case_signals", "declared_severity", "rules_of"]
