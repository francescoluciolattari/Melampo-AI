"""Extract DPO pairs from confirmed cases, then purge what extraction consumed.

The piece case_confirmation.py's own docstring named as not yet done:
preference_pairs.py already builds well-formed (prompt, preferred,
rejected) pairs from ConfirmationRegistry's admitted Confirmations plus
each case's raised alternatives -- this module is the connection between
that extraction and ConfirmedCaseStore, the retained (encrypted,
anonymised) record extraction reads from, deciding which records
extraction has now consumed and can be purged.

**Purge only what was genuinely usable, decided directly, not
defaulted into.** A confirmed case with no registered Confirmation
(the registry rejected it -- unspecified evidentiary source, or the
reviewer was not blinded, see governance/confirmation_registry.py),
with no alternative raised (nothing to contrast against), or where the
confirmed diagnosis was never raised at all (a real miss, worth a human
looking at it, not silently discarded before anyone sees it) produces no
pair and is left retained -- deleting a record extraction could not use
would not be "training consumed it", it would just be deleting evidence
of what went wrong before anyone reviewed it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..governance.confirmation_registry import ConfirmationRegistry
from .confirmed_case_store import ConfirmedCaseStore
from .preference_pairs import ExtractionReport, extract_preference_pairs


@dataclass(frozen=True)
class ExtractionAndPurgeReport:
    """What was extracted, and which confirmed cases were purged as a result -- data the caller reports back."""

    extraction: ExtractionReport
    purged_case_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"extraction": self.extraction.as_dict(), "purged_case_ids": self.purged_case_ids}


def extract_and_purge(confirmed_store: ConfirmedCaseStore, registry: ConfirmationRegistry) -> ExtractionAndPurgeReport:
    """Build DPO pairs from every confirmed case currently retained, then delete the ones extraction could use.

    `raised_by_case` and `prompt_for_case` are both built here from
    ConfirmedCaseStore's own records -- raised_labels (the nexus branch's
    full alternative set, stored by case_confirmation.py's
    _close_one_record()) and the case's own diagnostic_question, when
    present, standing in for the prompt extract_preference_pairs()
    otherwise falls back to a neutral placeholder for.
    """
    records = list(confirmed_store.load())
    raised_by_case: dict[str, list[str]] = {}
    prompt_for_case: dict[str, str] = {}
    for record in records:
        case_id = record.get("case_id")
        if not case_id:
            continue
        raised_by_case[case_id] = list(record.get("raised_labels", []))
        question = record.get("case_context", {}).get("diagnostic_question")
        if question:
            prompt_for_case[case_id] = question

    report = extract_preference_pairs(raised_by_case, registry, prompt_for_case=prompt_for_case)

    usable_case_ids = {pair.case_id for pair in report.pairs}
    for case_id in usable_case_ids:
        confirmed_store.delete(case_id)

    return ExtractionAndPurgeReport(extraction=report, purged_case_ids=sorted(usable_case_ids))
