"""The dual path, executed (14 November 2026): when the router picks ``dual_path`` the case's documents are read
again by the two root models through the recursive engine, and what they read is reconciled with the one-shot
retrieval.

Until now ``ModelRouter.pick_mode`` computed ``dual_path`` and nothing ran it; ``root_model_cross_check`` and
``retrieval_reconciliation`` were imported by nothing. The root models are chosen by measure (14 November 2026,
``root_model_cross_check.DEFAULT_PAIR``): Nemotron 3 Ultra and Gemma 4 31B, without reasoning, 25 of 25 graded
cases right with the engine that shows the text, 16 of 16 mechanisms right on vetting-bench v3.

Configuration, local only (a patient's documents never leave the machine; ``report_router.local_chat`` refuses any
other host): ``MELAMPO_LOCAL_CHAT_ENDPOINT`` and ``MELAMPO_ROOT_MODELS``, the two models as the local server names
them, primary first. Unset, the dual path says it is not configured and the one-shot retrieval stands as it was.

What it gives the case:
- ``retrieval``: the reconciled payload (``retrieval_reconciliation``): what both paths found is confirmed, what
  only the recursive path found enters only with verifiable offsets, the rest is discarded as overreach;
- ``cross_check``: the two models' answers and whether they agree; a disagreement, or a model that did not finish,
  goes to a person with both answers (the principle of doubt: nothing is chosen between them here).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from ..memory.context_environment import EnvironmentDocument
from .retrieval_reconciliation import build_dual_path_payload, reconcile
from .rlm_engine import PHASE_ONE_DATA_CLASSES, Budget
from .root_model_cross_check import DEFAULT_PAIR, cross_check

DUAL_PATH_MODES = ("dual_path", "dual_path_extended_budget")
DEFAULT_QUESTION = (
    "Which findings in these documents bear on the diagnosis? Name each finding with the document it comes from."
)


def root_models_from_env() -> tuple[dict[str, Callable[[str], str]], str]:
    """The two local root models, or why there are none."""
    endpoint = os.environ.get("MELAMPO_LOCAL_CHAT_ENDPOINT", "").strip()
    names = [n.strip() for n in os.environ.get("MELAMPO_ROOT_MODELS", "").split(",") if n.strip()]
    if not endpoint or len(names) != 2:
        return {}, ("not configured: MELAMPO_LOCAL_CHAT_ENDPOINT and MELAMPO_ROOT_MODELS (two local models, "
                    f"measured pair {DEFAULT_PAIR[0]}, {DEFAULT_PAIR[1]})")
    from ..data.report_router import local_chat

    try:
        return {name: local_chat(endpoint, name, timeout=120.0, max_tokens=1024) for name in names}, "configured"
    except ValueError as error:          # a host other than this machine
        return {}, f"refused: {error}"


def case_documents(case_id: str, report_text: str, attachments: Sequence[dict[str, Any]] = (), *,
                   data_class: str = "patient") -> list[EnvironmentDocument]:
    """The case's documents for the engine, each with its data class: ``patient`` unless the payload declares
    ``synthetic`` or ``deidentified``. The engine admits only those two in its first phase (rlm_engine,
    PHASE_ONE_DATA_CLASSES): a patient's text is refused there, and the dual path says so."""
    documents = []
    if str(report_text or "").strip():
        documents.append(EnvironmentDocument(document_id=f"{case_id}:report", text=str(report_text), source="case",
                                             metadata={"data_class": data_class}))
    for index, attachment in enumerate(attachments or (), start=1):
        text = str((attachment or {}).get("text") or "").strip()
        if text:
            name = str(attachment.get("name") or attachment.get("document_id") or f"attachment_{index}")
            documents.append(EnvironmentDocument(document_id=f"{case_id}:{name}", text=text, source="case",
                                                 metadata={"data_class": data_class}))
    return documents


def run_dual_path(case_id: str, documents: Sequence[EnvironmentDocument], one_shot: dict[str, Any], mode: str, *,
                  question: str = "", models: dict[str, Callable[[str], str]] | None = None) -> dict[str, Any]:
    """The dual path for one case: executed, not needed, or why not."""
    if mode not in DUAL_PATH_MODES:
        return {"status": "not_needed", "mode": mode}
    if models is None:
        models, reason = root_models_from_env()
        if not models:
            return {"status": "not_configured", "mode": mode, "reason": reason}
    if len(models) != 2:
        return {"status": "not_configured", "mode": mode, "reason": "two root models are needed"}
    if not documents:
        return {"status": "no_documents", "mode": mode}
    classes = {str(d.metadata.get("data_class", "")) for d in documents}
    if not classes <= set(PHASE_ONE_DATA_CLASSES):
        # the engine's own rule, kept: a patient's text is read by the engine only once it is de-identified
        return {"status": "refused", "mode": mode, "reason": f"data_class {sorted(classes)}: the recursive engine "
                "admits only synthetic or de-identified text in its first phase"}
    (first_name, first), (second_name, second) = list(models.items())
    extended = mode == "dual_path_extended_budget"

    def budget() -> Budget:
        return Budget(max_iterations=16 if extended else 12, max_wall_clock_seconds=180.0 if extended else 90.0)

    checked = cross_check(case_id, documents, question or DEFAULT_QUESTION, first, second,
                          primary_name=first_name, secondary_name=second_name, budget_factory=budget)
    # the recursive path is what the models that finished read: a model that did not finish brings nothing
    recursive_evidence = ((checked.primary_evidence if checked.primary_completed else [])
                          + (checked.secondary_evidence if checked.secondary_completed else []))
    recursive = {"query": question or DEFAULT_QUESTION, "evidence": recursive_evidence}
    verdict = reconcile(one_shot or {}, recursive)
    return {
        "status": "executed",
        "mode": mode,
        "retrieval": build_dual_path_payload(one_shot or {}, recursive, verdict),
        "cross_check": {
            "models": [first_name, second_name],
            "answers": [checked.primary_answer, checked.secondary_answer],
            "completed": [checked.primary_completed, checked.secondary_completed],
            "agree": checked.answers_agree,
            "needs_review": not checked.answers_agree,
            "notes": list(checked.notes),
        },
        "reconciliation": verdict.as_dict(),
    }
