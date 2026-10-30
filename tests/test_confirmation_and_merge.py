"""The pending case's follow-ups through the real pipeline (30 October 2026): a confirmation closes the case and
runs no diagnosis; its clinical basis decides the learning cycle; what arrives with it is kept for learning and a
finding away from the diagnosis goes to the physician; a merge keeps the earlier findings and is read whole."""

from melampo.app import build_default_runtime
from melampo.data.ingestion import PHYSICIAN_NOTE_KEY
from melampo.data.report_reading import ReportReaders, build_linker
from melampo.governance.confirmation_registry import ConfirmationRegistry
from melampo.training.case_confirmation import cases_awaiting_source, declare_confirmation_source
from melampo.training.confirmed_case_store import InMemoryConfirmedCaseStore
from melampo.training.training_extraction import extract_and_purge

LINKER = None


def _pipeline():
    global LINKER
    if LINKER is None:
        LINKER = build_linker()[0]
    pipeline = build_default_runtime().pipeline
    pipeline.ingestion.report_readers = ReportReaders(LINKER)
    return pipeline


def _pending(pipeline, case_id, **extra):
    pipeline.run({"case_id": case_id, "report_text": "Tosse persistente da tre settimane.", **extra})
    pipeline._nexus_scheduler_instance().run_once(activity={"active_requests": 0, "idle_seconds": 100})
    store = pipeline._nexus_scheduler_instance().candidate_store
    assert store.find_by_case_id(case_id, statuses=("candidate", "needs_review")) is not None
    return store


def test_a_confirmation_closes_the_case_and_runs_no_diagnosis():
    pipeline = _pipeline()
    store = _pending(pipeline, "conf-1")
    scheduler = pipeline._nexus_scheduler_instance()
    queued = len(scheduler.queue) if hasattr(scheduler, "queue") else None
    result = pipeline.run({"case_id": "conf-1", "confirmed_diagnosis": "Sarcoidosis",
                           "confirmation_source": "histopathology"})
    assert result["status"] == "case_closed_by_confirmation"
    assert result["pending_case_routing"]["action"] == "confirm_and_train"
    assert "diagnostic_result" not in result and "nexus" not in result          # no diagnosis was run
    assert result["learning"] == {"admitted": True, "rejection": None, "awaiting_clinical_basis": False}
    assert store.find_by_case_id("conf-1", statuses=("candidate", "needs_review")) is None
    if queued is not None:
        assert len(pipeline._nexus_scheduler_instance().queue) == queued     # no new pending job for it
    [kept] = list(pipeline.confirmed_case_store.load())
    assert kept["confirmed_diagnosis"] == "Sarcoidosis" and kept["confirmation_source"] == "histopathology"
    assert result["confirmed_case_store"] == "this_process_only"            # no DB_PASSWORD in the test


def test_without_a_clinical_basis_the_case_is_kept_but_waits_to_be_declared():
    pipeline = _pipeline()
    _pending(pipeline, "conf-2")
    result = pipeline.run({"case_id": "conf-2", "confirmed_diagnosis": "Sarcoidosis"})
    assert result["status"] == "case_closed_by_confirmation"
    assert result["learning"]["admitted"] is False and result["learning"]["awaiting_clinical_basis"]
    store, registry = pipeline.confirmed_case_store, pipeline.confirmation_registry
    assert [c["case_id"] for c in cases_awaiting_source(store, registry)] == ["conf-2"]
    # a review that saw the suggestion is still not independent
    assert declare_confirmation_source("conf-2", "independent_review", confirmed_case_store=store,
                                       registry=registry, reviewer_blinded_to_suggestion=False) == \
        (False, "reviewer_saw_the_suggestion_before_confirming")
    # once recorded, a basis is not rewritten
    assert declare_confirmation_source("conf-2", "histopathology", confirmed_case_store=store,
                                       registry=registry) == (False, "confirmation_source_already_recorded")


def test_a_basis_declared_after_closure_admits_the_case():
    pipeline = _pipeline()
    _pending(pipeline, "conf-3")
    pipeline.run({"case_id": "conf-3", "confirmed_diagnosis": "Sarcoidosis"})
    store, registry = pipeline.confirmed_case_store, pipeline.confirmation_registry
    assert declare_confirmation_source("conf-3", "histopathology", confirmed_case_store=store,
                                       registry=registry) == (True, None)
    assert cases_awaiting_source(store, registry) == []


def test_the_new_report_is_kept_for_learning_and_a_finding_elsewhere_goes_to_the_physician():
    pipeline = _pipeline()
    _pending(pipeline, "conf-4")
    report = ("TC del torace. Nel lobo superiore destro massa solida di 32 mm a margini spiculati. "
              "Nel lobo epatico destro lesione ipodensa di 15 mm.")
    result = pipeline.run({"case_id": "conf-4", "confirmed_diagnosis": "adenocarcinoma of the right upper lobe",
                           "confirmation_source": "histopathology", "report_text": report})
    assert result["after_confirmation"]["findings"] == 2
    [elsewhere] = result["to_the_physician"]                    # the lobe is the diagnosis's site; the liver is not
    assert elsewhere["structure"] == "liver" and elsewhere["reason"] == "site_not_named_by_the_diagnosis"
    [kept] = list(pipeline.confirmed_case_store.load())
    assert "massa solida di 32 mm" in kept["after_confirmation"]["report_text"]
    assert len(kept["after_confirmation"]["report_reading"]["findings"]) == 2


def test_when_the_diagnosis_names_no_site_every_new_finding_is_for_the_physician():
    pipeline = _pipeline()
    _pending(pipeline, "conf-5")
    result = pipeline.run({"case_id": "conf-5", "confirmed_diagnosis": "Sarcoidosi",
                           "confirmation_source": "histopathology",
                           "report_text": "TC del torace. Nel lobo superiore destro nodulo di 9 mm."})
    [item] = result["to_the_physician"]
    assert item["reason"] == "diagnosis_names_no_site_the_linker_can_place"


def test_a_merge_keeps_the_earlier_findings_and_reads_the_whole_case():
    pipeline = _pipeline()
    seen = []

    class Spy:
        """The ingestion, recording the payload it is given (the real one is a slots dataclass)."""

        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def from_payload(self, payload):
            seen.append(dict(payload))
            return self.inner.from_payload(payload)

    pipeline.ingestion = Spy(pipeline.ingestion)
    _pending(pipeline, "merge-1", findings=["Cough"])
    pipeline.run({"case_id": "merge-1", "report_text": "TC: nel lobo superiore destro nodulo di 9 mm.",
                  "findings": ["Fever", "cough"]})
    merged = seen[-1]
    assert merged["findings"] == ["Cough", "Fever"]                        # earlier first, none repeated
    assert "Tosse persistente" in merged["report_text"] and "nodulo di 9 mm" in merged["report_text"]
    assert "Tosse persistente" in merged[PHYSICIAN_NOTE_KEY] and "nodulo di 9 mm" in merged[PHYSICIAN_NOTE_KEY]


def test_training_extraction_keeps_a_record_with_its_post_confirmation_reports():
    store, registry = InMemoryConfirmedCaseStore(), ConfirmationRegistry()
    from melampo.governance.confirmation_registry import Confirmation

    for case_id, after in (("kept", {"report_text": "istologico: margini liberi"}), ("plain", {})):
        store.persist({"case_id": case_id, "confirmed_diagnosis": "Sarcoidosis",
                       "raised_labels": ["Sarcoidosis", "Tuberculosis"], "after_confirmation": after})
        registry.register(Confirmation(case_id, "Sarcoidosis", source="histopathology"))
    first = extract_and_purge(store, registry)
    assert len(first.extraction.pairs) == 2 and first.purged_case_ids == ["plain"]
    [record] = list(store.load())
    assert record["case_id"] == "kept" and record["dpo_extracted"]
    second = extract_and_purge(store, registry)
    assert second.extraction.pairs == [] and len(list(store.load())) == 1    # pairs not built twice
