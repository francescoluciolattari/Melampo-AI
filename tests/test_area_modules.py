from melampo.areas.case_context_area import CaseContextArea
from melampo.areas.epidemiology_area import EpidemiologyArea
from melampo.areas.language_listening_area import LanguageListeningArea
from melampo.areas.visual_diagnostic_area import VisualDiagnosticArea


def test_area_modules_expose_focus_and_salience_metadata():
    visual = VisualDiagnosticArea().integrate(
        volume_features={"study_id": "s1"},
        pathology_features={"slide_id": "p1"},
        patient_visual={"face": "pale"},
        labs_snapshot={"crp": 12},
    )
    assert visual["focus"] == "imaging_led"
    assert visual["signal_count"] == 4
    assert visual["salience_score"] > 0.0

    language = LanguageListeningArea().integrate(
        report_text="possible lesion",
        ehr_text="follow up suggested",
        patient_complaints="cough and weight loss",
        voice_features={"tone": "strained"},
    )
    assert language["focus"] == "language_led"
    assert language["signal_count"] == 4
    assert language["text_length"] > 0

    epidemiology = EpidemiologyArea().integrate(
        demographics={"age": 64},
        provenance={"region": "urban"},
        exposures={"smoking": True},
    )
    assert epidemiology["focus"] == "epidemiology_led"
    assert epidemiology["signal_count"] == 3

    context = CaseContextArea().integrate({"bundle_keys": ["Condition"], "priority": "high"})
    assert context["focus"] == "multimodal_context"
    assert context["signal_count"] == 2
    assert context["salience_score"] > 0.0
