"""Tests for patient_matching.py: finding a pending case without a case_id,
via fiscal code, or via name+surname+date+diagnostic question together --
never any single field alone. Two technical corrections drove this design,
both verified directly, not assumed: a cryptographic hash has no
meaningful "closeness" (fiscal code / name / surname matching is always
exact), and this project's only embedding function is not semantic (the
diagnostic question is compared by graph concept overlap, falling back to
plain word overlap, never vector similarity).
"""

from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.training.patient_matching import (
    PatientIdentifiers,
    hash_identifying_value,
    identifiers_match,
    normalize_name_component,
)

_GRAPH = InMemoryConceptGraph.from_edges(
    [ConceptEdge("Marfan syndrome", "has_phenotype", "Aortic root aneurysm", weight=0.9)]
)


def _identifiers(name, surname, date, question, fiscal_code=None, password="secret"):
    payload = {"patient_name": name, "patient_surname": surname, "case_date": date, "diagnostic_question": question}
    if fiscal_code:
        payload["patient_fiscal_code"] = fiscal_code
    return PatientIdentifiers.from_payload(payload, password)


# --------------------------------------------------------------------------
# normalize_name_component: the property everything else depends on
# --------------------------------------------------------------------------


def test_case_is_normalized():
    assert normalize_name_component("MARIO") == normalize_name_component("mario")


def test_accents_are_stripped():
    assert normalize_name_component("José") == normalize_name_component("Jose")


def test_internal_double_spaces_collapse():
    assert normalize_name_component("Mario  Rossi") == normalize_name_component("Mario Rossi")


def test_leading_and_trailing_whitespace_is_trimmed():
    assert normalize_name_component("  Mario  ") == "mario"


def test_the_same_value_and_password_always_hash_the_same():
    a = hash_identifying_value(normalize_name_component("Mario"), "secret")
    b = hash_identifying_value(normalize_name_component("MARIO"), "secret")
    assert a == b


def test_a_different_password_changes_the_hash():
    a = hash_identifying_value(normalize_name_component("Mario"), "secret-one")
    b = hash_identifying_value(normalize_name_component("Mario"), "secret-two")
    assert a != b


# --------------------------------------------------------------------------
# identifiers_match: fiscal code decisive alone; otherwise all four
# required together
# --------------------------------------------------------------------------


def test_fiscal_code_alone_is_decisive_regardless_of_case():
    a = _identifiers("Mario", "Rossi", "2026-01-01", "unrelated question", fiscal_code="RSSMRA80A01H501U")
    b = _identifiers("Different", "Name", "2099-12-31", "totally unrelated", fiscal_code="rssmra80a01h501u")
    assert identifiers_match(a, b, _GRAPH) is True


def test_a_different_fiscal_code_never_matches_even_with_everything_else_equal():
    a = _identifiers("Mario", "Rossi", "2026-01-01", "same question", fiscal_code="RSSMRA80A01H501U")
    b = _identifiers("Mario", "Rossi", "2026-01-01", "same question", fiscal_code="VRDLGU75B02H501X")
    assert identifiers_match(a, b, _GRAPH) is False


def test_name_surname_date_and_overlapping_question_together_match():
    a = _identifiers("Mario", "Rossi", "2026-09-01", "Evaluate for aortic root aneurysm")
    b = _identifiers("MARIO", "  Rossi  ", "2026-09-01", "Suspected aortic root aneurysm, follow-up")
    assert identifiers_match(a, b, _GRAPH) is True


def test_a_different_surname_never_matches_even_with_everything_else_equal():
    a = _identifiers("Mario", "Rossi", "2026-09-01", "Evaluate for aortic root aneurysm")
    b = _identifiers("Mario", "Bianchi", "2026-09-01", "Evaluate for aortic root aneurysm")
    assert identifiers_match(a, b, _GRAPH) is False


def test_a_different_date_never_matches_even_with_everything_else_equal():
    a = _identifiers("Mario", "Rossi", "2026-09-01", "Evaluate for aortic root aneurysm")
    b = _identifiers("Mario", "Rossi", "2026-09-02", "Evaluate for aortic root aneurysm")
    assert identifiers_match(a, b, _GRAPH) is False


def test_an_unrelated_diagnostic_question_never_matches_even_with_everything_else_equal():
    a = _identifiers("Mario", "Rossi", "2026-09-01", "Evaluate for aortic root aneurysm")
    b = _identifiers("Mario", "Rossi", "2026-09-01", "Routine follow-up, no specific complaint")
    assert identifiers_match(a, b, _GRAPH) is False


def test_missing_name_or_surname_never_matches_without_a_fiscal_code():
    a = PatientIdentifiers.from_payload({"case_date": "2026-09-01", "diagnostic_question": "x"}, "secret")
    b = _identifiers("Mario", "Rossi", "2026-09-01", "x")
    assert identifiers_match(a, b, _GRAPH) is False


# --------------------------------------------------------------------------
# The Italian-language finding: the graph's concept names are English, so
# real Italian clinical phrasing needs the word-overlap fallback to work
# at all
# --------------------------------------------------------------------------


def test_italian_phrasing_with_genuine_overlap_matches_via_the_word_fallback():
    a = _identifiers("Mario", "Rossi", "2026-09-01", "Valutare aneurisma della radice aortica")
    b = _identifiers("Mario", "Rossi", "2026-09-01", "Sospetto aneurisma della radice aortica")
    assert identifiers_match(a, b, _GRAPH) is True


def test_italian_phrasing_with_no_real_overlap_does_not_match():
    a = _identifiers("Mario", "Rossi", "2026-09-01", "Valutare aneurisma della radice aortica")
    b = _identifiers("Mario", "Rossi", "2026-09-01", "Controllo di routine, nessun sintomo specifico")
    assert identifiers_match(a, b, _GRAPH) is False


def test_an_empty_diagnostic_question_on_either_side_never_matches():
    a = _identifiers("Mario", "Rossi", "2026-09-01", "")
    b = _identifiers("Mario", "Rossi", "2026-09-01", "Evaluate for aortic root aneurysm")
    assert identifiers_match(a, b, _GRAPH) is False
