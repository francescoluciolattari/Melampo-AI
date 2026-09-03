import pytest

from melampo.memory.concept_resolution import (
    CLINICAL_MODIFIER_ROOT,
    INHERITANCE_ROOT,
    ROLE_FINDING,
    ROLE_INHERITANCE,
    ROLE_MODIFIER,
    ConceptResolver,
    TermIndex,
    attach_modifiers,
)
from melampo.reasoning.family_history import (
    BLOCK_ALREADY_ASSESSED,
    BLOCK_NOT_HERITABLE,
    BLOCK_ONSET_NOT_REACHED,
    DEGREE_FIRST,
    DEGREE_SECOND,
    INHERITANCE_AUTOSOMAL_DOMINANT,
    INHERITANCE_POLYGENIC,
    INHERITANCE_SPORADIC,
    ROLE_SCREENING_HYPOTHESIS,
    FamilyHistoryChannel,
    FamilyHistoryEntry,
    assert_not_a_finding,
)

# Hierarchy reproduced from hp.obo: modifiers and inheritance sit in their own branches.
OBO_SAMPLE = (
    "[Term]\n"
    "id: HP:0012823\n"
    "name: Clinical modifier\n"
    "\n"
    "[Term]\n"
    "id: HP:0012832\n"
    "name: Bilateral\n"
    "is_a: HP:0012823 ! Clinical modifier\n"
    "\n"
    "[Term]\n"
    "id: HP:0003676\n"
    "name: Progressive\n"
    "is_a: HP:0012823 ! Clinical modifier\n"
    "\n"
    "[Term]\n"
    "id: HP:0000005\n"
    "name: Mode of inheritance\n"
    "\n"
    "[Term]\n"
    "id: HP:0000006\n"
    "name: Autosomal dominant inheritance\n"
    "is_a: HP:0000005 ! Mode of inheritance\n"
    "\n"
    "[Term]\n"
    "id: HP:0002094\n"
    "name: Dyspnea\n"
    "\n"
    "[Term]\n"
    "id: HP:0002202\n"
    "name: Pleural effusion\n"
    "\n"
    "[Term]\n"
    "id: HP:0012735\n"
    "name: Cough\n"
)


def _resolver() -> ConceptResolver:
    return ConceptResolver(index=TermIndex.from_obo(OBO_SAMPLE.splitlines()))


# --------------------------------------------------------------------------
# Roles come from the hierarchy, not from a model
# --------------------------------------------------------------------------


def test_is_a_relations_are_parsed():
    index = TermIndex.from_obo(OBO_SAMPLE.splitlines())
    assert index.by_id["HP:0012832"].parents == ("HP:0012823",)
    assert index.by_id["HP:0002094"].parents == ()


def test_ancestors_are_transitive_and_memoised():
    index = TermIndex.from_obo(OBO_SAMPLE.splitlines())
    assert index.ancestors("HP:0012832") == frozenset({CLINICAL_MODIFIER_ROOT})
    assert index.ancestors("HP:0012832") == index.ancestors("HP:0012832")
    assert index.descends_from("HP:0012832", CLINICAL_MODIFIER_ROOT)
    assert index.descends_from("HP:0012823", CLINICAL_MODIFIER_ROOT), "a root descends from itself"


def test_modifiers_and_inheritance_are_separated_from_findings():
    index = TermIndex.from_obo(OBO_SAMPLE.splitlines())
    assert index.role_of("HP:0012832") == ROLE_MODIFIER
    assert index.role_of("HP:0003676") == ROLE_MODIFIER
    assert index.role_of("HP:0000006") == ROLE_INHERITANCE
    assert index.role_of("HP:0002094") == ROLE_FINDING
    assert index.descends_from("HP:0000006", INHERITANCE_ROOT)


# --------------------------------------------------------------------------
# Modifiers attach, never stand alone
# --------------------------------------------------------------------------


def test_modifiers_become_attributes_of_the_nearest_finding():
    """The noise that previously entered the graph as free-standing concepts."""
    text = "Presents with progressive dyspnea and bilateral pleural effusion."
    result = attach_modifiers(_resolver().resolve_text(text))

    assert [item.label for item in result.findings] == ["Dyspnea", "Pleural effusion"]
    assert [m.label for m in result.findings[0].modifiers] == ["Progressive"]
    assert [m.label for m in result.findings[1].modifiers] == ["Bilateral"]
    assert result.collapsed_modifiers == []


def test_a_modifier_never_appears_among_the_findings():
    result = attach_modifiers(_resolver().resolve_text("Bilateral pleural effusion noted."))
    assert "Bilateral" not in result.concepts
    assert result.concepts == ["Pleural effusion"]


def test_an_isolated_modifier_collapses_and_the_discard_is_reported():
    result = attach_modifiers(_resolver().resolve_text("The course has been progressive."))
    assert result.findings == []
    assert [item.label for item in result.collapsed_modifiers] == ["Progressive"]


def test_a_modifier_beyond_the_window_does_not_attach():
    text = "Progressive." + " filler text " * 12 + "Cough."
    result = attach_modifiers(_resolver().resolve_text(text), window=10)
    assert result.collapsed_modifiers
    assert result.findings[0].modifiers == ()


def test_inheritance_statements_are_separated_from_findings():
    concepts = _resolver().resolve_text("Autosomal dominant inheritance with cough.")
    result = attach_modifiers(concepts)
    assert [item.label for item in result.inheritance_statements] == ["Autosomal dominant inheritance"]
    assert result.concepts == ["Cough"]


def test_the_finding_keeps_its_identifier_and_stays_traversable():
    result = attach_modifiers(_resolver().resolve_text("Bilateral pleural effusion."))
    payload = result.findings[0].as_dict()
    assert payload["term_id"] == "HP:0002202"
    assert payload["modifiers"][0]["term_id"] == "HP:0012832"


# --------------------------------------------------------------------------
# Family history: destination, never origin
# --------------------------------------------------------------------------


def _entry(**kwargs) -> FamilyHistoryEntry:
    defaults = {
        "condition": "Diabetes mellitus",
        "degree": DEGREE_FIRST,
        "inheritance": INHERITANCE_POLYGENIC,
    }
    return FamilyHistoryEntry(**{**defaults, **kwargs})


def test_an_undiagnosed_heritable_condition_becomes_a_screening_hypothesis():
    result = FamilyHistoryChannel().evaluate([_entry()])
    assert len(result.screening) == 1
    payload = result.screening[0].as_dict()
    assert payload["role"] == ROLE_SCREENING_HYPOTHESIS
    assert payload["usable_as_evidence"] is False
    assert payload["belongs_in_differential"] is False
    assert "undiagnosed" in payload["rationale"]


def test_the_screening_hypothesis_is_not_a_patient_finding():
    result = FamilyHistoryChannel().evaluate([_entry()])
    with pytest.raises(ValueError):
        assert_not_a_finding([item.as_dict() for item in result.screening])


def test_a_dominant_mode_shifts_the_prior_more_than_a_polygenic_one():
    channel = FamilyHistoryChannel()
    dominant = channel.prior_shift(_entry(inheritance=INHERITANCE_AUTOSOMAL_DOMINANT))
    polygenic = channel.prior_shift(_entry(inheritance=INHERITANCE_POLYGENIC))
    assert dominant[0] > polygenic[0]
    assert dominant[1] > polygenic[1]


def test_a_distant_relative_attenuates_toward_no_effect_rather_than_reversing():
    channel = FamilyHistoryChannel()
    first = channel.prior_shift(_entry(inheritance=INHERITANCE_AUTOSOMAL_DOMINANT, degree=DEGREE_FIRST))
    second = channel.prior_shift(_entry(inheritance=INHERITANCE_AUTOSOMAL_DOMINANT, degree=DEGREE_SECOND))
    assert 1.0 < second[0] < first[0]
    assert 1.0 < second[1] < first[1]


def test_a_sporadic_condition_is_blocked():
    result = FamilyHistoryChannel().evaluate([_entry(inheritance=INHERITANCE_SPORADIC)])
    assert result.screening == []
    assert result.blocked[0].reason == BLOCK_NOT_HERITABLE


def test_an_adult_onset_condition_is_blocked_for_a_child():
    entries = [_entry(onset_age_years=40)]
    channel = FamilyHistoryChannel()
    assert channel.evaluate(entries, patient_age_years=8).blocked[0].reason == BLOCK_ONSET_NOT_REACHED
    assert channel.evaluate(entries, patient_age_years=55).screening


def test_an_already_assessed_condition_is_not_re_proposed():
    """Otherwise the same consideration returns at every visit."""
    result = FamilyHistoryChannel().evaluate(
        [_entry()], already_assessed=["diabetes  MELLITUS"]
    )
    assert result.screening == []
    assert result.blocked[0].reason == BLOCK_ALREADY_ASSESSED


def test_unknown_inheritance_is_admitted_by_default_but_configurable():
    entries = [_entry(inheritance=None)]
    assert FamilyHistoryChannel().evaluate(entries).screening
    strict = FamilyHistoryChannel(unknown_inheritance_is_actionable=False)
    assert strict.evaluate(entries).blocked[0].reason == BLOCK_NOT_HERITABLE


def test_prior_shifts_are_exposed_per_condition():
    result = FamilyHistoryChannel().evaluate(
        [_entry(condition="Long QT syndrome", inheritance=INHERITANCE_AUTOSOMAL_DOMINANT)]
    )
    shifts = result.prior_shifts()
    assert "Long QT syndrome" in shifts
    assert shifts["Long QT syndrome"][0] > 1.0


def test_the_result_states_that_family_history_is_not_a_finding():
    payload = FamilyHistoryChannel().evaluate([_entry()]).as_dict()
    assert "never enters patient findings" in payload["note"]


def test_ordinary_findings_pass_the_guard():
    assert_not_a_finding([{"term_id": "HP:0002094", "label": "Dyspnea"}])
