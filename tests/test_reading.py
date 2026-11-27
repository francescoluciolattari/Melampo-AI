"""The one procedure of reading (25 November 2026): the same card, comparison and coding for every text."""

import json

import pytest

from melampo.memory import disease_profiles as dp
from melampo.reading import Thesaurus, read_text
from melampo.reading import agreement as ag
from melampo.reading import card as c
from melampo.reading.normalize import HpoIndex
from melampo.reading.prompt import reading_prompt

OBO = """[Term]
id: HP:0001945
name: Fever
synonym: "Pyrexia" EXACT []

[Term]
id: HP:0011227
name: Elevated circulating C-reactive protein concentration
synonym: "Elevated C-reactive protein" EXACT []

[Term]
id: HP:0002027
name: Abdominal pain
"""
HPO = HpoIndex(dp.Terms.from_obo(OBO.splitlines()))
REPORT = "Paziente con dolore in fossa iliaca destra. PCR 52 mg/L, elevata. Non febbre."


def card(**kw):
    base = {"words": "", "name": "", "category": "manifestation", "specific": True, "what": "",
            "interpretation": "present", "value": None, "threshold": None, "site": "", "side": "",
            "observed_by": "laboratory", "time": "", "nature": "pathological", "part": "findings", "state": "present",
            "temporality": "current", "subject": "patient", "quote": ""}
    return {**base, **kw}


def local(answers):
    """A local interpreter: the reading, then the answer to each question by its first words."""
    def chat(prompt):
        if prompt.startswith("You read"):
            return json.dumps(answers["reading"])
        for start, answer in answers.get("questions", {}).items():
            if start in prompt:
                return answer
        return "[]"
    chat.local_only = True
    return chat


CRP_A = card(words="PCR 52 mg/L, elevata", name="elevated CRP", what="CRP", interpretation="increased",
             value={"number": 52, "unit": "mg/L"}, quote="PCR 52 mg/L, elevata.")
CRP_B = card(words="PCR 52 mg/L", name="elevated C-reactive protein", what="C-reactive protein",
             interpretation="increased", value={"number": 52, "unit": "mg/L"}, quote="PCR 52 mg/L, elevata.")
FEVER = card(words="Non febbre", name="fever", what="body temperature", observed_by="examination", state="absent",
             quote="Non febbre.")


def test_one_card_one_directive_for_every_context():
    assert c.problems(c.EXAMPLE["case"], "case") == [] and c.problems(c.EXAMPLE["publication"], "publication") == []
    assert "field_of_the_publication:measures" in c.problems({**c.EXAMPLE["case"], "measures": {"a": 1}}, "case")
    case, pub = reading_prompt("case", [("", "x")]), reading_prompt("publication", [("", "x")], "appendicitis")
    shared = case.split("- quote:")[0].split("one object:\n", 1)[1].split("- state:")[0]
    assert shared in pub                                  # the same description, word for word
    assert "give no code" in case and "hpo_term" not in case


def test_other_words_for_one_finding_are_paired_when_each_interpreter_says_it_of_the_other():
    """Each interpreter is shown, once, what only the other named (name and sentence) and its own unpaired cards
    in those sentences; "the same" pairs only when both say it of each other."""
    readings = {"nemotron": [c.read(CRP_A, "case")], "muse": [c.read(CRP_B, "case")]}
    shown = {}

    def ask(reader, prompt):
        shown[reader] = prompt
        return '[{"item": 1, "same_as": "a"}]'
    agreed, asked = ag.agree(readings, "case", ask)
    assert len(agreed) == 1 and agreed[0]["paired_by"] == "same_in_context" and agreed[0]["doubts"] == []
    assert len(asked) == 2 and "elevated CRP" in shown["muse"] and "a. elevated C-reactive protein" in shown["muse"]
    assert '"increased"' not in shown["muse"].split("every field of")[0]      # the other's fields are never shown
    # one says "the same", the other says the sentence does not state it: nothing paired, the second a doubt
    answers = {"muse": '[{"item": 1, "same_as": "a"}]', "nemotron": '[{"item": 1, "same_as": null, "card": null}]'}
    agreed, _ = ag.agree(readings, "case", lambda r, p: answers[r])
    assert [x["paired_by"] for x in agreed] == ["one_interpreter", "one_interpreter"]
    by = {x["readers"][0]: x for x in agreed}
    assert "not_stated_for_the_other" in by["muse"]["doubts"] and "not_stated_for_the_other" not in by["nemotron"]["doubts"]


def test_a_finding_only_one_named_is_read_again_by_the_other_independently():
    """Muse never wrote Nemotron's finding: shown its name and sentence, it gives its own card, and the two
    readings are compared field by field (25 November 2026, after the sixth run)."""
    readings = {"nemotron": [c.read(CRP_A, "case"), c.read(FEVER, "case")], "muse": [c.read(CRP_A, "case")]}
    own = {**FEVER, "state": "absent", "observed_by": "examination", "value": None}
    agreed, _ = ag.agree(readings, "case",
                         lambda r, p: json.dumps([{"item": 1, "same_as": None, "card": own}]) if r == "muse" else "[]")
    fever = next(x for x in agreed if x["name"] == "fever")
    assert fever["paired_by"] == "read_again_by_the_other" and fever["readers"] == ["nemotron", "muse"]
    assert fever["state"] == "absent" and fever["doubts"] == []


def test_a_field_read_differently_is_a_doubt_never_resolved():
    other = {**CRP_A, "value": {"number": 5.2, "unit": "mg/dL"}}
    agreed, _ = ag.agree({"nemotron": [c.read(CRP_A, "case")], "muse": [c.read(other, "case")]}, "case", None)
    assert agreed[0]["paired_by"] == "same_name" and agreed[0]["value"] is None and agreed[0]["doubts"] == ["value"]


def test_cards_of_different_sentences_are_never_paired():
    elsewhere = {**CRP_B, "name": "elevated CRP", "what": "CRP", "quote": "Paziente con dolore in fossa iliaca destra."}
    agreed, _ = ag.agree({"a": [c.read(CRP_A, "case")], "b": [c.read(elsewhere, "case")]}, "case",
                         lambda r, p: '[{"item": 1, "same_as": "a"}]')
    assert len(agreed) == 2 and {x["paired_by"] for x in agreed} == {"one_interpreter"}


def test_a_patients_text_is_read_only_on_this_machine():
    far = lambda p: "[]"   # noqa: E731
    with pytest.raises(ValueError, match="only on this machine"):
        read_text([("", REPORT)], "case", {"remote": far})


def test_the_same_procedure_codes_afterwards_and_a_patient_never_writes_the_thesaurus():
    chats = {"nemotron": local({"reading": [CRP_A, FEVER],
                                "questions": {"You and another reader": '[{"item": 1, "same_as": "a"}]', "Findings agreed": '[{"item": 1, "same": 1}]'}}),
             "muse": local({"reading": [CRP_B, FEVER],
                            "questions": {"You and another reader": '[{"item": 1, "same_as": "a"}]', "Findings agreed": '[{"item": 1, "same": 1}]'}})}
    thesaurus = Thesaurus()
    got = read_text([("", REPORT)], "case", chats, hpo=HPO, thesaurus=thesaurus)
    by = {x["name"]: x for x in got["cards"]}
    assert by["fever"]["concept"]["code"] == "HP:0001945" and by["fever"]["state"] == "absent"
    assert by["elevated CRP"]["concept"]["code"] == "HP:0011227"     # chosen by both, in the sentence
    assert thesaurus.concepts == {}                                   # a patient's sentence is never knowledge
    assert got["format"]["nemotron"] == {"objects": 2}


def test_a_publication_writes_what_both_agreed_into_the_thesaurus_and_finds_it_again():
    pub_a = {k: v for k, v in CRP_A.items() if k not in ("state", "temporality", "subject")}
    pub_a.update(words="CRP elevated", quote="An elevated CRP was found in 49% of patients.", value=None,
                 direction=None, compares="disease_vs_without", measures=None)
    pub_b = {**pub_a, "name": "elevated C-reactive protein", "what": "C-reactive protein"}
    chats = {"n": lambda p: json.dumps([pub_a]) if p.startswith("You read") else (
                 '[{"item": 1, "same_as": "a"}]' if p.startswith("You and another") else '[{"item": 1, "same": 1}]'),
             "m": lambda p: json.dumps([pub_b]) if p.startswith("You read") else (
                 '[{"item": 1, "same_as": "a"}]' if p.startswith("You and another") else '[{"item": 1, "same": 1}]')}
    thesaurus = Thesaurus()
    text = [("Abstract", "An elevated CRP was found in 49% of patients.")]
    got = read_text(text, "publication", chats, subject="appendicitis", hpo=HPO, thesaurus=thesaurus,
                    evidence={"source": "PMID:1"})
    assert got["cards"][0]["concept"]["code"] == "HP:0011227"
    concept = thesaurus.concepts["HP:0011227"]
    assert [x["label"] for x in concept.labels] == ["elevated CRP", "elevated C-reactive protein"]
    assert concept.labels[0]["source"] == "PMID:1"
    again = read_text(text, "publication", chats, subject="appendicitis", hpo=HPO, thesaurus=thesaurus)
    assert again["cards"][0]["concept"]["how"] == "thesaurus"


def test_when_one_interpreter_gives_no_reading_nothing_is_agreed_and_nothing_is_coded():
    """(25 November 2026, after the fifth run) Nemotron's answer was cut off: Muse's cards had counted as agreed
    with themselves and written the thesaurus."""
    chats = {"nemotron": local({"reading": "cut"}), "muse": local({"reading": [CRP_B]})}
    chats["nemotron"] = (lambda p: '[{"words": "PCR 52 mg/L", "name": ')
    chats["nemotron"].local_only = True
    thesaurus = Thesaurus()
    got = read_text([("", REPORT)], "case", chats, hpo=HPO, thesaurus=thesaurus)
    assert got["not_read"] == {"nemotron": "no_json_list"} and got["asked"] == []
    [card] = got["cards"]
    assert card["paired_by"] == "one_interpreter" and card["concept"]["how"] == "not_coded:one_interpreter"


def test_a_label_agreed_by_one_interpreter_alone_never_enters_a_thesaurus():
    from melampo.reading.thesaurus import signature
    other = Thesaurus.from_rows([{"id": "MLP:1", "name": "ultrasound", "code": "", "labels": [
        {"label": "ultrasound", "signature": signature({}), "readers": ["muse"]},
        {"label": "abdominal ultrasound", "signature": signature({}), "readers": ["nemotron", "muse"]}]}])
    mine = Thesaurus()
    mine.merge(other)
    assert [x["label"] for x in mine.concepts["MLP:1"].labels] == ["abdominal ultrasound"]


def test_a_field_one_reading_does_not_give_is_not_a_field_read_otherwise():
    """(26 November 2026, after the seventh run) 270 second readings came without a category: a required field
    given by one reading alone is that reading's, with a doubt; a field both give differently stays a doubt."""
    a = c.read(CRP_A, "case")
    b = c.read({k: v for k, v in CRP_A.items() if k not in ("category", "specific")}, "case")
    card = ag.merged([("nemotron", a), ("muse", b)], "case")
    assert card["category"] == "manifestation" and card["specific"] is True
    assert sorted(card["doubts"]) == ["category_given_by_one", "specific_given_by_one"]
    other = ag.merged([("nemotron", a), ("muse", {**a, "category": "test"})], "case")
    assert other["category"] is None and other["doubts"] == ["category"]
