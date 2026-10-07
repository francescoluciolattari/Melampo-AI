"""Forms that can mean more than one thing, found from the data; the language of an abbreviation;
the contextual check of the models for the forms the data flags."""

import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory import form_ambiguity as fa

_DATA = Path(__file__).resolve().parents[1] / "data" / "linking"


@pytest.fixture(scope="module")
def lexicon():
    return al.Lexicon.from_json(
        json.loads((_DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )


@pytest.fixture(scope="module")
def flags(lexicon):
    return fa.audit(lexicon)


def _signals(flags, text):
    return flags.get(al.normalise(text), fa.FormFlag((), frozenset(), ())).signals


def test_a_name_that_lost_its_head_noun_is_flagged(flags):
    assert "head_dropped" in _signals(flags, "left paraspinal muscles")
    assert "head_dropped" in _signals(flags, "left hip bone")


def test_abbreviations_in_capitals_are_flagged(flags):
    assert "short" in _signals(flags, "SVC")
    assert "short" in _signals(flags, "LID")


def test_a_spatial_prefix_on_an_adjective_flags_a_region(lexicon):
    flags = fa.audit(lexicon, [("aorta sottorenale", "aorta"), ("fegato", "liver")])
    assert "region" in _signals(flags, "aorta sottorenale")


def test_an_ordinary_full_name_is_not_flagged(flags):
    assert al.normalise("fegato") not in flags
    assert al.normalise("spleen") not in flags


def test_a_key_shared_by_two_classes_is_flagged():
    data = {
        "classes": {
            "a": {"it": ["xyz"], "en": []},
            "b": {"it": ["xyz"], "en": []},
        }
    }
    flags = fa.audit(al.Lexicon.from_json(data))
    assert "shared" in flags[al.normalise("xyz")].signals


def test_exposed_leaves_out_forms_that_have_a_sense_profile(lexicon):
    flags = fa.audit(lexicon)
    everything = fa.exposed(flags, [])
    covered = fa.exposed(flags, ["paraspinal"])
    assert len(covered) <= len(everything)


# --- the language of an abbreviation ----------------------------------------------------------


@pytest.fixture(scope="module")
def linker(lexicon):
    return al.AnatomyLinker(lexicon, [], {})


def test_an_italian_abbreviation_in_an_english_sentence_is_not_the_structure(linker):
    result = linker.link(
        "lid", "Conjunctival prolapse was observed through the lid fissure."
    )
    assert result.status == al.ABSTAINED
    assert result.reason == "abbreviation_of_another_language"


def test_the_same_abbreviation_in_an_italian_sentence_is_accepted(linker):
    result = linker.link("LID", "Nodulo nel lobo inferiore destro (LID) del polmone.")
    assert (result.status, result.cid) == (al.ACCEPTED, "lung_lower_lobe_right")


def test_an_english_abbreviation_in_an_english_sentence_is_accepted(linker):
    result = linker.link(
        "SVC", "The catheter tip is in the mid SVC and there is no effusion."
    )
    assert (result.status, result.cid) == (al.ACCEPTED, "superior_vena_cava")


def test_a_sentence_of_unknown_language_never_clashes(linker):
    assert linker.link("LID", "LID.").status == al.ACCEPTED


def test_a_full_name_is_not_held_to_the_language_rule(linker):
    result = linker.link("thyroid", "La thyroid appare regolare e ben delimitata.")
    assert result.reason != "abbreviation_of_another_language"


# --- the models read the sentence for the forms the data flags ----------------------------------


class Counter:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        assert "<tgt>" in prompt
        return self.answer


def _linker(lexicon, flags, a, b, **kw):
    chats = {"a": a, "b": b}
    return al.AnatomyLinker(
        lexicon,
        [],
        {},
        chats=chats,
        ambiguous=fa.keys(flags),
        verify=kw.get("verify", True),
    )


def test_both_models_confirm_so_the_link_stands(lexicon, flags):
    a, b = Counter("YES"), Counter("Yes.")
    result = _linker(lexicon, flags, a, b).link(
        "thyroid", "Ultrasound of the thyroid shows a nodule."
    )
    assert (result.status, result.cid) == (al.ACCEPTED, "thyroid_gland")
    assert (a.calls, b.calls) == (1, 1)
    assert [e.stream for e in result.trace if e.stream == "verify"] == ["verify"]


@pytest.mark.parametrize(
    ("first", "second", "reason"),
    [
        ("YES", "NO", "context_check_failed:no"),
        ("NO", "NO", "context_check_failed:no"),
        ("YES", "UNSURE", "context_check_failed:unsure"),
        ("YES", "maybe", "context_check_failed:unsure"),
    ],
)
def test_one_dissent_or_doubt_is_an_abstention_with_the_reason(
    lexicon, flags, first, second, reason
):
    result = _linker(lexicon, flags, Counter(first), Counter(second)).link(
        "thyroid", "Thyroid, parathyroid and vitamin D were normal."
    )
    assert result.status == al.ABSTAINED and result.reason == reason
    assert result.stage == "verify"


def test_a_form_the_data_does_not_flag_is_not_asked(lexicon, flags):
    a, b = Counter("NO"), Counter("NO")
    result = _linker(lexicon, flags, a, b).link("fegato", "Il fegato è nei limiti.")
    assert result.status == al.ACCEPTED
    assert (a.calls, b.calls) == (0, 0)


def test_verification_is_off_unless_asked_for(lexicon, flags):
    a, b = Counter("NO"), Counter("NO")
    result = _linker(lexicon, flags, a, b, verify=False).link(
        "thyroid", "Thyroid, parathyroid and vitamin D were normal."
    )
    assert result.status == al.ACCEPTED
    assert (a.calls, b.calls) == (0, 0)


def test_a_form_with_a_sense_profile_is_read_by_its_profile_not_asked(lexicon, flags):
    a, b = Counter("NO"), Counter("NO")
    result = _linker(lexicon, flags, a, b).link(
        "left paraspinal", "Fatty atrophy of the left paraspinal."
    )
    assert result.status == al.ACCEPTED
    assert (a.calls, b.calls) == (0, 0)


def test_an_english_abbreviation_in_an_italian_sentence_is_tolerated(linker):
    # Italian reports use CCA, SVC, IVC: the rule is not symmetric
    result = linker.link(
        "CCA sn", "Angio-TC: il calibro e l'opacizzazione di CCA sn sono conservati."
    )
    assert (result.status, result.cid) == (al.ACCEPTED, "common_carotid_artery_left")


# --- a model that does not answer is an abstention, never a vote ----------------------------------


def _down(prompt):
    raise RuntimeError("HTTP 429 from model-a")


def test_a_model_that_does_not_answer_makes_the_verify_check_abstain(lexicon, flags):
    result = _linker(lexicon, flags, _down, Counter("YES")).link(
        "thyroid", "Ultrasound of the thyroid shows a nodule."
    )
    assert result.status == al.ABSTAINED and result.reason == "model_unavailable"
    assert result.stage == "verify"
    assert "429" in result.votes["error"]
