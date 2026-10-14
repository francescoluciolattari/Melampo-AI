"""The question to a chat model about the phrase of a mention, and the check of its answer."""

import json

from melampo.evaluation import llm_reading as lr

S = "Five strains were chosen to represent low, mid, and high brain weights (Fig 1A)."
AT = S.index("brain")
END = AT + len("brain")


def test_the_mark_means_only_the_word_and_examples_use_the_same_notation():
    assert lr.mark(S, AT, END).count("<w>brain</w> weights") == 1
    assert lr.remark_example("Late effects may appear as [[brain tumors]] or impairment.", "brain") == (
        "Late effects may appear as <w>brain</w> tumors or impairment.")
    p = lr.prompt(S, AT, END, examples=[{"marked": "x <w>brain</w> tumors", "phrase": "brain tumors", "type": "disease or finding"}])
    assert "[[" not in p and "never the answer" in p and "phrase «brain tumors»" in p
    assert "uncertain" in p and "inherent_location" in p


def test_a_bare_word_followed_by_more_words_gets_one_follow_up():
    c = lr.check({"phrase": "brain", "head": "brain", "type": "anatomical structure", "role": "structure"}, S, AT, END)
    assert c.valid and "weights" in c.follow_up
    whole = lr.check({"phrase": "brain weights", "head": "weights", "type": "measurement or property",
                      "role": "inherent_location"}, S, AT, END)
    assert whole.valid and not whole.follow_up and whole.span == (AT, END + 8)


def test_no_follow_up_when_nothing_can_continue_the_phrase():
    s = "The head with brain was placed in a vial."
    at = s.index("brain")
    c = lr.check({"phrase": "brain", "head": "brain", "type": "anatomical structure", "role": "structure"}, s, at, at + 5)
    assert c.valid and not c.follow_up
    assert lr.next_words("the liver, then", 9) == []


def test_malformed_answers_are_rejected_with_their_reason():
    bad = lr.check({"phrase": "lung weights", "head": "x", "type": "nonsense", "role": "site"}, S, AT, END)
    assert not bad.valid
    assert any("role" in p for p in bad.problems) and any("piece of the sentence" in p for p in bad.problems)
    # a phrase elsewhere in the sentence that does not cover the marked word
    s = "brain weights and brain volume"
    other = lr.check({"phrase": "brain volume", "head": "volume", "type": "measurement or property",
                      "role": "inherent_location"}, s, 0, 5)
    assert not other.valid


def test_ask_follows_up_once_and_counts_the_change():
    answers = iter([
        {"phrase": "brain", "head": "brain", "type": "anatomical structure", "role": "structure"},
        {"phrase": "brain weights", "head": "weights", "type": "measurement or property", "role": "inherent_location"},
    ])
    seen = []

    def chat(prompt):
        seen.append(prompt)
        return json.dumps(next(answers))

    got = lr.ask(chat, lr.prompt(S, AT, END), S, AT, END)
    assert len(seen) == 2 and "followed by «weights»" in seen[1]
    assert got.asked_again and got.changed and got.final.role == "inherent_location"


def test_an_unreadable_answer_is_asked_again_then_lost():
    got = lr.ask(lambda _p: "no json here", lr.prompt(S, AT, END), S, AT, END)
    assert got.asked_again and not got.final.valid


def test_balanced_examples_keep_a_longer_and_a_bare_phrase():
    ex = [{"phrase": "brain"}, {"phrase": "brain"}, {"phrase": "brain"}, {"phrase": "brain weights"}]
    chosen = lr.balanced(ex, "brain", 2)
    assert {e["phrase"] for e in chosen} == {"brain", "brain weights"}


def test_a_copied_mark_is_taken_out_of_the_phrase():
    from melampo.evaluation import llm_reading as lr

    s = "There is minimal opacity within the lingula favored as atelectasis."
    a = s.index("lingula")
    got = lr.check({"phrase": "the <w>lingula</w>", "head": "<w>lingula</w>", "type": "anatomical structure",
                    "role": "structure"}, s, a, a + len("lingula"))
    assert got.valid and got.answer["phrase"] == "the lingula"


def test_a_placeholder_left_out_at_the_edge_still_covers_the_marked_words():
    from melampo.evaluation import llm_reading as lr

    s = "A density overlying the left  costophrenic XXXX is due to soft tissues."
    m = "left  costophrenic XXXX"
    a = s.index(m)
    got = lr.check({"phrase": "left costophrenic", "head": "costophrenic", "type": "anatomical structure",
                    "role": "structure"}, s, a, a + len(m))
    assert got.valid


def test_articles_and_prepositions_at_the_edge_and_spaced_hyphens():
    from melampo.evaluation import llm_reading as lr

    s = "La biopsia evidenziava depositi nelle membrane basali glomerulari."
    m = "nelle membrane basali"
    a = s.index(m)
    got = lr.check({"phrase": "membrane basali glomerulari", "head": "membrane", "type": "anatomical structure",
                    "roles": ["structure"]}, s, a, a + len(m))
    assert got.valid
    s = "AngioTac dell'asse mesenterico- portale."
    m = "asse mesenterico- portale"
    a = s.index(m)
    got = lr.check({"phrase": "asse mesenterico-portale", "head": "asse", "type": "anatomical structure",
                    "roles": ["structure"]}, s, a, a + len(m))
    assert got.valid
