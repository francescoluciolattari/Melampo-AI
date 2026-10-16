"""The report router: the kind of report first, then the interpreters (16 October 2026)."""

import pytest

from melampo.data import report_router as rr

BIOPSY = """ESAME ISTOLOGICO
Materiale: biopsia epatica.
Macroscopia: frustolo di 2 cm.
Microscopia: infiltrato linfocitario portale.
Immunoistochimica:
Ki-67          20%
ER             90%
HER2           2+
Diagnosi: epatite cronica attiva.
"""
BLOOD = """EMOCROMO
Emoglobina        11,8   g/dL      12,0 - 16,0   L
Leucociti         7,20   10^3/uL   4,00 - 10,00
"""
CT = """TC TORACE CON MDC
Tecnica: scansioni assiali.
Reperto: nodulo di 8 mm al lobo superiore destro.
Conclusioni: nodulo polmonare da rivalutare.
"""
SPE = """ELETTROFORESI SIEROPROTEICA
Albumina           55,2   %        55,8 - 66,1   L
Alfa 1              4,1   %        2,9 - 4,9
Alfa 2             12,3   %        7,1 - 11,8    H
Beta               11,0   %        8,4 - 13,1
Gamma              17,4   %        11,1 - 18,8
"""


def answer(*kinds):
    return lambda prompt: '{"kinds": [' + ",".join(f'{{"kind": "{k}", "from": 1, "to": 3}}' for k in kinds) + "]}"


def test_the_model_and_the_cues_agree_on_a_biopsy_and_its_tables_are_read_without_its_sentences():
    d = rr.route(BIOPSY, chat=answer("pathology"))
    assert "pathology" in d.kinds and d.agreement in ("agree", "model_adds", "cues_add")
    assert {"pathology", "lab_table", "narrative_reader"} <= set(d.interpreters)
    lab = rr.lab_reading(BIOPSY, d)
    assert [r.analyte for r in lab.results] == ["Ki-67", "ER", "HER2"]
    assert lab.results[2].qualitative_value == "score:2+"
    assert lab.unparsed_lines == ()   # "Macroscopia: frustolo di 2 cm." is a sentence, not a lost row


@pytest.mark.parametrize("text,kind,interpreter", [
    (BLOOD, "laboratory", "lab_table"), (CT, "imaging_report", "narrative_reader"),
])
def test_each_kind_goes_to_its_interpreter(text, kind, interpreter):
    d = rr.route(text, chat=answer(kind))
    assert d.agreement == "agree" and interpreter in d.interpreters and not d.needs_review


def test_without_a_model_the_cues_route_and_nothing_is_dropped():
    d = rr.route(SPE)
    assert d.agreement == "no_model" and not d.needs_review
    assert {"lab_table", "electrophoresis_trace"} <= set(d.interpreters)
    assert any("trace present and not read" in n for n in d.notes)


def test_an_unusable_answer_or_an_unreachable_model_falls_back_to_the_cues_and_asks_for_review():
    d = rr.route(BLOOD, chat=lambda p: "this is a blood test")
    assert d.model_kinds is None and d.needs_review and "lab_table" in d.interpreters

    def down(prompt):
        raise TimeoutError

    d = rr.route(BLOOD, chat=down)
    assert d.needs_review and any(n.startswith("router_model_error") for n in d.notes)


def test_a_kind_only_the_model_sees_is_read_too_and_a_contradiction_is_flagged():
    d = rr.route(CT, chat=answer("imaging_report", "laboratory"))
    assert "lab_table" in d.interpreters and "narrative_reader" in d.interpreters
    d = rr.route(BLOOD, chat=answer("pathology"))
    assert d.agreement == "both_differ" and d.needs_review
    assert {"lab_table", "pathology"} <= set(d.interpreters)


def test_an_unknown_kind_in_the_answer_is_not_accepted():
    assert rr.parse_kind_answer('{"kinds": [{"kind": "cardiology"}]}', 10) is None
    got = rr.parse_kind_answer('{"kinds": [{"kind": "laboratory", "from": 5, "to": 99}]}', 10)
    assert got == [{"kind": "laboratory", "from": 1, "to": 10}]


def test_the_prompt_asks_only_for_the_kind():
    p = rr.kind_prompt(BLOOD)
    assert "do not read, interpret or repeat any value" in p and "  1| EMOCROMO" in p


def test_an_external_model_is_refused():
    with pytest.raises(ValueError):
        rr.local_chat("https://api.example.com/v1", "m")
    assert callable(rr.local_chat("http://localhost:8000/v1", "m"))


def test_images_add_the_imaging_interpreter():
    assert "imaging" in rr.route(CT, images=2).interpreters
