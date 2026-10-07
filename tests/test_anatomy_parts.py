import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory import anatomy_parts as ap

DATA = Path(__file__).resolve().parents[1] / "data" / "linking"


@pytest.fixture(scope="module")
def lexicon():
    return al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )


@pytest.fixture(scope="module")
def table(lexicon):
    return ap.PartTable.from_json(
        json.loads((DATA / "anatomy_parts.json").read_text("utf-8")), lexicon
    )


@pytest.mark.parametrize(
    ("mention", "cid", "relation"),
    [
        ("sigma", "colon", "part_of"),
        ("colon discendente", "colon", "part_of"),
        ("sigmoid colon", "colon", "part_of"),
        ("lingula", "lung_upper_lobe_left", "part_of"),
        ("ileo terminale", "small_bowel", "part_of"),
        ("emibacino sinistro", "hip_left", "approx"),
        ("osso iliaco destro", "hip_right", "equal"),
        ("profilo cardiaco", "heart", "contour_of"),
        ("Cardiac silhouette", "heart", "contour_of"),
        ("processo odontoideo", "vertebrae_C2", "part_of"),
        ("testa femorale sinistra", "femur_left", "part_of"),
        ("right femoral neck", "femur_right", "part_of"),
        ("polo superiore della milza", "spleen", "part_of"),
        ("polo inferiore del rene destro", "kidney_right", "part_of"),
        ("lobo destro del fegato", "liver", "part_of"),
        ("left thyroid lobe", "thyroid_gland", "part_of"),
        ("esofago distale", "esophagus", "part_of"),
        ("Aorta ascendente", "aorta", "part_of"),
        ("muscolo psoas destro", "iliopsoas_right", "part_of"),
    ],
)
def test_known_parts_link_to_their_whole_with_the_relation(
    table, mention, cid, relation
):
    link = table.resolve(mention)
    assert isinstance(link, ap.PartLink)
    assert (link.cid, link.relation) == (cid, relation)


@pytest.mark.parametrize(
    "mention",
    [
        "testa del fegato",
        "coda della milza",
        "polo del cuore",
        "lobo del pancreas",
        "xyz distale",
    ],
)
def test_a_part_word_with_a_whole_it_does_not_belong_to_is_not_linked(table, mention):
    assert table.resolve(mention) is None


def test_a_part_of_a_paired_structure_needs_the_side(table):
    assert table.resolve("emibacino") == "part_of_a_paired_structure_without_a_side"
    assert (
        table.resolve("testa femorale") == "part_of_a_paired_structure_without_a_side"
    )
    assert table.resolve("femoral head").__class__ is str


def test_an_adjective_two_structures_share_is_not_linked(table):
    assert not isinstance(table.resolve("iliaco destro"), ap.PartLink)
    assert table.resolve("osso iliaco destro").cid == "hip_right"
    assert table.resolve("muscolo iliaco destro").cid == "iliopsoas_right"
    assert table.resolve("right iliac muscle").cid == "iliopsoas_right"


@pytest.mark.parametrize(
    "mention",
    [
        "porta hepatis",
        "ilo epatico",
        "canale midollare",
        "spinal canal",
        "vertebral canal",
    ],
)
def test_known_traps_are_never_linked(table, mention):
    assert isinstance(table.resolve(mention), str)


def test_the_wrapper_a_carico_di_is_removed():
    assert ap.strip_wrapper("a carico della colecisti") == "colecisti"
    assert ap.strip_wrapper("a carico dell'esofago") == "esofago"
    assert ap.strip_wrapper("colecisti") == "colecisti"


def _linker(lexicon, table, chats=None, pool=()):
    pool_, equivalent = al.build_pool(lexicon, list(pool))
    return al.AnatomyLinker(lexicon, pool_, equivalent, parts=table, chats=chats or {})


def test_the_linker_reports_the_relation(lexicon, table):
    result = _linker(lexicon, table).link("sigma", "Diverticoli del sigma.")
    assert (result.status, result.cid, result.stage) == (al.ACCEPTED, "colon", "parts")
    assert (result.relation, result.part) == ("part_of", "sigma")


def test_a_lexicon_name_is_still_equal(lexicon, table):
    result = _linker(lexicon, table).link("colon", "Colon nei limiti.")
    assert (result.stage, result.relation) == ("lexicon", "equal")


def test_a_carico_di_links_the_organ_named(lexicon, table):
    result = _linker(lexicon, table).link(
        "a carico della colecisti", "Calcoli a carico della colecisti."
    )
    assert (result.status, result.cid) == (al.ACCEPTED, "gallbladder")


def test_a_trap_is_an_abstention_with_its_reason(lexicon, table):
    result = _linker(lexicon, table).link(
        "porta hepatis", "Linfonodi alla porta hepatis."
    )
    assert (result.status, result.reason) == (
        al.ABSTAINED,
        "hilum_is_not_the_organ_or_its_vessel",
    )


def test_every_target_of_the_table_is_a_class_of_the_lexicon(lexicon, table):
    data = json.loads((DATA / "anatomy_parts.json").read_text("utf-8"))
    for entry in data["direct"]:
        assert table._family(entry["whole"]), entry
    for entry in data["parts"]:
        for whole in entry["wholes"]:
            assert table._family(whole), (entry, whole)


def test_no_name_of_the_table_is_also_a_different_lexicon_class(lexicon, table):
    """A table name that the lexicon already knows must agree with it (the lexicon answers first)."""
    data = json.loads((DATA / "anatomy_parts.json").read_text("utf-8"))
    for entry in data["direct"]:
        for name in entry["names"]:
            hits, how = lexicon.recognise(name, name)
            if how == "recognised":
                assert hits[0] in table._family(entry["whole"]), (name, hits)


# ---- the translation stage ------------------------------------------------

_OBO = """[Term]
id: UBERON:0001159
name: sigmoid colon
xref: FMA:1

[Term]
id: UBERON:0000995
name: uterus
xref: FMA:2

[Term]
id: UBERON:0002119
name: left ovary
xref: FMA:3

[Term]
id: UBERON:0002118
name: right ovary
xref: FMA:4

[Term]
id: UBERON:0001165
name: wall of stomach
xref: FMA:5
"""


def _translating(lexicon, answers):
    pool = al.load_obo_terms(_OBO.splitlines())
    chats = {
        name: (lambda a: lambda prompt: a)(answer) for name, answer in answers.items()
    }
    pool_, equivalent = al.build_pool(lexicon, pool)
    return al.AnatomyLinker(lexicon, pool_, equivalent, chats=chats)


def test_two_models_translating_to_the_same_pool_name_link_an_italian_mention(lexicon):
    linker = _translating(lexicon, {"a": "uterus", "b": "Uterus."})
    result = linker.link("utero", "Utero in antiversoflessione.")
    assert (result.status, result.cid, result.stage) == (
        al.ACCEPTED,
        "UBERON:0000995",
        "translation",
    )


def test_translations_that_differ_abstain(lexicon):
    result = _translating(lexicon, {"a": "uterus", "b": "sigmoid colon"}).link(
        "utero", "Utero."
    )
    assert (result.status, result.reason) == (
        al.ABSTAINED,
        "translations_do_not_agree_on_one_concept",
    )


def test_a_translation_that_is_not_a_pool_name_abstains(lexicon):
    result = _translating(lexicon, {"a": "womb", "b": "womb"}).link("utero", "Utero.")
    assert result.status == al.ABSTAINED


def test_unsure_or_empty_translations_abstain(lexicon):
    result = _translating(lexicon, {"a": "UNSURE", "b": "uterus"}).link(
        "utero", "Utero."
    )
    assert (result.status, result.reason) == (al.ABSTAINED, "a_model_did_not_translate")
    result = _translating(lexicon, {"a": "", "b": ""}).link("utero", "Utero.")
    assert result.reason == "a_model_did_not_translate"


def test_the_side_of_the_mention_must_agree_with_the_translated_term(lexicon):
    # both models drop the side: "left ovary" is the only pool name, the mention says right
    result = _translating(lexicon, {"a": "left ovary", "b": "left ovary"}).link(
        "ovaio destro", "Cisti dell'ovaio destro."
    )
    assert result.status == al.ABSTAINED
    ok = _translating(lexicon, {"a": "right ovary", "b": "right ovary"}).link(
        "ovaio destro", "Cisti dell'ovaio destro."
    )
    assert (ok.status, ok.cid) == (al.ACCEPTED, "UBERON:0002118")


def test_a_translation_with_a_different_tissue_word_is_not_the_organ(lexicon):
    result = _translating(
        lexicon, {"a": "wall of stomach", "b": "wall of stomach"}
    ).link("stomaco", "Stomaco.")
    # "stomaco" is a lexicon class: it is recognised before any translation
    assert (result.stage, result.cid) == ("lexicon", "stomach")


def test_the_translation_prompt_marks_the_mention(lexicon):
    seen = []

    def chat(prompt):
        seen.append(prompt)
        return "uterus"

    pool_, equivalent = al.build_pool(lexicon, al.load_obo_terms(_OBO.splitlines()))
    al.AnatomyLinker(lexicon, pool_, equivalent, chats={"a": chat, "b": chat}).link(
        "utero", "Utero in antiversoflessione."
    )
    assert "<tgt>Utero</tgt>" in seen[0]


def test_colluding_wrong_translations_cannot_pass_the_traps(lexicon, table):
    linker = _translating(lexicon, {"a": "sigmoid colon", "b": "sigmoid colon"})
    linker.parts = table
    result = linker.link("canale midollare", "Canale midollare ampio.")
    assert result.status == al.ABSTAINED


# ---- found by the adversarial review of the parts table -----------------------


@pytest.mark.parametrize(
    "mention",
    [
        "lume esofageo",
        "lume del colon",
        "gastric lumen",
        "aortic lumen",
        "lume tracheale",
    ],
)
def test_a_lumen_is_never_the_organ(lexicon, table, mention):
    result = _linker(lexicon, table).link(mention, f"{mention} regolare.")
    assert result.status == al.ABSTAINED


@pytest.mark.parametrize(
    ("mention", "cid", "relation"),
    [
        ("corpo gastrico", "stomach", "part_of"),
        ("corpo della colecisti", "gallbladder", "part_of"),
        ("parete aortica", "aorta", "part_of"),
        ("parete cardiaca", "heart", "part_of"),
        ("parenchima epatico", "liver", "part_of"),
        ("colonic wall", "colon", "part_of"),
    ],
)
def test_the_wall_or_body_of_a_structure_is_a_part_not_the_whole(
    lexicon, table, mention, cid, relation
):
    result = _linker(lexicon, table).link(mention, f"{mention} regolare.")
    assert (result.status, result.cid, result.relation) == (al.ACCEPTED, cid, relation)


@pytest.mark.parametrize(
    "mention", ["muscolo sternale", "sternal muscle", "sternal bone"]
)
def test_a_tissue_word_that_changes_the_structure_is_not_dropped(
    lexicon, table, mention
):
    result = _linker(lexicon, table).link(mention, f"{mention} regolare.")
    assert result.status == al.ABSTAINED


@pytest.mark.parametrize(
    "mention",
    [
        "splenica",
        "epatica",
        "gastrica",
        "cardiaca",
        "aortica",
        "pancreatica",
        "tiroidea",
        "vescicale",
        "cranica",
        "colica",
    ],
)
def test_a_bare_adjective_names_no_structure(lexicon, table, mention):
    assert _linker(lexicon, table).link(mention, f"{mention}.").status == al.ABSTAINED


@pytest.mark.parametrize("mention", ["ventricolo", "ventricle", "atrio", "atrium"])
def test_a_bare_ventricle_or_atrium_is_not_linked_to_the_heart(lexicon, table, mention):
    assert not isinstance(table.resolve(mention), ap.PartLink)


def test_the_left_ventricle_of_a_brain_sentence_is_not_the_heart(lexicon, table):
    result = _linker(lexicon, table).link(
        "ventricolo sinistro", "TC cranio: dilatazione del ventricolo sinistro."
    )
    assert result.status == al.ABSTAINED
    ok = _linker(lexicon, table).link(
        "ventricolo sinistro", "Ipertrofia del ventricolo sinistro."
    )
    assert (ok.status, ok.cid) == (al.ACCEPTED, "heart")


@pytest.mark.parametrize(
    "mention",
    ["lingula destra", "right lingula", "segmento laterale del lobo medio sinistro"],
)
def test_a_side_that_contradicts_the_structure_is_not_dropped(lexicon, table, mention):
    result = _linker(lexicon, table).link(mention, f"{mention}.")
    assert result.status == al.ABSTAINED


def test_the_wrapper_needs_a_word_boundary():
    assert (
        ap.strip_wrapper("a carico diffuso del fegato") == "a carico diffuso del fegato"
    )


# ---- translation guards found by the adversarial review ------------------------

_OBO2 = """[Term]
id: UBERON:0009853
name: body of uterus
xref: FMA:11

[Term]
id: UBERON:0000995
name: uterus
xref: FMA:12

[Term]
id: UBERON:0002098
name: apex of heart
xref: FMA:13

[Term]
id: UBERON:0002170
name: apex of lung
xref: FMA:14

[Term]
id: UBERON:0001222
name: neck of uterus
xref: FMA:15
"""


def _t(lexicon, term):
    pool = al.load_obo_terms(_OBO2.splitlines())
    pool_, equivalent = al.build_pool(lexicon, pool)
    chats = {n: (lambda a: lambda prompt: a)(term) for n in ("a", "b")}
    return al.AnatomyLinker(lexicon, pool_, equivalent, chats=chats)


@pytest.mark.parametrize(
    ("mention", "term"),
    [
        ("collo dell'utero", "body of uterus"),
        ("corpo dell'utero", "uterus"),
        ("fondo dell'utero", "uterus"),
        ("utero", "body of uterus"),
        ("apice del polmone", "apex of heart"),
        ("apice del cuore", "apex of lung"),
    ],
)
def test_colluding_translations_that_swap_a_part_or_an_organ_do_not_pass(
    lexicon, mention, term
):
    assert (
        _t(lexicon, term).link(mention, f"{mention} regolare.").status == al.ABSTAINED
    )


@pytest.mark.parametrize(
    ("mention", "term", "cid"),
    [
        ("collo dell'utero", "neck of uterus", "UBERON:0001222"),
        ("corpo dell'utero", "body of uterus", "UBERON:0009853"),
        ("utero", "uterus", "UBERON:0000995"),
        ("apice del polmone", "apex of lung", "UBERON:0002170"),
    ],
)
def test_faithful_translations_still_link(lexicon, mention, term, cid):
    result = _t(lexicon, term).link(mention, f"{mention} regolare.")
    assert (result.status, result.cid) == (al.ACCEPTED, cid)


@pytest.mark.parametrize(
    ("mention", "sentence", "cid"),
    [
        ("anca sinistra", "RM pelvi: anca sinistra con regolare segnale.", "hip_left"),
        ("anca dx", "RM pelvi: anca dx con regolare segnale.", "hip_right"),
        ("left hip", "Pain in the left hip after a fall.", "hip_left"),
    ],
)
def test_the_hip_without_bone_is_the_nearest_class_said_as_approx(
    lexicon, table, mention, sentence, cid
):
    """Decided 7 Oct 2026: "anca" / "hip" is the region or the joint (UBERON:0001464 hip is a
    region), the class is the hip bone; the link says approx, never equal."""
    result = _linker(lexicon, table).link(mention, sentence)
    assert (result.status, result.cid, result.relation) == (al.ACCEPTED, cid, "approx")


def test_the_hip_bone_named_with_its_head_noun_is_equal(lexicon, table):
    result = _linker(lexicon, table).link(
        "osso dell'anca destro", "Frattura dell'osso dell'anca destro."
    )
    assert (result.status, result.cid, result.relation) == (
        al.ACCEPTED,
        "hip_right",
        "equal",
    )


@pytest.mark.parametrize(
    ("mention", "sentence"),
    [
        ("left innominate", "The left innominate is compressed by the mass."),
        ("coxale sinistro", "Dolore coxale sinistro."),
    ],
)
def test_a_name_without_its_head_noun_is_not_the_lexicon_name(
    lexicon, table, mention, sentence
):
    """ "left innominate bone" minus "bone" can be the innominate vein or artery: the head noun of
    a name says what it refers to, so the shortened form is never recognised as the name."""
    recognised, how = lexicon.recognise(mention, sentence)
    assert (recognised, how) == ([], "name_without_its_head_noun")
    assert _linker(lexicon, table).link(mention, sentence).cid is None
