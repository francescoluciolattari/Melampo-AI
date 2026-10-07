"""Tests for the anatomy linker: normalisation, recognition, the integration check, deliberation, abstention."""

import json
from pathlib import Path

import pytest

from melampo.evaluation import linking_bench as lb
from melampo.memory import anatomy_linker as al

_DATA = Path(__file__).resolve().parents[1] / "data" / "linking"


@pytest.fixture(scope="module")
def lexicon():
    return al.Lexicon.from_json(
        json.loads((_DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )


@pytest.fixture(scope="module")
def heldout():
    return {
        lang: [
            json.loads(line)
            for line in (_DATA / f"heldout_{lang}.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        for lang in ("it", "en")
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("rene di dx", ("<dx>", "rene")),
        ("rt kidney", ("<dx>", "kidney")),
        ("L kidney", ("<sn>", "kidney")),
        ("VI costa destra", ("#6", "<dx>", "costa")),
        ("undicesima costa sinistra", ("#11", "<sn>", "costa")),
        ("3a costa sx", ("#3", "<sn>", "costa")),
        ("left 7th rib", ("#7", "<sn>", "rib")),
        ("D7", ("@D7",)),
        ("terza vertebra lombare", ("@L3",)),
        ("fourth lumbar vertebra", ("@L4",)),
        ("T4 vertebral body", ("@T4",)),
        ("terzo medio del femore destro", ("<dx>", "<mid>", "femore")),
        ("right kidney's upper pole", ("<dx>", "<sup>", "kidney", "pole")),
        ("L 5 rib", ("#5", "<sn>", "rib")),
        ("arteria ipogastrica sinistra", ("<art>", "<int>", "<sn>")),
        ("C5-C6", ("@C5", "@C6")),
        ("a. iliaca comune dx", ("<art>", "<com>", "<dx>", "iliaca")),
        ("fegato e milza", ("<and>", "fegato", "milza")),
        ("parenchima splenico", ("milza",)),
        ("hepatic segment IV", ("#4", "liver", "segment")),
        ("coste di destra", ("<dx>", "costa")),
    ],
)
def test_normalise(text, expected):
    assert al.normalise(text) == expected


def test_lowercase_i_is_an_article_not_a_roman_one():
    assert "#1" not in al.normalise("i polmoni")


@pytest.mark.parametrize(
    ("mention", "sentence", "expected"),
    [
        ("LID", "TC torace: nodulo nel LID.", ["lung_lower_lobe_right"]),
        ("RUL", "Nodule in the RUL.", ["lung_upper_lobe_right"]),
        (
            "XII costa di destra",
            "Frattura della XII costa di destra.",
            ["rib_right_12"],
        ),
        ("D12", "Rachide: crollo di D12.", ["vertebrae_T12"]),
        ("Th12", "Crollo di Th12.", ["vertebrae_T12"]),
        ("L 1", "Frattura di L 1.", ["vertebrae_L1"]),
        ("S1", "Fegato: lesione in S1.", ["liver_segment_1"]),
        ("S1", "Rachide: lisi del soma di S1.", ["vertebrae_S1"]),
        ("epistrofeo", "Frattura dell'epistrofeo.", ["vertebrae_C2"]),
    ],
)
def test_recognition(lexicon, mention, sentence, expected):
    assert lexicon.recognise(mention, sentence) == (expected, "recognised")


def test_a_context_dependent_name_without_its_context_is_not_recognised(lexicon):
    """S6 is a liver segment only when the sentence is about the liver; in a lung report it is not."""
    assert lexicon.recognise("S6", "Polmone: addensamento in S6.") == (
        [],
        "context_does_not_support_the_name",
    )


def test_names_shared_by_several_classes_all_need_context(lexicon):
    owners: dict[tuple, set] = {}
    for cid, entry in lexicon.classes.items():
        for name in entry["it"] + entry["en"]:
            owners.setdefault(al.normalise(name), set()).add(
                (cid, name in entry.get("requires_context", {}))
            )
    for key, members in owners.items():
        if len({cid for cid, _ in members}) > 1:
            assert all(needs for _, needs in members), (key, members)


def test_every_name_is_recognised_as_its_own_class(lexicon):
    for cid, entry in lexicon.classes.items():
        for name in entry["it"] + entry["en"]:
            region = entry.get("requires_context", {}).get(name)
            cue = (
                sorted(al.REGION_CUES[region.replace("_no_signal", "")])[0]
                if region
                else ""
            )
            found, _ = lexicon.recognise(name, f"{cue} {name}")
            assert cid in found, (cid, name, found)


def _cand(cid, *names, target=True):
    return al.Candidate(
        cid=cid,
        label=names[0],
        names=tuple(al.normalise(n) for n in names),
        is_target_class=target,
    )


@pytest.mark.parametrize(
    ("mention", "candidate", "reason"),
    [
        ("rene destro", _cand("kidney_left", "left kidney"), "side_mismatch"),
        (
            "rene destro",
            _cand("U:kidney", "kidney", target=False),
            "candidate_lacks_the_side_the_mention_states",
        ),
        (
            "succlavia",
            _cand("subclavian_artery_left", "succlavia sinistra"),
            "side_not_stated_in_mention",
        ),
        (
            "costa",
            _cand("rib_left_10", "costa 10 sinistra"),
            "side_not_stated_in_mention",
        ),
        (
            "X costa sinistra",
            _cand("rib_left_7", "costa 7 sinistra"),
            "number_or_level_mismatch",
        ),
        (
            "VI costa destra",
            _cand("U:rib", "rib", target=False),
            "candidate_lacks_the_number_the_mention_states",
        ),
        ("C5-C6", _cand("vertebrae_C5", "C5"), "mention_spans_several_levels"),
        (
            "fegato e milza",
            _cand("liver", "fegato"),
            "mention_names_more_than_one_structure",
        ),
        (
            "reni bilaterali",
            _cand("kidney_left", "left kidney"),
            "mention_names_more_than_one_structure",
        ),
        (
            "lobo polmonare destro",
            _cand("lung_upper_lobe_right", "lobo superiore destro"),
            "position_not_stated_in_mention",
        ),
        (
            "lobo superiore dx",
            _cand("lung_lower_lobe_right", "lobo inferiore destro"),
            "position_mismatch",
        ),
    ],
)
def test_verify_rejects(mention, candidate, reason):
    lateral = al.lateralised_bases(
        [candidate, _cand("U:left kidney", "left kidney", target=False)]
    )
    assert al.verify(al.normalise(mention), candidate, lateral) == reason


@pytest.mark.parametrize(
    ("mention", "candidate"),
    [
        (
            "segmento laterale del lobo medio",
            _cand("lung_middle_lobe_right", "lobo medio", "lobo medio destro"),
        ),
        ("lobo destro del fegato", _cand("liver", "fegato")),
        ("VI costa destra", _cand("rib_right_6", "costa 6 destra")),
    ],
)
def test_verify_accepts_what_is_compatible(mention, candidate):
    assert al.verify(al.normalise(mention), candidate, frozenset()) is None


_OBO = """format-version: 1.2

[Term]
id: UBERON:0004538
name: left kidney
synonym: "levorenal organ" EXACT []
xref: FMA:7205

[Term]
id: UBERON:0004601
name: left hemikidney
xref: FMA:9001

[Term]
id: UBERON:0004602
name: right hemikidney
xref: FMA:9002

[Term]
id: UBERON:0009001
name: paired organ one
synonym: "twin viscus" EXACT []
xref: FMA:9003

[Term]
id: UBERON:0009002
name: paired organ two
synonym: "twin viscus" EXACT []
xref: FMA:9004

[Term]
id: UBERON:0002113
name: kidney
synonym: "renal organ" EXACT []
xref: FMA:7203

[Term]
id: UBERON:0000001
name: insect thing

[Term]
id: UBERON:0000002
name: old thing
xref: FMA:1
is_obsolete: true

[Term]
id: UBERON:0002376
name: cranial muscle
xref: FMA:71287

[Term]
id: UBERON:0005396
name: left common carotid artery plus branches
xref: FMA:4058

[Typedef]
id: part_of
"""


def test_load_obo_terms_keeps_human_non_obsolete_terms():
    terms = al.load_obo_terms(_OBO.splitlines())
    assert [t["id"] for t in terms][:1] == ["UBERON:0004538"]
    assert "UBERON:0002113" in [t["id"] for t in terms]
    by_id = {t["id"]: t for t in terms}
    assert by_id["UBERON:0002113"]["synonyms"] == ["renal organ"]


def test_build_pool_maps_an_ontology_term_named_like_a_class(lexicon):
    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(_OBO.splitlines()))
    assert equivalent == {"UBERON:0004538": "kidney_left"}
    assert len(pool) == len(lexicon.classes) + 8


def _linker(lexicon, ranked, votes):
    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(_OBO.splitlines()))
    chats = {
        name: (lambda answer: lambda prompt: answer)(answer)
        for name, answer in votes.items()
    }
    return al.AnatomyLinker(
        lexicon, pool, equivalent, retriever=lambda m, s, k: ranked, chats=chats
    )


def test_the_lexicon_answers_before_any_model(lexicon):
    linker = _linker(lexicon, [], {"a": "boom", "b": "boom"})
    result = linker.link("LID", "Nodulo nel LID.")
    assert (result.status, result.cid, result.stage) == (
        al.ACCEPTED,
        "lung_lower_lobe_right",
        "lexicon",
    )


def test_two_models_must_agree(lexicon):
    ranked = [
        "UBERON:0009001",
        "UBERON:0009002",
    ]  # both carry the synonym "twin viscus"
    mention, sentence = "twin viscus", "Twin viscus nei limiti."
    agree = _linker(lexicon, ranked, {"a": "2", "b": "2"}).link(mention, sentence)
    disagree = _linker(lexicon, ranked, {"a": "1", "b": "2"}).link(mention, sentence)
    unsure = _linker(lexicon, ranked, {"a": "0", "b": "1"}).link(mention, sentence)
    garbled = _linker(lexicon, ranked, {"a": "boh", "b": "1"}).link(mention, sentence)
    assert (agree.status, agree.cid, agree.stage) == (
        al.ACCEPTED,
        "UBERON:0009002",
        "deliberation",
    )
    assert (disagree.status, disagree.reason) == (al.ABSTAINED, "models_disagree")
    assert (unsure.status, unsure.reason) == (al.ABSTAINED, "a_model_was_not_sure")
    assert (garbled.status, garbled.reason) == (al.ABSTAINED, "a_model_was_not_sure")


def test_options_that_contradict_the_mention_are_never_offered(lexicon):
    seen = []

    def chat(prompt):
        seen.append(prompt)
        return "1"

    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(_OBO.splitlines()))
    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, k: ["UBERON:0004601", "UBERON:0004602"],
        chats={"a": chat, "b": chat},
    )
    result = linker.link("hemikidney destro", "Cisti dell'hemikidney destro.")
    assert "left hemikidney" not in seen[0]
    assert result.cid == "UBERON:0004602"


def test_an_agreed_ontology_term_named_like_a_class_resolves_to_the_class(lexicon):
    result = _linker(lexicon, ["UBERON:0004538"], {"a": "1", "b": "1"}).link(
        "levorenal organ sinistro", "Cisti al levorenal organ sinistro."
    )
    assert (result.status, result.cid) == (al.ACCEPTED, "kidney_left")


def test_no_surviving_candidate_is_an_abstention(lexicon):
    result = _linker(lexicon, ["kidney_left"], {"a": "1", "b": "1"}).link(
        "emirene destro", "x"
    )
    assert (result.status, result.reason) == (
        al.ABSTAINED,
        "no_candidate_survives_the_checks",
    )


def test_heldout_mentions_sit_at_their_offsets(heldout):
    for rows in heldout.values():
        for row in rows:
            assert (
                row["sentence"][row["start"] : row["start"] + len(row["mention"])]
                == row["mention"]
            )


def test_deterministic_stages_make_no_silent_error_on_the_heldout_sets(
    lexicon, heldout
):
    """Regression guard: a lexicon or normaliser change that links anything wrongly fails here."""
    pool, equivalent = al.build_pool(lexicon)
    linker = al.AnatomyLinker(lexicon, pool, equivalent)
    for rows in heldout.values():
        report = lb.evaluate_anatomy_linker(linker, rows, workers=1)
        assert report["outcomes"].get("wrong", 0) == 0, [
            d for d in report["details"] if d["outcome"] == "wrong"
        ]
        assert report["outcomes"]["abstained_as_expected"] == sum(
            1 for r in rows if r["target"] is None
        )


@pytest.mark.parametrize(
    ("errors", "n", "bound"),
    [(0, 150, 0.0198), (0, 0, 1.0), (1, 10, 0.3942), (10, 10, 1.0)],
)
def test_upper_error_bound(errors, n, bound):
    assert lb.upper_error_bound(errors, n) == pytest.approx(bound, abs=5e-4)


def test_linker_report_counts_four_outcomes(lexicon):
    rows = [
        {"mention": "LID", "sentence": "nel LID", "target": "lung_lower_lobe_right"},
        {"mention": "LID", "sentence": "nel LID", "target": "lung_lower_lobe_left"},
        {"mention": "appendice", "sentence": "appendice", "target": None},
        {"mention": "parola ignota", "sentence": "x", "target": "spleen"},
    ]
    pool, equivalent = al.build_pool(lexicon)
    report = lb.evaluate_anatomy_linker(
        al.AnatomyLinker(lexicon, pool, equivalent), rows, workers=1
    )
    assert report["outcomes"] == {
        "correct": 1,
        "wrong": 1,
        "abstained_as_expected": 1,
        "abstained": 1,
    }
    assert report["precision_of_accepted"] == 0.5
    assert "links to check" in lb.render_linker_markdown({"x": report})


def test_retriever_merges_mention_and_description_rankings():
    pool = [_cand("a", "alpha"), _cand("b", "beta"), _cand("c", "gamma")]

    def embedder(texts):
        table = {
            "alpha": [1, 0, 0],
            "beta": [0, 1, 0],
            "gamma": [0, 0, 1],
            "m": [1, 0, 0],
            "desc": [0, 0, 1],
        }
        return [table[t] for t in texts]

    retriever = lb.EmbeddingRetriever(embedder, pool, describe=lambda prompt: "desc")
    assert retriever("m", "s", 2) == ["a", "c"]


def _adversarial(lexicon, pick):
    """Retrieval offers the whole pool; both models choose `pick` whenever it is offered."""
    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(_OBO.splitlines()))
    by_id = {c.cid: c for c in pool}

    def chat(prompt):
        for line in prompt.splitlines():
            number, _, label = line.partition(". ")
            if number.isdigit() and int(number) > 0 and label == by_id[pick].label:
                return number
        return "0"

    # The wrong pick is ranked first, as if retrieval itself had been fooled.
    ranked = [pick] + [c.cid for c in pool if c.cid != pick]
    return al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, k: ranked,
        chats={"a": chat, "b": chat},
    )


# Each case: the mention, its sentence, the wrong class two colluding models
# would pick, and the right class (None when abstaining is the only right answer).
_ATTACKS = [
    ("L'omero", "L'omero presenta una frattura.", "humerus_left", None),
    (
        "L'arteria succlavia",
        "L'arteria succlavia è pervia.",
        "subclavian_artery_left",
        None,
    ),
    (
        "L'ARTERIA SUCCLAVIA",
        "L'ARTERIA SUCCLAVIA È PERVIA.",
        "subclavian_artery_left",
        None,
    ),
    ("L'auricola", "L'auricola è libera.", "atrial_appendage_left", None),
    ("L'osso coxale", "L'osso coxale è integro.", "hip_left", None),
    ("L'ileopsoas", "L'ileopsoas è ispessito.", "iliopsoas_left", None),
    ("L'iliaca comune", "L'iliaca comune è ectasica.", "iliac_artery_left", None),
    (
        "S1",
        "Lesione sclerotica in S1 in paziente con metastasi epatiche.",
        "liver_segment_1",
        None,
    ),
    ("S3", "Frattura composta di S3; fegato nei limiti.", "liver_segment_3", None),
    (
        "S6",
        "Nodulo polmonare in S6 a ridosso della cupola epatica.",
        "liver_segment_6",
        None,
    ),
    ("LM", "Stenosi critica del LM e della IVA.", "lung_middle_lobe_right", None),
    (
        "T2",
        "Lesione iperintensa in T2 nel lobo destro del fegato.",
        "vertebrae_T2",
        None,
    ),
    ("D2", "Diverticolo periampollare in D2.", "vertebrae_T2", None),
    (
        "muscoli masticatori",
        "Ipertrofia dei muscoli masticatori.",
        "UBERON:0002376",
        None,
    ),
    (
        "carotide interna sinistra",
        "Stenosi della carotide interna sinistra.",
        "UBERON:0005396",
        None,
    ),
    ("VI costa", "Frattura della VI costa.", "liver_segment_6", None),
    ("D 12", "Crollo di D 12.", "rib_right_12", "vertebrae_T12"),
    ("D3", "Stenosi in D3 da compressione aorto-mesenterica.", "vertebrae_T3", None),
    ("T 12", "Crollo di T 12.", "rib_right_12", "vertebrae_T12"),
    ("Th12", "Crollo di Th12.", "rib_left_12", "vertebrae_T12"),
    ("L 1", "Frattura di L 1.", "rib_left_1", "vertebrae_L1"),
    ("L 5", "Listesi di L 5.", "rib_left_5", "vertebrae_L5"),
    ("C 5", "Crollo di C 5.", "rib_left_5", "vertebrae_C5"),
    (
        "VII costa sinistra nel tratto dorsale",
        "Frattura della VII costa sinistra nel tratto dorsale.",
        "vertebrae_T7",
        None,
    ),
    ("radice S1", "Rachide: compressione della radice S1.", "liver_segment_1", None),
    ("segmento S1", "Rachide: alterazione del segmento S1.", "liver_segment_1", None),
    (
        "arco posteriore della VI costa",
        "Frattura dell'arco posteriore della VI costa.",
        "rib_right_6",
        None,
    ),
    (
        "succlavia prossimale",
        "Stenosi della succlavia prossimale.",
        "subclavian_artery_left",
        None,
    ),
    (
        "polo superiore del rene",
        "Cisti al polo superiore del rene.",
        "kidney_left",
        None,
    ),
    (
        "margine costale destro",
        "Dolorabilità al margine costale destro.",
        "rib_right_10",
        None,
    ),
    (
        "carotide interna destra",
        "Placca della carotide interna destra.",
        "common_carotid_artery_right",
        None,
    ),
    (
        "arteria iliaca esterna sinistra",
        "Stenosi dell'arteria iliaca esterna sinistra.",
        "iliac_artery_left",
        None,
    ),
    ("vena cava", "Vena cava pervia.", "superior_vena_cava", None),
    ("I RENI", "I RENI SONO IN SEDE.", "kidney_left", None),
    # second review round
    ("T2", "Alterazione di segnale del midollo in T2.", "vertebrae_T2", None),
    ("T1", "Rachide: lesione ipointensa in T1.", "vertebrae_T1", None),
    ("T2", "Colonna lombare: ernia iperintensa in T2.", "vertebrae_T2", None),
    ("S1", "Metastasi ossee in S1 e nel fegato.", "liver_segment_1", None),
    ("S1", "Lesione litica di S1 con secondarismi epatici.", "liver_segment_1", None),
    ("LM", "Torace: calcificazioni del LM.", "lung_middle_lobe_right", None),
    # E3C Italian cases: GB is the white cell count, not the gallbladder
    (
        "GB",
        "Hb 12,3 g/dL; GB 5040/mmc (N 48%; L 42%); Plt 247000/mmc.",
        "gallbladder",
        None,
    ),
    ("GB", "Nei limiti la crasi ematica (GB: 11250/mmc; N 20%).", "gallbladder", None),
    ("T4", "Neoplasia del retto in stadio T4.", "vertebrae_T4", None),
    (
        "terzo distale della clavicola sinistra",
        "Frattura del terzo distale della clavicola sinistra.",
        "rib_left_3",
        "clavicula_left",
    ),
    (
        "terzo medio del femore destro",
        "Frattura del terzo medio del femore destro.",
        "rib_right_3",
        "femur_right",
    ),
    (
        "terzo medio del rene destro",
        "Cisti al terzo medio del rene destro.",
        "rib_right_3",
        "kidney_right",
    ),
    (
        "distal third of the left clavicle",
        "Fracture of the distal third of the left clavicle.",
        "rib_left_3",
        "clavicula_left",
    ),
    (
        "quinto metatarso sinistro",
        "Frattura del quinto metatarso sinistro.",
        "rib_left_5",
        None,
    ),
    (
        "quinto dito della mano sinistra",
        "Frattura del quinto dito della mano sinistra.",
        "rib_left_5",
        None,
    ),
    ("XII dx", "Paralisi del XII dx.", "rib_right_12", None),
    (
        "right kidney's upper pole",
        "Cyst at the right kidney's upper pole.",
        "lung_upper_lobe_right",
        "kidney_right",
    ),
    (
        "arteria ipogastrica sinistra",
        "Aneurisma dell'arteria ipogastrica sinistra.",
        "iliac_artery_left",
        None,
    ),
    (
        "carotide destra",
        "Placca della carotide destra.",
        "common_carotid_artery_right",
        None,
    ),
    ("L 5 rib", "Fracture of the L 5 rib.", "vertebrae_L5", "rib_left_5"),
    ("L 1 rib", "Fracture of the L 1 rib.", "vertebrae_L1", "rib_left_1"),
    ("colica renale", "Colica renale destra.", "colon", None),
    # third review round
    ("T2", "Rachide cervicale: mielopatia in T2.", "vertebrae_T2", None),
    ("T2", "Colonna dorsale: edema in T2 senza crollo.", "vertebrae_T2", None),
    ("T1", "Rachide: enhancement dopo gadolinio in T1.", "vertebrae_T1", None),
    ("D2", "Diverticolo della parete dorsale di D2.", "vertebrae_T2", None),
    (
        "sede renale destra",
        "Esiti di nefrectomia destra; in sede renale destra non recidive.",
        "kidney_right",
        None,
    ),
    (
        "regione surrenalica sinistra",
        "Regione surrenalica sinistra libera dopo surrenectomia.",
        "adrenal_gland_left",
        None,
    ),
    ("sede splenica", "Esiti di splenectomia; sede splenica libera.", "spleen", None),
    ("T3", "Carcinoma del retto cT3N1.", "vertebrae_T3", None),
    (
        "vena succlavia sinistra",
        "Trombosi della vena succlavia sinistra.",
        "subclavian_artery_left",
        None,
    ),
    (
        "arteria splenica",
        "Aneurisma dell'arteria splenica.",
        "portal_vein_and_splenic_vein",
        None,
    ),
    ("arteria polmonare", "Embolia nell'arteria polmonare.", "pulmonary_vein", None),
    (
        "arteria femorale sinistra",
        "Stenosi dell'arteria femorale sinistra.",
        "femur_left",
        None,
    ),
    (
        "vena femorale destra",
        "Trombosi della vena femorale destra.",
        "femur_right",
        None,
    ),
    (
        "vena renale sinistra",
        "Trombosi della vena renale sinistra.",
        "kidney_left",
        None,
    ),
    ("arteria epatica", "Arteria epatica pervia.", "liver", None),
    (
        "loggia renale destra",
        "Esiti di nefrectomia: loggia renale destra libera.",
        "kidney_right",
        None,
    ),
    ("ipocondrio destro", "Dolorabilità all'ipocondrio destro.", "liver", None),
    ("ilo epatico", "Linfoadenopatie all'ilo epatico.", "liver", None),
    ("L5", "Radicolopatia L5 sinistra.", "vertebrae_L5", None),
]

# Semantic errors that no attribute check can see: the mention is not a
# structure at all. Only the models' own "0" and the review list guard these.
_KNOWN_SEMANTIC_LIMITS = [
    # A symptom phrase whose organ word is a class: every attribute agrees with the
    # organ, so if retrieval and both models are fooled together, only review catches it.
    ("dolore epatico", "Dolore epatico da distensione capsulare.", "liver", None),
    # A part named without structural words, with retrieval and both models fooled
    # together: no attribute can show that the spleen is not the lingula. Needs the
    # part-of knowledge (curated parts table or ontology part_of) planned next.
    ("lingula", "Atelettasia della lingula.", "spleen", None),
]


@pytest.mark.parametrize(("mention", "sentence", "pick", "right"), _ATTACKS)
def test_colluding_models_cannot_force_a_wrong_class(
    lexicon, mention, sentence, pick, right
):
    result = _adversarial(lexicon, pick).link(mention, sentence)
    if result.status == al.ACCEPTED and result.cid in lexicon.classes:
        assert result.cid == right, (result.cid, result.stage, result.reason)


@pytest.mark.parametrize(
    ("mention", "sentence", "pick", "right"), _KNOWN_SEMANTIC_LIMITS
)
def test_words_the_candidate_does_not_cover_are_never_linked(
    lexicon, mention, sentence, pick, right
):
    result = _adversarial(lexicon, pick).link(mention, sentence)
    assert result.status != al.ACCEPTED


# Silent errors found by the live run with the two real models (2026-10-05): both models agreed,
# the checks passed, and the candidate was a different concept.
_OBO_LIVE = """[Term]
id: UBERON:0004151
name: cardiac chamber
xref: FMA:1

[Term]
id: UBERON:0035495
name: hilum of lymph node
xref: FMA:2

[Term]
id: UBERON:0002237
name: true rib
xref: FMA:3

[Term]
id: UBERON:0001154
name: vermiform appendix
synonym: "appendix" EXACT []
xref: FMA:4
"""


@pytest.mark.parametrize(
    ("mention", "sentence", "pick"),
    [
        (
            "Cardiac silhouette",
            "Cardiac silhouette within normal limits.",
            "UBERON:0004151",
        ),
        (
            "Right hilar lymph node",
            "Right hilar lymph node of 12 mm.",
            "UBERON:0035495",
        ),
        ("right ribs", "Fractures of the right ribs.", "UBERON:0002237"),
        (
            "porta hepatis",
            "Lymphadenopathy at the porta hepatis.",
            "portal_vein_and_splenic_vein",
        ),
    ],
)
def test_live_run_silent_errors_are_abstentions(lexicon, mention, sentence, pick):
    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(_OBO_LIVE.splitlines()))
    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, k: [pick],
        chats={"a": lambda p: "1", "b": lambda p: "1"},
    )
    assert linker.link(mention, sentence).status == al.ABSTAINED


def test_a_name_the_candidate_has_exactly_is_still_linked(lexicon):
    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(_OBO_LIVE.splitlines()))
    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, k: ["UBERON:0001154"],
        chats={"a": lambda p: "1", "b": lambda p: "1"},
    )
    result = linker.link("Appendix", "Appendix of normal caliber.")
    assert (result.status, result.cid) == (al.ACCEPTED, "UBERON:0001154")


@pytest.mark.parametrize(
    ("mention", "name"),
    [
        ("Appendix", "wall of appendix"),
        ("Subclavian artery", "wall of subclavian artery"),
    ],
)
def test_the_wall_of_a_structure_is_not_the_structure(lexicon, mention, name):
    obo = f"[Term]\nid: UBERON:0090001\nname: {name}\nxref: FMA:5\n"
    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(obo.splitlines()))
    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, k: ["UBERON:0090001"],
        chats={"a": lambda p: "1", "b": lambda p: "1"},
    )
    assert linker.link(mention, f"{mention} regular.").status == al.ABSTAINED


def test_a_homonym_from_the_nervous_system_is_not_offered_in_a_lung_sentence(lexicon):
    obo = (
        "[Term]\nid: UBERON:0004074\nname: cerebellum vermis lobule I\n"
        'synonym: "lingula" EXACT []\nxref: FMA:6\n'
    )
    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(obo.splitlines()))
    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, k: ["UBERON:0004074"],
        chats={"a": lambda p: "1", "b": lambda p: "1"},
    )
    assert linker.link("lingula", "Atelettasia della lingula.").status == al.ABSTAINED
    assert (
        linker.link("lingula", "Lingula del cervelletto, encefalo normale.").status
        == al.ACCEPTED
    )


@pytest.mark.parametrize(
    ("sentence", "linked"),
    [
        ("GB wall thickening with gallstones.", True),
        ("The GB is distended with sludge; the liver is normal.", True),
        ("Colecistectomia pregressa, GB non visualizzata, fegato regolare.", True),
        ("Hb 12,3 g/dL; GB 5040/mmc (N 48%; L 42%).", False),
        ("Nei limiti la crasi ematica (GB: 11250/mmc).", False),
        ("GB 11.2 with a fatty liver on ultrasound.", False),
        ("GB wbc 11,2 and hepatic steatosis.", False),
        ("GB: normal.", False),
    ],
)
def test_gb_is_the_gallbladder_only_with_hepatobiliary_evidence(
    lexicon, sentence, linked
):
    pool, equivalent = al.build_pool(lexicon)
    result = al.AnatomyLinker(lexicon, pool, equivalent).link("GB", sentence)
    if linked:
        assert result.cid == "gallbladder"
    else:
        assert result.cid is None and result.status == al.ABSTAINED


def test_a_model_that_does_not_answer_is_an_abstention_not_a_crash(lexicon):
    ranked = ["UBERON:0009001", "UBERON:0009002"]

    def down(prompt):
        raise RuntimeError("HTTP 429 from model-a")

    linker = _linker(lexicon, ranked, {"a": "2", "b": "2"})
    linker.chats["a"] = down
    result = linker.link("twin viscus", "Twin viscus nei limiti.")
    assert (result.status, result.reason, result.stage) == (
        al.ABSTAINED,
        "model_unavailable",
        "models",
    )


def test_a_retriever_that_does_not_answer_is_an_abstention_too(lexicon):
    pool, equivalent = al.build_pool(lexicon, al.load_obo_terms(_OBO.splitlines()))

    def broken(mention, sentence, k):
        raise RuntimeError("encoder 429")

    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=broken,
        chats={"a": lambda p: "1", "b": lambda p: "1"},
    )
    result = linker.link("twin viscus", "Twin viscus nei limiti.")
    assert (result.status, result.reason) == (al.ABSTAINED, "model_unavailable")


def test_a_structure_that_only_modifies_a_measurement_is_not_linked(lexicon):
    linker = al.AnatomyLinker(lexicon, [], {})
    for mention, sentence in [
        ("heart", "Her heart rate was high."),
        ("liver", "Liver function was abnormal."),
        ("thyroid", "Thyroid-stimulating hormone was normal."),
    ]:
        result = linker.link(mention, sentence)
        assert result.status == al.ABSTAINED
        assert result.reason.startswith("attribute_head_names_a_measurement:")
    # the same names, as structures, are still linked
    assert (
        linker.link("liver", "The liver biopsy showed steatosis.").status == al.ACCEPTED
    )
    assert linker.link("heart", "The heart is enlarged.").cid == "heart"
