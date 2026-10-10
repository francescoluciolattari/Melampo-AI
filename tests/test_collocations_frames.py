"""Collocations (point 3), anatomy names in the lattice memory (point 1), procedure frames (point 2)."""

import gzip
import io

from melampo.memory import chunk_lattice as cl
from melampo.memory.collocations import Collocations, runs
from melampo.memory.frames import DEVICE, SITE, ProcedureFrames

TEXT = (["The inferior vena cava filter placement was done. A filter placement in the inferior vena cava."] * 6
        + ["The liver was normal, and the spleen was large."] * 6)


def test_runs_are_not_broken_by_a_defined_abbreviation_but_by_punctuation():
    got = runs("Inferior vena cava (IVC) filter placement, then the liver.")
    assert got[0] == ["inferior", "vena", "cava", "filter", "placement"]
    assert got[1] == ["then", "the", "liver"]


def test_npmi_is_one_for_words_only_together_and_none_below_the_count():
    c = Collocations().add_texts(TEXT)
    assert c.npmi("vena", "cava") > 0.9
    assert c.npmi("vena", "cava") <= 1.0
    assert c.npmi("was", "large") > c.npmi("the", "liver") - 1  # defined
    assert c.npmi("cava", "liver") is None  # never together
    assert Collocations().add_texts(TEXT[:1]).npmi("vena", "cava") is None  # seen once: no value
    assert c.cohesion(["inferior", "vena", "cava"]) == min(c.npmi("inferior", "vena"), c.npmi("vena", "cava"))
    assert c.forward("vena", "cava") == 1.0


def test_collocations_roundtrip_and_lossy_counting_keeps_frequent_pairs(tmp_path):
    c = Collocations().add_texts(TEXT)
    path = tmp_path / "c.json.gz"
    c.prune().save(path)
    again = Collocations.load(path)
    assert again.npmi("vena", "cava") == c.npmi("vena", "cava")
    lossy = Collocations(epsilon=0.05).add_texts(TEXT * 3)
    assert lossy.count("vena", "cava") >= 30  # frequent pairs survive the buckets


def test_a_familiar_composed_block_costs_less_and_never_below_a_remembered_name():
    memory = cl.BlockMemory.from_json({"heads": {"placement": ["procedure", 0.9], "filter": ["device", 0.9]},
                                       "names": {}})
    colloc = Collocations().add_texts(TEXT)
    plain, familiar = cl.ChunkLattice(memory), cl.ChunkLattice(memory, collocations=colloc)
    assert plain._familiar(["filter", "placement"]) == 0.0
    bonus = familiar._familiar(["filter", "placement"])
    assert 0.0 < bonus <= cl.C_COMPOSED - cl.C_MEMORY
    assert familiar._familiar(["liver", "placement"]) == 0.0  # unknown pair: nothing


def test_anatomy_names_are_remembered_and_lose_to_a_name_of_another_kind():
    memory = cl.BlockMemory.from_json({"heads": {"bulb": ["device", 1.0], "layer": ["anatomy", 1.0]},
                                       "names": {"vena cava filter": "device"}},
                                      anatomy_names=["olfactory bulb", "vena cava filter", "inferior vena cava"])
    assert memory.names["olfactory bulb"] == "anatomy"
    assert memory.names["vena cava filter"] == "device"
    lat = cl.ChunkLattice(memory)
    # the owner is a structure, not a device: the layer is not "part of an object"
    assert lat.read("layer", "the outer layer of the olfactory bulb").outcome == "link"
    plain = cl.ChunkLattice(cl.BlockMemory.from_json({"heads": {"bulb": ["device", 1.0]}, "names": {}}))
    assert plain.read("layer", "the outer layer of the olfactory bulb").outcome == "not_a_site"


def _frames(site=40, device=30, n=60):
    f = ProcedureFrames()
    f.observe("placement", "text", set(), n - site - device)
    f.observe("placement", "text", {SITE}, site)
    f.observe("placement", "text", {DEVICE}, device)
    f.observe("assessment", "text", set(), 400)
    f.observe("assessment", "text", {SITE}, 10)
    return f


def test_a_slot_is_what_distinguishes_a_head_from_procedure_heads_in_general():
    f = _frames()
    assert f.slots("placement") == {SITE, DEVICE}
    assert f.slots("assessment") == set()
    assert f.slots("unknown") == set()
    few = ProcedureFrames()
    few.observe("biopsy", "ncit", {SITE}, 2)
    assert few.slots("biopsy") == set()  # two observations say nothing


def test_frames_learned_from_text_before_the_head_and_through_a_preposition():
    kinds = {"placement": "procedure", "filter": "device", "stent": "device", "cava": "anatomy",
             "duct": "anatomy", "liver": "anatomy"}
    f = ProcedureFrames().add_texts(
        ["Inferior vena cava filter placement.", "Placement of a stent in the bile duct.", "The placement."],
        lambda w: kinds.get(w, ""))
    c = f.evidence["placement"]["text"]
    assert c["n"] == 2 and c[SITE] == 2 and c[DEVICE] == 2


def test_frames_from_ncit_relations_and_snomed_rf2(tmp_path):
    obo = tmp_path / "n.obo"
    obo.write_text("[Term]\nid: NCIT:C1\nname: Liver Biopsy\nrelationship: NCIT:R163 NCIT:C2\n\n"
                   "[Term]\nid: NCIT:C3\nname: Stent Placement\nrelationship: NCIT:R181 NCIT:C4\n", "utf-8")
    f = ProcedureFrames().add_ncit(obo)
    assert f.evidence["biopsy"]["ncit"][SITE] == 1 and f.evidence["placement"]["ncit"][DEVICE] == 1
    rel = tmp_path / "rel.txt"
    rel.write_text("id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\n"
                   "1\t2025\t1\tm\t100\t200\t0\t405813007\n1\t2025\t1\tm\t100\t300\t0\t424226004\n"
                   "1\t2025\t0\tm\t101\t200\t0\t405813007\n", "utf-8")
    desc = tmp_path / "desc.txt"
    desc.write_text("id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n"
                    "1\t2025\t1\tm\t100\ten\tt\tInsertion of filter into inferior vena cava (procedure)\tc\n"
                    "2\t2025\t1\tm\t100\ten\tt\tInferior vena cava filter placement\tc\n"
                    "3\t2025\t1\tm\t101\ten\tt\tInactive thing placement\tc\n", "utf-8")
    f.add_snomed(rel, desc)
    assert f.evidence["placement"]["snomed"][SITE] == 1 and f.evidence["placement"]["snomed"][DEVICE] == 1
    assert f.evidence["cava"]["snomed"]["n"] == 1  # the semantic tag is dropped before the head is taken


def test_the_lattice_reads_the_structure_as_the_site_of_the_framed_procedure():
    memory = cl.BlockMemory.from_json({"heads": {"placement": ["procedure", 0.9], "filter": ["device", 0.9],
                                                 "assessment": ["procedure", 0.9]}, "names": {}})
    sentence = "safety of inferior vena cava filter placement in the office"
    plain = cl.ChunkLattice(memory).read("inferior vena cava", sentence)
    framed = cl.ChunkLattice(memory, frames=_frames()).read("inferior vena cava", sentence)
    assert plain.role == "device_site"  # the cheapest block is [inferior vena cava filter]
    assert (framed.outcome, framed.role, framed.source) == ("role", "procedure_site", "frame")
    assert framed.why == "frame:placement:device=filter"
    # a head without a site slot does not frame the structure
    other = cl.ChunkLattice(memory, frames=_frames()).read("inferior vena cava", "inferior vena cava filter assessment")
    assert other.source != "frame"


def test_pubmed_abstracts_are_read_from_the_xml():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_collocations import abstracts

    xml = (b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article><ArticleTitle>IVC "
           b"<i>filter</i> placement</ArticleTitle><Abstract><AbstractText>We placed filters.</AbstractText>"
           b"</Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>")
    got = list(abstracts(io.BytesIO(gzip.decompress(gzip.compress(xml)))))
    assert got == [("42", "IVC filter placement We placed filters.")]
