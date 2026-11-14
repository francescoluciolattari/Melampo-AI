"""The disease profiles' pilot (15 November 2026): structured sources first, the words before the judges, an
invented quote refused, the judges only on what the words leave open."""

import importlib.util
import json
from pathlib import Path

from melampo.clinical import case_graph as cg
from melampo.memory import disease_profiles as dp

ROOT = Path(__file__).resolve().parents[1]

HP = """format-version: 1.2

[Term]
id: HP:0002027
name: Abdominal pain
synonym: "Stomach pain" EXACT []

[Term]
id: HP:0001945
name: Fever
synonym: "Pyrexia" EXACT []

[Term]
id: HP:0002013
name: Vomiting

[Term]
id: HP:0031094
name: Elevated circulating lipase concentration

[Term]
id: HP:0000001
name: Obsolete thing
is_obsolete: true
"""

JATS = """<article><body>
<sec><title>Introduction</title><p>Pancreatitis is an inflammation of the pancreas.</p></sec>
<sec><title>Clinical presentation</title>
<p>Epigastric abdominal pain radiating to the back is the hallmark of acute pancreatitis. Vomiting is common.</p>
<p>Fever is rare at onset.</p></sec>
<sec><title>Diagnosis and imaging</title>
<p>A serum lipase above three times the upper limit supports the diagnosis.
Contrast CT shows peripancreatic fat stranding.</p></sec>
<sec><title>Treatment</title><p>Fluids.</p></sec>
</body></article>"""


def _hpo():
    return dp.Terms.from_obo(HP.splitlines())


def test_the_ontology_finds_a_term_by_its_name_or_exact_synonym_never_an_obsolete_one():
    hpo = _hpo()
    assert hpo.find("pyrexia") == "HP:0001945" and hpo.find("Stomach pain") == "HP:0002027"
    assert hpo.find("Obsolete thing") == ""


def test_hpo_annotations_bring_their_frequency_and_their_not():
    hpoa = ["#description", ("database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\t"
                             "frequency\tsex\tmodifier\taspect\tbiocuration"),
            "OMIM:1\tX\t\tHP:0002027\tPMID:1\tPCS\t\tHP:0040281\t\t\tP\tx",
            "OMIM:1\tX\t\tHP:0002013\tPMID:1\tPCS\t\t7/10\t\t\tP\tx",
            "OMIM:1\tX\tNOT\tHP:0001945\tPMID:1\tPCS\t\t\t\t\tP\tx",
            "OMIM:1\tX\t\tHP:0002027\tPMID:2\tTAS\t\t40%\t\t\tP\tx",
            "OMIM:2\tY\t\tHP:0002013\tPMID:1\tPCS\t\t\t\t\tP\tx",
            "OMIM:1\tX\t\tHP:0000006\tPMID:1\tPCS\t\t\t\t\tI\tx"]
    got = {m.manifestation_id: m for m in dp.from_hpoa(hpoa, {"OMIM:1"}, _hpo())}
    assert set(got) == {"HP:0002027", "HP:0002013", "HP:0001945"}
    assert got["HP:0002027"].frequency == 0.895 and len(got["HP:0002027"].citations) == 2
    assert got["HP:0002013"].frequency == 0.7
    assert got["HP:0001945"].typicality == "excluding" and got["HP:0001945"].frequency == 0.0


def test_only_the_sections_that_describe_the_disease_are_read():
    sections = dp.sections_from_jats(JATS)
    assert [t for t, _ in sections] == ["Clinical presentation", "Diagnosis and imaging"]
    assert dp.sections_from_jats(JATS, limit=60)[0][1] == sections[0][1][:60]
    assert dp.sections_from_jats("<broken") == []
    page = "<h2>Etiology</h2>x<h2>History and Physical</h2><p>Pain <b>migrates</b>.</p><h2>Evaluation</h2>CT."
    assert dp.html_sections(page, ("History and Physical", "Evaluation")) == [
        ("History and Physical", "Pain migrates ."), ("Evaluation", "CT.")]


def _item(name, quote, typicality="", kind="symptom", modality="", region=""):
    return {"manifestation": name, "quote": quote, "typicality": typicality, "kind": kind, "modality": modality,
            "region": region}


def test_an_invented_quote_is_refused_a_named_one_is_verified_and_the_rest_waits_for_the_judges():
    hpo, sections = _hpo(), dp.sections_from_jats(JATS)
    a = [_item("Abdominal pain", "Epigastric abdominal pain radiating to the back is the hallmark of acute "
                                 "pancreatitis."),
         _item("Pyrexia", "Fever is rare at onset.", "rare"),
         _item("Jaundice", "Jaundice is frequent."),
         _item("Elevated lipase", "A serum lipase above three times the upper limit supports the diagnosis.",
               kind="laboratory"),
         _item("Peripancreatic fat stranding", "Contrast CT shows peripancreatic fat stranding.", kind="imaging",
               modality="CT", region="abdomen")]
    b = [_item("Abdominal pain", "Epigastric abdominal pain radiating to the back is the hallmark of acute "
                                 "pancreatitis.", "hallmark"),
         _item("Vomiting", "Vomiting is common.", "common")]
    links = {l.manifestation: l for l in dp.merge_readings({"r1": a, "r2": b}, sections, {"id": "PMC1"}, hpo)}
    assert links["Abdominal pain"].verdict == "names" and links["Abdominal pain"].readers == ["r1", "r2"]
    assert links["Pyrexia"].verdict == "names"                 # the HPO synonym's term is named in the quote
    assert links["Jaundice"].verdict == "refused_quote"
    assert links["Elevated lipase"].verdict == ""     # "elevated" is not in the quote: a paraphrase, for the judges
    assert links["Peripancreatic fat stranding"].verdict == "names"
    m = dp.to_manifestation(links["Abdominal pain"], hpo, 0)
    assert m.concept.code == "HP:0002027" and m.typicality == "hallmark"
    assert m.citations[0]["quote"].startswith("Epigastric") and m.citations[0]["id"] == "PMC1"
    img = dp.to_manifestation(links["Peripancreatic fat stranding"], hpo, 1)
    assert img.verified_by == [{"kind": "imaging", "modality": "CT", "region": "abdomen"}]
    assert dp.to_manifestation(links["Pyrexia"], hpo, 2).typicality == "rare"


def test_the_judges_must_agree_and_a_split_is_a_doubt():
    link = dp.Link("Hyperlipasemia", "A serum lipase above three times the upper limit supports the diagnosis.", {})
    assert dp.judged(link, {"a": "yes", "b": "yes"}) == "judges"
    assert dp.judged(link, {"a": "no", "b": "no"}) == "judges_no"
    assert dp.judged(link, {"a": "yes", "b": "unclear"}) == "doubt"
    assert dp.judged(link, {"a": "yes", "b": ""}) == "doubt"
    assert dp.read_judgment('Sure. {"states": "Yes"}') == "yes" and dp.read_judgment("maybe") == ""
    assert dp.read_extraction("no list here") == [] and dp.read_extraction('[{"manifestation": "x"}]') == []


def test_the_profile_merges_sources_by_concept_and_keeps_every_citation():
    hpo = _hpo()
    structured = cg.Manifestation("HP:0002027", "Abdominal pain", concept=cg.Concept("HP", "HP:0002027"),
                                  frequency=0.9, citations=[{"source": "HPO annotations"}])
    text = cg.Manifestation("HP:0002027", "Abdominal pain", concept=cg.Concept("HP", "HP:0002027"),
                            typicality="hallmark", citations=[{"source": "Europe PMC review"}])
    other = cg.Manifestation("text:1", "fat stranding", citations=[{"source": "Europe PMC review"}])
    profile = dp.merge_profile("MONDO:1", "acute pancreatitis", [structured, text, other])
    assert len(profile.manifestations) == 2
    pain = profile.manifestations[0]
    assert pain.frequency == 0.9 and pain.typicality == "hallmark" and len(pain.citations) == 2
    assert hpo.find("abdominal pain") == "HP:0002027"
    graph = cg.CaseGraph("c")
    graph.add_disease(profile)
    assert graph.validate() == []


def test_the_pilot_list_was_written_before_the_run_and_comes_from_the_common_diagnoses():
    spec = importlib.util.spec_from_file_location("disease_profiles_script", ROOT / "scripts/disease_profiles.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    sspec = importlib.util.spec_from_file_location("source_coverage", ROOT / "scripts/source_coverage.py")
    sc = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(sc)
    assert len(script.PILOT) == 10 and set(script.PILOT) <= set(sc.DIAGNOSES)
    assert set(script.READERS) == {"nemotron-3-ultra", "gemma-4-31b"}
    assert json.dumps(script.JUDGES)
