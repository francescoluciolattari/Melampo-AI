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


def test_a_term_named_by_its_own_name_wins_over_the_terms_that_carry_it_as_a_synonym():
    obo = ["[Term]", "id: MONDO:1", "name: pneumonia", "", "[Term]", "id: MONDO:2", "name: lung inflammation",
           'synonym: "pneumonia" EXACT []', "", "[Term]", "id: MONDO:3", "name: pneumonitis",
           'synonym: "pneumonia" EXACT []']
    terms = dp.Terms.from_obo(obo)
    assert terms.find("pneumonia") == "MONDO:1"                 # the first pilot found nothing here
    assert terms.find("lung inflammation") == "MONDO:2"
    two = dp.Terms.from_obo(["[Term]", "id: A:1", 'name: x', 'synonym: "y" EXACT []', "", "[Term]", "id: A:2",
                             "name: z", 'synonym: "y" EXACT []'])
    assert two.find("y") == ""                                  # two synonyms, no name: nothing chosen


def test_an_answer_with_prose_and_citations_before_its_list_is_read():
    answer = ('Based on the text [1], here is the list:\n```json\n[{"manifestation": "Fever", "quote": "Fever is '
              'rare at onset."}]\n```')
    assert [i["manifestation"] for i in dp.read_extraction(answer)] == ["Fever"]
    assert dp.read_extraction('{"manifestations": [{"manifestation": "Fever", "quote": "x."}]}')[0]["quote"] == "x."


def test_a_sentence_about_a_tests_value_or_an_outcome_goes_to_the_judges_even_when_it_names_the_finding():
    hpo = _hpo()
    item = {"manifestation": "Fever", "quote": "Fever was an independent predictor of mortality."}
    assert not dp.names_it(item, hpo)
    assert dp.names_it({"manifestation": "Fever", "quote": "Fever is rare at onset."}, hpo)
    xml = ("<article><sec><title>Diagnostic accuracy of CT</title><p>CT had 95% sensitivity.</p></sec>"
           "<sec><title>Clinical features</title><p>Pain.</p></sec>"
           "<sec><title>Risk factors and prognosis</title><p>Age.</p></sec></article>")
    assert [t for t, _ in dp.sections_from_jats(xml)] == ["Clinical features"]


def test_mondo_is_looked_up_among_its_own_terms():
    obo = ["[Term]", "id: MONDO:0005249", "name: pneumonia", "", "[Term]", "id: HP:0002090", "name: Pneumonia"]
    terms = dp.Terms.from_obo(obo)
    assert terms.find("pneumonia") == ""                        # two terms with that name: none chosen
    assert terms.find("pneumonia", prefix="MONDO:") == "MONDO:0005249"
    assert dp.NOT_A_MANIFESTATION.search("SAP was associated with increased vWF in rats")


def test_a_sentence_is_verified_by_its_words_only_when_it_names_the_disease_too():
    text = ("Acute cholecystitis (AC) presents with fever. Bartter syndrome presents with hypokalemic alkalosis. "
            "Fever occurs in rats with AC.")
    sections = [("Clinical presentation", text)]
    words = dp.disease_words("acute cholecystitis", ["cholecystitis, acute"], sections)
    assert {"AC", "acute cholecystitis"} <= words
    readings = {"r": [{"manifestation": "Fever", "quote": "Acute cholecystitis (AC) presents with fever.",
                       "typicality": "", "kind": "symptom", "modality": "", "region": ""},
                      {"manifestation": "Hypokalemic alkalosis",
                       "quote": "Bartter syndrome presents with hypokalemic alkalosis.", "typicality": "",
                       "kind": "laboratory", "modality": "", "region": ""},
                      {"manifestation": "Fever", "quote": "Fever occurs in rats with AC.", "typicality": "",
                       "kind": "symptom", "modality": "", "region": ""}]}
    links = dp.merge_readings(readings, sections, {}, _hpo(), words)
    assert [link.verdict for link in links] == ["names", "", ""]     # Bartter and the rats go to the judges
    assert "animals" in dp.judge_prompt("acute cholecystitis", {"quote": "q", "manifestation": "m"})


WIKI = """<div class="mw-parser-output"><p>Cholecystitis is inflammation of the gallbladder.<sup class="reference">[1]
</sup></p><div class="mw-heading mw-heading2"><h2 id="Signs_and_symptoms">Signs and symptoms</h2></div>
<p>Pain in the right upper abdomen is the most common symptom.<sup class="reference">[2]</sup></p>
<ul><li>Fever</li><li>Nausea and vomiting</li></ul>
<div class="mw-heading mw-heading3"><h3 id="Murphy">Murphy's sign</h3></div><p>A positive Murphy sign.</p>
<div class="mw-heading mw-heading2"><h2 id="Diagnosis">Diagnosis</h2></div><table><tr><td>score</td></tr></table>
<p>Ultrasound shows a thickened gallbladder wall.</p>
<div class="mw-heading mw-heading2"><h2 id="Treatment">Treatment</h2></div><p>Surgery.</p></div>"""


def test_wikipedia_gives_its_presentation_and_diagnosis_sections_without_references_or_tables():
    sections = dp.wikipedia_sections(WIKI)
    assert [t for t, _ in sections] == ["Signs and symptoms", "Diagnosis"]
    symptoms = sections[0][1]
    assert "most common symptom." in symptoms and "[2]" not in symptoms and "Fever" in symptoms
    assert "Murphy sign" in symptoms                       # a subsection belongs to its section
    assert "score" not in sections[1][1] and "thickened gallbladder wall" in sections[1][1]


def test_only_licences_open_for_a_commercial_product_are_read():
    assert dp.open_for_use("cc by") and dp.open_for_use("CC BY 4.0") and dp.open_for_use("cc0")
    assert dp.open_for_use("cc by-sa")
    assert not dp.open_for_use("cc by-nc") and not dp.open_for_use("cc by-nc-nd") and not dp.open_for_use("")
    assert not dp.open_for_use("cc by-nd")


WIKI_REFS = ('<h2>Signs and symptoms</h2><p>Fever.<sup class="reference"><a href="#cite_note-a">[1]</a></sup> Pain.'
             '<sup class="reference"><a href="#cite_note-b">[2]</a></sup></p><h2>Treatment</h2><p>Drugs.'
             '<sup class="reference"><a href="#cite_note-c">[3]</a></sup></p><h2>References</h2><ol>'
             '<li id="cite_note-a"><a href="https://pubmed.ncbi.nlm.nih.gov/111/">PMID 111</a> '
             '<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC222/">PMC222</a></li>'
             '<li id="cite_note-b"><a href="https://doi.org/10.1000/xyz.1">doi</a></li>'
             '<li id="cite_note-c"><a href="https://pubmed.ncbi.nlm.nih.gov/333/">PMID 333</a></li></ol>')


def test_wikipedia_gives_the_publications_its_presentation_sections_cite_never_its_own_text_as_a_source():
    refs = dp.wikipedia_references(WIKI_REFS)
    assert refs == [{"pmid": "111", "pmcid": "PMC222", "doi": ""}, {"pmid": "", "pmcid": "", "doi": "10.1000/xyz.1"}]
    text = ("Acute cholecystitis is common. In acute cholecystitis, fever was present in 30% of patients. "
            "Fever predicted mortality in acute cholecystitis. Fever is frequent in sepsis.")
    found = dp.supporting_sentences(text, "Fever", _hpo(), {"acute cholecystitis"})
    assert found == ["In acute cholecystitis, fever was present in 30% of patients."]
    assert dp.supporting_sentences("Fever is frequent.", "Fever", _hpo(), {"acute cholecystitis"},
                                   title="Fever in acute cholecystitis") == ["Fever is frequent."]
    body = ("<article><body><sec><title>Methods</title><p>We enrolled.</p></sec><sec><title>Results</title>"
            "<p>Fever in 30%.</p></sec></body><back><ref-list><ref>x</ref></ref-list></back></article>")
    assert dp.body_text(body) == "Fever in 30%."


def test_two_independent_sources_establish_a_manifestation_one_reports_it():
    one = cg.Manifestation("HP:1", "x", citations=[{"source": "Europe PMC review", "id": "PMC1"},
                                                   {"source": "Europe PMC review", "id": "PMC1"}])
    two = cg.Manifestation("HP:1", "x", citations=[{"source": "Europe PMC review", "id": "PMC1"},
                                                   {"source": "HPO annotations", "id": "OMIM:1", "reference": "PMID:9"}])
    assert dp.evidence_of(one) == "reported" and dp.evidence_of(two) == "established"
    profile = dp.merge_profile("MONDO:1", "d", [two])
    assert profile.manifestations[0].evidence == "established"


def test_a_term_read_agrees_with_a_curated_profile_when_it_is_the_same_more_specific_or_a_little_more_general():
    obo = ["[Term]", "id: HP:1", "name: root", "", "[Term]", "id: HP:2", "name: mid", "is_a: HP:1", "",
           "[Term]", "id: HP:3", "name: leaf", "is_a: HP:2", "", "[Term]", "id: HP:4", "name: deep", "is_a: HP:3", "",
           "[Term]", "id: HP:9", "name: other", "is_a: HP:1"]
    hpo = dp.Terms.from_obo(obo)
    assert hpo.ancestors("HP:4") == {"HP:3", "HP:2", "HP:1"}
    assert dp.agrees_with_curation("HP:3", {"HP:3"}, hpo)
    assert dp.agrees_with_curation("HP:4", {"HP:2"}, hpo)          # more specific than the curators
    assert dp.agrees_with_curation("HP:2", {"HP:4"}, hpo)          # two steps more general
    assert not dp.agrees_with_curation("HP:1", {"HP:4"}, hpo)      # three steps: too general to agree
    assert not dp.agrees_with_curation("HP:9", {"HP:4"}, hpo)
