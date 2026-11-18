"""The sources of the disease profiles (18 November 2026): read through the index, one connector per kind of
source, switched on and off without touching the script; the HPO term chosen in the same reading; what is not a
manifestation kept out; one finding in other words is one manifestation."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from melampo.memory import disease_profiles as dp
from melampo.memory import literature_connectors as lc
from melampo.memory import sources_index as si

ROOT = Path(__file__).resolve().parents[1]

HP = """format-version: 1.2

[Term]
id: HP:0002027
name: Abdominal pain

[Term]
id: HP:0033279
name: Right upper quadrant pain
synonym: "Pain in the right upper quadrant" EXACT []

[Term]
id: HP:0001945
name: Fever
"""

JATS = """<article><body>
<sec><title>Clinical presentation</title>
<p>Right upper quadrant pain is the hallmark of acute cholecystitis. Fever is common in acute cholecystitis.
Older age is a risk factor for acute cholecystitis and many other conditions in this population.</p></sec>
</body></article>"""


def _hpo():
    return dp.Terms.from_obo(HP.splitlines())


def _script(name="disease_profiles"):
    spec = importlib.util.spec_from_file_location(f"{name}_script", ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(**kw):
    base = dict(id="s", kind="database", title="S", url="https://example.org", access="open", status="available")
    return si.Source(**{**base, **kw})


# -- the index says what the profiles read ----------------------------------------------------------------------

def test_the_index_in_the_repository_marks_the_profile_sources_and_every_connector_exists():
    index = si.SourcesIndex.load()
    assert index.problems() == []
    assert set(lc.CONNECTORS) | {"wikipedia_index"} == set(si.CONNECTOR_NAMES)
    marked = {s.id: s for s in index.sources if s.profiles}
    assert {"europe-pmc", "pubmed", "ncbi-bookshelf", "wikipedia-index", "linked-documents", "open-pdfs",
            "private-pdfs"} <= set(marked)
    # the limit of two reviews is gone: the number of documents is the index's
    assert marked["europe-pmc"].max_documents >= 5
    # what a product may not use is read only by a research run
    assert marked["pubmed"].research_only and marked["ncbi-bookshelf"].research_only
    assert marked["private-pdfs"].research_only and not marked["open-pdfs"].research_only
    assert "data/private/" in (ROOT / ".gitignore").read_text()


def test_a_source_is_read_only_when_switched_on_reachable_and_allowed_by_the_run():
    index = si.SourcesIndex([
        _source(id="on", profiles=True, connector="europe_pmc_reviews"),
        _source(id="off", profiles=False, connector="europe_pmc_reviews"),
        _source(id="closed", profiles=True, connector="pubmed_abstracts", status="to_register",
                how_to_reach="register"),
        _source(id="research", profiles=True, connector="pubmed_abstracts", research_only=True),
        _source(id="wiki", profiles=True, connector="wikipedia_index"),
    ])
    use, skipped = lc.profile_sources(index)
    assert [s.id for s in use] == ["on"]
    assert {x.source for x in skipped} == {"closed", "research"}
    use, _ = lc.profile_sources(index, research=True)
    assert [s.id for s in use] == ["on", "research"]


def test_the_index_refuses_a_profile_source_without_a_way_to_read_it_and_keeps_its_types(tmp_path):
    assert any("connector" in p for p in _source(profiles=True).problems())
    assert any("connector" in p for p in _source(connector="nope").problems())
    assert any("path" in p for p in _source(connector="pdf_folder").problems())
    path = tmp_path / "sources.yml"
    index = si.SourcesIndex([], path, "# header\n")
    assert index.put(_source(id="src-a", profiles=True, connector="pdf_folder", path="x", max_documents=7,
                             research_only=True)) == []
    again = si.SourcesIndex.load(path).by_id()["src-a"]
    assert again.profiles is True and again.research_only is True and again.max_documents == 7
    assert si.source_from({"profiles": "1", "max_documents": "3", "diseases": "a, b"}).diseases == ["a", "b"]


def test_a_source_is_removed_but_not_one_that_documents_belong_to(tmp_path):
    index = si.SourcesIndex([], tmp_path / "s.yml")
    index.put(_source(id="lib", connector="documents"))
    index.put(_source(id="doc-1", kind="document", of="lib"))
    assert index.remove("lib") and "lib" in si.SourcesIndex.load(tmp_path / "s.yml").by_id()
    assert index.remove("doc-1") == [] and index.remove("lib") == []
    assert si.SourcesIndex.load(tmp_path / "s.yml").sources == []


# -- the connectors --------------------------------------------------------------------------------------------

class FakeWeb:
    def __init__(self, pages):
        self.pages, self.asked = pages, []

    def __call__(self, url, accept="application/json"):
        self.asked.append(url)
        for key, body in self.pages.items():
            if key in url:
                return body
        return ""


def _epmc_hit(pmcid, pmid, licence="cc by"):
    return {"pmcid": pmcid, "pmid": pmid, "doi": f"10.1/{pmid}", "title": "Acute cholecystitis: diagnosis",
            "license": licence, "pubYear": "2024", "inEPMC": "Y"}


def test_europe_pmc_reads_as_many_open_reviews_as_the_index_says_and_skips_closed_ones():
    long_jats = JATS.replace("</p></sec>", " " + "Murphy sign is often positive in acute cholecystitis. " * 40
                             + "</p></sec>")
    hits = [_epmc_hit(f"PMC{i}", str(1000 + i)) for i in range(1, 9)] + [_epmc_hit("PMC99", "99", "cc by-nc")]
    web = FakeWeb({"search?": json.dumps({"resultList": {"result": hits}}), "fullTextXML": long_jats})
    docs = lc.EuropePmcReviews(_source(connector="europe_pmc_reviews", max_documents=6), web).documents(
        "acute cholecystitis", ["acute cholecystitis"])
    assert len(docs) == 6 and all(d.citation["licence"] == "cc by" for d in docs)
    assert docs[0].citation["source_id"] == "s" and docs[0].citation["pmid"] == "1001"


def test_the_same_publication_from_two_sources_is_read_once_and_a_broken_source_is_reported(monkeypatch):
    index = si.SourcesIndex([_source(id="a", profiles=True, connector="europe_pmc_reviews"),
                             _source(id="b", profiles=True, connector="documents")])
    same = lc.Document({"id": "PMC1", "pmid": "1"}, [("t", "x")])
    monkeypatch.setattr(lc.EuropePmcReviews, "documents", lambda self, d, w: [same])

    def broken(self, d, w):
        raise ValueError("down")

    monkeypatch.setattr(lc.Documents, "documents", broken)
    reading = lc.read_sources("d", ["d"], index)
    assert len(reading.documents) == 1 and reading.per_source == {"a": 1}
    assert reading.skipped[0].source == "b" and "down" in reading.skipped[0].reason


def test_pubmed_abstracts_are_independent_documents_marked_research_only():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
    <ArticleTitle>Acute cholecystitis</ArticleTitle><Abstract><AbstractText>Fever is common.</AbstractText>
    </Abstract></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId>
    </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
    web = FakeWeb({"esearch": json.dumps({"esearchresult": {"idlist": ["111"]}}), "efetch": xml})
    docs = lc.PubmedAbstracts(_source(connector="pubmed_abstracts", research_only=True), web).documents("x", ["x"])
    assert docs[0].citation["pmid"] == "111" and docs[0].citation["doi"] == "10.1/x"
    assert docs[0].citation["use"] == "research only" and docs[0].sections == [("Abstract", "Fever is common.")]


def test_a_document_added_by_its_link_is_read_for_the_diseases_it_names():
    lib = _source(id="lib", kind="library", connector="documents", profiles=True)
    doc = _source(id="doc-pmc5", kind="document", of="lib", identifier="PMC5", title="A review",
                  diseases=["acute cholecystitis"])
    other = _source(id="doc-pmc6", kind="document", of="lib", identifier="PMC6", title="Gout")
    index = si.SourcesIndex([lib, doc, other])
    web = FakeWeb({"search?": json.dumps({"resultList": {"result": [_epmc_hit("PMC5", "5")]}}),
                   "fullTextXML": JATS})
    docs = lc.Documents(lib, web, index).documents("acute cholecystitis", ["acute cholecystitis"])
    assert len(docs) == 1 and docs[0].citation["document"] == "doc-pmc5"
    assert not any("PMC6" in u for u in web.asked)


@pytest.mark.skipif(not shutil.which("pdftotext"), reason="poppler not installed")
def test_a_pdf_in_a_folder_is_read_with_its_citation(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "cholecystitis.pdf"
    c = canvas.Canvas(str(pdf))
    for i, line in enumerate(["Clinical Presentation",
                              "Right upper quadrant pain is the hallmark of acute cholecystitis.",
                              "Fever is common in acute cholecystitis."]):
        c.drawString(72, 760 - 16 * i, line)
    c.save()
    pdf.with_suffix(".yml").write_text("title: Cholecystitis chapter\nidentifier: ISBN 1, p. 12\n"
                                       "licence: personal subscription\ndiseases: [acute cholecystitis]\n")
    src = _source(connector="pdf_folder", path=str(tmp_path), research_only=True)
    docs = lc.PdfFolder(src).documents("acute cholecystitis", ["acute cholecystitis"])
    assert len(docs) == 1 and docs[0].citation["id"] == "ISBN 1, p. 12"
    assert docs[0].citation["use"] == "research only"
    assert "hallmark" in docs[0].sections[0][1]
    assert lc.PdfFolder(src).documents("gout", ["gout"]) == []


def test_a_text_without_structure_gives_its_clinical_sections_or_all_of_it():
    text = "INTRODUCTION\nHistory of the name.\nClinical Features\nFever is common.\nTREATMENT\nFluids."
    assert lc.clinical_sections(text) == [("Clinical Features", "Fever is common.")]
    assert lc.clinical_sections("Fever is common in the disease.", "T") == [("T", "Fever is common in the disease.")]


# -- the corrections of the reading (problems 1 to 3 of the pilot) ---------------------------------------------

def test_one_finding_in_other_words_is_one_key_and_a_different_finding_is_not():
    assert dp.words_key("Right upper quadrant pain") == dp.words_key("Pain in the right upper quadrant")
    assert dp.words_key("Chest pain radiating to neck, jaw or back") == dp.words_key(
        "Chest pain radiating to the neck, jaw or back")
    assert dp.words_key("Abscesses") == dp.words_key("abscess")
    assert dp.words_key("Tenderness in the right upper quadrant") != dp.words_key("Right upper quadrant pain")
    assert dp.words_key("Severe pain") != dp.words_key("pain")


def test_what_every_reader_calls_a_risk_factor_is_not_a_manifestation_and_a_disagreement_goes_to_judges():
    sections = dp.sections_from_jats(JATS)
    quote = "Older age is a risk factor for acute cholecystitis and many other conditions in this population."
    item = {"manifestation": "Older age", "quote": quote, "typicality": "", "kind": "risk_factor",
            "modality": "", "region": "", "hpo": ""}
    both = dp.merge_readings({"a": [item], "b": [dict(item)]}, sections, {"id": "PMC1"}, _hpo(),
                             {"acute cholecystitis"})
    assert both[0].verdict == "not_a_manifestation"
    split = dp.merge_readings({"a": [item], "b": [{**item, "kind": "sign"}]}, sections, {"id": "PMC1"}, _hpo(),
                              {"acute cholecystitis"})
    assert split[0].verdict == ""


def test_a_reader_proposes_an_hpo_term_only_when_the_ontology_has_it():
    sections = dp.sections_from_jats(JATS)
    item = {"manifestation": "RUQ pain", "quote": "Right upper quadrant pain is the hallmark of acute cholecystitis.",
            "typicality": "", "kind": "symptom", "modality": "", "region": "", "hpo": "Right upper quadrant pain"}
    invented = {**item, "hpo": "Hepatic quadrant algia"}
    links = dp.merge_readings({"a": [item], "b": [invented]}, sections, {"id": "PMC1"}, _hpo())
    assert links[0].proposed == {"HP:0033279": "Right upper quadrant pain"}
    assert links[0].unknown_terms == ["Hepatic quadrant algia"]
    prompt = dp.judge_prompt("acute cholecystitis", links[0].__dict__, links[0].proposed)
    assert "HP:0033279" in prompt and '"term"' in prompt and "RUQ pain" in prompt
    assert dp.read_term('{"states": "yes", "term": "HP:0033279"}', links[0].proposed) == "HP:0033279"
    assert dp.read_term('{"states": "yes", "term": "HP:9999999"}', links[0].proposed) == ""
    assert dp.read_term('{"states": "yes", "term": "none"}', links[0].proposed) == "none"
    assert "risk factor" in dp.judge_prompt("x", {"quote": "q", "manifestation": "m"})


class FakeRouter:
    def __init__(self, answers):
        self.answers, self.prompts = answers, []

    def chat(self, model, prompt, **kw):
        self.prompts.append((model, prompt))
        return self.answers[model]


def _link():
    link = dp.Link("RUQ pain", "Right upper quadrant pain is the hallmark of acute cholecystitis.", {"id": "PMC1"})
    link.proposed = {"HP:0033279": "Right upper quadrant pain"}
    return link


def test_the_term_enters_only_when_every_judge_asked_chooses_it_and_the_words_stay():
    script = _script()
    hpo = _hpo()
    agree = {m: '{"states": "yes", "term": "HP:0033279"}' for m, _ in script.JUDGES.values()}
    pilot = script.Pilot(FakeRouter(agree), hpo, Path("/tmp/none"))
    link = _link()
    assert pilot.decide("acute cholecystitis", link) == "judges"
    pilot.settle_term(link)
    m = dp.to_manifestation(link, hpo, 0)
    assert m.concept.code == "HP:0033279" and m.words == "RUQ pain"
    assert m.citations[0]["hpo_votes"] == {j: "HP:0033279" for j in script.JUDGES}

    split = dict(agree)
    split[script.JUDGES["gemma-4-31b"][0]] = '{"states": "yes", "term": "none"}'
    pilot = script.Pilot(FakeRouter(split), hpo, Path("/tmp/none"))
    link = _link()
    assert pilot.decide("acute cholecystitis", link) == "judges"
    pilot.settle_term(link)
    m = dp.to_manifestation(link, hpo, 0)
    assert m.concept is None and m.words == "RUQ pain"


def test_findings_in_other_words_merge_into_one_manifestation_with_both_sources():
    hpo = _hpo()
    a = dp.to_manifestation(dp.Link("Pain radiating to the back", "q1", {"id": "PMC1", "pmid": "1"}), hpo, 0)
    b = dp.to_manifestation(dp.Link("pain radiating to back", "q2", {"id": "PMC2", "pmid": "2"}), hpo, 1)
    profile = dp.merge_profile("MONDO:1", "d", [a, b])
    assert len(profile.manifestations) == 1 and profile.manifestations[0].evidence == "established"


# -- the mask ----------------------------------------------------------------------------------------------------

def test_the_mask_switches_a_source_adds_a_link_and_keeps_a_closed_pdf_out_of_the_repository(tmp_path, monkeypatch):
    mask = _script("sources_mask")
    path = tmp_path / "sources.yml"
    shutil.copy(si.SOURCES, path)
    index = si.SourcesIndex.load(path)
    html = mask.page(index, {"status": ["profiles"]})
    assert "Profili delle malattie" in html and "disattiva" in html and 'action="/pdf"' in html

    doc = mask.document_from_link({"link": "doi:10.1000/ABC", "title": "A guideline", "of": "linked-documents",
                                   "diseases": "acute pancreatitis, cirrhosis"})
    assert doc.identifier == "10.1000/abc" and doc.url == "https://doi.org/10.1000/abc"
    assert index.put(doc) == [] and si.SourcesIndex.load(path).by_id()[doc.id].diseases == [
        "acute pancreatitis", "cirrhosis"]

    open_dir, private_dir = tmp_path / "open", tmp_path / "private"
    for s in index.sources:
        if s.id == "open-pdfs":
            s.path = str(open_dir)
        if s.id == "private-pdfs":
            s.path = str(private_dir)
    msg, bad = mask.save_pdf(index, {"title": "Chapter", "licence": "personal subscription",
                                     "folder": "open-pdfs"}, b"%PDF-1.4 x", "chap ter.pdf")
    assert not bad and (private_dir / "chap_ter.pdf").exists() and not open_dir.exists()
    msg, bad = mask.save_pdf(index, {"title": "Review", "licence": "CC BY 4.0", "folder": "open-pdfs"},
                             b"%PDF-1.4 y", "review.pdf")
    assert not bad and (open_dir / "review.pdf").exists() and "licence: CC BY 4.0" in (open_dir / "review.yml").read_text()
    assert mask.save_pdf(index, {"title": "x"}, b"not a pdf", "x.pdf")[1]

    body = (b"--B\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nT\r\n--B\r\nContent-Disposition: "
            b"form-data; name=\"file\"; filename=\"a.pdf\"\r\nContent-Type: application/pdf\r\n\r\n%PDF-1\r\n--B--\r\n")
    fields, data, name = mask.multipart("multipart/form-data; boundary=B", body)
    assert fields == {"title": "T"} and data == b"%PDF-1" and name == "a.pdf"


def test_the_mask_page_is_valid_for_every_source_of_the_repository():
    mask = _script("sources_mask")
    html = mask.page(si.SourcesIndex.load(), {})
    assert html.count("<tr>") == len(si.SourcesIndex.load().sources) + 1


def test_a_whole_run_reads_the_index_sources_and_writes_concepts_sources_and_skipped(tmp_path, monkeypatch):
    """The script end to end, with the web and the models replaced: what the index switches on is read, what it
    switches off or keeps for research is listed with why, and the profile has its concepts."""
    script = _script()
    files = {"mondo": "[Term]\nid: MONDO:0043994\nname: acute cholecystitis\n", "hpo": HP, "hpoa": "",
             "symp": "", "doid-owl": ""}
    for name, body in files.items():
        (tmp_path / name).write_text(body)
    index_path = tmp_path / "sources.yml"
    shutil.copy(si.SOURCES, index_path)
    long_jats = JATS.replace("</p></sec>", " " + "Fever is common in acute cholecystitis. " * 60 + "</p></sec>")
    web = FakeWeb({"europepmc/webservices/rest/search?": json.dumps(
        {"resultList": {"result": [_epmc_hit("PMC1", "1"), _epmc_hit("PMC2", "2")]}}),
        "fullTextXML": long_jats})
    monkeypatch.setattr(script, "get", web)
    monkeypatch.setattr(script, "parse_doid_owl", lambda path: [])
    monkeypatch.setattr(script, "wikipedia_for", lambda name: ({}, [], []))
    reading = json.dumps([
        {"manifestation": "RUQ pain", "quote": "Right upper quadrant pain is the hallmark of acute cholecystitis.",
         "hpo": "Right upper quadrant pain", "typicality": "hallmark", "kind": "symptom"},
        {"manifestation": "Older age", "quote": "Older age is a risk factor for acute cholecystitis and many other "
         "conditions in this population.", "hpo": "", "typicality": "", "kind": "risk_factor"},
        {"manifestation": "Fever", "quote": "Fever is common in acute cholecystitis.", "hpo": "Fever",
         "typicality": "common", "kind": "sign"}])

    class Router:
        def __init__(self, key, budget):
            self.cost, self.budget, self.calls, self.tokens, self.last = 0.0, budget, {}, {}, {}

        def chat(self, model, prompt, **kw):
            if "List every manifestation" in prompt:
                return reading
            return '{"states": "yes", "term": "HP:0033279"}'

    monkeypatch.setattr(script, "OpenRouter", Router)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    out = tmp_path / "out" / "profiles.json"
    out.parent.mkdir()
    assert script.main(["--mondo", str(tmp_path / "mondo"), "--hpo", str(tmp_path / "hpo"), "--hpoa",
                        str(tmp_path / "hpoa"), "--doid-owl", str(tmp_path / "doid-owl"), "--symp",
                        str(tmp_path / "symp"), "--diagnoses", "acute cholecystitis", "--gold", "0",
                        "--sources", str(index_path), "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    row = got["summary"]["rows"][0]
    assert row["sources"]["europe-pmc"] == 2 and "pubmed" in row["sources_skipped"]
    ms = {m["words"]: m for m in got["profiles"][0]["manifestations"]}
    assert set(ms) == {"RUQ pain", "Fever"}                      # the risk factor is not a manifestation
    assert ms["RUQ pain"]["concept"]["code"] == "HP:0033279"     # proposed, chosen by the judges, words kept
    assert ms["Fever"]["evidence"] == "established"              # two reviews state it
    assert row["verdicts"].get("not_a_manifestation") == 2
