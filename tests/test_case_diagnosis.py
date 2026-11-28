"""The diagnostic path on the case graph (20 November 2026): profiles to the named diseases, candidates only from the
path, states, ranking, exams to propose, and the profile library that holds the knowledge."""

import json

from melampo.clinical import case_graph as cg
from melampo.clinical.case_graph_builder import add_declared_diseases
from melampo.memory.profile_library import ProfileLibrary
from melampo.reasoning.case_diagnosis import diagnose

SOURCE = [{"source": "Europe PMC review", "id": "PMC1"}]


def m(mid, words, code, typicality="", **kw):
    return cg.Manifestation(mid, words, cg.Concept("HPO", code, words), typicality=typicality, citations=SOURCE,
                            evidence="established", **kw)


APPENDICITIS = cg.Disease("MONDO:0004974", "hypothesized_diseases", "acute appendicitis", "profile", manifestations=[
    m("m1", "right lower quadrant pain", "HP:0100281", "hallmark"),
    m("m2", "fever", "HP:0001945", "common"),
    m("m3", "nausea", "HP:0002018", "common"),
    m("m4", "leukocytosis", "HP:0001974", "common")])
CHOLECYSTITIS = cg.Disease("MONDO:0001033", "hypothesized_diseases", "acute cholecystitis", "profile", manifestations=[
    m("m1", "right upper quadrant pain", "HP:0100280", "hallmark"),
    m("m2", "nausea", "HP:0002018", "common"),
    m("m3", "positive Murphy sign", "HP:0031277", "common")])
PANCREATITIS = cg.Disease("MONDO:0005084", "hypothesized_diseases", "acute pancreatitis", "profile", manifestations=[
    m("m1", "epigastric pain", "HP:0410019", "hallmark"),
    m("m2", "nausea", "HP:0002018", "common")])


def library(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": [d.as_dict() for d in (APPENDICITIS, CHOLECYSTITIS, PANCREATITIS)]}))
    names = tmp_path / "names.json"
    names.write_text(json.dumps({"MONDO:0004974": ["appendicite acuta"]}))
    return ProfileLibrary.load(path, names)


def case():
    graph = cg.CaseGraph("case-1")
    graph.documents.append(cg.Document("note", "note"))
    for fid, words, code, polarity in (("f1", "dolore in fossa iliaca destra", "HP:0100281", cg.AFFIRMED),
                                       ("f2", "apiretico", "HP:0001945", cg.NEGATED),
                                       ("f3", "nausea", "HP:0002018", cg.AFFIRMED),
                                       ("f4", "dolore all'ipocondrio destro", "HP:0100280", cg.AFFIRMED)):
        graph.findings.append(cg.Finding(fid, "note", words, cg.Provenance("note", words), polarity=polarity,
                                         concept=cg.Concept("HPO", code, words)))
    add_declared_diseases(graph, [{"words": "sospetta appendicite acuta", "status": "suspected"}])
    return graph


def test_the_library_has_a_version_and_finds_by_concept_and_by_name(tmp_path):
    lib = library(tmp_path)
    assert len(lib.version) == 16 and len(lib.profiles) == 3
    assert lib.find(concept=cg.Concept("MONDO", "MONDO:0001033"))[1] == "concept"
    assert lib.find(words="Sospetta appendicite acuta")[0].disease_id == "MONDO:0004974"
    assert lib.find(words="colica renale") == (None, "no_profile")
    again = ProfileLibrary.load(tmp_path / "profiles.json", tmp_path / "names.json")
    assert again.version == lib.version


def test_the_path_attaches_profiles_adds_candidates_and_ranks(tmp_path):
    graph = case()
    assert graph.branch(cg.CANDIDATE_DISEASES) == []
    out = diagnose(graph, library(tmp_path), proposals=[{"condition": "acute pancreatitis", "specificity_score": 0.4}])
    assert out["graph_problems"] == []
    assert out["named"][0]["profile"] == "name" and out["named"][0]["states"] == {
        "present": 2, "absent": 1, "not_assessed": 1}
    proposed = {c["words"]: c["proposed_by"][0]["proposed_by"] for c in out["candidates"]}
    # pancreatitis from the differential; cholecystitis from the lateral part of the intuition engine, because it
    # accounts for the pain nothing else explains (with appendicitis, as a pair: each explains what the other leaves)
    assert proposed == {"acute pancreatitis": "differential_from_findings",
                        "acute cholecystitis": "lateral_intuition"}
    lateral = out["lateral"]
    assert lateral["salient"] == [{"observation": "f4", "words": "dolore all'ipocondrio destro",
                                   "reasons": ["unexplained"]}]
    assert [a["kind"] for a in lateral["accepted"]] == ["pair"]
    assert lateral["accepted"][0]["status"] == "hypothesis_to_exclude" and lateral["accepted"][0]["insight"]
    assert all(d.origin == cg.DIAGNOSTIC_PATH for d in graph.branch(cg.CANDIDATE_DISEASES))
    ranking = [r["words"] for r in out["ranking"]]
    assert set(ranking) == {"sospetta appendicite acuta", "acute cholecystitis", "acute pancreatitis"}
    assert ranking[-1] == "acute pancreatitis"            # its hallmark is not there, nothing else of it is special
    first = out["ranking"][0]
    assert first["contributions"] and abs(sum(r["posterior"] for r in out["ranking"]) - 1) < 1e-3
    assert out["knowledge"]["disease_profiles"] == graph.snapshot.sources["disease_profiles"]
    assert out["unexplained_findings"] == []


def test_the_exams_to_propose_are_what_nothing_in_the_case_could_show(tmp_path):
    out = diagnose(case(), library(tmp_path))
    assert {"disease": "sospetta appendicite acuta", "manifestation": "leukocytosis"}.items() <= next(
        e for e in out["exams_to_propose"] if e["manifestation"] == "leukocytosis").items()


def test_a_disease_without_a_profile_says_so_and_is_not_ranked_on_nothing(tmp_path):
    graph = cg.CaseGraph("case-2")
    add_declared_diseases(graph, [{"words": "sospetta colica renale", "status": "suspected"}])
    out = diagnose(graph, library(tmp_path))
    assert out["named"][0]["profile"] == "no_profile" and out["ranking"] == []
    assert any("no_profile" in n for n in out["notes"])


def test_two_files_join_their_manifestations_each_with_its_source(tmp_path):
    folder = tmp_path / "profiles"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps({"profiles": [APPENDICITIS.as_dict()]}))
    other = cg.Disease("MONDO:0004974", "hypothesized_diseases", "acute appendicitis", "profile", manifestations=[
        cg.Manifestation("x1", "anorexia", cg.Concept("HPO", "HP:0002039"), citations=[{"id": "PMC2"}])])
    (folder / "b.json").write_text(json.dumps({"profiles": [other.as_dict()]}))
    lib = ProfileLibrary.load(folder)
    assert [x.words for x in lib.profiles["MONDO:0004974"].manifestations][-1] == "anorexia"
    assert lib.files == ["a.json", "b.json"]


def test_the_pipeline_result_comes_from_the_path(tmp_path, monkeypatch):
    from melampo.app import build_default_runtime

    library(tmp_path)
    monkeypatch.setenv("MELAMPO_DISEASE_PROFILES", str(tmp_path / "profiles.json"))
    monkeypatch.setenv("MELAMPO_DISEASE_NAMES", str(tmp_path / "names.json"))
    monkeypatch.delenv("MELAMPO_HPO_OBO", raising=False)
    runtime = build_default_runtime()
    result = runtime.pipeline.run({"case_id": "case-path", "report_text": "dolore addominale",
                                   "diseases": [{"words": "sospetta appendicite acuta", "status": "suspected"}]})
    path = result["case_diagnosis"]
    assert path["status"] == "ok", path
    assert path["named"][0]["profile"] == "name" and path["ranking"][0]["words"] == "sospetta appendicite acuta"
    # since 28 November 2026 the result is the path's ranking; the October path is only compared
    final = result["diagnostic_result"]
    assert final["result_label"] == path["ranking"][0]["words"]
    assert final["differential"][0]["disease"] == path["ranking"][0]["disease"]
    assert final["diagnosis"]["status"] == "ok"
    assert final["october_path"]["role"] == "comparison_only_until_december_2026_evaluation"


def test_without_profiles_the_pipeline_says_so(monkeypatch):
    from melampo.app import build_default_runtime

    monkeypatch.delenv("MELAMPO_DISEASE_PROFILES", raising=False)
    result = build_default_runtime().pipeline.run({"case_id": "case-none", "report_text": "dolore addominale"})
    assert result["case_diagnosis"]["status"] in ("no_profiles_configured", "no_case_graph")
    assert result["diagnostic_result"]["result_label"] == "no_candidate_found"
