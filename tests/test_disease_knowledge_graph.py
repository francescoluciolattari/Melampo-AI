"""The disease knowledge in FalkorDB, version by version (22 November 2026), on the embedded engine."""

import importlib.util
import json
from pathlib import Path

import pytest

from melampo.clinical import case_graph as cg
from melampo.memory import disease_knowledge_graph as kg
from melampo.memory.profile_library import ProfileLibrary

SPEC = importlib.util.spec_from_file_location(
    "knowledge_graph_script", Path(__file__).resolve().parents[1] / "scripts" / "knowledge_graph.py")
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


def _folder(tmp_path, extra=()):
    d = cg.Disease("MONDO:0004974", "hypothesized_diseases", "acute appendicitis", "profile", manifestations=[
        cg.Manifestation("m1", "right lower quadrant pain", cg.Concept("HPO", "HP:0100281"), frequency=0.8,
                         frequency_basis="one_study", citations=[{"pmid": "1"}]),
        cg.Manifestation("m2", "fever", cg.Concept("HPO", "HP:0001945"),
                         accuracy={"sensitivity": 0.4, "specificity": 0.7, "setting": "emergency department"},
                         citations=[{"pmid": "2"}]), *extra])
    p = cg.Disease("MONDO:0005249", "hypothesized_diseases", "pneumonia", "profile", manifestations=[
        cg.Manifestation("p1", "fever", cg.Concept("HPO", "HP:0001945"), typicality="common",
                         citations=[{"pmid": "3"}])])
    folder = tmp_path / "kb"
    folder.mkdir(exist_ok=True)
    (folder / "profiles.json").write_text(json.dumps({"profiles": [d.as_dict(), p.as_dict()]}))
    (folder / "names.json").write_text(json.dumps({"MONDO:0004974": ["appendicite acuta"]}))
    (folder / "tests.json").write_text(json.dumps({"acute appendicitis": [
        {"name": "CT", "accuracy": {"sensitivity": 0.98, "specificity": 0.95, "setting": "emergency department"}}]}))
    (folder / "reviews.json").write_text(json.dumps({"acute appendicitis": {"date": "2026-11-22"}}))
    return folder


@pytest.fixture
def graph(tmp_path):
    pytest.importorskip("redislite")
    from redislite.falkordb_client import FalkorDB

    return FalkorDB(str(tmp_path / "k.db")).select_graph("knowledge_test")


def test_a_version_is_loaded_once_and_read_back_as_the_library(tmp_path, graph, capsys):
    folder = _folder(tmp_path)
    assert script.main(["--knowledge", str(folder)], graph=graph) == 0
    first = json.loads(capsys.readouterr().out)
    assert (first["diseases"], first["manifestations"], first["tests"], first["already_there"]) == (2, 3, 1, False)
    script.main(["--knowledge", str(folder)], graph=graph)
    assert json.loads(capsys.readouterr().out)["already_there"] is True          # not loaded twice
    lib = ProfileLibrary.from_graph(graph)
    on_file = ProfileLibrary.load(folder / "profiles.json", folder / "names.json")
    assert lib.version == on_file.version == first["version"]                   # the same id everywhere
    assert lib.find(words="sospetta appendicite acuta")[0].disease_id == "MONDO:0004974"
    fever = next(m for m in lib.profiles["MONDO:0004974"].manifestations if m.words == "fever")
    assert fever.accuracy["sensitivity"] == 0.4


def test_the_diseases_that_show_some_manifestations_are_one_query(tmp_path, graph):
    script.main(["--knowledge", str(_folder(tmp_path))], graph=graph)
    assert kg.diseases_with(graph, ["HP:0001945", "HP:0100281"]) == [("MONDO:0004974", 2), ("MONDO:0005249", 1)]


def test_a_new_version_becomes_current_and_the_old_one_can_still_be_read(tmp_path, graph):
    script.main(["--knowledge", str(_folder(tmp_path))], graph=graph)
    old = kg.current_version(graph)
    vomiting = cg.Manifestation("m3", "vomiting", cg.Concept("HPO", "HP:0002013"), citations=[{"pmid": "9"}])
    script.main(["--knowledge", str(_folder(tmp_path, [vomiting]))], graph=graph)
    new = kg.current_version(graph)
    assert new != old and len(kg.versions(graph)) == 2
    assert len(ProfileLibrary.from_graph(graph, old).profiles["MONDO:0004974"].manifestations) == 2
    assert len(ProfileLibrary.from_graph(graph).profiles["MONDO:0004974"].manifestations) == 3
