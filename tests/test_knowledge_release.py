"""A version of the disease knowledge for a GitHub Release (22 November 2026): made only when it changed, with its
graph ready in FalkorDBLite and what changed written."""

import importlib.util
import json
from pathlib import Path

import pytest

from melampo.clinical import case_graph as cg

SPEC = importlib.util.spec_from_file_location(
    "knowledge_release", Path(__file__).resolve().parents[1] / "scripts" / "knowledge_release.py")
kr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kr)


def _run(tmp_path, name, extra=()):
    out = tmp_path / name
    out.mkdir()
    d = cg.Disease("MONDO:0004974", "hypothesized_diseases", "acute appendicitis", "profile", manifestations=[
        cg.Manifestation("m1", "right lower quadrant pain", cg.Concept("HPO", "HP:0100281"), frequency=0.8,
                         frequency_basis="one_study", citations=[{"pmid": "1"}]), *extra])
    (out / "profiles.json").write_text(json.dumps({"profiles": [d.as_dict()]}))
    (out / "reviews.json").write_text(json.dumps({"acute appendicitis": {"date": "2026-11-22", "texts_read": 6,
                                                                         "stopped_by": "saturation"}}))
    (out / "measures.jsonl").write_text("")
    return out


def _say(capsys) -> dict:
    return dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines() if "=" in line)


def test_a_release_is_made_only_when_the_knowledge_changed(tmp_path, capsys):
    pytest.importorskip("redislite")
    first = tmp_path / "release1"
    assert kr.main(["--run", str(_run(tmp_path, "out1")), "--release", str(first)]) == 0
    said = _say(capsys)
    assert said["changed"] == "true" and said["tag"].startswith("knowledge-")
    assert (first / "knowledge.db").exists() and "New: 1 manifestations" in (first / "changes.md").read_text()
    # the same review again: nothing changed, no Release
    again = tmp_path / "release2"
    assert kr.main(["--previous", str(first), "--run", str(_run(tmp_path, "out2")), "--release", str(again)]) == 0
    assert _say(capsys)["changed"] == "false"
    # a week later a manifestation more
    fever = cg.Manifestation("m2", "fever", cg.Concept("HPO", "HP:0001945"), citations=[{"pmid": "2"}])
    third = tmp_path / "release3"
    assert kr.main(["--previous", str(first), "--run", str(_run(tmp_path, "out3", [fever])),
                    "--release", str(third)]) == 0
    assert _say(capsys)["changed"] == "true"
    assert "Added: fever." in (third / "changes.md").read_text()
    version = json.loads((third / "version.json").read_text())
    assert version["previous"] == json.loads((first / "version.json").read_text())["version"]


def test_the_diagnosis_opens_the_graph_of_a_release(tmp_path, capsys, monkeypatch):
    pytest.importorskip("redislite")
    release = tmp_path / "release"
    kr.main(["--run", str(_run(tmp_path, "out")), "--release", str(release)])
    capsys.readouterr()
    from melampo.memory.profile_library import ProfileLibrary

    monkeypatch.setenv("MELAMPO_DISEASE_PROFILES", str(release / "knowledge.db"))
    lib = ProfileLibrary.from_env()
    assert lib is not None and lib.version == json.loads((release / "version.json").read_text())["version"]
    assert lib.find(words="acute appendicitis")[1] == "name"
