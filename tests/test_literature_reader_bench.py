"""The literature reader bench, without network: a fake router, texts already read (20 November 2026)."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "literature_reader_bench", Path(__file__).resolve().parents[1] / "scripts" / "literature_reader_bench.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

HPO = """format-version: 1.2

[Term]
id: HP:0002027
name: Abdominal pain

[Term]
id: HP:0001945
name: Fever
"""
MONDO = """format-version: 1.2

[Term]
id: MONDO:0004974
name: appendicitis
"""
TEXT = ("Right lower quadrant abdominal pain is typical of acute appendicitis. "
        "Fever is common in acute appendicitis.")


def fake_chat(answers):
    def chat(self, model, prompt, reasoning, max_tokens):
        if self.cost >= self.limit:            # the check of the real chat, before the call
            raise bench.OverBudget("over")
        self.cost += 0.01
        if prompt.startswith("Disease:"):
            return {"content": answers["judge"](prompt), "finish_reason": "stop", "reasoning_chars": 0,
                    "cost": 0.01, "seconds": 0.1, "tokens_out": 5}
        return {"content": answers[model], "finish_reason": "stop", "reasoning_chars": 0, "cost": 0.01,
                "seconds": 0.1, "tokens_out": 50}
    return chat


def run(tmp_path, monkeypatch, answers, budget=5.0):
    for name, body in (("hp.obo", HPO), ("mondo.obo", MONDO), ("phenotype.hpoa", "#\n")):
        (tmp_path / name).write_text(body)
    texts = [{"diagnosis": "acute appendicitis", "kind": "common", "curated": [],
              "reviews": [{"id": "PMC1", "title": "t", "licence": "cc by", "sections": [["Presentation", TEXT]]}]}]
    (tmp_path / "texts.json").write_text(json.dumps(texts))
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(bench.Router, "chat", fake_chat(answers))
    code = bench.main(["--mondo", str(tmp_path / "mondo.obo"), "--hpo", str(tmp_path / "hp.obo"),
                       "--hpoa", str(tmp_path / "phenotype.hpoa"), "--budget", str(budget),
                       "--out", str(tmp_path / "bench.json")])
    return code, json.loads((tmp_path / "bench_summary.json").read_text()) if code == 0 else None


PAIN = {"manifestation": "abdominal pain",
        "quote": "Right lower quadrant abdominal pain is typical of acute appendicitis."}
FEVER = {"manifestation": "fever", "quote": "Fever is common in acute appendicitis."}
MADE_UP = {"manifestation": "vomiting", "quote": "Vomiting is seen in every patient."}


def test_the_rule_prefers_coverage_and_rejects_invented_quotes(tmp_path, monkeypatch):
    answers = {bench.NEMOTRON: json.dumps([PAIN]), bench.GEMMA: json.dumps([PAIN, MADE_UP]),
               bench.MUSE: json.dumps([PAIN, FEVER]), "judge": lambda p: '{"states": "yes"}'}
    code, summary = run(tmp_path, monkeypatch, answers)
    assert code == 0
    table = summary["table"]
    assert table["A"]["invented_quotes"] == 1 and table["B"]["invented_quotes"] == 0
    assert table["B"]["coverage"] == 1.0 and table["A"]["coverage"] == 0.5
    assert summary["winner"] == "B"   # B and C equal: the same answers and the same cost, B first


def test_a_link_one_judge_refuses_is_not_accepted(tmp_path, monkeypatch):
    def judge(prompt):
        return '{"states": "no"}' if "Fever" in prompt else '{"states": "yes"}'
    answers = {bench.NEMOTRON: json.dumps([PAIN, FEVER]), bench.GEMMA: json.dumps([PAIN]),
               bench.MUSE: json.dumps([PAIN]), "judge": judge}
    code, summary = run(tmp_path, monkeypatch, answers)
    assert code == 0 and summary["accepted"] == 1
    assert summary["table"]["A"]["precision"] == 0.5 and summary["table"]["B"]["precision"] == 0.5


def test_reading_stops_at_its_share_and_keeps_configurations_comparable(tmp_path, monkeypatch):
    answers = {bench.NEMOTRON: json.dumps([PAIN]), bench.GEMMA: json.dumps([PAIN]), bench.MUSE: json.dumps([PAIN]),
               "judge": lambda p: '{"states": "yes"}'}
    code, summary = run(tmp_path, monkeypatch, answers, budget=0.03)   # reading may spend 0.018: not one whole text
    assert code == 0
    assert all(v["answers"] == 0 for v in summary["table"].values())


@pytest.mark.parametrize("config", ["A", "B", "C"])
def test_every_configuration_has_two_readers(config):
    assert len(bench.CONFIGS[config]) == 2
