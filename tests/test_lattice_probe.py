"""Offline test of the lattice-probe report on three hand-made rows."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lattice_probe as lp  # noqa: E402

from melampo.memory.chunk_lattice import BlockMemory, ChunkLattice  # noqa: E402

MEMORY = BlockMemory.from_json(
    {
        "heads": {
            "weight": ["property", 1.0, 0],
            "cheese": ["food", 1.0, 0],
            "removal": ["procedure", 0.9, 40],
        }
    }
)


def row(mention, sentence, rule, labels, corpus="medmentions"):
    return {
        "corpus": corpus,
        "mention": mention,
        "sentence": sentence,
        "status": "accepted",
        "by_project_rule": rule,
        "labels": labels,
    }


def test_report_counts_readings_roles_and_conventions():
    rows = [
        row("heart", "Samples from heart of Maroilles cheese.", "error", ["C1|T073"]),
        row("liver", "The liver is large.", "agrees", ["C2|T023"]),
        row(
            "brain",
            "Brain weight was lower.",
            "convention:imaging_measure_of_the_structure",
            ["C3|T081"],
        ),
        row("gallbladder", "Removal of the gallbladder.", "agrees", ["C4|T061"]),
    ]
    report = lp.summarise(rows, ChunkLattice(MEMORY))
    j = report["judged"]["medmentions"]
    assert j["errors"] == 1 and j["errors_by_reading"]["not_a_site"] == 1
    assert j["right_by_reading"] == {
        "plain": 1,
        "role": 1,
        "not_a_site": 0,
        "underspecified": 0,
    }
    assert report["conventions"]["convention:imaging_measure_of_the_structure"] == {
        "inherent_location": 1
    }
    assert report["roles"]["procedure_site"]["same_role"] == 1
    assert report["roles"]["structure"]["same_role"] == 1
    assert "Experiment lattice-probe" in lp.markdown(report, [])
