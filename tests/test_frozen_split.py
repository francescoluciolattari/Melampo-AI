"""The frozen test split: digest, guard, held-back rows, and the manifest in the repository."""

import json

import pytest

from melampo.evaluation import frozen_split as fs

# Pinned on purpose: changing the frozen list means editing this line too, in a commit someone reads.
FROZEN_SHA256 = "5e010646a711b49878700c5e93c24a83be5c3a23864fe872b1f1464d8cf9d33f"


def split():
    return fs.FrozenSplit.from_ids("medmentions", ["3", "1", "2", " "], "2026-10-10")


def test_the_digest_does_not_depend_on_order_and_detects_an_edit(tmp_path):
    assert fs.digest(["1", "2", "3"]) == fs.digest(["3", "2", "1", "2"])
    s = split()
    path = tmp_path / "f.json"
    s.save(path)
    assert fs.FrozenSplit.load(path).ids == frozenset({"1", "2", "3"})
    data = json.loads(path.read_text())
    data["ids"].append("4")
    path.write_text(json.dumps(data))
    with pytest.raises(fs.FrozenSplitError):
        fs.FrozenSplit.load(path)


def test_refuse_names_the_document_and_the_purpose():
    s = split()
    s.refuse(["9", "8"], "the trace memory")
    with pytest.raises(fs.FrozenSplitError, match="the trace memory"):
        s.refuse(["9", "2"], "the trace memory")
    s.refuse(["2"], "a craft step", corpus="craft")  # another corpus is not frozen
    assert s.open_docs(["1", "7"]) == ["7"]


def test_development_rows_hold_back_frozen_documents_unless_final():
    s = split()
    rows = [{"corpus": "medmentions", "doc": "1"}, {"corpus": "medmentions", "doc": "5"},
            {"corpus": "craft", "doc": "1"}]
    keep, held = fs.development_rows(rows, s)
    assert [r["doc"] for r in keep] == ["5", "1"] and held == 1
    assert fs.development_rows(rows, s, final=True) == (rows, 0)
    assert fs.development_rows(rows, None) == (rows, 0)


def test_the_manifest_in_the_repository_is_intact_and_pinned():
    s = fs.FrozenSplit.load()
    assert s.corpus == "medmentions" and len(s.ids) == 879
    assert s.sha256 == FROZEN_SHA256
    assert "not unseen" in s.seen_before_freeze["note"]
