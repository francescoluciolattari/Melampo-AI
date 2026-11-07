"""RadGraph annotations as items and labels (6 November 2026), on a record written here in RadGraph's schema."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from radgraph_items import items_of

RECORD = {
    "text": "There is a small effusion at the right lung base . The heart is normal . No pneumothorax .",
    "entities": {
        "1": {"tokens": "small", "label": "OBS-DP", "start_ix": 3, "end_ix": 3, "relations": [["modify", "2"]]},
        "2": {"tokens": "effusion", "label": "OBS-DP", "start_ix": 4, "end_ix": 4, "relations": [["located_at", "4"]]},
        "3": {"tokens": "right", "label": "ANAT-DP", "start_ix": 7, "end_ix": 7, "relations": [["modify", "4"]]},
        "4": {"tokens": "lung", "label": "ANAT-DP", "start_ix": 8, "end_ix": 8, "relations": []},
        "5": {"tokens": "base", "label": "ANAT-DP", "start_ix": 9, "end_ix": 9, "relations": [["modify", "4"]]},
        "6": {"tokens": "heart", "label": "ANAT-DP", "start_ix": 12, "end_ix": 12, "relations": []},
        "7": {"tokens": "normal", "label": "OBS-DP", "start_ix": 14, "end_ix": 14, "relations": [["located_at", "6"]]},
        "8": {"tokens": "pneumothorax", "label": "OBS-DA", "start_ix": 17, "end_ix": 17, "relations": []},
    },
}


def test_an_anatomy_head_takes_what_is_located_at_it_and_its_modifiers_are_not_items():
    got = {i["mention"]: kind for i, kind in items_of("r1", RECORD)}
    # "right" and "base" modify the lung; "normal" is an observation present at the heart (RadGraph says where
    # and whether, not whether it is abnormal)
    assert got == {"lung": "observed", "heart": "observed"}
    item = next(i for i, _ in items_of("r1", RECORD) if i["mention"] == "lung")
    assert item["clause"] == "There is a small effusion at the right lung base ."
    assert item["item_id"] == "radgraph-r1-4"


def test_the_two_labelers_of_the_test_set_are_read_apart():
    record = {"text": RECORD["text"], "labeler_1": {"entities": RECORD["entities"]},
              "labeler_2": {"entities": {"6": RECORD["entities"]["6"]}}}
    assert {i["mention"] for i, _ in items_of("r1", record, "labeler_2")} == {"heart"}
    assert {i["mention"]: k for i, k in items_of("r1", record, "labeler_2")} == {"heart": "nothing"}
