"""The numpy parts of the F1 experiment (scripts/head_probe.py). The models are not loaded here."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import head_probe as hp


def test_auc_is_one_when_the_score_separates_and_half_when_it_does_not():
    labels = np.array([1, 1, 0, 0, 0])
    assert hp.auc(np.array([0.9, 0.8, 0.2, 0.1, 0.3]), labels) == 1.0
    assert hp.auc(np.array([0.5, 0.5, 0.5, 0.5, 0.5]), labels) == 0.5
    assert hp.auc(np.array([0.1, 0.2, 0.9, 0.8, 0.7]), labels) == 0.0
    assert hp.auc(np.array([0.1, 0.2]), np.array([1, 1])) is None


def test_operating_points_report_errors_caught_for_right_links_lost():
    scores = np.array([0.9, 0.8, 0.6, 0.7, 0.5, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    points = {p["allowed_lost"]: p["best"] for p in hp.operating_points(scores, labels)}
    assert points[0][1:] == (2, 0)  # two errors, nothing lost
    assert points[5][1:] == (3, 1)  # all three errors, one right link lost


def test_tokens_inside_a_character_span_and_pooling():
    offsets = np.array([[0, 0], [0, 3], [4, 9], [10, 13], [0, 0]])
    assert hp.token_ids(offsets, 4, 13) == [2, 3]
    hidden = np.arange(3 * 5 * 2, dtype=float).reshape(3, 5, 2)
    pooled = hp.pool(hidden, [2, 3], (1, 2))
    assert pooled.shape == (2,)
    assert np.isclose(hp.cosine(pooled, pooled), 1.0)


def test_the_word_a_mention_attends_to_most_is_found_across_wordpieces():
    sentence = "liver fatty acid binding protein"
    # tokens: [CLS] liver fat ##ty acid binding protein [SEP]
    offsets = np.array(
        [[0, 0], [0, 5], [6, 9], [9, 11], [12, 16], [17, 24], [25, 32], [0, 0]]
    )
    attn = np.zeros((2, 1, 8, 8))
    attn[:, :, 1, 6] = 0.7  # the mention attends to "protein"
    attn[:, :, 1, 2] = 0.1
    attn[:, :, 1, 3] = 0.1
    attn[:, :, 1, 0] = 0.1  # to [CLS], which is ignored
    word, share, right = hp.attention_head(offsets, attn, sentence, [1], (0, 1))
    assert word == "protein"
    assert abs(share - 0.7 / 0.9) < 1e-9
    assert abs(right - 1.0) < 1e-9


def test_the_summary_counts_errors_and_lists_the_right_links_a_signal_would_stop():
    cases = [
        {"source": "craft", "mention": "liver", "sentence": "a", "label": 1, "phrase_nonsite": 1.0, "phrase_head": "protein"},
        {"source": "craft", "mention": "brain", "sentence": "b", "label": 0, "phrase_nonsite": 0.0, "phrase_head": ""},
        {"source": "control_it", "mention": "fegato", "sentence": "c", "label": 0, "phrase_nonsite": 1.0, "phrase_head": "test"},
    ]
    report = hp.summarise(cases, frozenset(), None)
    assert report["n"] == 3 and report["errors"] == 1
    assert report["features"]["phrase_nonsite"]["auc"] == 0.75
    assert [e["mention"] for e in report["examples"]["right_links_with_a_nonsite_head"]] == ["fegato"]
    assert "phrase_nonsite" in hp.markdown(report)
