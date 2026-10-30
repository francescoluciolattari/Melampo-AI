"""The size check's tolerance (30 October 2026): fixed, or the limits of agreement between two radiologists."""

import sys
from pathlib import Path

import pytest

from melampo.imaging.size_tolerance import FIXED, READERS, consistent, tolerance_mm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reader_agreement as ra  # noqa: E402


def test_fixed_is_the_old_rule_and_readers_grows_with_the_band():
    assert tolerance_mm(8, FIXED) == 2.0 and tolerance_mm(40, FIXED) == pytest.approx(4.0)
    assert tolerance_mm(8, READERS) == pytest.approx(3.5)
    assert tolerance_mm(15, READERS) == pytest.approx(7.3)
    assert tolerance_mm(25, READERS) == pytest.approx(12.7)
    assert consistent(None, 10) is None and consistent(10, None) is None
    assert consistent(12.9, 9, READERS) is False and consistent(12.9, 9, FIXED) is False
    assert consistent(14.0, 16, READERS) and not consistent(13.0, 16, FIXED)
    with pytest.raises(ValueError):
        tolerance_mm(10, "other")


def _row(i, readers, texture="solido", seg=None, sweep=None):
    return {"id": f"n{i}", "stratum": f"{texture} x",
            "readers": {"readers": [{"axial_long_mm": d} for d in readers]},
            "prompt": {"found": {"found": seg is not None}, "segmenter": {"axial_long_mm": seg},
                       "sweep": {"0.1": {"axial_long_mm": sweep}, "0.3": {"axial_long_mm": sweep}}}}


def test_limits_of_agreement_and_who_goes_to_a_person():
    rows = [_row(0, [5.0, 5.0, 5.0, 5.0], seg=5.0, sweep=5.2),
            _row(1, [6.0, 6.0, 6.0, 6.0], seg=9.0, sweep=6.0),
            _row(2, [25.0, 27.0, 29.0, 31.0], "vetro_smerigliato", seg=28.0, sweep=40.0),
            _row(3, [22.0, 22.0, 22.0, 22.0], seg=None),
            {"id": "bad", "error": "x"}]
    res = ra.report(rows)
    assert res["nodules"] == 4 and res["agreement"]["<10"]["limits_of_agreement_mm"] == 0
    assert res["agreement"][">=20"]["limits_of_agreement_mm"] > 0
    frozen = res["to_a_person"]["frozen_rule"][READERS]
    assert frozen["<10"] == {"to_a_person": 0, "of": 8, "share": 0.0}          # 5.2 vs 5, 6.0 vs 6
    assert frozen[">=20"]["of"] == 4 and frozen[">=20"]["to_a_person"] == 0    # ground glass: VoxTell's 28 mm
    seg = res["to_a_person"]["segmenter"][FIXED]
    assert seg["<10"]["to_a_person"] == 4                                       # 9 mm against a 6 mm report
    assert res["not_found"][">=20"] == 1
