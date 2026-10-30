"""How far the measured size may sit from the report's size before a person looks (30 October 2026).

The check compares two measures of the same lesion: ours and the one the radiologist wrote. Two radiologists
measuring the same nodule already differ, so the yardstick is how much they differ, not a number chosen by hand.

* ``FIXED`` -- max(2 mm, 10%), the rule used since the sixth probe. On the LIDC set "misura" it would send to a
  person 28% of the radiologists' own measures (one reader against the mean of the other three, the report rounded
  to the millimetre).
* ``READERS`` -- the 95% limits of agreement between two radiologists (Bland-Altman: 1.96 x sqrt(2) x the
  within-nodule standard deviation of the readers' longest axial diameter), by size band, plus half a millimetre for
  the report's rounding. Estimated on the 40 nodules of the LIDC set "misura" (four readers each), by
  ``scripts/reader_agreement.py``; numbers in ``data/lidc/reader_agreement_misura_2026-10-30.json``. To be checked,
  unchanged, on the set "verifica".

The tolerance never changes the measure (always the segmenter's region, or the reading fixed before the run): it only
decides whether a person looks. It comes from the readers among themselves, never from our errors.

The limits are wide for large nodules (12 mm over 20 mm): two radiologists' longest diameters of an irregular
nodule differ that much (a spicule or a vessel taken in one slice and not in another). A size check therefore only
catches gross mistakes there; the lesion's identity rests on its place (the lobe the report names, the atlas) and on
two independent readings agreeing.
"""

from __future__ import annotations

FIXED, READERS = "fixed", "readers"
POLICIES = (FIXED, READERS)

FIXED_MM, FIXED_SHARE = 2.0, 0.10
# (upper bound of the band on the report's size in mm, limits of agreement between two readers in mm)
READER_LIMITS_MM = ((10.0, 3.0), (20.0, 6.8), (float("inf"), 12.2))
REPORT_ROUNDING_MM = 0.5


def tolerance_mm(report_size_mm: float, policy: str = FIXED) -> float:
    """The largest difference between our measure and the report's size that needs no person."""
    if policy == FIXED:
        return max(FIXED_MM, FIXED_SHARE * report_size_mm)
    if policy == READERS:
        for upper, limit in READER_LIMITS_MM:
            if report_size_mm < upper:
                return limit + REPORT_ROUNDING_MM
    raise ValueError(f"unknown size policy {policy!r}: one of {POLICIES}")


def consistent(measured_mm: float | None, report_size_mm: float | None, policy: str = FIXED) -> bool | None:
    """True when the measure is within the tolerance of the report's size; None when one of them is missing."""
    if measured_mm is None or not report_size_mm:
        return None
    return abs(measured_mm - report_size_mm) <= tolerance_mm(report_size_mm, policy)
