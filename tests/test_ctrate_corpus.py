"""CT-RATE in PARROT's form (6 November 2026): the parts become sections, a report shared by reconstructions is
kept once, and the section reader reads the clinical information as the indication."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ctrate_corpus import columns, convert

from melampo.memory.report_sections import (
    FINDINGS,
    IMPRESSION,
    INDICATION,
    TECHNIQUE,
    section_at,
)

HEADER = ["VolumeName", "ClinicalInformation_EN", "Technique_EN", "Findings_EN", "Impressions_EN"]


def test_the_parts_become_sections_and_a_shared_report_is_kept_once():
    rows = [{"VolumeName": "valid_1_a_1.nii.gz", "ClinicalInformation_EN": "Cough and fever",
             "Technique_EN": "Non-contrast chest CT", "Findings_EN": "Heart size is normal. A nodule in the right "
             "upper lobe.", "Impressions_EN": "Pulmonary nodule"},
            {"VolumeName": "valid_1_a_2.nii.gz", "ClinicalInformation_EN": "Cough and fever",
             "Technique_EN": "Non-contrast chest CT", "Findings_EN": "Heart size is normal. A nodule in the right "
             "upper lobe.", "Impressions_EN": "Pulmonary nodule"}]
    assert columns(HEADER)["finding"] == "Findings_EN"
    out = convert(rows, HEADER)
    assert len(out) == 1 and out[0]["no"] == "ctrate-1" and out[0]["patient"] == "valid_1"
    text = out[0]["translation"]
    assert section_at(text, text.index("Cough")).kind == INDICATION
    assert section_at(text, text.index("Non-contrast")).kind == TECHNIQUE
    assert section_at(text, text.index("A nodule")).kind == FINDINGS
    assert section_at(text, text.index("Pulmonary nodule")).kind == IMPRESSION
