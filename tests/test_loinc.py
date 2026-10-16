"""Laboratory rows to LOINC: one code only when exactly one term fits (16 October 2026). A tiny index built here
with the shape of the LOINC release (codes shown are real LOINC codes, used under the LOINC licence)."""

import io
import json
import zipfile

from melampo.data.lab_results import extract_lab_results
from melampo.data.loinc import LoincIndex, LoincTerm, map_rows

TERMS = [
    LoincTerm("718-7", "Hemoglobin", "MCnc", "Pt", "Bld", "Qn", "", ("hemoglobin", "emoglobina", "hgb", "hb")),
    LoincTerm("20509-6", "Hemoglobin", "MCnc", "Pt", "Bld", "Qn", "Calculated", ("hemoglobin", "emoglobina")),
    LoincTerm("2160-0", "Creatinine", "MCnc", "Pt", "Ser/Plas", "Qn", "", ("creatinine", "creatinina")),
    LoincTerm("2161-8", "Creatinine", "MCnc", "Pt", "Urine", "Qn", "", ("creatinine", "creatinina")),
    LoincTerm("14682-9", "Creatinine", "SCnc", "Pt", "Ser/Plas", "Qn", "", ("creatinine", "creatinina")),
    LoincTerm("6690-2", "Leukocytes", "NCnc", "Pt", "Bld", "Qn", "Automated count", ("leukocytes", "leucociti", "wbc")),
    LoincTerm("26464-8", "Leukocytes", "NCnc", "Pt", "Bld", "Qn", "Manual count", ("leukocytes", "leucociti", "wbc")),
]
INDEX = LoincIndex(TERMS)

REPORT = """ESAMI EMATOCHIMICI
Emoglobina        11,8   g/dL      12,0 - 16,0   L
Creatinina        1,45   mg/dL     0,50 - 1,20   H
Leucociti         7,20   10^3/uL   4,00 - 10,00
Glicemia           95    mg/dL     70 - 110
"""


def test_mapped_only_when_one_term_fits():
    rows = {m["analyte"]: m for m in map_rows(extract_lab_results(REPORT).results, INDEX)}
    assert rows["Emoglobina"]["code"] == "718-7"           # the method-less term is the general one
    assert rows["Creatinina"]["code"] == "2160-0"          # mg/dL -> MCnc; serum by the section "ematochimici"
    assert rows["Leucociti"]["status"] == "ambiguous" and len(rows["Leucociti"]["candidates"]) == 2
    assert rows["Glicemia"]["status"] == "unmapped"


def test_the_unit_decides_the_property_and_the_section_the_specimen():
    assert INDEX.match("Creatinina", "umol/L", "SIERO").code == "14682-9"
    assert INDEX.match("Creatinina", "mg/dL", "URINE").code == "2161-8"
    assert INDEX.match("Creatinina", "furlongs").status == "unmapped"


def test_without_the_index_nothing_is_guessed():
    rows = map_rows(extract_lab_results(REPORT).results, None)
    assert {r["status"] for r in rows} == {"no_index"}


def test_the_index_is_built_from_the_release_zip(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_loinc_index as b

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("LoincTable/Loinc.csv",
                   "LOINC_NUM,COMPONENT,PROPERTY,TIME_ASPCT,SYSTEM,SCALE_TYP,METHOD_TYP,STATUS,CLASSTYPE,"
                   "LONG_COMMON_NAME,RELATEDNAMES2\n"
                   "718-7,Hemoglobin,MCnc,Pt,Bld,Qn,,ACTIVE,1,Hemoglobin [Mass/volume] in Blood,Hgb; HB\n"
                   "99999-9,Old thing,MCnc,Pt,Bld,Qn,,DEPRECATED,1,x,\n")
        z.writestr("AccessoryFiles/LinguisticVariants/itIT15LinguisticVariant.csv",
                   "LOINC_NUM,COMPONENT\n718-7,Emoglobina\n")
    path = tmp_path / "loinc.zip"
    path.write_bytes(buf.getvalue())
    index = b.build(path)
    assert [t["code"] for t in index["terms"]] == ["718-7"]
    assert {"hemoglobin", "emoglobina", "hgb", "hb"} <= set(index["terms"][0]["names"])
    (tmp_path / "index.json").write_text(json.dumps(index))
    assert LoincIndex.load(tmp_path / "index.json").match("Emoglobina", "g/dL").code == "718-7"
