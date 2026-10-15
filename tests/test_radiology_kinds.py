"""Radiology kinds from a RadLex-like CSV: branch kinds, exact correspondences, merge into the block memory."""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_radiology_kinds as br  # noqa: E402

from melampo.memory import chunk_lattice as cl  # noqa: E402

CSV = """Class ID,Preferred Label,Synonyms,Definitions,Obsolete,CUI,Semantic Types,Parents
R1,RadLex entity,,,false,,,
R2,procedure,,,false,,,R1
R3,imaging modality,,,false,,,R2
R4,computed tomography,CT|CAT scan,,false,,,R3
R5,radiograph,plain film,,false,,,R2
R6,anatomical entity,,,false,,,R1
R7,brain,,,false,,,R6
R8,imaging observation,,,false,,,R1
R9,opacity,,,false,,,R8
R10,clinical finding,,,false,,,R1
R11,view,,,false,,,R2
R12,view,,,false,,,R10
"""


def test_kinds_from_branches_and_exact_correspondence():
    classes = br.read_classes(gzip.compress(CSV.encode()))
    table = br.build(classes)
    assert table["heads"]["ct"] == ["procedure", 1.0, 0]
    assert table["heads"]["radiograph"][0] == "procedure"
    assert table["heads"]["opacity"][0] == "disease"
    assert table["names"]["computed tomography"] == "procedure"
    assert "view" in table["conflicts"] and "view" not in table["heads"]   # two kinds: no correspondence


def test_the_radiology_kinds_fill_ncit_gaps_only(tmp_path):
    memory = cl.BlockMemory.from_json({"heads": {"opacity": ["property", 1.0], "view": ["conceptual", 0.9]}, "names": {}})
    added = memory.add_radiology({"heads": {"ct": ["procedure", 1.0, 0], "opacity": ["disease", 1.0, 0],
                                            "view": ["procedure", 1.0, 0]},
                                  "names": {"computed tomography": "procedure"}})
    assert memory.head("ct") == ("procedure", 1.0)
    assert memory.head("view") == ("procedure", 1.0)          # NCIt "conceptual" is replaced
    assert memory.head("opacity") == ("property", 1.0)        # NCIt kind kept where both give one
    assert memory.names["computed tomography"] == "procedure" and added == {"heads": 2, "names": 1}
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"heads": {"ct": ["procedure", 1.0, 0]}, "names": {}}))
    loaded = cl.BlockMemory.load(tmp_path / "missing.json", radiology=path)
    assert loaded.head("ct") == ("procedure", 1.0)
