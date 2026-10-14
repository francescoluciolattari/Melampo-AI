"""Governor kinds from RadLex and UMLS: exact matches only, one kind or none (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_governor_kinds as bg  # noqa: E402


def fake_get(responses):
    def get(url, headers=None):
        for key, value in responses.items():
            if key in url:
                return value
        return {}
    return get


UMLS = {
    "string=pain": {"result": {"results": [{"ui": "C0030193"}]}},
    "CUI/C0030193": {"result": {"semanticTypes": [{"uri": "x/TUI/T184", "name": "Sign or Symptom"}]}},
    "string=view": {"result": {"results": [{"ui": "C1"}, {"ui": "C2"}]}},
    "CUI/C1": {"result": {"semanticTypes": [{"uri": "x/TUI/T060"}]}},
    "CUI/C2": {"result": {"semanticTypes": [{"uri": "x/TUI/T081"}]}},
    "string=appearance": {"result": {"results": [{"ui": "C3"}]}},
    "CUI/C3": {"result": {"semanticTypes": [{"uri": "x/TUI/T080"}]}},
}
RADLEX = {
    "q=radiograph": {"collection": [{"@id": "http://radlex.org/RID/RID1"}]},
    "RID1/ancestors": [{"prefLabel": "radiography"}, {"prefLabel": "imaging modality"}, {"prefLabel": "procedure"}],
    "q=pain": {"collection": [{"@id": "http://radlex.org/RID/RID2"}]},
    "RID2/ancestors": [{"prefLabel": "clinical finding"}],
}


def test_one_kind_per_word_or_none():
    get = fake_get({**UMLS, **RADLEX})
    table = bg.build(["pain", "view", "appearance", "radiograph"], "u", "b", bg.Cache(None), get=get)
    assert table["kinds"] == {"pain": "disease", "radiograph": "procedure"}
    assert "view" in table["conflicts"]          # two UMLS concepts of two kinds: not a correspondence
    assert "appearance" not in table["kinds"]    # a qualitative concept gives no kind


def test_sources_must_agree():
    get = fake_get({"string=pain": UMLS["string=pain"], "CUI/C0030193": UMLS["CUI/C0030193"],
                    "q=pain": {"collection": [{"@id": "http://radlex.org/RID/RID9"}]},
                    "RID9/ancestors": [{"prefLabel": "procedure"}]})
    table = bg.build(["pain"], "u", "b", bg.Cache(None), get=get)
    assert table["kinds"] == {} and table["conflicts"] == ["pain"]


def test_no_snomed_in_the_umls_search():
    assert "SNOMED" not in bg.UMLS_SABS
