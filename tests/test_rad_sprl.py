"""Rad-SpRL reader: landmarks, discontinuous spans, gold rows with the labelled class."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rad_sprl as rs  # noqa: E402

TEXT = "\nOpacity in the right upper lobe. Fracture of the right and left lung.\n"


def _xml(tmp_path):
    words = [(m.start(), m.end()) for m in __import__("re").finditer(r"\S+", TEXT)]
    # strip the final period off tokens, as the corpus tokenises it apart
    toks = []
    for s, e in words:
        if TEXT[e - 1] == ".":
            toks += [(s, e - 1), (e - 1, e)]
        else:
            toks.append((s, e))
    tok = "".join(f'<Token cs="{s}" cl="{e - s}" />' for s, e in toks)
    text_tokens = [TEXT[s:e] for s, e in toks]
    i = text_tokens.index
    xml = f"""<Corpus numDocs="1"><Document version="1.0" id="00007"><Text><![CDATA[{TEXT}]]></Text><Annotations>{tok}
<Sentence ts="0" tl="7" /><Sentence ts="7" tl="8" />
<RadSpRLRelation ts="{i('in')}" tl="1"><Trajector ts="0" tl="1" type="Disorder" />
<Landmark ts="{i('right')}" tl="3" type="Anatomy" /></RadSpRLRelation>
<RadSpRLRelation ts="{i('of')}" tl="1"><Trajector ts="{i('Fracture')}" tl="1" type="Disorder" />
<Landmark ts="{text_tokens.index('right', 8)},{text_tokens.index('lung')}" tl="1,1" type="Anatomy" /></RadSpRLRelation>
</Annotations></Document></Corpus>"""
    p = tmp_path / "r.xml"
    p.write_text(xml, "utf-8")
    return p


def test_landmarks_sentences_and_discontinuous_spans(tmp_path):
    (rid, text, lms), = list(rs.read(_xml(tmp_path)))
    assert rid == "radsprl-00007" and text == TEXT
    assert lms[0]["mention"] == "right upper lobe" and lms[0]["indicator"] == "in"
    assert lms[0]["sentence"] == "Opacity in the right upper lobe."
    assert lms[1]["mention"] == "right and left lung" and lms[1]["pieces"] == "right lung"


def test_gold_rows_carry_the_labelled_class_and_the_structure_role(tmp_path):
    classes = tmp_path / "c.txt"
    classes.write_text("# comment\nright upper lobe|lung_upper_lobe_right|equal|\n", "utf-8")
    out, gold = tmp_path / "l.jsonl", tmp_path / "g.jsonl"
    assert rs.main(["convert", "--xml", str(_xml(tmp_path)), "--out", str(out), "--gold", str(gold),
                    "--classes", str(classes)]) == 0
    rows = [json.loads(x) for x in gold.read_text().splitlines()]
    assert rows[0]["structure"] == "lung_upper_lobe_right" and rows[0]["role"] == "structure"
    assert rows[1]["structure"] == rs.UNLABELLED
