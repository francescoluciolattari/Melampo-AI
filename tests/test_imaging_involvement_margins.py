"""The structures a lesion involves, its position in the one that holds it, and its margins as data (18 October
2026). Synthetic volumes only."""

import numpy as np
import pytest

from melampo.imaging import findings as fd
from melampo.imaging.density import density_profile
from melampo.imaging.involvement import CONTACT, CROSSES, INSIDE, StructureLabels, involvement
from melampo.imaging.margins import compare_margins_with_report, margin_profile
from melampo.imaging.segmenters import MaskSegmenter
from melampo.memory.site_qualifiers import read_site_qualifiers

AFF = np.diag([1.0, 1.0, 1.0, 1.0])      # 1 mm isotropic: (column, row, slice) -> mm
SHAPE = (72, 72, 72)
Z, Y, X = np.mgrid[0:SHAPE[0], 0:SHAPE[1], 0:SHAPE[2]].astype(float)


def ball(c, r):
    return (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= r ** 2


LOBE = ball((36, 36, 36), 30)
BELOW = (Z < 36) & ~LOBE & (np.abs(X - 36) < 30) & (np.abs(Y - 36) < 30)   # a neighbour outside the lobe


def test_a_lesion_deep_in_a_lobe_is_inside_it_and_central():
    got, pos = involvement(ball((36, 36, 36), 5), {"lobe": LOBE, "below": BELOW}, AFF)
    assert [i.structure for i in got] == ["lobe"] and got[0].relation == INSIDE and got[0].lesion_share > 0.99
    assert pos.host == "lobe" and pos.zone == "central" and not pos.touches_surface


def test_a_lesion_on_the_surface_crosses_into_the_neighbour():
    lesion = ball((36, 36, 8), 5)                  # at the bottom of the lobe, across its border
    got, pos = involvement(lesion, {"lobe": LOBE, "below": BELOW}, AFF)
    rel = {i.structure: i.relation for i in got}
    assert rel["lobe"] in (INSIDE, CROSSES) and rel["below"] in (CROSSES, CONTACT)
    assert pos.host == "lobe" and pos.touches_surface and pos.zone == "peripheral"


def test_a_tumour_that_replaced_the_tissue_is_held_by_the_structure_around_it():
    lesion = ball((36, 36, 36), 8)
    hollow = LOBE & ~lesion                         # the organ mask is the tissue: the tumour is a hole in it
    got, pos = involvement(lesion, {"lobe": hollow}, AFF)
    assert got[0].relation == CONTACT and got[0].surround_share > 0.9
    assert pos.host == "lobe" and pos.host_chosen_by == "surround_share" and pos.zone == "central"


def test_a_label_map_gives_the_same_reading_as_separate_masks():
    labels = np.zeros(SHAPE, np.int16)
    labels[LOBE] = 3
    labels[BELOW] = 7
    lesion = ball((36, 36, 8), 5)
    a, pa = involvement(lesion, {"lobe": LOBE, "below": BELOW}, AFF)
    b, pb = involvement(lesion, StructureLabels(labels, {3: "lobe", 7: "below", 9: "absent"}), AFF)
    assert [i.as_dict() for i in a] == [i.as_dict() for i in b] and pa == pb


def _spiked(r=9, n=8, length=7):
    m = ball((36, 36, 36), r)
    rng = np.random.default_rng(1)
    for _ in range(n):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for t in np.arange(r - 1, r + length, 0.5):
            p = np.round(np.array([36, 36, 36]) + t * d).astype(int)
            m[p[2], p[1], p[0]] = True
            m[p[2], p[1], p[0] + 1] = True
    return m


def test_contours_smooth_spiculated_lobulated():
    smooth = margin_profile(ball((36, 36, 36), 9), AFF)
    assert smooth.contour == "smooth" and smooth.spicules == 0
    spiked = margin_profile(_spiked(), AFF)
    assert spiked.contour == "spiculated" and spiked.spicules >= 3
    lobes = ball((36, 36, 36), 7)                  # a centre and four lobes around it, notches between them
    for dx, dy in ((-7, -7), (7, -7), (-7, 7), (7, 7)):
        lobes |= ball((36 + dx, 36 + dy, 36), 7)
    assert margin_profile(lobes, AFF).contour == "lobulated"


def test_a_small_lesion_gets_numbers_but_no_contour():
    m = margin_profile(ball((36, 36, 36), 3), AFF)
    assert m.contour == "" and m.spicule_share is not None and m.notes[0].startswith("contour_not_judged")


def _image(lesion, ramp_mm):
    """Lung (-850 HU) and a solid lesion (+30 HU); the border ramps over ``ramp_mm`` mm (0: a step)."""
    from scipy.ndimage import distance_transform_edt
    out = distance_transform_edt(~lesion)
    w = np.clip(1 - out / ramp_mm, 0, 1) if ramp_mm else np.zeros(SHAPE)
    w[lesion] = 1
    return -850 + 880 * w


def test_sharp_and_blurred_borders():
    pytest.importorskip("scipy")
    lesion = ball((36, 36, 36), 10)
    sharp = margin_profile(lesion, AFF, density_profile(_image(lesion, 0), lesion, AFF, "CT"))
    blurred = margin_profile(lesion, AFF, density_profile(_image(lesion, 8), lesion, AFF, "CT"))
    assert sharp.sharpness == "sharp" and blurred.sharpness == "blurred"
    assert sharp.transition_mm < blurred.transition_mm


@pytest.mark.parametrize("sentence,want", [
    ("Nodulo spiculato di 2 cm nel lobo superiore destro.", ["margin:spiculated"]),
    ("Formazione a margini netti nel lobo inferiore.", ["margin:sharp"]),
    ("Lesione a contorni irregolari nel fegato.", ["margin:irregular"]),
    ("Lesione di forma irregolare nel fegato.", []),
    ("Nodulo non spiculato nel lobo superiore.", ["margin:not_spiculated"]),
    ("Nodulo subpleurico nel lobo superiore.", ["position:surface"]),
    ("Ill-defined opacity in the right upper lobe.", ["margin:blurred", "density:high"]),
])
def test_report_words_on_margins_and_position(sentence, want):
    word = "lobo" if "lobo" in sentence else ("fegato" if "fegato" in sentence else "lobe")
    at = sentence.index(word)
    q = read_site_qualifiers(sentence, at, at + len(word), "en" if "lobe" == word else "it").as_dict()
    got = [c for c in q["character"] if c.split(":")[0] in ("margin", "position", "density")]
    assert sorted(got) == sorted(want)


def test_a_spiculated_report_on_a_smooth_lesion_goes_to_a_person():
    smooth = margin_profile(ball((36, 36, 36), 9), AFF)
    assert compare_margins_with_report(["margin:spiculated"], smooth)
    assert not compare_margins_with_report(["margin:spiculated"], margin_profile(_spiked(), AFF))
    assert not compare_margins_with_report(["margin:lobulated"], smooth)   # no clear opposite: not compared


def test_the_finding_carries_involvement_position_and_margins_into_the_graph(tmp_path):
    lesion = ball((36, 36, 36), 9)
    image = -850 + 880 * lesion.astype(float)
    seg = MaskSegmenter({"p": lesion}).segment(None, AFF, ["p"])[0]
    side = fd.image_evidence(seg, AFF, image, None, 18.0, modality="CT", structure="lobe",
                             structures={"lobe": LOBE, "below": BELOW})
    assert side.status == fd.AGREE and side.position["host"] == "lobe" and side.margins.contour == "smooth"
    t = fd.TextSide("rep1", "Nodulo di 18 mm nel lobo.", "lobo", 0, 4, "lobe", (), {}, fd.sizes_in_text("18 mm"))
    f = fd.build_finding("pt1", "st1", "2026-01-10", "CT", t, side)
    row = f.as_row()
    assert row["host_structure"] == "lobe" and row["involved_structures"] == ["lobe"]
    assert row["margin_contour"] == "smooth" and row["position_zone"] == "central"
    pytest.importorskip("redislite")
    from redislite.falkordb_client import FalkorDB

    graph = FalkorDB(str(tmp_path / "g.db")).select_graph("involves_test")
    fd.load_findings(graph, [f])
    fd.load_findings(graph, [f])
    got = graph.query("MATCH (:Finding)-[r:INVOLVES]->(s:Structure) RETURN s.cid, r.relation, count(r)").result_set
    assert got == [["lobe", "inside", 1]]
