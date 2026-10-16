"""Image findings: measures from a mask, the checks against the organ and the text, the graph, the follow-up
(16 October 2026). Synthetic volumes only: the segmenters are given masks."""

import math

import numpy as np
import pytest

from melampo.imaging import findings as fd
from melampo.imaging.lesion_measures import lps_to_ras_affine, measure_region
from melampo.imaging.longitudinal import link_exams, load_follows, relative_position
from melampo.imaging.segmenters import MaskSegmenter

AFFINE = np.array([[0.8, 0, 0, -100.0], [0, 0.8, 0, -120.0], [0, 0, 2.5, 50.0], [0, 0, 0, 1.0]])  # col,row,slice
SHAPE = (40, 128, 128)


def ball(center_mm, radius_mm, affine=AFFINE, shape=SHAPE):
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    pts = np.stack([x, y, z, np.ones_like(x)], -1) @ affine.T
    return ((pts[..., 0] - center_mm[0]) ** 2 + (pts[..., 1] - center_mm[1]) ** 2
            + (pts[..., 2] - center_mm[2]) ** 2) <= radius_mm ** 2


LIVER = ball((-50, -70, 100), 40)


def test_measures_of_a_sphere_on_an_anisotropic_grid():
    m = measure_region(ball((-50, -70, 100), 10), AFFINE)
    assert abs(m.volume_mm3 - 4 / 3 * math.pi * 1000) / (4 / 3 * math.pi * 1000) < 0.06
    assert abs(m.longest_axial_diameter_mm - 20) < 2.5 and abs(m.surface_mm2 - 4 * math.pi * 100) / 1257 < 0.12
    assert 0.85 < m.sphericity <= 1.05 and abs(m.principal_axes_mm[0] - 20) < 2.5
    assert np.allclose(m.centroid_mm, (-50, -70, 100), atol=1.0)


def test_ras_affine_flips_x_and_y():
    assert np.allclose(lps_to_ras_affine(AFFINE)[:2, 3], [100.0, 120.0])


def text(size="1,8 cm", structure="liver", start=10):
    clause = f"Lesione ipodensa di {size} nel fegato."
    return fd.TextSide("rep1", clause, "fegato", start, start + 6, structure, (), {"character": ["density:low"]},
                       fd.sizes_in_text(clause))


def seg(mask):
    return MaskSegmenter({"p": mask}).segment(None, AFFINE, ["p"])[0]


def test_sizes_written_in_the_text():
    got = fd.sizes_in_text("nodulo di 8 mm e formazione di 2,1 x 1,5 cm")
    assert [s["largest_mm"] for s in got] == [8.0, 21.0] and got[1]["mm"] == [21.0, 15.0]


def test_a_lesion_inside_the_named_organ_agrees_and_its_size_matches_the_text():
    image = fd.image_evidence(seg(ball((-50, -70, 100), 9)), AFFINE, organ_mask=LIVER,
                              text_size_mm=text().largest_size_mm)
    assert image.status == fd.AGREE and image.size_check == "consistent" and image.in_organ_share > 0.99


def test_outside_the_organ_disagrees_and_the_image_part_is_set_aside():
    image = fd.image_evidence(seg(ball((-10, -30, 100), 6)), AFFINE, organ_mask=LIVER, text_size_mm=18.0)
    assert image.status == fd.DISAGREE
    f = fd.build_finding("pt1", "st1", "2026-01-10", "CT", text(), image)
    row = f.as_row()
    assert "volume_mm3" not in row and "shape" not in f.vectors and row["image_status"] == "disagree"


@pytest.mark.parametrize("mask,status", [
    (np.zeros(SHAPE, bool), fd.NOT_FOUND),
    (ball((-60, -70, 100), 6) | ball((-30, -70, 100), 6), fd.AMBIGUOUS),
])
def test_nothing_found_or_two_regions(mask, status):
    assert fd.image_evidence(seg(mask), AFFINE, organ_mask=LIVER).status == status


def test_no_image_is_explicit_never_zeros():
    f = fd.build_finding("pt1", "st1", "2026-01-10", "CT", text(), fd.image_evidence(None, AFFINE))
    assert f.image.status == fd.NO_IMAGE and "volume_mm3" not in f.as_row() and f.vectors == {}


def _exam(study, date, radius, center=(-50, -70, 100), frame="1.2.3"):
    image = fd.image_evidence(seg(ball(center, radius)), AFFINE, organ_mask=LIVER, text_size_mm=2 * radius,
                              frame_of_reference=frame)
    t = text(size=f"{2 * radius} mm")
    t.report_id = study
    return fd.build_finding("pt1", study, date, "CT", t, image)


def test_the_same_lesion_is_followed_and_its_growth_measured():
    a, b = _exam("st1", "2026-01-10", 8), _exam("st2", "2026-04-10", 10)
    follows = link_exams([a], [b], comparable_acquisitions=True)
    assert len(follows) == 1 and follows[0].earlier == a.id and follows[0].later == b.id
    f = follows[0]
    assert f.days == 90 and 20 < f.diameter_change_pct < 30 and 1.7 < f.volume_ratio < 2.3
    assert 70 < f.doubling_time_days < 130 and not f.ambiguous


def test_a_different_structure_or_a_negated_finding_is_never_followed():
    a, b = _exam("st1", "2026-01-10", 8), _exam("st2", "2026-04-10", 8)
    b.text.structure = "spleen"
    assert link_exams([a], [b]) == []
    b.text.structure = "liver"
    b.text.qualifiers = {"polarity": "negated"}
    assert link_exams([a], [b]) == []


def test_two_equally_good_candidates_are_flagged():
    a1 = _exam("st1", "2026-01-10", 6, center=(-60, -70, 100), frame="")
    a2 = _exam("st1b", "2026-01-10", 6, center=(-40, -70, 100), frame="")
    b = _exam("st2", "2026-04-10", 6, center=(-50, -70, 100), frame="")
    follows = link_exams([a1, a2], [b])
    assert len(follows) == 1 and follows[0].ambiguous


def test_position_inside_the_organ():
    assert relative_position((5, 5, 5), ((0, 0, 0), (10, 20, 10))) == (0.5, 0.25, 0.5)


def test_findings_and_follows_in_falkordb(tmp_path):
    pytest.importorskip("redislite")
    from redislite.falkordb_client import FalkorDB

    graph = FalkorDB(str(tmp_path / "g.db")).select_graph("findings_test")
    a, b = _exam("st1", "2026-01-10", 8), _exam("st2", "2026-04-10", 10)
    assert fd.load_findings(graph, [a, b]) == 2
    assert fd.load_findings(graph, [a, b]) == 2  # idempotent
    assert load_follows(graph, link_exams([a], [b], comparable_acquisitions=True)) == 1
    rows = fd.findings_of(graph, "pt1")
    assert [r["study"] for r in rows] == ["st1", "st2"] and rows[0]["image_status"] == "agree"
    n = graph.query("MATCH (:Finding)-[r:FOLLOWS]->(:Finding) RETURN count(r), r.days").result_set[0]
    assert n == [1, 90]


def test_the_interpreter_makes_findings_only_where_the_clause_describes_something():
    from melampo.imaging.interpreter import interpret_exam

    report = ("TC addome con mdc. Fegato di dimensioni nei limiti. Al VII segmento epatico lesione ipodensa di 18 mm. "
              "Milza nei limiti. Non lesioni focali renali.")
    def site(word, cid, roles=()):
        at = report.index(word)
        return {"mention": word, "start": at, "end": at + len(word), "cid": cid, "roles": list(roles)}
    sites = [site("addome", "abdomen", ["procedure_site"]), site("Fegato", "liver"), site("epatico", "liver"),
             site("Milza", "spleen"), site("renali", "kidney")]
    clause = "Al VII segmento epatico lesione ipodensa di 18 mm"
    lesion = MaskSegmenter({clause: ball((-50, -70, 100), 9)})
    organ = MaskSegmenter({"liver": LIVER})
    got = interpret_exam("pt1", "st1", "2026-01-10", "CT", "rep1", report, sites, volume=np.zeros(SHAPE),
                         affine=AFFINE, frame_of_reference="1.2.3", lesion_segmenter=lesion, organ_segmenter=organ)
    assert [f.text.mention for f in got] == ["epatico"]
    f = got[0]
    assert f.text.clause == clause and f.text.largest_size_mm == 18.0
    assert f.image.status == fd.AGREE and f.image.size_check == "consistent" and "shape" in f.vectors


def test_without_the_segmenter_installed_the_finding_stays_on_the_text():
    from melampo.imaging.interpreter import interpret_exam
    from melampo.imaging.segmenters import VoxTellSegmenter

    report = "Lesione ipodensa di 18 mm nel fegato."
    at = report.index("fegato")
    got = interpret_exam("pt1", "st1", "2026-01-10", "CT", "rep1", report,
                         [{"mention": "fegato", "start": at, "end": at + 6, "cid": "liver"}],
                         volume=np.zeros(SHAPE), affine=AFFINE, lesion_segmenter=VoxTellSegmenter())
    assert got[0].image.status == fd.NO_IMAGE and got[0].image.notes[0].startswith("segmenter_unavailable")


def test_the_parts_of_a_region_are_counted_without_scipy_too():
    two = ball((-60, -70, 100), 6) | ball((-30, -70, 100), 6)
    sizes = fd._components_numpy(two)
    assert len(sizes) == 2 and sizes == fd._components(two)


# --- 16 October 2026, after the first Kaggle probe (lung_006: a good overlap measured 137 mm against 27 mm) ---

from melampo.imaging import threshold_check as tc  # noqa: E402

LUNG = ball((-50, -70, 100), 45)


def ct_with(lesion_hu=40.0, extra=None):
    """A synthetic chest CT: aerated lung (-850 HU) around a lesion of radius 9 mm."""
    img = np.full(SHAPE, -850.0)
    img[ball((-50, -70, 100), 9)] = lesion_hu
    if extra is not None:
        img[extra] = 30.0
    return img


def test_a_detached_speck_does_not_enter_the_size_but_is_reported():
    speck = np.zeros(SHAPE, bool)
    speck[20, 120, 120] = True                               # far from the lesion, same slice
    image = fd.image_evidence(seg(ball((-50, -70, 100), 9) | speck), AFFINE, organ_mask=LIVER, text_size_mm=18.0)
    assert abs(image.measures.longest_axial_diameter_mm - 18) < 2.5 and image.whole_axial_mm > 60
    assert len(image.fragments) == 1 and image.fragments[0]["distance_mm"] > 40
    assert not image.needs_review            # a speck of one voxel is noted, not a second lesion


def test_a_detached_part_of_some_size_goes_to_a_person():
    image = fd.image_evidence(seg(ball((-50, -70, 100), 9) | ball((-50, -40, 100), 5)), AFFINE, organ_mask=LIVER,
                              text_size_mm=18.0)
    assert image.status == fd.AGREE and fd.DETACHED_PARTS in image.review


def test_a_size_different_from_the_report_goes_to_a_person_and_is_not_used_for_growth():
    image = fd.image_evidence(seg(ball((-50, -70, 100), 9)), AFFINE, organ_mask=LIVER, text_size_mm=30.0)
    assert image.status == fd.AGREE and image.size_check == "inconsistent" and fd.SIZE_DIFFERS in image.review
    f = fd.build_finding("pt1", "st1", "2026-01-10", "CT", text("3 cm"), image)
    assert f.as_row()["needs_review"] and "shape" not in f.vectors


def test_a_measure_nothing_confirms_goes_to_a_person():
    image = fd.image_evidence(seg(ball((-50, -70, 100), 9)), AFFINE, organ_mask=LIVER)   # no size written, no CT
    assert image.review == (fd.UNCONFIRMED,)


def test_not_found_and_outside_say_why():
    assert fd.image_evidence(seg(np.zeros(SHAPE, bool)), AFFINE).review == (fd.NOT_IN_IMAGE,)
    out = fd.image_evidence(seg(ball((-10, -30, 100), 6)), AFFINE, organ_mask=LIVER, text_size_mm=12.0)
    assert fd.OUTSIDE_STRUCTURE in out.review


def test_the_threshold_reading_confirms_a_solid_nodule_with_a_vessel_attached():
    z, y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1], 0:SHAPE[2]]
    vessel = (np.abs(y - 50) <= 0) & (np.abs(z - 20) <= 0) & (x > 20) & (x < 75)     # one voxel thick
    image = ct_with(extra=vessel)
    first = ball((-50, -70, 100), 8.5)                                                  # VoxTell a little short
    r = tc.second_reading(first, image, AFFINE, "CT", "lung_upper_lobe_right")
    assert r.status == tc.APPLIED and r.agree and r.dice > 0.7 and abs(r.second_axial_mm - 18) < 2.5
    # 3 mm short is beyond the agreement: the readings are reported apart, not averaged
    short = tc.second_reading(ball((-50, -70, 100), 7.5), image, AFFINE, "CT", "lung_upper_lobe_right")
    assert short.agree is False and short.second_axial_mm > short.first_axial_mm + 2
    ev = fd.image_evidence(seg(first), AFFINE, image, organ_mask=LUNG, modality="CT",
                           structure="lung_upper_lobe_right")
    assert ev.second["agree"] and not ev.needs_review       # no size written, but two readings agree


def test_a_ground_glass_lesion_is_not_confirmed_by_a_solid_threshold():
    image = ct_with(lesion_hu=-600.0)
    ev = fd.image_evidence(seg(ball((-50, -70, 100), 9)), AFFINE, image, organ_mask=LUNG, modality="CT",
                           structure="lung_lower_lobe_left")
    assert ev.second["agree"] is False and fd.SECOND_DISAGREES in ev.review


def test_a_lesion_on_the_chest_wall_leaks_and_goes_to_a_person():
    wall = np.zeros(SHAPE, bool)
    wall[:, :, 73:] = True                                     # dense wall touching the lesion's side
    image = ct_with(extra=wall)
    r = tc.second_reading(ball((-50, -70, 100), 9), image, AFFINE, "CT", "lung")
    assert r.agree is False and "reaches_the_search_zone_edge" in r.notes


def test_the_threshold_reading_is_not_applied_outside_lung_ct():
    image = ct_with()
    assert tc.second_reading(ball((-50, -70, 100), 9), image, AFFINE, "CT", "liver").status == tc.NOT_APPLICABLE
    assert tc.second_reading(ball((-50, -70, 100), 9), image, AFFINE, "MR", "lung").status == tc.NOT_APPLICABLE


def test_parts_are_labelled_largest_first_with_and_without_scipy():
    from melampo.imaging.lesion_measures import _label_numpy, label_parts

    m = ball((-50, -70, 100), 5) | ball((-50, -40, 100), 9)
    labels, sizes = label_parts(m)
    assert sizes[0] > sizes[1] and (labels == 1).sum() == sizes[0]
    assert _label_numpy(m)[1] == 2
