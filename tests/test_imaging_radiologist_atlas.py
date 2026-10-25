"""The radiologist's reading on the patient's atlas (25 October 2026): what the eleventh probe's readings ran
into -- a thick vessel in the lung, the chest wall, the mediastinal fat, a ring mixing tissues around a large
mass -- is left out because the atlas names it. Synthetic volumes only."""

import numpy as np

from melampo.imaging.involvement import StructureLabels
from melampo.imaging.radiologist_reading import radiologist_readings

A = np.diag([0.8, 0.8, 1.0, 1.0])
Z, Y, X = np.mgrid[0:60, 0:110, 0:110].astype(float)


def _dist(cx, cy, cz=30):
    return np.sqrt(((X - cx) * 0.8) ** 2 + ((Y - cy) * 0.8) ** 2 + (Z - cz) ** 2)


def _lung():
    return np.full(X.shape, -860.0) + np.random.default_rng(3).normal(0, 20, X.shape)


def test_a_thick_vessel_the_atlas_names_is_not_followed():
    img = _lung()
    d = _dist(55, 55)
    img[d <= 8] = 30.0                                          # lesion 16 mm
    vessel = (np.abs((Y - 55) * 0.8) <= 3.0) & (np.abs(Z - 30) <= 3.0) & (X > 55)   # 6 mm: the opening keeps it
    img[vessel] = 40.0
    lobe = (_dist(55, 55) <= 45) & ~vessel
    bare = radiologist_readings(img, d <= 7.5, A, "CT", "lung_upper_lobe_right", {"lung_upper_lobe_right": lobe})
    assert "reaches_the_reading_zone_edge" in bare["window"]["notes"]           # without the atlas: it leaks
    labels = np.zeros(X.shape, np.int16)
    labels[lobe] = 1
    labels[vessel] = 2
    atlas = StructureLabels(labels, {1: "lung_upper_lobe_right", 2: "lung_arteries"})
    r = radiologist_readings(img, d <= 7.5, A, "CT", "lung_upper_lobe_right", atlas)
    assert abs(r["window"]["axial_long_mm"] - 16) < 1.5 and not r["window"]["notes"]


def test_the_lesion_stays_in_its_cavity_against_an_unnamed_chest_wall():
    img = _lung()
    wall = X >= 80                                              # soft tissue the atlas does not name
    img[wall] = 45.0
    d = _dist(68, 55)
    img[d <= 9] = 35.0                                          # lesion 18 mm touching the wall
    cavity = ~wall
    labels = np.zeros(X.shape, np.int16)
    labels[cavity & ~(d <= 9)] = 1
    no_cavity = radiologist_readings(img, d <= 8.5, A, "CT", "lung_upper_lobe_right",
                                     StructureLabels(labels, {1: "lung_upper_lobe_right"}))
    assert "reaches_the_reading_zone_edge" in no_cavity["window"]["notes"]
    atlas = StructureLabels(labels, {1: "lung_upper_lobe_right"}, regions={"thoracic_cavity": cavity})
    r = radiologist_readings(img, d <= 8.5, A, "CT", "lung_upper_lobe_right", atlas)
    assert r["cavities"] == ["thoracic_cavity"] and abs(r["window"]["axial_long_mm"] - 18) < 2.5
    assert not (r["window"]["mask"] & wall & ~np.roll(cavity, 1, axis=2)).any()


def test_mediastinal_fat_is_a_plane_not_lesion():
    img = _lung()
    fat = (X >= 75) & (X < 100)
    img[fat] = -100.0 + np.random.default_rng(4).normal(0, 12, X.shape)[fat]
    img[X >= 100] = 40.0
    d = _dist(66, 55)
    img[d <= 8] = 35.0                                          # lesion 16 mm against the fat
    lobe = (X < 75) & ~(d <= 8)
    labels = np.zeros(X.shape, np.int16)
    labels[lobe] = 1
    r = radiologist_readings(img, d <= 7.5, A, "CT", "lung_upper_lobe_right",
                             StructureLabels(labels, {1: "lung_upper_lobe_right"}))
    assert r["fat_plane_voxels"] > 0 and r["window"]["axial_long_mm"] < 16 + 4.0


def test_the_tissue_around_a_large_mass_is_its_parenchyma_not_a_mixed_ring():
    img = _lung()
    wall, media = X >= 95, X < 12
    img[wall] = 45.0
    img[media] = 40.0
    d = _dist(55, 55)
    img[d <= 16] = 30.0 + np.random.default_rng(5).normal(0, 15, X.shape)[d <= 16]   # a 32 mm mass
    img[(d > 16) & (d <= 18.5)] = -600.0                        # ground glass around it
    lobe = ~wall & ~media & ~(d <= 16)
    labels = np.zeros(X.shape, np.int16)
    labels[lobe] = 1
    labels[wall] = 2
    labels[media] = 3
    atlas = StructureLabels(labels, {1: "lung_lower_lobe_right", 2: "autochthon_right", 3: "heart"})
    r = radiologist_readings(img, d <= 15.5, A, "CT", "lung_lower_lobe_right", atlas)
    assert r["surrounding_from"] == "holding_structure" and r["noise"] < 60
    assert abs(r["total"]["axial_long_mm"] - 37) < 2.5        # with its ground glass, not a fragment
    assert abs(r["solid"]["axial_long_mm"] - 32) < 2.0


def test_the_band_reading_refines_only_near_the_segmenter_edge():
    img = _lung()
    d = _dist(55, 55)
    img[d <= 8] = 30.0
    img[(d > 8) & (d <= 9.5)] = -450.0                          # a rim 1.5 mm, brighter than the window level
    far = _dist(55 + 8 / 0.8, 55) <= 3                          # a blob 6-8 mm beyond, joined by a bridge
    img[far] = 30.0
    img[(np.abs((Y - 55) * 0.8) <= 0.5) & (np.abs(Z - 30) <= 0.5) & (X > 55) & (X < 66)] = -400.0
    r = radiologist_readings(img, d <= 8, A, "CT", "lung_upper_lobe_right")
    assert abs(r["window_band"]["axial_long_mm"] - 19) < 1.6   # the rim, within 3 mm of the seed's edge
