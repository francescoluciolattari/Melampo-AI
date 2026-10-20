"""The lesion read as a radiologist reads it: window, solid part, total with its ground glass, long and short
axis (20 October 2026). Synthetic volumes only."""

import numpy as np

from melampo.imaging.lesion_measures import measure_region
from melampo.imaging.radiologist_reading import radiologist_readings, window_for

A = np.diag([0.8, 0.8, 1.0, 1.0])
Z, Y, X = np.mgrid[0:60, 0:100, 0:100].astype(float)
D = np.sqrt(((X - 50) * 0.8) ** 2 + ((Y - 50) * 0.8) ** 2 + (Z - 30) ** 2)


def _lung(rim_hu=-650.0):
    img = np.full(X.shape, -860.0) + np.random.default_rng(0).normal(0, 20, X.shape)
    img[D <= 11] = rim_hu                                     # ground glass to 22 mm
    img[D <= 8] = 30.0                                        # solid part 16 mm
    vessel = (np.abs((Y - 50) * 0.8) <= 1.2) & (np.abs(Z - 30) <= 1.2) & (X > 50)
    img[vessel] = 40.0                                        # a thin vessel leaving the lesion
    aorta = np.sqrt(((X - 50) * 0.8) ** 2 + ((Y - 50) * 0.8 - 14) ** 2) <= 5
    img[aorta] = 40.0                                         # a named structure touching the lesion
    return img, aorta


def test_window_solid_and_total_on_a_part_solid_lesion_beside_the_aorta():
    img, aorta = _lung()
    lobe = (D <= 40) & ~aorta
    r = radiologist_readings(img, D <= 7.5, A, "CT", "lung_upper_lobe_right",
                             {"lung_upper_lobe_right": lobe, "aorta": aorta})
    assert r["window"]["window"] == "lung" and r["window"]["level"] == -600.0
    assert abs(r["window"]["axial_long_mm"] - 16) < 1.5          # -650 HU is darker than the window's grey
    assert abs(r["solid"]["axial_long_mm"] - 16) < 1.5
    assert abs(r["total"]["axial_long_mm"] - 22) < 1.5 and not r["total"]["notes"]
    assert not (r["total"]["mask"] & aorta).any() and r["kept_structures"] == ["lung_upper_lobe_right"]


def test_a_ground_glass_brighter_than_the_window_level_is_inside_the_window_edge():
    img, aorta = _lung(rim_hu=-450.0)
    r = radiologist_readings(img, D <= 7.5, A, "CT", "lung_upper_lobe_right", {"aorta": aorta})
    assert abs(r["window"]["axial_long_mm"] - 22) < 1.5 and abs(r["solid"]["axial_long_mm"] - 16) < 1.5


def test_outside_the_lung_the_soft_tissue_window_or_half_way():
    assert window_for("liver") == "soft_tissue" and window_for("lung_lower_lobe_left") == "lung"
    img = np.full(X.shape, 60.0) + np.random.default_rng(1).normal(0, 8, X.shape)   # liver
    img[D <= 10] = 20.0                                                            # hypodense lesion
    r = radiologist_readings(img, D <= 9, A, "CT", "liver", {"liver": D <= 40})
    assert r["window"]["window"] == "soft_tissue" and r["solid"] is None
    assert abs(r["window"]["axial_long_mm"] - 20) < 2.0
    img2 = img.copy()
    img2[D <= 10] = 140.0                                                          # both above the level
    r2 = radiologist_readings(img2, D <= 9, A, "CT", "liver", {"liver": D <= 40})
    assert any(n.startswith("window_level_not_between") for n in r2["notes"])
    assert abs(r2["window"]["axial_long_mm"] - 20) < 2.0


def test_not_on_mr():
    assert radiologist_readings(np.zeros(X.shape), D <= 9, A, "MR")["window"] is None


def test_long_and_short_axis_like_a_radiologist():
    m = (((X - 50) * 0.8 / 10) ** 2 + ((Y - 50) * 0.8 / 5) ** 2 + ((Z - 30) / 4) ** 2) <= 1
    got = measure_region(m, A)
    assert abs(got.longest_axial_diameter_mm - 20) < 1.0 and abs(got.axial_short_mm - 10) < 1.0
    assert abs(got.axial_mean_mm - 15) < 1.0
