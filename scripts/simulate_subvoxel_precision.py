"""How precisely can a high-contrast lesion's centre and volume be estimated, as a function of slice thickness?

Evidence behind the precision thresholds in data/dicom_volume.py
(docs/imaging_decision_record.md). No published source gave a number for
localising a lesion's centre along z, so this measures it under IDEAL
conditions -- an upper bound on what real exams allow:

- a rigid sphere (nodule +40 HU in lung -800 HU: contrast 840 HU);
- voxel value = the fraction of the voxel inside the sphere (partial
  volume), in-plane pixel 0.7 mm, contiguous slices with a box profile;
- white Gaussian noise, 30 HU at 1.25 mm, scaled by 1/sqrt(thickness);
- random sub-voxel position of the true centre in every trial.

Not modelled, and each makes real exams worse: scanner blur (point-spread
function), correlated noise, motion, irregular shapes, attached vessels,
segmentation variability. The result is precision (repeatability of the
estimate), not accuracy against true anatomy.

Centre: background-subtracted, intensity-weighted centroid. Volume: sum of
partial-volume fractions above 0.25 (biased low by that cut; the bias is
correctable, the spread is what limits), and the count of voxels above 50%.

Run: uv run python scripts/simulate_subvoxel_precision.py [--trials 40]
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np

PIXEL_MM = 0.7
CONTRAST_HU = 840.0
NOISE_HU_AT_1_25_MM = 30.0


@dataclass(frozen=True)
class Result:
    diameter_mm: float
    thickness_mm: float
    rms_error_x_mm: float
    rms_error_z_mm: float
    volume_bias_partial: float
    volume_sd_partial: float
    volume_bias_threshold: float
    volume_sd_threshold: float


def _fractions(diameter, thickness, centre, sub_xy=6):
    radius = diameter / 2
    half = radius + 3 * max(PIXEL_MM, thickness)
    nxy = int(np.ceil(2 * half / PIXEL_MM))
    nz = int(np.ceil(2 * half / thickness))
    sub_z = max(6, int(np.ceil(thickness / 0.1)))
    edges_xy = np.arange(nxy) * PIXEL_MM - half
    edges_z = np.arange(nz) * thickness - half
    xs = (edges_xy[:, None] + ((np.arange(sub_xy) + 0.5) / sub_xy * PIXEL_MM)[None, :]).ravel()
    zs = (edges_z[:, None] + ((np.arange(sub_z) + 0.5) / sub_z * thickness)[None, :]).ravel()
    inside = (
        (xs - centre[0])[:, None, None] ** 2 + (xs - centre[1])[None, :, None] ** 2 + (zs - centre[2])[None, None, :] ** 2
    ) <= radius * radius
    fraction = inside.reshape(nxy, sub_xy, nxy, sub_xy, nz, sub_z).mean(axis=(1, 3, 5))
    return fraction, edges_xy + PIXEL_MM / 2, edges_z + thickness / 2


def trial(diameter, thickness, rng):
    centre = rng.uniform(-0.5, 0.5, 3) * np.array([PIXEL_MM, PIXEL_MM, thickness])
    fraction, centres_xy, centres_z = _fractions(diameter, thickness, centre)
    noise = NOISE_HU_AT_1_25_MM * math.sqrt(1.25 / thickness)
    weights = (fraction * CONTRAST_HU + rng.normal(0, noise, fraction.shape)) / CONTRAST_HU
    kept = np.where(weights > 0.25, weights, 0.0)
    x, y, z = np.meshgrid(centres_xy, centres_xy, centres_z, indexing="ij")
    estimate = np.array([(kept * x).sum(), (kept * y).sum(), (kept * z).sum()]) / kept.sum()
    voxel = PIXEL_MM * PIXEL_MM * thickness
    true_volume = math.pi * diameter**3 / 6
    return (
        estimate - centre,
        (kept.sum() * voxel - true_volume) / true_volume,
        ((weights > 0.5).sum() * voxel - true_volume) / true_volume,
    )


def simulate(diameters, thicknesses, trials, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for diameter in diameters:
        for thickness in thicknesses:
            runs = [trial(diameter, thickness, rng) for _ in range(trials)]
            errors = np.array([run[0] for run in runs])
            partial = np.array([run[1] for run in runs])
            threshold = np.array([run[2] for run in runs])
            results.append(Result(
                diameter, thickness,
                float(np.sqrt((errors[:, 0] ** 2).mean())), float(np.sqrt((errors[:, 2] ** 2).mean())),
                float(partial.mean()), float(partial.std()), float(threshold.mean()), float(threshold.std()),
            ))
    return results


def render(results):
    lines = [
        "| Diameter | Slice | RMS error x | RMS error z | Volume, partial volume | Volume, 50% threshold |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.diameter_mm:g} mm | {r.thickness_mm:g} mm | {r.rms_error_x_mm:.3f} mm | {r.rms_error_z_mm:.3f} mm "
            f"| {r.volume_bias_partial:+.1%} ± {r.volume_sd_partial:.1%} | {r.volume_bias_threshold:+.1%} ± {r.volume_sd_threshold:.1%} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(render(simulate((4.0, 6.0, 10.0), (0.625, 1.25, 2.5, 5.0), args.trials, args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
