# Imaging Decision Record

Decisions, measurements and defects for Melampo's imaging work: lesions in CT/MRI, their measurement over time for the same patient, and their comparison across patients. Entries are appended with their date; earlier entries are not rewritten.

The full evaluation behind these entries (sources, licences, alternatives considered) is the project document "valutazione_pipeline_imaging_2026-09-26" (rev. 3).

---

## 2026-09-26 — Scope

**Decided by the owner.**

- **Report↔image discrepancy detection is out of scope.** Pillar-0 is not used to check a radiologist's report against the image.
- **Two scenarios.**
  - **(A) Same patient over time.** Precision is what confirms and diagnoses the evolution of a pathology.
  - **(B) Different patients.** Alterations are compared by shape, density/texture ("consistency") and anatomical position. Size and 3D rotation are treated as transformable.
- **Each abnormal region is isolated and described** by size, shape and consistency. Its extension into neighbouring structures ("invasiveness") is measured. Every involved structure is labelled at the finest anatomical granularity available and turned into a vector.
- **Still in force from earlier decisions.**
  - No atlas-based normalisation for registration.
  - An unguided reading always runs in parallel to any guided one.
  - Compute cost is not a constraint; precision is.

## 2026-09-26 — What limits precision: a measurement

No published source gave a number for how precisely a lesion's centre can be located along z, so it was measured. `scripts/simulate_subvoxel_precision.py` models a rigid high-contrast sphere, partial volume, 0.7 mm pixels, contiguous slices and white noise, with 40 trials per configuration and seed 42:

| Diameter | Slice | RMS error x | RMS error z | Volume, partial volume | Volume, 50% threshold |
|---|---|---|---|---|---|
| 4 mm | 0.625 mm | 0.010 mm | 0.011 mm | -4.5% ± 0.7% | -3.7% ± 2.4% |
| 4 mm | 1.25 mm | 0.015 mm | 0.028 mm | -6.8% ± 1.4% | -6.2% ± 4.6% |
| 4 mm | 2.5 mm | 0.027 mm | 0.035 mm | -11.0% ± 2.5% | -12.2% ± 10.7% |
| 4 mm | 5 mm | 0.047 mm | 1.086 mm | -20.0% ± 10.1% | -26.9% ± 42.9% |
| 6 mm | 0.625 mm | 0.007 mm | 0.007 mm | -2.9% ± 0.3% | -1.3% ± 1.2% |
| 6 mm | 1.25 mm | 0.009 mm | 0.010 mm | -3.9% ± 0.5% | -2.1% ± 1.8% |
| 6 mm | 2.5 mm | 0.015 mm | 0.056 mm | -6.4% ± 0.9% | -4.0% ± 2.6% |
| 6 mm | 5 mm | 0.023 mm | 0.463 mm | -11.4% ± 2.4% | -11.9% ± 12.2% |
| 10 mm | 0.625 mm | 0.005 mm | 0.004 mm | -1.7% ± 0.2% | -0.7% ± 0.4% |
| 10 mm | 1.25 mm | 0.006 mm | 0.008 mm | -2.1% ± 0.2% | -0.9% ± 1.0% |
| 10 mm | 2.5 mm | 0.006 mm | 0.027 mm | -3.4% ± 0.5% | -1.7% ± 2.7% |
| 10 mm | 5 mm | 0.011 mm | 0.215 mm | -6.2% ± 1.7% | -3.2% ± 12.5% |

**Reading.**

- **Thin slices:** under ideal conditions and with slices ≤ 2.5 mm, a high-contrast lesion's centre is located well below 0.1 mm, along z too.
- **5 mm slices:** a small lesion falls in one or two slices, and z is lost (up to 1.1 mm; small-lesion volume ±43%).
- **Precision, not accuracy:** this is the repeatability of an estimate. Scanner blur, correlated noise, motion, irregular shape, attached vessels and segmentation variability are all absent from the model.
- **Real-world volume repeatability is an order of magnitude worse** than the simulated 0.2–4.6% at ≤ 1.25 mm. The QIBA small-nodule profile (2023) gives a coefficient of variation of 0.29 at 6 mm and 0.14 at 10 mm. Precision work therefore belongs in acquisition consistency and segmentation, not in sub-voxel estimation itself.

## 2026-09-26 — DICOM geometry completed (`data/dicom_volume.py`)

The assessment already checked size, pixel spacing, orientation, slice positions along the normal (duplicates, uneven spacing), slice count, mixed series and multi-frame. Four things were missing for measurements in millimetres. Each was added and tested on synthetic series (`tests/test_dicom_geometry.py`).

1. **Voxel → patient mm (`VolumeAssessment.affine`, `index_to_patient_mm`).**
   - Built from ImagePositionPatient, ImageOrientationPatient and PixelSpacing (row spacing first, then column spacing: PS3.3 C.7.6.2).
   - The slice step is (last − first) / (N − 1) from the positions, as NiBabel does. It never comes from SliceThickness ("nominal") or SliceLocation (relative to an unspecified reference).
   - Fractional indices give sub-voxel positions.
   - The affine is built only when the slices form one regular grid. With uneven spacing, duplicate or off-line positions, mixed orientation, size, spacing, series or frame of reference, one step vector would misplace the inner slices, so the affine is withheld and `index_to_patient_mm` refuses.
   - A test uses non-square pixels (0.5 × 0.8 mm) so that a row/column swap cannot pass.
2. **Gantry tilt, measured from the positions.**
   - GantryDetectorTilt is "not intended for mathematical computations".
   - The angle between the slice step and the plane normal is `tilt_degrees`.
   - Above 0.01° the series is noted as `sheared_volume_gantry_tilt` and stays usable. The affine carries the shear, so the last slice maps exactly to where the scanner placed it.
   - A slice displaced sideways from the line through the others is refused (`slice_positions_not_collinear`).
3. **Slice thickness against the measured spacing.**
   - `gaps_between_slices`: anatomy is never sampled between slices, and QIBA requires spacing ≤ thickness.
   - `overlapping_slices`: a normal reconstruction choice.
   - `inconsistent_slice_thickness`.
   - `spacing_between_slices_tag_disagrees`.
   - These are reported as notes, not refusals.
4. **Frame of reference.**
   - A series mixing FrameOfReferenceUIDs is refused.
   - `share_frame_of_reference` answers whether two series are spatially related by the scanner (C.7.4). An unknown frame is never assumed shared.

Two functions turn the geometry into claims.

- **`measurement_precision`** declares one of four levels:
  - `high`: QIBA conditions, ≤ 1.25 mm with no gaps;
  - `reduced`: ≤ 2.5 mm, where the simulation still shows sub-voxel z;
  - `low`: thicker slices, gaps, mixed or unknown thickness;
  - `none`: not a usable volume.

  MR is flagged `mr_intensities_not_absolute`, and sheared volumes `sheared_volume_measure_through_affine_only`.
- **`compare_acquisitions`** lists what differs between two exams. QIBA comparability requires the same thickness, reconstruction kernel and scanner model. An unknown value on either side counts as a difference.

`Manufacturer` and `ManufacturerModelName` were added to the de-identification allowlist so the scanner model can be compared. Neither identifies a patient. A kernel or scanner model that changes inside one series is reported (`inconsistent_convolution_kernel`, `inconsistent_manufacturer_model`).

**Behaviour change, intended.** Two new problems make series that were usable before unusable:
- `mixed_frame_of_reference`, including a series where only some instances carry the UID;
- `slice_positions_not_collinear`.

Such series are no longer eligible for Pillar-0 or for measurement. A series with missing frame-of-reference UIDs on every instance is still usable, but is never considered spatially related to another series.

## 2026-09-26 — Defect: visual imprints destroyed their input vectors (`memory/visual_imprint.py`)

**Found while evaluating** whether the existing "morphing algorithm" (`VisualImprintMorpher`) could compare lesion shapes.

**What was wrong.** Every vector given to `VisualRecognitionImprint.from_payload` went through `_matrix_to_vector`. That function:
- read at most 256 values;
- folded them into 64 buckets through `tanh`;
- hashed strings into pseudo-random values.

`_cosine` silently truncated vectors of different length to the shorter one. `as_dict()` → `from_payload()` was not idempotent: the stored vector was re-hashed.

**Reproduced.**

| Input | True cosine | Cosine after imprint |
|---|---|---|
| Two 1152-value embeddings differing from position 257 on | 0.214 | 1.000 |
| `{"volume_mm3": 1000, "sphericity": 0.9}` vs `{"volume_mm3": 30000, ...}` | — | 1.000 (identical vectors) |

**Fixed.**
- **A flat sequence of finite numbers** (list, tuple or numpy array) is now a `numeric_embedding`: kept whole and exact, only L2-normalised.
- **Structured payloads** remain `hashed_signature` fingerprints, documented as such.
- **Round trips:** an `as_dict()` read back keeps its kind and never re-hashes.
- **Cosine of different-length vectors** is 0, never truncated.
- **The morpher** pairs and compares only imprints of the same kind and dimension. It reports `incomparable_pair_count` and ignores diagnostic targets it cannot compare.
- **Regression tests:** `tests/test_visual_imprint_vectors.py`. The existing morpher, trainer and Weaviate tests pass unchanged.

**Defects in the first version of this fix, found by an independent review the same day and corrected before commit.**
- Numpy scalar elements (`list(np.ones(300, np.float32))`) and a `(1, N)` batch dimension were rejected as numbers, which sent embeddings back down the lossy signature path. Both are now accepted; booleans still are not.
- The round trip was not exact: re-normalising an already rounded unit vector moved the 6th decimal of about 1% of vectors, changing `matrix_signature_hash`. A vector that is already a rounded unit vector is now kept as is (tested on 900 random vectors, dimensions 3, 8 and 300).
- The new payload-key selection differed from the old `or` chain: a falsy scalar in `vector` was chosen over `embedding`, and a 0-d numpy array crashed. The old rule is restored, without the chain's crash on multi-element arrays.
- Skipped pairs were visible only as a counter. The morpher now returns a `warnings` entry. Callers that mix builder signatures with numeric embeddings get no morphs for those pairs, and now say so.
- Weaviate fixes a named vector's dimension at the first insert, so embeddings of mixed dimension written into `recognition_matrix_vector` would be rejected by a real server. The two kinds are now stored under separate named vectors:
  - signatures (always 64 values) in `recognition_matrix_vector`;
  - embeddings in the new `numeric_embedding_vector`.

  The adapter enforces one embedding dimension per store (`rejected_embedding_dimension_mismatch`).

**Migration note.** Imprints serialised before this change carry no `vector_kind`. Read back, a stored 64-value signature is taken as a `numeric_embedding` and no longer pairs with new signatures. Any stored imprints should be rebuilt from their source.

**Not fixed, and why.**
- **Cosine cannot compare sizes.** It ignores scale by definition, so a measurement vector still cannot be compared through an imprint: [1000, 0.9] vs [30000, 0.9] scores 0.99999. A test pins this limit. Measurements (size, shape indices, density) need standardised features and a distance in their own comparison, which is Phase 4/7 of the plan.
- **The "bridge" term** added to each morph is derived from a SHA-256 hash of the concept names, not from any learned or geometric direction. Changing the morph's semantics needs a decision, so it is recorded here, not changed.
- **Negative similarities score 0** (`_cosine` clamps to [0, 1]). This is acceptable for a score, but "opposite" and "unrelated" are indistinguishable.
- **The morpher morphs vectors, not shapes.** The geometric shape morphing proposed for scenario B is: normalise both masks, register one onto the other in both directions, and use deformation plus residual mismatch as the cost. It is separate work and still to decide (D5).

## 2026-09-26 — Manifest correction

The `rave` note in `melampo-assets.yaml` said that `export_series_directory` writes RAVE's input CSV. It writes one series directory; the CSV that lists such directories is not written by anything yet. The note is corrected. Verified licences were recorded the same day:
- RAVE: ECL-2.0 (LICENSE file);
- Pillar-0 AbdomenCT, HeadCT and BreastMRI: ECL-2.0 (model cards).

## Open decisions (from the evaluation, rev. 3)

| # | Decision |
|---|---|
| D1 | Lesion isolation tools under a commercial-use constraint |
| D2 | Anatomical vocabulary and Italian labels |
| D3 | Brain: an atlas used only to label, never to measure (an exception to "no atlas"), and which atlas |
| D4 | Learned block of the lesion vector: Pillar-0, a JEPA encoder, or both |
| D5 | Geometric morphing method |
| D6 | Acquisition requirements for declaring maximum precision (now implemented as `measurement_precision` levels; thresholds to confirm) |
| D7 | Who confirms lesion masks |
| D8 | Surface anatomy (nasal subunits, fingertip zones): outside CT/MRI scope, or a separate photographic pipeline |
| D9 | Where the GPU runs |
| D10 | Archive of cases with confirmed diagnoses for scenario B |
