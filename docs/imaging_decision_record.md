# Imaging Decision Record

Decisions, measurements and defects for Melampo's imaging work: lesions in CT/MRI, their measurement over time for the same patient, and their comparison across patients. Entries are appended with their date; earlier entries are not rewritten.

The full evaluation behind these entries (sources, licences, alternatives considered) is the project document "valutazione_pipeline_imaging_2026-09-26" (rev. 4); the 2026-10-02 architecture is in "architettura_voxtell_sdf_vljepa_2026-10-02".

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

## 2026-09-26 — Scope clarified: the report says what, the image says where and how

**Decided by the owner, correcting a misunderstanding that ran through the earlier entries.**

- **The radiology report is the source of what is in the exam.** Analysing reports is not the imaging model's job. The imaging model only has to **find in the image what the report describes**, isolate it, identify it, and apply the transformations needed to compare it with:
  - other exams of the same patient (evolution of the disease);
  - diagnosed cases in the medical literature.
- **Normalisation happens before any vector is made.** The findings already decoded in the report are normalised to the project's own synonyms and disease database by the existing text pipeline, with the Nemotron/Gemma control models. VL-JEPA and energy-based models then work only on those normalised terms. They do not find anything themselves.
- **Two paired objects.**
  - **Textual object** (from the report): finding, anatomical region at the finest granularity, size, consistency, invasiveness (which regions, how).
  - **Visual object** (from the image): the region those attributes refer to.

  Both become vectors in one space, compared by concept and through the project's morphing.
- **All vectors are kept in memory** (symptoms, diagnoses, reports, visual objects) for the "intuition" model, which is still to be built.

**Consequences for earlier entries.**

- **Invasiveness is the radiologist's statement**, carried in the textual object. The geometric contact measure is its visual counterpart, used to locate and to follow it over time. The earlier concern that geometry cannot certify infiltration no longer applies: certification was never this model's task.
- **The unguided reading "always in parallel" was a safeguard against missing unreported findings**, which is discrepancy detection and out of scope. It is proposed for withdrawal; the owner decides. What remains is grounding quality: when no candidate, or more than one, matches the description, the visual object is reported as not found or ambiguous, never forced.
- **Registration is not needed to find a finding in its own exam.** It serves:
  - the same patient over time: correspondence of a lesion across exams, and the direction of change;
  - reports that describe a finding only relative to the previous exam ("invariato", "in riduzione"), where the location must come from the earlier exam.

**Facts that shape the design (verified 2026-09-26).**

- **VL-JEPA** (arXiv 2512.10942) has three parts:
  - a frozen V-JEPA 2 ViT-L vision encoder (video/2D frames);
  - a predictor from Llama-3.2-1B;
  - a text Y-encoder initialised from EmbeddingGemma-300M and fine-tuned, with 1,536-d projections.

  Its only weights are under a FAIR non-commercial research licence. No medical or 3D version exists. An own VL-JEPA-style model with commercially usable components, trained on paired objects, is the route that fits an MDR product (decision `imaging-vector-model` in the manifest).
- **Text embeddings are unreliable with numbers.** Retrieval between texts differing only in a number averaged 0.54 across 13 models, near chance (Deng et al., EACL 2026). Sizes and other measurements stay as structured numbers beside the vector, never only inside it.
- **Report-to-volume grounding models are all non-commercial or unlicensed**, and the benchmarks and grounded datasets are mostly chest-only and English-only:
  - VoxTell: weights CC-BY-NC-SA; best on real report sentences, Dice 50.2;
  - BiomedParse v2: weights CC-BY-NC-SA;
  - SAT: no licence.

  The best result on ReXGroundingCT is Dice 0.32. The commercially usable route is:
  1. anatomical segmentation of the region named in the report (TotalSegmentator free tasks);
  2. lesion candidates inside that region;
  3. selection by reported size and density.

  Candidates that are not selected are discarded, never shown: showing them would be unreported-finding detection by another route. "Not found" and "ambiguous" describe the localisation, never an error in the report.

  No published accuracy exists for step 3, so it must be measured on Melampo's own data.
- **Diagnosed cases with commercial rights are few:**
  - TCIA collections under CC BY;
  - the Medical Segmentation Decathlon (CC BY-SA 4.0);
  - PMC open-access figures of the commercial subset. ROCOv2 and MultiCaRe need per-image licence filtering, because their dataset licence sits on per-article licences that include CC BY-NC.

  Radiopaedia, MedPix, CT-RATE, Merlin, AbdomenAtlas, KiTS and LiTS are not usable commercially. Literature figures are 2D, so comparison with them runs on key slices through the lesion plus concept-level matching.
- **Vectors are only comparable within one encoder.** Every stored vector carries the encoder id and version, and the normalised object it came from stays the source of truth, so vectors can be recomputed when the encoder changes. Vectors derived from non-commercial data sit in a separate partition, by the same rule `connectors/pmc_case_reports.py` applies to articles. The same partition applies to training pairs: share-alike data (e.g. CC BY-SA) may pass its terms to a trained model. Using processed patient exams as training pairs is a proposal, not a decision: it needs a legal basis for secondary use of health data (GDPR art. 9), de-identification, and the phase-one rule (D14).

## 2026-09-28 — Every image-bearing exam; measurements beside the vector; the licence question

**Decided by the owner.**
- **Every image-bearing exam is in scope**, not only CT and MRI: X-ray (including mammography), ultrasound and the other exams that come with images. The semantic association concerns a normalised medical term and the image region it describes, in every modality.
- **Measurements stay as numeric fields beside the vector**, and the vector encodes the normalised concept. This was agreed after the 2026-09-26 evidence that text embeddings handle numbers near chance.

**Licence question: changing Melampo's own licence does not lift third-party restrictions.** This is an engineering reading of primary texts, not legal advice, and it is to be confirmed by a lawyer (D15).
- **A licensee cannot widen the rights it received.** CC BY-NC 4.0 grants its rights "for NonCommercial purposes only" (Section 2(a)(1)) and nothing lets a licensee extend them. "NonCommercial" is defined by the purpose of the use ("not primarily intended for or directed towards commercial advantage or monetary compensation"), not by the user's own licence.
- **MDR Art. 2(27):** making a device available on the market is supply "in the course of a commercial activity, whether in return for payment or free of charge". In our reading, a device made available on the market is supplied commercially even when free. Whether this regulatory definition settles the meaning of "NonCommercial" in CC or FAIR terms is interpretation.
- **MDR Art. 5(5)** (manufactured and used only within one health institution, "not transferred to another legal entity", no equivalent device on the market) is the only route that avoids placing on the market. Whether such use counts as non-commercial under third-party licences is a separate question.
- **The FAIR Noncommercial Research License** (VL-JEPA weights) permits only noncommercial research uses. Derivatives stay under the same terms, and its acceptable-use policy bars medical professional services without proper licensing.
- **ShareAlike** may extend to models trained or fine-tuned on SA material. Creative Commons has not said whether a trained model is Adapted Material.
- **Creative Commons recommends against CC licences for software** (CC0 excepted).
- **Text and data mining exceptions (Directive 2019/790):**
  - Art. 4 (general TDM) can yield to contracts: Art. 7(1) protects only Arts. 3, 5 and 6 from contractual override. Terms accepted to access gated models or data may therefore prevail; whether a particular agreement binds is a contract-law question.
  - Art. 3 (research organisations) is contract-proof, but covers research, not a product.
- **The repository LICENSE file** calls itself Business Source License 1.1 but paraphrases it. The official text allows only its parameters to change ("Not to modify this License in any other way").
- **The planned commercial licensing and the 2029 change to Apache-2.0** are, in our reading, incompatible with non-commercial components inside the product.
- **Routes:**
  - a research partition that never enters the product (the rule `connectors/pmc_case_reports.py` already applies);
  - commercial licences on request (Stanford AIMI about USD 70,000 per dataset per year; PadChest and BIMCV case by case; HyperKvasir by written permission);
  - retraining permissively licensed code on permissively licensed data;
  - hospital partnerships with a legal basis for secondary use, which would also supply Italian reports.

**Sources surveyed** (full map in the project document "fonti_modelli_dati_per_modalita_2026-09-28").
- **Commercially usable encoders:**
  - CT/MRI: Pillar-0 (ECL-2.0, aligned to Qwen3-Embedding-8B), CT-FM (Apache-2.0), Merlin (weights MIT, abdominal CT; trained on the non-commercial Merlin data, so a training-data risk).
  - Chest X-ray: MedSigLIP and CXR Foundation (Google HAI-DEF terms: commercial use allowed; derivatives, including distillation, stay bound; Google must not become a device "manufacturer").
  - Text: Qwen3-Embedding (Apache-2.0, over 100 languages; Italian not named explicitly on the card), BGE-M3 and multilingual-e5 (MIT).
- **Gaps:** no commercially licensed text-aligned model for brain MRI, mammography or ultrasound.
- **Several MIT-licensed Microsoft models state "research only"** on their cards (RAD-DINO, BiomedCLIP, COLIPRI) and need legal review. A permissive weight licence does not clear restrictions on the training data (Merlin, M3D-CLIP, CheXagent).
- **No public dataset pairs images with free-text reports under a clearly commercial licence**, and none exists in Italian. The commercially usable ones carry structured descriptors: LIDC-IDRI, CBIS-DDSM, CMMD, Breast-Lesions-USG, BraTS 2021, NLST images, OpenNeuro CC0, the CC0/CC BY part of ISIC. The Medical Segmentation Decathlon is CC BY-SA, with the share-alike caveat above.
- **This fits the design:** the textual object is a set of normalised concepts, not free text, so structured descriptors convert directly into textual objects for training pairs.

## 2026-10-02 — Correction: the vector store is FalkorDB, not Weaviate

**Found by the owner.** Graph and vector storage were decided as FalkorDB (`docs/recursive_engine_decision_record.md`, "FalkorDBLite scelto per il progetto"). `docs/architecture_consistency_matrix.md` marks the Weaviate component as superseded and not being built out. The 2026-09-26 fix nevertheless also changed `memory/weaviate_adapter.py` and `memory/weaviate_schema.py`.

- **What stands.** The fix in `memory/visual_imprint.py` is storage-independent and stays. The Weaviate changes are harmless and keep their tests, but they target a component that will not be used. That work should not have been spent there.
- **The rule carries over to FalkorDB.** A FalkorDB vector index is declared per label and property, with a fixed `dimension` (1–4096) and a `similarityFunction` (FalkorDB documentation, Cypher cheat sheet, read 2026-10-02): `CREATE VECTOR INDEX FOR (n:Label) ON (n.prop) OPTIONS {dimension: …, similarityFunction: 'cosine'}`. Vectors from different encoders therefore go into different properties, one index each, never into the same one. `memory/falkordb_graph.py` has no vector index yet.
- **Gap found while checking.** `_comparable` in `visual_imprint.py` checks only vector kind and length. Two different encoders with the same dimension would be compared silently, and their scores would mean nothing. Imprints need the encoder id and version, and comparison must require them to match. This is not fixed here; it is listed as D21.

## 2026-10-02 — Melampo's own licence changed to CC BY-NC 4.0

**Decided by the owner** (commits 6a9ea65 and ba7998e on main). This supersedes the remark on the Business Source License in the 2026-09-28 entry. The observations below are an engineering reading of the CC BY-NC 4.0 legal code (SPDX copy, read 2026-10-02). They are not legal advice and go to the legal review (D15).

- **The added attribution terms go beyond what the licence lets a licensor require**, in our reading. Section 3(a)(1)(A)(i) lets the licensor request "any reasonable manner" of identifying the creator. Section 3(a)(2) lets the licensee satisfy the conditions "in any reasonable manner based on the medium, means, and context", for example "by providing a URI or hyperlink". A mandatory, permanent credit inside the user interface may not be enforceable as part of the CC licence.
- **Termination is stated incompletely.** Section 6(a) does end the rights automatically on a breach. Section 6(b) reinstates them "automatically as of the date the violation is cured, provided it is cured within 30 days of Your discovery of the violation". The LICENSE text omits the reinstatement.
- **A modified CC licence is a different licence.** Creative Commons does not authorise its trademark "in connection with any unauthorized modifications to any of its public licenses" (closing notice of the legal code). Either the plain CC BY-NC 4.0 is used, or a custom licence is written that does not carry the CC name.
- **Form.**
  - The LICENSE file links only the Creative Commons home page, not the licence deed or legal code (https://creativecommons.org/licenses/by-nc/4.0/).
  - Its conditions are numbered 1, 3, 4.
  - The licence grants no patent rights (Section 2(b)(2)).
  - Creative Commons recommends against CC licences for software (2026-09-28 entry).
- **The licensor is not bound by its own licence.** The owner can still grant separate commercial licences. Contributions from others received under CC BY-NC would need a contributor agreement for that to remain possible. The README no longer gives a commercial contact.
- **Effect on third-party non-commercial components.**
  - With the project now non-commercial, using non-commercial weights such as VoxTell's (CC BY-NC-SA 4.0) is consistent with the project's declared use, in our reading.
  - ShareAlike: VoxTell weights fine-tuned by us would have to be shared under CC BY-NC-SA, not under Melampo's CC BY-NC. Calling the weights from our code does not, in our reading, make the code Adapted Material of them.
  - The FAIR Noncommercial Research License (Meta's VL-JEPA weights) is narrower: research only. Its acceptable-use policy bars medical professional services without proper licensing.
  - The MDR applies or not according to how the software is used and supplied, not according to its licence.
  - A future commercial version would need every non-commercial component replaced. The licence partition per component therefore stays.

## 2026-10-02 — New imaging architecture: VoxTell → neural geometry → VL-JEPA-style pairing → graph nodes

**Decided by the owner.** The pipeline has four stages:
1. **VoxTell** isolates the region the report describes.
2. **A neural geometric representation** models the lesion's continuous geometry and distances: a signed distance field (SDF), or 3D Gaussian splatting.
3. **A VL-JEPA-style model** pairs the semantic meaning with that geometry.
4. **The result is a node in the knowledge graph**, with edges carrying continuous proximity to neighbouring structures.

No mesh, B-rep or VTK. The repository uses no VTK today.

This fits the 2026-09-26 scope ("the report says what, the image says where and how"). VoxTell is a tool for finding in the image what the report names. Each claim in the proposal was checked; the table records what holds and what changes.

| # | Claim in the proposal | Finding | Consequence |
|---|---|---|---|
| 1 | VoxTell maps report text to a 3D region | Holds. Free-text prompts → volumetric masks on CT, PET and MRI. Its prompt encoder is the frozen Qwen3-Embedding-4B. Images must be in RAS orientation, and they are not resampled (about 1.5 mm is typical). `voxtell-finetune` exists since v0.1.2 (README and PyPI 0.1.2, read 2026-10-02). | The prompt is the normalised textual object (finding, fine region, laterality), not the raw report. The published prompts are English; Italian prompts are unmeasured (D20). Orientation comes from `dicom_volume`'s affine. |
| 2 | VoxTell outputs a probabilistic field, not a binary mask | Partly. The predictor thresholds the logits at 0, which is sigmoid > 0.5 (`voxtell/inference/predictor.py`). The logits exist before the threshold, so a soft map is obtainable by changing the predictor. | A sigmoid output is not a calibrated probability. Calibration would have to be measured before the soft map is read as confidence. |
| 3 | VoxTell isolates the region reliably | Not established. Dice 50.2 on real report sentences (2026-09-26 survey). VoxTell returns a mask for any prompt. | The outcomes found / not found / ambiguous must stay possible. The VoxTell mask is checked against the organ and segment from TotalSegmentator (free tasks) and against the reported size and density before it is accepted. |
| 4 | VoxTell is usable | Code Apache-2.0 (verified 2026-10-02). Weights CC BY-NC-SA 4.0 (verified 2026-09-26; Hugging Face unreachable from this session on 2026-10-02). | Usable within the project's non-commercial licence, subject to D15. Fine-tuned weights stay CC BY-NC-SA. |
| 5 | An SDF gives "infinite resolution" of the boundary | Does not hold as information. The zero level set interpolates between voxels. Boundary accuracy stays bounded by voxel spacing, partial volume and the segmentation (2026-09-26 simulation). A neural SDF trained with an eikonal term is only approximately a distance away from the surface. | The SDF gives a smooth sub-voxel surface and fast queries, not new information. It does not move the 0.1 mm goal. Every measure carries `measurement_precision`. |
| 6 | Lesion–vessel distance is a subtraction of two SDFs | Does not hold as stated. The minimum distance is the minimum of the vessel's SDF over points on the lesion's surface: sample, then evaluate, which is fast on GPU. | The vessel needs its own segmentation (TotalSegmentator `liver_vessels`, free). An exact Euclidean distance transform on the voxel masks gives the same quantity at voxel precision and is the cross-check. Edges store distance, method, precision level and uncertainty. |
| 7 | One small network per lesion (`weights/sdf_lesion_001.pt`) | Works for one lesion, but networks fitted separately cannot be compared. | Proposed instead: one shared decoder with a latent code per lesion (DeepSDF, Park et al., CVPR 2019). The lesion is fitted after removing position, orientation and scale, which are kept as numeric fields. The latent code is then a shape vector comparable across patients. Interpolating between two codes is a geometric morph. This is scenario B (size and rotation transformable) and answers D5. |
| 8 | Gaussian splatting represents infiltration: opacity α near 1 is tumour, falling to 0 is microscopic infiltration | No support. In medical imaging, Gaussian splatting is used for reconstruction and rendering (for example CT reconstruction from projections). α is a rendering parameter with no ground truth for infiltration, and CT/MRI do not resolve microscopic infiltration. | The SDF (option A) is used for measurement. Option B is at most a visualisation (D18). |
| 9 | VL-JEPA takes the 3D patch and the text and gives a 768-d vector, self-supervised | Partly. Meta's VL-JEPA has a 1,536-d shared space, a 2D video encoder and FAIR non-commercial research weights. It is trained on paired data with InfoNCE, not on unpaired data. | An own VL-JEPA-style model (D13). X = the VoxTell crop, possibly with the SDF latent code. Y = the normalised textual object. The vector says *what* the lesion is; position and measures stay as fields. |
| 10 | The whole pipeline is trained end to end | Possible in principle. | Three problems. Paired training data is needed (D14). Training the segmentation to agree with the semantic vector pushes it to find what the report names even where there is nothing, which conflicts with "not found". MDR verification is simpler with modules frozen and validated separately. Proposed: staged training with frozen interfaces; end-to-end only later, with negative cases (D19). |
| 11 | The node is a purely neural object in a "neural knowledge graph" | The graph is FalkorDB. | The node stores the latent code with decoder id and version, not a path to per-lesion weights. It stores one vector property per encoder (2026-10-02 correction), the normalised text and structured fields as the source of truth, and the raw report sentence as provenance. A graph neural network over these nodes is the future intuition model. |
| 12 | Digital twins: simulating how the lesion deforms under pressure or breathing | Needs a biomechanical model and tissue properties. | Out of current scope. |

**Precision is unchanged by this architecture.** None of the three stages raises the information limit measured on 2026-09-26. Same-patient precision still rests on DICOM geometry, rigid alignment on bone and sub-voxel estimation of high-contrast points.

## 2026-10-03 — Decisions, the encoder question, and the encoder bench

**Decided by the owner.**
- **D15 closed for the VoxTell weights:** they may be used.
- **D18:** a shared-decoder SDF with one latent code per lesion. Gaussian splatting is eliminated, including for visualisation. Fitting the decoder is deferred: until a need appears, shape is described by computed descriptors (volume, sphericity, elongation, surface-to-volume ratio, boundary irregularity), which need no training.
- **Training:** the only model Melampo trains is the diagnostic model. Report reading, segmentation and perception modules are used frozen. Consequence: D13 (an own VL-JEPA-style pairing model) is suspended, and no visual encoder is needed on this path for now (D12, D16). The Pillar-0 report-versus-image comparison had already left scope.
- **D19:** staged, with an explicit check between stages. When VoxTell and TotalSegmentator disagree, the node says so, the image part is ignored, and the vector is built from the report text only. The node records three distinct states -- concordant, discordant, not found or ambiguous -- and "visual evidence absent" is an explicit flag, never zeros, so the diagnostic model cannot read a missing density as a density of zero.
- **D20:** Melampo's normalisation cascade writes the VoxTell prompt. It has to be extended beyond disease concepts to fine anatomy, laterality, measures and consistency, with negation as an explicit field (encoders place "nodule present" and "nodule absent" close together, so negation must not live in the vector). Anatomy maps to TotalSegmentator class names, which is also what the agreement check needs. The prompt language, Italian or English from a fixed vocabulary, is still to be measured.

**Refinement of D19.** With nothing upstream trained, the remaining risk is training the diagnostic model on idealised inputs and deploying it on VoxTell masks of Dice 50. It is trained on the pipeline's real outputs, failures included, with the outcomes of the checks as inputs, and evaluated on a negative set (cases where the named lesion is not there). Cases discarded as unusable are kept as that negative set rather than thrown away; the discarding must not be decided by the model's own output, or only the easy cases remain.

**Corrections to the 2026-10-02 entries.**
- The list of "trained things" there was wrong: only the diagnostic model is trained. The SDF decoder and the VL-JEPA-style pairing are optional components, not obligations.
- Pillar-0 was mentioned as an encoder candidate only because the manifest ties its image encoder to the Qwen3 text space. That is not a requirement.
- What the repository holds for the "Nemotron and Gemma" normalisation, checked 2026-10-03: the cascade in `memory/concept_normalisation.py` (lexical match with curated synonyms from HPO and UMLS, then SapBERT, then an injected language model) links phrases to graph concepts; the Gemma 4 contracts are disabled by default and run no live inference; Nemotron-Parse serves document extraction. No Nemotron or Gemma normaliser for imaging terms is wired in.

**D21: the text encoder. Not decided; this is what was established.**
- There is no universal embedding space. Each encoder defines its own, and vectors from two encoders are not comparable even at the same dimension. A swap cannot be avoided by a clever choice; it can only be made cheap.
- Under discussion: one encoder with open weights, pinned in `melampo-assets.yaml`; its identity kept at index level (one index per encoder, queries from a different encoder refused) rather than on every vector; vectors regenerated from the stored source (the normalised object, later the crop and mask) by an automatic migration -- build the new index, recalibrate thresholds, run a regression gate, generate the change-control report, obtain human approval, switch the pointer, keep the old index for rollback. Retraining is needed only where a learned model takes vectors as input; where vectors are only compared, rebuilding the index and recalibrating the thresholds is enough. The intuition model is proposed to learn at the level of concept codes and measures, with a case memory, and to see vectors only through a small per-encoder adapter.
- Ruled out as the encoder: gpt-oss-120b (generative, not trained to produce embeddings; tying the vector space to the reasoning model would force a migration with every change of root model). Hosted-only encoders (OpenAI text-embedding-3, Voyage, mistral-embed) cannot be pinned and are kept in the bench only as references. Anthropic offers no embedding model (its documentation says so and recommends Voyage AI).
- Shortlist: Qwen3-Embedding-8B (Apache-2.0, over 100 languages, 70.58 on MTEB multilingual at release, 4,096 dimensions, which is FalkorDB's limit). Non-Chinese alternatives asked for by the owner: IBM Granite Embedding Multilingual R2 (Apache-2.0, Italian among 52 enhanced languages), EmbeddingGemma-300M (Gemma terms, to verify), NVIDIA Nemotron 3 Embed 1B (licence to verify), multilingual-e5-large (MIT). Several western encoders are fine-tuned from Chinese base models; provenance is checked model card by model card.
- For the legal review: the encoder receives only the normalised object, never the raw report, and an automatic check must show the normalised object carries no names, dates or places. The raw report is processed only locally or under a data-processing agreement. Pseudonymised data remains personal data under the GDPR.

**The encoder bench** (`src/melampo/evaluation/encoder_bench.py`, `scripts/run_encoder_bench.py`, `.github/workflows/encoder-bench.yml`, gold data in `data/encoder_bench/`). It decides D17 and D21 by measurement.
- *Concept linking:* 189 Italian anatomical phrases (formal, abbreviated, colloquial) must retrieve the right one of 63 structures, identified by TotalSegmentator class names (`total` and `liver_segments`, read from totalsegmentator 2.18.0). The pool is embedded with Italian labels, English labels and both, which also informs D20.
- *Hard triplets:* 68 cases where a paraphrase must score above a near-miss that differs in laterality, adjacent organ, liver segment, vertebral level, negation, units, comparison with a prior exam, or consistency. Reported per category, with 95% Wilson intervals.
- All phrases are synthetic and hand-written, so nothing patient-related is sent to a hosted endpoint. The result is a screen that removes unsuitable encoders, not a clinical validation; the sets are small and a gap inside the interval is not a gap.
- **Not yet run live.** The session that built it could not reach OpenRouter and held no key. The code is tested offline (23 tests; the whole suite passes). Run it from GitHub Actions (manual, needs the `OPENROUTER_API_KEY` secret). Encoders that are not on OpenRouter (Granite R2, EmbeddingGemma) need a local backend, which is not built.

## Open decisions (updated 2026-10-03)

| # | Decision |
|---|---|
| D1 | Lesion isolation: **VoxTell chosen by the owner (2026-10-02), weights approved for use (D15, 2026-10-03)**; TotalSegmentator kept as the organ/segment cross-check; on disagreement the image part is ignored |
| D2 | Anatomical vocabulary and Italian labels |
| D3 | Brain: an atlas used only to label, never to measure (an exception to "no atlas"), and which atlas |
| D5 | Geometric morphing method: shared-decoder SDF latent codes proposed (2026-10-02, item 7) |
| D6 | Acquisition requirements for declaring maximum precision (now implemented as `measurement_precision` levels; thresholds to confirm) |
| D7 | Who confirms lesion masks and text-to-image pairings |
| D8 | Surface anatomy (nasal subunits, fingertip zones): outside CT/MRI scope, or a separate photographic pipeline |
| D9 | Where the GPU runs |
| D10 | Archive of cases with confirmed diagnoses for scenario B |
| D11 | Withdraw the "unguided reading always in parallel" requirement (proposed), or keep it |
| D12 | Pillar-0: no role on the current path (2026-10-03). The image is described by measures and shape descriptors; no visual encoder is needed while D13 is suspended |
| D13 | **Suspended (2026-10-03).** Own VL-JEPA-style vector model: only the diagnostic model is trained, so the pairing model is not built unless a need appears (absorbs the former D4) |
| D14 | Validation and training data: where exams with masks and training pairs come from, legal basis (GDPR art. 9), compatibility with the phase-one rule (synthetic or de-identified data only) |
| D15 | Legal review. **Closed for the use of the VoxTell weights (2026-10-03).** Still for the review: intended use (research, in-house under MDR Art. 5(5), or placing on the market), the CC BY-NC 4.0 LICENSE text and its added terms, NC, SA, HAI-DEF and FAIR materials, and hosted-API use with de-identification of anything sent out |
| D16 | Visual encoder per modality: not needed on the current path (2026-10-03); reopens only if a visual vector is wanted |
| D17 | Text encoder for normalised concepts: to be decided by `encoder_bench` (2026-10-03); Qwen3-Embedding-8B is the reference candidate, with non-Chinese alternatives in the shortlist |
| D18 | **Decided 2026-10-03:** shared-decoder SDF with one latent code per lesion; Gaussian splatting eliminated. Fitting the decoder deferred; shape descriptors meanwhile |
| D19 | **Decided 2026-10-03:** staged, with an explicit check between stages; on VoxTell and TotalSegmentator disagreement the vector comes from the report text only |
| D20 | The normalisation cascade writes the VoxTell prompt (**decided 2026-10-03**); prompt language, Italian or English from the vocabulary: to be measured (linking pools in `encoder_bench`, then VoxTell Dice on the four prompt types) |
| D21 | Open (2026-10-03). Encoder to pin: chosen by `encoder_bench`. Proposed: identity at index level, automatic migration with a regression gate and human approval; see the 2026-10-03 entry |

## 2026-10-03 (later) — Encoder bench: first results and the local backend

First run through OpenRouter (7 encoders, 63 structures, 189 phrases, 68 triplets).
Screening order: gemini-embedding-001 0.802, qwen3-embedding-8b with instruction
0.779, qwen3-embedding-8b 0.748, openai text-embedding-3-large 0.747,
qwen3-embedding-4b 0.742, bge-m3 0.714, mistral-embed 0.707. The top two are not
separable (95% intervals overlap). No encoder is reliable on size/unit
(0-25%), consistency (17-50%) or negation (50-75%): those stay structured fields,
never carried by the vector. Indexing Italian and English labels together lifts
linking to 83-89% for every encoder. The instruction prefix helps Qwen3-8B on
English queries only.

Extension: the bench now also covers every OpenRouter embedding model that can
handle Italian (Gemini embedding 2, OpenAI 3-small, Voyage 4/large/lite,
Nemotron 3 Embed 1B, Perplexity pplx-embed 4B/0.6B, Liquid LFM 2.5 350M,
multilingual-e5-large) and, through a local sentence-transformers backend run on
the Actions runner, IBM Granite Multilingual R2 (311M, 97M), EmbeddingGemma-300M,
Snowflake Arctic-Embed-L v2 and Nomic Embed v2 MoE. Models that document
document-side or symmetric-task prefixes get them (e5, EmbeddingGemma, Nomic).
English-only models are left out deliberately. Run: Actions > "Text encoder bench",
scope `both`; EmbeddingGemma needs the secret `HF_TOKEN` of an account that accepted
the Gemma terms. D17/D21 stay open until these results are read.

Added the same day: Cohere Embed v5 Pro and Fast (direct API, `COHERE_API_KEY`;
the two share one embedding space, so one can index and the other query; roles are
sent as `input_type`). NV-Embed-v2 is opt-in only (`--backend local --roster
nv-embed-v2`): 7.9B parameters (no standard runner can hold it) and CC-BY-NC-4.0,
which this project could not ship, so it is measured only for reference and on
the owner's own hardware.
