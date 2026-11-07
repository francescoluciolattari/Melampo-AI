# Validation and data lineage

*English · [Italiano](../it/VALIDATION.md) · as of 4 November 2026*

What Melampo claims is what its probes measured, on samples nobody tuned the system on, against criteria
written before the run. Today nothing is certified: the gold set of the radiologists is not yet labelled, and
the labels used so far are written by one labeller (Claude) and wait for a radiologist's review.

## 1. Method

1. **A fresh sample for every measure that counts.** Reports already seen during development never count again:
   each sample is drawn with its own seed and excludes every report of the earlier samples
   (`scripts/owner_probe.py sample --seed …`).
2. **Labels before any reading.** The labeller sees the sentence, the marked words, the linked class and the
   exam, never the readers' answers. Doubtful labels end with "?" and go to a radiologist.
3. **Criteria before the run**, in this file or in the pull request that adds the probe.
4. **A change that fails its criterion is switched off**, and the result is written here. A run that loses more
   than 5% of the model answers (`lost_answers`) does not count.
5. **No tuning toward the reference.** A sample used to find an error becomes development data; the fix is
   measured on the next fresh sample.
6. **Two readers that must agree.** A deterministic reader and the language models; two models count as one
   stream; a person decides on contradiction. Agreement is measured on what changes the outcome.

## 2. Data lineage

Texts of public corpora are never committed: only labels, seeds and the scripts that draw the samples. The
licence and status of every external asset (ontologies, corpora, model weights) are in
[`melampo-assets.yaml`](../../melampo-assets.yaml); Python packages are in [`uv.lock`](../../uv.lock) and in the
generated [SOUP list](../registers/soup.md).

| Source | Used for | In the repository |
|---|---|---|
| PARROT v1.0 (radiology reports in several languages, with English translations; CC BY-NC-SA 4.0) | finding owner, statement and claim samples; HPO samples | labels in `data/gold/*_2026-1*` |
| E3C (Italian and English clinical cases) | silver gold set, role probe | labels in `data/gold/silver_2026-10-12` |
| IU X-ray and Rad-SpRL (spatial roles, CC BY 4.0) | role probe, landmark role | labels in `data/gold/rad_sprl_2026-10-12` |
| CRAFT, MedMentions | external check of the linker | the MedMentions test split is frozen: `data/linking/frozen_test_split.json` |
| MSD Task06 Lung, LIDC-IDRI | imaging probes on Kaggle | notebooks in `notebooks/` |
| HPO, MONDO, DOID, LOINC, CIPI, ICD-9-CM; UBERON, PATO | reading, linking, graph | DVC pins in `data/*.dvc`; UBERON and PATO downloaded by the workflows at their latest release |

**Samples of the finding-owner question** (all English PARROT items, 200 each unless said):

| Sample | Seed | Excludes | Labels | Status |
|---|---|---|---|---|
| HPO development | default | — | `data/gold/hpo_silver_2026-10-31` | seen |
| HPO held out | 2026103102 | HPO development | `data/gold/hpo_heldout_2026-10-31` | seen |
| owner (209 items, 9 Italian) | 20261101 | both HPO | `data/gold/finding_owner_2026-11-01` | seen |
| statement | 20261102 | the above | `data/gold/statement_2026-11-01` | seen |
| claims | 20261103 | the above | `data/gold/claims_2026-11-02` | seen |
| claims2 | 20261104 | the above | `data/gold/claims_2026-11-02b` | seen |
| claims3 (174 items) | 20261105 | the above | `data/gold/claims_2026-11-03` | seen |
| claims4 (121 items, up to 16 per structure) | 20261106 | the above | `data/gold/claims_2026-11-04` | seen |

## 3. The gold set of the radiologists

The certificate the project aims at: "error ≤ 1% on accepted links, 95% confidence" (Clopper-Pearson: 299
accepted links with 0 errors, 628 with ≤ 2, 1,049 with ≤ 5, 1,941 with ≤ 12). Target: 2,000–2,300 mentions from
real pseudonymised reports, one mention per report, two radiologists labelling blind and alone, a third deciding
the disagreements, the set frozen once labelled. How to label is in the
[radiologists' guide](RADIOLOGIST_GUIDE.md); the commands are `scripts/gold_set.py sample | check | agree | merge
| evaluate`. After go-live, a blind 2–5% sample of accepted links keeps the bound up to date.

### The labelling convention (6 November 2026)

Until the radiologists label, the samples are labelled by one labeller (Claude). Doubtful cases are decided by
written rules, each with its source, not by the labeller's judgement of the moment; the rules are fixed before a
sample is labelled and never corrected by looking at the models' answers.

| Case | Rule | Source |
|---|---|---|
| a variant (accessory spleen, Riedel's lobe) | a variant HPO holds as a phenotype of a structure is a finding of that structure | HPO (e.g. HP:0001747) |
| mesentery, parietal pleura, lobar bronchus | a structure is part of another only if UBERON says so (`part_of`) | UBERON |
| what the referral or the clinical question says | an indication, never a current finding | the report's sections (LOINC 18785-6) |
| an embolization, a resection | a procedure: history | — |
| an implanted valve, a tube, a stent | a device, even when "its shadow is seen" | — |
| "at the level of", "in the projection of" | they place, unless the sentence says the structure is touched | Rad-SpRL spatial indicators |

A label that the rules still leave open is marked "?". **The criteria are computed on the items without "?"**
(`by_set.<sample>.criteria.without_doubt`); the "?" items are reported apart, with their outcome
(`criteria.all`, `claims_kinds.for_a_radiologist`).

## 4. Results

Newest first. Detail of each run: the document named in the last column, in the tag `docs-archive-2026-11`
(path `docs/linker/…`) or in the pull request.

| Date | Question | Result | Decision | Detail |
|---|---|---|---|---|
| 2026-11-05 | Claim by claim, sample `claims4` | valid (0 answers lost). 3 of 7 attributed silently, all accessory spleens that HPO holds as a phenotype of the spleen (HP:0001747): most likely my labels; 0 of 74 findings set aside; 15 of 114 to a person (13.2%); 7.1% attached otherwise, 4 of them sentences of the referral read as current findings. 38 measures compared with a cited reference, 4 sent to a person by the reader's recomputation, all rightly | **off**; the report read as a document (sections, conclusion, sentence before), indication and procedure as claims, devices as devices, the size of a lesion never compared with the organ's reference, ambiguous placing words read by the models only; the labelling convention, criteria on the items without "?" | this pull request |
| 2026-11-04 | Claim by claim, sample `claims3` | valid (0 answers lost). 4 of 17 attributed silently (criterion 1 failed), 0 of 110 findings set aside, 33 of 157 to a person (21%, criterion 3 failed), 2.4% attached otherwise, none a normal state (claims2: 4 of 5). To a person: 13 measures and tables, compared by the models with values they remember; 2 rib fractures read as absences (the negation was read on the last word); 1 from extending the reader's check; 5 "?" labels; the rest older reader readings and model disagreements | **off**; negation read on the head of the observation; the reference library: a measure is compared only with a cited reference value, recomputed by the reader | this pull request |
| 2026-11-03 | Claim by claim, sample `claims2` | valid (0 answers lost). 5 of 17 attributed silently (criterion 1 failed), 0 of 134 findings set aside, 13 of 182 to a person (7.1%), 3.0% attached otherwise. Of the 5: two where the model said "its own" while its own words only placed the finding ("beneath", "in the topography of"); two where the question told the models that "at the level of" reaches the vertebra; one "?" label ("lucency at the base of the heart - pneumothorax?") | **off**; corrections: a model whose own words only place what it calls the structure's own contradicts itself and goes to a person, "beneath", "underneath" and "in the topography of" read as placing, the question no longer says that a level reaches the structure. On the same, already seen run: 3 of 17 silently, 0 removed, 13 to a person. Sample `claims3` | this pull request |
| 2026-11-02 | Claim by claim, sample `claims` | run not valid: credit ran out (Nemotron lost 47 of 200 answers). On the 153 items with both answers: 2 of 5 attributed silently, 0 right findings removed, 21% to a person, 4.3% attached otherwise | **off**; corrections (parts read with UBERON, question on the marked words); sample `claims2` | `affermazione_per_affermazione_2026-11-02` |
| 2026-11-02 | Role probe after the landmark fix | "site or not": 0 items with reader and model agreeing on a wrong outcome; Gemma fine roles 86.0% (simple) and 87.6% (method), was 89.4% and 91.5% | kept; the Rad-SpRL stratum must be relabelled | `che_cosa_dice_la_frase_della_struttura_2026-11-01` |
| 2026-11-02 | Statement types, sample `statement` | 3 of 15 attributed silently, 0 removed, 20% to a person; when accepted the type was always right (0 of 137) | **off** | `che_cosa_dice_la_frase_della_struttura_2026-11-01` |
| 2026-11-01 | Finding owner v1, fresh sample | 1 of 24 attributed silently (was 11 of 26 on seen cases), 0 of 184 right findings removed, 19 to a person (10.3%, criterion 10%) | kept on in ingestion; criteria 1 and 3 missed by one case each | `di_che_cosa_parla_la_frase_2026-10-31` |
| 2026-10-31 | Finding → HPO term, held-out sample | integrated arm 90.1% safe in English, 90% in Italian; threshold 95% | not in the graph | `dal_reperto_al_termine_hpo_2026-10-31` |
| 2026-10-30 | Measurement tolerance | taken from the variability between radiologists (LIDC), replacing max(2 mm, 10%) | in use | `tolleranza_dai_radiologi_2026-10-30` |
| 2026-10-16 | Imaging on Kaggle (VoxTell + TotalSegmentator, MSD lung, 5 cases) | first end-to-end run; second reading of the images | basis for the image interpreter | `primo_probe_kaggle_e_seconda_lettura_2026-10-16` |
| 2026-10-14 | Disagreements in English (role probe) | 10% (Nemotron 3 Ultra) and 13% (Gemma 4) of 484 items to a person | option b: agreement on what changes the outcome | `disaccordi_inglesi_2026-10-14` |
| 2026-10-13 | Role probe, complete run | 963 items, five strata, three models | method kept | `role_probe_run_completo_2026-10-13` |
| 2026-10-08 | External check of the linker | CRAFT 1,352 judged, 13 errors (99.04%); MedMentions 768 judged, 51 errors | error analysis | `analisi_errori_controllo_esterno_2026-10-08` |

## 5. Next measure: a chest CT sample from CT-RATE, and PhysioNet

PARROT has no varied sample left (62 items from 42 reports, mostly brain and aorta). The next measures:

- **CT-RATE** (chest CT, CC BY-NC-SA 4.0, public, usable with OpenRouter for research): a sample drawn by seed and
  labelled with the convention before any reading, with the same five criteria computed on the items without "?".
  It covers the chest only: it measures the corrections, not the whole system.
- **PhysioNet** (credentialed access requested): RadGraph-XL, radiologists' labels on chest X-ray, chest CT,
  abdomen-pelvis CT and brain MR; MIMIC-IV-Note for variety; RadGraph for the agreement between two radiologists;
  RadCoref for references across sentences. Run only inside a protected environment with models of our own or a
  service verified with PhysioNet, never on GitHub runners, never with OpenRouter until PhysioNet agrees; only
  aggregate numbers come back to the repository.

## 6. Known limits

- All labels so far are one labeller's; the "?" labels and the lists of findings of another structure wait for a
  radiologist.
- The reader does not recognise "nodule" or "free fluid" as abnormalities (they also name structures in UBERON);
  the negation of "not enlarged … with cysts" reaches the cysts.
- No Italian report of PARROT is left unused: the Italian side needs its own sample.
- The Rad-SpRL stratum of the role probe was labelled before the landmark role existed.
