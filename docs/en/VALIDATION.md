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
| ctrate (200 items, CT-RATE chest CT) | 20261107 | — (another corpus) | `data/gold/ctrate` | **fresh, labelled with the convention, not yet run** |

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
| "may", "cannot be excluded", a differential (7 November 2026) | the doubt of the report: the finding with the claim `uncertain` (`finding+uncertain`), kept in the diagnostic path | the principle of doubt |

A label that the rules still leave open is marked "?". **The criteria are computed on the items without "?"**
(`by_set.<sample>.criteria.without_doubt`); the "?" items are reported apart, with their outcome
(`criteria.all`, `claims_kinds.for_a_radiologist`).

### The principle of doubt (7 November 2026)

What is not certain is not a question for a person, who knows no more than the report says: it stays in the
diagnostic path as a **finding in doubt** (verdict `doubt`), with every reading beside it, never set aside. Other
reports, the history or a new exam may close it later; within the report, the conclusion or the reader's
unambiguous knowledge (HPO, PATO) may close a doubt of reading. A finding is certain when two sources of different
mechanism hold it: the two models agreeing (one stream) and the reader not contradicting them, or one model's
reading confirmed by the conclusion or the reader. A person sees only a **discordance of the document**: the
context and the sentence (a finding read in the referral), the sentence and the conclusion, "its" pointing to
another organ, a measure and its cited reference. The doubt the report itself states is kept with the finding
(`clinical`); our reading never closes it.

The kinds of doubt: `reading` (the models read differently), `state` (a model cannot tell it from the usual),
`site` (a reading gives it to another structure: it stays where the linker put it), `one_reading` (one answer
only), `reader` (the reader reads it otherwise than the models), `clinical`.

## 4. Results

Newest first. Detail of each run: the document named in the last column, in the tag `docs-archive-2026-11`
(path `docs/linker/…`) or in the pull request.

| Date | Question | Result | Decision | Detail |
|---|---|---|---|---|
| 2026-11-08 | Router probe, the cues alone (labels written first, 270 texts) | 20.4% of the texts with an interpreter missed (95% upper bound 24.8%): the table reader for 46 clinical notes the cues read as imaging reports only (they quote an MRI or an x-ray), the pathology reader for 9 of the 11 pathology texts; 14% with an interpreter added | the baseline the local model is measured against; the run with the models follows | this pull request |
| 2026-11-08 | Why asides are attached as certain, and why the reading that leads a doubt is often wrong (runs already seen) | Of the 11 asides attached as certain on claims2/3/4 (with the "?" ones): 4 are accessory spleens, findings of the spleen by the convention of 6 November, labelled before it (not relabelled: the samples are seen); 7 are placed by words of place ("at the level of", "at the base of", "in the topography of", "projected on") or said of another subject ("its origin from the aorta" of the celiac trunk): the convention that places them was given to the labeller, not to the models. Of the doubts, the PARROT answers were made with the questions before "differs" and the measure apart (normal states and measures listed as observed); on CT-RATE, with the current questions, the leading reading is right in 5 of 6; in the doubts of the reader its reading was often the right one. The label's reading is among those a doubt keeps in all but 1 of 448 items (0.2%) | the models are given the convention's rule of place and of variants (`CLAIM_STEPS`, measured on `ctrate2`); a piece only of a measure is a measure; the reader's reading is kept in the doubt and none leads; criterion 6 counts a doubt without the label's reading as an error (under 2%) | this pull request |
| 2026-11-07 | The principle of doubt, on runs already seen (the decisions made again on the stored answers) | CT-RATE (199 items): 0 to a person (95% upper bound 1.5%), 1.0% of the certain readings attached otherwise (bound 3.2%), 0 asides attached as certain, 0 findings set aside, 3.0% in doubt, 24 doubts of the report kept with their finding. PARROT claims2/3/4: to a person 1.1%, 0.6%, 4.6% (was 17.5%, 29.5%, 26.7%), certain readings attached otherwise 1.9%, 1.6%, 2.1%, asides attached as certain 2, 2, 2 (both models agreeing wrongly; on claims4 two had gone to a person only because the referral was read as reaching the end of the report), in doubt 9.9%, 17.6%, 4.6%, the leading reading of a doubt wrong in 13 of 18, 21 of 28, 5 of 5. Not one criterion is shown under 2% with 95% confidence on these samples | the principle of doubt **on** in the claims reading (itself still off in ingestion); a section before the findings is certain only to its line or first sentence. Sample `ctrate2` (300 items) with the criteria under 2% | this pull request |
| 2026-11-07 | Claim by claim, sample `ctrate` (chest CT, CT-RATE) | valid (0 of 400 answers lost); all five criteria met for the first time: 0 of 1 attributed silently, 0 findings set aside, 16 of 198 to a person (8.1%), 1.1% attached otherwise. Of the 16: 7 model disagreements, 5 of them nested (one model reads a piece the other leaves out); 9 from reader checks that recalled a sense outside the report (the amphibian septomaxilla for "corpus", the embryonic arch arteries for "aortic arch") or read two parallel places as one | context-dependent senses: the reader recalls no sense of another species and the embryo's only in a report about development (corrected 9 November 2026: it had excluded every embryonic structure), the conclusion and "its" count only when they name another organ, "its" only with a pronoun, parallel places each keep their finding; the reader as third reader of the piece nested readings differ on (**proposed**, it changes "the models must agree"). On the same, already seen run: 5 of 198 to a person (2.5%), nothing else changed. (Corrected the same day: the claims readings of claims2/3/4 cannot be replayed from the cache, their prompts having changed since; they are replayed from the stored answers in the next row.) To be measured on a second CT-RATE sample | this pull request |
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

  The sample (7 November 2026): 200 items from 101 reports, 182 with a finding, 17 to keep without one, 1 to set
  aside, 1 doubtful. Criteria, fixed before the run, on the 199 items without doubt: (1) the item to set aside is
  not attached silently; (2) none of the findings is set aside, at most two stated; (3) at most 10% of the items to
  keep go to a person; (4) at most 5% of the accepted items are attached otherwise than the label; (5) at most 5%
  of the answers lost. With one item to set aside, criterion 1 says almost nothing here: the sample measures
  above all what the reading does to real findings (2, 3, 4).
- **CT-RATE, second sample** (`ctrate2`, 7 November 2026): 300 items, seed 20261108, from reports the first
  sample did not use, at most 24 per structure, labelled with the convention (the doubt of the report included)
  before any reading. Criteria, fixed before labelling, on the items without "?", every error share under **2%**,
  the rate of clinically significant errors of radiology reports themselves (dictation 2.3%, speech recognition
  6–9.6% in du Toit et al., BMC Medical Imaging 2015), each with its one-sided 95% upper bound (Clopper-Pearson):
  (1) no item to set aside attached as certain; (2) no finding set aside; (3) at most 2% of the items to a person,
  each a discordance of the document; (4) at most 2% of the certain readings attached otherwise than the label,
  closed doubts included; (5) at most 2% of the answers lost; (6) at most 10% of the items in doubt (a bound of
  usefulness, not of error), a doubt that does not keep the label's reading among its readings counted as an
  error (under 2% of the items), the closed doubts wrong under 2% of the closed ones. With 300 items, 0 errors bound the share at 1.0%, 2 errors at 2.1%.
- **Router probe** (`scripts/router_probe.py`, workflow `router-probe`, 8 November 2026): does the local model
  help the report router say the kinds of a text? Public texts of known kinds: MTSamples (6 per specialty,
  English) and PARROT (20 Italian, 10 other languages), seed 20261108, labelled (`data/gold/router/labels.txt`)
  before any reading. Each model alone against the cues alone. Criteria, fixed before the labels: with the model,
  at most 2% of the texts with an interpreter missed (a kind of the label whose interpreters are not run), with
  its 95% upper bound; at most 2% of the model's answers unusable. Then `MELAMPO_REPORT_ROUTER=model` may be
  turned on; otherwise it stays off. Interpreters added that no kind needs are reported (a cost, not an error).
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
