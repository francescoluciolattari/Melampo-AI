# Gold set: guide for the radiologists

*English · [Italiano](../it/RADIOLOGIST_GUIDE.md) · version of 10 October 2026, landmark role and several roles
31 October, claims 2 November 2026*

Two radiologists label **alone and blind** the same mentions; a third decides the disagreements. The system is
never shown: a wrong suggestion seen before reading steers the reading (Dratsch et al., *Radiology* 2023: from
about 80% to under 20% accuracy).

## 1. What you do

Each row of your sheet (`annotator_A.csv` or `annotator_B.csv`, opens in Excel) is a **mention** in a sentence.
Read `sentence` and `mention` and fill **four columns**: `structure`, `relation`, `role`, `span_text`; `claims`
when the sheet has it. `side_in_text` and `note` are optional. Do not change any other column.

| Column | What to write |
|---|---|
| `structure` | the id from `valid_structures.txt`, or `NONE_IN_CLASSES` (a real structure not in the list: uterus, appendix, ovary), `NOT_ANATOMY` (the word does not name a structure here: the "T2" sequence, "heart rate", "American Heart Association"), `AMBIGUOUS` (the text does not say which: side missing in a paired structure, "iliac", an ambiguous abbreviation) |
| `relation` | only for a structure: `equal` (the structure or a synonym), `part_of` (a part: sigmoid → colon, femoral head → femur), `contour_of` (its outline on a radiograph), `approx` (a usage synonym, not identical: hemipelvis ≈ iliac bone) |
| `side_in_text` | the side written in the text (right, left, bilateral, none); never inferred from the rest of the report |
| `note` | optional, for doubts |

## 2. `role`: what the word does in the sentence

**For a structure** (an id or `NONE_IN_CLASSES`):

| `role` | When | Example |
|---|---|---|
| `structure` | the sentence speaks of the structure itself | "enlarged liver" |
| `landmark` | the structure only places something else: a node, a fluid, a mass or another organ near, along, around, between, outside or towards it | "nodes along the aortic arch", "fluid between the small-bowel loops" |
| `procedure_site` | the site of a procedure | "liver biopsy", "brain MRI" |
| `device_site` | the site of a device | "coronary stent", "atrial pacemaker" |
| `source_of` | where a substance or cells come from | "liver extract" |

**For `NOT_ANATOMY`:**

| `role` | When | Example |
|---|---|---|
| `inherent_location` | the structure is what a measure, function or process is of | "heart rate", "liver function" |
| `inside_a_name` | the word is part of the name of something else | "Dallas Heart Study", "thyroid peroxidase" |
| `not_a_body_site` | part of something that is not a body | "heart of Maroilles cheese" |
| `other` | none of these; explain in `note` | |

For `AMBIGUOUS` the column stays empty.

**In doubt between `structure` and another role:** would the reader of the report think of that part of the
patient's body? **In doubt between `structure` and `landmark`:** does the sentence say something of this
structure, or use it to say where something else is? The test is what the measure and the character of the
sentence belong to, not the preposition: "plaques along the femoro-popliteal axis" are in the artery wall
(`structure`).

**Several roles together** are written joined by `+`, the main one first: for "wall thickening of the colon near
the splenic flexure", the mention "splenic flexure" is `structure+landmark`.

## 3. `span_text`: the whole name

Usually empty. Fill it **only** when the mention is part of a longer name that contains it, writing the whole
name as in the text: "Heart" → "Dallas Heart Study". If unsure where the name ends, leave it empty and write in
`note`.

## 4. `claims`: what the sentence states of the structure

Write everything the sentence states **of it**, with its own words, adding nothing you know. Do not choose one
label: mark every claim present, joined by `+`.

| Claim | When | Example |
|---|---|---|
| `finding` | the sentence observes something different from the usual in it or in a part of it, even mild or only probable | "pancreas **hyperechoic**", "**likely** calcified cartilages" |
| `involved` | a finding of another structure reaches it: extends into it, compresses it, narrows it, or lies at the level or in the joint it forms | "disc herniation **at the L5-S1 level**", "the mass **molds** the pancreas" |
| `normal` | it is said to be normal, or an abnormality is said to be absent | "**not enlarged**", "**no stones**" |
| `measure` | a measure of it | "measures **10.5 cm**" |
| `history` | a fact of its past | "**history of** thyroid surgery" |
| `device` | a device in it | "**intubation tube** in the trachea" |
| `not_assessable` | it cannot be assessed | "the wall **is not assessable**" |
| `aside` | what is observed belongs to another structure, and this one only says where it is | "nodes **along** the aorta", "a collection **between** the flank and the psoas" |
| `nothing` | it is named without anything stated of it: a title, the exam's name | "THYROID ULTRASOUND" |

**Said, not inferred.** A value you know to be normal ("IMT 0.5 mm") is only a measure if the sentence does not
say "normal". "Hyperechoic and not enlarged" are two claims: `finding+normal`. **Near is not contact:** "between",
"adjacent", "along" say where it is (`aside`); "extends into", "infiltrates", "compresses" say that it reaches it
(`involved`).

## 5. Rules that always hold

- Label the structure **named**, not the disease or its presence: "absence of the right kidney" → right kidney.
- A lumen or a space is not the organ: "renal fossa", "oesophageal lumen" → `NONE_IN_CLASSES`.
- Wall, parenchyma, body of an organ → the organ, with `part_of`.
- A structure named only as the modifier of a measure or a test ("heart rate", "liver function tests",
  "thyroid-stimulating hormone") → `NOT_ANATOMY`; the same word among the findings of an imaging report names
  the structure.
- "Hip" without "bone" is the hip region or joint: label the closest class (`hip_left`, `hip_right`) with
  `approx`; "hip bone", "iliac bone", "os coxae" → `equal`.
- Side not written for a paired structure → `AMBIGUOUS`, even if the rest of the report suggests it. If the side
  is only in the exam's name, write "side from the exam" in `note`.
- A word inside a proper name (institution, study, scale, molecule, device) → `NOT_ANATOMY`.
- **If unsure: `AMBIGUOUS`.** The system's abstention on those cases is correct and is measured as such.

## 6. The procedure

1. **Alignment session**, once, before the 2,000 mentions: 40 sentences from public corpora
   (`python scripts/gold_set.py alignment … --n 40`). Each radiologist fills the sheet alone (about 40 minutes);
   then agreement and kappa (`gold_set.py agree`), every disagreement discussed, the reference reading opened only
   after filling. If the kappa of `structure` or `role` stays below 0.8, the missing rule is written and the
   session repeated with 40 new sentences. The session is not part of the study.
2. **Labelling:** each radiologist alone, without seeing the other or the system. Check the sheet with
   `gold_set.py check`.
3. **Agreement and adjudication:** `gold_set.py agree` writes the disagreements; the third reviewer decides them;
   `gold_set.py merge` writes the final labels.
4. **Freeze:** the labels and the commit of the system evaluated are frozen; nobody looks at them during
   development.

## 7. What you hand in

The two filled sheets, never changed after delivery. No report leaves the hospital; the text is pseudonymised
at the source.
