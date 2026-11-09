# The reference library

What Melampo compares a measure with (4 November 2026). Two parts:

- **`values.yml`** (in the repository): reference values, one per row, each with its structure, aspect,
  technique, population, source (URL, section, reference number), the source's licence and its status. Only
  facts with their citation are here.
- **the paragraphs of the sources** (not in the repository): a local JSON-lines copy built with
  `scripts/reference_passages.py`, read from `MELAMPO_REFERENCE_PASSAGES`. Paragraphs only, at most 250 words,
  never a whole chapter, each with its source, section and licence, and the sections it links to.

Sources for research, for now: open-access literature and Radiopaedia (CC BY-NC-SA 3.0, modified: not for a
commercial product). Classic textbooks only through an institution with lawful access (art. 70-ter of the
Italian copyright law) or a licence. Every source used is listed in `melampo-assets.yaml`; a product replaces
the non-commercial ones.

A row is `to_verify` until the words it cites (`quoted`) have been found on the page itself, word for word, by
the weekly check of the sources index (`scripts/sources_index.py check-quotes`, workflow `sources-index`); then it
is `verified`, with the date and the fingerprint of the page. A value holds because its source says it: there
is no status for a person's memory (8 November 2026). The sources themselves, and how each may be reached, are
in `sources.yml` (the index of the sources), kept up to date by hand (`scripts/sources_mask.py`) and by the
project's weekly search.

## State on 8 November 2026

29 rows, all `to_verify`. Each row now carries `quoted`: the sentence of the page it rests on, as a web reading
quoted it when asked for the exact words. Reading them so corrected three rows (a threshold without a reference
number cited as ref 2, a missing ref 3, the section of the IVC threshold) and confirmed the others. A quotation
read through a web reading may still be wrong: the check on the page decides.

Added: the ascending aorta (dilatation above 40 mm) and the descending thoracic aorta (aneurysm above 40 mm), the
adrenal limbs and bodies, the thyroid lobes, isthmus and volume.

Not yet there, and why:
- the oesophageal wall: the Radiopaedia page gives no number;
- the prostate and the urinary bladder: their pages could not be read from here;
- the attenuation of the liver (steatosis) and of other tissues: the table has no unit for Hounsfield units yet;
- the cardiothoracic ratio and the pulmonary trunk: a ratio has no unit in the table, and the linker has no class
  for the pulmonary artery.
