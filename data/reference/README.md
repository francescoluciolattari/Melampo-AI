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

A row is `to_verify` until a person has checked it against the page, and only a `reviewed` row (by a
radiologist) is fit for clinical use. The first rows were read by Claude through a summarising web reading:
they must be checked against the pages before they are trusted.
