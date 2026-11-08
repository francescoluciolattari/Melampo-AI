# Labels of the router probe (8 November 2026)

`labels.txt`: one line per text, `item_id<TAB>kind+kind<TAB>note`, with the kinds of `report_router.KINDS`
(laboratory, protein_electrophoresis, microbiology, pathology, imaging_report, clinical_note, other). Written
before any reading, from the text alone: the kinds the text contains, not the specialty it is filed under. The
texts (MTSamples, PARROT) are not committed. `scripts/router_probe.py` explains the criteria.
