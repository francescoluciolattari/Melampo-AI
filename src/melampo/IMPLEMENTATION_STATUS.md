# Implementation Status

This document tracks the current status of the Melampo baseline scaffold.

| Area | Purpose | Status |
|---|---|---|
| clinical | standards and clinical schemas | baseline scaffold in place |
| data | ingestion, normalization, FHIR and DICOM placeholders | baseline scaffold in place |
| models | modality and control-layer placeholders | partial scaffold in place |
| memory | episodic, semantic, graph, retriever | baseline scaffold in place |
| reasoning | workspace, differential, critique, escalation, policy stack | baseline scaffold in place |
| training | curriculum, meta-learning, EWC, replay, trainer | baseline scaffold in place |
| evaluation | validation, falsification, metrics, calibration, risk-coverage | baseline scaffold in place |
| orchestration | routing, contracts, bootstrap, MCP, A2A | baseline scaffold in place |
| resources | prompts and schemas | baseline scaffold in place |
| anatomy linker | streams, sense inventory, form ambiguity, report state, exam frame and area, roles (procedure site, inherent location), convergence profile, UBERON graph, verify stream (yes/no or balanced choice), blind reader (no model; optional veto, reads the side), exam side and discourse (stage 1), Greek and Latin roots as proposals (stage 3), proper names / molecules / devices / longer anatomical names / language of the name (stage 4), conflict policy (stage 8, `record` by default), compound names read by the head of the noun phrase (hyphen, abbreviation defined in the text, graft material, UBERON synonyms as longer names), developmental frame as a recorded conflict, head probe experiment (parser, attention), external check on CRAFT and MedMentions, calibration/test split for certification, Learn-then-Test certification tool (waits for the gold set) | in progress; see `docs/linker/`; process and function nouns after the structure ("heart development", "development of the pancreas") are not links and record the structure as inherent location (`process_heads`, 9 Oct 2026); names of other things (assessment tools, proteins, genes, chemicals) read from NCIt and Protein Ontology by type (`longer_names.json`, 6,705 names, 9 Oct 2026; written in one run, with a real word outside the mention); chunk lattice (the phrase read as blocks, off by default, trace only: `chunk_lattice.py`, `block_memory.json`, `lattice-probe`, 9 Oct 2026); E1 recount by concept (`concept_check.py`, `external-check concept_level`), E2 informed LLM reader and E4a UMLS memory arm in `phrase-probe` (`phrase_knowledge.py`, `umls_lookup.py`; criteria fixed before the runs), probes mark the judged occurrence (`at`), 9 Oct 2026; LLM segmenter measured with `phrase-probe llm=sample` (10/22 MedMentions errors, 29 % of right links flagged, role 3/12: proposal only, not a decider, 9 Oct 2026); E1/E2/E4a measured 10 Oct (pre-registered criteria not met); phrase-probe has a time budget, per-phase log and always-saved UMLS cache; construction-integration reader (`ci_reader.py`, `ci_probe.py`, `ci-probe`: LSA space, predication, trace memory + gist, exhaustive construction, constraint integration; measured 10 Oct, off, decides nothing: no configuration meets the criteria, context words are noise at this space size; the context is now one ambito signal per sentence and document, the NCIt kinds of the other things named, centred on the base rate: 41/710 right links changed, 3/22 errors, adds no information at this training size); FalkorDB server job in CI |
| tests | smoke coverage | growing |

The scaffold is intentionally lightweight and provider-neutral. Concrete model and infrastructure bindings should be added incrementally.
