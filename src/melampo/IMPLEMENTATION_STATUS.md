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
| anatomy linker | streams, sense inventory, form ambiguity, report state, exam frame and area, roles (procedure site, inherent location), convergence profile, UBERON graph, verify stream (yes/no or balanced choice), blind reader (no model; optional veto, reads the side), exam side and discourse (stage 1), Greek and Latin roots as proposals (stage 3), proper names / molecules / devices / longer anatomical names / language of the name (stage 4), conflict policy (stage 8, `record` by default), external check on CRAFT and MedMentions, calibration/test split for certification, Learn-then-Test certification tool (waits for the gold set) | in progress; see `docs/linker/` |
| tests | smoke coverage | growing |

The scaffold is intentionally lightweight and provider-neutral. Concrete model and infrastructure bindings should be added incrementally.
