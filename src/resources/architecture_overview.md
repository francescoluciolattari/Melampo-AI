# Architecture Overview

The new `src` baseline is organized around the following axes:

- `melampo/clinical`: standards, schema, and interoperability anchors
- `melampo/data`: ingestion, normalization, FHIR and DICOM placeholders
- `melampo/models`: modality-specific and control-layer placeholders
- `melampo/memory`: episodic, semantic, graph, and retrieval interfaces
- `melampo/reasoning`: differential, critique, escalation, abstention, and workspace
- `melampo/training`: curriculum, meta-learning, replay, continual-learning placeholders
- `melampo/evaluation`: calibration, risk-coverage, validation, and falsification
- `melampo/orchestration`: routing, contracts, MCP, A2A, and service registry

This structure is intended as the stable baseline for all future implementation work.
