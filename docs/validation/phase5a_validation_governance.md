# Melampo Phase 5A Validation Governance

Phase 5A adds research-validation governance around Melampo's model, memory,
retrieval and dream-evolution stack. It does not make Melampo a validated medical
device and does not authorize clinical use.

## New components

- `evaluation/dataset_manifest.py`: dataset metadata, de-identification, label schema, license and required-slice checks.
- `evaluation/validation_protocol.py`: lockable research protocols with model version, memory snapshot and endpoint thresholds.
- `evaluation/slice_analysis.py`: coverage and selective-accuracy checks by modality, site, pathology family, learning status and other slices.
- `evaluation/model_release_gate.py`: research-only gate that combines benchmark, calibration, RAG, slice, dataset and change-control evidence.
- `governance/change_control.py`: PCCP-like change records for model, memory, retriever, policy and dream-branch changes.
- `safety/rails.py`: deterministic input, retrieval and output rails for research-only clinical safety boundaries.

## Governance boundary

Passing the Phase 5A release gate means only that a research artifact is eligible
for stricter review. It does not imply prospective validation, regulatory
clearance, clinical deployment, or diagnostic authority.

Required clinical translation steps remain external to this scaffold:

1. human specialist review;
2. ethics and regulatory protocol approval;
3. locked prospective validation;
4. monitoring and post-market style change governance;
5. formal medical-device review where applicable.
