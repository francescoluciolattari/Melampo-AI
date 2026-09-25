# Training

This package hosts the training-side baseline for Melampo.

Included:

- generative replay engine

The curriculum-progression, meta-learning, elastic-weight-consolidation and
top-level-trainer placeholders from the original scaffold were removed
(2026-09-25 cleanup): each was a never-developed stub with zero consumers and
zero tests, frozen since the April baseline commit. See
`claude/audit_file_obsoleti_2026-09-24.md` in the project for the full audit.

The goal is to make future multimodal and continual-learning work easier to organize and validate.
