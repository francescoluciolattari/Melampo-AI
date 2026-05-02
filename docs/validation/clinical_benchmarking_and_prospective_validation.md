# Clinical Benchmarking, Prospective Validation and Calibration

## Scope

This document defines the research-grade validation path for Project Melampo. It does not claim clinical validity and does not authorize clinical use.

Melampo must be evaluated in three distinct layers:

1. **Retrospective clinical benchmarking** on labeled and de-identified datasets.
2. **Prospective validation** where predictions are locked before outcomes are known.
3. **Calibration on real datasets** to align confidence, abstention and escalation behavior with observed correctness.

## 1. Retrospective benchmark layer

Implemented base module:

```text
src/melampo/evaluation/clinical_benchmark.py
```

Primary classes:

- `ClinicalBenchmarkRecord`
- `ClinicalBenchmarkRunner`
- `ClinicalBenchmarkReport`

Expected JSONL input format:

```json
{"case_id":"case-001","payload":{"case_id":"case-001"},"gold_labels":["pneumonia"],"slices":{"modality":"XR","site":"demo"},"provenance":{"dataset":"demo"}}
```

The benchmark runner computes:

- sample count;
- answered count;
- abstained count;
- top-1 accuracy;
- coverage;
- selective accuracy;
- slice reports.

Important distinction:

- `top1_accuracy` counts abstentions as non-correct over the full sample.
- `selective_accuracy` evaluates accuracy among non-abstained predictions.
- `coverage` measures how often Melampo produced a non-abstained answer.

## 2. Prospective validation layer

Implemented base module:

```text
src/melampo/evaluation/prospective_validation.py
```

Primary classes:

- `ProspectivePrediction`
- `ProspectiveValidationRegistry`

Prospective validation requires predictions to be created and locked before outcomes are known.

Required workflow:

1. Register a case payload and diagnostic result.
2. Store prediction ID, payload hash and timestamp.
3. Lock the prediction.
4. Attach outcome only later.
5. Evaluate only completed prediction/outcome pairs.

The current implementation is file-based and intended for research scaffolding. Production validation should use immutable storage, protocol identifiers, de-identification checks, IRB/ethics approval and access controls.

## 3. Calibration layer

Implemented base module:

```text
src/melampo/evaluation/calibration.py
```

Primary classes:

- `ConfidenceCalibrationEvaluator`
- `CalibrationReport`
- `CalibrationBin`

Metrics:

- Expected Calibration Error, ECE;
- Maximum Calibration Error, MCE;
- Brier score;
- calibration bins;
- threshold suggestion by target empirical accuracy.

Calibration is not clinical validation. A calibrated model can still be wrong, biased or unsafe on distribution-shifted data.

## Required dataset governance

Every benchmark or validation dataset must include:

- dataset card;
- de-identification report;
- license/access class;
- site/source metadata;
- modality metadata;
- gold-label creation protocol;
- adjudication procedure;
- inclusion/exclusion criteria;
- known limitations;
- demographic and prevalence slices where lawful and available.

## Required validation slices

At minimum:

- modality;
- site/source;
- age band;
- sex/gender where lawful and available;
- pathology family;
- prevalence band;
- scanner/protocol where relevant;
- source type: case, literature, guideline, synthetic;
- learning status: candidate, promoted, rejected, needs_review, retired.

## Clinical safety gates

Before any clinical deployment, Melampo requires:

1. retrospective benchmark performance above pre-registered thresholds;
2. prospective validation on target clinical workflow;
3. calibration by modality/pathology/site;
4. abstention and escalation analysis;
5. bias and subgroup review;
6. human specialist review;
7. regulatory classification review;
8. monitoring and drift detection plan;
9. incident response plan.

## Current implementation status

Implemented:

- retrospective benchmark runner;
- prospective validation registry;
- confidence calibration evaluator;
- tests for these primitives.

Not implemented in core:

- real clinical datasets;
- immutable clinical-grade registry;
- IRB workflow;
- statistical power analysis;
- FDA/MDR regulatory workflow;
- monitoring dashboards;
- deployment-specific integrations.
