# Open Clinical Data for Melampo Prototype

This document lists open or credentialed clinical datasets that can be used to test and extend the Melampo research prototype.
The repository must not store real patient data, downloaded DICOM files, or restricted credentialed datasets.

## Safe default

Use the synthetic example first:

```bash
melampo-prototype examples/open_clinical_case_chest_xray_synthetic.json --runtime-profile local_research
```

The example is synthetic and contains no real patient data.

## Dataset catalog

The code catalog is available in:

`src/melampo/datasets/open_catalog.py`

It currently tracks:

| Dataset | Access | Best use | Caution |
|---|---|---|---|
| NIH ChestX-ray14 | Public download with terms | Imaging-led weak-label chest X-ray experiments | Labels are NLP-derived weak labels, not adjudicated ground truth. |
| Open-i / Indiana University Chest X-rays | Open collection with license terms | Paired report/image style examples | Verify redistribution terms before copying reports/images. |
| The Cancer Imaging Archive | Open collections with collection-specific terms | Oncology DICOM workflow tests | Track collection-specific license and provenance. |
| MIMIC-IV | Credentialed PhysioNet access | Structured EHR/ICU/hospital trajectories | Requires training, credentialing, and DUA. Do not commit data. |
| MedQuAD | Public repository | QA/retrieval/grounding experiments | Educational QA, not patient-level records. |

## Integration rules

1. Keep real downloaded clinical data outside the repository.
2. Store only paths, metadata, or synthetic fixtures in Git.
3. Preserve source dataset provenance in `payload.provenance`.
4. Treat weak labels as weak supervision only.
5. Do not use credentialed datasets such as MIMIC-IV unless the local user/institution has completed the required training and DUA.
6. Keep prototype outputs marked as research-only until clinical validation exists.

## Suggested next loaders

Future loader modules can be added without changing the clinical core:

- `melampo.datasets.chestxray14_loader`
- `melampo.datasets.openi_loader`
- `melampo.datasets.tcia_manifest_loader`
- `melampo.datasets.mimic_mapping`

Each loader should convert source-specific records into the existing prototype JSON payload shape before calling `run_prototype_case`.
