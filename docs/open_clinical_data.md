# Open Clinical Data for Melampo Prototype

This document lists open or credentialed clinical datasets that can be used to test and extend the Melampo research prototype.
The repository must not store real patient data, downloaded DICOM files, or restricted credentialed datasets.

## Safe default

Use the synthetic JSON example first:

```bash
melampo-prototype examples/open_clinical_case_chest_xray_synthetic.json --runtime-profile local_research
```

The example is synthetic and contains no real patient data.

## Adding local images to JSON payloads

For a single JSON case, attach one or more local image paths without editing the JSON file:

```bash
melampo-prototype examples/open_clinical_case_chest_xray_synthetic.json \
  --image-path /local/path/image_001.png \
  --image-path /local/path/image_002.png \
  --include-raw
```

The paths are inserted into the first imaging study as `series_paths` and are carried through `volume_features` as:

- `series_paths`
- `image_count`
- `has_local_images`

The prototype currently carries image paths and metadata. It does not yet perform real pixel/DICOM inference unless a real image encoder service is connected later.

## ChestX-ray14-style CSV workflow

Use the synthetic CSV metadata fixture to test the metadata loader and prototype runner:

```bash
melampo-prototype-cxr examples/chestxray14_metadata_sample.csv --limit 1 --runtime-profile local_research
```

The command reads ChestX-ray14-style metadata, converts each row into a Melampo payload, runs the prototype, and prints JSON output.
It does not download or bundle images. If local images are available after accepting dataset terms, pass an image root:

```bash
melampo-prototype-cxr path/to/Data_Entry_2017.csv --limit 5 --image-root /local/path/to/images
```

## Open-i / Indiana-style CSV workflow

Use the synthetic Open-i-style metadata fixture to test paired report/image metadata conversion:

```bash
melampo-prototype-openi examples/openi_metadata_sample.csv --limit 1 --runtime-profile local_research
```

The command reads Open-i / Indiana-style report metadata, converts each row into a Melampo payload, runs the prototype, and prints JSON output.
It does not download, bundle, or redistribute reports or images. If local images are available after checking source terms, pass an image root:

```bash
melampo-prototype-openi path/to/openi_metadata.csv --limit 5 --image-root /local/path/to/images
```

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
