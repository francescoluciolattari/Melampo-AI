# Imaging Model Strategy

Melampo treats imaging as a future-facing adapter layer rather than a fixed model choice.
The current implementation carries image paths and metadata through the clinical reasoning pipeline, extracts technical local-imaging features, selects an imaging provider family, and prepares a remote-provider request contract when configured.

It does not yet perform validated clinical pixel-level or DICOM-volume inference.

## Current behavior

`VolumeEncoder` currently exposes:

- local `series_paths`
- `image_count`
- `has_local_images`
- `input_kind`
- `supported_modalities`
- `preferred_future_models`
- `provider_strategy`
- `provider_selection`
- `provider_readiness`
- `requires_remote`
- `readiness_requirement`
- `remote_result`
- `fallback_mode`
- `encoder_ready`
- `real_pixel_inference`

This lets the rest of the pipeline reason over imaging availability now, while keeping the interface stable for later real providers.

## Runtime strategies

The runtime config field is:

`RuntimeConfig.imaging_provider_strategy`

Supported strategies:

| Strategy | Current behavior | Future target |
|---|---|---|
| `local_metadata` | metadata/path features only | local technical metadata validation |
| `local_pixels` | provider family selected, local pixel inference not yet implemented | local pixel/image encoder |
| `remote_radiology_vlm` | remote request contract prepared, fallback to local features | projection radiology VLM |
| `remote_dicom_3d` | remote request contract prepared, fallback to local features | DICOM-native 3D foundation model |
| `hybrid_multimodal` | chooses projection or 3D provider family by input kind | modality-aware multimodal provider routing |

## Remote provider contract

`RemoteImagingProviderClient` defines the request/response contract for future remote providers.
It currently does not perform network calls. Instead, it returns:

- `status: not_called` when no endpoint is configured
- `status: request_prepared` when an endpoint exists but network inference is still disabled in the research stub
- `fallback_required: true`

`VolumeEncoder` then exposes:

- `remote_result`
- `fallback_mode: remote_stub_to_local_features`

This keeps the system executable while preparing the exact boundary where a real provider will be connected.

## Future provider strategy

The encoder is designed to support the best available imaging model as the project evolves.
The preferred future classes are:

1. Specialized radiology vision-language models
2. DICOM-native 3D foundation models
3. Multimodal clinical VLMs
4. Self-supervised medical image encoders

A future provider should be attached behind the current `VolumeEncoder.encode(...)` interface so that CLI, dataset loaders, pipeline, differential reasoning, and critique do not need to change.

## Integration principle

Do not hard-code one model family into the clinical pipeline.
Instead, evolve `VolumeEncoder` as the adapter that chooses or calls the strongest available provider for the modality:

- CR / DX: projection radiography VLM or image encoder
- CT / MR / PT: volumetric DICOM-native model or 3D encoder
- US: ultrasound-capable image encoder
- mixed studies: multimodal routing with modality-aware fusion

## Research-only caution

Until a real encoder provider is connected and clinically validated, imaging outputs are metadata/path-aware only or remote-contract prepared.
They must not be interpreted as diagnostic pixel-level inference.
