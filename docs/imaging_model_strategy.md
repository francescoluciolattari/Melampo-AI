# Imaging Model Strategy

Melampo treats imaging as a future-facing adapter layer rather than a fixed model choice.
The current implementation carries image paths and metadata through the clinical reasoning pipeline.
It does not yet perform real pixel-level or DICOM-volume inference.

## Current behavior

`VolumeEncoder` currently exposes:

- local `series_paths`
- `image_count`
- `has_local_images`
- `input_kind`
- `supported_modalities`
- `preferred_future_models`
- `encoder_ready`
- `real_pixel_inference: false`

This lets the rest of the pipeline reason over imaging availability now, while keeping the interface stable for later real providers.

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

Until a real encoder provider is connected and clinically validated, imaging outputs are metadata/path-aware only.
They must not be interpreted as diagnostic pixel-level inference.
