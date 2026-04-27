from melampo.models.volume_encoder import VolumeEncoder


def test_volume_encoder_exposes_future_facing_imaging_adapter_metadata():
    encoder = VolumeEncoder()
    result = encoder.encode(
        study_id="study-001",
        series_paths=["/local/images/case001.png"],
        metadata={"modality": "CR"},
    )
    assert result["has_local_images"] is True
    assert result["image_count"] == 1
    assert result["input_kind"] == "projection_or_image_file"
    assert result["encoder_ready"] is True
    assert result["real_pixel_inference"] is False
    assert "multimodal_clinical_vlm" in result["preferred_future_models"]


def test_volume_encoder_detects_volumetric_modalities():
    encoder = VolumeEncoder()
    result = encoder.encode(
        study_id="ct-study-001",
        series_paths=["/local/dicom/1.dcm", "/local/dicom/2.dcm"],
        metadata={"modality": "CT"},
    )
    assert result["input_kind"] == "volumetric_dicom_or_series"
    assert "CT" in result["supported_modalities"]
