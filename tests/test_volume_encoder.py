from melampo.config import build_default_config
from melampo.models.imaging_provider_selector import ImagingProviderSelector
from melampo.models.volume_encoder import VolumeEncoder


def test_volume_encoder_exposes_future_facing_imaging_adapter_metadata(tmp_path):
    image_path = tmp_path / "case001.png"
    image_path.write_text("synthetic image placeholder", encoding="utf-8")

    encoder = VolumeEncoder()
    result = encoder.encode(
        study_id="study-001",
        series_paths=[str(image_path)],
        metadata={"modality": "CR"},
    )
    assert result["has_local_images"] is True
    assert result["image_count"] == 1
    assert result["input_kind"] == "projection_or_image_file"
    assert result["provider_strategy"] == "local_metadata"
    assert result["provider_selection"]["provider_kind"] == "local_metadata_adapter"
    assert result["provider_readiness"] == "metadata_ready"
    assert result["encoder_ready"] is True
    assert result["real_pixel_inference"] is False
    assert result["requires_remote"] is False
    assert result["remote_result"] is None
    assert result["fallback_mode"] == "local_features_only"
    assert result["routing_hint"] == "route_to_projection_radiology_provider"
    assert result["local_features"]["local_readiness"] == "ready"
    assert result["local_features"]["existing_path_count"] == 1
    assert "multimodal_clinical_vlm" in result["preferred_future_models"]


def test_volume_encoder_detects_volumetric_modalities():
    encoder = VolumeEncoder()
    result = encoder.encode(
        study_id="ct-study-001",
        series_paths=["/local/dicom/1.dcm", "/local/dicom/2.dcm"],
        metadata={"modality": "CT"},
    )
    assert result["input_kind"] == "volumetric_dicom_or_series"
    assert result["routing_hint"] == "route_to_3d_dicom_provider"
    assert result["local_features"]["missing_path_count"] == 2
    assert "CT" in result["supported_modalities"]


def test_volume_encoder_reads_imaging_strategy_from_runtime_config(tmp_path):
    image_path = tmp_path / "case001.png"
    image_path.write_text("synthetic image placeholder", encoding="utf-8")
    config = build_default_config(runtime_profile="remote_research")
    encoder = VolumeEncoder(config=config)
    result = encoder.encode(
        study_id="study-remote-001",
        series_paths=[str(image_path)],
        metadata={"modality": "CR"},
    )
    assert result["provider_strategy"] == "hybrid_multimodal"
    assert result["provider_selection"]["provider_kind"] == "remote_projection_radiology_vlm"
    assert result["provider_readiness"] == "remote_provider_configured"
    assert result["requires_remote"] is True
    assert result["real_pixel_inference"] is True
    assert result["remote_result"]["status"] == "not_called"
    assert result["remote_result"]["fallback_required"] is True
    assert result["fallback_mode"] == "remote_stub_to_local_features"


def test_imaging_provider_selector_routes_future_strategies():
    selector = ImagingProviderSelector()
    radiology = selector.select("remote_radiology_vlm", "projection_or_image_file").describe()
    dicom = selector.select("remote_dicom_3d", "volumetric_dicom_or_series").describe()
    hybrid = selector.select("hybrid_multimodal", "volumetric_dicom_or_series").describe()

    assert radiology["provider_kind"] == "remote_projection_radiology_vlm"
    assert dicom["provider_kind"] == "remote_3d_dicom_foundation_model"
    assert hybrid["provider_kind"] == "remote_3d_dicom_foundation_model"
    assert radiology["requires_remote"] is True
    assert dicom["real_pixel_inference"] is True
