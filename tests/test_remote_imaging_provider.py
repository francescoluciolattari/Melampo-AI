from melampo.models.remote_imaging_provider import RemoteImagingProviderClient


def test_remote_imaging_provider_contract_prepares_request_without_network_call():
    client = RemoteImagingProviderClient(
        provider_name="remote_radiology_vlm_provider",
        endpoint="https://example.invalid/radiology",
        timeout_seconds=45,
        enabled=True,
    )
    selection = {
        "strategy": "remote_radiology_vlm",
        "provider_kind": "remote_projection_radiology_vlm",
        "provider_name": "remote_radiology_vlm_provider",
        "real_pixel_inference": True,
        "requires_remote": True,
        "readiness_requirement": "local_or_remote_projection_images_required",
    }
    result = client.infer(
        study_id="study-001",
        series_paths=["/local/image.png"],
        metadata={"modality": "CR"},
        provider_selection=selection,
    )
    assert result["status"] == "request_prepared"
    assert result["fallback_required"] is True
    assert result["request"]["endpoint"] == "https://example.invalid/radiology"
    assert result["request"]["timeout_seconds"] == 45


def test_remote_imaging_provider_contract_safe_when_not_configured():
    client = RemoteImagingProviderClient(provider_name="remote_dicom_3d_provider")
    result = client.infer(
        study_id="ct-study-001",
        series_paths=["/local/1.dcm"],
        metadata={"modality": "CT"},
        provider_selection={"provider_kind": "remote_3d_dicom_foundation_model"},
    )
    assert result["status"] == "not_called"
    assert result["reason"] == "remote_provider_not_configured"
    assert result["fallback_required"] is True
