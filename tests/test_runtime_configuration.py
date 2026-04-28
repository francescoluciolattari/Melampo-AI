from melampo.app import build_default_runtime
from melampo.config import build_default_config


def test_runtime_config_profiles_and_runtime_description():
    local = build_default_config()
    local_description = local.describe()
    assert local_description["runtime_profile"] == "local_research"
    assert local_description["allow_remote_models"] is False
    assert local_description["imaging_provider_strategy"] == "local_metadata"
    assert "theoretical_quantum" in local_description["disabled_services"]

    remote = build_default_config(runtime_profile="remote_research")
    remote_description = remote.describe()
    assert remote_description["runtime_profile"] == "remote_research"
    assert remote_description["allow_remote_models"] is True
    assert remote_description["imaging_provider_strategy"] == "hybrid_multimodal"
    assert "theoretical_quantum" in remote_description["enabled_services"]

    explicit = build_default_config(runtime_profile="remote_research", imaging_provider_strategy="remote_dicom_3d")
    explicit_description = explicit.describe()
    assert explicit_description["imaging_provider_strategy"] == "remote_dicom_3d"

    runtime = build_default_runtime(config=remote)
    runtime_description = runtime.describe()
    assert runtime_description["config"]["runtime_profile"] == "remote_research"
    assert runtime_description["config"]["imaging_provider_strategy"] == "hybrid_multimodal"
    assert runtime_description["pipeline"] == "ClinicalInferencePipeline"
    assert runtime_description["validator"]["runtime_profile"] == "remote_research"
    assert runtime_description["validator"]["theoretical_quantum"] == "enabled_research"
