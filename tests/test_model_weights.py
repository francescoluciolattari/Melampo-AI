"""Tests for models/weights.py, melampo-storage.yaml and scripts/pin_model_revisions.py."""

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

from melampo.models.weights import (
    BACKEND_DVC,
    BACKEND_HF_MIRROR,
    BACKEND_UPSTREAM,
    ModelPin,
    PinnedFile,
    StorageConfig,
    model_pins,
    plan_download,
    repo_from_source,
    select_files,
    verify_files,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
SHA = "b" * 64


def _pin(**overrides):
    values = {"asset_id": "m", "repo": "org/Model", "revision": COMMIT, "files": (PinnedFile("w.safetensors", SHA, 10),)}
    values.update(overrides)
    return ModelPin(**values)


# --------------------------------------------------------------------------
# Storage configuration
# --------------------------------------------------------------------------


def test_the_committed_storage_file_is_valid_and_starts_upstream():
    config = StorageConfig.load(ROOT / "melampo-storage.yaml")
    assert config.problems() == []
    assert config.backend == BACKEND_UPSTREAM


def test_unknown_backend_is_a_problem():
    assert StorageConfig(backend="s3").problems()


def test_hf_mirror_needs_an_organisation_and_stays_private():
    assert StorageConfig(backend=BACKEND_HF_MIRROR).problems()
    assert StorageConfig(backend=BACKEND_HF_MIRROR, hf_organization="melampo-ai", hf_private=False).problems()
    assert StorageConfig(backend=BACKEND_HF_MIRROR, hf_organization="melampo-ai").problems() == []


def test_dvc_backend_must_be_in_the_eu():
    good = StorageConfig(backend=BACKEND_DVC, dvc_url="s3://b/p", dvc_region="eu-south-1", dvc_jurisdiction="EU")
    assert good.problems() == []
    assert good.dvc_remote_commands() == [["dvc", "remote", "add", "--default", "--force", "melampo-eu", "s3://b/p"]]
    us = StorageConfig(backend=BACKEND_DVC, dvc_url="s3://b/p", dvc_region="us-east-1", dvc_jurisdiction="US")
    assert any("EU" in problem for problem in us.problems())
    assert us.dvc_remote_commands() == []


def test_from_mapping_reads_nested_sections():
    config = StorageConfig.from_mapping(
        {"backend": "dvc", "dvc": {"url": "s3://x", "region": "eu-west-1", "jurisdiction": "EU"}, "hf_mirror": {"organization": "o"}}
    )
    assert (config.backend, config.dvc_url, config.dvc_region, config.hf_organization) == ("dvc", "s3://x", "eu-west-1", "o")


# --------------------------------------------------------------------------
# Pins and download plans
# --------------------------------------------------------------------------


def test_repo_from_source():
    assert repo_from_source("https://huggingface.co/YalaLab/Pillar0-ChestCT") == "YalaLab/Pillar0-ChestCT"
    assert repo_from_source("https://github.com/soda-inria/tabicl") == ""


def test_is_pinned_needs_a_commit_and_file_hashes():
    assert _pin().is_pinned
    assert not _pin(revision="to_pin").is_pinned
    assert not _pin(files=()).is_pinned
    assert not _pin(files=(PinnedFile("w", "short"),)).is_pinned


def test_upstream_plan_uses_the_pinned_commit():
    plan = plan_download(_pin(), StorageConfig())
    assert (plan.repo, plan.revision) == ("org/Model", COMMIT)


def test_mirror_plan_uses_the_organisation_and_mirror_revision():
    config = StorageConfig(backend=BACKEND_HF_MIRROR, hf_organization="melampo-ai")
    assert plan_download(_pin(mirror_revision="c" * 40), config).repo == "melampo-ai/Model"
    unrecorded = plan_download(_pin(), config)
    assert unrecorded.revision == "main" and unrecorded.notes


def test_dvc_plan_points_at_the_models_directory():
    config = StorageConfig(backend=BACKEND_DVC, dvc_url="s3://b", dvc_region="eu-south-1", dvc_jurisdiction="EU")
    assert plan_download(_pin(), config).dvc_target == "models/m.dvc"


def test_an_unpinned_model_cannot_be_planned():
    with pytest.raises(ValueError, match="not pinned"):
        plan_download(_pin(revision="to_pin"), StorageConfig())


def test_an_invalid_config_cannot_be_planned():
    with pytest.raises(ValueError):
        plan_download(_pin(), StorageConfig(backend=BACKEND_HF_MIRROR))


def test_verify_files(tmp_path):
    (tmp_path / "w.safetensors").write_bytes(b"weights")
    good = PinnedFile("w.safetensors", hashlib.sha256(b"weights").hexdigest())
    assert verify_files(tmp_path, [good]) == []
    assert "sha256" in verify_files(tmp_path, [PinnedFile("w.safetensors", SHA)])[0]
    assert "missing" in verify_files(tmp_path, [PinnedFile("absent", SHA)])[0]


def test_select_files():
    paths = ["config.json", "model-00001.safetensors", "original/model.bin", "metal/model.bin"]
    assert select_files(paths, ["*.json", "*.safetensors"], ["original/*", "metal/*"]) == ["config.json", "model-00001.safetensors"]
    assert select_files(paths, [], ["metal/*"]) == ["config.json", "model-00001.safetensors", "original/model.bin"]


# --------------------------------------------------------------------------
# The manifest's models
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return yaml.safe_load((ROOT / "melampo-assets.yaml").read_text(encoding="utf-8"))


def test_every_hf_model_is_listed_with_repo_revision_and_files(manifest):
    weights = [asset for asset in manifest["assets"] if asset["kind"] == "model_weights" and "huggingface.co" in asset["source"]]
    assert weights
    for asset in weights:
        assert asset["manager"] == "weights", asset["id"]
        assert asset["repo"] == repo_from_source(asset["source"]), asset["id"]
        assert "revision" in asset and "files" in asset, asset["id"]


def test_pins_that_exist_are_well_formed(manifest):
    for pin in model_pins(manifest):
        if pin.revision == "to_pin":
            assert not pin.files, f"{pin.asset_id}: files listed without a revision"
            continue
        assert pin.is_pinned, f"{pin.asset_id}: revision or a file hash is malformed"
        assert len({item.path for item in pin.files}) == len(pin.files)


# --------------------------------------------------------------------------
# The pin script's pure part
# --------------------------------------------------------------------------


def _load_pin_script():
    spec = importlib.util.spec_from_file_location("pin_model_revisions", ROOT / "scripts" / "pin_model_revisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INFO = {
    "sha": COMMIT,
    "siblings": [
        {"rfilename": "config.json", "size": 120},
        {"rfilename": "model.safetensors", "size": 134, "lfs": {"sha256": SHA, "size": 376000000}},
        {"rfilename": "original/model.bin", "size": 134, "lfs": {"sha256": "c" * 64, "size": 5}},
    ],
}


def test_pin_from_api_uses_lfs_hashes_and_hashes_small_files():
    script = _load_pin_script()
    commit, files = script.pin_from_api(INFO, exclude=["original/*"], hash_small_file=lambda path: "d" * 64)
    assert commit == COMMIT
    assert files == [
        {"path": "config.json", "sha256": "d" * 64, "size": 120},
        {"path": "model.safetensors", "sha256": SHA, "size": 376000000},
    ]


def test_pin_from_api_refuses_to_skip_an_unhashable_file():
    script = _load_pin_script()
    with pytest.raises(ValueError, match="config.json"):
        script.pin_from_api(INFO)


def test_write_pins_keeps_comments(tmp_path):
    pytest.importorskip("ruamel.yaml")
    script = _load_pin_script()
    path = tmp_path / "m.yaml"
    path.write_text(
        "# top comment\nassets:\n  # a model\n  - id: m\n    kind: model_weights\n    revision: to_pin\n    files: []\n",
        encoding="utf-8",
    )
    script.write_pins(path, {"m": (COMMIT, [{"path": "w", "sha256": SHA, "size": 1}])}, "2026-09-26")
    text = path.read_text(encoding="utf-8")
    assert "# top comment" in text and "# a model" in text
    data = yaml.safe_load(text)
    assert data["assets"][0]["revision"] == COMMIT
    assert data["assets"][0]["files"][0]["sha256"] == SHA
    assert data["assets"][0]["pinned_on"] == "2026-09-26"
