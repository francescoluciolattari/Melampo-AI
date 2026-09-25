"""melampo-assets.yaml must stay true to the repository it describes.

The manifest lists every asset the project uses beyond its own code. These
checks keep it from drifting into a document nobody can trust: every
DVC-managed asset marked present has a data/*.dvc pin importing exactly the
source URL the manifest states, every pin in data/ is listed, and every
consumer named is a module that exists.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"present", "to_download", "to_integrate", "blocked_decision", "blocked_licence"}
MANAGERS = {"uv", "dvc", "git", "apt", "api", "weights"}


@pytest.fixture(scope="module")
def manifest():
    return yaml.safe_load((ROOT / "melampo-assets.yaml").read_text(encoding="utf-8"))


def _dvc_source(pin: Path) -> str:
    return yaml.safe_load(pin.read_text(encoding="utf-8"))["deps"][0]["path"]


def test_every_asset_is_complete_and_uses_known_values(manifest):
    ids = [asset["id"] for asset in manifest["assets"]]
    assert len(ids) == len(set(ids))
    for asset in manifest["assets"]:
        assert {"id", "kind", "manager", "status", "source", "licence", "consumers"} <= set(asset), asset["id"]
        assert asset["status"] in STATUSES, asset["id"]
        assert asset["manager"] in MANAGERS, asset["id"]


def test_present_dvc_assets_are_pinned_to_the_stated_source(manifest):
    for asset in manifest["assets"]:
        if asset["manager"] == "dvc" and asset["status"] == "present":
            pin = ROOT / f"{asset['path']}.dvc"
            assert pin.is_file(), f"{asset['id']}: {pin.name} missing"
            assert _dvc_source(pin) == asset["source"], asset["id"]


def test_every_dvc_pin_in_data_is_in_the_manifest(manifest):
    listed = {asset.get("path") for asset in manifest["assets"] if asset["manager"] == "dvc"}
    pins = {str(pin.relative_to(ROOT))[: -len(".dvc")] for pin in (ROOT / "data").glob("*.dvc")}
    assert pins <= listed, f"not in melampo-assets.yaml: {sorted(pins - listed)}"


def test_hpo_files_are_pinned_to_a_single_release(manifest):
    """All HPO files move together: mixing releases would pair an ontology
    with annotations written against a different one."""
    releases = {asset["version"] for asset in manifest["assets"] if asset["id"].startswith("hpo-")}
    assert len(releases) == 1


def test_git_managed_data_is_actually_in_git(manifest):
    for asset in manifest["assets"]:
        if asset["manager"] == "git" and asset["status"] == "present":
            assert (ROOT / asset["path"]).is_file(), asset["id"]


def test_every_named_consumer_exists(manifest):
    for asset in manifest["assets"]:
        for consumer in asset["consumers"]:
            if consumer == "to_integrate":
                continue
            assert (ROOT / "src" / "melampo" / consumer).is_file(), f"{asset['id']}: {consumer}"


def test_every_blocked_asset_points_at_a_listed_decision(manifest):
    blocked = {asset_id for decision in manifest["decisions_pending"] for asset_id in decision["blocks"]}
    for asset in manifest["assets"]:
        if asset["status"] == "blocked_decision":
            assert asset["id"] in blocked, asset["id"]
