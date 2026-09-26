"""Where model weights come from: one pinned identity, one storage switch.

Two files, two jobs:

- ``melampo-assets.yaml`` says **which** weights: for every model, the
  Hugging Face repository, the exact commit (``revision``) and the sha256 of
  every file. Pins are written by ``scripts/pin_model_revisions.py`` from the
  Hub's own metadata, never typed by hand.
- ``melampo-storage.yaml`` says **where** they are fetched from, with a
  single ``backend`` key, following the phased plan decided on 2026-09-26:

  ============  ========================  ===========================================
  backend       phase                     weights come from
  ============  ========================  ===========================================
  upstream      now                       the original repository, at the pinned commit
  hf_mirror     first deployment          private copies in the project's HF organisation
  dvc           first training run        a DVC remote in an EU region
  ============  ========================  ===========================================

Whatever the backend, every downloaded file is checked against the sha256
in the manifest. That check -- not the backend -- is what makes a deployed
model the one that was validated: a mirror re-uploads the files (so its
commit ids differ from upstream's), and a DVC remote knows nothing about
Hugging Face at all, but the file content is the same everywhere or the
download is rejected.

The EU constraint on the ``dvc`` backend is enforced here rather than left
to a comment: model weights fine-tuned on this project's cases are its own
artefacts, and the decision was that they live in the EU.
"""

import hashlib
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKEND_UPSTREAM = "upstream"
BACKEND_HF_MIRROR = "hf_mirror"
BACKEND_DVC = "dvc"
BACKENDS = (BACKEND_UPSTREAM, BACKEND_HF_MIRROR, BACKEND_DVC)

UNPINNED = "to_pin"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HF_URL = re.compile(r"^https://huggingface\.co/([^/\s]+/[^/\s]+?)/?$")

DEFAULT_STORAGE_FILE = "melampo-storage.yaml"
DEFAULT_MANIFEST_FILE = "melampo-assets.yaml"


# ---------------------------------------------------------------------------
# Storage configuration (melampo-storage.yaml)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageConfig:
    backend: str = BACKEND_UPSTREAM
    hf_organization: str = ""
    hf_private: bool = True
    dvc_remote_name: str = "melampo-eu"
    dvc_url: str = ""
    dvc_region: str = ""
    dvc_jurisdiction: str = "EU"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StorageConfig":
        mirror = data.get("hf_mirror") or {}
        dvc = data.get("dvc") or {}
        return cls(
            backend=str(data.get("backend", BACKEND_UPSTREAM)).strip(),
            hf_organization=str(mirror.get("organization") or "").strip(),
            hf_private=bool(mirror.get("private", True)),
            dvc_remote_name=str(dvc.get("remote_name") or "melampo-eu").strip(),
            dvc_url=str(dvc.get("url") or "").strip(),
            dvc_region=str(dvc.get("region") or "").strip(),
            dvc_jurisdiction=str(dvc.get("jurisdiction") or "").strip(),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_STORAGE_FILE) -> "StorageConfig":
        import yaml

        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def problems(self) -> list[str]:
        """Everything that would make this configuration unusable, in words."""
        found: list[str] = []
        if self.backend not in BACKENDS:
            found.append(f"backend must be one of {list(BACKENDS)}, got {self.backend!r}")
        if self.backend == BACKEND_HF_MIRROR:
            if not self.hf_organization:
                found.append("hf_mirror.organization is required for the hf_mirror backend")
            if not self.hf_private:
                found.append("hf_mirror.private must stay true: a public mirror would redistribute the weights")
        if self.backend == BACKEND_DVC:
            if not self.dvc_url:
                found.append("dvc.url is required for the dvc backend")
            if not self.dvc_region:
                found.append("dvc.region is required for the dvc backend")
            if self.dvc_jurisdiction.upper() != "EU":
                found.append("dvc.jurisdiction must be EU (decided 2026-09-26: trained weights live in the EU)")
        return found

    def dvc_remote_commands(self) -> list[list[str]]:
        """The DVC commands that register this configuration's remote (dvc backend only)."""
        if self.backend != BACKEND_DVC or self.problems():
            return []
        return [["dvc", "remote", "add", "--default", "--force", self.dvc_remote_name, self.dvc_url]]


# ---------------------------------------------------------------------------
# Pins (melampo-assets.yaml, kind: model_weights)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinnedFile:
    path: str
    sha256: str
    size: int = 0


@dataclass(frozen=True)
class ModelPin:
    asset_id: str
    repo: str
    revision: str
    files: tuple[PinnedFile, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    gated: str = ""
    mirror_revision: str = ""

    @property
    def is_pinned(self) -> bool:
        return (
            bool(_COMMIT.match(self.revision))
            and bool(self.files)
            and all(_SHA256.match(item.sha256) for item in self.files)
        )

    @property
    def mirror_repo_name(self) -> str:
        return self.repo.split("/", 1)[-1]


def repo_from_source(source: str) -> str:
    """``https://huggingface.co/YalaLab/Pillar0-ChestCT`` -> ``YalaLab/Pillar0-ChestCT``."""
    match = _HF_URL.match(source.strip())
    return match.group(1) if match else ""


def model_pins(manifest: Mapping[str, Any]) -> list[ModelPin]:
    """Every Hugging Face-hosted model_weights asset in the manifest."""
    pins: list[ModelPin] = []
    for asset in manifest.get("assets", []):
        if asset.get("kind") != "model_weights":
            continue
        repo = str(asset.get("repo") or repo_from_source(str(asset.get("source", ""))))
        if not repo:
            continue
        pins.append(
            ModelPin(
                asset_id=str(asset["id"]),
                repo=repo,
                revision=str(asset.get("revision") or UNPINNED),
                files=tuple(
                    PinnedFile(path=str(item["path"]), sha256=str(item.get("sha256", "")), size=int(item.get("size", 0)))
                    for item in asset.get("files") or []
                ),
                include=tuple(asset.get("include") or ()),
                exclude=tuple(asset.get("exclude") or ()),
                gated=str(asset.get("gated") or ""),
                mirror_revision=str(asset.get("mirror_revision") or ""),
            )
        )
    return pins


# ---------------------------------------------------------------------------
# Resolving and verifying a download
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadPlan:
    asset_id: str
    backend: str
    repo: str = ""
    revision: str = ""
    dvc_target: str = ""
    files: tuple[PinnedFile, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


def plan_download(pin: ModelPin, config: StorageConfig) -> DownloadPlan:
    """Where this model would be fetched from under ``config``. Raises if it cannot be."""
    problems = config.problems()
    if problems:
        raise ValueError("; ".join(problems))
    if not pin.is_pinned:
        raise ValueError(
            f"{pin.asset_id} is not pinned yet (revision {pin.revision!r}): run the "
            "'Pin model revisions' workflow before downloading"
        )
    if config.backend == BACKEND_UPSTREAM:
        return DownloadPlan(pin.asset_id, config.backend, repo=pin.repo, revision=pin.revision, files=pin.files)
    if config.backend == BACKEND_HF_MIRROR:
        notes = () if pin.mirror_revision else ("no mirror_revision recorded: fetching the mirror's main, verified by sha256",)
        return DownloadPlan(
            pin.asset_id,
            config.backend,
            repo=f"{config.hf_organization}/{pin.mirror_repo_name}",
            revision=pin.mirror_revision or "main",
            files=pin.files,
            notes=notes,
        )
    return DownloadPlan(pin.asset_id, config.backend, dvc_target=f"models/{pin.asset_id}.dvc", files=pin.files)


def sha256_of(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(directory: str | Path, files: Iterable[PinnedFile]) -> list[str]:
    """Every pinned file that is missing or whose content differs, in words. Empty means verified."""
    root = Path(directory)
    problems: list[str] = []
    for item in files:
        target = root / item.path
        if not target.is_file():
            problems.append(f"{item.path}: missing")
            continue
        actual = sha256_of(target)
        if actual != item.sha256:
            problems.append(f"{item.path}: sha256 {actual} != pinned {item.sha256}")
    return problems


def download(pin: ModelPin, config: StorageConfig, destination: str | Path, *, token: str | None = None) -> Path:
    """Fetch the pinned files through the configured backend, then verify every one.

    Needs the ``models`` extra (huggingface_hub) for the two Hugging Face
    backends, and DVC for the ``dvc`` backend. Raises on any mismatch -- a
    model that is not byte-identical to its pin is not the validated model.
    """
    plan = plan_download(pin, config)
    destination = Path(destination)
    if plan.backend == BACKEND_DVC:  # pragma: no cover - needs a configured remote
        subprocess.run(["dvc", "pull", plan.dvc_target], check=True)
        destination = Path("models") / pin.asset_id
    else:  # pragma: no cover - network call
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=plan.repo,
            revision=plan.revision,
            allow_patterns=[item.path for item in plan.files],
            local_dir=destination,
            token=token,
        )
    problems = verify_files(destination, plan.files)
    if problems:
        raise ValueError(f"{pin.asset_id}: downloaded files do not match their pins: {problems}")
    return destination


def select_files(paths: Sequence[str], include: Sequence[str], exclude: Sequence[str]) -> list[str]:
    """Filter a repository listing with glob patterns (fnmatch), include first then exclude."""
    from fnmatch import fnmatch

    selected = [path for path in paths if not include or any(fnmatch(path, pattern) for pattern in include)]
    return [path for path in selected if not any(fnmatch(path, pattern) for pattern in exclude)]
