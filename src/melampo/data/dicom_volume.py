"""A DICOM series kept whole, de-identified, and assessed as a volume -- step 5 of the document-processing plan.

**What was lost before this.** dicom_handler.py turns each uploaded DICOM
file into at most 16 PNG frames, 8-bit, each frame stretched to its own
min/max -- right for showing an image, and for a vision-language model
reading a picture. Everything else was discarded when ingestion finished:
the Hounsfield values (every slice rescaled differently, so "-1000" and
"+40" no longer mean air and soft tissue), the geometry, and the file
itself. Nothing volumetric could ever run on a case's own CT.

**What Pillar-0 actually needs, verified at the source rather than
assumed.** Pillar-0's own preprocessing is RAVE (github.com/YalaLab/rave,
cited by the Pillar-0 paper, arXiv:2511.17803): its input is a CSV of
`series_path` entries -- DICOM series directories or NIfTI files -- and
RAVE itself does the HU range (-1024..3071 for CT), resampling, crop/pad
and multi-windowing (lung -600/1500, mediastinum 50/400, bone 400/1800,
brain 40/80, liver 80/150). So what a case must keep is not a rendered
picture but the series' original instances. Released Pillar-0 checkpoints
(huggingface.co/YalaLab): head CT, chest CT, abdomen(-pelvis) CT, breast
MRI -- nothing for radiographs, ultrasound or densitometry (RAVE has
x-ray and mammogram configs, but there is no Pillar-0 model for them).

**Privacy, by the same rule dicom_handler.py already holds: an allowlist.**
Keeping the uploaded bytes would keep everything in them -- patient name,
ID, birth date, institution, physicians, and on real scanners hundreds of
private tags. Each instance is therefore rebuilt from an explicit list of
the attributes pixels, geometry and series grouping need, and nothing
else: an allowlist cannot leak a tag nobody thought to block. Pixel data
is copied byte for byte, compressed or not -- never decoded and
re-encoded, so no fidelity is lost. Dates and times are left out too
(stricter than dicom_handler's metadata allowlist, which keeps StudyDate):
these instances are the ones that would be handed to a model service.
Not handled, and said so: text burned into the pixels themselves. A
declared BurnedInAnnotation="YES" is reported; undeclared burn-in (common
in ultrasound and screenshots) cannot be detected here.

**Assessed, not assumed to be a volume.** A series is a usable volume only
if every instance shares one orientation, size and pixel spacing, slice
positions (projected on the slice normal, not InstanceNumber, which
scanners do not guarantee to be spatial) are distinct and evenly spaced,
and there are enough of them. Each failed condition is returned by name.

**Not done here:** calling Pillar-0. The adapter stays disabled until a
deployment exists; `export_series_directory` is the hand-off RAVE's
`series_path` expects, and the only function here that writes to disk.
"""

from __future__ import annotations

import copy
import io
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

# Attributes an instance keeps -- everything else is dropped.
INSTANCE_ALLOWLIST = (
    # Identity of the object and its series (UIDs: pseudonymous links, needed
    # to group instances into series and to write a valid file).
    "SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID",
    "Modality", "SeriesNumber", "InstanceNumber", "AcquisitionNumber", "ImageType",
    "BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName", "PatientPosition",
    # Geometry.
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "SliceThickness",
    "SpacingBetweenSlices", "SliceLocation",
    # Image pixel module.
    "SamplesPerPixel", "PhotometricInterpretation", "PlanarConfiguration", "Rows", "Columns",
    "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation", "NumberOfFrames",
    "PixelPaddingValue", "PixelData",
    # Value transforms: stored value -> Hounsfield units (CT), display windows.
    "RescaleSlope", "RescaleIntercept", "RescaleType", "WindowCenter", "WindowWidth", "VOILUTFunction",
    # Acquisition parameters a model or reviewer may need.
    "KVP", "ConvolutionKernel", "ContrastBolusAgent",
    "MagneticFieldStrength", "RepetitionTime", "EchoTime", "ScanningSequence", "SequenceVariant", "MRAcquisitionType",
    # Enhanced multi-frame geometry (private tags stripped inside them).
    "SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence",
    # Reported, never acted on silently.
    "BurnedInAnnotation",
)

# DICOM LO: at most 64 characters.
DEIDENTIFICATION_METHOD = "Melampo allowlist: identity, dates, private tags removed"

# A usable volume needs at least this many slices: fewer is a localiser or a
# key-image set, not something a volumetric model can read.
MIN_SLICES_FOR_VOLUME = 16
# Relative tolerance for "evenly spaced": 1% of the median spacing.
SPACING_TOLERANCE = 0.01
ORIENTATION_TOLERANCE = 1e-3

CHECKPOINT_CHEST_CT = "Pillar0-ChestCT"
CHECKPOINT_ABDOMEN_CT = "Pillar0-AbdomenCT"
CHECKPOINT_HEAD_CT = "Pillar0-HeadCT"
CHECKPOINT_BREAST_MRI = "Pillar0-BreastMRI"

# Anatomy from BodyPartExamined (DICOM defined terms) and descriptions
# (Italian and English). Matched as whole words on upper-cased text.
_ANATOMY_WORDS = {
    "chest": ("CHEST", "THORAX", "LUNG", "TORACE", "TORACICA", "TORACICO", "POLMONI", "POLMONE"),
    "abdomen": ("ABDOMEN", "PELVIS", "ABDOMENPELVIS", "ADDOME", "ADDOMINALE", "PELVI", "ADDOMINOPELVICA"),
    "head": ("HEAD", "BRAIN", "SKULL", "ENCEFALO", "CRANIO", "CEREBRALE", "TESTA"),
    "breast": ("BREAST", "MAMMELLA", "MAMMELLE", "MAMMARIA", "MAMMARIO", "SENO"),
}


@dataclass(frozen=True)
class DeidentifiedInstance:
    data: bytes
    series_uid: str
    has_pixel_data: bool
    notes: tuple[str, ...] = ()


def deidentify_instance(data: bytes) -> DeidentifiedInstance | None:
    """The instance rebuilt from INSTANCE_ALLOWLIST only -- pixel data byte for byte. None if it is not readable DICOM."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.errors import InvalidDicomError

    try:
        source = pydicom.dcmread(io.BytesIO(data))
    except (InvalidDicomError, ValueError, OSError, AttributeError, KeyError, TypeError):
        return None
    transfer_syntax = getattr(getattr(source, "file_meta", None), "TransferSyntaxUID", None)
    if transfer_syntax is None or "SOPInstanceUID" not in source or "SOPClassUID" not in source:
        return None

    rebuilt = Dataset()
    for keyword in INSTANCE_ALLOWLIST:
        if keyword in source:
            element = copy.deepcopy(source[keyword])
            rebuilt[element.tag] = element
    rebuilt.remove_private_tags()  # recurses into the functional-group sequences
    rebuilt.PatientIdentityRemoved = "YES"
    rebuilt.DeidentificationMethod = DEIDENTIFICATION_METHOD

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = source.SOPClassUID
    meta.MediaStorageSOPInstanceUID = source.SOPInstanceUID
    meta.TransferSyntaxUID = transfer_syntax
    rebuilt.file_meta = meta

    buffer = io.BytesIO()
    rebuilt.save_as(buffer, enforce_file_format=True)
    notes = []
    if str(source.get("BurnedInAnnotation", "")).strip().upper() == "YES":
        notes.append("burned_in_annotation_declared")
    return DeidentifiedInstance(
        data=buffer.getvalue(), series_uid=str(source.get("SeriesInstanceUID", "")),
        has_pixel_data="PixelData" in source, notes=tuple(notes),
    )


@dataclass(frozen=True)
class VolumeAssessment:
    """Whether a series is a usable volume, and its geometry -- JSON-safe via as_dict()."""

    usable: bool
    problems: tuple[str, ...]
    modality: str | None
    slice_count: int
    rows: int | None = None
    columns: int | None = None
    pixel_spacing: tuple[float, float] | None = None
    slice_spacing: float | None = None
    slice_thickness: float | None = None
    orientation: str | None = None
    anatomy: str | None = None
    order: tuple[int, ...] = ()  # instance indices, sorted along the slice normal
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable, "problems": list(self.problems), "modality": self.modality,
            "slice_count": self.slice_count, "rows": self.rows, "columns": self.columns,
            "pixel_spacing": list(self.pixel_spacing) if self.pixel_spacing else None,
            "slice_spacing": self.slice_spacing, "slice_thickness": self.slice_thickness,
            "orientation": self.orientation, "anatomy": self.anatomy, "notes": list(self.notes),
        }


def _floats(value: Any, count: int) -> tuple[float, ...] | None:
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return numbers if len(numbers) == count else None


def _anatomy(dataset: Any) -> str | None:
    text = " ".join(
        str(dataset.get(keyword, "") or "") for keyword in ("BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName")
    ).upper()
    words = set(re.split(r"[^A-Z0-9]+", text))
    found = [anatomy for anatomy, names in _ANATOMY_WORDS.items() if words & set(names)]
    return found[0] if len(found) == 1 else ("multiple:" + "+".join(sorted(found)) if found else None)


def _orientation_label(normal: tuple[float, float, float]) -> str:
    axis = max(range(3), key=lambda index: abs(normal[index]))
    return ("sagittal", "coronal", "axial")[axis]


def assess_series(instances: Sequence[bytes]) -> VolumeAssessment:
    """Read headers only (no pixel decoding) and decide whether these instances form one usable volume."""
    import pydicom

    headers = []
    for data in instances:
        try:
            headers.append(pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True))
        except Exception:  # noqa: BLE001 -- an unreadable instance is a named problem, never a crash
            headers.append(None)
    problems: list[str] = []
    notes: list[str] = []
    readable = [(index, header) for index, header in enumerate(headers) if header is not None]
    if len(readable) != len(headers):
        problems.append("unreadable_instances")
    if not readable:
        return VolumeAssessment(usable=False, problems=("no_readable_instances",), modality=None, slice_count=0)

    first = readable[0][1]
    modality = str(first.get("Modality", "") or "") or None
    if any(_frames(header) > 1 for _, header in readable):
        problems.append("enhanced_multiframe_not_assembled")
    if len({str(header.get("SeriesInstanceUID", "")) for _, header in readable}) > 1:
        problems.append("mixed_series")
    if any(str(header.get("BurnedInAnnotation", "")).strip().upper() == "YES" for _, header in readable):
        notes.append("burned_in_annotation_declared")

    sizes = {(int(header.get("Rows", 0) or 0), int(header.get("Columns", 0) or 0)) for _, header in readable}
    spacings = {_floats(header.get("PixelSpacing"), 2) for _, header in readable}
    orientations = [_floats(header.get("ImageOrientationPatient"), 6) for _, header in readable]
    positions = [_floats(header.get("ImagePositionPatient"), 3) for _, header in readable]
    if len(sizes) != 1:
        problems.append("inconsistent_image_size")
    if len(spacings) != 1 or None in spacings:
        problems.append("inconsistent_or_missing_pixel_spacing")
    if any(item is None for item in orientations) or any(item is None for item in positions):
        problems.append("missing_geometry")
        return _assessment(False, problems, notes, modality, readable, sizes, spacings, None, None, None, first)
    reference = orientations[0]
    if any(max(abs(a - b) for a, b in zip(item, reference, strict=True)) > ORIENTATION_TOLERANCE for item in orientations):
        problems.append("inconsistent_orientation")

    row, column = reference[:3], reference[3:]
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    distances = [sum(p * n for p, n in zip(position, normal, strict=True)) for position in positions]
    order = sorted(range(len(readable)), key=lambda index: distances[index])
    ordered = [distances[index] for index in order]
    gaps = [round(later - earlier, 6) for earlier, later in pairwise(ordered)]
    slice_spacing = None
    if gaps:
        if min(gaps) <= 1e-6:
            problems.append("duplicate_slice_positions")
        median = sorted(gaps)[len(gaps) // 2]
        if median > 0:
            slice_spacing = round(median, 4)
            if max(abs(gap - median) for gap in gaps) > SPACING_TOLERANCE * median:
                problems.append("uneven_slice_spacing")
    if len(readable) < MIN_SLICES_FOR_VOLUME:
        problems.append("too_few_slices")
    return _assessment(
        not problems, problems, notes, modality, readable, sizes, spacings, slice_spacing,
        _orientation_label(normal), tuple(readable[index][0] for index in order), first,
    )


def _frames(header: Any) -> int:
    try:
        return int(header.get("NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _assessment(usable, problems, notes, modality, readable, sizes, spacings, slice_spacing, orientation, order, first):
    size = next(iter(sizes)) if len(sizes) == 1 else (None, None)
    spacing = next(iter(spacings)) if len(spacings) == 1 else None
    try:
        thickness = float(first.get("SliceThickness")) if first.get("SliceThickness") not in (None, "") else None
    except (TypeError, ValueError):
        thickness = None
    return VolumeAssessment(
        usable=usable, problems=tuple(problems), modality=modality, slice_count=len(readable),
        rows=size[0], columns=size[1], pixel_spacing=spacing, slice_spacing=slice_spacing,
        slice_thickness=thickness, orientation=orientation, anatomy=_anatomy(first),
        order=order or (), notes=tuple(notes),
    )


def load_volume_hu(instances: Sequence[bytes], assessment: VolumeAssessment) -> Any:
    """The volume as a (slices, rows, columns) array in stored units after the modality LUT -- Hounsfield units for CT.

    Decoded on demand, never kept: a 300-slice 512x512 CT is ~150 MB even as
    int16. Refuses an unusable series rather than stacking it anyway.
    """
    import numpy as np
    import pydicom
    from pydicom.pixels import apply_modality_lut

    if not assessment.usable:
        raise ValueError(f"series is not a usable volume: {', '.join(assessment.problems)}")
    slices = []
    for index in assessment.order:
        dataset = pydicom.dcmread(io.BytesIO(instances[index]))
        slices.append(np.asarray(apply_modality_lut(dataset.pixel_array, dataset)))
    volume = np.stack(slices)
    if np.all(np.equal(np.mod(volume, 1), 0)) and volume.min() >= -32768 and volume.max() <= 32767:
        return volume.astype(np.int16)
    return volume.astype(np.float32)


@dataclass(frozen=True)
class Pillar0Eligibility:
    eligible: bool
    checkpoint: str | None
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "checkpoint": self.checkpoint, "reason": self.reason}


def pillar0_eligibility(assessment: VolumeAssessment) -> Pillar0Eligibility:
    """Which released Pillar-0 checkpoint could read this series, or why none can -- from modality, anatomy and geometry only."""
    modality = (assessment.modality or "").upper()
    if modality not in ("CT", "MR"):
        return Pillar0Eligibility(False, None, f"modality_not_covered_by_pillar0:{modality or 'unknown'}")
    if not assessment.usable:
        return Pillar0Eligibility(False, None, "not_a_usable_volume")
    anatomy = assessment.anatomy
    if anatomy is None:
        return Pillar0Eligibility(False, None, "anatomy_not_stated_in_dicom")
    if anatomy.startswith("multiple:"):
        return Pillar0Eligibility(False, None, f"anatomy_ambiguous:{anatomy[len('multiple:'):]}")
    if modality == "CT":
        if assessment.orientation != "axial":
            return Pillar0Eligibility(False, None, "ct_series_not_axial")
        checkpoint = {"chest": CHECKPOINT_CHEST_CT, "abdomen": CHECKPOINT_ABDOMEN_CT, "head": CHECKPOINT_HEAD_CT}.get(anatomy)
    else:
        checkpoint = CHECKPOINT_BREAST_MRI if anatomy == "breast" else None
    if checkpoint is None:
        return Pillar0Eligibility(False, None, f"anatomy_not_covered_by_pillar0:{modality.lower()}_{anatomy}")
    return Pillar0Eligibility(True, checkpoint, None)


def preferred_series(candidates: Sequence[tuple[str, VolumeAssessment, Pillar0Eligibility]]) -> dict[str, str]:
    """Per checkpoint, the one series to send: thinnest slices, then most slices, then series UID.

    The Pillar-0 paper selects, per study, the axial series with the lowest
    slice thickness, choosing at random among ties; the tie-break here is
    deterministic instead, so the same case always sends the same series.
    """
    best: dict[str, tuple[tuple[float, int, str], str]] = {}
    for key, assessment, eligibility in candidates:
        if not eligibility.eligible or eligibility.checkpoint is None:
            continue
        thickness = assessment.slice_thickness if assessment.slice_thickness is not None else math.inf
        rank = (thickness, -assessment.slice_count, key)
        if eligibility.checkpoint not in best or rank < best[eligibility.checkpoint][0]:
            best[eligibility.checkpoint] = (rank, key)
    return {checkpoint: key for checkpoint, (_, key) in best.items()}


def export_series_directory(instances: Sequence[bytes], assessment: VolumeAssessment, directory: str | Path) -> list[Path]:
    """Write the de-identified instances, in slice order, as the series directory RAVE's `series_path` expects.

    The only function here that touches disk, and nothing calls it yet:
    the Pillar-0 adapter stays disabled until a deployment exists. A caller
    is responsible for the directory's lifetime.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    order = assessment.order or tuple(range(len(instances)))
    written = []
    for position, index in enumerate(order, start=1):
        path = target / f"{position:05d}.dcm"
        path.write_bytes(instances[index])
        written.append(path)
    return written
