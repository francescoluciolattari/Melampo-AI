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
if every instance shares one orientation, size, pixel spacing and frame of
reference, slice positions (projected on the slice normal, not
InstanceNumber, which scanners do not guarantee to be spatial) are
distinct, evenly spaced and on one straight line, and there are enough of
them. Each failed condition is returned by name.

**Geometry for measurement, in millimetres.** Every measurement on a
volume -- a lesion's centre, a distance, a volume -- is only as right as
the mapping from voxel indices to the patient. `VolumeAssessment.affine`
is that mapping, built from the attributes the DICOM standard defines for
it (PS3.3 C.7.6.2) and nothing it calls nominal:

- ImagePositionPatient is the centre of the first voxel, in mm;
  ImageOrientationPatient gives the row and column direction cosines;
  PixelSpacing is *row spacing first, then column spacing* -- swapping
  them silently distorts every in-plane measurement on non-square pixels.
- The slice step is the vector from the first to the last slice position
  divided by N-1, as NiBabel does, not SliceThickness (nominal) or
  SliceLocation (relative to an unspecified reference). On a tilted gantry
  that step is not perpendicular to the image plane: the volume is
  sheared. GantryDetectorTilt is "not intended for mathematical
  computations", so the tilt is measured from the positions instead
  (`tilt_degrees`), reported as a note, and carried in the affine -- any
  resampling that assumes orthogonal axes would misplace every voxel
  away from the first slice.
- SliceThickness is compared with the measured spacing: slices thinner
  than their spacing leave anatomy unsampled between them (QIBA's
  volumetry profile requires spacing <= thickness); thicker ones overlap,
  which is a normal reconstruction choice.
- FrameOfReferenceUID: series sharing it are spatially related (C.7.4);
  series that do not share it need a registration before any comparison.

`measurement_precision` turns this into a declared level, so a precision
is never claimed that the acquisition cannot support. The thresholds are
QIBA's (<= 1.25 mm, no gaps) and those of a sub-voxel simulation recorded
in docs/imaging_decision_record.md.

**Not done here:** calling Pillar-0. The adapter stays disabled until a
deployment exists. `export_series_directory` writes the de-identified
series directory that RAVE's `series_path` entries point to (RAVE's own
input is a CSV listing such paths, which nothing writes yet), and is the
only function here that writes to disk.
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
    "KVP", "ConvolutionKernel", "ContrastBolusAgent", "Manufacturer", "ManufacturerModelName",
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
# Relative tolerance for "evenly spaced": 1% of the median spacing. The same
# fraction bounds how far a slice may sit off the line through the others,
# and how far thickness and spacing may differ before a gap or an overlap
# is reported.
SPACING_TOLERANCE = 0.01
ORIENTATION_TOLERANCE = 1e-3
# Below this angle between the slice step and the plane normal the volume
# is treated as orthogonal; above it, as sheared (tilted gantry).
TILT_TOLERANCE_DEGREES = 0.01
# Floor for position tolerances, in mm: DICOM decimal strings are commonly
# rounded to a few micrometres.
POSITION_TOLERANCE_FLOOR_MM = 1e-3

# Declared measurement precision (see measurement_precision).
PRECISION_HIGH = "high"
PRECISION_REDUCED = "reduced"
PRECISION_LOW = "low"
PRECISION_NONE = "none"
# QIBA CT small-nodule volumetry profile (2023): reconstructed slice
# thickness <= 1.25 mm, slice interval <= thickness.
QIBA_MAX_SLICE_THICKNESS_MM = 1.25
# Sub-voxel simulation (docs/imaging_decision_record.md): up to 2.5 mm the
# centre of a 4-10 mm high-contrast lesion is still located to <0.06 mm
# along z; at 5 mm the error reaches ~1.1 mm and small-lesion volume +-43%.
REDUCED_MAX_SLICE_THICKNESS_MM = 2.5

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
    # Voxel (column i, row j, slice k in `order`) -> patient LPS mm, 4x4 row-major.
    affine: tuple[tuple[float, float, float, float], ...] | None = None
    tilt_degrees: float | None = None
    frame_of_reference_uid: str | None = None
    convolution_kernel: str | None = None
    manufacturer_model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable, "problems": list(self.problems), "modality": self.modality,
            "slice_count": self.slice_count, "rows": self.rows, "columns": self.columns,
            "pixel_spacing": list(self.pixel_spacing) if self.pixel_spacing else None,
            "slice_spacing": self.slice_spacing, "slice_thickness": self.slice_thickness,
            "orientation": self.orientation, "anatomy": self.anatomy, "notes": list(self.notes),
            "affine": [list(row) for row in self.affine] if self.affine else None,
            "tilt_degrees": self.tilt_degrees, "frame_of_reference_uid": self.frame_of_reference_uid,
            "convolution_kernel": self.convolution_kernel, "manufacturer_model": self.manufacturer_model,
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
    frames_of_reference = {str(header.get("FrameOfReferenceUID", "") or "") for _, header in readable}
    if len(frames_of_reference) > 1:
        problems.append("mixed_frame_of_reference")
    if any(item is None for item in orientations) or any(item is None for item in positions):
        problems.append("missing_geometry")
        return _assessment(False, problems, notes, modality, readable, sizes, spacings, None, None, None, first)
    reference = orientations[0]
    if any(max(abs(a - b) for a, b in zip(item, reference, strict=True)) > ORIENTATION_TOLERANCE for item in orientations):
        problems.append("inconsistent_orientation")

    row, column = reference[:3], reference[3:]
    normal = _cross(row, column)
    distances = [_dot(position, normal) for position in positions]
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

    geometry = _slice_geometry([positions[index] for index in order], normal, slice_spacing, problems, notes)
    _check_thickness(readable, slice_spacing, notes)
    for keyword, note in (("ConvolutionKernel", "inconsistent_convolution_kernel"), ("ManufacturerModelName", "inconsistent_manufacturer_model")):
        if len({_text(header.get(keyword)) for _, header in readable}) > 1:
            notes.append(note)
    affine = None
    spacing = next(iter(spacings)) if len(spacings) == 1 else None
    # A single step vector describes every slice only if the slices form one
    # regular grid; otherwise the affine would misplace the inner slices.
    if geometry is not None and spacing is not None and not set(problems) & _GRID_PROBLEMS:
        step, origin = geometry["step"], geometry["origin"]
        row_spacing, column_spacing = spacing  # PixelSpacing: row spacing first, then column spacing
        affine = (
            (row[0] * column_spacing, column[0] * row_spacing, step[0], origin[0]),
            (row[1] * column_spacing, column[1] * row_spacing, step[1], origin[1]),
            (row[2] * column_spacing, column[2] * row_spacing, step[2], origin[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    return _assessment(
        not problems, problems, notes, modality, readable, sizes, spacings, slice_spacing,
        _orientation_label(normal), tuple(readable[index][0] for index in order), first,
        affine=affine, tilt_degrees=geometry["tilt_degrees"] if geometry else None,
        frame_of_reference_uid=(next(iter(frames_of_reference)) or None) if len(frames_of_reference) == 1 else None,
    )


_GRID_PROBLEMS = frozenset({
    "inconsistent_orientation", "inconsistent_or_missing_pixel_spacing", "inconsistent_image_size",
    "duplicate_slice_positions", "uneven_slice_spacing", "slice_positions_not_collinear",
    "mixed_series", "mixed_frame_of_reference",
})


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _slice_geometry(
    ordered_positions: Sequence[tuple[float, ...]],
    normal: tuple[float, float, float],
    slice_spacing: float | None,
    problems: list[str],
    notes: list[str],
) -> dict[str, Any] | None:
    """Slice step vector, tilt and collinearity, from the positions in slice order.

    The step is (last - first) / (N - 1), never a thickness or spacing tag.
    Every position must lie on the line through the first one along that
    step: a slice displaced sideways is not part of the same sampled grid.
    """
    count = len(ordered_positions)
    if count < 2 or not slice_spacing:
        return None
    first, last = ordered_positions[0], ordered_positions[-1]
    step = tuple((b - a) / (count - 1) for a, b in zip(first, last, strict=True))
    length = math.sqrt(_dot(step, step))
    if length <= 0:
        return None
    unit = tuple(value / length for value in step)
    tolerance = max(SPACING_TOLERANCE * slice_spacing, POSITION_TOLERANCE_FLOOR_MM)
    for position in ordered_positions:
        offset = tuple(p - f for p, f in zip(position, first, strict=True))
        along = _dot(offset, unit)
        sideways = math.sqrt(max(_dot(offset, offset) - along * along, 0.0))
        if sideways > tolerance:
            problems.append("slice_positions_not_collinear")
            break
    cosine = min(abs(_dot(unit, normal)) / (math.sqrt(_dot(normal, normal)) or 1.0), 1.0)
    tilt = math.degrees(math.acos(cosine))
    if tilt > TILT_TOLERANCE_DEGREES:
        notes.append("sheared_volume_gantry_tilt")
    return {"step": step, "origin": tuple(first), "tilt_degrees": round(tilt, 4)}


def _check_thickness(readable: Sequence[tuple[int, Any]], slice_spacing: float | None, notes: list[str]) -> None:
    """Thickness against the measured spacing, and the SpacingBetweenSlices tag against it -- reported, not refused."""
    thicknesses = set()
    for _, header in readable:
        try:
            thicknesses.add(round(float(header.get("SliceThickness")), 4))
        except (TypeError, ValueError):
            continue
    if len(thicknesses) > 1:
        notes.append("inconsistent_slice_thickness")
    if slice_spacing:
        tolerance = max(SPACING_TOLERANCE * slice_spacing, POSITION_TOLERANCE_FLOOR_MM)
        if len(thicknesses) == 1:
            thickness = next(iter(thicknesses))
            if thickness < slice_spacing - tolerance:
                notes.append("gaps_between_slices")
            elif thickness > slice_spacing + tolerance:
                notes.append("overlapping_slices")
        tagged = set()
        for _, header in readable:
            try:
                tagged.add(round(float(header.get("SpacingBetweenSlices")), 4))
            except (TypeError, ValueError):
                continue
        if tagged and any(abs(value - slice_spacing) > tolerance for value in tagged):
            notes.append("spacing_between_slices_tag_disagrees")


def _frames(header: Any) -> int:
    try:
        return int(header.get("NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _assessment(usable, problems, notes, modality, readable, sizes, spacings, slice_spacing, orientation, order, first,
                *, affine=None, tilt_degrees=None, frame_of_reference_uid=None):
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
        order=order or (), notes=tuple(notes), affine=affine, tilt_degrees=tilt_degrees,
        frame_of_reference_uid=frame_of_reference_uid,
        convolution_kernel=_text(first.get("ConvolutionKernel")),
        manufacturer_model=_text(first.get("ManufacturerModelName")),
    )


def _text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, list | tuple) or type(value).__name__ == "MultiValue":
        return "\\".join(str(item).strip() for item in value) or None
    return str(value).strip() or None


def index_to_patient_mm(assessment: VolumeAssessment, slice_index: float, row: float, column: float) -> tuple[float, float, float]:
    """Patient coordinates (LPS, mm) of a voxel of `load_volume_hu`'s array -- indices may be fractional (sub-voxel).

    The array is (slices, rows, columns) in `assessment.order`; the affine
    is written for (column, row, slice), as DICOM and NiBabel do. The
    affine exists only for a regular grid, but a series can have one and
    still be unusable as a volume for other reasons (e.g. too few slices):
    positions are then correct, volumetric measurements are not supported
    (see measurement_precision).
    """
    if assessment.affine is None:
        raise ValueError("no patient geometry: " + (", ".join(assessment.problems) or "affine not available"))
    vector = (column, row, slice_index, 1.0)
    return tuple(sum(value * factor for value, factor in zip(line, vector, strict=True)) for line in assessment.affine[:3])


def share_frame_of_reference(first: VolumeAssessment, second: VolumeAssessment) -> bool:
    """Whether two series are spatially related by the scanner itself (PS3.3 C.7.4). Unknown is False, never assumed."""
    return bool(first.frame_of_reference_uid) and first.frame_of_reference_uid == second.frame_of_reference_uid


@dataclass(frozen=True)
class MeasurementPrecision:
    """The precision a series can support, and why -- declared, so nothing claims more than the acquisition allows."""

    level: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "reasons": list(self.reasons)}


def measurement_precision(assessment: VolumeAssessment) -> MeasurementPrecision:
    """high: QIBA volumetry conditions; reduced: sub-voxel still reliable along z; low: planar measures only; none: no geometry."""
    if not assessment.usable or assessment.affine is None:
        return MeasurementPrecision(PRECISION_NONE, ("not_a_usable_volume", *assessment.problems))
    reasons: list[str] = []
    notes = set(assessment.notes)
    if (assessment.modality or "").upper() == "MR":
        reasons.append("mr_intensities_not_absolute")
    if "sheared_volume_gantry_tilt" in notes:
        reasons.append("sheared_volume_measure_through_affine_only")
    thickness = assessment.slice_thickness
    if thickness is None:
        return MeasurementPrecision(PRECISION_LOW, ("slice_thickness_unknown", *reasons))
    if "gaps_between_slices" in notes:
        return MeasurementPrecision(PRECISION_LOW, ("gaps_between_slices", *reasons))
    if "inconsistent_slice_thickness" in notes:
        return MeasurementPrecision(PRECISION_LOW, ("inconsistent_slice_thickness", *reasons))
    if thickness <= QIBA_MAX_SLICE_THICKNESS_MM:
        return MeasurementPrecision(PRECISION_HIGH, tuple(reasons))
    if thickness <= REDUCED_MAX_SLICE_THICKNESS_MM:
        return MeasurementPrecision(PRECISION_REDUCED, (f"slice_thickness_above_{QIBA_MAX_SLICE_THICKNESS_MM}mm", *reasons))
    return MeasurementPrecision(PRECISION_LOW, (f"slice_thickness_above_{REDUCED_MAX_SLICE_THICKNESS_MM}mm", *reasons))


@dataclass(frozen=True)
class AcquisitionComparison:
    differences: tuple[str, ...]
    qiba_comparable: bool  # same thickness, kernel and scanner model, as QIBA requires for change measurement

    def as_dict(self) -> dict[str, Any]:
        return {"differences": list(self.differences), "qiba_comparable": self.qiba_comparable}


def compare_acquisitions(earlier: VolumeAssessment, later: VolumeAssessment) -> AcquisitionComparison:
    """What differs between two exams' acquisitions in ways that change measurements taken on them.

    An unknown value on either side counts as a difference: comparability
    is shown, never assumed.
    """
    differences = []

    def differ(name: str, left: Any, right: Any, *, numeric: bool = False) -> bool:
        if left is None or right is None:
            differences.append(f"{name}_unknown")
            return True
        different = abs(float(left) - float(right)) > POSITION_TOLERANCE_FLOOR_MM if numeric else left != right
        if different:
            differences.append(f"different_{name}:{left}->{right}")
        return different

    critical = [
        differ("slice_thickness", earlier.slice_thickness, later.slice_thickness, numeric=True),
        differ("convolution_kernel", earlier.convolution_kernel, later.convolution_kernel),
        differ("manufacturer_model", earlier.manufacturer_model, later.manufacturer_model),
    ]
    differ("modality", earlier.modality, later.modality)
    differ("pixel_spacing", earlier.pixel_spacing, later.pixel_spacing)
    same_modality = earlier.modality is not None and earlier.modality == later.modality
    return AcquisitionComparison(tuple(differences), qiba_comparable=not any(critical) and same_modality)


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
