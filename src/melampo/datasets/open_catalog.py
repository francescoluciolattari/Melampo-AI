from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenClinicalDataset:
    """Metadata for open or credentialed clinical datasets usable by Melampo."""

    name: str
    modality: str
    access_policy: str
    license_or_terms: str
    recommended_use: str
    caution: str
    url: str

    def describe(self) -> dict:
        return {
            "name": self.name,
            "modality": self.modality,
            "access_policy": self.access_policy,
            "license_or_terms": self.license_or_terms,
            "recommended_use": self.recommended_use,
            "caution": self.caution,
            "url": self.url,
        }


OPEN_CLINICAL_DATASETS: tuple[OpenClinicalDataset, ...] = (
    OpenClinicalDataset(
        name="NIH ChestX-ray14",
        modality="chest_xray_labels",
        access_policy="public_download_with_terms",
        license_or_terms="NIH dataset terms for academic/research use",
        recommended_use="imaging-led prototype cases, weak-label radiology experiments, multi-label thoracic finding scaffolds",
        caution="labels are NLP-derived from reports and should be treated as weak supervision, not adjudicated ground truth",
        url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
    ),
    OpenClinicalDataset(
        name="Open-i / Indiana University Chest X-rays",
        modality="chest_xray_images_and_reports",
        access_policy="open_access_collection_with_license_terms",
        license_or_terms="CC BY-NC-ND 4.0 on common mirrors; verify original Open-i terms before redistribution",
        recommended_use="paired radiology report and image style cases for language-listening plus visual diagnostic integration",
        caution="respect redistribution restrictions and avoid committing downloaded images or reports into this repository",
        url="https://openi.nlm.nih.gov/",
    ),
    OpenClinicalDataset(
        name="The Cancer Imaging Archive",
        modality="oncology_dicom_imaging",
        access_policy="open_access_collections_with_collection_specific_terms",
        license_or_terms="collection-specific TCIA terms and DOI citation requirements",
        recommended_use="oncology imaging collections, DICOM workflow tests, imaging-led multimodal cases",
        caution="use collection metadata and licenses carefully; do not mix collections without tracking provenance",
        url="https://www.cancerimagingarchive.net/",
    ),
    OpenClinicalDataset(
        name="MIMIC-IV",
        modality="deidentified_ehr_icu_hospital",
        access_policy="credentialed_access_with_training_and_dua",
        license_or_terms="PhysioNet Credentialed Health Data License / DUA",
        recommended_use="structured EHR, ICU/hospital trajectories, policy and reasoning experiments after credentialing",
        caution="not open-download; requires credentialing, training, and DUA compliance before use",
        url="https://physionet.org/content/mimiciv/",
    ),
    OpenClinicalDataset(
        name="MedQuAD",
        modality="medical_question_answering",
        access_policy="public_repository",
        license_or_terms="CC BY 4.0 according to dataset repository",
        recommended_use="retrieval, grounding, clinical question-answering, patient-education style text cases",
        caution="QA content is educational and should not be treated as patient-level clinical records",
        url="https://github.com/abachaa/MedQuAD",
    ),
)


def list_open_clinical_datasets() -> list[dict]:
    """Return known open/credentialed clinical dataset metadata for prototype planning."""

    return [dataset.describe() for dataset in OPEN_CLINICAL_DATASETS]


def datasets_by_modality(modality: str) -> list[dict]:
    """Filter dataset metadata by a simple modality string."""

    return [dataset.describe() for dataset in OPEN_CLINICAL_DATASETS if dataset.modality == modality]
