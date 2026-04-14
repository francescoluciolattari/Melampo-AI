from __future__ import annotations

FHIR_RESOURCE_TYPES = {
    "patient",
    "observation",
    "diagnostic_report",
    "imaging_study",
    "condition",
    "procedure",
    "encounter",
}

DICOM_MINIMUM_TAGS = {
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "BodyPartExamined",
    "PatientID",
}

TERMINOLOGY_SYSTEMS = {
    "SNOMED_CT": "http://snomed.info/sct",
    "LOINC": "http://loinc.org",
    "ICD_10": "http://hl7.org/fhir/sid/icd-10",
}
