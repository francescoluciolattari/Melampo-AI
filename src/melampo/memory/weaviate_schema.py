from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WeaviateProperty:
    name: str
    data_type: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "data_type": self.data_type, "description": self.description}


@dataclass(frozen=True, slots=True)
class WeaviateReference:
    name: str
    target: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "target": self.target, "description": self.description}


@dataclass(frozen=True, slots=True)
class WeaviateClassSchema:
    name: str
    description: str
    properties: tuple[WeaviateProperty, ...] = field(default_factory=tuple)
    references: tuple[WeaviateReference, ...] = field(default_factory=tuple)
    named_vectors: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "class": self.name,
            "description": self.description,
            "properties": [property_.as_dict() for property_ in self.properties],
            "references": [reference.as_dict() for reference in self.references],
            "named_vectors": list(self.named_vectors),
        }


@dataclass(slots=True)
class MelampoWeaviateSchema:
    """Schema contract for Melampo's semantic clinical memory.

    This module intentionally does not import or call Weaviate. It defines the
    governed object-property ontology that a real adapter can materialize.
    """

    version: str = "0.1.0"

    def classes(self) -> list[WeaviateClassSchema]:
        return [
            WeaviateClassSchema(
                name="Symptom",
                description="Clinical symptom or complaint with ontology references and links to candidate pathologies.",
                properties=(
                    WeaviateProperty("name", "text", "Human-readable symptom name."),
                    WeaviateProperty("description", "text", "Clinical description or normalized narrative."),
                    WeaviateProperty("snomed_code", "text", "SNOMED-like concept code when available."),
                    WeaviateProperty("temporal_pattern", "text", "Acute, chronic, recurrent, progressive or unknown pattern."),
                ),
                references=(
                    WeaviateReference("suggestsPathology", "Pathology", "Pathologies supported or suggested by this symptom."),
                    WeaviateReference("appearsInCase", "ClinicalCase", "Cases where this symptom appears."),
                ),
                named_vectors=("symptom_text_vector",),
            ),
            WeaviateClassSchema(
                name="Pathology",
                description="Candidate pathology or disease concept linked to symptoms, imaging patterns and risk factors.",
                properties=(
                    WeaviateProperty("name", "text", "Pathology name."),
                    WeaviateProperty("description", "text", "Clinical description and differential reasoning notes."),
                    WeaviateProperty("snomed_code", "text", "SNOMED-like concept code when available."),
                    WeaviateProperty("icd_code", "text", "ICD-like billing/classification code when available."),
                    WeaviateProperty("prevalence_band", "text", "Low, medium, high or unknown prevalence band."),
                ),
                references=(
                    WeaviateReference("hasSymptom", "Symptom", "Symptoms associated with this pathology."),
                    WeaviateReference("hasImagingPattern", "ImagingFinding", "Imaging patterns associated with this pathology."),
                    WeaviateReference("hasRiskFactor", "EpidemiologicalFactor", "Risk factors associated with this pathology."),
                    WeaviateReference("supportedByDocument", "ClinicalDocument", "Guidelines, articles or sources supporting this pathology."),
                ),
                named_vectors=("pathology_text_vector", "ontology_context_vector"),
            ),
            WeaviateClassSchema(
                name="ClinicalCase",
                description="A governed patient/case object used as post-training memory and RAG context.",
                properties=(
                    WeaviateProperty("case_id", "text", "Stable case identifier."),
                    WeaviateProperty("demographics", "object", "Age, sex and other structured demographics."),
                    WeaviateProperty("provenance", "object", "Source, license, de-identification and governance metadata."),
                    WeaviateProperty("learning_status", "text", "candidate, promoted, rejected, needs_review or retired."),
                ),
                references=(
                    WeaviateReference("hasSymptom", "Symptom", "Symptoms linked to this case."),
                    WeaviateReference("hasReport", "ClinicalDocument", "Reports or documents linked to this case."),
                    WeaviateReference("hasImage", "ImagingStudy", "Imaging studies linked to this case."),
                    WeaviateReference("hasDifferential", "Pathology", "Differential hypotheses linked to this case."),
                ),
                named_vectors=("case_summary_vector", "case_trace_vector"),
            ),
            WeaviateClassSchema(
                name="ImagingStudy",
                description="Imaging object with modality, visual vector slots, provenance and findings.",
                properties=(
                    WeaviateProperty("study_id", "text", "Stable imaging study identifier."),
                    WeaviateProperty("modality", "text", "CT, MRI, CR, XR, US, PET or unknown."),
                    WeaviateProperty("series_metadata", "object", "Local or PACS series metadata."),
                    WeaviateProperty("source_path_count", "int", "Number of local or referenced image paths."),
                ),
                references=(
                    WeaviateReference("belongsToCase", "ClinicalCase", "Case that owns this imaging study."),
                    WeaviateReference("hasFinding", "ImagingFinding", "Findings detected or suspected in this study."),
                ),
                named_vectors=("image_vector", "report_alignment_vector"),
            ),
            WeaviateClassSchema(
                name="ImagingFinding",
                description="Visual/radiological finding linked to imaging studies and pathologies.",
                properties=(
                    WeaviateProperty("name", "text", "Finding name."),
                    WeaviateProperty("description", "text", "Finding description and location if available."),
                    WeaviateProperty("anatomical_region", "text", "Region or organ system."),
                    WeaviateProperty("confidence", "number", "Model confidence or calibrated score."),
                ),
                references=(
                    WeaviateReference("supportsPathology", "Pathology", "Pathologies supported by this finding."),
                    WeaviateReference("foundInStudy", "ImagingStudy", "Imaging study containing this finding."),
                ),
                named_vectors=("finding_text_vector", "finding_visual_vector"),
            ),
            WeaviateClassSchema(
                name="ClinicalDocument",
                description="Literature, guideline, report or clinical document chunk with provenance.",
                properties=(
                    WeaviateProperty("source", "text", "Document source or URI."),
                    WeaviateProperty("section", "text", "Document section or heading."),
                    WeaviateProperty("page", "int", "Page number when available."),
                    WeaviateProperty("text", "text", "Chunk text."),
                    WeaviateProperty("publication_date", "date", "Publication or update date when available."),
                    WeaviateProperty("license", "text", "License or access class."),
                ),
                references=(
                    WeaviateReference("mentionsSymptom", "Symptom", "Symptoms mentioned by this document."),
                    WeaviateReference("mentionsPathology", "Pathology", "Pathologies mentioned by this document."),
                    WeaviateReference("mentionsFinding", "ImagingFinding", "Imaging findings mentioned by this document."),
                ),
                named_vectors=("document_text_vector", "document_layout_vector"),
            ),
            WeaviateClassSchema(
                name="EpidemiologicalFactor",
                description="Exposure, demographic or prevalence factor used to shape pre-test probability.",
                properties=(
                    WeaviateProperty("name", "text", "Risk or epidemiological factor name."),
                    WeaviateProperty("description", "text", "Factor description."),
                    WeaviateProperty("region", "text", "Geographic region when relevant."),
                    WeaviateProperty("prevalence_band", "text", "Prevalence band or qualitative likelihood."),
                ),
                references=(
                    WeaviateReference("increasesRiskOf", "Pathology", "Pathologies whose probability is increased by this factor."),
                    WeaviateReference("appearsInCase", "ClinicalCase", "Cases where this factor appears."),
                ),
                named_vectors=("epidemiology_text_vector",),
            ),
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "melampo_semantic_clinical_memory",
            "version": self.version,
            "backend": "Weaviate",
            "classes": [class_schema.as_dict() for class_schema in self.classes()],
            "governance": {
                "dream_generated_records_default_status": "candidate",
                "promotion_requires": [
                    "rational_control_validation",
                    "source_or_synthetic_provenance",
                    "audit_trace",
                    "human_review_before_clinical_use",
                ],
                "clinical_warning": "Research memory schema; not a validated medical device.",
            },
        }

    def class_names(self) -> list[str]:
        return [class_schema.name for class_schema in self.classes()]
