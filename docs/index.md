# Project Melampo Documentation Index

This index links the current documentation set for Project Melampo's enterprise-grade multimodal, multimodel clinical-intuition research scaffold.

## Start here

- [`../README.md`](../README.md) - project overview, architecture, CLI and roadmap.
- [`architecture.md`](architecture.md) - current architecture and module responsibilities.
- [`final_treatise_decision_record.md`](final_treatise_decision_record.md) - canonical decisions for the final treatise.

## Enterprise AI / RAG direction

- [`enterprise_ai_rag_evolution.md`](enterprise_ai_rag_evolution.md) - enterprise AI/RAG evolution plan.
- [`core_consolidation_map.md`](core_consolidation_map.md) - canonical core modules and consolidation priorities.

## Validation and governance

- [`validation/clinical_benchmarking_and_prospective_validation.md`](validation/clinical_benchmarking_and_prospective_validation.md) - benchmark, prospective validation and calibration framework.
- [`model_cards/melampo_model_stack_card.md`](model_cards/melampo_model_stack_card.md) - model stack card.
- [`dataset_cards/clinical_memory_dataset_card.md`](dataset_cards/clinical_memory_dataset_card.md) - clinical memory dataset card.

## Source implementation

- [`../src/README.md`](../src/README.md) - source-tree overview.
- `src/melampo/reasoning/diagnostic_orchestrator.py` - final Melampo diagnostic orchestrator.
- `src/melampo/orchestration/model_capability_registry.py` - model decision registry.
- `src/melampo/memory/weaviate_schema.py` - Weaviate object-property schema contract.
- `src/melampo/evaluation/clinical_benchmark.py` - retrospective benchmark primitives.
- `src/melampo/evaluation/prospective_validation.py` - prospective validation primitives.
- `src/melampo/evaluation/calibration.py` - confidence calibration primitives.

## Safety statement

Melampo is a research scaffold. It is not a validated medical device and must not be used for autonomous clinical diagnosis or patient-care decisions without formal validation, regulatory review and human specialist oversight.
