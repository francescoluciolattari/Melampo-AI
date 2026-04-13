# Melampo Source Scaffold

This `src/` tree is the rebuilt starting point for **Project Melampo**, derived from the consolidated scientific architecture described in the project treatise. The architecture couples:

- 3D and pathology perception modules
- multimodal fusion and model orchestration
- episodic, semantic, and prototypical memory
- differential reasoning and metacognitive control
- uncertainty-aware abstention and safety governance
- generative replay / statistical dreaming
- an optional theoretical-quantum research layer

## Why Python

Python is used instead of notebooks as the default implementation language because it is easier to version, test, lint, orchestrate, deploy, and connect to medical data standards, ML frameworks, and validation pipelines.

## Design Principles

1. **Production-friendly scaffolding**: package layout, clear boundaries, explicit contracts.
2. **Research-safe expansion**: optional theoretical-quantum modules remain isolated behind interfaces.
3. **Clinical interoperability**: explicit placeholders for FHIR, DICOM, KG-RAG, MCP, and A2A.
4. **Model-provider neutrality**: all external services use explicit placeholders such as `api_for_service_xyz`.
5. **English-only code comments and naming** as requested.

## High-Level Layout

```text
src/
├── README.md
├── resources/
│   ├── model_registry.example.yaml
│   ├── prompts/
│   └── schemas/
└── melampo/
    ├── app.py
    ├── config.py
    ├── types.py
    ├── clinical/
    ├── data/
    ├── models/
    ├── memory/
    ├── orchestration/
    ├── reasoning/
    ├── training/
    ├── evaluation/
    └── utils/
```

## Notes

- External model APIs are not hardcoded. They are described in `resources/model_registry.example.yaml` and loaded through configuration.
- All advanced modules are provided as safe, typed scaffolds that can later be bound to MONAI, PyTorch, Hugging Face, vLLM, LangGraph, MCP servers, custom PACS services, or institutional infrastructure.
- The folder is intentionally clean and self-contained so it can replace the previous `src/` tree as the new project baseline.
