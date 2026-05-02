# Melampo Source Scaffold

This `src/` tree is the current implementation spine for Project Melampo: a multimodal, multimodel, enterprise-grade research scaffold for clinical intuition.

The source tree preserves the original Melampo vision while adding explicit provider-neutral contracts, validation primitives, semantic memory, and an audit-first diagnostic orchestrator.

## Core implementation principles

1. **Melampo-owned final control**: external models provide signals; `MelampoDiagnosticOrchestrator` produces the final structured research output.
2. **Provider neutrality**: Pillar-0, Gemma 4, Claude, Weaviate and Docling are represented through contracts and registries, not hardcoded authority.
3. **No hidden network calls**: live integrations require explicit configuration or infrastructure-specific subclasses.
4. **Clinical safety boundary**: all outputs are research outputs, not validated medical decisions.
5. **Typed and auditable outputs**: every major module emits structured dictionaries with provenance, limitations and governance metadata.
6. **Research-safe expansion**: optional theoretical-quantum and external-provider paths remain isolated behind interfaces.

## Installation profiles

Use the profile that matches the work being performed:

```bash
# Baseline scaffold, tests and CI
python -m pip install -r requirements.txt

# Research profile: clinical metadata, document ingestion, Weaviate retrieval and visualization
python -m pip install -r requirements-research.txt

# Full enterprise profile: all optional imaging, ML, API, quantum and visualization extras
python -m pip install -r requirements-enterprise.txt
```

The same profiles map to `pyproject.toml` extras:

```bash
pip install -e .[dev]
pip install -e .[dev,clinical,document,retrieval,viz]
pip install -e .[enterprise]
```

The full enterprise profile may require platform-specific wheels or GPU/CPU choices for PyTorch and medical-imaging dependencies.

## Canonical flow

```text
ClinicalInferencePipeline
  -> area signals
  -> AreaCoherenceAnalyzer / NeuroDynamicMetrics
  -> DreamTrainer
  -> IntuitionEngine
  -> PipelineCoordinator / DifferentialEngine / PolicyStack
  -> CritiqueLoop
  -> MelampoDiagnosticOrchestrator
```

## Main directories

```text
src/melampo/
├── app.py                         # runtime assembly
├── cli.py                         # prototype and enterprise CLI commands
├── config.py                      # runtime service configuration
├── types.py                       # shared clinical/research datatypes
├── areas/                         # simulated functional areas
├── clinical/                      # clinical helpers
├── data/                          # ingestion, normalization, document processing
├── datasets/                      # public metadata loaders
├── evaluation/                    # quantum gate, calibration, benchmark, prospective validation
├── memory/                        # retrieval, vector memory, Weaviate schema/adapter contracts
├── models/                        # encoders, belief layer, rankers, specialist adapter contracts
├── orchestration/                 # service registry, router, model capability registry
├── reasoning/                     # pipeline, intuition, differential, critique, orchestrator
├── training/                      # dream/replay/counterfactual branch
└── utils/
```

## Current enterprise modules

- `orchestration/model_capability_registry.py`: records Pillar-0, Gemma 4, Claude, Weaviate and Docling roles.
- `reasoning/diagnostic_orchestrator.py`: final audit-first research diagnostic controller.
- `memory/weaviate_schema.py`: object-property clinical memory schema contract.
- `memory/weaviate_adapter.py`: safe Weaviate adapter contract and dry-run/live boundary.
- `models/specialist_adapters.py`: Pillar-0, Gemma 4 and Claude adapter contracts.
- `data/document_processing.py`: Docling-aware document processor with fallback.
- `evaluation/clinical_benchmark.py`: retrospective benchmark runner.
- `evaluation/prospective_validation.py`: prediction-lock prospective validation registry.
- `evaluation/calibration.py`: confidence calibration metrics.

## CLI commands

```bash
melampo-prototype examples/prototype_case.json
melampo-prototype-cxr metadata.csv --limit 5
melampo-prototype-openi metadata.csv --limit 5
melampo-decision-record
melampo-weaviate-schema
```

## Model strategy

- **Pillar-0**: primary radiology / volumetric imaging signal provider for `visual_diagnostic_area`.
- **Gemma 4**: grounded clinical text and agentic reasoning provider for language/context tasks.
- **Claude Healthcare / Life Sciences style critic**: optional external second-opinion, literature and regulatory critic.
- **Weaviate**: semantic object-property memory and ontology-aware RAG backend.
- **Docling**: document intelligence parser for clinical/literature ingestion.

None of these external systems is the final diagnostic authority.

## Validation strategy

Validation is split into:

1. retrospective benchmark evaluation;
2. prospective prediction-lock validation;
3. calibration against real-world correctness;
4. safety and abstention analysis;
5. clinical/regulatory review outside this scaffold.

See `docs/validation/clinical_benchmarking_and_prospective_validation.md`.
