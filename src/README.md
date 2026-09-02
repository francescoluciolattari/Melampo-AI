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

## Retrieval strategy modules

Foundation for the RLM-on-Memory migration recorded in
`docs/rlm_on_memory_decision_record.md`. Recursive retrieval replaces the
retrieval strategy only; the Weaviate memory substrate is unchanged and becomes
more central, since typed relations are what give a recursive strategy its
navigation affordances.

- `memory/context_environment.py`: navigable case environment with mandatory character-level provenance and an instrumented coverage ledger.
- `memory/retrieval_contract.py`: shared contract for one-shot and recursive strategies, with a validator for the silent failure modes.
- `reasoning/retrieval_reconciliation.py`: deterministic dual-path reconciliation; path divergence becomes an empirical conflict signal.
- `training/hypothesis_channel.py`: dream candidates delivered as exclusion hypotheses under an indeterminacy gate, structurally isolated from the evidence path.
- `memory/concept_resolution.py`: ontology parsing, term index, deterministic surface-to-concept resolution; modifier roles read from the hierarchy; separates a resolution gap from a coverage gap.
- `memory/ontology_import.py`: HPO annotations as interval edges, published frequency ranges preserved rather than collapsed to points.
- `memory/graph_coverage.py`: coverage against a reference relation set; guards evaluations that would measure the knowledge base while appearing to measure the architecture.
- `reasoning/family_history.py`: family history as screening hypothesis and prior modifier, never as a patient finding.
- `memory/assertion.py`: deterministic assertion detection — polarity, certainty, experiencer, temporality, source — producing an interval and an epistemic state rather than a scalar.
- `reasoning/findings_boundary.py`: enforced boundary admitting only current, asserted findings of this patient; every rejection carries its route.

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
- **Gemma 4**: grounded clinical text and agentic reasoning provider for language/context tasks. **Open item:** no verifiable downloadable artefact carries this name, which is a traceability defect for the model card. Replacement candidates are Gemma-3-27B-MeditronFO (fully open pipeline, an audit advantage) and MedGemma 1.5 27B. Identifiers in code are unchanged pending that decision; see `docs/rlm_on_memory_decision_record.md`.
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
