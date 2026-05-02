# Contributing to Project Melampo

Thank you for considering a contribution to Project Melampo.

Melampo is an ambitious multimodal, multimodel research scaffold for clinical intuition. Contributions are welcome, but every contribution must respect the project's safety, governance and enterprise-grade architecture boundaries.

## Clinical safety boundary

Melampo is not a validated medical device and must not be used for autonomous diagnosis or patient-care decisions.

Contributions must not:

- claim clinical validity without evidence;
- introduce autonomous diagnosis language;
- bypass abstention, escalation, provenance or audit paths;
- add hidden network calls;
- promote dream-generated or synthetic data into clinical truth;
- include identifiable patient data.

## Current architecture to preserve

The canonical chain is:

```text
clinical_pipeline
  -> area_coherence / neuro_dynamics
  -> dream_trainer
  -> intuition_engine
  -> differential_engine
  -> policy_stack / critique_loop
  -> diagnostic_orchestrator
```

External models such as Pillar-0, Gemma 4, Claude-style critics, Weaviate and Docling are specialist providers or infrastructure components. They are not final diagnostic authorities.

Final research output authority belongs to `MelampoDiagnosticOrchestrator`.

## High-priority contribution areas

### 1. Enterprise-safe model adapters

- Pillar-0 radiology / volumetric imaging adapter.
- Gemma 4 grounded clinical text / workflow adapter.
- Claude-style optional critique adapter.
- Local/private model backends behind provider-neutral contracts.

Adapter requirements:

- no hidden network calls;
- explicit configuration;
- structured response;
- confidence and uncertainty;
- provenance;
- limitations;
- tests for disabled, dry-run and configured paths.

### 2. Semantic memory and ontology RAG

- Weaviate infrastructure subclass for schema materialization.
- Governed upsert and search implementations.
- Object-property relation handling for symptoms, pathologies, cases, imaging studies, findings and documents.
- Ontology references such as SNOMED-like, ICD-like, RadLex-like or LOINC-like metadata.

### 3. Document intelligence

- Docling source catalog integration.
- Clinical section-aware chunking.
- Table/formula preservation.
- Page/section provenance.
- Document-to-Weaviate ingestion flow.

### 4. Clinical validation infrastructure

- Dataset-specific benchmark loaders.
- Prospective validation storage backends.
- Calibration and abstention analysis.
- Bias and slice reports.
- Model cards and dataset cards.

### 5. Core reasoning improvements

- Area signal contracts.
- Evidence ranking.
- Support/contradiction extraction.
- Dream branch curriculum.
- Neuro-dynamic calibration.
- Policy thresholds and escalation semantics.

## Code standards

Contributions should maintain:

- typed public contracts where practical;
- deterministic fallback behavior;
- provider-neutral interfaces;
- structured outputs;
- provenance and audit metadata;
- explicit safety limitations;
- backward-compatible CLI behavior;
- tests for normal, empty and adverse paths.

## Documentation standards

Any architectural change should update relevant documentation:

- `README.md`
- `src/README.md`
- `docs/architecture.md`
- `docs/core_consolidation_map.md`
- `docs/final_treatise_decision_record.md`
- model cards or dataset cards when applicable

## Testing

Run:

```bash
pip install -e .[dev]
pytest -q
melampo-decision-record
melampo-weaviate-schema
```

Optional enterprise dependencies may be installed with:

```bash
pip install -e .[enterprise]
```

## Pull request checklist

Before opening a pull request, confirm:

- [ ] no hidden network calls were added;
- [ ] no patient-identifiable data was committed;
- [ ] clinical safety language remains accurate;
- [ ] external models are not treated as final arbiters;
- [ ] provenance and limitations are preserved;
- [ ] tests were added or updated;
- [ ] documentation was updated;
- [ ] CLI behavior remains backward compatible.

## Submitting changes

1. Fork the repository.
2. Create a feature branch.
3. Commit focused changes.
4. Run tests.
5. Open a pull request with a clear summary, risk analysis and validation notes.

## Code of conduct

We are committed to a friendly, safe and welcoming environment for all contributors.

Let's build the future of safe, auditable Medical AI research together.
