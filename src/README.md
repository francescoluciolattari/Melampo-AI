# Melampo Source Scaffold

This `src/` tree is the current implementation spine for Project Melampo: a multimodal, multimodel, enterprise-grade research scaffold for clinical intuition.

The source tree preserves the original Melampo vision while adding explicit provider-neutral contracts, validation primitives, semantic memory, and an audit-first diagnostic orchestrator.

## Core implementation principles

1. **Melampo-owned final control**: external models provide signals; `MelampoDiagnosticOrchestrator` produces the final structured research output.
2. **Provider neutrality**: Pillar-0, Gemma 4, Claude, Weaviate and Nemotron-Parse are represented through contracts and registries, not hardcoded authority.
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

- `orchestration/model_capability_registry.py`: records Pillar-0, Gemma 4, Claude, Weaviate and Nemotron-Parse roles.
- `reasoning/diagnostic_orchestrator.py`: final audit-first research diagnostic controller.
- `memory/weaviate_schema.py`: object-property clinical memory schema contract.
- `memory/weaviate_adapter.py`: safe Weaviate adapter contract and dry-run/live boundary.
- `models/specialist_adapters.py`: Pillar-0, Gemma 4 and Claude adapter contracts.
- `data/document_processing.py`: Nemotron-Parse-aware document processor with fallback.
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
- `reasoning/rlm_engine.py`: recursive retrieval loop that dispatches named primitives instead of executing code; data class, budget and completion enforced in code; depth capped at one.
- `reasoning/rlm_wiring.py`: binds the engine to the semantic memory adapter (inheriting the quarantine) and writes trajectories to the audit store as health data.
- `reasoning/root_model_cross_check.py`: runs two root models independently over the same case and treats disagreement as an uncertainty signal, the same principle `retrieval_reconciliation` applies to the two retrieval paths; picks neither side and flags for review.
- `memory/information_content.py`: concept specificity as -log(p), used to weight path strength so a path through specific concepts outranks one of equal length through general ones (IC-based measures outperform path-based ones on MSH-WSD); plus ONTOSPREAD's converging-paths reward, sub-additive so connectivity cannot manufacture certainty.
- `connectors/europe_pmc.py`: primary literature connector -- PubMed's 37M citations plus PMC full text plus preprints (bioRxiv, medRxiv) in one search, no API key, producing checkable LiteraturePassage objects. Chosen over raw PubMed E-utilities specifically because PubMed does not index full text.
- `connectors/clinical_trials.py`: ClinicalTrials.gov as a second, complementary source -- trial registrations, not published findings; excludes terminated/withdrawn trials by default; NCT identifiers recognised as independently checkable.
- `memory/term_history.py`: append-only record of every HPO term rename and obsoletion ever detected -- a renamed term keeps every historical name as a permanent synonym, so nothing a case, a clinician, or an older record once called a concept is ever lost to an ontology update.
- `memory/differential_ranking.py`: ranks candidate diseases by IC-weighted specificity of shared findings, not raw overlap -- the "observe symptoms, discriminate against what the graph already associates" approach, which needs no separate diagnostic data source because it is the established phenotype-similarity method the HPO graph was built to support.
- `memory/structural_extraction.py`: the real (HTTP-backed) extractor for tier 3 of the normalisation cascade -- entities and relations out of free text, transport left to the deployment, degrading gracefully throughout.
- `memory/graph_sources.py`: loads the verification graph AND the hp.obo synonym index from real HPO files, always reporting which source it used -- refusing to fall back to a fixture when `require_real_data=True`, because an audit found three live bench runs scoring candidates against a 33-edge hand-written fixture while reporting it as grounding against the concept graph. loads the verification graph from a real HPO release and always reports which source it used -- refusing to fall back to a fixture when `require_real_data=True`, because an audit found three live bench runs scoring candidates against a 33-edge hand-written fixture while reporting it as grounding against the concept graph.
- `memory/concept_normalisation.py`: three-tier cascade bridging free-text clinical phrasing to graph concepts -- lexical (exact/containment/word-set, unchanged), then SapBERT-style embedding with both a threshold and a runner-up margin, then structural. Every result records which tier resolved it and whether that tier is reproducible.
- `memory/structural_comparison.py`: tier 3 -- a model extracts entities and relations from a claim and from a concept's cached literature description, and the comparison of those two structures is arithmetic. The model never judges equivalence; it only extracts. Descriptions persist as append-only JSONL, the same discipline as learned graph edges, and are created automatically whenever `DiagnosticAssembly.promote_confirmed` promotes a new edge -- from the Dream Engine or a newly confirmed RLM conjecture -- for any concept that does not already have one. tier 3 -- a model extracts entities and relations from a claim and from a concept's cached literature description, and the comparison of those two structures is arithmetic. The model never judges equivalence; it only extracts.
- `memory/gene_annotations.py`: parsers for genes_to_phenotype.txt, phenotype_to_genes.txt, genes_to_disease.txt -- a new `associated_gene`/`causes_disease` relation the graph never carried before, header-driven and alias-tolerant so a column-naming variant across HPO releases fails loudly rather than silently misattributing a field. tier 3 -- a model extracts entities and relations from a claim and from a concept's cached literature description, and the comparison of those two structures is arithmetic. The model never judges equivalence; it only extracts.
- `memory/literature_index.py`: literature as a retrieval source with citable provenance, never training data -- concept-matched (not embedding-matched, given this project's own measurement of what latent similarity costs clinically), citation-first formatting, and uncheckable sources excluded from retrieval by default.
- `memory/graph_store.py`: append-only JSONL persistence for edges the system learned, kept in a separate file from the derivable HPO import -- the learned layer is the only irreplaceable data, and an HPO refresh must not destroy it.
- `memory/candidate_retrieval.py`: finds which conditions the graph connects to a case's findings, so MechanismEnumerator can start from symptoms alone instead of a caller-supplied candidate list. Ranks by how many findings each condition touches; uses edge direction (not relation names) to avoid proposing a symptom as a diagnosis.
- `memory/spreading_activation.py`: constrained spreading activation from one or two origin concepts, reusing SCRIPT_RELATIONS as the default allowed-relation vocabulary; IC-weighted at the destination, multi-sourcing identifies mediating concepts for a relevance question.
- `training/dora_config.py`: DoRA hyperparameters as a plain dataclass, field names matching peft.LoraConfig exactly -- a decision recorded as code, with no base model chosen yet and no PEFT dependency required to use it.
- `training/dpo_config.py`: DPO hyperparameters (trl.DPOConfig-compatible) plus a readiness check on pair count, consuming preference_pairs.py's real output rather than a second copy of the extraction.
- `evaluation/enumeration_bench.py`: measures the differential itself -- recall of the confirmed condition, its rank, restraint where the graph cannot support a conclusion, and whether the open questions name real terms. Four properties kept separate because they fail independently.
- `evaluation/vetting_bench.py` (v2): 16 cases across ten organ systems (up from four), three restraint cases where the correct answer is declining a connection, repeated trials per case folded into one result, and a Wilson confidence interval on grounding_rate -- ranking sorts on restraint first, then the conservative interval bound, not the raw point estimate. measures a candidate on vetting rather than navigation -- propose a mechanism, and let `mechanism_verification` score it against the graph; the graph is the judge, never another model.
- `training/preference_pairs.py`: extracts DPO (prompt, chosen, rejected) triples from confirmed cases and the alternatives raised alongside them; builds no training code, only verifies the data would be there.
- `training/hypothesis_yield_wiring.py`: connects ConfirmationRegistry to HypothesisYieldModel and exposes it as a plain callable plus a function-calling/MCP-shaped tool spec -- data, not weights, so it works identically whichever model calls it.
- `memory/guided_graph_expansion.py`: optional, non-deterministic fallback when constrained spreading finds nothing -- a model chooses which real edge to follow, never invents one; every result is marked as coming from a guided walk.
- `memory/concept_paths.py` also now exposes `concept_names_match` (exact/containment/word-set text comparison, one shared rule) and `resolve_concept` (mentioned_concepts as a safe first tier, concept_names_match's word-set tolerance as fallback) -- one comparison philosophy for both factor/target resolution and mechanism matching, instead of two that happened to agree only on the cases tested so far.
- `reasoning/diagnostic_assembly.py`: the single assembly point -- builds the graph from its persistent layers, binds the enumerator, ledger and yield model to it, runs a case end to end, and promotes confirmed conjectures into the learned store. One file to check when asking "is it connected?".
- `reasoning/rlm_graph_bridge.py`: the link between what the RLM reads and what the graph knows -- document findings become graph entry points, ranked hypotheses return predicted findings to look for, and the RLM's own conjectures are vetted by the same `verify_mechanism` any model answer gets.
- `reasoning/mechanism_verification.py`: checks a relevance frame's claimed `mechanism` against the concept graph via `mediating_concepts` rather than against the other model's string, separating agreement from grounding so two models agreeing on an invented mechanism is visible instead of looking like clean corroboration.
- `reasoning/frame_answer.py`: compares two answers by frame slots (drug/dose/frequency, finding/site/polarity) rather than by characters, after character similarity was measured ranking a contradiction above a paraphrase; polarity slots make Fauconnier's mental spaces checkable.
- `evaluation/depth_comparison.py`: paired comparison of depth 0 against depth 1 before the recursion is trusted.

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
- **Nemotron-Parse**: document intelligence parser for clinical/literature ingestion.

None of these external systems is the final diagnostic authority.

## Validation strategy

Validation is split into:

1. retrospective benchmark evaluation;
2. prospective prediction-lock validation;
3. calibration against real-world correctness;
4. safety and abstention analysis;
5. clinical/regulatory review outside this scaffold.

See `docs/validation/clinical_benchmarking_and_prospective_validation.md`.
