# Project Melampo

**From Attention to Intuition**  
**Enterprise-grade research scaffold for multimodal, multimodel clinical intuition**

![Status](https://img.shields.io/badge/Status-Research_Scaffold-blue)
![Focus](https://img.shields.io/badge/Focus-Medical_AI_%7C_RAG_%7C_Intuition-purple)

Project Melampo is an experimental open-source research framework designed to evolve Computer-Aided Diagnosis (CAD) into **Computer-Aided Intuition (CAI)**.

Melampo is **not** a single-model diagnostic assistant and **not** a simple RAG wrapper. It is a multimodal, multimodel, neuro-symbolic and neuro-inspired architecture where specialist models produce governed signals, simulated functional areas compare and integrate them, and a Melampo-owned orchestrator produces a structured research diagnostic result with abstention, escalation, provenance and audit trace.

> **Clinical safety boundary:** Melampo is a research scaffold. It is not a validated medical device and must not be used for autonomous clinical diagnosis or patient-care decisions without formal clinical validation, regulatory review, monitoring and human specialist oversight.

## Current architectural decisions

The project now records these decisions in `docs/final_treatise_decision_record.md`:

- **Final diagnostic authority:** `MelampoDiagnosticOrchestrator`, not a single external LLM or imaging model.
- **Primary radiology / volumetric imaging candidate:** Pillar-0, used as a signal provider for `visual_diagnostic_area`.
- **Clinical text and agentic reasoning candidate:** Gemma 4, used for grounded language and workflow reasoning.
- **External critique candidate:** Claude Healthcare / Life Sciences style capability, used as optional second-opinion and scientific/regulatory critic.
- **Semantic memory / ontology RAG backend:** Weaviate, selected for object-property clinical memory and concept relationships.
- **Document intelligence parser:** Docling, used for layout-aware ingestion of PDFs, DOCX, PPTX, tables, formulas and clinical literature.

External models are specialist signal providers, critics, parsers, retrievers or explainers. They are not sole final diagnostic arbiters.

## Core architecture

```text
specialist models
  -> Melampo functional areas
  -> coherence / mismatch / neuro-dynamic metrics
  -> intuition engine
  -> differential reasoning
  -> policy, abstention and critique
  -> diagnostic orchestrator
  -> governed memory and validation loop
```

### Functional areas

- `visual_diagnostic_area`: imaging and radiology signals, future Pillar-0 / volumetric backend integration.
- `language_listening_area`: reports, EHR text, symptoms and narrative reasoning, future Gemma 4 integration.
- `case_context_area`: demographics, provenance and structured context.
- `epidemiology_area`: exposures, prevalence and risk-factor context.
- semantic memory: Weaviate object-property RAG for symptoms, pathologies, cases, imaging studies, documents and epidemiological factors.
- dream / replay branch: counterfactual, rare-case and mismatch-resolution candidates under strict governance.

### Melampo intuition

In Melampo, intuition is not an LLM guess. It is an emergent diagnostic candidate produced by multimodal convergence, semantic memory, coherence between simulated areas, controlled mismatch resolution, inhibition of noisy or bias-prone signals, dream replay and contextual belief update.

## Implemented scaffold

The repository currently includes:

- Python package scaffold under `src/melampo/`.
- End-to-end `ClinicalInferencePipeline`.
- `MelampoDiagnosticOrchestrator` final controller.
- `ModelCapabilityRegistry` for Pillar-0, Gemma 4, Claude, Weaviate and Docling roles.
- Weaviate schema and adapter contracts, including visual recognition imprint objects linked to semantic concepts.
- Specialist adapter contracts for Pillar-0, Gemma 4 and Claude.
- Docling-aware document processing contract with plain-text fallback.
- Neuro-dynamic metrics, `pi_score`, convergence, mismatch, prediction error and belief-update signals.
- Dream branch auto-evolution guardrails and candidate-only inferential visual imprint morphing over total or partial semantic concept overlap.
- Retrospective clinical benchmark primitives.
- Prospective validation registry primitives.
- Confidence calibration evaluator.
- CLI commands and CI workflow.

## Installation profiles

Melampo keeps the core scaffold lightweight and moves heavy research/enterprise dependencies into optional extras. Three requirements files are provided:

```bash
# Baseline development and CI profile
python -m pip install -r requirements.txt

# Research profile: dev + clinical metadata + Docling + Weaviate retrieval + visualization
python -m pip install -r requirements-research.txt

# Full enterprise profile: all optional research, imaging, ML, API, quantum and visualization extras
python -m pip install -r requirements-enterprise.txt
```

Equivalent direct editable installs:

```bash
pip install -e .[dev]
pip install -e .[dev,clinical,document,retrieval,viz]
pip install -e .[enterprise]
```

The enterprise profile may require platform-specific CPU/GPU choices for PyTorch, imaging and scientific-computing wheels.

## CLI commands

```bash
pip install -r requirements.txt
pytest -q

melampo-prototype examples/prototype_case.json
melampo-decision-record
melampo-weaviate-schema
```

## Documentation map

- `docs/final_treatise_decision_record.md` - canonical decisions for the final treatise.
- `docs/enterprise_ai_rag_evolution.md` - enterprise AI/RAG evolution plan.
- `docs/architecture.md` - current architecture overview.
- `docs/architecture_consistency_matrix.md` - theory/document/code alignment and enterprise constraints.
- `docs/core_consolidation_map.md` - canonical core module map.
- `docs/validation/clinical_benchmarking_and_prospective_validation.md` - benchmark, prospective validation and calibration plan.
- `docs/model_cards/melampo_model_stack_card.md` - model stack card.
- `docs/dataset_cards/clinical_memory_dataset_card.md` - clinical memory dataset card.

## Roadmap

- [x] Research architecture and theoretical direction.
- [x] Python research scaffold.
- [x] Multimodal pipeline spine.
- [x] Melampo-owned diagnostic orchestrator.
- [x] Weaviate semantic schema contract.
- [x] Model capability registry.
- [x] Benchmark, prospective validation and calibration primitives.
- [ ] Real Weaviate deployment adapter subclass.
- [ ] Real Pillar-0 inference adapter.
- [ ] Real Gemma 4 local/private backend adapter.
- [ ] Optional Claude critique adapter.
- [ ] Real Docling ingestion pipeline with governed source catalogs.
- [ ] Dataset-specific benchmark suites.
- [ ] Prospective clinical validation under ethics/regulatory governance.

## Contributing

We welcome contributors in medical imaging, clinical NLP, RAG, ontologies, validation, computational neuroscience, safety engineering and research infrastructure. See `CONTRIBUTING.md`.

## ⚖️ License and Terms of Use

This software is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** License. 

To view a copy of this license, visit: https://creativecommons.org

---

### SPECIFIC ATTRIBUTION REQUIREMENTS (Pursuant to Section 3(a) of the CC BY-NC 4.0 License)

As the Licensor of this software, the Author exercises the right to specify the reasonable manner of attribution. Any user, redistributor, or developer of derivative works **MUST strictly comply** with the following enforcement conditions:

1. **Graphical User Interface (GUI) or Prompt
   Attribution:** If the software, or any work derived from it, features a Graphical User Interface (GUI), a clear, permanent, and easily accessible credit to the original Author and source repository **MUST be displayed within the user interface itself** (e.g., in the "About" screen, "Credits" section, or prominently in the application footer).
3. **Required Format:** The attribution within the GUI must remain visible to the end-user and must include the following text (or a substantially identical equivalent):  
   *"Based on MelampoAI by Francesco Lattari — Source: https://github.com/francescoluciolattari/Melampo-AI"*
4. **Source Code Preservation:** All existing copyright notices, author credits, and license files within the source code must be preserved intact and unaltered in all copies or modifications.

**Failure to comply with these specific attribution terms constitutes a material breach of the license agreement and an immediate infringement of copyright, resulting in the automatic termination of your rights to use, modify, or distribute this software.**

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

See `LICENSE`.

## Contact

**Francesco Lattari** - Project Lead & Author  
Email: [flattari@chandra.it](mailto:flattari@chandra.it)  
Website: [www.chandra.it](https://www.chandra.it)

---

© 2026 Project Melampo Research Team
