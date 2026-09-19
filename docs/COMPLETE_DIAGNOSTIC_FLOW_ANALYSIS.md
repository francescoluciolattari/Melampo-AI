# Il percorso diagnostico completo — verificato, non presunto

**Scopo di questo documento.** Ogni affermazione qui è stata verificata
leggendo il codice o il documento citato per intero, non ricostruita a
memoria. Tre versioni dello stesso percorso — quello che esiste oggi,
quello previsto a maggio 2026, quello previsto dal documento di decisione
di settembre — messe fianco a fianco per trovare cosa è ridondante,
superato, o mai stato collegato.

---

## Parte 1 — Il percorso di oggi, verificato riga per riga

```
payload in ingresso
   │
   ▼
ClinicalInferencePipeline.run(payload)
   │
   ├─ ingestion.from_payload(payload)              → CaseContext
   ├─ normalizer.to_fhir_bundle(case)               → bundle FHIR
   ├─ codifica multimodale (testo, volume, istologia) → fused
   │
   ├─ _retrieve_and_rank()
   │     └─ MemoryRetriever.retrieve()  [recupero a UN passaggio]
   │           └─ SemanticMemoryStore — VUOTO, verificato
   │           └─ ricade sempre su _fallback_evidence()
   │                 3 prove INVENTATE, grounding_score fisso
   │     └─ EvidenceRanker → ranked_evidence
   │
   ├─ segnali delle 4 aree (visiva, linguistica, contesto, epidemiologia)
   │     └─ ognuna conta chiavi di dizionario, verificato, non contenuto
   │
   ├─ area_coherence.analyze() → area_dynamics
   │     └─ include neuro_dynamic_metrics (18 parametri, pesi a mano,
   │        "computational_abstraction_not_literal_neurobiology" —
   │        dichiarazione del codice stesso)
   │
   ├─ governance_scores = _derive_governance_scores(...)
   │
   ├─ nexus = _run_nexus_branch()  →  NexusTrainer.run()
   │     ├─ ReplayFilter.assess()
   │     ├─ CounterfactualSampler.sample()
   │     ├─ VisualImprintMorpher.nexus_morph()
   │     ├─ profilo di reiterazione (caso raro/limite/contraddizione)
   │     ├─ _alternative_hypotheses()
   │     │     SE payload["findings"] presente:
   │     │        MechanismEnumerator reale sul grafo (B1) → ipotesi vere
   │     │     ALTRIMENTI:
   │     │        etichette di ripiego (rare_case, adjacent_case, ...)
   │     ├─ _auto_evolution_plan()  ── calcola un verdetto MAI USATO
   │     │        da nessun altro modulo (verificato)
   │     └─ belief_layer.update()  [QuantumBeliefLayer, chiamata 1/2]
   │
   ├─ nexus_scheduler.enqueue(case_context, area_dynamics, nexus, ...)
   │     └─ mette in coda — nessun innesco automatico di bassa attività
   │        esiste ancora (collegato questo turno, mai svuotato da solo)
   │
   ├─ intuition_engine.infer(ranked_evidence, nexus, graph_candidates)
   │     ├─ rank_differential()  SE findings presenti → nomi VERI
   │     │        (collegato con B1-stile questa sessione)
   │     ├─ 3 modalità (rapida/razionale/contraddizione), somma di
   │     │        18 metriche con pesi fissi, NESSUN limite superiore
   │     │        (verificato: 7,341 osservato in un caso reale)
   │     └─ belief_layer.update()  [QuantumBeliefLayer, chiamata 2/2]
   │
   ├─ coordinator.run() → DifferentialEngine.rank(evidence, intuition, nexus)
   │     ├─ primaria = intuition, SE segnaposto → promossa da nexus reale
   │     └─ alternative = fuse da nexus.alternative_hypotheses
   │
   └─ _finalize_diagnostic_result() → DiagnosticResult (.nexus, .differential, ...)
         │
         ▼
   output in uscita
```

**Mai toccato da questo percorso, verificato con ricerche dirette nel
codice**:
- `rlm_engine.py` (navigazione ricorsiva dei documenti)
- `diagnostic_assembly.py` / `rlm_graph_bridge.py` (richiede una
  traiettoria RLM come ingresso, mai fornita)
- `retrieval_reconciliation.py`, `root_model_cross_check.py`
- `literature_persistence.py` e i connettori di letteratura
- `document_processing.py` (l'integrazione Nemotron-Parse/LlamaParse
  stessa, mai richiamata; la chiamata HTTP reale non è nemmeno
  implementata — `NotImplementedError`)
- `self.router` (`ModelRouter` — D1, non ancora costruito)
- `NexusScheduler.run_once()` (mai innescato automaticamente)

---

## Parte 2 — Cosa prevedeva la documentazione di maggio

Fonti: `architecture_consistency_matrix.md`, `final_treatise_decision_record.md`,
`neuro_vector_evolution_strategy.md`, `dream_self_evolution_governance.md`.

```
NexusTrainer (Fase 1, per caso, sincrona)
   │
   ▼ il proprio output "nexus"
NexusScheduler (Fase 2, offline, bassa attività — dream_scheduler.py)
   │
   ├─ NexusSelfEvolutionLoop.rehearse()
   │     └─ genera un candidato, valuta soglie proprie
   │        (pi_score, prediction_error, bias_suppression —
   │        DIVERSE da quelle di _auto_evolution_plan)
   │     └─ SCRIVE DAVVERO in InMemoryVectorStore
   │           stato: "promoted" o "candidate"
   │
   ├─ RationalControlValidator.evaluate()
   │     └─ verifica soglie aggiuntive prima della promozione finale
   │
   └─ PromotionPolicy.decide()
         └─ 5 stati: candidate → needs_review → promoted/rejected/retired
         └─ ambito dichiarato: "synthetic_curriculum_or_vector_memory_only"
            — MAI il grafo diagnostico vero

Deposito vettoriale previsto: Weaviate (poi Qdrant in un documento
successivo, mai riconciliati fra loro — superato ora dalla scelta di
FalkorDB, Parte 4)
Parser documentale previsto: Docling (rimosso dal progetto prima di
questa sessione, sostituito da Nemotron-Parse/LlamaParse)
Modello di ragionamento previsto: "Gemma 4" (mai confermato corrispondere
a un modello realmente rilasciato)
```

**Cosa di questo è già collegato, oggi**: la connessione
`NexusTrainer → NexusScheduler` (fatta questo turno). Il resto della
catena (`NexusSelfEvolutionLoop`, `RationalControlValidator`,
`PromotionPolicy`) esiste, è testato, ed è raggiungibile — ma nessun
innesco automatico di bassa attività esiste ancora per farla girare da
sola.

---

## Parte 3 — Cosa prevede il documento di decisione di settembre

Fonte: `rlm_on_memory_decision_record.md`, letto per intero.

```
D1 (ModelRouter, da costruire) — decide QUANTI percorsi far girare,
non quale:

   Ricerca fattuale, basso rischio       → solo un passaggio
   Complesso o alto rischio              → doppio percorso + riconciliazione
   Disaccordo fra aree non risolto       → doppio percorso, budget esteso
   Ramo nexus (bassa attività)           → solo ricorsivo

Un passaggio (recupero associativo, veloce)  →  IntuitionEngine
Ricorsivo (RlmEngine, navigazione documenti) →  DifferentialEngine
                                                  [MAI collegato oggi:
                                                   servirebbe passare
                                                   per rlm_graph_bridge.py,
                                                   mai raggiunto da
                                                   clinical_pipeline.py]

Se le due vie disaccordano:
   retrieval_reconciliation.py
      → dossier di prove fuse + conflict_signal (un NUMERO)
      → NON propone un percorso diagnostico ulteriore

Giustificazione, con numeri citati dal documento stesso:
   ricorsivo degrada le ricerche semplici del 15-30%
   ricorsivo soggetto a "overreach" (narrazioni senza fonte)
   un passaggio fallisce per omissione, non per invenzione
```

**Cosa di questo è già collegato, oggi**: nulla di questa catena
specifica. `IntuitionEngine` è collegato, ma non tramite "un passaggio
puro" nel senso del documento — riceve sia `ranked_evidence` (spesso
finta) sia, da questa settimana, `rank_differential()` reale.
`DifferentialEngine` non riceve mai l'output di `RlmEngine`.

---

## Parte 4 — Le ridondanze trovate, con una raccomandazione per ciascuna

| Ridondanza | Cosa sono | Raccomandazione |
|---|---|---|
| `_auto_evolution_plan()` contro `NexusSelfEvolutionLoop` | Due valutazioni indipendenti, soglie diverse, solo una scrive davvero in memoria | **Eliminare `_auto_evolution_plan()`** — non ha effetto, solo un dizionario che alimenta un'altra formula non calibrata |
| `QuantumBeliefLayer` chiamato due volte (`NexusTrainer` e `IntuitionEngine`) | Stessa formula, stesso tipo di input (metriche neuro-dinamiche), due istanze separate | Da decidere: una chiamata unica condivisa, o due deliberatamente separate? Non ancora chiarito |
| Weaviate (maggio) contro Qdrant (aprile) contro FalkorDB (questa sessione) | Tre decisioni sullo stesso problema, mai riconciliate nei documenti originali | **Risolta**: FalkorDB scelto e implementato. I riferimenti a Weaviate in `architecture_consistency_matrix.md` sono già marcati come superati |
| `retrieval_reconciliation.py` contro `root_model_cross_check.py` | Non ridondanti — riconciliano assi diversi (strategie contro modelli) — ma **entrambi** mai collegati | Da decidere insieme: quale, se non entrambi, va collegato a D1/D2 |
| `document_processing.py` | Contiene già l'integrazione Nemotron-Parse/LlamaParse, ma non è mai chiamato, e la chiamata HTTP reale non è implementata | Va completata (l'endpoint HTTP) e collegata, non è codice morto da eliminare — è infrastruttura reale, incompleta |
| "Gemma 4" | Nome mai confermato corrispondere a un modello reale, presente in quattro file | Va verificato o sostituito con un nome di modello reale e verificabile |

---

## Parte 5 — Una proposta di percorso unico, per procedere

Non un rifacimento — un collegamento di pezzi che esistono già, con le
ridondanze tolte:

```
1. D1 (ModelRouter) decide quanti percorsi, secondo la tabella già
   scritta a settembre

2. Percorso "un passaggio":
   IntuitionEngine, alimentato da rank_differential() (già reale)
   — non più da ranked_evidence finto, se possibile eliminare quella
   dipendenza o popolare davvero SemanticMemoryStore

3. Percorso "ricorsivo", SOLO quando D1 lo richiede:
   RlmEngine → rlm_graph_bridge.py → DifferentialEngine
   (collegamento che oggi non esiste, da costruire)

4. Se entrambi i percorsi girano:
   retrieval_reconciliation.py fonde le prove, produce conflict_signal

5. NexusTrainer resta il generatore di ipotesi dal grafo per il ramo
   nexus, con _auto_evolution_plan() rimosso

6. NexusScheduler processa la coda quando un vero innesco di bassa
   attività esiste (da costruire) — mai dentro una richiesta
```

Questo documento non decide da solo cosa fare — elenca cosa esiste,
cosa manca, e dove sono le scelte ancora aperte, perché si possa
decidere insieme senza ambiguità residua.
