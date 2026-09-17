# Registro di verifica — come è nata la roadmap unificata

**Non è una roadmap.** `src/melampo/ROADMAP.md` è l'unico piano attivo —
questo documento è la prova verificata che lo ha prodotto: cosa è stato
controllato nel repository reale, con data del commit e contenuto del
file, per distinguere ciò che il Report EG #10 (2 settembre 2026) dava
per aperto da ciò che nel frattempo si era già chiuso. Non contiene un
piano proprio, non va letto come alternativa a `ROADMAP.md`, e non
menziona la rotazione del token — gestita separatamente, fuori da questo
lavoro.

**Autore:** Francesco Lattari (verifica: sessione di sviluppo)
**Data:** 17 settembre 2026

---

## Perché serviva una verifica, non solo un confronto

Il Report EG #10 è datato 2 settembre 2026. Fra quella data e l'inizio di
questa sessione sono intercorsi **96 commit**, in una sessione intermedia
non osservata direttamente in questo lavoro. Confrontare EG #10 con
`ROADMAP.md` a tavolino avrebbe significato confrontare due fotografie,
nessuna delle due aggiornata a oggi. Ogni riga delle due tabelle sotto è
il risultato di una ricerca nel codice — file, data del commit, contenuto
— non un'assunzione dal silenzio di uno dei due documenti.

---

## Cosa era già chiuso, con la prova

| Voce EG #10 | Evidenza verificata |
|---|---|
| **A0** — risoluzione dei concetti | Già dichiarato chiuso in EG #10 stesso; confermato: `concept_resolution.py`, creato 2 settembre |
| **B0c/B0d** (parte) — ConText, asserzione | `memory/assertion.py`, creato 2 settembre, esteso 6 settembre: *"assertion detection and an enforced findings boundary"*, poi *"absence paraphrases, layered resolution and residue measurement"* |
| **B2** — isolamento fisico dei candidati in Weaviate | `memory/weaviate_adapter.py` + `weaviate_schema.py`, commit del 3 settembre: *"physical quarantine for candidates, gate wired to real metrics"* |
| **B3** — gate di indeterminazione sulle metriche reali | Stesso commit del 3 settembre di B2 — copre entrambe le voci in un solo intervento |
| **C1** — `rlm_engine.py`: loop root, budget, sandbox, FINAL() | Commit del 6 settembre, etichettato esplicitamente *"Block C"* nel proprio messaggio |
| **C2** (parziale) — ambiente collegato a Weaviate | `memory/context_environment.py`, 1 settembre: *"RLM-on-Memory retrieval foundation"* |
| **A3** (meccanica) — grafo popolato | Sostanzialmente superato in questa sessione: grafo reale, 1.273.466 archi, 29.053 concetti — nessuno di questo esisteva il 2 settembre |
| **G3** (parziale) — riconciliazione concetti | Indirizzato tramite UMLS crosswalk (`connectors/umls.py`), non MONDO come previsto — stesso obiettivo, meccanismo diverso |

---

## Cosa era ancora aperto, verificato per assenza

Non presunto dal silenzio di EG #10 — cercato direttamente: il file, la
funzione, o il commit atteso.

| Voce | Verifica effettuata | Esito |
|---|---|---|
| **A1** — equivalente ricorsivo di `mean_grounding_score` | Cercata la funzione e ogni file di grounding/retrieval adiacente | Nessuna implementazione trovata |
| **A2** — corpus di validazione longitudinale | Cercato qualunque file con "longitudinal" nel nome | Zero risultati |
| **A3.1** — set di riferimento per la copertura | Cercato in `graph_coverage.py` e altrove | Zero risultati |
| **A3.2** — importatori LOINC, ATC, ECTO | Cercati per nome | Zero risultati |
| **B0b** — estrazione guidata dall'indice | Bloccata dalla decisione (lingua di riferimento, ampiezza lessico) mai presa | Ancora aperta |
| **B1** — `MechanismEnumerator` dentro `_alternative_hypotheses()` | Il campo `enumerator: Any = None` esiste ma nessun punto reale di costruzione lo popola | Confermato ancora aperto |
| **D1** — `ModelRouter` come gate di complessità | Letto il file per intero: 20 righe, instradamento fisso, nessuna logica di complessità reale | Ancora uno stub |
| **E2** — sostituire "Gemma 4" con artefatto verificabile | Cercato letteralmente "Gemma 4" in tutto il codice | Ancora presente in quattro file, non sostituito |
| **E5** — protezione dei confini nei chunk Docling | Docling è stato rimosso dal progetto prima di questa sessione | Voce superata dagli eventi, non semplicemente da completare |
| **F1-F5** — oracolo a predicati, privacy | Cercata qualunque traccia | Nessuna implementazione dedicata trovata |
| **G1, G2, G4** — vocabolario di staging | Cercati per nome | Zero risultati |

**Non verificate in questo passaggio** (limiti di tempo dell'analisi, non
assunte chiuse né aperte): C3, C4, E1, E3, E4. Andrebbero controllate con
la stessa disciplina prima di essere pianificate con fiducia.

---

## Come questa verifica ha strutturato `ROADMAP.md`

Le voci sopra — chiuse, aperte, non verificate — sono la base diretta di
`ROADMAP.md` di oggi: le chiuse non vi compaiono più come attive, le
aperte vi sono riportate con lo sforzo residuo aggiornato, le non
verificate vi sono segnalate come tali. Un'osservazione strutturale,
utile a spiegare perché la nuova roadmap ha la forma che ha: EG #10 (7
blocchi, ~35 attività nominate, sforzo e dipendenze per ognuna) era già
uno strumento più operativo del vecchio `ROADMAP.md` a 4 fasi senza stime
né dipendenze — la nuova versione ne eredita la struttura perché è quella
su cui si può davvero lavorare, non quella con la vista più larga.
