# Riconciliazione delle roadmap — Report EG #10 vs ROADMAP.md, e proposta unificata

> **Nota**: questo documento ha prodotto la versione ottimizzata e unificata
> ora in vigore in `src/melampo/ROADMAP.md`. Resta qui come riconciliazione
> verificata voce per voce — la fonte delle decisioni, non il piano
> operativo corrente. Per il piano attivo, vedere `ROADMAP.md`.

**Autore:** Francesco Lattari (analisi e verifica: sessione di sviluppo)
**Data:** 17 settembre 2026
**Stato:** documento vivo, non definitivo

---

## Premessa metodologica

Il Report EG #10 è datato 2 settembre 2026. Fra quella data e oggi sono
intercorsi **96 commit** in una sessione intermedia non direttamente
osservata in questo lavoro, più i commit di questa sessione (11-17
settembre). Questo documento non confronta EG #10 con `ROADMAP.md` a
tavolino — **ogni affermazione sullo stato attuale di un'attività è stata
verificata nel repository reale**, con data del commit e contenuto del
file, non presunta da nessuno dei due documenti.

---

## Parte 1 — Confronto strutturale: cosa sono, davvero, i due documenti

Non sono due versioni della stessa cosa. Sono due strumenti diversi, e
questo spiega perché non si sovrappongono riga per riga.

| | `ROADMAP.md` | Report EG #10 |
|---|---|---|
| Granularità | 4 fasi, elenchi puntati | 7 blocchi (A-G), ~35 attività nominate |
| Stime di sforzo | Nessuna | Sì, per ogni attività |
| Dipendenze esplicite | Nessuna | Sì, con vincoli d'ordine dichiarati non negoziabili |
| Tracciamento dello stato | Nessuno | Aperto/Parziale/Chiuso per ogni voce |
| Claim bloccanti tracciati | Nessuno | Sì, con stato di misurabilità |
| Decisioni richieste all'utente | Nessuna | 6, esplicite, con cosa bloccano |
| Sequenza consigliata | Implicita nell'ordine delle fasi | Esplicita, con blocchi paralleli indicati |

**`ROADMAP.md` è uno strumento strategico** — cosa fare, in che ordine
generale, senza impegnarsi sui dettagli esecutivi. **EG #10 è uno strumento
operativo** — cosa fare questa settimana, cosa la blocca, quanto costa.
Non sono in conflitto: EG #10 è, di fatto, l'unica granularità a cui si
può davvero lavorare; `ROADMAP.md` è la vista che sopravvive quando i
dettagli esecutivi cambiano sotto di essa.

**Un'assenza degna di nota**: la Fase 4 di `ROADMAP.md` ("theoretical-
quantum track") non ha alcun corrispondente in EG #10 — il blocco più
vicino per argomento, il modello di cognizione quantistica per gli effetti
d'ordine (Tappa A completata in questa sessione), è nato da una
conversazione di questa settimana, non dalla pianificazione del 2
settembre. `ROADMAP.md`, su questo punto, è più aggiornato di EG #10 — non
il contrario.

---

## Parte 2 — Cosa in EG #10 è ormai superato, verificato con data e commit

### Chiuso nella sessione intermedia (3-6 settembre), prima di questa settimana

| Voce EG #10 | Evidenza verificata |
|---|---|
| **A0** — risoluzione dei concetti | Già dichiarato chiuso in EG #10 stesso; confermato: `concept_resolution.py`, creato 2 settembre |
| **B0c/B0d** (parte) — ConText, asserzione | `memory/assertion.py`, creato 2 settembre, esteso 6 settembre: *"assertion detection and an enforced findings boundary"*, poi *"absence paraphrases, layered resolution and residue measurement"* |
| **B2** — isolamento fisico dei candidati in Weaviate | `memory/weaviate_adapter.py` + `weaviate_schema.py`, commit del 3 settembre: *"physical quarantine for candidates, gate wired to real metrics"* — corrisponde quasi testualmente alla descrizione EG #10 |
| **B3** — gate di indeterminazione sulle metriche reali | **Stesso commit del 3 settembre** di B2 — *"gate wired to real metrics"* copre entrambe le voci in un solo intervento |
| **C1** — `rlm_engine.py`: loop root, budget, sandbox, FINAL() | Commit del 6 settembre, **etichettato esplicitamente "Block C" nel proprio messaggio**: *"feat: recursive engine without code execution (Block C)"* |
| **C2** (parziale) — ambiente collegato a Weaviate | `memory/context_environment.py`, 1 settembre: *"RLM-on-Memory retrieval foundation"* |

### Avanzato in questa sessione (11-17 settembre)

| Voce EG #10 | Stato reale oggi |
|---|---|
| **A3** — grafo popolato | Ben oltre "meccanica completa": grafo reale caricato, **1.273.466 archi, 29.053 concetti**, importatori geni-fenotipo e geni-malattia, MAxO, cronologia dei rinomini — nessuno di questo esisteva il 2 settembre |
| **A3.3** — verifica licenza SNOMED | Non chiusa, ma **tentata**: ricerca diretta sullo stato di appartenenza dell'Italia a SNOMED International, esito inconcludente (assenza consistente su più fonti, non conferma certa) — documentato nel trattato tecnico, Parte 1.3 |
| **G3** (parziale) — riconciliazione concetti | Non tramite MONDO come previsto, ma tramite **UMLS crosswalk** (`connectors/umls.py`) — stesso obiettivo (collegare il grafo ad altri vocabolari), meccanismo diverso da quello originariamente pianificato |

---

## Parte 3 — Cosa resta genuinamente aperto, verificato per assenza

Nessuna delle voci seguenti ha evidenza di completamento nel repository —
verificato cercando i file, le funzioni, o i commit attesi, non solo
assumendo dal silenzio di EG #10.

| Voce | Verifica effettuata | Esito |
|---|---|---|
| **A1** — equivalente ricorsivo di `mean_grounding_score` | Cercata la funzione e ogni file di grounding/retrieval adiacente | Nessuna implementazione trovata; ancora "progettato, non implementato" |
| **A2** — corpus di validazione longitudinale | Cercato qualunque file con "longitudinal" nel nome | Zero risultati |
| **A3.1** — set di riferimento per la copertura | Cercato in `graph_coverage.py` e altrove | Zero risultati — `measure_coverage` non ha ancora un denominatore |
| **A3.2** — importatori LOINC, ATC, ECTO | Cercati per nome | Zero risultati |
| **A4** — rotazione del token GitHub | **Verificato direttamente: il file delle credenziali contiene ancora un solo token, lo stesso** | **Non fatto — e il token è stato usato per ogni commit di questa intera sessione, decine di volte, da quando il rischio era già stato segnalato il 2 settembre** |
| **B0b** — estrazione guidata dall'indice | Bloccata dalla decisione #1/#2 mai presa (lingua di riferimento, ampiezza lessico) | Ancora aperta |
| **B1** — `MechanismEnumerator` dentro `_alternative_hypotheses()` | **Verificato in questa sessione stessa** (Parte VI del trattato tecnico): il campo `enumerator: Any = None` esiste ma nessun punto reale di costruzione lo popola | Confermato ancora aperto |
| **B4** — valutazione clinica su casi densi | Dipende da A3.1, mai soddisfatta | Ancora aperta |
| **D1** — `ModelRouter` come gate di complessità | **Letto il file per intero**: 20 righe, `routing_mode` fisso a `"static_research_router"`, nessuna logica di complessità reale | Ancora uno stub, non un gate |
| **D2, D3, D4, D5** | Dipendono da C (parzialmente pronto) e D1 (non pronto) | Ancora aperte |
| **E2** — sostituire "Gemma 4" con artefatto verificabile | Cercato letteralmente "Gemma 4" in tutto il codice | **Ancora presente ovunque**: `model_capability_registry.py`, `specialist_adapters.py`, `rlm_model_adapter.py`, `specialist_runtime.py` — non sostituito |
| **E1, E3, E4** | Nessuna evidenza cercata di completamento | Presumibilmente ancora aperte |
| **E5** — Contextual Boundary Protection nei chunk Docling | **Verosimilmente superata dagli eventi**: Docling è stato rimosso da questo progetto (PR storica #53, prima di questa sessione di analisi) in favore di Nemotron-Parse/LlamaParse. Questa voce va probabilmente riscritta per il nuovo parser, non semplicemente completata come scritta |
| **F1-F5** — oracolo a predicati, privacy | Cercata qualunque traccia (`predicate_budget`, `cumulative_disclosure`) | Nessuna implementazione dedicata trovata |
| **G1, G2, G4** — vocabolario di staging | Cercati per nome | Zero risultati |

---

## Parte 4 — La scoperta più urgente di questa riconciliazione

**A4 non è una voce come le altre.** Era già segnalata come bloccante e a
costo bassissimo (30 minuti) il 2 settembre. Oggi, 15 giorni e centinaia di
commit dopo — inclusi tutti quelli di questa sessione, ogni singolo commit
firmato `frankltr`, ogni pull request aperta e mergiata — **lo stesso
identico token risulta ancora in uso**. Non è un dettaglio amministrativo
rimasto indietro: è il rischio di sicurezza più a buon mercato da
risolvere nell'intero documento, ed è rimasto aperto più a lungo di
qualunque altra voce del blocco A.

---

## Parte 5 — Proposta di roadmap unificata

Struttura di EG #10 (più dettagliata, azionabile), aggiornata allo stato
verificato di oggi. Le voci chiuse sono rimosse dall'elenco attivo; quelle
avanzate sono annotate; quelle nuove (emergenti da questa sessione) sono
aggiunte con la stessa disciplina.

### Blocco A — completamento fondamenta (quasi chiuso)

| # | Attività | Stato oggi | Sforzo residuo |
|---|---|---|---|
| A4 | Rotazione token, secret scanning, push protection | **Aperto, urgente** — invariato dal 2 settembre | 30 min |
| A1 | Equivalente ricorsivo del grounding | Aperto | 1-2 sett. |
| A2 | Corpus longitudinale con aghi congiuntivi | Aperto | 2 sett. |
| A3.1 | Set di riferimento per la copertura | Aperto — **blocca B4** | 1 sett., curatela clinica |
| A3.2 | Importatori LOINC, ATC, ECTO | Aperto | 1-2 sett. ciascuno |
| A3.3 | Verifica licenza SNOMED | Tentata, inconcludente — richiede conferma diretta con SNOMED International | 1 gg |
| ~~A0~~ | ~~Risoluzione concetti~~ | **Chiuso** | — |
| ~~A3 (meccanica)~~ | ~~Grafo popolato~~ | **Sostanzialmente superato**: 1,27M archi reali, geni, MAxO, cronologia | — |

### Blocco B — pilota Dream Engine (parzialmente chiuso)

| # | Attività | Stato oggi | Sforzo residuo |
|---|---|---|---|
| B0b | Estrazione guidata dall'indice | Aperto — bloccato da decisioni #1/#2 mai prese | 2-3 sett. |
| B1 | `MechanismEnumerator` dentro `_alternative_hypotheses()` | **Aperto, confermato** in questa sessione | 2-3 gg |
| B4 | Valutazione clinica su casi densi | Aperto — blocca su A3.1 | 2 sett. |
| ~~B0c/B0d~~ | ~~ConText, asserzione~~ | **Chiuso** (2-6 settembre) | — |
| ~~B2~~ | ~~Isolamento Weaviate~~ | **Chiuso** (3 settembre) | — |
| ~~B3~~ | ~~Gate di indeterminazione~~ | **Chiuso** (3 settembre) | — |

### Blocco C — motore RLM (in gran parte chiuso)

| # | Attività | Stato oggi |
|---|---|---|
| C3 | Traiettorie in `decision_trace.py` e audit store | Da verificare — non controllato in questa riconciliazione |
| C4 | Baseline depth 0 accanto a depth 1 | Da verificare |
| ~~C1~~ | ~~`rlm_engine.py`~~ | **Chiuso** (6 settembre) |
| ~~C2~~ | ~~Ambiente Weaviate reale~~ | **Sostanzialmente pronto** (1 settembre) |

### Blocco D — dual-path diagnostico (ancora aperto)

Nessuna voce chiusa. D1 resta uno stub da 20 righe senza logica reale;
D2-D5 dipendono da D1 e non sono ancora sbloccabili.

### Blocco E — consolidamento

| # | Attività | Nota |
|---|---|---|
| E2 | Sostituire "Gemma 4" con artefatto verificabile | Ancora aperto, presente ovunque nel codice |
| E5 | Contextual Boundary Protection nei chunk Docling | **Da riscrivere**: Docling è stato rimosso dal progetto; questa voce va riformulata per Nemotron-Parse/LlamaParse |
| E1, E3, E4 | Non verificate in questa riconciliazione | — |

### Blocco F — privacy e oracolo a predicati

Nessuna evidenza di lavoro iniziato. Tutte le voci (F1-F5) restano aperte
come nel report originale.

### Blocco G — vocabolario e curatela

| # | Attività | Nota |
|---|---|---|
| G3 | Riconciliazione concetti via MONDO/UMLS | **Parzialmente indirizzato** tramite `connectors/umls.py` (crosswalk), meccanismo diverso da MONDO ma stesso obiettivo |
| G1, G2, G4 | Nessuna evidenza | Ancora aperte |

### Blocco H — nuovo, emerso da questa sessione (non presente in EG #10 né in ROADMAP.md)

| # | Attività | Sforzo |
|---|---|---|
| H1 | Riconciliare `diagnostic_assembly.py` con `clinical_pipeline.py` (due catene di ragionamento non ancora unificate — Parte VI del trattato tecnico) | Da stimare, architetturale |
| H2 | Rimuovere o riformulare E5 per il nuovo parser documentale | 1-2 gg |
| H3 | Indagine sulle prestazioni: 3,4 secondi per interrogazione singola sul grafo reale, misurato in questa sessione, mai investigato | 3-5 gg |
| H4 | Popolamento del deposito UMLS cifrato una volta ottenuta la registrazione | Dipende da registrazione UMLS (in corso) |

---

## Parte 6 — Le sei decisioni di EG #10, verificate: quali sono ancora bloccanti

| # | Decisione | Ancora blocca | Stato |
|---|---|---|---|
| 1 | Lingua di riferimento per il collegamento concettuale | B0b | **Ancora non presa** |
| 2 | Ampiezza del lessico nella prima versione | B0b | **Ancora non presa** |
| 3 | Categorie oltre il fenotipo, in ordine di priorità | A3.2 | **Ancora non presa** |
| 4 | Chi cura il set di riferimento per la copertura | A3.1, quindi B4 | **Ancora non presa — la più costosa delle sei rispetto al beneficio** |
| 5 | Classificatore di asserzione locale o addestrato di dominio | B0d | Probabilmente risolta implicitamente da `assertion.py` (3-6 settembre) — da confermare |
| 6 | Rotazione del token | A4 | **Ancora non presa — la più economica, ancora aperta dopo 15 giorni** |

---

## Nota di chiusura

Questo documento riflette lo stato del repository al 17 settembre 2026.
Non sostituisce né EG #10 né `ROADMAP.md` — li riconcilia, con ogni
affermazione di stato verificata contro il codice reale al momento della
scrittura. Non è definitivo: C3, C4, E1, E3, E4, G1, G2, G4 non sono stati
verificati con lo stesso livello di dettaglio delle altre voci per limiti
di tempo di questa analisi, e andrebbero controllati con la stessa
disciplina prima di considerarli affidabili.
