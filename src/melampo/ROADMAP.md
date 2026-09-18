# Technical Roadmap — versione ottimizzata e unificata

**Sostituisce** la precedente versione a 4 fasi di questo file e il Report
EG #10 come unica fonte operativa — questa è **l'unica roadmap**.
`docs/ROADMAP_VERIFICATION_LOG.md` non è una seconda roadmap: è il
registro delle verifiche contro il codice reale che ha prodotto questa
versione, utile solo a chi voglia capire perché una voce è segnata chiusa
o aperta.

**Aggiornata al**: 17 settembre 2026. **Non definitiva** — verificare lo
stato di ogni voce contro il codice reale prima di assumerla vera, la
stessa disciplina che ha prodotto questa versione dalla precedente.

**Non contiene** la rotazione del token GitHub, di proposito — gestita
separatamente, fuori da questo documento.

---

## Come leggere questa roadmap

Ogni voce ha uno stato (**Aperta**, **Parziale**, **Chiusa**), uno sforzo
stimato, e le dipendenze reali. Le priorità (P0-P3) non replicano l'ordine
di EG #10 — sono state **ricalcolate** sul valore e costo di oggi, non su
quelli del 2 settembre, perché diverse voci sono cambiate di costo da
allora (alcune sono diventate più economiche perché il lavoro propedeutico
è stato fatto; altre sono rimaste ferme).

---

## P0 — valore alto, costo basso oggi (non c'era a settembre)

Queste voci sono in cima non perché lo fossero in origine, ma perché il
lavoro di questa settimana ne ha abbassato il costo o ne ha rivelato
l'urgenza.

### ~~H1~~ — Riconciliare `diagnostic_assembly.py` con `clinical_pipeline.py`
**Riformulata, non "chiusa" nel senso semplice.** Un'indagine più
approfondita (`docs/rlm_on_memory_decision_record.md`, 1-6 settembre,
mai letto prima di questa settimana) ha mostrato che H1 era una domanda
mal posta: non "quale pipeline tenere", ma la ricostruzione di un piano
già esistente — `IntuitionEngine` (veloce, associativo) e
`DifferentialEngine` (seriale, controllato) sono **due motori già dentro
`clinical_pipeline.py`**, pensati per ricevere due tipi di recupero
diversi (a un passaggio per l'intuizione, ricorsivo per il differenziale)
ma oggi alimentati dallo stesso recupero superficiale.

**Verificato con numeri, non un'opinione**: il recupero ricorsivo degrada
le ricerche semplici di circa 15-30 punti (dipendenza dalla profondità),
ed è soggetto a "overreach" (narrazioni senza fonte che superano
comunque il controllo di provenienza). Il recupero a un passaggio fallisce
nel modo opposto, per omissione. Il doppio percorso è giustificato dalla
loro asimmetria, non dalla superiorità di uno dei due.

**`IntuitionEngine` verificato per intero**: si autodichiara *"research
scaffold"*, e lo è — le sue ipotesi primarie sono etichette segnaposto
(`candidate_1`, `candidate_2`), alimentate da metriche neuro-dinamiche i
cui pesi (`AreaInteractionPrior`) sono numeri scelti a mano, a loro volta
alimentate da aree (`CaseContextArea`, `EpidemiologyArea`) che calcolano
la salienza contando le chiavi di un dizionario, non il contenuto.

**Un primo collegamento reale è stato fatto**: `DifferentialEngine.rank()`
ora promuove la migliore ipotesi reale e collegata al grafo (prodotta da
B1, sopra) quando la primaria di `IntuitionEngine` è un segnaposto —
verificato end-to-end: "aneurysm-osteoarthritis syndrome" con 19 cammini
di provenienza reale, al posto di "candidate_1". Il contributo
dell'intuizione non viene scartato, solo retrocesso.

**Secondo collegamento fatto**: `IntuitionEngine` produce ora nomi di
diagnosi reali (`rank_differential()`, la stessa similarità fenotipica
pesata per specificità già costruita e mai collegata a produzione),
sostituendo `candidate_1`/`candidate_2` — risalita fino alla radice del
problema: `SemanticMemoryStore` in `app.py` è vuoto, quindi il recupero a
un passaggio ricade sempre su tre prove inventate a mano
(`grounding_score` fisso a 0,58/0,46/0,5). Il collegamento non risolve
quel vuoto — dà a `IntuitionEngine` una seconda fonte indipendente di
contenuto reale. Verificato: quando l'intuizione produce già un'etichetta
vera, `DifferentialEngine` correttamente **non promuove più nulla sopra
di essa** — la rete di sicurezza della sezione precedente resta intatta
solo per quando serve davvero.

**Una scoperta interessante durante la verifica**: le due fonti reali
(similarità fenotipica e enumerazione di meccanismi) danno **risposte
diverse** sugli stessi reperti — "contractural arachnodactyly, congenital"
contro "aneurysm-osteoarthritis syndrome". Non è un difetto: è
esattamente il disaccordo informativo che l'intera architettura a doppio
percorso prevede.

**Ancora aperto**: il vero cancello D1 (instrada *quanti* percorsi
girare, non quale) resta codice morto, mai chiamato in `run()`, e fa
instradamento di protocollo, non valutazione di complessità clinica; le
aree a monte (`CaseContextArea`, `EpidemiologyArea`) restano segnaposto;
le scale dei punteggi fra le due fonti restano non comparabili (verificato
ancora: 7,341 contro un massimo di 1,0 sul lato grafo) — la formula di
`IntuitionEngine` resta il problema più grande, non affrontato.

### ~~B1~~ — `MechanismEnumerator` dentro `_alternative_hypotheses()`
**Chiusa.** `clinical_pipeline.py` ora collega un `MechanismEnumerator`
reale, costruito contro il grafo HPO vero (non più `KnowledgeGraphClient`,
il segnaposto da 7 righe che c'era prima), attaccato al `DreamTrainer`
solo quando un caso fornisce reperti — mai a costruzione fissa, per non
caricare il grafo reale (6,6s) su richieste che non ne hanno bisogno.
Verificato end-to-end: ipotesi enumerate reali, con cammini del grafo
autentici (inclusi collegamenti genici) come provenienza.

**Una scoperta di prestazioni non prevista, che rafforza H3**:
`MechanismEnumerator.run()` costa circa 3,5-4 secondi **per candidato**
contro il grafo reale (misurato: 2 candidati ~7s, 5 ~20s, 10 ~40s —
lineare, non un costo fisso) — mai misurato prima, perché l'enumeratore
era sempre stato esercitato solo su grafi fixture piccoli. Un limite di
sicurezza (`DREAM_ENUMERATION_CANDIDATE_CAP = 8`) tiene il ramo dream
utilizzabile oggi; il costo per candidato resta un problema di
prestazioni a sé, non risolto qui.

**Un difetto trovato e corretto durante la chiusura**: una prima versione
caricava il grafo reale a ogni chiamata di `_build_runtime_components()`,
indipendentemente dalla presenza di reperti — ha rallentato l'intera
suite di test da 13 a 99 secondi, perché una dozzina di test preesistenti
esercita questa pipeline senza reperti. Corretto collegando l'enumeratore
solo quando serve davvero.

---

## P1 — blocchi reali su altro lavoro, costo moderato

### A3.1 — Set di riferimento per la copertura
**Stato**: Aperta. **Sforzo**: 1 settimana, curatela clinica.
**Blocca**: B4 (valutazione clinica). Senza denominatore, `measure_coverage`
non misura nulla. **Decisione ancora pendente**: chi la cura — la
decisione più costosa rispetto al proprio beneficio in tutto il documento
precedente, e resta tale.

### D1 — `ModelRouter` come gate di complessità reale
**Stato**: Aperta. Letto per intero: 20 righe, instradamento fisso,
nessuna logica di complessità. **Sforzo**: 1 settimana.
**Blocca**: l'intero blocco D (D2-D5) — un vero collo di bottiglia, non
solo una voce fra tante.

### A3.3 + G3 — Stato SNOMED e riconciliazione concettuale (fuse)
**Stato**: Parziale. La verifica sullo stato di appartenenza dell'Italia a
SNOMED International è stata tentata questa settimana — assenza
consistente su più fonti, non conferma certa. La riconciliazione
concettuale (G3) è già parzialmente indirizzata, ma non tramite MONDO come
previsto in origine — tramite crosswalk UMLS, in attesa di registrazione.
**Sforzo residuo**: 1 giorno per la conferma diretta con SNOMED
International; il resto dipende dalla registrazione UMLS, già in corso.
**Perché fuse**: entrambe dipendono dalla stessa incertezza di licenza, e
risolverla una volta serve a entrambe.

---

## P2 — valore reale, costo alto o sequenza-dipendente

### A1 — Equivalente ricorsivo del grounding
**Stato**: Aperta, invariata. **Sforzo**: 1-2 settimane.
**Vincolo non negoziabile, ereditato da EG #10 e confermato ancora
valido**: deve precedere D5 — un A/B eseguito senza questo confronta
grandezze non commensurabili.

### A2 — Corpus di validazione longitudinale
**Stato**: Aperta, nessuna evidenza di lavoro iniziato. **Sforzo**: 2
settimane. Nessun blocco diretto su altro lavoro identificato — può
scorrere in parallelo, non in cima.

### B0b → B0c → B0d — Estrazione guidata dall'indice
**Stato**: B0c/B0d **chiuse** questa settimana passata (verificato:
`assertion.py`, 2-6 settembre). **B0b resta aperta**, bloccata da due
decisioni mai prese: lingua di riferimento per il collegamento
concettuale, ampiezza del lessico nella prima versione. **Sforzo**: 2-3
settimane una volta decise.

### B4 — Valutazione clinica su casi selezionati per densità
**Stato**: Aperta. **Blocca su**: B1 (ora più vicina), A3.1 (ancora
lontana). **Sforzo**: 2 settimane.

### D2, D3, D4, D5 — Dual-path diagnostico
**Stato**: Aperte, bloccate da D1 e (per D5) da A1. Sforzo invariato da
EG #10: 1-2 settimane ciascuna, D5 in coda per il vincolo d'ordine.

### A3.2 — Importatori LOINC, ATC, ECTO
**Stato**: Aperta. **Bloccata da**: decisione sulle categorie oltre il
fenotipo, in ordine di priorità — ancora non presa. **Sforzo**: 1-2
settimane ciascuno, meccanico una volta decise le categorie.

---

## P3 — continuo, o dipendente da eventi esterni

### E2 — Sostituire "Gemma 4" con un artefatto verificabile
**Stato**: Aperta. Verificato: "Gemma 4" compare letteralmente in quattro
file (`model_capability_registry.py`, `specialist_adapters.py`,
`rlm_model_adapter.py`, `specialist_runtime.py`) come segnaposto per *"un
modello di ragionamento generale a pesi aperti"* — non confermato se
corrisponda a un modello realmente rilasciato con questo nome esatto.
**Sforzo**: 3 giorni, principalmente verifica, non implementazione.

### E5 — Protezione dei confini contestuali nei blocchi di testo
**Riscritta rispetto a EG #10**: la voce originale parlava di blocchi
Docling. Docling è stato rimosso dal progetto (rimosso prima di questa
sessione di analisi, sostituito da Nemotron-Parse/LlamaParse). La stessa
esigenza di protezione dei confini si applica ai blocchi prodotti dal
nuovo estrattore — da riformulare per quello, non da completare come
scritta in origine. **Sforzo**: 1 settimana.

### H3 — Prestazioni sul grafo reale (tentativo di correzione fatto, respinto con prove, causa più profonda del previsto)
**Tentato**: raggruppare le chiamate `edges_from()` per livello del
fronte di ricerca invece di farne una per nodo (`FalkorConceptGraph.edges_from_many()`),
per ridurre 533 andata-ritorno a una manciata. **Risultato: peggio, non
meglio, su tutti e tre i casi reali testati** (0,5x, 0,7x, 0,2x la
velocità originale) — un lotto con dimensione limitata (150 concetti)
non ha risolto il caso peggiore. Causa trovata, non presunta: reperti
comuni come "epatomegalia" espandono a 1.396 concetti al secondo
livello, e recuperare i loro archi — anche raggruppati — restituisce
oltre 100.000 archi in una singola risposta. **Il collo di bottiglia si
è spostato dal numero di andata-ritorno al volume di dati trasferiti**,
e raggruppare le richieste non riduce il volume totale. La stessa causa
è stata verificata anche per `MechanismEnumerator`: 20,25s su FalkorDB
contro 3,04s in Python, ancora più lento, senza nemmeno tentare una
correzione — la prova era già sufficiente a non procedere.

`edges_from_many()` resta nel codice come infrastruttura corretta e
testata (12 test dedicati), ma **non collegata al percorso critico** di
`retrieve_candidates`, che è stato riportato alla versione originale
funzionante.

**Cosa resta da valutare, non ancora deciso**: (a) spostare la logica di
ammissibilità (la distinzione gene/malattia) dentro la query Cypher
stessa, cosa che ridurrebbe il volume trasferito a soli i candidati
finali ammissibili — respinto finora per il rischio di replicare una
regola a più condizioni senza poter riusare i test esistenti
direttamente contro di essa; (b) un modello ibrido, dove FalkorDB resta
la fonte di verità persistente ma il livello di attraversamento a caldo
continua a usare una cache in memoria costruita da FalkorDB all'avvio —
non ancora esplorato. **Sforzo**: 1-2 settimane per (a) con validazione
rigorosa, o alcuni giorni per (b).

### H4 — Popolamento del deposito UMLS cifrato
**Nuova**: dipende dal completamento della registrazione UMLS, in corso.
Una volta ottenuta la chiave, il popolamento (crosswalk per i concetti già
tracciati) è meccanico. **Sforzo**: 1-2 giorni una volta sbloccata.

### E1, E3, E4 — Consolidamento
**Stato**: non riverificate in questa ottimizzazione — sforzo e stato
invariati da EG #10 (2 settimane, 2-3 settimane, 2 settimane
rispettivamente). Da controllare con la stessa disciplina prima di
pianificarle.

### F1-F5 — Privacy e oracolo a predicati
**Stato**: Aperte, nessuna evidenza di lavoro iniziato su nessuna delle
cinque. Sforzo invariato da EG #10. Tre claim bloccanti dipendono in
parte da questo blocco (vedi sotto).

### G1, G2, G4 — Vocabolario e curatela
**Stato**: Aperte, nessuna evidenza. Sforzo invariato da EG #10.

---

## Voci chiuse — verificate, non ripetute oltre questo elenco

A0 (risoluzione concetti), A3 nella sua parte meccanica (grafo reale
popolato, ben oltre l'ambizione originale: 1.273.466 archi, 29.053
concetti), B0c/B0d, B2, B3, C1 (`rlm_engine.py`, letteralmente etichettato
"Block C" nel proprio commit), C2 nella sua parte sostanziale
(`context_environment.py`).

---

## Decisioni ancora pendenti, con priorità aggiornata

| # | Decisione | Blocca | Priorità |
|---|---|---|---|
| 1 | Chi cura il set di riferimento per la copertura | A3.1 → B4 | **La più urgente**: costo di decisione basso, beneficio alto |
| 2 | Lingua di riferimento per il collegamento concettuale | B0b | Alta |
| 3 | Ampiezza del lessico nella prima versione | B0b, ne determina la durata | Alta |
| 4 | Categorie oltre il fenotipo, in ordine di priorità | A3.2 | Media |
| 5 | Classificatore di asserzione: locale o addestrato di dominio | Probabilmente risolta implicitamente da `assertion.py` — da confermare | Bassa, verifica |

---

## Claim bloccanti — stato aggiornato

| Claim | Bloccante | Stato oggi |
|---|---|---|
| `rlm.dual_path_beats_single_path` | Sì | Aperto — misurabile dopo A1 |
| `rlm.disagreement_is_informative` | Sì | Aperto — misurabile dopo D2 |
| `privacy.predicate_budget_prevents_reconstruction` | Sì | Aperto — nessuno strumento ancora (blocco F) |
| `rlm.dream_hypotheses_add_value` | No | Aperto — **più vicino a essere misurabile** ora che B1 è a basso costo |
| `rlm.coverage_predicts_grounding` | No | Aperto |
| `rlm.open_weight_root_is_sufficient` | Dormiente | Invariato |
| `rlm.recursive_helps_only_on_complex_cases` | — | Ritirato |

---

## Nota di chiusura

Questa versione sostituisce sia le 4 fasi precedenti di questo file sia il
Report EG #10 — è l'unica roadmap. Il registro di verifica che l'ha
prodotta, con ogni controllo documentato singolarmente, resta in
`docs/ROADMAP_VERIFICATION_LOG.md`. Non è definitiva: le
voci del blocco E non riverificate (E1, E3, E4) e l'intero blocco F
andrebbero controllate con la stessa disciplina di verifica diretta contro
il codice prima di pianificarle con fiducia.
