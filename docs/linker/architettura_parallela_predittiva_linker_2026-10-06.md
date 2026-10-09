# Linker anatomico a strati paralleli, predittivo, con grafo convergente (6 ottobre 2026)

Domanda di Frank: possiamo far lavorare i livelli in parallelo e in modo predittivo, con grafi convergenti e strati paralleli? Qual è la soluzione migliore secondo i lavori più recenti?

**Risposta breve.** Sì. La soluzione proposta trasforma i nove passi da una cascata di cancelli in **nove strati che lavorano insieme sullo stesso insieme di ipotesi**, un sottografo anatomico. L'ipotesi è attivata in anticipo dallo stato del referto (previsione dall'alto) e sostenuta o contraddetta dal basso da flussi di prova indipendenti. Le prove si combinano lungo il grafo (parte-di, è-un, sorelle, lato opposto), pesate per l'affidabilità misurata di ciascun flusso. Si accetta solo quando flussi di meccanismo diverso convergono e l'errore di previsione residuo è basso. Altrimenti si ripiega sul padre nel grafo, o ci si astiene dicendo quali flussi erano in disaccordo. I due LLM contano come **un solo flusso**, perché i loro errori sono correlati. Niente di questo è ancora costruito: qui ci sono il progetto, le fonti e il piano a tappe, e ogni tappa ha una misura che decide se si va avanti.

## 1. Che cosa dicono le fonti più recenti

**Neuroscienze del linguaggio: gerarchico, bidirezionale, iterativo.**
- Nel modello a codifica predittiva dell'N400 (Nour Eddine, Brothers, Wang, Spratling, Kuperberg, *Cognition* 2024) i livelli ortografico, lessicale e semantico lavorano tutti insieme. Ogni livello ha unità di stato (che cosa crede) e unità d'errore (che cosa la previsione dall'alto non spiega). Il contesto pre-attiva dall'alto i tratti semantici attesi. La lettura è un ciclo: aggiorna lo stato, calcola l'errore, ricostruisce dall'alto. L'N400 corrisponde all'errore di previsione lessico-semantico totale: sale, raggiunge il picco intorno alla quinta iterazione e scende quando il sistema converge sul significato. Il modello riproduce effetti di frequenza, priming, prevedibilità e le loro interazioni (gli effetti lessicali si riducono sulle parole prevedibili).
  → **Conseguenza:** non una cascata, ma livelli simultanei che si correggono a vicenda finché l'errore residuo scende. Se non scende, è il segnale di non aver capito.
- Integrazione di indizi pesata per affidabilità (Ernst & Banks 2002, risultato classico): quando due sensi danno una stima, il cervello le combina pesandole per la loro affidabilità. Uno studio del dicembre 2025 (BayesBench, arXiv 2512.02719) mostra che non tutti gli LLM lo fanno: alcuni superano il riferimento bayesiano, altri (GPT-5 Mini nell'esperimento) sono molto accurati ma non riducono il peso di un indizio inaffidabile.
  → **Conseguenza:** i pesi dei flussi vanno **misurati** sul gold set, non lasciati al modello né scelti a mano per sempre.
- Sistemi "Thousand Brains" (Leadholm, Clay, Knudstrup, Lee, Hawkins, luglio 2025, arXiv 2507.04494; implementazione Monty): molti moduli paralleli tengono ciascuno un insieme di ipotesi con un punteggio d'evidenza, che sale quando l'osservazione combacia con la previsione e scende quando no. Si modulano a vicenda votando lateralmente, e questo accelera il consenso.
  → **Conseguenza:** flussi paralleli sullo **stesso spazio di ipotesi**, ognuno con il proprio punteggio, che convergono per voto pesato.

**IA: il consenso non è verifica.**
- "Nine Judges, Two Effective Votes" (Kohli, arXiv 2605.29800): nove giudici LLM di sette famiglie valgono circa **2–2,5 voti indipendenti**. L'accuratezza del pannello resta 8–22 punti sotto quella attesa con voti indipendenti, e il miglior giudice singolo eguaglia o supera il pannello. Un'aggregazione più furba recupera al massimo l'11% del deficit di indipendenza.
- "Consensus Is Not Verification" (Denisov-Blanch, Kazdan, … Koyejo, ICML): nessuna strategia di aggregazione migliora in modo coerente la veridicità senza un verificatore esterno. Gli errori sono correlati tra famiglie di modelli (fino a 0,35 di correlazione anche su stringhe casuali), e la confidenza dichiarata segue l'accordo atteso, non la correttezza. Aggregare funziona **quando esiste un verificatore** (recupero, strumenti, esecuzione, ontologia, feedback umano).
- Aggregazione per propagazione di credenze (Mehrabi, Haeri Boroujeni, Razi, arXiv 2606.00405): un grafo di fattori fra modelli batte il voto di maggioranza (MMLU 79,8% contro 73,5%; GPQA 43,6% contro 37,3%) e protegge il modello più affidabile dal rumore dei più deboli.
  → **Conseguenza:** la convergenza va cercata tra flussi di **meccanismo diverso**, con un verificatore simbolico (grafo anatomico, controlli deterministici). La propagazione su grafo è il modo giusto di combinare, ma non crea indipendenza dove non c'è.

**Gerarchia ed errori.**
- MedPath (Mishra, Aziz, Calixto 2025, arXiv 2511.10887): sul linking biomedico circa il 20% degli errori è troppo generico (antenato) e il 20% troppo specifico (discendente). Con una metrica gerarchica, l'accuratezza@1 di un recuperatore TF-IDF passa dal 51% al 68,6%.
  → **Conseguenza:** molti errori sono di granularità. Un grafo permette di **ripiegare sul padre** invece di sbagliare o astenersi del tutto, e di misurare gli errori per distanza nel grafo.

**Garanzia sugli accettati.**
- Selective Conformal Risk Control (Xu, Guo, Wei, dicembre 2025, arXiv 2512.12844): garantisce che l'errore sui casi accettati resti sotto un livello α e che la frazione accettata superi una soglia ξ. Ipotesi: scambiabilità tra calibrazione e uso, e una regola di selezione simmetrica. La variante SCRC-T dà garanzie esatte su campioni finiti.
  → **Conseguenza:** è lo strumento per il passo 9, insieme a Learn-then-Test. Richiede il gold set e vale solo per la popolazione su cui è calibrato.

Non letto: "Better Later Than Sooner: Neuro-Symbolic Knowledge Graph Construction via Ontology-grounded Post-extraction Correction" (arXiv 2605.29168). Il server ha rifiutato la richiesta per limite di frequenza, quindi è citato solo per titolo.

## 2. Che cosa abbiamo già (verificato oggi sul codice e sui dati)
- **UBERON basic** (versione fissata 2025-05-28, sha256 verificato): 14.624 termini, 19.004 archi è-un, **9.149 archi parte-di**, 794 archi "mutualmente disgiunti" (vincoli negativi pronti) e 6.037 termini con riferimento FMA (il pool del linker, 6.162 candidati, ne deriva).
  - Esempi reali: la cistifellea è parte del sistema biliare; il ponte è parte del metencefalo e del tronco encefalico; ileo e digiuno sono parte dell'intestino tenue; il midollo osseo è parte dell'elemento osseo e del sistema emopoietico; il midollo spinale è parte del sistema nervoso centrale ed è mutualmente disgiunto dall'encefalo.
  - **Attenzione:** lato e suddivisioni stanno spesso nell'è-un, non nel parte-di ("left kidney" è-un "kidney"; "sigmoid colon" è-un "proximal-distal subdivision of colon", senza parte-di diretto). Il grafo per il ripiego deve combinare le due relazioni e ricavare il lato dai nomi e dalle famiglie TotalSegmentator.
  - Alcune voci non hanno quel nome in UBERON ("porta hepatis", "left main coronary artery", "fifth lumbar vertebra"): servono sinonimi o il lessico.
  - Oggi `load_obo_terms` legge solo nomi, sinonimi e riferimenti: **le relazioni non vengono caricate.**
- `memory/spreading_activation.py`: attivazione diffusa *vincolata* su grafo (vincoli di relazione, decadimento per salto, soglia), già scritta per le malattie. È il meccanismo per la previsione dall'alto.
- `memory/assertion.py`: rilevazione deterministica di negazione, incertezza, esperiente e temporalità (stile ConText). È un flusso di contesto.
- `memory/word_senses.py` (oggi): sensi in competizione su contesto, lingua e dati.
- I controlli deterministici del linker (`verify`, `covers`, `wrong_system`, `_anchored`, `_context_allows`), la tabella parti, il lessico, il recuperatore, i due LLM e la traduzione. Oggi sono stadi in cascata che si fermano al primo esito.
- FalkorDB è usato per il grafo delle malattie, non per l'anatomia. Per il linker non serve subito: un sottografo in memoria basta (poche decine di nodi per menzione).

## 3. L'architettura proposta

### 3.1 Uno spazio di ipotesi comune: il sottografo della menzione
Per ogni menzione si costruisce un piccolo grafo **H**:
- i candidati proposti da tutti i flussi di riconoscimento (lessico, parti, morfologia, encoder);
- i loro padri e figli (parte-di ed è-un), sorelle, controlaterali e i nodi disgiunti;
- i **sensi non anatomici** dell'inventario (globuli bianchi, gigabyte, bypass…) come nodi concorrenti;
- un nodo **NESSUNO**.

Tutti gli strati lavorano su questo stesso H: è il "grafo convergente".

### 3.2 La previsione dall'alto (strato 1, stato del referto)
Prima di leggere la menzione, lo stato del referto attiva H:
- intestazione e tecnica (modalità, distretto, lato dichiarato);
- lingua;
- strutture già collegate nelle frasi precedenti dello stesso referto;
- storia nota ("colecistectomia pregressa": cistifellea assente).

L'attivazione si diffonde con `spreading_activation` vincolato a parte-di ed è-un, con decadimento. È il priming: "TC torace" pre-attiva polmoni, mediastino e coste.
**La previsione non accetta mai nulla da sola.** Abbassa o alza la soglia di sorpresa e rende misurabile l'errore di previsione: una struttura forte dal basso ma non prevista dall'alto (ginocchio in una TC torace) genera un errore alto, che va spiegato o porta all'astensione.

### 3.3 I flussi dal basso, in parallelo
Ogni flusso guarda la menzione e la frase **indipendentemente dagli altri**. Per ogni nodo di H restituisce un'evidenza a favore, un'evidenza contro o un veto, e il proprio motivo.

| Flusso | Meccanismo | Passo | Tipo di prova |
|---|---|---|---|
| A. Lessico esatto + tabella parti | conoscenza curata | 2 | a favore (forte) |
| B. Morfologia (radici greco-latine) | regole | 3 | a favore (debole, solo proposta) |
| C. Encoder denso | somiglianza appresa | 3 | a favore (graduata, solo proposta) |
| D. Sensi (contesto, lingua, dati) | inventario + punteggio | 4 | a favore e contro tra sensi |
| E. Integrazione: lato, numero, tipo, tessuto/spazio, regione | regole deterministiche | 4 | **veto** |
| F. Vicini nel grafo: sorelle, controlaterale, padre/figli, disgiunti | struttura UBERON + famiglie TS | 5 | contro (se un vicino spiega altrettanto bene) |
| G. Asserzione (negazione, storia, incertezza) | ConText | 1/4 | modula, non collega |
| H. Scelta vincolata dei due LLM | modelli linguistici | 6 | a favore, **un solo flusso** |
| I. Ri-derivazione cieca (traduzione IT→EN, descrizione → ricerca) | meccanismo diverso | 7 | a favore/contro |

**Parallelo davvero, a costo controllato.** I flussi A–G sono deterministici e veloci (millisecondi) e partono tutti insieme. H e I costano tempo e denaro: partono solo se dopo A–G l'ipotesi migliore non ha già convergenza e margine. È il riconoscimento rapido delle parole frequenti nel cervello: quando basta, la deliberazione non serve.

### 3.4 La convergenza: propagazione sul grafo
Le prove si combinano su H come un piccolo **grafo di fattori**:
- **Evidenza dei flussi:** ogni flusso contribuisce con un peso pari alla sua affidabilità misurata (alla Ernst & Banks; il peso viene dal gold set, vedi 3.6).
- **Vincoli rigidi:** un veto di E azzera il nodo. Lo stesso vale per i nodi disgiunti tra loro quando la frase ne colloca uno.
- **Propagazione verticale:** la prova per un figlio sostiene in parte il padre ("sigmoide" sostiene "colon"), non il contrario. È ciò che rende possibile il ripiego.
- **Inibizione laterale:** sorelle e controlaterali competono (attivazione interattiva). Una prova che non distingue il rene destro dal sinistro non può far vincere nessuno dei due.
- **Poche iterazioni:** parte-di ed è-un sono quasi un albero, quindi la propagazione converge in 2–3 passi ed è esatta sugli alberi. Il grafo è piccolo, il costo è trascurabile.

Il risultato: una distribuzione su H ∪ {NESSUNO, sensi non anatomici}, più l'**errore di previsione residuo**, cioè la distanza fra ciò che lo stato del referto si aspettava e ciò che i flussi hanno trovato.

### 3.5 Decidere: accetta, ripiega, astieni (passi 8 e 9)
Si accetta il nodo migliore **solo se**:
1. ha margine sul secondo (incluse sorelle, controlaterale e sensi non anatomici);
2. è sostenuto da **almeno due flussi di meccanismo diverso**. A conta come uno, C come uno, H come uno solo anche se i modelli sono due. Per un nome esatto e non ambiguo del lessico, il secondo sostegno può essere F (nessun vicino spiega meglio) insieme a 1 (coerente con lo stato del referto);
3. nessun veto;
4. errore di previsione residuo sotto soglia, oppure spiegato da una prova forte dal basso.

Altrimenti si **ripiega sul padre**: il primo antenato di H che soddisfa le stesse condizioni, con relazione `part_of` o `is_a` dichiarata. MedPath indica che circa il 40% degli errori è di granularità, ed è qui che si recupera. Se nessun antenato regge, ci si **astiene** con un motivo strutturato: quali flussi erano d'accordo, quali contro, quale vicino o senso concorreva, quanto era l'errore di previsione. Il passo 8 diventa quindi una misura, non una lista di motivi.

### 3.6 Pesi e soglie: dai dati, con garanzia
- **Fino al gold set:** pesi prudenti fissati a mano, la regola dei due flussi indipendenti e le stesse astensioni di oggi. Nessuna promessa di miglioramento.
- **Con il gold set:**
  - affidabilità di ogni flusso (errore per flusso);
  - correlazione degli errori fra flussi, quindi un numero effettivo di flussi indipendenti, come i "voti effettivi" di Kohli: se A e C sbagliano insieme, contano meno di due;
  - pesi e soglie appresi sulla parte di calibrazione e certificati sulla parte congelata con Learn-then-Test o Selective Conformal Risk Control (errore ≤1% sugli accettati, copertura minima dichiarata).
- La garanzia vale per quella popolazione e quel sistema: si ricalibra a ogni cambio di modello, lessico o ospedale.

### 3.7 Tracciabilità (MDR)
Ogni decisione salva: l'attivazione dall'alto, l'evidenza di ogni flusso per ogni nodo, i veti, la distribuzione finale, l'errore residuo e la regola che ha deciso. È la "spiegazione" verificabile da un revisore, ed è più ricca di quella di oggi.

## 4. Perché questa e non le alternative
- **Un solo modello neurale end-to-end** (LLM o encoder addestrato): errori correlati, nessun verificatore, nessuna traccia, nessuna garanzia per sottogruppi. Scartato per un dispositivo medico.
- **Più LLM a maggioranza:** le fonti 2025–2026 mostrano che non crea indipendenza (circa 2 voti effettivi su 9) e che il consenso non è verifica.
- **Una rete a codifica predittiva vera** (stile Nour Eddine 2024): è il modello cognitivo più fedele, ma va addestrata su dati che non abbiamo ed è opaca. Ne prendiamo l'**architettura** (stato + errore, dall'alto + dal basso, iterazioni), non la rete.
- **Scelta:** neuro-simbolico, con flussi paralleli eterogenei, convergenza su grafo anatomico, previsione dall'alto, decisione selettiva calibrata. Simula l'organizzazione della comprensione umana e aggiunge la precisione dei modelli dove misurano bene (proposta, scelta tra candidati già compatibili), con la garanzia che solo la statistica sugli accettati può dare.

## 5. Piano a tappe (ogni tappa con la sua misura)
| Tappa | Cosa | Misura per andare avanti |
|---|---|---|
| T0 ✔ (7 ott) | Rifattorizzare gli stadi attuali in flussi che **restituiscono evidenze** invece di fermarsi al primo esito; stesse decisioni di oggi | Stesse decisioni su tutti i test e sul bench (nessuna differenza); traccia per flusso |
| T1 ◐ (7 ott) | Grafo anatomico da UBERON basic (parte-di + è-un + disgiunti) + famiglie TotalSegmentator per il lato; flusso F (vicini) e ripiego al padre | Zero nuovi errori silenziosi su dev/held-out IT/EN; quante astensioni diventano link al padre corretti |
| T2 ◐ (7 ott sera) | Stato del referto: parser dell'intestazione, lingua, strutture già collegate; previsione con `spreading_activation`; il referto intero passato come contesto | Errori di "distretto sbagliato" intercettati su casi costruiti e sui corpora pubblici; nessun calo di precisione |
| T3 ◐ (7 ott) | Convergenza: profilo dei meccanismi (support/conflicts), area dell'esame come previsione, Sistema 2 sul conflitto, ruoli; la soglia sulla convergenza attende il gold set | Copertura e precisione rispetto a T2; chiamate LLM risparmiate; latenza |
| T4 ◐ (7 ott: strumento pronto) | Gold set: affidabilità e correlazione per flusso, pesi appresi, soglia certificata (LTT/SCRC; `selective_calibration.py`) | Precisione ≥99% certificata sugli accettati, con copertura dichiarata |

T0 e T1 si possono fare subito e non dipendono dal gold set. T4 sì.

## 6. Limiti onesti
- Quando questo documento è stato scritto (6 ottobre) era un progetto. Al 7 ottobre T0 e T1 sono costruite (sezione 7); T2, T3 e T4 no. Le fonti sostengono l'organizzazione, non i nostri numeri.
- La propagazione su grafo non crea indipendenza: se tutti i flussi sbagliano per la stessa ragione (un nome sbagliato nel lessico), converge sull'errore. Per questo restano i veti deterministici, il ripiego e il gold set.
- Il grafo UBERON basic ha lacune (lato e suddivisioni in è-un, nomi mancanti). Il grafo va costruito con test, e una relazione sbagliata è un errore silenzioso nuovo.
- Più flussi vogliono dire più parametri: senza gold set i pesi sono scelte di progetto.
- Le cifre delle fonti sono quelle riportate negli articoli letti oggi. Gli studi su LLM (Kohli; Denisov-Blanch et al.; Mehrabi et al.; BayesBench) sono preprint o atti recenti, non tutti revisionati.

## Fonti
- Nour Eddine S., Brothers T., Wang L., Spratling M., Kuperberg G.R. (2024). A predictive coding model of the N400. *Cognition* 246:105755. https://kuperberg.mgh.harvard.edu/wp-content/uploads/Nour-Eddine-Kuperberg-Cognition-2024.pdf
- Leadholm N., Clay V., Knudstrup S., Lee H., Hawkins J. (2025). Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference. https://arxiv.org/abs/2507.04494
- Emergent Bayesian Behaviour and Optimal Cue Combination in LLMs (BayesBench, 2025). https://arxiv.org/abs/2512.02719
- Ernst M.O., Banks M.S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature* 415 (risultato classico, non riletto oggi).
- Kohli G. Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels. https://arxiv.org/abs/2605.29800
- Denisov-Blanch Y. et al. Consensus Is Not Verification: Why Crowd Wisdom Strategies Fail for LLM Truthfulness (ICML). https://arxiv.org/abs/2603.06612
- Mehrabi N., Haeri Boroujeni S.P., Razi A. From Talking Words to Sharing Thoughts: Scalable Multi-LLM Aggregation via Structured Message Passing. https://arxiv.org/abs/2606.00405
- Mishra N., Aziz W., Calixto I. (2025). MedPath: Multi-Domain Cross-Vocabulary Hierarchical Paths for Biomedical Entity Linking. https://arxiv.org/abs/2511.10887
- Xu Y., Guo W., Wei Z. (2025). Selective Conformal Risk Control. https://arxiv.org/abs/2512.12844
- Non letto (limite di frequenza del server): Better Later Than Sooner: Neuro-Symbolic Knowledge Graph Construction via Ontology-grounded Post-extraction Correction. https://arxiv.org/abs/2605.29168
- Dati: UBERON basic v2025-05-28, https://github.com/obophenotype/uberon/releases/download/v2025-05-28/uberon-basic.obo (conteggi calcolati oggi).
- Documenti del Project: `nove_stadi_dettaglio_2026-10-06.md`, `disambiguazione_contesto_lingua_2026-10-06.md`, `risultati_linker_live_2026-10-05.md`.


## 7. Stato al 7 ottobre 2026: T0 fatto, T1 fatto a metà (e perché)

Branch `feat/linker-streams` (commit 08b8563 e 2112adb), bundle `melampo-linker-streams.bundle`.

**Correzione trovata strada facendo.** Fotografando le decisioni sul bench è emerso che l'inventario dei sensi del giorno prima bloccava "muscolo ileo-psoas", perché leggeva "ileo" dentro la parola composta. Ora una forma conta solo come parola intera. Rispetto al commit precedente all'inventario, tutte le decisioni del bench sono identiche, tranne il motivo di una riga "LM" che resta un'astensione. C'è anche un test di regressione su tutti gli held-out.

**T0, fatto.**
- I flussi economici (sensi, controlli d'integrazione, codici di livello, lessico, tabella parti) vengono valutati tutti, ciascuno da solo, e restano nella traccia (`LinkResult.trace`).
- La decisione è separata (`_decide`) e usa lo stesso ordine di autorità di prima.
- I due LLM vengono interrogati in parallelo. Un test lo dimostra: le due chiamate devono incontrarsi su una barriera.
- **Misura:** 858 decisioni su 858 identiche fra prima e dopo (5 set, linker deterministico e con modelli finti deterministici). Le suite del linker passano. Nella suite completa restano gli stessi 34 problemi preesistenti (FalkorDB, HPO, pydicom).
- **Esempio di traccia:** in "GB 5040/mmc" si vede il conflitto. Il lessico da solo direbbe cistifellea, i sensi lo vietano.

**T1, grafo e vicini fatti; il ripiego al padre resta una proposta.**
- `anatomy_graph.py` legge è-un, parte-di e "spazialmente disgiunto" da UBERON basic.
- Le ancore classe↔UBERON stanno in un file curato (`anatomy_graph_anchors.json`). L'abbinamento automatico per sinonimi proponeva "insect arista" per il segmento epatico 6 e un "inferior lobe" di zebrafish per i lobi inferiori.
- **Vicini (passo 5):** se la scelta dei modelli ha fra le opzioni ammesse un vicino nel grafo (l'altro lato, una sorella, padre/figlio, una struttura disgiunta), il linker si astiene. Sui dati del bench con modelli finti non scatta mai: `covers` richiede già le stesse parole di contenuto. È una rete di sicurezza, non una fonte di copertura.
- **Ripiego al padre (passo 9):**
  - Sui 6.037 termini UBERON del pool: 1.196 risalgono a una classe, 70 coincidono con una classe e gli altri vengono rifiutati o non trovano una classe vicina.
  - Tre revisioni cieche indipendenti, di 100 risalite casuali ciascuna, hanno trovato rispettivamente 3 sbagliate e 9 dubbie, 1 e 13, 5 e 12. Errori tipici: l'ipofisi "parte" dell'encefalo, il piccolo omento dello stomaco, l'uraco della vescica, una cripta duodenale "parte" del tenue, che nella segmentazione è una classe separata.
  - Ogni revisione ha aggiunto regole: niente risalita attraverso spazi, superfici, solchi, forami, vasi, linfonodi, nervi, legamenti, mesenteri, ventricoli, bronchi, fontanelle o strutture embrionali; niente "tipo di"; rifiuto se il termine raggiunge due classi. Ogni campione nuovo ha comunque trovato errori nuovi.
  - Conclusione: il parte-di di UBERON è anatomia, non contenimento nella maschera TC. Qualche punto percentuale d'errore è lontano dall'obiettivo dell'1%.
  - **Decisione:** il ripiego al padre è una **proposta** (`result.fallback`) per la coda di revisione e per la misura sul bench (`graph_fallback_proposals`: quante proposte coincidono con il target). Diventa un link solo con `accept_parent_fallback=True`, da attivare dopo la certificazione sul gold set.
- Il bench e il gold set caricano il grafo quando c'è il file UBERON (`--no-graph` per il confronto).
- **Misura offline:** sul bench deterministico la copertura non cambia e restano 0 errori. Sulle 9 righe "altro concetto" prodotte dai modelli finti, tutte con target fuori dalle classi (appendice, utero, ovaio, falce…), il grafo non ha proposto nessuna classe sbagliata.

**Cosa resta.**
- Il run live del linking-bench con Nemotron e Gemma, per vedere quante "altro concetto" reali avrebbero una proposta corretta.
- T2 (stato del referto e previsione dall'alto) e T3 (convergenza pesata).
- La certificazione del ripiego e dei pesi sul gold set (T4).

## 8. Prima prova su testo reale (7 ottobre 2026, sera)

Corpora: PARROT italiano (285 referti, 229 menzioni campionate) e IU X-ray inglese (3.927 referti, 3.103 menzioni). Linker deterministico, senza rete, con il grafo. **Non c'è un gold set: le percentuali sotto sono quote di accettazione, non precisione.**

| | PARROT IT | IU X-ray |
|---|---|---|
| Accettate | 126 (55%) | 2.610 (84%) |
| Astenute | 103 | 493 |
| Il grafo cambia decisioni | 0 | 0 |
| Proposte di ripiego | 0 | 0 |

- **Astensioni, per motivo.** PARROT: forme ambigue per lingua/contesto 31 (polo, seno, base, corpo), codice di livello senza evidenza 29, aggettivo nudo 22. IU X-ray: aggettivo nudo 318 (cardiac 228, aortic 76), nome non riconosciuto 76 ("left base" 41, "right base" 20), "base" ambigua 56.
- **Codici di livello (T1/T2 della RM vs vertebre).** Su 25 menzioni PARROT di T1/T2/L4/L5/S1 il filtro ha astenuto tutte le sequenze RM (T1, T2) e accettato le vertebre solo con evidenza ("Crollo di L4", "livello L5-S1"). "Anterolistesi di L4 su L5" è astenuta: prudente, da recuperare con un indizio in più.
- **Controllo a occhio di 85 link accettati** (45 PARROT, 40 IU X-ray, campione casuale): 83 corretti; 1 errore silenzioso (**"left paraspinal" → muscolo autoctono sinistro** in "left paraspinal/retrocrural adenopathy": è la regione, non il muscolo) e 1 dubbio ("colon-sigmoidea" → colon).
  - **Causa dell'errore:** il lessico scarta la parola "muscles" dalla chiave di "left paraspinal muscles", quindi "paraspinal" da solo veniva letto come il muscolo, qualunque fosse la frase.
  - **Correzione (7 ott, sera, seconda versione):** una prima correzione per parola ("il nome nudo non indica il muscolo") è stata scartata perché ignorava il contesto. Ora "paraspinal" e "paravertebrale/i" sono una forma ambigua dell'inventario dei sensi: muscolo oppure regione accanto alla colonna. Decide la frase: "adenopathy", "massa", "linfonodi" → regione, astensione `sense_conflict:region`; "atrophy", "steatosi", "muscoli" → muscolo, link accettato; nessun indizio → `sense_unresolved`, astensione. Bench invariato (0 errori silenziosi); sul referto reale l'item ora si astiene.
  - **Limite:** l'inventario resta un elenco di forme ambigue, con un meccanismo unico ma conoscenza nei dati. Altre forme con la stessa struttura (nome del lessico a cui la chiave toglie una parola) non sono ancora elencate: la scansione sistematica (UMLS) è il passo che manca.
- **Difetti del campionatore** trovati e corretti (branch `fix/public-reports-loaders`): 19 menzioni attraversavano il confine di frase ("T12. Right", "destra. Aorta"); la frase passata al linker contiene a volte l'intestazione ("Quesito clinico: …" 14% in PARROT), cosa che T2 deve risolvere. IU X-ray eseguito senza `cap`: "heart" è 1.919 menzioni su 3.103; con `cap=30` il campione scende a 676 menzioni su 103 forme.
- **MultiCaRe:** `cases.parquet` ha le colonne `cases` e `article_id` (più casi per articolo); loader corretto, id unici.

## 9. T2, prima parte (7 ottobre 2026, sera)

Costruito: `report_state.py` (sezioni, titolo, quesito attaccato ai reperti, lingua, modalità, "referto sul rachide"); il linker accetta `report` e `at`; l'intestazione diventa `context`; gold set: frasi pulite, campo `section`, `evaluate --reports` passa lo stato come in uso.

Prima previsione dall'alto: i codici di livello. Misura su testo reale (nessun gold set): PARROT 126 → 129 accettate su 229 (le 3 nuove controllate a mano: "L2 - L3" in una scoliosi, due "Anterolistesi di L4 su L5"), IU X-ray invariato. Il primo tentativo accettava anche "spazi intersomatici del tratto C3-C7" e "C3": il controllo a occhio le ha trovate e il filtro ora cerca le radici delle parole (disc-, radic-, foram-, interso-, spazi) e non le parole intere.

Bench sintetico invariato: 0 errori silenziosi. Test: 2.180 passati (31 nuovi); gli stessi 34 problemi di ambiente.

Limiti: il "solo rachide" è una regola a parole, non un modello; un referto del rachide con una riga sul rene (frequente nei referti RM lombari completi) non conta come referto sul rachide e i suoi codici restano astenuti. È prudente, non completo.

## 10. Tutti i termini, non uno alla volta (7 ottobre 2026, notte)

**Domanda di Frank:** il contesto deve correggere ogni termine, non uno per volta. Risposta: sì, in linea di principio ogni termine si legge nel contesto, ma l'inventario dei sensi (7 forme) è scritto a mano e non scala. Mancava la conoscenza di *quali* forme sono ambigue. Ora si trova dai dati, e il contesto decide con tre strati.

**Misura (testo reale, senza gold set).** Su IU X-ray con cap, PARROT e un campione di 600 menzioni MultiCaRe, il 8% (IU X-ray), 23% (PARROT) e 21% (MultiCaRe) dei link accettati dal solo nome riguarda forme che i segnali strutturali marcano come a rischio; solo 1 su 129 era letto da un profilo dei sensi. In 70 link marcati controllati a mano: **1 errore netto** ("lid fissure" collegato al lobo inferiore destro: LID è una sigla italiana in un testo inglese) e circa 8 dubbi (thyroid function tests, hip flexion, pulmonary venous hypertension). Nei link non marcati i controlli a occhio precedenti (85) avevano 1 errore, già marcato dal segnale "nome senza il sostantivo di testa" (paraspinal).

**I tre strati.**
1. *Rilevamento dai dati* (`form_ambiguity.py`): segnali strutturali sul lessico e sulla tabella parti (nome senza testa, sigla in maiuscole, aggettivo singolo, prefisso spaziale, chiave condivisa). 113 chiavi su 732. È orientato alla copertura: un segnale non è un verdetto.
2. *Lingua della frase contro lingua del nome, per tutte le sigle:* il lessico sa già in quale lista (it/en) ogni nome è scritto. Una sigla scritta solo in italiano dentro una frase chiaramente inglese non è la struttura. Non è simmetrica: un referto italiano usa sigle inglesi come CCA, SVC, IVC, uno inglese non usa sigle italiane (la prima versione, simmetrica, faceva astenere "CCA sn" nel bench; corretta). Un profilo dei sensi, se c'è, decide al posto suo.
3. *Verifica del contesto con i due modelli* (`verify`, spenta di default): per i link dal solo nome a una forma marcata e senza profilo, entrambi i modelli leggono la frase; serve YES da tutti e due, altrimenti astensione con motivo `context_check_failed:no|unsure`. Testata con modelli finti; **non misurata con modelli veri**.

**Misure offline.** Bench sintetico invariato (0 errori silenziosi, copertura uguale). 2.206 test passati (stessi 34 problemi di ambiente). Sul testo reale il caso "lid" è astenuto; gli altri link non cambiano.

**Limiti onesti.** (a) I segnali sono strutturali: forme ambigue senza segnale restano esposte; solo il gold set lo misura. (b) La verifica con i modelli non è indipendente dal resto quanto sembra (Kohli: 9 giudici ≈ 2–2,5 voti indipendenti) e non è ancora misurata: serve il run live con `verify` acceso. (c) Il rilevamento non produce da solo i profili dei sensi: i profili si scrivono (o si propongono con i modelli e si rivedono) partendo dall'elenco delle forme esposte.


## 11. Come si lanciano i run di misura (7 ottobre 2026, notte)

Guida completa: `docs/linker_operativo_github_actions.md` nel repo (copia nel Project: `operativo_workflow_actions_2026-10-07.md`).

- **`public-reports`**: i campi `n` e `cap` vogliono il numero nudo (`600`, `30`). Scrivere `n=600` dava `invalid int value: 'n=600'`; ora lo script e il workflow tolgono il prefisso e un valore non numerico ferma il run con un messaggio.
- **`linking-bench`**: `graph` e `verify` sono caselle del modulo, ma compaiono solo se il ramo scelto contiene la modifica. Se `verify` non c'è, il ramo non ha il bundle. L'artifact ora si chiama `linking-bench-results-graph-<…>-verify-<…>` e il riepilogo riporta ramo, commit e opzioni, così i due run (verify spento e acceso) si distinguono.
- Misura prevista: due run con `mode=linker`, `graph` acceso, `verify` spento/acceso. Il confronto dice quanti link a forme a rischio la verifica ferma, e se ne ferma di corretti. Resta non misurato fino ai due run.

## 12. Il run del 7 ottobre (verify spento e acceso) e cosa ha mostrato

**Esito dei due run di `linking-bench`: nessuna misura.** I due file `linking_results.json` sono identici byte per byte. Le righe con i modelli sono 0: Nemotron ha risposto HTTP 429 (limite di frequenza) e il client ha ritentato solo per 7 secondi in tutto (attese di 1, 2 e 4 secondi), poi la sezione con i modelli si è fermata. Gli stadi deterministici (senza modelli) sono uguali ai miei numeri locali: dev_it 157 corretti, 0 errati, 32 astenuti; held-out IT 70/0; EN 62/0; held-out2 23/0. Quindi **la verifica con i modelli non è ancora misurata**.

**Correzioni (bundle `melampo-linker-t2-fix-nparam-20261007-…`, seconda versione).**
- Client dei modelli: con 429 aspetta quanto dice `Retry-After` (fino a 60 s) oppure raddoppia l'attesa (tetto 60 s); 8 tentativi; almeno 0,5 s fra due richieste dello stesso modello, anche da thread diversi.
- Un modello (o il recupero denso) che non risponde non ferma più il run: la riga si astiene con motivo `model_unavailable`, e il riepilogo scrive in grassetto quante righe hanno perso la risposta ("WARNING … lost a model answer"). Un run che perde righe lo dice in cima.

**Il testo reale: 600 menzioni MultiCaRe (inglese, case report), lette a mano.** Con `cap=30` il campione ha 192 forme diverse. Il linker deterministico accetta 356 su 600 dal solo nome (stadi lessico 283, tabella parti 73) e ne astiene 244 (aggettivo nudo 95, codice di livello senza prove 85, parte di struttura pari senza lato 19…). Nessun gold set: sono le mie letture.
- Fra i 71 link accettati su forme che i segnali strutturali marcano, il caso più chiaro è questo: **"thyroid function / thyroid-stimulating hormone / thyroid hormone tests" → ghiandola tiroide, 16 su 24 menzioni di thyroid**.
- Il difetto più grosso stava fuori dai segnali strutturali: "heart" (12 "heart rate", 3 "heart sounds", 1 "fetal heart tones"), "liver" (10 "liver function", 3 "liver enzymes", 1 "profiles"), "axis" letto come seconda vertebra in "short axis view", "normal axis" (ECG), "Axis 90" (oculistica), "12:00 axis of the breast" (4 su 4 controllati). In tutto circa 57 link su 356 (16%). Il 16% vale per questo registro (case report con analisi di laboratorio e segni vitali), non per i referti di radiologia, dove "heart rate" è raro; ma il linker è pensato anche per la memoria clinica.
- I segnali strutturali del passo 10 marcavano il 20% dei link accettati ma non vedevano questi: la forma "heart" non ha niente di strano, è il **contesto a destra** che la cambia.

**Regola di costruzione, valida per tutte le strutture** (`attribute_heads` in `word_senses.json`). Se la parola subito dopo la menzione (o subito prima: "anti-") è un nome di misura o di esame (funzione, frequenza, toni, enzimi, profilo, pannello, ormone, anticorpo, "stimulating", "transcription", saggio, marcatore; fase, contrasto, "lymph"; e le forme italiane), la struttura è solo il modificatore e il linker si astiene (`attribute_head_names_a_measurement:<parola>`). Non è una voce per parola anatomica: è una proprietà del vicino, quindi vale per ogni struttura presente e futura. Non contiene parole di malattia o di procedura sulla struttura ("liver biopsy", "heart failure", "brain MRI" restano il fegato, il cuore, l'encefalo).
Inoltre "axis" entra nell'inventario dei sensi (vertebra contro geometria): una voce di dati.
- **Effetto misurato:** sui 600 la regola del vicino e l'inventario cambiano 57 link (48 il vicino, 9 "axis") e a mano **tutti e 57 sono astensioni giuste** (non erano strutture; i link accettati scendono da 356 a 299); su PARROT e IU X-ray 0 cambi; bench sintetico identico (0 errori silenziosi). Rimane accettato per errore, che io veda: "Thyroid, parathyroid, and vitamin D assay" (elenco di esami), "hip" come osso o articolazione, alcuni casi di vena porta.
- **Limiti:** (a) l'elenco dei vicini è scritto a mano e va allargato con il gold set; (b) "thyroid" in "thyroid cancer" resta la tiroide (decisione di etichettatura da confermare coi radiologi); (c) le 600 letture sono mie, non di due radiologi.

**Proposta di regola per i radiologi (da confermare):** una menzione che modifica una misura o un esame (frequenza cardiaca, funzione epatica, ormone tiroideo) si etichetta `NOT_ANATOMY`. Il protocollo del gold set lo riporta.

**Misurare la verifica sul testo reale: `verify-probe`.** Nuovo workflow e script `scripts/verify_probe.py`: legge i fogli di un run `public-reports` (id del run), collega ogni menzione con gli stadi deterministici e, per i link dal solo nome, chiede ai due modelli se il testo marcato nomina quella struttura in quella frase. Con `scope=all` chiede per ogni link dal solo nome (non solo per le forme marcate), che è ciò che serve dopo aver visto i difetti fuori dai segnali. Esce l'elenco dei link fermati (da leggere: era giusto fermarli?) e un campione di quelli confermati (c'è ancora qualcosa di sbagliato?). Provato con modelli finti sui 600 reali: gira da capo a fondo.

## 13. Il secondo run live, e il contesto come tipo, quadro e area dell'esame (7 ottobre 2026, notte)

**Secondo run di `linking-bench` (graph acceso, con i modelli veri).** I modelli hanno risposto: 0 righe perse per il limite di frequenza, quindi la correzione ha funzionato.

| Set | verify spenta: corretti / altro concetto / astenuti | verify accesa |
|---|---|---|
| dev_it (189) | 158 / 0 / 31 | 153 / 0 / 36 |
| heldout_it (87) | 70 / 6 / 11 | 69 / 7 / 11 |
| heldout_en, heldout2_it, heldout2_en | invariati | invariati |

Errori silenziosi: 0 in entrambi. La verifica ha fermato 6 link, **tutti corretti**: "anca sinistra/destra/sn/dx" (Nemotron dice NO, Gemma SÌ: l'anca è una regione, non l'osso della classe, vedi sotto), "LIS" (stesso disaccordo) e "Vena splenica" (NO da entrambi, la classe si chiama "portal vein and splenic vein"). Costo: 3% di copertura su dev_it. Beneficio sul bench sintetico: nessuno, perché non ci sono errori da evitare. Il beneficio, se c'è, sta nel testo reale: lo misura `verify-probe`. Il 7° caso che cambia fra i due run ("coste di destra" → altro concetto, stadio di traduzione) non passa dalla verifica: è la variabilità dei modelli fra due run (i due modelli hanno tradotto "right ribs"/"ribs right" in un run e non nell'altro), un motivo in più per non trattare un singolo run come misura.

**La domanda: il contesto è il tipo, il quadro e l'area dell'esame?** Sì, e le fonti lo confermano. Tre livelli, dal più alto al più locale:
1. **Tipo dell'esame.** I referti radiologici dichiarano il nome o tipo di esame e le sezioni: ACR (Practice Parameter for Communication of Diagnostic Imaging Findings, "Components of the Report": nome o tipo di esame, informazioni cliniche, corpo del referto con tecnica, reperti, limiti, confronti, impressione) e RSNA RadReport (storia clinica, tecnica, confronto, osservazioni organizzate per organo, impressione). Il linker lo legge già (`report_state.py`: titolo, sezioni, modalità dell'intestazione).
2. **Quadro.** I documenti clinici si dividono in sezioni con significati diversi: LOINC/HL7 distingue i dati di laboratorio (30954-2), i segni vitali (8716-3), il reperto obiettivo (29545-1), l'anamnesi della malattia attuale (10164-2), le procedure (29554-3). SecTag (Vanderbilt) sostiene che la sezione in cui compare un concetto ne cambia il significato (una malattia in anamnesi personale o familiare). "Thyroid, parathyroid, and vitamin D assay" è un elenco di esami: nel quadro "risultati di laboratorio" la tiroide è il modificatore di una misura, non un reperto.
3. **Area.** Il distretto dell'esame (rachide, torace, collo): oggi solo il rachide. Serve per le forme come "axis" (vertebra nel rachide, vista o asse in ecocardiografia).

**Costruito ora: il quadro della frase** (`exam_frame.py`, dati in `data/linking/exam_frames.json`). Per ogni frase si stima se è scritta nel quadro delle immagini, dei risultati di laboratorio o dei segni vitali, da parole e schemi di unità del quadro (nessun nome anatomico: un test lo verifica). Un quadro di misura vale con almeno 4 punti (due indizi forti, o uno forte e uno schema) e nessun indizio di immagini nella stessa frase: una parola sola come "laboratory" o "bradycardia" non fa un quadro. In un quadro di misura il linker si astiene (`frame_is_a_measurement:laboratory|vital_signs`). Il **tipo** entra così: se il referto ha un'intestazione che nomina un esame di immagini e la frase è nei reperti o nelle conclusioni, il quadro di misura non si applica (i reperti parlano delle immagini); nel quesito clinico e nella tecnica sì.

**Il vicino immediato resta, ma è un indizio del quadro, non "il contesto".** "Heart rate", "liver function", "anti-thyroid" sono la stessa idea a distanza di una parola: il testo sta parlando di una misura. Il quadro copre i casi che il vicino non vede (elenchi di esami, "assay" in fondo alla frase); il vicino copre le frasi brevi senza altri indizi ("Her heart rate was high"). Entrambi sono dati sul testo, non sulla struttura.

**Misura (600 MultiCaRe, 229 PARROT, 676 IU X-ray, nessun gold set).** Il quadro cambia 2 link sui 600 e 0 sugli altri due corpora; bench sintetico identico. I due: "Thyroid, parathyroid, and vitamin D assay" (giusto) e "liver injury" in una frase che comincia con "Laboratory data revealed leukocytosis and…" (la struttura è davvero nominata: è una perdita di copertura accettata). Con la soglia di 2 punti il quadro cambiava 9 link (8 su MultiCaRe, 1 su PARROT) e 7 erano strutture davvero nominate ("shortness of breath" letto come segno vitale, "laboratory" da sola, "aortica sottorenale"): la soglia di 4 l'ha decisa la lettura dei casi. In totale, dopo il vicino, "axis" e il quadro, i link accettati sui 600 passano da 356 a 297, e ho letto a mano tutte le variazioni.

**Altri conflitti, letti sulla documentazione.**
- **Anca (hip).** UBERON, citando FMA, definisce "hip" come la regione ("hip region", FMA:24964); "hip bone" e "hip joint" sono altre voci. La classe TotalSegmentator è l'osso dell'anca. Il linker accetta "left hip" / "anca sinistra" come equivalenti della classe, perché la chiave toglie la parola "bone". Nemotron dice NO a "anca sinistra" proprio per questo. Proposta: la relazione è `approx` (la regione che contiene l'osso, come "emibacino") e non `equal`, e un'articolazione ("hip replacement", "coxartrosi") non è l'osso. Non l'ho cambiata: cambia l'esito del bench sintetico (4 righe dev_it) e va decisa insieme ai radiologi.
- **Cancro e malattia dell'organo** ("bladder cancer", "heart failure"): la struttura è nominata e va collegata con la polarità/patologia come attributo; è la regola del protocollo, non un conflitto del linker.
- **Elenchi di esami senza parole del quadro** restano accettati (non ne ho trovati nei 600 oltre a quello risolto).

**Limiti.** (a) Le parole e gli schemi del quadro sono scelti da me e provati sui tre corpora; non sono un gold set. (b) Il quadro è per frase: una frase che mescola immagini e laboratorio non è vetata (prudente per la copertura, non per la precisione); un'intestazione di laboratorio su più frasi non è ancora letta come sezione. (c) Il distretto dell'esame fuori dal rachide non è ancora stato costruito. (d) "Fonti ACR": ho letto le sezioni della Practice Parameter dalla pagina pubblica; non ho potuto leggere il testo completo di ogni sezione.

Fonti: ACR Practice Parameter for Communication of Diagnostic Imaging Findings (https://gravitas.acr.org/PPTS/GetDocumentView?docId=74); RSNA RadReport (https://reportingwiki.rsna.org/images/c/ce/ReportingChairOrientationSR.pdf); HL7 FHIR valueset doc-section-codes (https://www.hl7.org/fhir/valueset-doc-section-codes.html); SecTag (https://www.vumc.org/cpm/sectag-tagging-clinical-note-section-headers); UBERON basic v2025-05-28.

## 14. `verify-probe` sui 600 MultiCaRe reali (7 ottobre 2026, notte)

Run con `scope=all`, branch con il quadro dell'esame: 600 elementi, 297 link accettati dal solo nome, 297 chiesti ai due modelli, **0 righe perse** (nessun 429). Esito: 285 confermati, 12 fermati (9 con un NO, 3 "incerti"). Ho letto a mano tutti i 12 e 95 dei 285 confermati (nessun gold set: sono le mie letture).

**I 12 stop.**
- *Giusti e chiari (4):* "heart teams" (squadre), "Cranio"-facial e "cranio"-caudally (forme di composizione, non il cranio), "liver pancreas antigen [SLA/LP]" (un antigene, non il pancreas).
- *Difendibili (4):* "L1-level paraparesis" (livello neurologico, non la vertebra), "bladder reflex of the uterus peritoneum" (riflessione peritoneale), "bladder irritation symptoms", "right iliac bone" → osso dell'anca (in inglese l'iliaco è l'ileo, in italiano è l'osso coxale).
- *Sbagliati (4):* "aortic valve" e "aorta" (Gemma ha risposto vuoto e "WORD"), "sigma resection" (Nemotron NO) e "RML" nel "ritorno della vena polmonare" (Nemotron NO). Tre degli stop su 12 vengono da risposte malformate di Gemma, non da un giudizio.

**Dei 95 confermati letti:** 1 errore chiaro rimasto ("renal and liver parameters", misura di laboratorio) e 2 dubbi ("left hip pain", "portal venous hypertension"). Stima ruvida dell'errore residuo sui confermati: 1–3%, da 95 casi letti da me; non è una misura certificabile.

**Quanto vale la verifica.** Sui 600 ha evitato 4 errori chiari (1,3% dei link chiesti) al prezzo di 2–4 link giusti persi (0,7–1,3%). Nessuno dei 4 era già preso dalle regole di contesto. Il tasso d'errore sul testo reale è sceso da circa 16% (356 link accettati con ~57 non-strutture) a circa 1–3%, ma il merito va soprattutto alle regole di contesto (vicino, quadro, sensi di "axis"), non ai modelli. Il costo della verifica sul bench sintetico (3% di copertura, nessun errore evitato) resta.

**Correzioni fatte dopo il probe (stesso bundle).**
1. *Lettore delle risposte.* Il verdetto era la prima parola della risposta: "Word: YES" diventava "WORD" e valeva incerto. Ora è la parola YES/NO/UNSURE che la risposta usa; nessuna o due diverse = incerto. Una risposta vuota è una chiamata fallita: il client ritenta (2 volte) invece di contarla come voto.
2. *Forma di composizione.* Una parola unita alla successiva da un trattino ("cranio-facial", "cranio-caudale", "ileo-cecale") è una forma di composizione, non la struttura: il linker si astiene (`mention_is_the_first_part_of_a_compound`). I codici di livello (C5-6) non sono parole. Effetto: sui 600 toglie "Cranio" e "cranio"; su PARROT toglie 4 link accettati (125 invece di 129) e le 9 menzioni toccate erano tutte composti (acromion-claveare e sacro-iliaco sono le articolazioni, non l'osso; ileo-cecale è la valvola; colon-sigmoidea; corpo-coda; sacro-ileite): nessuna era la struttura. Ripasso della mia lettura precedente di PARROT: ne avevo segnalato uno solo (colon-sigmoidea).
3. *Vicini.* Si aggiungono team, score, parameter, antigen (e le forme italiane): "heart teams" e "liver parameters" si astengono.
Bench sintetico identico (0 errori silenziosi); IU X-ray invariato.

**Ancora aperti.** (a) L'accordo dei due modelli è un controllo severo (Nemotron dice NO più spesso di Gemma: 8 stop su 12 sono NO di Nemotron contro SÌ di Gemma). (b) "left hip" come osso e "hip pain" (regione/articolazione): serve la decisione sulla relazione `approx`. (c) Elenchi di esami senza parole del quadro. (d) Un gold set vero: queste sono 600 menzioni inglesi di case report, non referti di radiologia italiani.

## 15. Ragionamento umano simulato e rigore in parallelo: verifica, idee dalle fonti, T3 (7 ottobre 2026, tarda notte)

Domanda di Frank: il ragionamento umano simulato e il rigore dell'IA lavorano davvero in parallelo e rispettano l'obiettivo? Quali idee recenti (neurobiologia, filosofia della comprensione, strumenti di IA, nuove relazioni) migliorano il modello? Poi avanti con i prossimi passi. Prima, due decisioni approvate: "anca" come `approx` e la regola `NOT_ANATOMY`.

### 15.1 Verifica sul codice e sui 600 MultiCaRe

| Domanda | Risposta (misurata) |
|---|---|
| La lettura è parallela? | **Sì.** `read()` è pura: sensi, quadro, integrazione, codici di livello, lessico e tabella parti leggono la menzione ognuno per conto suo. I due modelli sono chiamati insieme. |
| La decisione è convergente? | **No, prima di oggi.** `_decide` accettava il primo flusso a favore senza veti. Sui 293 link accettati dei 600 MultiCaRe, **tutti** poggiavano su un solo meccanismo (il nome curato). I flussi "umani" lavoravano solo come inibizione (veto), mai come conferma. |
| L'area dell'esame era letta? | Solo per il rachide. Il "tipo" (modalità) e il "quadro" sì; l'**area** no. |
| La sintassi era della lingua giusta? | **No: difetto trovato.** Le teste di misura erano cercate come in inglese (testa a destra: "liver function"). In italiano la testa sta a sinistra: "Funzione del fegato", "Ormoni della tiroide", "Toni del cuore", "Dosaggio ormoni tiroide" erano **accettati**. Corretto (15.3). |
| Un nome accorciato era trattato come il nome? | **Sì: difetto trovato.** "anca" = "osso dell'anca" senza "osso"; "left innominate" = "left innominate bone" senza "bone" (ma può essere la vena o l'arteria anonima); "paravertebrali" = "muscoli paravertebrali". La testa del sintagma dice a che cosa si riferisce; toglierla cambia il referente. Corretto (15.2). |
| C'era una misura di confidenza? | No. Nessuna lettura di secondo ordine ("quanto è sostenuto questo link"), quindi niente da calibrare. |
| L'obiettivo (≤1% sugli accettati, astensione motivata) è rispettato? | Astensione motivata: sì (ogni astensione ha il motivo). Bench sintetico: 0 errori silenziosi. Testo reale: stima 1–3% dalla mia lettura, **non ancora ≤1% e non certificabile senza gold set**. |

### 15.2 Le due decisioni approvate
- **"Anca" / "hip" senza "osso"** è la regione o l'articolazione (UBERON:0001464 *hip* è una regione). La classe TotalSegmentator *hip* è l'osso coxale. Il link va alla classe più vicina con `relation = approx` (tabella parti); "osso dell'anca", "osso iliaco" e "os coxae" restano `equal`. Effetto: le 4 righe "anca" di dev_it passano da `equal` ad `approx`, e "left hip" di MultiCaRe diventa `approx`.
- **Regola generale (non per parola): un nome senza la sua parola-testa non è quel nome.** Il lessico non riconosce più una chiave che esiste solo perché un nome scritto ha perso la testa ("osso", "bone", "muscoli", "gland"…), a meno che l'inventario dei sensi abbia letto la forma nella frase. Nel lessico erano 18 chiavi: le 8 di "anca"/"hip"/"coxale"/"innominate", le 8 dei paravertebrali, 2 "suprarenal".
- **`NOT_ANATOMY`** per la struttura nominata solo come modificatore di una misura: approvata il 7 ottobre come regola del progetto (protocollo aggiornato). I radiologi la applicano dalla sessione di allineamento.

### 15.3 Idee dalle fonti e che cosa ne è stato fatto

| Fonte (anno) | Idea | Che cosa è diventata |
|---|---|---|
| Fillmore, semantica dei frame; SNOMED CT, modello dei concetti osservabili (*inherent location*, *finding site*, *procedure site*) | La stessa parola ha **ruoli** diversi nella scena: sede di un reperto, sede di una procedura, struttura di cui si misura una proprietà | Campo `role`: `procedure_site` ("liver biopsy", "biopsia del fegato", "resezione epatica"); `inherent_location` per le misure ("heart rate", "funzione del fegato"): il link si astiene come prima, ma `about` conserva la struttura misurata. **Nessuna informazione persa.** Sui 600: 52 misure con la struttura conservata, 9 sedi di procedura (tutte giuste alla lettura). |
| Sintassi testa–modificatore (linguistica generale) | La posizione della testa dipende dalla lingua | Teste inglesi a destra, italiane a sinistra, con preposizione in entrambe ("function of the liver", "toni del cuore"). |
| LOINC/RSNA Radiology Playbook (*Region Imaged*, *Imaging Focus*, *Laterality*); Campbell, AJR 2005 (la TC torace copre in media ~7 cm sotto le basi) | L'**area** dell'esame è un attributo standard della procedura | `exam_area`: area letta dal nome dell'esame (titolo, tecnica, prima frase, o la frase stessa: "RM pelvi:", "brain MRI", "CT of the chest, abdomen and pelvis"), mai dalla storia clinica. Struttura dell'area: sostegno. Area vicina: neutra. Area lontana: **errore di previsione** registrato. |
| Codifica predittiva (Nour Eddine *et al.*, Cognition 2024): l'errore di previsione è il segnale per guardare meglio; De Neys, BBS 2023: la deliberazione parte quando le intuizioni sono in conflitto | **Sistema 2 sul conflitto** | Una forma ambigua lontana dall'area va ai modelli anche senza `--verify`; senza modelli ci si astiene (`ambiguous_form_outside_the_exam_area`). Un nome non ambiguo lontano dall'area resta accettato con il conflitto registrato: un reperto incidentale o un dato anamnestico fuori campo è normale radiologia. |
| Fleming, Annual Review of Psychology 2024 (confidenza come lettura di secondo ordine); Ernst & Banks; architettura §3.5 | Separare la decisione dalla **misura di quanto è sostenuta** | `LinkResult.support` / `conflicts` / `convergence`. I meccanismi sono name, sense, frame, area e models (i due modelli contano **uno**). Sui 600: 159 link solo con il nome, 134 con due o più meccanismi. Per ora non decide nulla: è il punteggio da calibrare. |
| Learn-then-Test (Angelopoulos *et al.*, arXiv 2110.01052; MAPIE, controllo della precisione con fixed-sequence testing); Mondrian conformal; Kahneman & Klein 2009 (l'intuizione vale solo dove è stata validata) | Scegliere la soglia di accettazione con una **garanzia** sull'errore tra gli accettati, per strato | `evaluation/selective_calibration.py`: test a sequenza fissa delle soglie di convergenza, dalla più severa, per strato (lingua \| stadio). `gold_set.py evaluate` lo riporta. Con 0 errori servono **299** link per strato per certificare l'1% al 95%. È lo strumento di T4, pronto per il gold set. |
| Chain-of-Verification (Dhuliawala *et al.*, ACL 2024); MedAbstain (EACL 2026: l'opzione esplicita di astensione aumenta l'astensione sicura); Kim *et al.*, ICML 2025 (errori correlati tra LLM) | Domande non suggestive e opzione "non si può dire" | `verify_style=choice`: cinque opzioni bilanciate. Solo l'opzione 1 conferma; "5. the sentence does not settle it" vale incerto. Si confronta con `yes_no` lanciando il `verify-probe` con `style`. Diventa predefinito solo se misura meglio. |
| Mohri & Hashimoto, ICML 2024 (ripiego verso un'affermazione meno specifica); MedPath 2025 | Ripiegare sul padre invece di sbagliare | Già presente come proposta (`fallback`); da certificare con LTT come gli altri link (T4). |

**Idee valutate e non adottate (per ora).** L'entropia semantica (Farquhar *et al.*, Nature 2024) costa più chiamate per link; nel nostro caso il raggruppamento è esatto (stesso ID), quindi è facile da aggiungere dopo il gold set, se la convergenza non basta. Il dibattito tra più agenti non crea indipendenza (errori correlati). Le "sonde" sugli stati nascosti richiedono modelli a pesi aperti in casa.

### 15.4 Misure
- **Bench sintetico** (dev_it, held-out IT/EN): nessuna decisione cambiata, 0 errori silenziosi. Cambia solo la relazione di 4 righe ("anca" → `approx`).
- **MultiCaRe 600:** accettati 293 (invariato rispetto al bundle precedente); area nota per 65 accettati (61 attesa, 4 vicina, 0 lontana: i case report raramente nominano l'esame nella frase); 52 misure con la struttura conservata; 9 sedi di procedura.
- **Suite:** 509 test passati nei file del linker; nel resto, solo i fallimenti di ambiente noti (FalkorDB, pydicom, lab_phenotypes).

### 15.5 Limiti
- La convergenza non decide ancora: senza gold set i pesi e la soglia sarebbero scelte mie. La regola "almeno due meccanismi" della §3.5 escluderebbe oggi il 54% dei link giusti del solo nome. Va decisa con LTT sul gold set, per strato.
- L'area si legge solo dove l'esame è nominato. Nei case report è spesso sconosciuta; nei referti radiologici italiani (titolo "RM PELVI", "TC TORACE") dovrebbe esserci quasi sempre. Va misurato sui referti veri.
- Le parole di regione, le adiacenze e le teste di procedura sono scelte da me su fonti standard; un radiologo deve rivederle.
- Il quadro resta per frase (una sezione "Laboratorio" su più frasi non è ancora letta come sezione).

## 16. `verify-probe` a confronto: `yes_no` contro `choice` (8 ottobre 2026)

Stessi 600 MultiCaRe, `scope=all`, 293 link chiesti ai due modelli, ramo con la correzione del lettore delle risposte. Letture mie, senza gold set.

| | `yes_no` | `choice` (cinque opzioni) |
|---|---|---|
| Confermati | 287 | 275 |
| Fermati | 6 | 18 |
| Stop in comune | 3 (pancreas/autoanticorpi, "right iliac bone", "RML") | 3 |

- **`yes_no`**, 6 stop: "bladder reflex of the uterus peritoneum" (riflessione peritoneale) e "NOSAs … pancreas" (autoanticorpi) sono catture giuste; "right iliac bone" è difendibile (iliaco inglese = ileo); "bladder irritation symptoms" è difendibile; "sigma resection" (sede di procedura, il link al colon è giusto) e "RML" (lobo medio) sono stop probabilmente sbagliati. Con il lettore corretto gli stop passano da 12 a 6 e spariscono quelli dovuti a risposte malformate.
- **`choice`**, 15 stop in più rispetto a `yes_no`: tutti link a mio avviso giusti ("liver edge", "liver ultrasound", "transverse colon", "right distal femur", "spinal cord") e, soprattutto, **sette livelli vertebrali** ("C1/C2 root", "C5-6 level", "L2/3 level"…), per cui Gemma sceglie un'opzione diversa da 1 mentre Nemotron conferma. Le opzioni 2–4 non sono neutre: "una regione accanto alla struttura" attira i livelli e le sedi.
- **Esito:** `choice` costa circa 4% di copertura in più e non cattura nessun errore che `yes_no` non cattura. Resta `yes_no` come predefinito; `choice` rimane come opzione ma non va usata così com'è. Se si riprova, le opzioni vanno riscritte e rimisurate (non basta cambiare formato).
- **Limite:** nessuno dei due è un giudizio sul confermato: non ho riletto un campione dei 287 confermati in questo run.

## 17. Passo 7: il lettore cieco senza modelli (8 ottobre 2026)

**Perché non un LLM.** Gli errori dei modelli sono correlati: ~60% delle volte due LLM sbagliano nello stesso modo (Kim et al., ICML 2025); nove giudici valgono circa due voti indipendenti (Kohli); il consenso non è verifica (Denisov-Blanch). Due LLM che dicono sì contano come un meccanismo solo. Il lettore cieco ha un meccanismo diverso: nessun modello, nessuna rete, nessuna vista sulla prima risposta.

**Come legge** (`src/melampo/memory/blind_reader.py`). Coseno TF-IDF su n-grammi di caratteri (3–5) tra la menzione grezza e tutti i nomi grezzi: nomi del lessico (esclusi i `requires_context`), nome e sinonimi UBERON, nomi della tabella delle parti. Poi:
- **support**: tra i concetti entro 0,03 dal migliore ce n'è uno che è la struttura scelta, un suo nodo UBERON o una sua parte (risalita nel grafo, fino a 6 passi);
- **against**: solo se il migliore ha punteggio ≥ 0,80, nessuno dei vicini è la struttura scelta, tutte le parole della menzione sono spiegate e al vertice ci sono al massimo due strutture;
- **silent**: meno di 4 lettere, codici di livello, qualsiasi cifra, nessun nome abbastanza vicino. Il silenzio non è un voto.

**Nel linker.** `AnatomyLinker(blind=..., blind_veto=False)`: il verdetto va nella traccia (`Evidence("blind", ...)`); `support` conta come meccanismo (`blind`), `against` come conflitto registrato (`blind_reader_disagrees`). Con `blind_veto=True` un `against` fa astenere con motivo `blind_reader_disagrees:<nome letto>`. Il predefinito è **senza veto**: finché non c'è il gold set non so quanti buoni link fermerebbe.

**Misure** (`scripts/blind_reader_check.py`, nostre menzioni, nessun gold set):
| Prova | support | silent | against |
|---|---|---|---|
| contro la classe giusta (373 menzioni) | 268 | 105 | **0** |
| contro un'altra classe a caso | 0 | 129 | 244 |
| contro l'altro lato | 113 | 36 | 0 |
| 600 MultiCaRe, 293 link accettati | 257 | 36 | 0 |
Bench sintetico: con e senza veto i risultati sono identici (nessun falso allarme sui link del lessico).

**Limiti.**
- Il lato non lo vede: sull'altro lato dà support (113 su 149), perché cerca la struttura, non il lato. Lato, livello e area restano ad altri meccanismi.
- È debole sull'italiano (UBERON è inglese): circa metà silenzio. Non è una prova di correttezza, solo un secondo parere indipendente quando parla.
- I numeri sono su menzioni scritte da noi: dicono che il meccanismo funziona, non quanti errori veri cattura. Serve il gold set.

**Set liberi in inglese come controllo esterno.** Esistono etichette umane gratuite, ma di popolazioni diverse dai referti: CRAFT (ID UBERON, manuali, CC BY 3.0, articoli completi di genetica del topo), MedMentions (UMLS, abstract PubMed), AnatEM (anatomia, abstract), RadGraph (referti radiologici annotati da radiologi, accesso PhysioNet con credenziali). Servono per controllare lessico e lettore cieco su etichette non mie (precisione dei link, falsi allarmi del lettore), non per certificare: la certificazione richiede un campione nuovo, non visto, etichettato da radiologi sulla popolazione d'uso.


## 18. I nove stadi completati come meccanismi, e il primo controllo su etichette non nostre (8 ottobre 2026)

Richiesta di Frank: completare i nove stadi per arrivare oltre il 99%. Ora ogni stadio ha il suo meccanismo nel codice e la sua misura. Il 99% resta **non dimostrato** sui referti: lo dimostra solo il gold set dei radiologi (stadio 9). Su testo etichettato da altri, il linker è al 99,0% su CRAFT e al 96% su MedMentions secondo le nostre regole (dettagli sotto).

### 18.1 Che cosa è stato aggiunto, stadio per stadio

| Stadio | Aggiunto oggi | Effetto misurato |
|---|---|---|
| 1. Stato del referto | **Lato dell'esame** ("RM ginocchio destro"), letto come l'area. Stesso lato: sostegno. Lato opposto senza parole di confronto: astensione (`side_contradicts_the_exam`). Struttura pari senza lato: il lato dell'esame è solo una **proposta** (`inherit_exam_side` spento finché i radiologi non decidono). **Discorso**: la struttura nominata in un'altra frase dello stesso referto sostiene il link. | Bench sintetico identico. Testo reale: discorso a favore di 124 link su 291 (MultiCaRe), 40 su 471 (IU X-ray). Il lato dell'esame scatta di rado: nei corpora pubblici l'esame raramente dichiara il lato. |
| 3. Candidati morfologici | Radici greco-latine (`morphology_roots.json`, 22 radici di una sola struttura; escluse le ambigue come cist- e col-). Su un link accettato contano come meccanismo; su un'astensione vanno tra le proposte. Non accettano mai. | Proposte su 125 astensioni di MultiCaRe e 73 di IU X-ray; sostegno a 29 e 55 link. |
| 4. Significati e integrazione | Nome proprio (istituzioni, studi, scale); nome di un'altra cosa fatto con l'organo (molecola, canale, dispositivo, modello, enzima in "-asi", "anti-"); parola dentro un nome anatomico più lungo ("blood-brain barrier"); nome solo italiano in frase inglese ("sigma"); sigla fuori da un testo radiologico (`document_type`); la lingua da sola non decide un senso; "atlas" con i suoi sensi; misura condivisa da due organi coordinati ("liver and renal function"). **Correzione generale:** i vicini si leggono all'occorrenza giusta e a parola intera ("thyroid" non dentro "Hypothyroidism"). | Bench sintetico identico; 1.695 menzioni reali di 4 corpora: 2 decisioni cambiate, entrambe giuste ("brain arteries", "sigma resection"). |
| 5. Vicini | Per i link del lessico i vicini sono coperti da quattro controlli: nomi condivisi da più classi, lato (lettore cieco ed esame), nome più lungo nel grafo, parte o tipo nel grafo. | Nessun errore di struttura sorella o di lato nei link accettati su CRAFT e MedMentions. |
| 7. Lettore cieco | Legge anche il lato dalle parole della menzione. Non sostiene una lettura che aggiunge una parola. | Su testo esterno non dissente mai e sostiene anche gli errori: legge solo il nome (§18.3). |
| 8. Monitor di conflitto | `conflict_policy`: `record` (predefinito) scrive il conflitto, `review` manda in coda i link con un flusso contrario e pochi sostegni. | Su testo esterno e reale i conflitti sono quasi assenti: oggi la politica non cambierebbe nulla. |
| 9. Certificazione | `sample --exclude` (referti mai visti) e `--split` (calibrazione/test); `evaluate --split test`. Pacchetto inglese per i radiologi (518 + 1.150 menzioni). Workflow `external-check`. | Vedi `docs/gold_set_protocollo.md`: il test del pacchetto certifica l'inglese solo con 0 errori. |

### 18.2 Controllo esterno: CRAFT e MedMentions

Due corpora liberi con etichette umane, scritti per altri progetti: **CRAFT v5.1.0** (97 articoli di genetica del topo, concetti UBERON annotati a mano, CC BY 3.0) e **MedMentions** (4.392 abstract PubMed, concetti UMLS, CC0). Le menzioni si propongono come per le schede del gold set e il linker è quello deterministico (`document_type="literature"`). Si confronta ogni link accettato con l'etichetta che lo copre.

| | CRAFT | MedMentions |
|---|---|---|
| Menzioni proposte / link accettati | 2.715 / 1.354 | 4.138 / 1.884 |
| Giudicati | 1.352 | 768 (regole del progetto) |
| Errori | 13 (10 "right middle lobe" del topo, 3 granularità) | 51 secondo lo script, **30 alla mia lettura** |
| Precisione | **99,0%** (limite superiore dell'errore al 95%: 1,5%) | **96,1%** alla lettura (93,4% secondo lo script) |

MedMentions etichetta il concetto più lungo. Per le nostre regole molte sue etichette non sono errori: 877 link stanno dentro una malattia o una procedura ("prostate cancer", che per noi nomina la prostata). Ci sono poi 27 cellule dell'organo ("liver cells"), 13 misure di immagine ("brain volume"), 11 sedi di dispositivo ("IVC filter") e 18 soggetti di malattia. Lo script li conta a parte e li elenca, così chi non è d'accordo con la regola può ricontarli.

**Errori trovati e corretti con regole generali** (contati dallo script: su MedMentions da 221 a 51 nel corso della giornata, su CRAFT da 23 a 13): nomi propri, molecole e dispositivi, vicini letti all'occorrenza sbagliata o dentro un'altra parola, misure coordinate, "atlas" e "axis" fuori dalla colonna, la lingua che da sola faceva un senso.

**Errori rimasti (30), da leggere in `external_check.md`:**
- teste composte a due o tre parole ("prostate symptom score", "liver fatty acid binding protein", "spinal cord K(+) channel");
- processi dell'organo ("heart development", "brain connectivity"): convenzione da decidere con i radiologi;
- "heart" del formaggio;
- "DENS" come sigla di una scala.

### 18.3 Che cosa dicono questi numeri per gli stadi 8 e 9

- Gli errori rimasti hanno **lo stesso profilo di convergenza** dei link giusti. Su CRAFT gli errori sono a 2 e a 4 meccanismi, i giusti soprattutto a 3. Il lettore cieco sostiene 50 dei 51 errori di MedMentions e nessun conflitto scatta. Una soglia sulla convergenza non li separa: il test Learn-then-Test non certifica nessuna soglia.
- Sono errori di **contesto** (a che cosa si riferisce la parola nella frase), non di nome. Li vede solo chi legge la frase: i due modelli (`verify`) o il radiologo. La prossima misura è quindi `external-check` con `verify` acceso. Dirà quanti di questi errori i modelli fermano, e a che prezzo in link giusti.
- Per i referti radiologici il dato che conta resta il gold set: articoli e abstract hanno molte più molecole, istituzioni e processi dei referti.

### 18.4 Limiti

- CRAFT e MedMentions sono letteratura in inglese, non referti in italiano.
- La classificazione "convenzione" contro "errore" su MedMentions segue le nostre regole, che i radiologi non hanno ancora confermato.
- Le liste di parole (teste, nomi propri, radici) sono dati scelti da me su questi corpora: vanno riviste con il gold set, che dirà anche cosa costano in copertura sui referti.

## 19. Gli errori del controllo esterno, uno per uno, e le correzioni (8 ottobre 2026, sera)

Richiesta di Frank: analizzare nel dettaglio gli errori rimasti (CRAFT 13, MedMentions 51) e correggerli con regole generali, senza casi particolari. L'elenco completo dei 64 casi, con causa, è in `docs/linker/analisi_errori_controllo_esterno_2026-10-08.md`.

### 19.1 Che cosa erano

- **CRAFT (13).** 10 sono differenze di granularità dell'oro (7 "right middle lobe" contro "anatomical lobe", 3 "bladder" contro "bladder organ"): il linker è più specifico del testo. 3 sono "aortic arch" in testi di sviluppo, dove UBERON usa lo stesso nome per l'arco embrionale e per quello adulto.
- **MedMentions (51).** 16 nomi composti (molecole, scale, assi, abbreviazioni), 16 processi o funzioni dell'organo ("heart development"), 4 materiale d'innesto, 3 formaggio, 4 granularità, 4 rumore dell'oro, 2 mappatura UMLS→UBERON dell'oro, 2 misure o campi.

### 19.2 Che cosa è stato aggiunto

| Meccanismo | Che cosa legge | Veto |
|---|---|---|
| Trattino (`compounds.hyphen_compound`) | la menzione è unita da un trattino a un'altra parola, e il sintagma che segue ha per testa una cosa che non è una sede ("gut-brain **axis**", "pro-brain natriuretic **peptide**") | `mention_is_joined_by_a_hyphen_to:<parola>:<testa>` |
| Sigla definita (`compounds.defined_abbreviation`) | il testo scrive "forma lunga (SIGLA)" (algoritmo di Schwartz e Hearst 2003) e la menzione è parte della forma lunga, che ha per testa una cosa che non è una sede ("liver fatty acid binding **protein** (L-FABP)") | `mention_is_a_word_of_the_name_defined_as:<SIGLA>` |
| La menzione è la sigla (`defined_short_form`) | "(DENS) **scale**": la sigla ha il significato che il testo le dà | `abbreviation_defined_in_the_text_as:<forma lunga>` |
| Materiale d'innesto (`word_senses.material_qualifier`) | un qualificatore d'origine subito prima ("autologous", "homologous", "allogeneic", "donor-specific") e la menzione chiude il sintagma ("autologous costal cartilage as graft") | `tissue_taken_as_graft_material:<qualificatore>` |
| Sinonimi di UBERON come nomi più lunghi | "aortic arch artery" è un nome della *pharyngeal arch artery*: la menzione è dentro un'altra struttura. Il nome è letto come sequenza ordinata di parole, con una testa propria diversa da quella della menzione ("embryonic brain" resta un cervello) e non è solo un luogo ("heart region") | `mention_is_inside_a_longer_name:<nome>` |
| Cornice di sviluppo (`development`) | un nome condiviso fra una struttura in sviluppo e una adulta (66 nomi, letti dall'ontologia: "aortic arch", "phallus", "mesenteron") in un testo con parole di stadio (embryo, E12.5, fetal). **È una prova registrata, non un veto.** | conflitto `name_shared_with_a_developing_structure_in_a_developmental_text` nel profilo |

La **testa del sintagma** è la regola comune: `word_senses.noun_phrase_after` prende le parole a destra della menzione fino a una parola di arresto (preposizione, congiunzione, verbo della classe chiusa, punteggiatura) e la testa è l'ultima, tolte le nominalizzazioni ("expression", "levels"). Le teste che non sono una sede sono le teste già note (`attribute_heads.after`) più `phrase_final` in `data/linking/word_senses.json`.

### 19.3 Una prima versione sbagliata, e perché

La prima versione non guardava la testa: si fermava a ogni trattino, a ogni sigla definita e a ogni sinonimo. Misurata riga per riga contro la base, fermava 15 errori ma anche **49 link giusti** e cambiava 106 casi non giudicati: "congenital heart disease (CHD)", "Mouse Brain Library (MBL)", "neck–liver", "fetal-liver-derived macrophages", "embryonic brain". La causa era una sola: il trattino o la sigla da soli non dicono se il sintagma nomina una sede. Lo dice la testa ("axis", "protein", "scale" sì; "disease", "library", "macrophages" no). Con la testa le perdite sono scese da 49 a **1** link giusto ("islet-to-pancreas volume ratios", una misura) e 1 non giudicato ("hypothalamus-pituitary-gonadal-liver axis").

### 19.4 Misura prima e dopo (stessi corpora, stessi commit fissati)

| | CRAFT | MedMentions |
|---|---|---|
| Errori prima → dopo | 13 → 11 | 51 → 38 |
| Giudicati | 1.352 → 1.349 | 768 → 755 |
| Precisione (regola del progetto) | 99,04% → 99,18% | 93,36% → 94,97% |
| Link giusti persi | 1 | 0 (1 non giudicato) |

Bench sintetico: identico (108 valori su 108). Test del linker: 628 passano.

### 19.5 Che cosa non ha funzionato (e va detto)

- **Dominio del documento come veto.** Nei documenti di sviluppo di CRAFT gli annotatori hanno collegato "aortic arch" all'arco aortico adulto 25 volte su 28 ("the definitive aortic arch", "left-sided aortic arch" di un embrione). Un veto sulla cornice avrebbe perso 25 link giusti per evitarne 3 sbagliati. Per questo la cornice è una prova registrata. Ciò che distingue i 3 errori è locale ("fourth aortic arch artery", "aortic arch arteries") e lo legge il sinonimo di UBERON.
- **Firma del senso dalle definizioni dell'ontologia** (confronto probabilistico e sovrapposizione di parole distintive): su 385 link accettati con un nome rivale dava come "insetto" molti testi di cardiologia umana. I rivali reali nei due corpora sono 5 coppie di nomi, 3 di specie irrilevanti (cuore dorsale di insetto, faringe di nematode). `uberon-basic` non porta il taxon. Scartato.
- **Il solo parser non basta**: la menzione non è testa del suo sintagma nel 56% degli errori e nel 36% dei link giusti. La testa che compare più spesso fra i link giusti è "development" (26 volte in CRAFT, dove l'annotatore etichetta l'organo in "heart development"); in MedMentions la stessa costruzione è un errore. La convenzione cambia da un oro all'altro (§19.7).

### 19.6 Esperimento F1: la testa del sintagma con parser e attention

`scripts/head_probe.py` e il workflow `head-probe` misurano tre modi di trovare la testa (scansione a parole di arresto, parser a dipendenze spaCy, attention di un encoder biomedico) e tre segnali di che cosa sia (lista, spostamento di significato della menzione fra sola e in frase, somiglianza della testa a prototipi di sede o non-sede). Riporta per ogni segnale l'AUC (errore contro giusto) e quanti errori ferma per quanti link giusti perde. Include i set di controllo inglese e italiano del progetto. Non decide nulla. Si lancia a mano (§ guida operativa).

### 19.7 Aperto: F6 e F7

- **F6, processo o funzione dell'organo.** Risolto il 9 ottobre 2026 (§20).
- **F7, granularità.** "brain parenchyma" e "hepatic parenchyma" portano all'organo per convenzione del progetto; "bladder" è la vescica urinaria e "right middle lobe" il lobo del polmone destro nei referti. "large bowel" porta a `colon` per scelta del lessico: va confermata o marcata come approssimata. Nessuna modifica al codice.

## 20. F6: processo e funzione della struttura (9 ottobre 2026)

**Domanda di Frank:** scegliere dalla documentazione clinica o con una soluzione generale tra astenersi e collegare con `subject_of_process`.

**Risposta.** Né l'una né l'altra come erano state poste. Il progetto aveva già il meccanismo giusto per le misure: il link non si fa e la struttura è registrata come sede intrinseca (`role = inherent_location`, `about`). Lo stesso vale per un processo o una capacità: la struttura è il portatore (SNOMED CT *Inherent location* 718497002 e *Inheres in* 704319004; BFO/GO: il processo è un occurrent con partecipante la struttura), non è il processo. Un'etichetta nuova (`subject_of_process`) sarebbe stata un secondo canale con lo stesso significato.

**Come funziona** (`SenseInventory.process_head`, dati in `process_heads` di `word_senses.json`, stesso codice per ogni struttura e per le due lingue):
1. il nome di processo è l'intero sintagma a destra, al massimo due parole, tolte le nominalizzazioni (`_TAIL`: level, expression, pattern(s), process(es), placement…): "heart development", "brain state dynamics", "brain connectivity patterns". Una parola estranea in mezzo ("brain tumour growth") non decide: il processo è del tumore;
2. oppure il nome sta a sinistra con una preposizione e la struttura chiude il sintagma: "development of the pancreas", "sviluppo del cuore". "development of left lower lobe airspace disease" non è fermato: lo sviluppo è della malattia.

Veto `process_head_names_a_process:<testa>`, registrato come sede intrinseca. Le teste vengono da Gene Ontology (15, frequenza dei nomi di processo che cominciano con un nome anatomico) più 18 scelte dall'autore (funzione e sistema funzionale: circuits, connectivity, dynamics, arousal, state, plasticity, interaction, research, regulation, control, contractility, motility…); 18 teste italiane. La provenienza di ogni gruppo è in chiaro nel file. Escluse le parole che in un referto sono reperti o procedure (remodeling, repair, enhancement, uptake, "formazione", "controllo", "ricerca").

**Misura.** CRAFT 11 errori su 1.326 (99,17%), nessun errore in meno e 23 link giusti non fatti, tutti registrati con la struttura giusta (91 registrati, 87 uguali all'etichetta, 4 senza etichetta, 0 sbagliati). MedMentions 38 → 25 errori (94,97% → 96,60%), 13 errori fermati, 7 link giusti non fatti. Bench interno identico. Sui 3.969 casi dei referti pubblici (iu-xray, iu-xray 2, e3c-it) nessuna decisione cambia. **Limite:** le 18 teste aggiunte dall'autore sono scelte guardando MedMentions, quindi quella misura è ottimistica; il confronto non truccato è CRAFT (teste solo da GO) e i referti.

**Nuova misura nel controllo esterno.** `external_check.py` aggiunge `recorded_as_inherent_location`: per ogni link non fatto ma registrato confronta l'etichetta del corpus con la struttura registrata.

**`verify`** (due modelli per frase) sui due corpora: CRAFT 0 errori fermati su 11 (7 giusti fermati), MedMentions 2 su 38 (2 giusti fermati). Gli errori rimasti li vede solo un lettore con il contesto di tutta la frase e della convenzione dell'oro, o un radiologo.

## 21. F1: nomi lunghi tipizzati da ontologia, e fallimento dei flussi su MedMentions (9 ottobre 2026)

**Nome lungo.** `LongerNames` (dati in `data/linking/longer_names.json`, costruiti da `scripts/build_longer_names.py` da NCIt e, se fornita, Protein Ontology) elenca nomi di 2–9 parole con il loro tipo (assessment_tool, chemical, protein, gene). Il linker pone il veto `mention_is_inside_the_name_of_another_thing:<tipo>:<nome>` quando la menzione è in una finestra che coincide con un nome elencato; vince il nome più lungo. Dal 9 ottobre il file ha 6.705 nomi (anche Protein Ontology 73.1); un nome vale solo se scritto in continuo (nessun `;`, `:`, a capo, né virgola o punto seguiti da spazio tra le sue parole) e se fuori dalla menzione ha una parola vera (tre lettere o più, non un numero), così "rib 1" resta una costola e "2, brain; 3" resta un elenco. I tipi vengono dall'ontologia (antenati di classe), non da elenchi di parole. Esclusi domande e risposte di questionario, codici CDISC e nomi dove la parola anatomica è solo un qualificatore.

**Perché i flussi non vedono l'errore.** Su MedMentions gli errori hanno lo stesso profilo dei link giusti (15 su 22: nome, cieco, discorso; nessun conflitto): ogni flusso legge la parola, nessuno il sintagma. Per questo il rimedio è un nuovo flusso che legge il sintagma contro un'ontologia, non una soglia diversa. Dettagli, strumenti usati e limiti: `analisi_fallimenti_interpretativi_medmentions_2026-10-09.md`.

**CI.** Il job `falkordb-service` di `ci.yml` avvia un server FalkorDB (`falkordb/falkordb-server:v4.20.7`) come service container e lancia i test di grafo su TCP reale (`MELAMPO_FALKORDB_SERVICE=localhost:6379`).

## 22. Flussi del sintagma a tre livelli (9 ottobre 2026)

Gli errori rimasti non sembrano incerti (un cancello su conflitto o convergenza si apre su 10 errori su 33), quindi i flussi che leggono il sintagma non possono stare dietro a un cancello di incertezza del link. Livello 1, sempre: consultazioni deterministiche (nomi lunghi tipizzati, teste di processo, "X of <oggetto>" con un senso per discorso, ipotesi di refuso). Livello 2, quando la menzione sta dentro un sintagma più lungo: riconoscitore di intervalli con tipo (GLiNER-BioMed) e recupero del sintagma su NCIt (SapBERT). Livello 3, solo quando i flussi dei livelli 1–2 non concordano sul tipo del sintagma: LLM che segmenta la frase. Nessun flusso del sintagma aggiunge link; ferma o assegna il ruolo. Misura ed esperimento: `analisi_fallimenti_interpretativi_medmentions_2026-10-09.md` §9, workflow `phrase-probe`.

## 23. Dal voto dei flussi al reticolo di blocchi (9 ottobre 2026)

I flussi leggono la parola e votano sul link; il cervello costruisce prima l'unità (blocco) e poi decide che cosa farne. Il disegno nuovo (documento `lettura_del_sintagma_cervello_e_modelli_2026-10-09.md`): un **reticolo di blocchi** a livelli con finestre crescenti; i nodi vengono dalla memoria (nomi di ontologia, entità note), dalla composizione per tipo e da un segmentatore appreso; si sceglie la segmentazione di costo minimo e si impegna il minimo (astensione o ruolo meno impegnativo quando due segmentazioni sono vicine); solo dopo si assegna il link o il ruolo della struttura. Parallelismo limitato alle segmentazioni candidate (Christiansen e Chater 2016). I flussi del §22 (GLiNER, recupero, LLM) diventano fornitori di nodi e di punteggi. Non implementato: tappe A–E nel documento.

## 24. Esito del run `phrase-probe` e ricerca sui limiti (9 ottobre 2026)

I lettori zero-shot (GLiNER-BioMed, SapBERT su NCIt) non separano gli errori dai link giusti (AUC 0,50–0,60); GLiNER tipizza la parola anatomica stessa in 26 errori su 33. Dei 22 errori MedMentions 9 non sono isolamento di sintagma e 13 hanno uno span dell'oro più lungo che NCIt non contiene. Conseguenze per il piano: (1) metrica di prodotto = correttezza di span e ruolo; (2) il reticolo (§23) richiede una memoria più grande di NCIt e un lessico di teste curato; (3) la precisione certificabile viene dall'accettazione selettiva con insiemi calibrati (Learn-then-Test, Clopper-Pearson, ≥ 299 link accettati senza errori per ≤ 1 % al 95 %), non da un modello perfetto; (4) il gold dei radiologi comprende span e ruolo; (5) convenzione bersaglio RadGraph (Anatomia/Osservazione, Located_At). Dettaglio: `lettura_del_sintagma_cervello_e_modelli_2026-10-09.md` §7 e `analisi_fallimenti_interpretativi_medmentions_2026-10-09.md` §9.6.

## 25. Reticolo di blocchi (tappa B), realizzato e spento (9 ottobre 2026)

`src/melampo/memory/chunk_lattice.py` legge il sintagma attorno alla menzione come blocchi (memoria: nomi NCIt e `longer_names.json`; composizione: testa a destra, tipo dalla classe NCIt; cammino di costo minimo; nodo aperto; minimo impegno) e solo dopo decide: collegamento, ruolo (`procedure_site`, `device_site`, `inherent_location`, `source_of`), non-sito, non letto o sottospecificato. È opzionale (`AnatomyLinker(chunk_lattice=...)`), scrive solo nella traccia (flusso `blocks`, `LinkResult.block`) e non decide. Misure e limiti: `lettura_del_sintagma_cervello_e_modelli_2026-10-09.md` §8. In breve: legge 10 errori MedMentions su 22 e 1 su 11 in CRAFT come ruolo o non-sito, 13 su 13 delle misure per immagini, la procedura giusta 68 % delle volte e la misura 83 % sui link con etichetta; sui referti reali i ruoli sono plausibili in circa 6 casi su 10 (letti a mano su 42). Per questo non diventa un decisore finché lessico di teste curato, etichettatura grammaticale e gold set dei radiologi (span + ruolo) non l'hanno corretto.

## 26. Il segmentatore LLM misurato, e dove può stare (9 ottobre 2026)

`phrase-probe` con `llm=sample` (Nemotron 3 Super e Gemma 3 27B; 33 errori + 300 link giusti) legge 10 errori MedMentions su 22, 0 su 11 in CRAFT, segnala 29 % dei link giusti, dà il ruolo giusto 3 volte su 12. Non raggiunge i criteri fissati prima. Decisione di architettura: nessun LLM nel percorso che decide. L'LLM resta un possibile **proponente di blocco** nei casi già incerti, con effetto solo dopo il gold set dei radiologi (span + ruolo) e con prove a coppie minime. Dettaglio: `lettura_del_sintagma_cervello_e_modelli_2026-10-09.md` §8.5.

## 27. Perché l'esperto legge e noi sbagliamo (9 ottobre 2026)

Classificati a mano i 33 errori del controllo esterno: 12 nomi composti presenti in UMLS e non nella nostra memoria, 13 differenze di vocabolario o granularità con l'etichetta, 3 memoria troppo grossolana (arco aortico), 2 che chiedono il documento intero, 3 etichette anomale. La lettura vera pesa per il 42 %. Risposta: prima la memoria (nomi composti curati), poi la traduzione fra vocabolari, poi una lettura a due direzioni con revisione; un LLM diverso da solo non cambia l'esito, un LLM con definizioni curate ed esempi di convenzione, controllato da un arbitro deterministico, va provato (E2, criteri fissati prima). Dettaglio: `perche_il_medico_legge_e_noi_sbagliamo_2026-10-09.md`.

## 28. Costruzione esaustiva e integrazione a vincoli (9 ottobre 2026, sera)

Letti i testi originali di Kintsch (1998, 2001, 2011) ed Ericsson e Kintsch (1991). La comprensione è costruzione esaustiva (tutti i sensi, tutti i nomi noti, anche quelli sbagliati) seguita da integrazione per soddisfacimento di vincoli; dove manca la conoscenza la memoria di lavoro dell'esperto non c'è. Conseguenze per l'architettura: (1) la memoria va estesa a nomi composti di ogni tipo (E4a misura il guadagno con UMLS, E4b la cura dei radiologi); (2) le regole deterministiche vanno trattate come nodi di una rete di integrazione, non come filtri a posteriori; (3) proposta E5, un integratore a vincoli deterministico (letture concorrenti con collegamenti negativi, conoscenza e contesto come nodi, attivazione fino alla stabilità, astensione senza vincitore). Gli LLM restano proponenti di letture nella fase di costruzione. Corretto un difetto delle prove (prima occorrenza invece di quella giudicata). Dettaglio: `perche_il_medico_legge_e_noi_sbagliamo_2026-10-09.md`.

