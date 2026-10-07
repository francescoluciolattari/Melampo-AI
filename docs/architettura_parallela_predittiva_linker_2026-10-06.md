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
| T2 | Stato del referto: parser dell'intestazione, lingua, strutture già collegate; previsione con `spreading_activation`; il referto intero passato come contesto | Errori di "distretto sbagliato" intercettati su casi costruiti e sui corpora pubblici; nessun calo di precisione |
| T3 | Convergenza con propagazione e decisione a quattro condizioni; LLM come flusso unico e solo se serve | Copertura e precisione rispetto a T2; chiamate LLM risparmiate; latenza |
| T4 | Gold set: affidabilità e correlazione per flusso, pesi appresi, soglia certificata (LTT/SCRC) | Precisione ≥99% certificata sugli accettati, con copertura dichiarata |

T0 e T1 si possono fare subito e non dipendono dal gold set. T4 sì.

## 6. Limiti onesti
- È un progetto: **nessuna delle tappe è costruita**. Le fonti sostengono l'organizzazione, non i nostri numeri.
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
