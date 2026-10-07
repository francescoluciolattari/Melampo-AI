# I nove stadi del linker, in dettaglio (6 ottobre 2026, rev. 5 del 7 ottobre notte: flussi paralleli, grafo, stato del referto, vicino a destra)

Aggiornamento del 7 ottobre: gli stadi sono ora flussi indipendenti con traccia, e i due LLM sono interrogati in parallelo. Sulla scelta dei modelli c'è un controllo dei vicini nel grafo UBERON, e il ripiego al padre è una proposta. Dettagli in `architettura_parallela_predittiva_linker_2026-10-06.md`, sezione 7.

Obiettivo misurabile: **precisione ≥99% sui link accettati automaticamente** (errore ≤1% al 95% di confidenza), con la massima copertura possibile; tutto il resto va in coda di revisione con il motivo. Il 100% letterale non è dimostrabile; il ≥99% certificato sì, con ~2.000 menzioni reali etichettate (vedi `docs/gold_set_protocollo.md`).

Principio aggiunto in questa revisione (richiesta di Frank): **dove una parola può voler dire più cose, decidono il contesto, la lingua e i dati che la accompagnano, in tutti i casi e non parola per parola.** Non è un decimo passo: è il cuore del passo 4, e tocca i passi 1, 2 e 8.

Stato: ✔ fatto, ◐ parziale, ○ da fare.

| # | Stadio | Stato |
|---|---|---|
| 1 | Stato del referto (modalità, regione, lato, **lingua**, **quadro**) | ◐ quadro della frase (immagini, laboratorio, segni vitali) sì (7 ott notte, `exam_frame.py`); sezioni, titolo, intestazione, lingua, modalità e "referto sul rachide" sì (7 ott, `report_state.py`); la previsione agisce solo sui codici di livello; altre regioni, lato dell'intestazione e strutture già collegate no |
| 2 | Lessico esatto bilingue + tabella parti (una forma ambigua non si accetta dal solo nome) | ✔ |
| 3 | Candidati morfologici e recupero denso (solo proposta) | ◐ encoder valutati, morfologia no |
| 4 | **Significati in competizione + integrazione** (recupero e integrazione separati) | ◐ inventario dei sensi su 7 forme; le altre forme a rischio si trovano dai dati (`form_ambiguity`, 113 chiavi su 732), la lingua della sigla vale per tutte, la verifica con i due modelli è pronta ma non misurata; punteggio continuo no |
| 5 | Controllo dei vicini (fratelli, controlaterale, padre, figli) | ◐ lato/numero/tipo sì; vicini nel grafo UBERON sulla scelta dei modelli sì (7 ott); vicini per lessico e parti no |
| 6 | Scelta vincolata con "nessuna delle precedenti" (Nemotron + Gemma) | ✔ |
| 7 | Ri-derivazione cieca da un lettore di meccanismo diverso | ◐ la traduzione IT→EN ci somiglia, manca il lettore indipendente |
| 8 | Monitor di conflitto → astensione con motivo (ora anche tra sensi) | ◐ motivi sì, punteggio unico no |
| 9 | Soglia certificata (Learn-then-Test) e ripiego al padre `part_of` | ◐ ripiego al padre calcolato dal grafo come **proposta** (7 ott; ~1–5% di errori nelle revisioni cieche, quindi non applicato); soglia: serve il gold set |

## 1. Stato del referto
Umano: l'esperto capisce il "gist" prima delle parole e si aspetta certe strutture (TC torace: polmone, mediastino). Il cervello prevede più in là e più gerarchicamente dei modelli di linguaggio (Caucheteux 2023). La lingua in cui è scritto il testo è parte dello stato: la stessa sigla vuol dire cose diverse in italiano e in inglese.
Cosa fa: dall'intestazione e dalla tecnica estrae modalità, regione, lato, sistema atteso. Ogni link deve essere coerente. Ginocchio in una TC torace → segnalato, non forzato. Lato nell'intestazione in conflitto col corpo → coda.
**Novità:** la lingua della frase viene stimata (parole che appartengono a una sola lingua; risponde solo se c'è almeno un indizio e almeno il doppio dell'altra lingua, altrimenti "non so") ed entra come indizio nel passo 4. `link()` accetta anche un `context` opzionale (il resto del referto, un'intestazione) che pesa la metà della frase.
Rischio coperto: omonimi di altri distretti, lato sbagliato dettato, copia-incolla fra referti, sigle inglesi in testo italiano.
**Fatto il 7 ottobre sera (T2, prima parte).** `report_state.py` legge, senza decidere nulla:
- le sezioni dichiarate dal referto (Quesito clinico, Indicazione, Dati tecnici, Tecnica, Sequenze, Referto, Conclusioni; Clinical history, Technique, Findings, Impression), il titolo dell'esame in maiuscolo ("RM RACHIDE LOMBOSACRALE") e le frasi di tecnica non etichettate ("L'esame è stato eseguito con…");
- il quesito clinico che si attacca ai reperti senza punto ("…sospetta colelitiasi Il fegato…");
- lingua, modalità (TC, RM, ecografia, RX, PET) e se il referto parla **solo del rachide**: lo dice l'intestazione, oppure almeno due frasi, e nessuna parola di un altro distretto compare in tutto il referto.
Le frasi passate al linker non contengono più l'intestazione (parole di intestazione nella frase: da 18 a 2 su PARROT) e l'intestazione diventa il `context` del passo 4 quando non se ne dà un altro.
**Dove la previsione agisce:** un codice di livello (L4, C7…) in un referto sul rachide è una vertebra ("Anterolistesi di L4 su L5"), ma mai T1/T2, mai nell'intestazione, mai se la frase parla di segnale RM, stadio tumorale, radice, disco, forame, spazi intersomatici o di un altro distretto. Su PARROT: 3 accettazioni in più su 229 menzioni, controllate a mano; su IU X-ray nessuna variazione.
**Ancora da fare:** altre regioni oltre al rachide, lato dell'intestazione in conflitto con il corpo, strutture già collegate come previsione (`spreading_activation`).

## 2. Lessico esatto
Umano: riconoscimento istantaneo regolato dalla frequenza (area fusiforme media: 180 ms per parole frequenti, 400 ms per rare; la frequenza spiega il 73% dell'attività, Woolnough 2021). Gli esperti non "ragionano" su LNL o LID, le conoscono. Ma per una parola ambigua nemmeno l'esperto si fida del nome: guarda la frase.
Cosa fa: corrispondenza esatta dopo normalizzazione, senza punteggi. Tabella curata parte→intero con relazione. **Una forma che compare nell'inventario dei sensi (passo 4) non si accetta mai solo perché il nome è nel lessico**: prima si decide quale significato è in gioco. "GB" è nel lessico come cistifellea (inglese) e per questo è stato un errore silenzioso sui casi italiani (globuli bianchi).
Stato: fatto. Si estende solo con doppia revisione e test di regressione: una voce sbagliata è peggio di una mancante.

## 3. Candidati morfologici e recupero denso
Umano: scomposizione cieca in morfemi (Rastle 2004), poi controllo del significato.
Cosa fa: radici greco-latine (nefro-, epato-, -ectomia) e un encoder multilingue (BioELX/SapBERT-like) *propongono* candidati. Non accettano mai.
Attenzione: insegnare i morfemi aiuta sulle parole insegnate (SMD 0,83) ma trasferisce poco a parole nuove (0,31) e quasi nulla sulla comprensione (0,13): usarli per trovare candidati, mai come prova.

## 4. Significati in competizione, poi integrazione
Umano, tre fatti distinti:
- *Accesso riordinato.* Leggendo una parola ambigua si attivano tutti i significati; la frequenza di ciascuno e il contesto li riordinano; un contesto forte elimina il significato che non c'entra (Duffy, Morris & Rayner 1988; Rodd, Gaskell & Marslen-Wilson 2002/2005).
- *Costruzione-integrazione.* La costruzione è permissiva (ogni significato è un candidato); l'integrazione tiene ciò che i vincoli sostengono. Lo schema del dominio è un vincolo fra gli altri (Kintsch).
- *Recupero ≠ integrazione.* N400 = quanto è facile recuperare il significato; P600 = quanto costa integrarlo. A parità di priming il N400 non cambia con l'implausibilità, il P600 sì (Aurnhammer 2023).

Cosa fa, in quest'ordine:
1. **Competizione dei sensi.** Per ogni forma scritta elencata in `data/linking/word_senses.json` (oggi: GB, LM, ponte, ileo, digiuno, midollo) ogni senso raccoglie prove dalla frase: parole forti (3 punti) e deboli (1), schemi di numeri e unità (forti: `GB 5040/mmc`, `42%`), la lingua (+1 se il senso è attestato in quella lingua, −2 se non lo è), il contesto più ampio a metà peso.
2. **Accettazione.** Il senso anatomico passa solo con almeno 2 punti *e* 1 punto di margine sul miglior altro senso. Il silenzio non è prova. Altrimenti il linker si astiene col motivo `sense_conflict:<senso concorrente>` o `sense_unresolved`, stadio `senses`.
3. **Classi consentite.** Un senso può limitare le classi a cui può portare (midollo spinale → `spinal_cord`); un link a un'altra classe viene fermato.
4. **Integrazione deterministica** come prima: lato, numero, tipo di struttura, tessuto/spazio, trappole, `covers`, regione. Un candidato molto simile ma che non si integra viene scartato ("porta hepatis" → vena porta).

Esempi (verificati nei test):
- "GB wall thickening with gallstones." → cistifellea (inglese +1, "gallstones" +3, "wall" +1; nessun rivale).
- "Colecistectomia pregressa, GB non visualizzata, fegato regolare." → cistifellea anche in italiano (−2 per la lingua, +3 colecistectomia, +1 fegato = 2): la prova forte supera la lingua.
- "Hb 12,3 g/dL; GB 5040/mmc (N 48%)." → astensione `sense_conflict:white_blood_cells`.
- "GB 11.2 with a fatty liver on ultrasound." → pareggio 3 a 3 → astensione.
- "GB: normal." → `sense_unresolved`: nessuna prova, nessun link.
- "Il file DICOM pesa 5 GB." → `sense_conflict:gigabyte`.
- LM: "Stenosi critica del LM e della IVA" → coronaria sinistra, astensione; "Atelettasia del LM al polmone destro" → lobo medio.

Questo risolve il compromesso che Frank ha rifiutato: non si perde "GB" inglese per cistifellea quando il contesto lo dice, e non si commette l'errore silenzioso quando il contesto dice altro.

**Cosa NON è dimostrato.** Pesi, soglie (2 e 1) e le sei voci dell'inventario sono scelte di progetto, provate su frasi costruite e sui casi E3C dove è nato l'errore; non sono misurate su un gold set. Le forme non elencate non sono controllate (il resto del lessico non è passato al vaglio per ambiguità). La lingua è un'euristica a parole-spia. Non c'è distanza sintattica, né negazione, né ambito. Le parole-indizio sono forme esatte (singolare/plurale, flessioni: vanno elencate).
**Come cresce.** Nuova forma ambigua = una voce JSON + un test, nessun codice. Ogni errore di senso trovato dal gold set diventa una voce e un test di regressione.
**Scoperta sistematica (da fare).** Invece di aspettare che le sigle ambigue emergano una per una, scansionare il lessico e il pool con UMLS (la chiave c'è): una stringa che mappa su concetti di tipi semantici diversi (struttura anatomica / test di laboratorio / unità / procedura) è una forma da inventariare. Per le forme trovate, gli indizi si ricavano dalle definizioni e dai contesti dei corpora, poi vanno rivisti da una persona.
Da fare anche: rendere l'integrazione un punteggio per soglia (accetta / ripiega / astieni), non solo un sì/no.

**Aggiornamento 7 ottobre (notte): il vicino a destra e a sinistra.** Una parola anatomica seguita da un nome di misura o di esame ("heart rate", "liver function", "thyroid hormone", "anti-thyroid") non nomina la struttura: il linker si astiene con `attribute_head_names_a_measurement:<parola>`. È una proprietà del vicino, valida per tutte le strutture (dati in `word_senses.json`, chiave `attribute_heads`). Su 600 menzioni di case report ha corretto 48 link, tutti verificati a mano; "axis" (seconda vertebra o asse geometrico) è ora un profilo dei sensi. Dettagli e limiti in `architettura_parallela_predittiva_linker_2026-10-06.md`, sezione 12.

**Il contesto è il tipo, il quadro e l'area dell'esame (7 ottobre, notte).** Tipo: titolo, sezioni e modalità dell'intestazione (ACR, RSNA). Quadro: la frase è scritta nei reperti di immagini, nei risultati di laboratorio o nei segni vitali (LOINC/HL7, SecTag): in un quadro di misura una struttura nominata è il modificatore di una misura e il linker si astiene (`frame_is_a_measurement`), tranne nei reperti di un referto di immagini. Area: oggi solo il rachide. Il vicino immediato ("heart rate") è un indizio dello stesso quadro. Dettagli, misure e fonti: `architettura_parallela_predittiva_linker_2026-10-06.md`, sezione 13.

## 5. Controllo dei vicini
Umano: l'illusione di Mosè: la sostituzione di una parola simile passa inosservata (5–60% dei casi). È l'errore "struttura sorella" o "lato opposto".
Cosa fa: per ogni link con somiglianza alta, confronto esplicito con sorelle (milza/rene), controlaterale, padre e figli. Senza una ragione verificabile per preferire il candidato → astensione.
**Fatto (7 ottobre):** `anatomy_graph.py` genera i vicini da UBERON (padre e figlio, sorelle sotto un genitore non generico, strutture dichiarate disgiunte) e dalle famiglie sinistra/destra. Se la scelta dei due modelli ha un vicino fra le opzioni che hanno superato gli stessi controlli, il linker si astiene (`neighbour_passes_the_same_checks:<tipo>`). Sul bench non scatta, perché `covers` esige già le stesse parole: è una rete di sicurezza.
Da fare: lo stesso confronto per i link del lessico e della tabella parti, e una misura sul run live.

## 6. Scelta vincolata
Umano: decisione fra alternative note.
Cosa fa: Nemotron e Gemma scelgono una lettera fra candidati già filtrati, con la menzione marcata `<tgt>`, e l'opzione "nessuna". Mai codici liberi (GPT-4 sbaglia i codici ICD nel 54–66% dei casi, 18,5% inventati: NEJM AI 2024). Le forme ambigue arrivano qui solo dopo il passo 4.
Stato: fatto. Limite: due LLM che sbagliano scelgono la stessa risposta ~60% delle volte (Kim 2025): sono copertura, non prova. Anche sulle sigle cliniche gli LLM hanno mostrato calo in lingue diverse dall'inglese e sovra-confidenza (vedi documento di sintesi): un motivo in più per non lasciare il senso a loro.

## 7. Ri-derivazione cieca
Umano: la metacomprensione migliora (r da 0,14 a 0,41) solo se si ricostruisce il significato dopo un intervallo; spiegare la propria risposta non aiuta (Prinz 2020).
Cosa fa: un secondo lettore *di meccanismo diverso* (encoder + regole, oppure un LLM che non vede la prima risposta) deriva il concetto dalla sola menzione. Il disaccordo è un'astensione.
Da fare: lettore indipendente non-LLM; misurare l'errore congiunto dei due LLM sul gold set per sapere quanto l'accordo vale davvero.

## 8. Monitor di conflitto
Umano: il segnale di errore del cingolato anteriore compare ~50 ms dopo l'errore, anche senza consapevolezza. Nei lettori, un buon rilevamento del conflitto predice quanti errori di significato vengono notati. Il controllo semantico (corteccia frontale inferiore sinistra) interviene quando più significati competono.
Cosa fa: un punteggio unico (accordo dei canali, rango nel recupero, esito dei controlli, coerenza con lo stato del referto). Un canale molto sicuro e uno contrario → astensione con motivo strutturato per la coda. **Il conflitto tra sensi (`sense_conflict:*`) è già un motivo strutturato**: la coda mostra quale altro significato era in gioco.
Principio: astenersi dal *disaccordo fra segnali indipendenti*, non dalla bassa confidenza di un solo modello (gli LLM non sanno quando sbagliano, MetaMedQA 2025).

## 9. Soglia certificata e ripiego
Umano: nessun corrispettivo; gli umani sono mal calibrati.
Cosa fa: la soglia di accettazione si sceglie con Learn-then-Test / controllo del rischio conforme su un insieme di calibrazione del gold set; si certifica su un insieme separato congelato. Prima di astenersi, ripiego al padre con `part_of` (come Mohri 2024: togliere l'affermazione incerta invece di scartare tutto). Coerenza fra testo, intestazione e lato.
**Fatto in parte (7 ottobre):** il grafo calcola il padre per un termine UBERON più fine di ogni classe, ma solo come **proposta** (`result.fallback`), non come link. Tre revisioni cieche di 100 risalite ciascuna hanno trovato 3, 1 e 5 errori più 9–13 casi dubbi, perché il parte-di di UBERON non coincide con il contenuto di una maschera TC (ipofisi ed encefalo, uraco e vescica, duodeno e tenue). Le regole aggiunte dopo ogni revisione non hanno impedito errori nuovi.
Da fare: serve il gold set, per certificare il ripiego (attivabile con `accept_parent_fallback=True`) e per tarare pesi e soglie del passo 4 (2 e 1), non a mano.

## Come si arriva al 99%
1. Gold set reale (2.000–2.300 menzioni, due radiologi, terzo revisore). Fino ad allora "0 errori" è una misura del nostro stesso lavoro.
2. Tassonomia degli errori reali: critici (lato/organo/sistema), di senso (omonimi, sigle, lingua), di relazione (parte↔intero, spazio↔tessuto), di granularità.
3. Ogni errore diventa un test di regressione e una voce di inventario o una regola (come già fatto con porta hepatis e GB).
4. Stadi 1, 5, 7, 8 per intercettare ciò che i controlli attuali non vedono; scansione UMLS per trovare le forme ambigue prima che sbaglino.
5. Soglia con Learn-then-Test e ripiego al padre.
6. Coda di revisione progettata contro l'anchoring: il revisore codifica prima di vedere il sistema; audit del 2–5% degli accettati; casi sentinella.
Realistico: ≥99% di precisione sugli accettati a copertura alta è plausibile, non garantito: lo stabilisce il gold set. L'errore certificato vale per questa coppia sistema-popolazione e va riverificato a ogni cambio di modello, lessico o ospedale.
