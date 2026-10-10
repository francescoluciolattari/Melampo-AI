# Perché il medico legge un quadro clinico senza problemi e noi sbagliamo: contesto, scomposizione, modo di lettura, o un altro modello?

9 ottobre 2026. Risposta alla domanda di Frank dopo il run `phrase-probe` con il segmentatore LLM (`docs/linker/lettura_del_sintagma_cervello_e_modelli_2026-10-09.md` §8.5). Qui "quadro" è letto come "quadro clinico". Per le fonti sul cervello e sull'esperto vale anche `comprensione_umana_linking_clinico_2026-10-06.md`, che non ripeto.

**Revisione della sera (stesso giorno).** Frank ha fornito i testi originali: Kintsch 1998 (*Comprehension: A paradigm for cognition*, libro intero), Kintsch 2001 (*Predication*), Kintsch e Mangalath 2011 (*The construction of meaning*), il rapporto tecnico di Ericsson e Kintsch del 1991 (ICS 91-13, versione preliminare dell'articolo del 1995 sulla memoria di lavoro a lungo termine), la recensione di Gernsbacher e McKinney al libro di Kintsch, Landauer e Dumais 1997 (LSA), Sweller 2024 (carico cognitivo) e Sanchez e Wiley 2006 (dettagli seduttivi). Li ho letti nelle parti che servono (indicate con le pagine). Questa versione:

- sostituisce la sintesi online di Ericsson e Kintsch con il testo originale;
- **corregge la classificazione dei 33 errori** (§2): due sviste mie e un difetto degli strumenti di misura, trovati rifacendo il conteggio a livello di concetto (E1);
- aggiorna il §5 sui modelli: i lavori citati usavano modelli vecchi (GPT-3, GPT-4); ho aggiunto i lavori del 2026 che ho trovato e reso il modello una scelta di chi lancia l'esperimento;
- aggiunge il §6 su che cosa dicono i testi di Kintsch e il §7 con il modello che ne segue;
- descrive E1, E2 ed E4a come **realizzati** (in attesa dei run su GitHub) e fissa i criteri prima dei run.

## 1. Risposta breve

1. **Prima di tutto è la memoria, poi il modo di leggere; il contesto conta in pochi casi precisi.** Kintsch lo dice in modo netto: dove manca la conoscenza la memoria di lavoro a lungo termine non c'è (1998, p. 221), e quando il suo algoritmo di predicazione sbaglia il senso di *foil* (scherma) "indica una mancanza di conoscenza, non necessariamente un difetto dell'algoritmo" (2001, p. 197). È il nostro caso: 9 errori su 33 sono nomi composti di un'altra cosa che la nostra memoria non ha.
2. **Il modo di leggere è diverso.** Il lettore umano costruisce **tutte** le letture possibili delle parole (anche quelle sbagliate), poi le integra per soddisfacimento di vincoli e tiene quella coerente con il resto (modello di costruzione-integrazione, Kintsch 1998, cap. 4-5). Il nostro flusso costruisce solo letture anatomiche e decide in una passata. L'esperto, inoltre, si accorge delle incoerenze e abbandona più facilmente un'ipotesi sbagliata (Ericsson e Kintsch 1991, pp. 59-60).
3. **Il contesto conta per 4 errori su 33**: una parola con un altro senso ("heart" come *cuore del formaggio*, 3 volte; "aortic arch" come arteria dell'arco faringeo in un embrione, 1 volta). Di questi, due richiedono il documento intero, non la frase.
4. **La metà degli errori (16 su 33) non è lettura**: sono differenze di granularità o di mappatura fra vocabolari. Il conteggio a livello di concetto (E1) lo rende visibile.
5. **Un LLM diverso, da solo, non risolve.** Il motivo è lo stesso del punto 1: un lettore senza la conoscenza del dominio elabora lo stesso, ma in modo sbagliato (Moravcsik e Kintsch 1993, in Kintsch 1998, pp. 288-289). La prova che conta è E2: lo stesso LLM con in mano la conoscenza e gli esempi. Il modello si sceglie nel workflow, quindi si può provare anche il più recente.
6. **Il modello nuovo che i testi indicano non è un LLM più grande**, ma un integratore a vincoli alla Kintsch (§7): letture concorrenti come nodi, conoscenza e regole come nodi, attivazione che si stabilizza, astensione se non c'è un vincitore. È deterministico e ispezionabile, quindi adatto a un dispositivo medico. Lo propongo come E5, dopo i numeri di E2 ed E4a.

## 2. Perché sbagliamo: i 33 errori, classificazione corretta

**Che cosa è cambiato rispetto alla mattina.**

- **Difetto degli strumenti di misura.** `phrase-probe`, `lattice-probe` e `head-probe` marcavano la *prima* occorrenza della parola nella frase, non quella giudicata. Nei 2.058 link giudicati la parola compare più di una volta in 278 frasi (13,5 %), 5 delle quali sono errori. Il caso più evidente è "brain" nella frase "Canine brain phantoms … agarose brain parenchyma": l'etichetta riguarda la seconda occorrenza ("brain parenchyma"), mentre il segmentatore LLM ha letto la prima ("Canine brain phantoms"). Quella "riuscita" del §8.5 del documento sul sintagma era quindi un artefatto. Il controllo esterno ora salva la posizione della menzione nella frase (`at`), e tutte e tre le prove la usano. I numeri di ieri su quelle 278 frasi vanno rimisurati.
- **Le etichette lette davvero.** Nella prima classificazione avevo usato soltanto CUI e tipo semantico. Rileggendo anche il testo etichettato (`label_texts`) e le mappature UBERON, 6 casi cambiano classe:
  - "heart" nel formaggio ha come etichetta la sola parola "heart" (C1254362, tipo T082 *concetto spaziale*): è un **altro senso**, non un nome composto;
  - "mucosa of the colon" ha come etichetta la sola parola "colon";
  - "brain white matter property" ha un'etichetta **anatomica** (C0152381, T023), che UBERON mappa alla sostanza bianca del cervelletto;
  - il CUI di "aortic arch" (C0003489, T023) è mappato da UBERON a *pharyngeal arch artery*, una struttura embrionale: errore di mappatura;
  - in CRAFT l'"aortic arch" della frase sulle 4e arterie faringee è proprio l'arteria embrionale: un altro senso dato dal contesto.

| causa | n | casi | è un errore di lettura? |
|---|---|---|---|
| **Granularità o mappatura dell'etichetta** | 16 | CRAFT: "right middle lobe" ×7 (etichetta *anatomical lobe*), "bladder" ×3 (etichetta *bladder organ*): l'etichetta è una classe più generale della nostra. MedMentions: "aortic arch" ×2 (CUI mappato da UBERON a un'arteria embrionale), "brain parenchyma" ×2 (CUI mappato a *parenchyma* generico), "brain white matter" (mappato alla sostanza bianca del cervelletto), "large bowel" (UBERON lo dà come sinonimo esatto di *colon*, UMLS come intestino crasso) | no: vocabolari che non si allineano |
| **Nome composto di un'altra cosa** | 9 | "inferior vena cava filter placement" ×2 (procedura), "donor-specific spleen cells transfusion", "liver living donors" (gruppo di persone), "developing brain" ×2 (funzione), "development of pancreas" (funzione), "levels of heart interleukin-6" (procedura di laboratorio), "brain controls" (processo mentale) | sì: manca il nome in memoria |
| **Altro senso della stessa parola** | 4 | "heart" ×3 = cuore del formaggio (1 riconoscibile nella frase, "heart of Maroilles cheese"; 2 solo dal documento); CRAFT "aortic arch" = arteria dell'arco faringeo nell'embrione | sì: senso dal contesto |
| **Etichetta dubbia** | 4 | "colon" ×2 con lo stesso CUI C3888384 di tipo T204 (*eucariote*: in UMLS è un organismo, forse il genere di coleotteri *Colon*); "colon" C0391907 (tipo sconosciuto); "brain retained inside the cranium" etichettato come cranio | da verificare con i nomi UMLS (E1 li stampa) |

Conclusioni:

- **Gli errori di lettura sono 13 su 33 (39 %)**: 9 di memoria e 4 di senso.
- **Il conteggio per concetto (E1) cambia la precisione misurata.** Anche solo togliendo dal giudizio i link la cui etichetta è più generale della struttura nominata (sono "non giudicati", non "giusti") e senza UMLS, CRAFT passa da 11 errori su 1.326 a **1 su 1.316 (99,92 %)** e MedMentions da 22 su 732 a 21 su 731 (97,13 %). La mappatura attraverso i codici FMA e NCIt di UMLS (E1, con la chiave) dirà quanti dei 6 casi di MedMentions erano solo mappatura.
- **Gli annotatori di MedMentions cercavano, non leggevano come un medico.** Hanno cercato a mano i termini nel Metathesaurus UMLS 2017AA e scelto il concetto più specifico, senza menzioni sovrapposte ([Mohan & Li](https://arxiv.org/pdf/1902.09476)). L'articolo non riporta un accordo fra annotatori, solo il 97,3 % fra revisori e annotatori su 469 concetti.
- Per il prodotto la convenzione bersaglio resta RadGraph con le classi di TotalSegmentator, non MedMentions né CRAFT.

## 3. Che cosa fa il cervello e che cosa fa il nostro flusso

| meccanismo | fonte | il lettore esperto | il nostro flusso |
|---|---|---|---|
| **Costruzione esaustiva, poi integrazione** | Kintsch 1998 §4.1 pp. 96-101; §5.1.2-5.1.3 pp. 128-133 | Attiva tutti i sensi di una parola, anche quelli fuori contesto: subito dopo la parola entrambi i sensi di un omografo sono attivi. Entro circa 350 ms il contesto lascia solo quello giusto. Le inferenze sul tema della frase arrivano dopo 500-1.000 ms. Le costruzioni alternative delle stesse parole si inibiscono a vicenda; l'integrazione è un'attivazione che si propaga fino a stabilizzarsi. | Costruisce solo letture anatomiche (lessico, parti). Il reticolo sceglie il cammino di costo minimo e si ferma. Non ci sono letture concorrenti di altro tipo né una fase di integrazione. |
| **Memoria di lavoro a lungo termine** | Ericsson e Kintsch 1991 (ICS 91-13), abstract e pp. 58-61; Kintsch 1998 §7.1 pp. 217-221 | Usa la memoria a lungo termine come memoria di lavoro attraverso strutture di recupero; un recupero richiede 300-400 ms. Questo è possibile solo con molta conoscenza del dominio e strategie di codifica allenate: "dove manca la conoscenza, la LT-WM non è disponibile". | NCIt, UBERON e 6.705 nomi lunghi. Pochi nomi composti di procedura, dispositivo o misura, quasi nessuno in italiano. |
| **Il medico esperto** | Ericsson e Kintsch 1991, pp. 59-60; Kintsch 1998 pp. 233-234 | Chi ha meno esperienza ha più conoscenza *sbagliata* delle malattie. L'esperto accede alla conoscenza in modo più affidabile e la integra meglio, scopre le incoerenze e si riprende più facilmente da un'ipotesi diagnostica sbagliata (Feltovich et al. 1984; Johnson et al. 1981). Nel ricordo di un caso, gli specializzandi riproducono il testo senza distinguere l'essenziale; i medici esperti ricostruiscono sintomi essenziali e diagnosi (Groen e Patel 1988; Schmidt e Boshuizen 1993). | Nessuna revisione: la prima lettura stabile vince. Le voci sbagliate del lessico producono errori sicuri. |
| **Conoscenza contro abilità** | Kintsch 1998 §9.1.3 pp. 287-290; Sweller 2024 | La conoscenza del dominio può compensare anche una bassa abilità verbale (Recht e Leslie 1988; Schneider et al. 1989). Moravcsik e Kintsch 1993: i lettori con poca conoscenza, aiutati da un testo ben scritto, ricordano il testo, ma le loro elaborazioni e inferenze sono **sbagliate e fantasiose**; solo chi ha conoscenza elabora correttamente. Per Sweller la conoscenza in memoria a lungo termine è la principale (forse l'unica) differenza modificabile fra le persone: i blocchi (*chunk*) riducono l'interazione fra elementi. | Un LLM senza memoria del dominio somiglia al lettore con poca conoscenza: elabora comunque. |
| **Predicazione: il senso nasce dal contesto** | Kintsch 2001 pp. 178-183, 196-197; Kintsch e Mangalath 2011 | Il significato di una parola non si prende già pronto da un lessico: si costruisce prendendo i vicini semantici della parola che sono collegati anche all'argomento (P(A) = centroide di P, A e dei k vicini di P più attivati; k fra 1 e 5). Nel test sugli omonimi è giusto in 9 casi su 10; l'unico errore (*foil* come scherma) è una mancanza di conoscenza dello spazio semantico. Servono due memorie: una del *gist* (LSA o topic model) e una esplicita (co-occorrenze e dipendenze sintattiche), e la combinazione delle due batte ciascuna da sola. | "heart" si legge come cuore in ogni contesto, anche vicino a "cheese". |
| **Conoscenza formale contro associazione** | Kintsch 1998 §11.2.1 pp. 399-401 | Nel problema di Linda una rete costruita solo sulla storia preferisce "cassiera e femminista" (attivazione 0,75 contro 0,31). Aggiungendo alla rete un nodo di conoscenza formale (la regola delle probabilità) la preferenza si rovescia (0,34 contro 0,01). | Le nostre regole deterministiche (lato, sistema, parti) agiscono come filtri a posteriori, non come nodi della rete. |
| **Dettagli seduttivi** | Sanchez e Wiley 2006, Memory & Cognition 34 | Chi ha meno capacità di controllo dell'attenzione si fa distrarre di più dalle informazioni non pertinenti, e le guarda più spesso e più a lungo. | Per un LLM: aggiungere contesto non pertinente alla frase lo distrae (il *focus-mismatch* di BioELX nel documento del 6 ottobre). |
| **Memoria distribuzionale** | Landauer e Dumais 1997 | LSA, addestrata solo su testo, risponde bene a 64,4 % degli 80 item TOEFL, contro il 64,5 % dei candidati non anglofoni ai college USA; la maggior parte di ciò che apprende viene da inferenze indirette. Ma "conosce solo quello che le è stato insegnato" (Kintsch 2001, p. 176). | Gli LLM sono discendenti di questa idea, con lo stesso limite: senza i testi del dominio non hanno la conoscenza del dominio. |
| **Un senso per discorso; modello della situazione** | Gale, Church e Yarowsky 1992; Zwaan e Radvansky 1998 | Il senso si tiene per tutto il discorso; il lettore costruisce un modello di ciò che il testo descrive. | Il braccio `discourse` è locale. |
| **Circolo ermeneutico** | [SEP, Hermeneutics](https://plato.stanford.edu/entries/hermeneutics/) | Proiezione e superamento ripetuti delle interpretazioni inadeguate. | Una sola passata. |

## 4. Le tre domande di Frank

**È il contesto che fa la differenza?** Per 4 errori su 33. Le forme sono tre: la frase (predicazione: "heart of Maroilles cheese"), il documento (un senso per discorso, inferenza del tema in 500-1.000 ms) e la cornice dell'esame (embrione o adulto). Nei referti il documento conterà di più: lato ereditato dalla frase precedente, "la lesione", termini ripetuti.

**È il metodo di scomposizione?** No, se per metodo si intende l'algoritmo che taglia la frase. La scomposizione umana riconosce blocchi che sono già in memoria. Con la memoria giusta anche un algoritmo semplice funziona; senza memoria, nessun algoritmo trova il nome di una procedura che non conosce.

**È la modalità di lettura?** Sì, ed è la parte che non abbiamo ancora costruito: costruzione esaustiva (tutte le letture, di ogni tipo), integrazione per vincoli (comprese le regole come nodi) e revisione quando non c'è un vincitore.

## 5. Serve un LLM diverso? I modelli recenti

- **I lavori citati la mattina usavano modelli vecchi** (GPT-3, GPT-4, GPT-4.1). Non posso verificare quali modelli siano usciti dopo la mia data di conoscenza (giugno 2026); il workflow permette di metterne qualunque id OpenRouter (`llm_models`), così il modello più recente si misura direttamente invece di discuterne.
- **Lavori del 2026 trovati:**
  - **Allucinazioni dei modelli di frontiera nel linking a SNOMED CT** (workshop ClinicalNLP di LREC 2026, letto solo il riassunto). Gli LLM di frontiera inventano codici medici in misura tale da renderli "non adatti alla codifica clinica autonoma". Vincolarli agli span giusti peggiora le allucinazioni invece di ridurle. Gli LLM generici rendono molto meno dei metodi specializzati di entity linking zero-shot.
  - **Linking su MedMentions senza fine-tuning** (Studies in Health Technology and Informatics 2026). Un Qwen da 7 miliardi di parametri già specializzato su UMLS, con 5 candidati da ricerca su 8,18 milioni di nomi UMLS, ottiene 0,560 di accuratezza su 69.891 menzioni; sale a 0,799 quando il concetto giusto è fra i 5 candidati. I modelli supervisionati citati fanno 0,70-0,81, ma su un altro sottoinsieme (il confronto non è diretto).
- **Il quadro resta quello di Kintsch:** senza conoscenza del dominio un lettore elabora comunque, e sbaglia (Moravcsik e Kintsch 1993). I due modelli provati concordano sul tipo nel 79 % dei casi e mancano gli stessi errori; gli errori di LLM diversi sono correlati (circa il 60 % delle volte la stessa risposta sbagliata, Kim et al. 2025).
- **Definizioni:** quelle prese da una fonte curata aiutano, quelle scritte dal modello stesso no (arXiv 2404.00152, con GPT-4).
- **Esempi:** con pochi dati annotati, esempi simili e un riassunto della guida di annotazione nel prompt funzionano meglio di un modello piccolo addestrato (EvalLLM 2025).

## 6. Esperimenti, con i criteri fissati prima dei run

| # | esperimento | stato | criterio di riuscita |
|---|---|---|---|
| E1 | Controllo esterno contato per concetto: (a) i link la cui etichetta è una classe più generale della struttura nominata escono dal giudizio; (b) la mappatura dei CUI attraverso i codici FMA e NCIt di UMLS. Nella direzione opposta, elenca i link giusti che la seconda mappatura contraddice; stampa i nomi UMLS delle etichette degli errori. | realizzato (`scripts/concept_check.py`, `external-check` con `concept_level`). Senza chiave: CRAFT 1 errore su 1.316, MedMentions 21 su 731 | nessun criterio di riuscita: è una correzione della misura, e ogni riga spostata è elencata perché si possa contestare la regola |
| E2 | Segmentatore LLM *informato*: regole di annotazione; ciò che NCIt e UMLS sanno dei gruppi di parole attorno alla menzione (tutte le letture, anche quella anatomica, senza suggerire la risposta); 5 frasi annotate di altri documenti con la stessa parola, prese dallo split di addestramento di MedMentions e mai dal documento del caso. Risultati anche per split (gli errori MedMentions sono 17 in addestramento, 2 in sviluppo, 3 in test). | realizzato (`phrase-probe`, `llm_mode = informed`, modelli scelti in `llm_models`) | almeno 12 errori MedMentions su 22 trovati con al più il 3 % dei link giusti letti segnalati, **oppure** il ruolo dell'etichetta in almeno il 70 % degli errori tipizzati (gli stessi criteri del run precedente) |
| E3 | Flusso di documento: senso unico per discorso e tema del documento | da fare, sui referti iu-xray ed E3C | nessun link giusto peggiorato; trova i casi di lato ereditato |
| E4a | Braccio di memoria UMLS: il gruppo di parole più lungo attorno alla menzione che UMLS nomina esattamente; se il suo tipo non è anatomia o malattia, il sintagma nomina un'altra cosa. Nessun modello. Misura direttamente "è la memoria?". | realizzato (`phrase-probe`, `umls = true`) | gli stessi criteri di E2 |
| E4b | Memoria curata: i nomi trovati da E4a (`umls_compound_candidates.json`: nome, tipo, frequenza, frasi, errori) rivisti dai radiologi, poi in `longer_names.json` | il generatore c'è; la revisione richiede radiologi | gli errori con nome composto scendono sotto un terzo senza perdere link giusti sul gold set |
| E5 | Lettore a costruzione-integrazione alla Kintsch (§7): spazio semantico LSA, predicazione, memoria a due vie (tracce esplicite con recupero per indizio più gist del documento), costruzione di tutte le letture, integrazione per vincoli | **realizzato come strumento** (`src/melampo/memory/ci_reader.py`, `scripts/ci_probe.py`, workflow `ci-probe`); misurato il 10 ottobre; **non spento nel linker** (risultati in §6.1) | stessi criteri di E2 ed E4a |

**Risultato di E1 (run `external-check` del 9 ottobre, con UMLS, 0 ricerche perse).** CRAFT: 11 errori per identificatore, 10 escono dal giudizio perché l'etichetta è più generale della struttura (7 "right middle lobe" con *anatomical lobe*, 3 "bladder" con *bladder organ*), ne resta 1; precisione 0,9917 → 0,9992 su 1.316 giudicati. MedMentions: 22 → 19 errori su 731 (0,9699 → 0,974): 2 "aortic arch" rientrano con la seconda mappatura (FMA:3768 → UBERON:0001508; NCIT:C32123 → UBERON:0004363), 1 "large bowel" esce perché C0021851 è *Large Intestine*, più generale di *colon*. **Nessun link giusto è contraddetto** dalla seconda mappatura (0). Il lato CRAFT si conferma quasi tutto convenzione dell'etichetta; il lato MedMentions no: restano 19 casi che E1 non tocca. I nomi UMLS delle etichette li rendono leggibili: *Spatial Concept* (T082) per "heart" ×3, *Colon <Coloninae>* (T204, un taxon) per "colon" ×2, *Bone structure of cranium* per "brain retained inside the cranium", *Cerebellar white matter* e *Parenchyma* per "brain white matter" e "brain parenchyma", *Living Donors*, *Transfusion - action*, *Interleukin 6 Measurement*, *brain/pancreas development* per i nomi composti. Cinque etichette (heart ×3, colon taxon ×2) sono dubbie come etichette e vanno riviste dai radiologi, non corrette da noi. CUI C0596948 (brain controls) è sconosciuto a UTS.

**Run #4 di `phrase-probe` (9 ottobre): interrotta a 5 ore, nessun report.** Dal solo zip dell'external-check non si vede la fase lenta (non ho il log del run). Cause plausibili, tutte nel flusso: l'indice SapBERT sui nomi NCIt su CPU (1–2 ore da solo, già nelle stime), le circa 6.700 ricerche UMLS e le chiamate LLM con modelli a ragionamento (timeout di 90 s x 4 tentativi per chiamata). Il difetto vero è un altro: il report e la cache UMLS si scrivevano solo a fine run, quindi il limite di 300 minuti ha cancellato tutto. Rimedio nel bundle: `max_minutes`, cache salvata ogni 200 richieste e anche a run fallito, log a tempi per fase in `phrase_probe.log`.

**Risultato di E2 ed E4a (run `phrase-probe` del 10 ottobre, con `gliner` ed `encoder` accesi; 37 minuti in tutto, 0 ricerche UMLS perse, 7.135 richieste).** I criteri fissati prima **non passano**.

| segnale | errori MedMentions segnalati | link giusti MedMentions segnalati | errori CRAFT segnalati | link giusti CRAFT segnalati | ruolo tipizzato | esito |
|---|---|---|---|---|---|---|
| E2 `llm_other` (2 modelli, prompt informato) | 6/22 | 16/114 (14 %) | 0/11 | 14/184 (7,6 %) | 1 su 12 | non passa |
| E4a `umls_other` | 2/22 | 9/710 (1,3 %) | 0/11 | 49/1.315 (3,7 %) | 1 su 1 | non passa (vedi sotto) |

* **Il "True" del ruolo di E4a è un artefatto.** Il braccio ha tipizzato un solo errore su 22 (1 su 1 = 100 %). Il criterio non fissava un minimo di errori tipizzati; la lettura corretta è che **non passa**. Il codice va corretto con una copertura minima (almeno metà dei 22).
* **E2 informato legge peggio nel recupero e meglio nella precisione rispetto al run semplice** (semplice: 10/22 errori con il 29 % dei link giusti segnalati; informato: 6/22 con il 14 %). Nessuno dei due arriva al 3 %. L'AUC di `llm_other` è 0,538, vicina al caso.
* **E4a trova un nome UMLS più lungo per 3 errori su 22.** Il motivo: il nome in UMLS è quasi sempre diverso dalla stringa annotata ("donor-specific spleen cells transfusion" ha come concetto *Transfusion - action*); la ricerca esatta per stringa non lo vede. L'ipotesi "UMLS conosce il nome composto" è **respinta per i nomi esatti**; resta possibile con una ricerca per termine normalizzato o per parti, da provare solo se serve.
* **Falsi allarmi di E4a su CRAFT: 39 dei 49 sono "brain" in "brain weight/volume" (tipo *property*).** Gli annotatori di CRAFT segnano l'organo dentro la misura; quelli di MedMentions segnano l'intera frase ("brain white matter property" è un errore del nostro link). La convenzione cambia da un corpus all'altro, quindi **una regola lessicale universale "se il sintagma nomina una misura, il link è sbagliato" è falsa**. Va registrato il ruolo (la misura *di* quell'organo), non un astenersi.
* **Tutti i segnali separano gli errori dai link giusti poco più che a caso** (AUC fra 0,50 e 0,60 su 2.058 link). Con E1 (13 dei 33 sono errori di lettura veri, il resto convenzione dell'etichetta) il quadro è coerente: non c'è un segnale di lettura che isoli quei 33 casi.
* Secondo la regola decisa prima: **nessuno dei due passa, quindi servono il gold set dei radiologi (span e ruolo) e un modello addestrato**; E5 (integratore a vincoli) non è giustificato dai numeri di questi due esperimenti.
* Tempi misurati: caricamento di `gliner` 34 s, indice `encoder` 1.543 s (26 minuti), raccolta dei casi 272 s, braccio UMLS 8 minuti (1.871 s in UTS), fase LLM 456 s per 333 casi x 2 modelli. La run #4 interrotta a 5 ore non è spiegata da questi tempi: non so dire perché lì l'indice non finisse.

### 6.1 E5 realizzato e misurato (10 ottobre 2026, in sessione, sui 2.058 link del controllo esterno)

**Che cosa è stato costruito**, una voce per ogni idea dei testi che Frank ha chiesto di implementare:

| idea | dove sta nel codice |
|---|---|
| costruzione esaustiva (Kintsch 1998) | sei letture sempre costruite (struttura, sito di procedura, sito di dispositivo, oggetto di una misura, origine di una molecola, pezzo del nome di altro); ogni blocco del reticolo che contiene la menzione è un nodo (`ChunkLattice.candidates`), non solo il più economico |
| integrazione per vincoli | rete con legami eccitatori dove le cose combaciano, inibitori fra le letture della stessa stringa; `A(t+1) = W·A(t)`, negativi a zero, divisione per il massimo, arresto sotto 0,001 |
| predicazione (Kintsch 2001) | vettore del predicato (la testa) + argomento (la menzione) + i k=3 fra i m=100 vicini del predicato più legati all'argomento |
| memoria di lavoro a lungo termine (Ericsson e Kintsch 1991) | tracce delle annotazioni dei documenti di addestramento di MedMentions, recuperate per indizio dal più specifico al più generale: parola+successiva, parola+testa, parola+precedente, poi gli stessi con il *tipo* del vicino, poi "qualunque struttura anatomica"+testa, poi la parola sola; più specifico l'indizio, più peso |
| memoria doppia CI-II (Kintsch e Mangalath 2011) | tracce esplicite (relazionali) più gist: argomento del documento nello spazio, confrontato con i documenti in cui ogni lettura è stata vista |
| spazio semantico (Landauer e Dumais 1997) | LSA con pesi log-entropia e SVD (100 dimensioni, 39.074 parole) sui testi dei corpora e sulle 146.096 definizioni NCIt; i prototipi delle letture sono centroidi di definizioni per tipo |
| testa del sintagma tipizzata (richiesta di Frank, metodo 1) | il reticolo già c'era (§ lettura del sintagma, 8); qui la testa entra come nodo e, in più, come indizio *appreso dal dominio*: dalle annotazioni di addestramento si impara che cosa vuol dire "una struttura seguita da X" |

Le costanti sono fissate per principio e scritte nel codice; il documento in lettura non è mai nella memoria con cui si legge (tutto "lascia fuori il documento").

**Risultati.** Cinque versioni, tutte riportate: sono state modificate dopo aver visto la precedente, quindi i 2.058 link sono qui un insieme di sviluppo e **nessun numero è un test**.

| versione | che cosa cambia | MedMentions: errori letti "non struttura" / link giusti cambiati | AUC | CRAFT: link giusti cambiati |
|---|---|---|---|---|
| v1 | tutto per principio, nodo "il nome è noto" | 4 su 22 / 68 su 710 | 0,525 | 96 su 1.315 |
| v2 | tolto il nodo "nome noto" (contava due volte la stessa cosa del blocco), contrasto delle parole di contesto riportato a una scala libera da etichette, indizi per tipo del vicino | 13 su 22 / 348 su 710 (49 %) | 0,533 | 235 su 1.315 |
| v3 | aggiunti gli indizi "qualunque struttura + parola successiva / testa" | 10 su 22 / 333 su 710 (47 %) | 0,514 | 213 su 1.315 |
| v3 senza le parole di contesto | | 3 su 22 / 41 su 710 (5,8 %) | 0,632 | 62 su 1.315 |
| v4a | **ambito** al posto delle parole di contesto: un solo segnale per frase (e uno per documento), cioè il tipo NCIt delle altre cose nominate; indizi "parola + ambito", "qualunque struttura + ambito", "qualunque struttura + ambito del documento" | 3 su 22 / 28 su 710 (3,9 %) | 0,605 | 37 su 1.315 |
| v4b | l'ambito vale solo per **quanto aggiunge alla norma** (eccesso di ogni ruolo sulla frequenza di base, come il gist è centrato); un ambito sparso non sceglie più un tipo a caso; il tipo vuoto "conceptual" non conta come ambito | 3 su 22 / 41 su 710 (5,8 %) | 0,632 | 62 su 1.315 |
| riferimento: reticolo da solo | | 9 su 22 / 156 su 710 (22 %) | | 223 su 1.315 |

**Che cosa ne ricavo, senza abbellire.**
* **Nessuna configurazione passa i criteri** (12 errori su 22 con al più il 3 % di link giusti cambiati).
* **Le parole di contesto sono rumore a questa scala.** Con lo spazio costruito da 4.392 abstract, 97 articoli e le definizioni NCIt (Landauer usava 4,6 milioni di parole) la loro parentela è troppo debole: cambiano la lettura di quasi metà dei link giusti. Tolte, i falsi allarmi scendono al 5,8 %, ma si trovano solo 3 errori su 22. Va rifatto con uno spazio molto più grande (PubMed intero o referti), non con altre costanti.
* **L'ambito (v4) elimina il rumore, ma non recupera gli errori.** Con l'ambito al posto delle parole di contesto i link giusti cambiati scendono da 333 a 28 su 710 (da 47 % a 3,9 %), ma gli errori trovati restano 3 su 22. In v4a il guadagno viene soprattutto dal fatto che l'ambito, così com'è, *ripete la norma* (nei testi di ogni ambito la struttura è quasi sempre il ruolo giusto): è una prior, non un'informazione sul contesto. Per questo in v4b l'ho centrato sulla frequenza di base: il nodo smette di spingere verso "struttura" e i falsi allarmi tornano a 41 su 710, identici a quelli senza ambito, cioè **l'ambito appreso da 4.392 abstract non aggiunge informazione**. Per tipo di documento (v4a) i falsi allarmi sono 7 su 151 (sviluppo), 6 su 120 (test), 15 su 439 (addestramento): il test non è migliore dello sviluppo, come deve essere.
* **Perché i tre "heart" del formaggio non si muovono.** La frase nomina agar (sostanza chimica), ceppi (organismo), MRS (procedura), "sample" (generico) e formaggio (cibo): cinque tipi con una parola ciascuno. Con la prima versione l'ambito scelto era il primo inserito («conceptual»), cioè un caso; ora è «chemical+food» in modo deterministico. Ma nell'addestramento un ambito «cibo» ha solo 11 esempi di una struttura (10 struttura, 1 "non sito"): con undici tracce nessun nodo può rovesciare un blocco anatomico sicuro. Servirebbe un corpus in cui il cibo sia comune (una memoria che ha *letto* di formaggi), non altre costanti.
* **Dei 22 errori, 12 sono quelli che una lettura può correggere** (9 nomi composti, 3 altro senso); i 6 di granularità o convenzione e i 4 con etichetta dubbia non sono errori di lettura. Il criterio preregistrato "12 su 22" coincide quindi col massimo possibile: va tenuto presente quando si dice che un lettore "non passa".
* **La memoria serve.** Senza tracce il lettore cambia 472 link giusti su 710 e trova 14 errori; con le tracce 333 e 10: le tracce tolgono 139 falsi allarmi al prezzo di 4 errori. Le tracce con il *tipo* del vicino (generalizzazione) sono quelle che funzionano; le tracce con la sola parola esatta quasi mai trovano qualcosa, perché i composti sono rari.
* **I tre "heart" del formaggio sono letti come "non è un sito del corpo"** (v2 e v3), cosa che il reticolo non vede; "Liver Donation" è sottospecificato; "developing brain" non è letto correttamente. Dove il lettore trova l'errore, lo trova per la ragione giusta, ma costa troppi link giusti.
* **L'AUC resta fra 0,51 e 0,63.** Un lettore che integra tutto non separa gli errori dai link giusti meglio dei segnali singoli: la causa non è un segnale che manca, ma che 16 dei 33 errori sono convenzioni di etichetta e 4 sono etichette dubbie (§2). Combinando i segnali del run di `phrase-probe` con una regressione e validazione incrociata a 5 pezzi l'AUC sale solo a 0,65 (MedMentions 0,63; 33 errori, quindi con grande incertezza).
* **Il lettore resta spento.** Non decide nulla nel linker. È uno strumento di misura per quando ci saranno lo spazio grande e il gold set.

**Che cosa farebbe salire l'AUC** (nell'ordine in cui ha senso provare): (1) cambiare il compito: non "è un errore?" ma "quale ruolo registrare?"; i 16 errori di etichetta spariscono come errori quando il ruolo è registrato e i radiologi decidono la convenzione; (2) un gold set con span e ruolo, che dà il segnale pulito (oggi le etichette mescolano convenzioni di annotatori diversi); (3) uno spazio semantico grande e di dominio (PubMed, referti); (4) un modello appreso sul gold set, con questi segnali come ingressi (con 33 errori non si può addestrare nulla).

**Come ragionano le persone sui nomi composti** (dalla mia conoscenza, non verificata in questa sessione: nei documenti del Project non c'è). La letteratura sulla comprensione dei composti in inglese dice due cose: nei composti noti il significato è recuperato intero dalla memoria (via lessicale), negli altri la testa a destra fissa la categoria ("liver donation" è una specie di donazione) e il modificatore riempie una relazione con quella testa (di, per, fatto di, che causa); la relazione scelta dipende da quali relazioni quel modificatore ha già avuto con altre teste. È ciò che fa il metodo 1 (tipo della testa, ruolo per il modificatore) con una differenza: gli umani la relazione la imparano dalle esposizioni, noi la leggiamo da NCIt e dalle annotazioni. Il metodo 1 è quindi coerente con la letteratura, ma non è una sua replica: la parte appresa (le tracce) è quella che ha funzionato, quella derivata dalle definizioni NCIt è la più debole.

Che cosa decide che cosa: se E4a passa, la risposta è la memoria e si va a E4b ed E5. Se E4a non passa ma E2 sì, il guadagno viene dall'integrazione che fa il modello, e va resa deterministica (E5). Se nessuno dei due passa, servono il gold set dei radiologi e un modello addestrato.

### 6.2 Dopo la v4: split congelato, testa tipizzata, filtro grammaticale, lettore nel linker (10 ottobre 2026)

**Split di test congelato.** I 879 documenti del test ufficiale di MedMentions sono registrati con la loro impronta SHA-256 (`data/linking/frozen_test_split.json`). Da ora nessuna misura di sviluppo ne legge le righe (a meno di `--final`, l'unica esecuzione che scrive il certificato), lo spazio semantico non li contiene e un test fissa l'impronta. **Non era un test vergine**: di quei documenti erano stati giudicati 347 link (3 errori, 120 corretti) e letti prima. Lo dice il manifesto. Conseguenza sui numeri: le misure di sviluppo ora sono su 1.935 link (30 errori; MedMentions 19 errori e 590 link giusti), non più 2.058 (33; 22 e 710). I criteri dei 12 errori su 22 vanno riformulati prima del giro finale: il massimo che una lettura può correggere è ora 10 su 19 (le due frasi congelate di "spleen cells" e "colon microbiome" erano nomi composti). Una proposta, da decidere: *almeno l'80 % degli errori correggibili da una lettura (8 su 10) con al più il 3 % dei link giusti cambiati*. Con i numeri attuali nessuna configurazione passa né il criterio vecchio né questo.

**Numeri attuali del lettore (sviluppo, MedMentions 19 errori / 590 link giusti, ambito centrato).** Tutte le prove: 2 errori trovati, 36 link giusti cambiati (6,1 %), AUC 0,60; con il filtro grammaticale nel reticolo 2 errori e 35 link giusti (5,9 %); il reticolo da solo agisce su 8 errori e 121 link giusti, con il filtro su 8 e 117. Il filtro toglie quindi 4 falsi allarmi su 590 senza perdere errori: un guadagno piccolo e reale, ma sullo stesso insieme su cui l'ho visto.

**Testa tipizzata da UMLS.** `UmlsHeadTyper` (`src/melampo/memory/head_typing.py`) chiede a UMLS i concetti che si chiamano esattamente come la parola-testa, ne mappa i tipi semantici sui tipi del reticolo e dà (tipo, quota): sotto la stessa quota minima della memoria (0,75) non agisce; se UMLS non risponde non indovina (conta le perdite). Entra solo per le parole che la memoria NCIt non conosce (`ChunkLattice(memory, typer=...)`). Il confronto con e senza UMLS **non è stato misurato**: serve la chiave UMLS e non è raggiungibile da qui. Si misura con il workflow `ci-probe` (`umls = true`): scrive nel rapporto il reticolo da solo per variante (memoria sola, con filtro grammaticale, con filtro e testa UMLS) e il lettore su ciascuno. Le tracce restano scritte con i tipi NCIt; sul lato lettura UMLS riempie i tipi mancanti.

**Filtro grammaticale.** `src/melampo/memory/grammar.py`: parole chiuse, avverbi, participi regolari e irregolari inglesi, con la lista dei nomi che somigliano a un participio (bed, hundred, family) e la regola che una parola nota alla memoria non viene mai filtrata. **Non è un analizzatore grammaticale**: dove serve un vero tagger ("-ing" nomi o verbi) c'è un punto solo dove collegarlo (`Grammar.verbal`).

**Lettore nel linker, solo traccia.** `AnatomyLinker(ci_reader=...)` scrive la lettura in `LinkResult.reading` ("decisione:lettura:margine") e nel flusso silenzioso `reader`; non cambia nessuna decisione (un test lo verifica sullo stesso caso con e senza lettore). `scripts/build_ci_resources.py` scrive le risorse (`space.npz`, `protos.npz`, `traces.json.gz`) e `ci_reader.load_reader` le rilegge. Resta spento perché non ha passato i criteri: serve a misurarlo sui referti veri quando c'è il gold set.

**Lettore acceso, con perimetro (decisione di Frank, 10 ottobre).** Il lettore può ora parlare nel linker, ma solo dove nessun altro metodo ha già risposto, e solo per la domanda che gli altri metodi non si pongono: *questa parola è, qui, un sito del corpo?* `AnatomyLinker(ci_reader=..., ci_reader_mode=...)` con tre modi: `trace` (scrive la lettura, silenzioso: il modo di prima), `record` (se legge la parola come "non un sito" scrive il conflitto `reader_reads_a_non_site` nel profilo, che il gold set calibrerà), `review` (e un link con al più `review_below` supporti va in coda di revisione, con il link che avrebbe fatto offerto al revisore). Non crea e non cambia mai un link: può abbassare la copertura, non la precisione. Il lettore **non è usato**: (a) dove il reticolo ha già agito sulla frase (ruolo, nome di altra cosa, due letture); (b) dove c'è un vicino di procedura ("biopsia epatica"), che ha la sua regola di ruolo; (c) per le forme ambigue con profilo di sensi, che leggono i sensi e, se richiesto, i modelli. Non parla neanche per i ruoli (sito di procedura, misura, dispositivo): sono convenzioni, e le decide la politica dei ruoli. Il codice di perimetro è `_another_method_decides`.

*Quanto vale, misurato (sviluppo: 19 + 11 errori, 590 + 1.315 link giusti).* Dove il reticolo tace, il lettore legge "non un sito" su 3 link giusti di MedMentions e su nessuno di CRAFT, e **su nessuno dei 30 errori**. È un limite superiore dei link giusti che andrebbero in revisione (0,16 % di 1.905) perché il conteggio non conosce i vicini di procedura e le forme ambigue. Il beneficio misurato è zero: i soli due errori che il lettore trovava (`inferior vena cava filter placement`) erano già del reticolo. Lo acceso così costa pochissimo e non può peggiorare la precisione, ma **oggi non migliora nulla di dimostrato**; il suo valore si misura sul gold set dei radiologi. Il default resta `trace`: la modalità si sceglie nell'istanza che carica le risorse del lettore (`build_ci_resources.py`, 50 MB circa, non nel repo).

**Aggiornamento dopo l'analisi "dove il lettore diverge dall'uomo" (10 ottobre, sera).** Vedi `lettore_kintsch_dove_diverge_dall_uomo_2026-10-10.md`. Tre meccanismi che questo documento indicava come mancanti (§3: regole come nodi, un senso per discorso, unità attraverso la sigla definita) sono ora nel lettore. Sullo sviluppo: errori letti come "non struttura" 2 → 6 su 19 (MedMentions), link giusti cambiati 36 → 38 su 590, di cui solo 4 non cambiati già dal reticolo (0,7 %); nel linker, dove nessun altro metodo decide, il lettore trova 2 errori su 19 (prima 0) con gli stessi 3 link giusti. Il caso del discorso è un documento solo e vale per proprietari di tipo "cibo". 22 dei 30 errori di sviluppo sono letti dal lettore come li leggerebbe un medico: il problema lì è l'etichetta.

**Italiano.** Vedi `supporto_italiano_piano_2026-10-10.md`: il meccanismo a testa sinistra c'è e ha i test; manca tutto il dato (memoria di blocchi, nomi lunghi, tracce, gold set italiano).

**Gold set.** Le schede chiedono anche ruolo e span; sessione di allineamento di 40 frasi dai corpora pubblici (`gold_set.py alignment`) e guida per i radiologi (`gold_set_guida_radiologi.md`).

## 7. Il modello che i testi indicano: un integratore a vincoli (E5, progetto)

Una rete per ogni menzione, costruita in modo esaustivo e risolta per soddisfacimento di vincoli (Kintsch 1998, cap. 4):

- **Nodi.**
  - Ogni lettura delle parole attorno alla menzione: i blocchi del reticolo, i nomi della memoria (NCIt, UMLS, `longer_names`) di ogni tipo, non solo anatomici.
  - I nodi del contesto: cornice dell'esame, tema del documento, sensi già fissati nel discorso.
  - I nodi di regola: lato, sistema, parti, tessuto contro spazio. Come nel problema di Linda, la conoscenza formale entra nella rete e non resta un filtro a posteriori.
- **Collegamenti.**
  - Positivi fra letture compatibili, con forza data dalla predicazione: vicini della parola che sono anche vicini dell'argomento, in uno spazio semantico del dominio (*gist*) più le co-occorrenze esplicite (Kintsch e Mangalath 2011).
  - Negativi fra letture alternative delle stesse parole, sulla sola base formale (Kintsch 1998, p. 96).
- **Integrazione.** A(t+1) = W·A(t), normalizzato al massimo, finché nessun valore cambia più di 0,001 (Kintsch 1998, pp. 98-99).
- **Decisione.**
  - Se vince la lettura anatomica con un margine, si collega.
  - Se vince un'altra lettura, si registra un ruolo o un non-sito.
  - Se non c'è un vincitore stabile, si ripropone con il candidato successivo (la revisione dell'esperto) e poi ci si astiene.

Perché questo e non un LLM più grande:

- È **deterministico** e **ispezionabile**: ogni decisione ha una rete che la spiega.
- Gli errori si correggono nella memoria (una voce, un peso), non riaddestrando.
- Usa l'LLM solo come proponente di letture nella fase di costruzione (Kintsch: la costruzione può essere "stupida", purché esaustiva), non come giudice.

## 8. Limiti

- 33 errori (22 in MedMentions; 3 nel test e 2 nello sviluppo). La classificazione è mia e va ripetuta da un radiologo; 4 etichette restano dubbie finché E1 non stampa i nomi UMLS.
- Articoli, non referti.
- Del libro di Kintsch ho letto i capitoli 4, 5.1, 7, 9.1 e 11.2, non tutto. Dell'articolo di Ericsson e Kintsch ho letto il rapporto tecnico del 1991 (riconoscimento ottico del testo), non la versione pubblicata nel 1995. Del lavoro di LREC 2026 solo il riassunto. Drew et al. 2013 non era leggibile e non è usato.
- Le prove sugli LLM con definizioni ed esempi riguardano il riconoscimento di entità, non il linking.
- E2 ed E4a usano UMLS, la stessa fonte delle etichette di MedMentions: provano l'ipotesi "è la memoria" nella forma più favorevole. Per il prodotto la memoria dovrà venire da fonti con licenza adatta al dispositivo e approvate dai radiologi.

## Fonti

- Kintsch W. *Comprehension: A paradigm for cognition*. Cambridge University Press 1998 (cap. 4, 5.1, 7, 9.1, 11.2).
- Kintsch W. Predication. *Cognitive Science* 25, 2001, 173-202.
- Kintsch W, Mangalath P. The construction of meaning. *Topics in Cognitive Science* 3, 2011, 346-370.
- Ericsson KA, Kintsch W. Memory in comprehension and problem solving: a long-term working memory. Institute of Cognitive Science, University of Colorado, Publication 91-13, 1991 (poi *Psychological Review* 102, 1995, 211-245).
- Gernsbacher MA, McKinney VM. Recensione di Kintsch 1998. *American Scientist* 87(6), 1999.
- Landauer TK, Dumais ST. A solution to Plato's problem: the latent semantic analysis theory of acquisition, induction, and representation of knowledge. *Psychological Review* 104, 1997, 211-240.
- Sweller J. Cognitive load theory and individual differences. *Learning and Individual Differences* 110, 2024, 102423.
- Sanchez CA, Wiley J. An examination of the seductive details effect in terms of working memory capacity. *Memory & Cognition* 34(2), 2006, 344-355.
- Mohan S, Li D. MedMentions. arXiv 1902.09476. https://arxiv.org/pdf/1902.09476
- Kuperberg GR, Jaeger TF. What do we mean by prediction in language comprehension? *Language, Cognition and Neuroscience* 31, 2016. https://kuperberg.mgh.harvard.edu/wp-content/uploads/kuperbergjaeger_lcn_15.pdf
- Zwaan RA, Radvansky GA. Situation models in language comprehension and memory. *Psychological Bulletin* 123, 1998. https://sites.ualberta.ca/~dmiall/Cognitive/Readings/Zwaan_Radvansky_1998.pdf
- Gale W, Church K, Yarowsky D. One sense per discourse. 1992. https://preview.aclanthology.org/fix_video/H92-1044.pdf
- Stanford Encyclopedia of Philosophy, Hermeneutics. https://plato.stanford.edu/entries/hermeneutics/
- Evans et al. *Cognitive Research* 2021. https://link.springer.com/article/10.1186/s41235-021-00339-5
- Profiling hallucinations in frontier LLMs for entity linking to medical ontologies. LREC 2026, workshop ClinicalNLP. https://lrec.elra.info/lrec2026-ws-clinicalnlp-41
- Medical entity linking in low-resource settings with fine-tuning-free LLMs. *Stud Health Technol Inform* 2026. https://journals.sagepub.com/doi/10.3233/SHTI251402
- On-the-fly definition augmentation of LLMs for biomedical NER. arXiv 2404.00152. https://arxiv.org/html/2404.00152v2
- Sistema francese per la sfida EvalLLM 2025. arXiv 2510.03577. https://arxiv.org/html/2510.03577v1
- Gutiérrez BJ et al. Findings of EMNLP 2022. https://preview.aclanthology.org/author-url/2022.findings-emnlp.329/
- Kim E et al. Correlated errors in large language models. ICML 2025. https://arxiv.org/html/2506.07962v1
