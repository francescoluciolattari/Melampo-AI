# Come il cervello isola il sintagma, e che modello ne deriva per il linker — 2026-10-09

Domanda di Frank: il linker deve leggere il sintagma intero, in modo il più possibile univoco come nel pensiero umano. Come lo isola il cervello? Con un modello parallelo a grafo, a strati paralleli? Leggere i lavori più recenti di neurobiologia, neurofisiologia e filosofia della mente e della comprensione, e cercare o modificare modelli di IA innovativi.

Convenzione di questo documento: **[letto]** = ho letto il testo della fonte; **[abstract]** = ho letto solo il riassunto; **[inferenza]** = trasposizione mia al linker, da misurare. Le fonti PMC dello studio di Nelson non erano raggiungibili da PMC: i numeri vengono dalle copie del gruppo di ricerca.

## 1. Che cosa ho detto nella tabella a tre livelli, e che cosa vale

La tabella (§22 dell'architettura) era una scelta di ingegneria, non un modello del cervello: metteva i controlli economici sempre accesi, i modelli di sintagma (GLiNER, SapBERT) solo quando la menzione sta dentro un sintagma più lungo, e l'LLM solo quando i due livelli sotto non concordano. L'ultima frase ("l'incertezza che apre il livello 3 riguarda la lettura del sintagma, non il link") voleva dire: l'LLM non si chiama quando il *link* è incerto (gli errori non lo sembrano: un cancello su conflitto e convergenza si apre su 10 errori su 33), ma quando i *lettori del sintagma* si contraddicono sul tipo di cosa è il sintagma.

Il difetto della tabella è che i flussi restano **lettori di parole che votano su un link**. Il cervello non fa così: prima costruisce l'unità, poi decide che cosa farne. Il resto del documento ridisegna il linker su questa osservazione.

## 2. Che cosa dicono le fonti sul modo in cui il cervello isola il sintagma

**2.1 Il sintagma è costruito dalla conoscenza, non letto dal segnale.** In un esperimento MEG con sillabe a 4 Hz senza alcun indizio acustico di struttura, la corteccia segue la sillaba (4 Hz), il sintagma (2 Hz) e la frase (1 Hz). Le sillabe in ordine casuale mostrano solo i 4 Hz, e ascoltatori che non conoscono la lingua mostrano solo i 4 Hz. Gli autori concludono che il tracciamento di sintagmi e frasi non si spiega con acustica, prosodia o probabilità di transizione, ma dipende dalla comprensione e dalla conoscenza grammaticale ([Ding et al., Nature Neuroscience 2016](https://doi.org/10.1038/nn.4186)) **[letto]**.

**2.2 Si costruisce parola per parola, e si chiude quando le parole possono fondersi.** In elettrodi intracranici su 12 pazienti (721 elettrodi) l'attività di alta frequenza cresce a ogni parola e **cala di colpo quando le parole possono fondersi in un sintagma**; segue il numero di nodi aperti. I modelli di analisi dal basso e "left-corner" spiegano i dati meglio di quello dall'alto; il numero di nodi aperti correla con la lunghezza della pila del parser (r = 0,97). Le regioni sono temporale anteriore e posteriore, polo temporale e giro frontale inferiore ([Nelson et al., PNAS 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5422821/)) **[letto, copia del gruppo]**. Il lavoro non ha testato frasi senza senso (Jabberwocky), quindi non separa sintassi e semantica.

**2.3 La combinazione è locale, rapida e prima di tutto di significato.** Nella rete del linguaggio la risposta resta alta se si mescolano le parole in modo locale, purché le parole combinabili restino vicine, e crolla se le parole combinabili vengono spinte a circa otto parole di distanza (finestra di integrazione stimata 5–7 parole); l'ordine delle parole non è necessario ([Mollica et al., Neurobiology of Language 2020](https://www.mit.edu/~hopekean/files/composition.pdf)) **[letto]**. Nel lobo temporale anteriore sinistro, 200–250 ms dopo l'inizio del nome, l'attività sale quando il nome può combinarsi con la parola precedente, e sale di più se la prima parola restringe i possibili riferimenti ("Indian food" contro "Asian food"); la risposta è concettuale, non strettamente sintattica, e la corteccia ventromediale prefrontale segue circa 200 ms dopo ([Pylkkänen, Science 2019](https://acesin.letras.ufrj.br/wp-content/uploads/2023/08/pylkkanen_science_2019.pdf)) **[letto]**. L'autrice lascia aperto se il cervello costruisca struttura sintattica online.

**2.4 Una parte dei sintagmi non si costruisce: si richiama dalla memoria.** Il modello Memoria–Unificazione–Controllo: la memoria (temporale e angolare) conserva forme di parola e *cornici sintattiche* legate alle parole; l'unificazione (giro frontale inferiore sinistro) assembla i pezzi in strutture più grandi, con competizione e selezione tra significati candidati; il controllo gestisce attenzione e selezione ([Hagoort, Frontiers in Psychology 2013](https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00416/full)) **[letto]**. Sulle espressioni di più parole: per le espressioni frequenti conta la frequenza dell'espressione intera oltre a quella delle parole, quindi sono in parte conservate e riusate in blocco; per quelle nuove si compongono con regole astratte; gli autori propongono un passaggio graduale legato alla frequenza, ma hanno provato solo gli estremi ([Morgan e Levy, Cognition 2016](https://www.mit.edu/~rplevy/papers/morgan-levy-2016-cognition.pdf)) **[letto]**. Per la filosofia, Jackendoff stima circa 25.000 idiomi inglesi, e se trattarli come voci di lessico sia una difesa legittima della composizionalità è controverso ([Stanford Encyclopedia, Compositionality §4.2.2](https://plato.stanford.edu/entries/compositionality/)) **[letto]**.

**2.5 Il cervello raggruppa subito, a più livelli, e non tiene aperte molte interpretazioni.** Teoria "Now-or-Never / Chunk-and-Pass": l'input è fugace, quindi viene ricodificato subito in blocchi astratti con perdita di dettaglio; i blocchi salgono di livello (suono, parola, sintagma, discorso), ogni livello ha una finestra più lunga e tiene poche unità; la previsione guida la ricodifica; l'elaborazione è incrementale. Gli autori scartano i modelli che tengono aperte molte interpretazioni complete in parallelo e quelli che accumulano costituenti incompleti fino a fine frase: **il parallelismo ammesso riguarda le decisioni di raggruppamento**, e l'ambiguità rimasta si lascia sottospecificata ("principio del minimo impegno") finché l'input successivo non la risolve, con una pressione a "giusto la prima volta" ([Christiansen e Chater, Behavioral and Brain Sciences 2016](https://csl-lab.psych.cornell.edu/files/2021/02/2016-cc-BBS.pdf)) **[letto]**.

**2.6 Un modello computazionale che riproduce il 4–2–1 Hz.** DORA ha unità per parole, sintagmi e frasi; i livelli alti si accendono quando le unità sotto si accendono vicine nel tempo; ogni unità ha un inibitore e i componenti di un sintagma si accendono a turno (asincronia), così non si perde l'identità delle parole. Riproduce 4, 2 e 1 Hz; una rete ricorrente addestrata sugli stessi stimoli non li mostra. Gli autori dicono che è un modello di *rappresentazione*, non di analisi sintattica, con stimoli limitati e apprendimento assente ([Martin e Doumas, PLOS Biology 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5333798)) **[letto]**.

**2.7 Il cervello prevede su più livelli e a più lunga portata dei modelli.** Aggiungere previsioni a lungo raggio a GPT-2 migliora la corrispondenza con l'attività cerebrale, e la corteccia fronto-parietale prevede contenuti più astratti e più lontani ([Caucheteux, Gramfort e King, Nature Human Behaviour 2023](https://arxiv.org/abs/2111.14232)) **[già in uso nel progetto, comprensione_umana]**.

**2.8 Filosofia della comprensione.** Due tradizioni che sembrano opposte descrivono lo stesso fatto. Il principio del contesto di Frege (una parola ha significato nel contesto della frase) e la composizionalità (il tutto è determinato dalle parti) non si contraddicono se la determinazione corre nei due sensi: dal basso le parti, dall'alto il contesto ([Stanford Encyclopedia, Compositionality §2.4](https://plato.stanford.edu/entries/compositionality/)) **[letto]**. Il circolo ermeneutico (Schleiermacher, Dilthey) dice che si capisce il tutto dalle parti e le parti dal tutto, e Gadamer descrive la comprensione come un ciclo di proiezioni di interpretazioni che vengono superate fino a quando l'interpretazione diventa sufficiente, a partire da "pre-giudizi" che non si possono rendere del tutto espliciti ([Stanford Encyclopedia, Hermeneutics §1.3, §4.1](https://plato.stanford.edu/entries/hermeneutics/)) **[letto]**. Millière e Buckner sostengono che il successo dei modelli di linguaggio mette in discussione assunzioni consolidate sulle reti neurali e chiedono metodi empirici per guardare dentro ([A Philosophical Introduction to Language Models](https://arxiv.org/abs/2401.03910)) **[abstract]**.

## 3. Che cosa se ne ricava: la risposta alla domanda "grafo parallelo o strati paralleli?"

Le fonti non dicono né l'uno né l'altro in forma pura. Dicono questo, con i gradi di certezza della legenda:

1. **Unità prima del giudizio.** Il sintagma è un oggetto costruito (2.1, 2.2), non un'etichetta data a una parola. Un linker che decide sulla parola e poi controlla il contorno inverte l'ordine. **[fonti 2.1–2.2 + inferenza]**
2. **Due sorgenti di unità: memoria e composizione.** I blocchi noti si richiamano in blocco (memoria: nomi di ontologia, entità note, espressioni frequenti), quelli nuovi si compongono se le parole *possono* combinarsi e sono vicine (unificazione, 5–7 parole). **[fonti 2.3–2.4 + inferenza]**
3. **Si chiude quando non si può più fondere.** La chiusura è un evento della costruzione, non una lista di parole di arresto. **[fonte 2.2 + inferenza]**
4. **Parallelismo limitato alle scelte di raggruppamento, a più livelli con finestre crescenti.** Non si tengono in parallelo interpretazioni complete: si tengono in parallelo *segmentazioni candidate* e si impegna il minimo. **[fonte 2.5 + inferenza]**
5. **Il significato scende dall'alto.** Il contesto superiore (tipo di documento, entità vicine) vincola il raggruppamento sotto (2.5, 2.8). **[letto + inferenza]**

Da qui l'architettura proposta, **un reticolo di blocchi** (grafo) **a livelli con finestre crescenti**:

- **Nodi.** Ogni gruppo contiguo di parole che potrebbe essere un blocco (come una cella di un parser a tabella) è un nodo con un tipo (struttura, procedura, dispositivo, proteina, cibo, processo…).
- **Origine di un nodo.** (a) *Memoria*: il gruppo è un nome di un'ontologia con tipo (UBERON, NCIt, Protein Ontology), un'entità nota o un'espressione frequente nel corpus; (b) *composizione*: modificatore + testa i cui tipi sono combinabili secondo le classi NCIt (struttura + procedura → "procedura su quella struttura"); (c) *modello appreso*: un segmentatore addestrato (§4).
- **Archi.** Segmentazioni che coprono tutta la frase senza sovrapporsi. "inferior vena cava filter placement" ha almeno due: [inferior vena cava] [filter placement] e [inferior vena cava filter placement].
- **Punteggio di una segmentazione.** Somma di: corrispondenza a blocchi noti, coesione interna del blocco (§4.2), combinabilità dei tipi, accordo con il livello superiore. La scelta è un cammino di costo minimo sul reticolo (programmazione dinamica), quindi deterministica e ispezionabile.
- **Impegno minimo.** Se le due migliori segmentazioni sono troppo vicine, il sintagma resta *sottospecificato* e il linker si astiene o registra il ruolo meno impegnativo, come già fa con `inherent_location`.
- **Poi il link.** Solo dopo che il blocco è fissato: se la testa del blocco che contiene la menzione è una struttura, si collega; se è una procedura, un dispositivo, un processo, un nome di proteina, la struttura prende il ruolo corrispondente (`procedure_site`, `device_site`, `inherent_location`, `inside_a_name`) o non è corpo (`not_a_body_site`).

I **flussi** della tabella a tre livelli (nomi lunghi, GLiNER, recupero SapBERT, LLM) restano, ma cambiano ruolo: non votano sul link, **propongono nodi e punteggi** per il reticolo. Nel quadro del cervello, GLiNER e il recupero fanno la parte della memoria, l'LLM la parte del controllo quando due segmentazioni sono vicine.

**Cosa non copio.** Il cervello sbaglia: i "garden path" nascono dall'impegno precoce (2.5), l'83% dei radiologi non ha visto il gorilla (comprensione_umana). Il reticolo tiene più segmentazioni proprio perché il nostro obiettivo è meno errori del cervello; ma il numero di segmentazioni va limitato (finestre di 5–7 parole) per non esplodere.

## 4. Modelli di IA da cercare, modificare o prendere in prestito

| Modello | Che cosa fa | Che cosa ne prendiamo | Limite per noi |
|---|---|---|---|
| **H-Net, taglio dinamico** ([Hwang, Wang e Gu, arXiv 2507.07955](https://arxiv.org/abs/2507.07955)) **[letto]** | Impara i confini dei blocchi dal solo compito di prevedere: la probabilità di confine è la dissimilarità (coseno) fra la rappresentazione di una posizione e quella della precedente; più stadi annidati. A 2 stadi forma parole e poi unità più ampie, "anche sintagmi di più parole" (evidenza qualitativa) | L'idea di **confine = salto fra stati adiacenti**, e di **più livelli di blocchi**; si può misurare su un encoder biomedico congelato senza addestrarlo | Modello di linguaggio a byte, testato fino a 1,3 miliardi di parametri; nessuna misura quantitativa di allineamento con parole o morfemi nel testo; non è un classificatore |
| **EM-LLM, segmentazione per sorpresa** ([Fountas et al., ICLR 2025](https://arxiv.org/abs/2407.09450)) **[letto]** | Un confine di evento si apre dove la sorpresa (log-verosimiglianza negativa) supera una soglia dalla media recente; poi un affinamento con modularità sul grafo delle similarità; i confini somigliano a quelli degli annotatori umani sui testi | **Sorpresa come segnale di confine** (prima regola, economica) e **affinamento per modularità** dei confini sul grafo delle parole | Pensato per memoria a lungo contesto, non per sintagmi; confini di eventi di testo lungo, non di sintagmi nominali |
| **Chunking per previsione reciproca** ([Asabuki, Hiratani, Fukai, PLOS CB 2018](https://www.biorxiv.org/content/10.1101/215392v1)) **[letto]** | Due reti si insegnano a vicenda senza etichette e imparano blocchi di lettere ricorrenti; la coesione viene dalla ricorrenza | Blocchi imparati da **coesione statistica** senza supervisione | Giocattolo: sequenze di lettere, blocchi piatti, numero di blocchi fissato in anticipo |
| **DORA** (Martin e Doumas 2017) **[letto]** | Rappresentazione di parole, sintagmi, frasi per asincronia | Una **prova di principio** che la struttura gerarchica serve più della sola serialità (RNN senza) | Non analizza; stimoli molto limitati |
| **Binding problem** ([Greff, van Steenkiste, Schmidhuber 2020](https://arxiv.org/abs/2012.05208)) **[abstract]** | Distingue segregazione, rappresentazione e composizione; le reti standard non legano in modo flessibile | Il **vocabolario** per progettare: nodo = segregazione, tipo = rappresentazione, combinabilità = composizione | Il testo non nomina qui le soluzioni; da leggere per esteso |
| **Riconoscitore con tipo (GLiNER-BioMed)** | Dà intervallo e tipo in linguaggio naturale | Fa la parte "memoria con tipo" per forme non elencate | Solo inglese, F1 medio 56,9: un segnale, non un giudice |

### 4.1 Una proposta nuova da provare: il segmentatore appreso sul modello di MedMentions

MedMentions è, di fatto, un corpus di **confini di blocco**: oltre 350.000 menzioni, ciascuna il concetto più specifico senza sovrapposizioni (da [Mohan e Li](https://arxiv.org/abs/1902.09476)). Quel che serve al linker è esattamente un modello che dica, data una parola anatomica, **se è il blocco intero o un pezzo di un blocco più lungo**. Si può addestrare un piccolo segmentatore sulla parte `trng` dell'insieme, tarare su `dev`, e lasciare `test` congelato. È la versione "impara a elaborare" di Christiansen e Chater (apprendimento online e locale). Rischi dichiarati: l'etichetta di MedMentions segue la sua convenzione (§5 dell'analisi), per cui le uscite vanno lette come "è dentro un blocco più lungo", non come "quel blocco è di quel tipo"; il test di MedMentions non serve più come misura indipendente se si addestra sulla stessa distribuzione, quindi la certificazione resta sui radiologi.

### 4.2 Segnali di coesione e di confine da misurare subito (senza addestrare)

Tutti si calcolano con un modello già pubblico sulla CPU di GitHub:
1. **Sorpresa dopo la menzione**: dopo la parola anatomica, quanto è sorprendente la parola successiva per un modello causale biomedico? Alta sorpresa = confine (EM-LLM).
2. **Coesione** del gruppo: log P(gruppo) meno la somma dei log P(parola) (informazione mutua puntuale), sul modello; è la versione misurabile della frequenza del blocco intero oltre le parti (Morgan e Levy) e della prossimità fra parole combinabili (Mollica).
3. **Salto fra stati adiacenti** (H-Net): coseno fra gli stati nascosti di parole vicine in un encoder biomedico congelato.
4. **Combinabilità per tipo**: tipo della testa dalle classi NCIt e tipo del modificatore (già in `phrase_probe`, braccio `head`, con i suoi limiti: la parola finale da sola segnala 472 link giusti).

## 5. Piano a tappe, ciascuna con la sua misura

| Tappa | Che cosa | Misura | Dove gira |
|---|---|---|---|
| A | Aggiungere a `phrase-probe` i bracci 4.2: sorpresa, coesione, salto fra stati | AUC e punti di lavoro come per gli altri segnali; quanti dei 33 errori separano senza perdere link giusti | GitHub (Hugging Face) |
| B (fatta, §8) | Reticolo di blocchi **deterministico**: nodi dalla memoria (NCIt, UBERON, Protein Ontology, lessico) e dalla composizione per tipo, cammino minimo; nessun modello | Errori fermati e link giusti persi sui link giudicati, sulle held-out, sui referti reali (3.969 casi) | Qui |
| C | Aggiungere i punteggi dei modelli (GLiNER, SapBERT, A) come pesi dei nodi; l'LLM solo come tie-break tra segmentazioni vicine | Come B, con ablazione per braccio | GitHub, poi qui |
| D | Segmentatore appreso su `trng` di MedMentions (§4.1) | Su `dev`; `test` aperto una volta sola, a decisione presa | GitHub (GPU o CPU piccola) |
| E | Ruoli nel controllo esterno e nel protocollo dei radiologi | Percentuale di link con ruolo corretto, non solo etichetta | Qui + radiologi |

Il rapporto di `phrase-probe` è arrivato (§7.1): i segnali "parola vicina" non separano (AUC 0,50–0,60) e la memoria di NCIt non copre i 13 errori a span lungo; B va fatto con una memoria più grande e con la metrica del ruolo.

## 6. Limiti onesti (rivisti dopo la ricerca del §7)

- **Prove neurofisiologiche.** Ding 2016 è MEG su volontari sani (ascoltatori di cinese e di inglese); solo Nelson 2017 usa elettrodi intracranici in pazienti con epilessia. In entrambi i casi sono frasi di lingua comune, non sintagmi nominali di un referto: la trasposizione resta **inferenza**. Una ricerca mirata (§7.4) non ha trovato nessuno studio sulla lettura di sintagmi di referto; esistono prove su esperti (memoria di lavoro a lungo termine, conoscenza incapsulata, percezione olistica del radiologo) che sostengono i blocchi di dominio ma non sono sul testo.
- DORA e l'asincronia sono un modello di rappresentazione, non un parser; H-Net è stato provato su byte fino a 1,3 miliardi di parametri; EM-LLM e il chunking per previsione reciproca non sono nati per sintagmi nominali biomedici.
- **Nessun modello pronto isola e tipizza con precisione certificabile.** Soffitti trovati: riconoscimento di UBERON in CRAFT circa F1 0,82; MedMentions, baseline TaggerOne F1 0,453; GLiNER-BioMed F1 56,9 su 8 insiemi; RadGraph F1 di entità 0,94/0,905 ma in-dominio (inglese, accesso con credenziali). La precisione certificabile non può venire da un modello quasi perfetto: viene dall'accettazione selettiva con insiemi calibrati (§7.3).
- **Millière e Buckner** ora letti per esteso (§7.5): sonde e attenzione non sono spiegazioni; per attribuire una struttura interna servono interventi causali; l'evidenza solo comportamentale non basta. Per noi: certificazione comportamentale più prove a coppie minime, non "feature" di attenzione.
- Il parallelismo "a grafo" qui è un parallelismo di **segmentazioni candidate**; Christiansen e Chater sono espliciti nel non ammettere parallelismo di interpretazioni complete.
- Il reticolo (tappa B) è implementato e misurato nel §8; le sue letture non decidono nulla finché i ruoli non sono confrontati con il gold set dei radiologi.

## 7. Risultati reali di `phrase-probe` e ricerca per superare i tre limiti

### 7.1 Che cosa ha misurato il run di Frank (2.058 link giudicati: 22 errori MedMentions su 732, 11 CRAFT su 1.326)

Nessun lettore zero-shot separa gli errori dai link giusti (AUC 0,50–0,60; il migliore, i voti economici, 0,60 e in CRAFT sotto 0,5). Il braccio del parser è caduto per un modulo mancante (`click`): corretto nel workflow, da rilanciare.

- **GLiNER-BioMed tipizza la parola anatomica stessa come "anatomia" in 26 errori su 33**: legge la parola, non il sintagma che la contiene (inferior vena cava *filter placement*).
- **L'unione economica** (oggetto "X of <oggetto>", un senso per discorso, refuso, GLiNER ≥ 0,5) ferma **9 errori MedMentions su 22 al prezzo di 65 link giusti su 710**; oggetto+discorso+refuso fermano 4 errori a costo zero. I 65 sono quasi tutti `brain` (31), `liver` (13), `heart`, `prostate`: casi di convenzione (attivazione cerebrale, peso del cervello) che andrebbero **registrati con un ruolo**, non persi. Perciò la misura del prodotto è la **correttezza del ruolo**, non il punto di lavoro link/veto.
- **Il cancello "solo quando incerto"** si apre su 10 errori su 33 (conflitto o convergenza < 3): gli errori non sembrano incerti.
- Dei 22 errori MedMentions, **9 non sono un problema di isolamento** (lo span dell'annotatore coincide con la menzione: mappatura, convenzione o tipo) e **13 hanno uno span dell'annotatore più lungo**. **NCIt non contiene nessuno di questi 13 come nome**: la memoria di NCIt da sola non li ripara; servono composizione o una memoria più grande.
- Un lessico di teste derivato dai nomi NCIt che contengono un tratto anatomico (ultima parola, ≥ 5 nomi, ≥ 90 % non-sito) è troppo rumoroso: 2/22 errori fermati con 48/710 link giusti segnalati. Va curato, come si è fatto con i GO per `process_heads`.

### 7.2 Convenzione bersaglio: RadGraph, non MedMentions né CRAFT

Lo scopo del prodotto è dire in quale struttura sta il reperto. La convenzione più vicina è RadGraph: entità Anatomia e Osservazione con relazioni Located_At, Modify, Suggestive_Of. Dal punto di vista del linker: lo span utile è la struttura con il suo modificatore e la relazione che la lega al reperto. RadGraph/RadGraph-XL sono in-dominio (referti radiologici in inglese) ma l'accesso è con credenziali PhysioNet e non è in italiano: serve come **banco di prova della convenzione**, non come modello da accendere. Per i ruoli, la post-coordinazione di SNOMED CT offre un vocabolario già normato (sito, procedura, morfologia) a cui mappare `inherent_location` e `procedure_site`.

### 7.3 Come ottenere precisione certificabile senza un modello perfetto

Si certifica la **procedura di accettazione**, non il modello. Previsione conforme per NER (arXiv 2601.16999) e Learn-then-Test: si fissa la soglia su un insieme di calibrazione in modo che il tasso di errore tra i link *accettati* sia ≤ 1 % con confidenza 95 % (Clopper-Pearson: servono almeno 299 link accettati senza errori). Applicato al reticolo: l'insieme conforme è l'insieme di segmentazioni vicine al costo minimo; se contiene più di un tipo/ruolo, il sistema si astiene o registra un ruolo. Il gold dei radiologi (due lettori ciechi più aggiudicatore) deve quindi contenere **span e ruolo**, non solo il link.

### 7.4 Il ponte cervello → referto

Per gli esperti, le fonti sostengono blocchi di dominio: memoria di lavoro a lungo termine (Ericsson e Kintsch), conoscenza biomedica incapsulata (Boshuizen e Schmidt), percezione olistica del radiologo in circa 200 ms. Nessuno studio diretto sulla lettura dei sintagmi nominali di un referto: la trasposizione si **misura** con il protocollo dei radiologi, aggiungendo il compito span+ruolo.

### 7.5 Che cosa cambia in pratica

1. Metrica: correttezza del ruolo e dello span, con link giusto solo se lo span è giusto.
2. Tappa A (bracci sorpresa/coesione/salto fra stati) resta utile, ma gli esiti del run dicono di non aspettarsi molto dai segnali "parola vicina".
3. Tappa B (reticolo deterministico) richiede una memoria più grande di NCIt per i 13 casi a span lungo: candidati da valutare (licenza e copertura da verificare), non assunti.
4. Prove a coppie minime (metamorphic) nel banco di verifica: cambiare l'oggetto del sintagma deve cambiare il ruolo e non il resto.
5. Lessico di teste: curato prima di entrare in un flusso.

## 8. Il reticolo di blocchi, realizzato e misurato (tappa B)

### 8.1 Che cosa c'è

- `src/melampo/memory/chunk_lattice.py`: il lettore. Nessun modello, nessuna rete, deterministico e ispezionabile.
- `data/linking/block_memory.json`: la memoria, costruita da NCIt con `scripts/build_block_memory.py` (tipo preso dalla classe dell'ontologia, mai da elenchi di parole): 10.433 parole-testa con il tipo dominante dei nomi che finiscono con quella parola (quota e numero) e 8.771 nomi NCIt di altro tipo che contengono una struttura (procedura, dispositivo, processo, proprietà, malattia…), quelli non già dati dall'ultima parola. Le proteine e i geni sono in `longer_names.json` (6.705 nomi). Non si contano item di questionari e righe di standard (CDISC), e una parola il cui tipo coincide con un simbolo di gene o con una sostanza ("scar", "air") non prende il tipo da lì.
- `scripts/lattice_probe.py` e il workflow manuale `lattice-probe`: rileggono con il reticolo ogni link giudicato del controllo esterno.
- `AnatomyLinker(chunk_lattice=...)`, spento di default: scrive la lettura nella traccia (flusso `blocks`, silenzioso) e in `LinkResult.block`; non decide nulla. `external_check.py --blocks` la mette nelle righe.

### 8.2 Come legge (le cinque idee del §3, nell'ordine)

1. **Finestra.** Il sintagma nominale attorno alla menzione: a sinistra fino a 5 parole, a destra fino a 7 (la finestra di integrazione di 5–7 parole di Mollica); la chiudono punteggiatura, parole funzionali, un elenco chiuso di verbi da referto ("appears", "measuring") e i numeri.
2. **Nodi.** La menzione è un blocco di memoria (una struttura); un nome noto è un blocco di memoria con il suo tipo; modificatori + testa è un blocco composto (le composizioni inglesi hanno la testa a destra: il tipo è quello dell'ultima parola); una parola sola è l'ultima risorsa.
3. **Costi (per principio, non adattati ai dati):** memoria 0,6; composto 1,0 + 0,1 per parola + 1,0 × (1 − quota del tipo della testa); parola sola 1,2. **Nodo aperto:** una testa di tipo non anatomico dentro il blocco, che avrebbe potuto chiuderlo, costa 2,0 (Nelson et al.: un nodo resta aperto finché le parole non si fondono): "brain weight strains" è [brain weight] + strains, non un blocco con testa "strains". Una testa anatomica ("cell", "tissue") non chiude: "spleen cells transfusion" arriva a "transfusion".
4. **Cammino di costo minimo** (programmazione dinamica) sulla finestra, con il blocco che contiene la menzione. **Minimo impegno:** se la migliore lettura alternativa che mette la menzione in una classe di decisione diversa (collegamento, ruolo, non-sito) è entro 0,15, il sintagma resta *sottospecificato* e non si decide.
5. **Poi la decisione**, solo a blocco fissato: struttura/malattia → collegamento; procedura, dispositivo, processo, misura → la struttura si tiene con un ruolo; molecola composta ("liver extracts") → ruolo `source_of` (nuovo, da registrare, mai un veto); nome noto di una molecola o oggetto ("heart of Maroilles cheese", "Liver Fatty Acid Binding Protein") → non è un sito. Un secondo livello unisce blocchi con "of": "removal of the gallbladder" → `procedure_site`.

Tipi che agiscono solo tramite memoria o "of", mai per composizione: cibo, organismo, concettuale. NCIt mette sotto "cibo" i nutrienti ("sterol", "glutamate") e sotto "organismo" ogni "Whole …": una parola-testa è un segnale troppo grezzo per dire "non è un sito". Costo pagato: "colon microbiome" non si legge più.

### 8.3 Misure (CRAFT e MedMentions; 2.058 link giudicati, 1.836 link MedMentions con tipo semantico dell'etichetta, 4.841 menzioni di referti reali iu-xray)

**Link giudicati** (le regole sono state scritte dopo aver letto i 33 errori del primo controllo: questa parte è ottimistica):

| corpus | errori | letti come ruolo / non-sito | link giusti: con ruolo / persi | tasso d'errore dei link semplici |
|---|---|---|---|---|
| CRAFT | 11 | 1 / 0 | 227 / 0 | 0,83 % → 0,91 % |
| MedMentions | 22 | 9 / 1 | 160 / 1 | 3,01 % → 2,14 % |

In CRAFT gli errori sono quasi tutti di mappatura ("right middle lobe", "bladder"): il reticolo non li tocca, come ci si aspetta. In MedMentions il ruolo coincide con quello dell'etichetta per 3 dei 10 errori letti; negli altri c'è un ruolo diverso, ma comunque il link non è più "semplice".

**Ruoli contro il tipo semantico dell'etichetta** (link che il reticolo non ha visto quando si scrivevano le regole):

| etichetta | n | stesso ruolo |
|---|---|---|
| procedura (`procedure_site`) | 180 | 123 (68 %) |
| misura, processo (`inherent_location`) | 18 | 15 (83 %) |
| dispositivo | 11 | 3 |
| non è un sito | 18 | 1 |
| struttura / malattia: nessun ruolo aggiunto | 851 / 755 | 650 / 636 (76 % / 84 %) |

Convenzioni del progetto: misura per immagini ("brain volume") 13 su 13 `inherent_location`; origine di cellule 7 su 27 con ruolo; sito di un dispositivo 5 su 11 con ruolo (3 `device_site`).

**Stabilità.** Con il costo del nodo aperto ≥ 2 il risultato non dipende dagli altri parametri: 11 errori letti e da 1 a 6 link giusti persi su 2.025. Con costo 1 il blocco ingloba le teste dopo la prima e si perdono da 38 a 53 link giusti. Il costo 2 è quindi una condizione, non una taratura fine.

**Referti reali** (iu-xray, 4.841 menzioni proposte): 61 % collegamento semplice; 37,5 % misura (`inherent_location`, quasi sempre "heart size", "cardiac size"); procedure 0,8 %; dispositivi 0,7 %. Ho letto a mano 42 letture non banali: 26 plausibili (62 %). Misure 9 su 14, procedure 6 su 14, dispositivi 11 su 14. Gli errori hanno una causa comune: **la parola-testa non è un nome o il suo tipo NCIt è idiosincratico** — aggettivi ("focal", "vascular"), nomi con senso diverso ("appearance", "presence", "base", "junction" tipizzati come procedura o dispositivo), verbi non in elenco ("suggesting").

### 8.4 Che cosa dice questo sul progetto

- Il reticolo fa quello che doveva fare **come lettore**: costruisce l'unità prima, poi decide; la scelta del ruolo dipende dal blocco, non dalla parola vicina. Dove l'etichetta ha un ruolo di misura o procedura il ruolo è giusto 70–80 % delle volte, con regole che non ho adattato su quei link.
- **Non è ancora un decisore.** Un ruolo giusto nel 60–80 % dei casi non basta per essere scritto in un referto: resta nella traccia, spento di default, e va confrontato con il gold set dei radiologi (con il compito span + ruolo del §7.4).
- **Il limite è la memoria, non l'algoritmo.** I 13 errori a span lungo non sono in NCIt come nomi, e una testa derivata da NCIt sbaglia sui nomi comuni. Ci sono tre strade, da misurare nell'ordine: (1) lessico di teste curato (una lista breve di nomi di procedura, dispositivo e misura approvata dai radiologi, che sostituisce quella derivata per i tipi che agiscono); (2) un'etichettatura grammaticale per escludere aggettivi e verbi dalle teste; (3) il segmentatore appreso del §4.1, che porta la conoscenza dei sintagmi che NCIt non ha. Nessuna delle tre è in questa consegna.
- **Italiano:** non ancora. In italiano la testa viene prima ("filtro della vena cava inferiore") e i tipi dei nomi sono in italiano: serve un lessico di teste italiano e una regola di direzione per lingua.
- Un senso per discorso resta nel braccio `discourse` di `phrase-probe` (serve il documento intero); i due "heart" del formaggio che il reticolo non vede si fermano con quello.

### 8.5 Il segmentatore LLM misurato (`phrase-probe`, `llm=sample`, zip di Frank del 9 ottobre 2026)

Prova: Nemotron 3 Super 120B e Gemma 3 27B ricevono la frase, la dividono in concetti e dicono quale contiene la menzione, di che tipo è e se è una parte del corpo. Campione: tutti i 33 errori + 300 link giusti a caso (331 casi letti; 22 errori MedMentions, 11 CRAFT, 114 e 184 link giusti). Criteri fissati prima di guardare: almeno 12 errori MedMentions su 22 trovati perdendo al più il 3 % dei link giusti, oppure ruolo giusto almeno nel 70 % dei casi.

| misura | risultato |
|---|---|
| errori MedMentions letti come "sintagma più lungo che non è la struttura" | 10 su 22 (9 se i due modelli concordano) |
| errori CRAFT letti | 0 su 11 |
| link giusti segnalati | 33 su 114 MedMentions (29 %), 50 su 184 CRAFT (27 %); al punto di lavoro del rapporto: 9 errori contro 35 link giusti persi |
| AUC del segnale | 0,53 (MedMentions 0,61; CRAFT 0,36) |
| ruolo uguale a quello dell'etichetta | 3 su 12 (25 %) |
| i due modelli danno lo stesso tipo | 263 su 331 (79 %) |
| unione con il reticolo, errori MedMentions | 13 su 22 (LLM da solo 3, reticolo da solo 3, entrambi 7); ma 45 link giusti su 114 cambiano lettura (39 %) |
| intersezione con il reticolo, errori MedMentions | 7 su 22, 12 link giusti su 114 (10,5 %) |

Nessuno dei due criteri è raggiunto. **Dove funziona:** trova lo span lungo che NCIt non ha ("inferior vena cava filter placement", "Living-Related Liver Donation", "simulated colon microbiome", "heart interleukin-6", "Canine brain phantoms", "development of pancreas", "heart of Maroilles cheese"): lì fa quello per cui è stato provato, e sa dire "alimento", "organismo", "procedura". **Dove non funziona:** (1) non vede gli errori in cui il sintagma è solo anatomia ("aortic arch", "colon", "developing brain", "brain parenchyma", "large bowel", 2 "heart" su 3): sono errori di convenzione dell'etichetta, non di lettura; (2) segnala come "altra cosa" i link che il corpus considera giusti per convenzione (22 dei 83 segnalati sono "brain weight/volume", "prostate mass", "spleen sterol contents": misura o origine della struttura, che il progetto registra con un ruolo e non perde; gli altri sono processi e procedure in cui la struttura è il luogo, "the role of ERK5 in the heart"); (3) il tipo è instabile nei casi in cui la parola è anche altro (il liver in "Mouse liver microsomes" è "proteina" per un modello e "organo" per l'altro).

**Correzione (sera del 9 ottobre).** Il probe marcava la prima occorrenza della parola nella frase, non quella giudicata (278 frasi su 2.058 hanno la parola due volte, 5 sono errori). "Canine brain phantoms" è un artefatto: l'etichetta riguarda la seconda occorrenza ("agarose brain parenchyma"). Rileggendo le etichette, "heart" nel formaggio è etichettato come la sola parola con un altro senso (concetto spaziale), non come nome composto. Il difetto è corretto (`at` nelle righe del controllo esterno); la classificazione corretta degli errori è in `perche_il_medico_legge_e_noi_sbagliamo_2026-10-09.md` §2, e il nuovo run (E2, informato) misura di nuovo tutto.

**Conclusione.** Il segmentatore LLM non è un decisore e non è un filtro: segnala, non decide. Il suo uso ammissibile è **solo come proposta di blocco** nei casi già incerti per le altre vie (reticolo non letto, o reticolo e LLM d'accordo sullo stesso ruolo), con la ripetizione sul gold set dei radiologi prima di qualunque effetto. Resta la tappa D (segmentatore appreso su MedMentions) e il lessico di teste curato. Limiti della misura: 33 errori, articoli e non referti, due modelli non certificabili e non deterministici, un solo campione casuale (seme 11). Millière e Buckner indicano la strada: certificazione di comportamento con coppie minime (stessa frase con la struttura come sede / come misura / come origine), non lettura dei modelli.

## Fonti

- Ding N, Melloni L, Zhang H, Tian X, Poeppel D. Cortical tracking of hierarchical linguistic structures in connected speech. Nat Neurosci 2016. https://doi.org/10.1038/nn.4186
- Nelson MJ, et al. Neurophysiological dynamics of phrase-structure building during sentence processing. PNAS 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5422821/
- Martin AE, Doumas LAA. A mechanism for the cortical computation of hierarchical linguistic structure. PLOS Biol 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5333798
- Mollica F, et al. Composition is the core driver of the language-selective network. Neurobiol Lang 2020. https://www.mit.edu/~hopekean/files/composition.pdf
- Pylkkänen L. The neural basis of combinatorial syntax and semantics. Science 2019. https://acesin.letras.ufrj.br/wp-content/uploads/2023/08/pylkkanen_science_2019.pdf
- Hagoort P. MUC (Memory, Unification, Control) and beyond. Front Psychol 2013. https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00416/full
- Morgan E, Levy R. Abstract knowledge versus direct experience in processing of binomial expressions. Cognition 2016. https://www.mit.edu/~rplevy/papers/morgan-levy-2016-cognition.pdf
- Christiansen MH, Chater N. The Now-or-Never bottleneck. Behav Brain Sci 2016. https://csl-lab.psych.cornell.edu/files/2021/02/2016-cc-BBS.pdf
- Caucheteux C, Gramfort A, King J-R. Evidence of a predictive coding hierarchy in the human brain listening to speech. Nat Hum Behav 2023. https://arxiv.org/abs/2111.14232
- Stanford Encyclopedia of Philosophy: Compositionality (https://plato.stanford.edu/entries/compositionality/); Hermeneutics (https://plato.stanford.edu/entries/hermeneutics/)
- Millière R, Buckner C. A philosophical introduction to language models, parts I e II. arXiv 2401.03910 e 2405.03207.
- Conformal NER, arXiv 2601.16999; Learn-then-Test (Angelopoulos et al.); RadGraph (Jain et al. 2021) e RadGraph-XL; SNOMED CT Compositional Grammar.
- Hwang S, Wang B, Gu A. Dynamic chunking for end-to-end hierarchical sequence modeling (H-Net). arXiv 2507.07955.
- Fountas Z, et al. Human-inspired episodic memory for infinite context LLMs (EM-LLM). ICLR 2025. https://arxiv.org/abs/2407.09450
- Asabuki T, Hiratani N, Fukai T. Chunking sequence information by mutually predicting recurrent neural networks. PLOS Comput Biol 2018 (preprint https://www.biorxiv.org/content/10.1101/215392v1)
- Greff K, van Steenkiste S, Schmidhuber J. On the binding problem in artificial neural networks. arXiv 2012.05208.
- Mohan S, Li D. MedMentions: a large biomedical corpus annotated with UMLS concepts. arXiv 1902.09476.
- Ihor/gliner-biomed-bi-small-v1.0 (scheda del modello, Apache-2.0), https://huggingface.co/Ihor/gliner-biomed-bi-small-v1.0
