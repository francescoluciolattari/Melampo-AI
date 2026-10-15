# Il metodo del lettore chiesto al modello, RadLex nel vocabolario di base, l'addestramento dalle revisioni (14 ottobre 2026)

## 1. Braccio `llm_method`: il metodo, non le conclusioni

Il lettore Kintsch costruisce le letture possibili, raccoglie gli indizi per ciascuna e le integra. Al modello si chiede la stessa cosa, scritta passo per passo e fatta con la sua conoscenza della lingua (`llm_reading.method_prompt`). Il modello deve indicare:

1. il sintagma e la sua testa;
2. **il reggente**: la parola da cui il sintagma dipende, anche fuori dal sintagma (oltre "e", dopo "di"/"of", il nome o il verbo), e che cosa nomina: esame, procedura, dispositivo, reperto o malattia, misura o proprietà, altro;
3. il predicato, cioè che cosa dice la frase della struttura;
4. le letture possibili, ciascuna con l'indizio a favore e quello contro;
5. i ruoli che l'evidenza sostiene, il più importante per primo.

Il modello **non vede** le risposte del lettore: così i due restano indipendenti e il loro accordo conserva valore. Un modello a cui si mostrano le risposte del lettore ne copia gli errori (misurato il 13 ottobre).

Nella sonda è il braccio `llm_method`. Gli arms predefiniti di `role-probe` diventano `lattice,reader,llm,llm_method`: `llm+reader` è tolto, perché copia gli errori e costa. La tabella dell'accordo sull'esito ha una riga per `llm_method:<modello>` accanto a quella del braccio `llm`.

## 2. RadLex nel vocabolario di base del lettore

`scripts/build_radiology_kinds.py` scarica tutto RadLex da BioPortal (esportazione CSV, `BIOPORTAL_API_KEY`). Il tipo di ogni classe si legge dal ramo in cui sta: modalità d'esame o procedura → procedura; osservazione d'immagine o reperto → reperto; entità anatomica → anatomia; dispositivo → dispositivo; proprietà → proprietà. Una parola o un nome riceve un tipo solo se tutte le classi RadLex che lo usano danno lo stesso tipo.

Il risultato è `data/linking/radiology_kinds.json`. `BlockMemory.load` lo legge se esiste:

- **riempie** le parole che NCIt non tipizza, o tipizza solo come "conceptual" ("ct", "mri", "radiograph");
- **aggiunge** i nomi di più parole che NCIt non ricorda;
- **dove NCIt e RadLex danno due tipi diversi, resta quello di NCIt.**

Il workflow `radiology-kinds` (a mano e il primo del mese) lo costruisce e apre una pull request.

**Nessuna regola "esame davanti a struttura".** Frank l'ha giudicata troppo deterministica. In "CT brain" il reticolo ha ora anche il blocco "ct" con tipo procedura; l'indizio entra nell'integrazione del lettore insieme agli altri, e decide il contesto. Quanto cambi lo dirà il run.

**Licenza.** La RadLex License consente il materiale derivato, con la nota "based on the RadLex® Ontology developed at the Radiological Society of North America", che è nel file e nella PR. Il file non si chiama RadLex e non è RadLex.

## 3. Tolto il reggente nel lettore

Erano stati tolti dal codice:

- il parser spaCy (`governor.py`);
- i tipi RadLex/UMLS delle parole che reggono (`build_governor_kinds.py`);
- le opzioni `--governor` e `--governor-kinds` della sonda;
- l'input `governor` del workflow.

Il motivo: misurati, peggioravano il lettore (decidendo) o non lo miglioravano (come indizio). Decisione di Frank: non si tiene nulla che peggiori, nemmeno spento. Restano i documenti che raccontano le misure. Il metodo passa al modello (punto 1), che il reggente lo trova da sé.

## 4. L'addestramento dalle revisioni dei medici

`melampo.evaluation.review_feedback` registra le decisioni prese nella coda di revisione:

- la correzione o la conferma del medico;
- la seconda conferma dell'interprete;
- i referti confermati per intero.

Le regole sono tre:

- **solo testo pseudonimizzato**: il registro rifiuta il resto. Nessun nome e nessun identificativo del paziente; l'id del referto è una chiave pseudonima;
- **una decisione diventa dato di addestramento solo se confermata da una seconda persona**, diversa da chi ha deciso: una correzione da sola è una proposta, non un'etichetta;
- le righe hanno il formato di `lora_roles.py` (domanda, ruoli come risposta, cinque fold per referto), quindi la stessa procedura addestra Gemma 4 31B o Nemotron 3 Ultra.

Per il MDR un modello riaddestrato è una nuova versione: si valida sul gold set tenuto da parte prima di sostituire quello in uso.

**Etichette reali di radiologi già disponibili:**

- **RadGraph / RadGraph-XL**: referti MIMIC-CXR e altre modalità annotati da radiologi, con anatomia, osservazioni e relazioni "si trova in". Servono le credenziali PhysioNet, e i dati restano sulla macchina della persona accreditata.
- **Rad-SpRL** (già nel progetto): 2.000 referti IU X-ray con le sedi annotate.

In italiano non conosco un dataset pubblico di referti con questi ruoli. PARROT ha referti scritti da radiologi, ma senza ruoli. Per l'italiano la fonte saranno le revisioni stesse e il gold set dei radiologi.
