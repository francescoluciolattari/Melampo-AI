# Memoria del dominio come la legge un medico: collocazioni, blocchi, cornici (10 ottobre 2026)

Risposta all'analisi di Frank su "inferior vena cava filter placement". Un medico (1) riconosce due blocchi già in memoria, [vena cava inferiore] e [posizionamento del filtro]; (2) lascia che la testa "placement" attiri le parole prima di lei nei suoi posti: un oggetto (il filtro) e un luogo (la vena); (3) tiene insieme le parole perché le ha viste insieme molte volte. Ordine deciso con Frank: prima il punto 3, poi l'1, poi il 2.

**Nessun numero qui è un test.** Misure sullo sviluppo (MedMentions dev, 1.935 link del controllo esterno); lo split congelato non è stato letto. Due scelte sono state fatte dopo aver visto i dati e lo dico dove cadono.

## 1. In breve

| punto | che cosa c'è ora | che cosa cambia oggi | che cosa serve |
|---|---|---|---|
| 3. Frequenza e co-occorrenza | `collocations.py`: coppie ordinate di parole, npmi, probabilità di transizione, conteggio con perdita per corpora grandi; `collocation_probe.py`; costruttore PubMed e workflow `collocations` | Le coppie di parole separano "dentro uno span annotato" da "a cavallo di un confine" con AUC 0,84-0,87, contro 0,77 della regola banale "due parole piene". La copertura cresce con il testo: 22 % delle coppie con 0,15 M parole, 50 % con 5,7 M. Nel reticolo non cambia nessuna decisione. | Testo molto più grande: PubMed (workflow pronto) e referti |
| 1. Blocchi già in memoria | `anatomy_names.json`: 33.799 nomi di strutture di 2-5 parole (UBERON e lessico), caricabili nella memoria dei blocchi | Un falso allarme in meno su 1.315 link giusti di CRAFT. "the outer layer of the olfactory bulb" non è più "parte di un dispositivo". Nessun errore perso. | Nomi di procedure e dispositivi: NCIt non ha "filter placement" |
| 2. Selezione semantica (valenza) | `frames.py`: per ogni testa di procedura, i posti "sito" e "dispositivo", da NCIt, dal testo e (con licenza) da SNOMED CT; il reticolo li usa (`frames=`) | "inferior vena cava filter placement" (2 errori): da "sito di un dispositivo" a **"sito della procedura, dispositivo = filter"**, cioè la lettura di Frank. Nessun altro ruolo cambia sui 1.905 link giusti. | SNOMED CT: l'Italia non è membro, serve la licenza affiliata |

Tutto è spento di default e misurato come variante (`ci_probe --collocations --frames`; nomi anatomici sempre misurati).

## 2. Punto 3: collocazioni

**Che cosa misura.** In MedMentions ogni span è un'unità segnata da una persona. Due parole adiacenti senza punteggiatura in mezzo sono "dentro" (stesso span) o "a cavallo" (una dentro e una fuori, o in due span diversi); le coppie tutte fuori non si giudicano. I conteggi vengono solo da testo senza etichette e non contengono mai i documenti valutati (dev).

| conteggi da | parole | copertura | AUC npmi | AUC, solo coppie di parole piene | F1 "dentro" | span letti come un'unità |
|---|---|---|---|---|---|---|
| MedMentions addestramento 25 % | 153.631 | 0,22 | 0,861 | 0,776 | 0,25 | 13,8 % |
| 50 % | 309.794 | 0,29 | 0,863 | 0,758 | 0,32 | 19,1 % |
| 100 % | 622.392 | 0,37 | 0,865 | 0,770 | 0,38 | 24,3 % |
| + CRAFT | 1.272.442 | 0,41 | 0,858 | 0,759 | 0,40 | 25,7 % |
| + definizioni NCIt | 5.702.229 | 0,50 | 0,843 | 0,733 | 0,45 | 29,9 % |

21.219 span di più parole, 125.895 coppie giudicate. Regola banale (entrambe parole piene, senza conteggi): AUC 0,765.

**Lettura.**
- Le coppie che si conoscono separano bene (0,86) e meglio della regola banale anche fra parole piene (0,73-0,78): è conoscenza di sequenze, non solo "le parole vuote stanno fuori".
- Quello che cresce con il testo è la **copertura**, non la qualità della singola coppia: da 22 % a 50 % delle coppie, e gli span letti interi da 14 % a 30 %. Le definizioni NCIt aggiungono copertura ma abbassano un po' l'AUC: sono un altro registro. È l'argomento di Frank in numeri: serve molto più testo del dominio.
- **Soglia.** La regola fissata prima ("unisci se npmi > 0", cioè sopra il caso) univa quasi tutte le coppie: nel testo due parole vicine stanno quasi sempre sopra il caso. La soglia ora si stima sui soli documenti di addestramento (conteggi da metà, F1 sull'altra metà): npmi > 0,3. È una correzione fatta dopo il primo run e lo dichiaro.

**Nel reticolo.** Un blocco composto le cui parole stanno insieme nel testo costa meno: fino al costo di un nome ricordato quando la coppia più debole ha npmi 1 (`ChunkLattice(collocations=...)`). Sui 1.935 link **non cambia nessuna lettura**: le decisioni vengono dal tipo della testa, e nessun errore ha accanto una parola familiare di tipo sconosciuto (controllato: 0 casi). Le collocazioni contano quando le teste non bastano, cioè con testo grande e su parole che la memoria non tipizza.

**PubMed.** `scripts/build_collocations.py` legge i file baseline di PubMed (titolo e abstract) ed **esclude tutti i PMID di MedMentions**, perché sono abstract di PubMed e alcuni sono documenti di valutazione. Conta con perdita (Manku e Motwani 2002, ε = 2·10⁻⁷) e salva solo le coppie viste almeno 3 volte. Il workflow `collocations` (input `files`, di default 10 file, circa 300.000 abstract) conta, rimisura la tabella sopra con PubMed, costruisce le cornici anche da PubMed e misura il reticolo sul controllo esterno. Gli abstract di CRAFT possono essere in PubMed: il loro testo, non le etichette, può entrare nei conteggi.

*Licenza.* La baseline PubMed è distribuita da NLM con i suoi termini; gli abstract possono essere coperti dal copyright degli editori. Nel file finisce un conteggio di coppie di parole, non testo. Se possa entrare nel dispositivo lo decide chi firma il fascicolo tecnico.

## 3. Punto 1: i blocchi delle strutture

La memoria del reticolo ricordava i nomi di **altre cose** che contengono una struttura (NCIt: "vena cava filter", "liver transplantation"), non i nomi delle **strutture**. Un vicino anatomico della menzione si leggeva parola per parola, e la sua ultima parola poteva avere il tipo di un altro senso: in NCIt "bulb" è un dispositivo.

`scripts/build_anatomy_names.py` scrive `data/linking/anatomy_names.json`:
- nomi e sinonimi esatti di UBERON (rilascio fissato 2025-05-28, CC BY 3.0) più il lessico del progetto in inglese e italiano;
- 2-5 parole, tokenizzati come il reticolo;
- `BlockMemory.load(..., anatomy_names=load_anatomy_names())`; un nome di altro tipo vince ("vena cava filter" resta un dispositivo).

| variante del reticolo | CRAFT: errori / giusti su cui agisce | MedMentions |
|---|---|---|
| memoria sola | 1 / 223 | 8 / 123 |
| + nomi delle strutture | 1 / **222** | 8 / 123 |
| + filtro grammaticale + nomi | 1 / 219 | 8 / 119 |

Effetto collaterale utile. Dei 6 sensi "fissati nel discorso" da teste di tipo dispositivo (`lettore_kintsch_dove_diverge…` §4), 2 erano "of the olfactory bulb": con i nomi delle strutture spariscono. Restano "sequence of development", "domain of mr-s forms" e "neurons of the LGE form": un verbo (form, forms) e una testa tipata male (development = dispositivo), che servono a un lessico delle teste curato.

**Che cosa manca.** I nomi di procedure e dispositivi: NCIt non ha "filter placement", "catheter placement", "stent placement". Le collocazioni da PubMed possono proporli come candidati (sequenze coese con una testa di procedura); vanno poi tipizzati e revisionati.

## 4. Punto 2: le cornici delle teste di procedura

**SNOMED CT: la licenza.** Ho controllato il 10 ottobre 2026: l'Italia **non è** fra i 54 membri di SNOMED International ([snomed.org/members](https://www.snomed.org/members)). Per un paese non membro la pagina [Get SNOMED CT](https://www.snomed.org/get-snomed) dice che serve una licenza affiliata tramite MLDS, con dichiarazione d'uso annuale. "Possono applicarsi tariffe" ed esistono esenzioni; per una stima chiedono di scrivere a info@snomed.org con l'uso previsto. Quindi:
- il costruttore legge i file RF2 di SNOMED (attributi Procedure site, Procedure site - Direct, Procedure site - Indirect, Using device, Direct device) **quando una release con licenza è disponibile a chi costruisce il file**; nel repository non c'è nulla di SNOMED;
- nel frattempo le cornici vengono da fonti aperte.

**Fonti aperte, nessuna cornice scritta a mano.**
- NCIt (CC BY 4.0): 331 procedure con relazioni di sito (Target, Imaged, Excised Anatomy), nessuna con il dispositivo (`Procedure_Uses_Manufactured_Object` è definita ma non usata). Troppo poco da solo.
- Testo letto (addestramento MedMentions e definizioni NCIt): per ogni parola che la memoria tipizza come procedura, i tipi delle parole che riempiono i suoi posti. Prima della testa, nella stessa frase senza punteggiatura (5 parole); dopo la testa, attraverso una preposizione (fino a 7 parole: "placement of a stent in the bile duct").
- **Un posto esiste** se la testa lo mostra più spesso delle teste di procedura in generale, con il 95 % di confidenza: il limite inferiore di Wilson della sua quota supera la frequenza di base (sito 14,4 %, dispositivo 2,7 %). Un posto è ciò che distingue la testa.

1.455 teste, 188 con un posto sito, 144 con un posto dispositivo. Esempi: *placement* sito e dispositivo, *biopsy* sito e dispositivo, *transplantation* sito, *resection* nessuno (nel testo letto prima di "resection" c'è quasi sempre una malattia: "tumor resection"; con più testo cambierà).

*Scelta fatta dopo aver visto.* La prima versione guardava 5 parole anche dopo la preposizione e "placement" non aveva il posto sito. L'ho portata a 7, la finestra destra del reticolo, per coerenza: è una ragione di principio, ma la modifica viene dopo aver visto il caso.

**Nel reticolo** (`ChunkLattice(frames=...)`). Se un blocco che contiene la menzione finisce, a destra, con una testa di procedura che ha un posto sito, e ogni parola fra menzione e testa riempie un posto (un dispositivo per il posto dispositivo, una struttura per il sito), allora la struttura è il sito di quella procedura, qualunque blocco più piccolo preferiscano i costi. La lettura scrive la cornice.

| | prima | con le cornici |
|---|---|---|
| "office-based inferior vena cava filter placement" | sito di un dispositivo ([IVC filter]) | **sito della procedura**, `frame:placement:device=filter` |
| "inferior vena cava (IVC) filter placement" | sito di un dispositivo | **sito della procedura**, `frame:placement:device=filter` |
| 1.905 link giusti | | nessun ruolo cambia; 55 già "sito di procedura" ora hanno anche la cornice scritta |

Fra le cornici scritte ce ne sono di teste tipate male come procedure ("cholesterol", "arising", "type", "total"). Non cambiano ruolo, ma mostrano di nuovo il rumore del lessico delle teste.

## 5. Che cosa resta

- **Più testo.** Lanciare il workflow `collocations` (10 file per cominciare; poi 50). Dirà quanto crescono copertura, span letti interi e cornici, e se il reticolo cambia decisioni.
- **Lessico delle teste curato** (inglese e italiano): è la causa comune di "bulb", "form", "development", "total", "cholesterol".
- **SNOMED CT**: decidere se chiedere la licenza affiliata (MLDS). Se arriva, si aggiunge con `--snomed-relationships/--snomed-descriptions` senza toccare il codice.
- **Italiano**: le stesse tre cose servono sui referti italiani. Le collocazioni e le cornici dal testo funzionano su qualunque lingua, ma servono referti italiani pseudonimizzati.
- Niente è acceso di default. Per accendere nomi, cornici e collocazioni nel linker aspetterei il run su PubMed e il gold set: oggi l'effetto misurato è "nessun danno, due errori letti come li legge un medico, un falso allarme in meno".

## 6. Come si ripete

```
python scripts/collocation_probe.py --medmentions ext/mm --craft ext/craft --ncit ncit.obo [--extra pubmed.json.gz]
python scripts/build_anatomy_names.py --uberon data/linking/uberon-basic.obo
python scripts/build_procedure_frames.py --ncit ncit.obo --ncit-definitions --medmentions ext/mm [--pubmed ...]
python scripts/ci_probe.py ... --collocations colloc.json.gz --frames data/linking/procedure_frames.json
```
Workflow: `collocations` (Actions, manuale). Test: `tests/test_collocations_frames.py` (10).
