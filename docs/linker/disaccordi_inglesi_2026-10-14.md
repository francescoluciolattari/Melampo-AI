# I disaccordi in inglese: perché sono il 10–13% e cosa serve per scendere sotto l'1% (14 ottobre 2026)

Obiettivo di Frank: con la regola "lettore e modello d'accordo, altrimenti a una persona", scendere sotto l'1% di item mandati a una persona. In inglese oggi sono il 10% (Nemotron 3 Ultra) e il 13% (Gemma 4 31B), su 484 item anatomici.

## I 71 item in disaccordo, per causa

Li ho letti uno per uno: sono gli item in cui almeno uno dei due modelli non concorda con il lettore. Le cause sono quattro.

| causa | item | esempi |
|---|---|---|
| **A. Il lettore non conosce la parola che regge** (procedura o dispositivo) | 15 | "CT brain", "views of the chest", "radiograph of the knees", "extraction of the … incisor tooth", "incising the conjunctiva", "skin graft surgery", "the Cx was stented" |
| **B. Il vocabolario (NCIt) dà un tipo sbagliato a una parola vicina** | 7 | "knees pain" ("pain" vi risulta procedura), "a normal liver appearance" ("appearance" vi risulta procedura), "the cervix looked" |
| **C. Il modello sbaglia, o usa una convenzione diversa dal gold** | ~40 | "pleural effusion" letto come `source_of` (Gemma, 6 volte), "pulmonary vascularity" e "cardiac contours" come proprietà, "cardiomegaly" come non-sede |
| **D. La convenzione non è decisa** (anche il gold è discutibile) | ~10 | "low lung volumes", "size of the cardiac silhouette", "surgical clips in the right upper abdomen", "patching of the right eye", "partial resection of the posterior fourth rib" |
| **E. Difetto tecnico**: Nemotron copia il segno `<w>` nella risposta, che viene scartata | 6 | "the \<w\>lingula\</w\>" |

## Cosa ho corretto e cosa ho misurato

1. **E è corretto:** il controllo della risposta toglie il segno copiato prima di verificare il sintagma (`llm_reading.check`). Le risposte già in cache valgono subito.
2. **Il lettore riceve la parola che regge** (`governor.py`):
   - ora risale le catene di nomi ("skin graft surgery" porta a *surgery*);
   - vede l'esame scritto prima della menzione ("CT brain" porta a *CT*);
   - con un vocabolario esatto delle parole che reggono (`governor_kinds`), quella parola **decide** la lettura, invece di essere un voto tra gli altri.
3. **Misura**:
   - **Come voto tra gli altri, con NCIt, la parola che regge non basta**: nell'integrazione pesano le tracce e il contesto, che dicono quasi sempre "struttura", e un nodo solo non le ribalta. Se le si dà più peso peggiora (38 errori invece di 25), perché NCIt dà a molte di queste parole un tipo sbagliato ("pain", "appearance").
   - **Stima del tetto, con un vocabolario corretto**: ho tipizzato a mano le parole che reggono (una sessantina) come farebbero UMLS e RadLex, e le ho lasciate decidere. È una stima, non un dato: la lista scritta a mano non è nel repo.

| | Gemma 4 31B | Nemotron 3 Ultra |
|---|---|---|
| oggi | 63 (13,0%) | 49 (10,1%) |
| + parola che regge con tipi corretti + correzione E | 49 (10,1%) | **30 (6,2%)** |
| d'accordo ma sbagliati (rispetto al gold di oggi) | 12 | 14 |

## Perché non si scende sotto l'1% senza decidere le convenzioni

Dei 30 disaccordi che restano con Nemotron, 17 sono errori del lettore e 13 del modello. I 14 "d'accordo ma sbagliati" sono quasi tutti casi in cui lettore e modello concordano contro il gold, e il gold applica una convenzione non scritta ("size of the cardiac silhouette": il gold dice struttura, la parola che regge dice misura). Finché la convenzione non è decisa e scritta nello stesso modo per il gold, per il lettore e per il prompt del modello, questi casi restano disaccordi o errori.

**Le convenzioni da decidere** (radiologi e Frank):

1. **Struttura dentro una frase di misura** ("low lung volumes", "size of the cardiac silhouette", "uterus size", "elevation of the hemidiaphragm"): struttura o `inherent_location`?
2. **"Pulmonary vascularity", "cardiac contours", "mediastinal silhouette"**: struttura o proprietà?
3. **Dispositivo che si trova in una struttura** ("surgical clips in the right upper abdomen", "PICC tip near the brachiocephalic vein"): `device_site` o struttura?
4. **Struttura oggetto di un atto terapeutico** ("patching of the right eye", "partial resection of the posterior fourth rib", "autotransplanted to the sternocleidomastoid muscle", "tumor removed from head/neck"): `procedure_site`? Il gold oggi dice struttura; i modelli dicono sede di procedura.
5. **Struttura mostrata normale da un esame** ("bronchoscopy showed a normal tracheobronchial tree"): struttura o sede dell'esame?
6. **Aggettivo di sede in un reperto** ("pleural" in "pleural effusion", "pulmonary" in "pulmonary edema"): struttura. Questo è già il gold e va solo scritto nel prompt.

## Il piano per arrivare sotto l'1%

1. **Decidere le 6 convenzioni** e scriverle: il gold corretto (registrando che la correzione è fatta dopo aver visto le risposte), le definizioni dei ruoli nel prompt, esempi bilanciati.
2. **Il vocabolario delle parole che reggono**: RadLex per le parole della radiologia, UMLS (tipi semantici) per le altre, dove la corrispondenza è esatta. Script che costruisce `governor_kinds` nel workflow.
3. **I verbi che reggono** ("incising", "stented", "packed"): dal verbo al nome dell'atto (incise → incision), con il vocabolario.
4. **Rilanciare `role-probe`** con `governor` acceso e il nuovo prompt, e misurare su un set di prova che non sia quello su cui si sono scelte le regole.

## RadLex: come scaricarlo

1. Aprire **https://radlex.org/** e premere **"Download RadLex"**.
2. Alla domanda *"Do you agree to the RadLex License Agreement and Terms of Use?"* premere **"I agree"**, poi **"Next"**. La versione attuale è la 4.3. La licenza (RadLex License Agreement V2.1) è gratuita anche per uso commerciale; chiede l'attribuzione e vieta di usare il nome RadLex/RSNA per versioni modificate.
3. Il file si allega qui in chat, oppure resta da te.

In alternativa si usa **BioPortal** (bioportal.bioontology.org, account gratuito). Dal profilo si copia la API key e la si mette tra i secret del repo come `BIOPORTAL_API_KEY`; il workflow scarica RadLex 4.3 da solo.

Nel repo entra solo il derivato (parola → tipo), con l'attribuzione richiesta: "is based on the RadLex® Ontology developed at the Radiological Society of North America". RadLex stesso non entra.

## Fonti

- [RadLex.org](https://radlex.org/)
- [RSNA, RadLex radiology lexicon](https://www.rsna.org/practice-tools/data-tools-and-standards/radlex-radiology-lexicon)
- [RadLex License Agreement and Terms of Use V2.1](https://www.rsna.org/uploadedFiles/RSNA/Content/Informatics/RadLex_License_Agreement_and_Terms_of_Use_V2_Final.pdf)
- [BioPortal, RADLEX 4.3](https://bioportal.bioontology.org/ontologies/RADLEX)

## Aggiornamento della sera: cause C e D spiegate, vocabolari con le chiavi di Frank

**Causa C, il modello sbaglia.** Lettore e gold sono d'accordo, ma il modello dà un altro ruolo. Esempio: in "No pleural effusion" la parola chiesta è "pleural". È la struttura in cui si trova il versamento, quindi struttura. Gemma risponde `source_of`, cioè "la struttura è da dove viene una sostanza", e lo fa 6 volte. È un errore di lettura del modello: si corregge scrivendo meglio la definizione dei ruoli nel prompt, con esempi.

**Causa D, la regola non c'è.** Lettore e modello concordano tra loro, oppure non concordano, ma la risposta giusta dipende da una convenzione che non abbiamo mai scritto. Esempio: in "low lung volumes" la parola "lung" è struttura (il gold) o "misura del polmone"? Non è un errore di nessuno: manca una decisione. Le 6 decisioni sono elencate sopra.

**Le chiavi.** Nei secret ci sono `BIOPORTAL_API_KEY` e `UMLS_API_KEY`, con la licenza UMLS già accettata.

- `scripts/build_governor_kinds.py` raccoglie le parole che reggono le menzioni (dalle frasi, mai dalle etichette). Per ciascuna chiede:
  - a **RadLex**, via BioPortal, la classe con quel nome esatto e il ramo sopra di essa: procedura o modalità d'esame, osservazione o reperto, anatomia, dispositivo, proprietà;
  - a **UMLS** i concetti con quel nome esatto e i loro tipi semantici (procedura diagnostica o terapeutica, segno o sintomo, reperto, dispositivo, attributo…).
- **Una parola riceve un tipo solo se tutte le corrispondenze esatte danno lo stesso tipo**, dentro ogni fonte e tra le due. Altrimenti resta senza tipo ed è registrata come conflitto.
- **Da UMLS non uso SNOMED CT.** L'Italia non è membro di SNOMED International, e nei Paesi non membri SNOMED CT richiede la licenza affiliata. Le fonti cercate sono MeSH, NCI, HPO, Metathesaurus, CHV, LOINC e MedlinePlus.
- **La tabella si costruisce nel workflow e non si committa né si carica come artefatto**: il contenuto UMLS non si ridistribuisce. Nell'artefatto vanno solo i conteggi.
- **Uso:** con l'input `governor: true`, il workflow `role-probe` costruisce la tabella e la passa al lettore (`--governor-kinds`). Quando la parola che regge è nella tabella, decide la lettura.

**BioPortal: quali ontologie servono.** BioPortal ne ospita centinaia. Per il nostro compito servono poche:

| ontologia | a cosa serve | decisione |
|---|---|---|
| **RADLEX** (4.3, dic. 2025) | parole della radiologia: esami, proiezioni, osservazioni, anatomia | **usata subito** |
| **HP** (Human Phenotype Ontology) | reperti e segni; ha una **traduzione italiana** | candidata per i reperti italiani |
| **ORDO_IT** (Orphanet, italiano) | malattie rare in italiano | poco utile per i ruoli |
| MESH, NCIT, LOINC, HPO via UMLS | già coperte dalla ricerca UMLS | non serve BioPortal |
| SNOMEDCT | completa | **non utilizzabile** dall'Italia senza licenza affiliata |
| UBERON, FMA | anatomia | UBERON è già nel linker |
| ICD9CM, ICD10CM (inglese) | diagnosi e procedure in inglese | coperte da UMLS; per l'italiano servono ICD-9-CM IT e CIPI |

## Il run del 14 ottobre (governor acceso, tipi RadLex/UMLS) e il cambio di impostazione

**Tabella dei tipi.** Delle 419 parole che reggono una menzione, 99 hanno ricevuto un tipo e 25 sono finite in conflitto.

**Il lettore è peggiorato:** 88,1% → 80,0% di ruoli giusti, e gli errori sede/non-sede sono passati da 3 a 39. I casi rotti sono 65 in italiano e 24 in inglese. Le cause sono tre:

1. **Il vocabolario inglese applicato a parole italiane.** "via" (vie biliari) è stato tipizzato come il gene *VIA*, "tc" come un gene.
2. **Una parola con più sensi, fissata a un solo tipo.** UMLS dà "elevation" come procedura; in "elevation of the hemidiaphragm" è un reperto.
3. **Il peso contro "struttura" sotto una parola sconosciuta.** Non ha prodotto un "non so": ha fatto vincere un altro ruolo (sede di dispositivo, nome di altro).

È la conferma di quanto ha detto Frank: il tipo di una parola fuori dal contesto non può decidere.

**Corretto:**

- il tipo da RadLex/UMLS è solo un'evidenza tra le altre, mai una decisione;
- vale solo per le parole inglesi;
- non si usa se il tipo è gene, sostanza chimica o organismo;
- nessun peso contro "struttura".

Il lettore con il reggente resta spento per default. Misurato in locale con i tipi del run (79 parole), non migliora l'accordo: gli errori del lettore inglese passano da 25 a 21, ma i suoi "non so" aumentano.

**Corretti anche, nel controllo delle risposte dei modelli:**

- il segno `<w>` copiato nella risposta;
- un sintagma che lascia fuori il segnaposto "XXXX" dei referti anonimizzati.

Dei 9 disaccordi sede/non-sede di Nemotron in inglese, 7 venivano da questi due difetti.

### C e D rivisti (Frank)

- **C.** In "No pleural effusion", `source_of` (la pleura come origine del versamento) non è sbagliato. La struttura è la sede del reperto e anche da dove il liquido viene. Lo schema chiede *un* ruolo dove il testo ne sostiene più d'uno: questo non è un disaccordo tra due letture, è un limite dello schema.
- **D.** Niente regole caso per caso. "Misura del polmone" (`inherent_location`) vuol dire che la frase parla di una quantità dell'organo, il suo volume. "Low lung volumes" è insieme una misura e un reperto radiologico dei polmoni. Quale dei due conta lo dice il contesto, come nella lettura umana. Le 6 "decisioni" proposte sono ritirate.

### Che cosa deve concordare

Dove serve davvero l'accordo tra lettore e modello è ciò che cambia l'esito. Prima di tutto: la parola indica qui una sede del corpo o no? Poi, se il link si fa o se la struttura è solo l'oggetto di una misura.

Disaccordi (lettore senza reggente, risposte dei modelli con le due correzioni):

| | ruolo esatto | link / misura / non-sede | sede / non-sede |
|---|---|---|---|
| inglese, Gemma 4 31B | 13,0% | 6,4% | 1,9% |
| inglese, Nemotron 3 Ultra | 9,1% | 3,7% | **0,6%** (3 su 484) |
| italiano, Gemma 4 31B | 30,1% | 8,0% | 4,5% |
| italiano, Nemotron 3 Ultra | 26,4% | 3,2% | 1,9% |

**Sulla domanda che conta di più (sede o no) l'inglese con Nemotron è già sotto l'1%, senza errori quando i due concordano.**

I 3 casi che restano:

- "Granulomatous disease in the bilateral": una frase mozzata;
- "pulmonary outflow and XXXX": anonimizzata;
- "T11 level": il lettore non sa.

I ruoli più fini (sede di un reperto, di una procedura, oggetto di una misura) possono essere veri insieme. Vanno registrati come **insieme di ruoli che il contesto sostiene**, non scelti con una regola. Proposta:

1. il modello indica *tutti* i ruoli che si applicano, non uno;
2. il lettore conserva le letture alternative con le loro quote;
3. si registrano i ruoli condivisi;
4. una persona interviene solo quando le letture si contraddicono sull'esito: sede contro non-sede, oppure link contro solo misura.
