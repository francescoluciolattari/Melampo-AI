# Il reggente nel lettore Kintsch, il vocabolario italiano, una procedura di addestramento (13 ottobre 2026)

## 1. Il reggente sintattico nel lettore

**Cosa fa.** `melampo.memory.governor` usa spaCy (licenza MIT; `en_core_web_md` e `it_core_news_md` 3.8.0) per trovare la parola da cui dipende la menzione. La ricerca segue le coordinazioni fino al primo membro e distingue tre casi:

- "X of/di \<menzione\>" e "\<menzione\> X" (composto): il reggente è X;
- il verbo di cui la menzione è soggetto o oggetto: il reggente è il verbo;
- una preposizione di luogo ("opacity **in** the lower lobes", "lesione **nel** fegato"): la menzione è un luogo.

Il lettore (`CIReader(governor=...)`) aggiunge un nodo all'integrazione, come le altre evidenze. Non decide da solo:

- **reggente con un tipo nel vocabolario** (NCIt): sostiene la lettura di quel tipo. "incision" porta a sede di procedura;
- **preposizione di luogo**: sostiene "struttura" (oppure sede di dispositivo, se il reggente è un dispositivo);
- **reggente noto ma senza ruolo** (un'azione, un cambiamento: "changes of the spine"): sostiene "struttura";
- **reggente nome sconosciuto al vocabolario**: pesa *contro* "struttura", con un peso per lingua (inglese 0, italiano 1). Il lettore non legge "struttura" sotto una parola che non sa nominare.

Il nodo è spento per default. Si accende con `role_probe.py --governor` e con l'input `governor` del workflow `role-probe`, che installa spaCy e i due modelli fissati alla release.

**Misura.** Lettore da solo, item anatomici del run del 13 ottobre. Sono frasi rilette in locale senza il testo del referto intero, quindi i numeri della riga "senza" differiscono di poco dal run.

| | inglese, 484 item | italiano, 375 item |
|---|---|---|
| senza reggente | 25 errori, 1 incerto | 77 errori, 1 incerto |
| con reggente (sconosciuto: peso 0) | 24 errori, 12 incerti | 75 errori, 4 incerti |
| con reggente + suffissi di procedura e sigle d'esame (\*) | **23 errori**, 13 incerti | 59 errori, 9 incerti |
| come sopra, sconosciuto con peso 1 | 27 errori, 68 incerti | **26 errori**, 189 incerti |

(\*) I suffissi (-scopia/-scopy, -grafia/-graphy, -tomia/-tomy, -ectomia/-ectomy, -stomia, -plastica, -centesi, biopsia) e le sigle (TC, TAC, RM, RMN, RX, ECO, PET, CT, MRI) sono stati provati solo nella misura e **non sono nel codice**: sono una regola, non una corrispondenza di vocabolario. Entrano se il vocabolario italiano (§2) non basta.

- **Inglese:** il reggente aiuta poco. Il parser trova quasi sempre il reggente giusto, ma il vocabolario dei tipi (NCIt) sbaglia o non ha proprio le parole comuni che reggono: "appearance" e "pain" vi risultano procedure; "radiograph", "view", "extraction", "change" non ci sono. Pesare il reggente sconosciuto in inglese peggiora, perché quasi sempre è una parola di reperto ("tortuosity of the aorta").
- **Italiano:** con il reggente gli errori del lettore scendono da 77 a 26, ma metà degli item diventa "non so". È quello che si chiedeva: niente "struttura" detta con certezza sotto una parola sconosciuta. Per passare dal "non so" alla risposta giusta serve un vocabolario italiano dei tipi (§2).

**Nella regola "lettore e modello d'accordo, altrimenti a una persona"** (Gemma 4 31B oppure Nemotron 3 Ultra):

| | inglese | italiano |
|---|---|---|
| errori | **3**, tutti etichette sbagliate del gold set (uterus size, abdominal wall incisions, abdominal cavity) | 2 (appendice, piede torto congenito) |
| a una persona | 10–13% | 26–30% |

**In inglese gli errori veri sono già zero su questo set di sviluppo.** Il reggente non cambia questo numero. Cambia il lettore da solo (in italiano) e rende onesto il suo "non so". Gli errori italiani restano finché "liberazione" e "intervento" non hanno un tipo.

## 2. Vocabolari dei tipi: inglese più completo di NCIt, italiano (rivisto la sera, decisione di Frank)

### Cosa vuol dire "a una persona"

Con la regola "lettore e modello d'accordo, altrimenti a una persona", un item va a una persona quando le due letture indipendenti non concordano: il sistema non decide e chiede a chi rivede. In inglese sono 63 item su 484 con Gemma (13%) e 49 con Nemotron (10%). Chi ha ragione in quei casi:

| | lettore giusto | modello giusto | entrambi sbagliati |
|---|---|---|---|
| Gemma 4 31B | 40 | 20 | 3 |
| Nemotron 3 Ultra | 26 | 21 | 2 |

Il disaccordo più frequente è il lettore che dice struttura e il modello che dice sede di procedura (19 casi): "CT brain", "incising the conjunctiva", "extraction of the canine and lateral incisor tooth". Lì il modello ha quasi sempre ragione, e il lettore sbaglia perché il suo vocabolario non conosce la parola che regge.

### I 3 errori inglesi della regola

Tutti e tre sono etichette sbagliate del gold set (le ho scritte io):

- "The **uterus** size was 6cm × 4cm × 3cm": lettura `inherent_location`, gold struttura. Per la convenzione del gold ("heart size" ha ruolo `inherent_location`), il gold è sbagliato;
- "the **abdominal wall** incisions were located…": lettura `procedure_site`, gold struttura. L'incisione è una procedura sulla parete, quindi il gold è sbagliato;
- "no specific findings were found in the **abdominal cavity**": lettura struttura, gold `procedure_site`. "Found in" fa della cavità la sede di un reperto, quindi il gold è sbagliato.

### Inglese: più completo di NCIt

| fonte | cosa dà | licenza |
|---|---|---|
| **UMLS** (Metathesaurus + Semantic Network) | ogni concetto di oltre 200 vocabolari (SNOMED CT US, MeSH, NCIt, RadLex, LOINC…) con un tipo semantico: "Diagnostic Procedure", "Finding", "Body Part"… | licenza UMLS (gratuita su richiesta, non redistribuibile); `head_typing.py` è già pronto e aspetta la chiave |
| **RadLex** (RSNA) | il lessico della radiologia: procedure ("radiograph", "view"), osservazioni d'immagine ("opacity", "density"), anatomia | licenza pubblica RadLex: uso anche commerciale e redistribuzione, con attribuzione |

**Proposta:** RadLex per le parole della radiologia, che sono il nostro dominio e si possono tenere nel repo, e UMLS per il resto, se arriva la chiave. NCIt resta dove ha la corrispondenza.

### Italiano (decisione di Frank: ICD, poi MeSH italiano e LOINC)

**ICD-10-CM non esiste in italiano.** È la modifica statunitense. In Italia:

- **ICD-9-CM 2007**, versione italiana del Ministero: in uso per le SDO fino al 31 dicembre 2026;
- **ICD-10-IM** (diagnosi) e **CIPI**, la Classificazione italiana delle procedure e degli interventi chirurgici: adottate dal DM Salute 23 ottobre 2025 (GU n. 286 del 10 dicembre 2025). Valgono dal 1° gennaio 2027, con il 2026 come anno di doppia codifica sperimentale. Le versioni prototipo ("gamma") sono sul portale del Ministero (progetto NSIS-CLASS) da febbraio 2026; la versione finale è attesa per la fine del 2026. Il decreto non indica una licenza d'uso.

**Ordine proposto:**

1. **CIPI** per gli interventi e le procedure, perché è la classificazione italiana che resterà. In attesa della versione finale si usa **ICD-9-CM 2007 italiano** (volume 3, procedure), che è in vigore ora; le tavole di transcodifica del Ministero collegano le due.
2. **ICD-10-IM / ICD-9-CM** per le diagnosi.
3. **MeSH italiano** (ISS, in UMLS): procedure, malattie, anatomia, con l'albero dei tipi; licenza UMLS.
4. **LOINC italiano** (CNR): esami di laboratorio e radiologia ("TC torace"); licenza LOINC.

La licenza di riuso dei file del Ministero va verificata prima di mettere nel repo qualcosa di derivato. Fino ad allora, solo script che scaricano e costruiscono, come per NCIt.

## 3. Una procedura di addestramento per Gemma 4 e Nemotron 3 Ultra (Nemotron 3 Super tolto, decisione di Frank)

`melampo.evaluation.model_profiles` raccoglie in un posto i modelli previsti, con quanto è verificato e quanto viene solo dalle pagine dei fornitori:

| profilo | pesi (Hugging Face) | uso | addestramento LoRA | verificato qui |
|---|---|---|---|---|
| `gemma-4-31b` | `google/gemma-4-31B-it`, Apache 2.0 | 1 GPU da 80 GB | NeMo AutoModel `gemma4_31b_peft.yaml` (esempio NVIDIA su 8 GPU); QLoRA su 1 GPU stimato | ruoli via OpenRouter |
| `nemotron-3-ultra` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (anche NVFP4) | 1 nodo da 8 GPU | ricetta LoRA NeMo AutoModel, 4 nodi (16 GB200 o 32 H100) | ruoli via OpenRouter |

**La procedura è la stessa per tutti** (`scripts/lora_roles.py`):

1. `prepare`: dati del solo ruolo, cinque fold raggruppati per referto;
2. `export`: il fold come JSONL in formato chat (`messages`), cioè gli stessi file per ogni addestratore;
3. **addestramento:**
   - NeMo AutoModel con la ricetta NVIDIA del modello (Gemma 4 e Nemotron sono entrambi coperti), puntata ai file esportati;
   - in alternativa, per una sola GPU, `train --profile` (QLoRA con TRL);
4. `evaluate --endpoint`: base e base + adattatore serviti dietro un endpoint compatibile OpenAI (vLLM, NIM), con la stessa domanda e gli stessi punteggi della sonda.

`plan --profile <nome>` stampa i passi. Nulla è stato addestrato: servono GPU e le etichette dei radiologi. Addestrare sulle etichette silver insegnerebbe anche i loro errori (§1).

## Fonti

- [spaCy models, it_core_news_md / en_core_web_md 3.8.0](https://github.com/explosion/spacy-models/releases)
- [NeMo AutoModel, Gemma 4 31B](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/google/gemma-4-31B-it.md)
- [NeMo AutoModel, Fine-Tune Nemotron-3-Ultra-550B](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/nemotron-3-ultra)
- [NeMo AutoModel, LLM coverage](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/llm/index.html)
- [UMLS, LOINC Italian (LNC-IT-IT)](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/LNC-IT-IT/metadata.html)
- [Università di Padova, la classificazione ICD-9-CM](https://sdb.unipd.it/laclassificazioneicd-9-cm)
- Hugging Face API, modelli `nvidia/*Nemotron-3*` e `google/gemma-4*` (13 ottobre 2026)
- [DM Salute 23 ottobre 2025, ICD-10-IM e CIPI](https://www.mefop.it/cms/doc/29766/dm-salute-23-ottobre-2025.pdf)
- [ITDRG, ICD-10-IM e CIPI](https://www.italiandrg.com/itdrg-icd10-cipi-ent.html)
- [RadLex public license 1.0.1](https://rsna.org/uploadedFiles/RSNA/Content/Informatics/RadLex/radlex_public_license_version_1-0-1.pdf)
