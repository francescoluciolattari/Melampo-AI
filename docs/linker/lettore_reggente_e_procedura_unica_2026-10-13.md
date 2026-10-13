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

## 2. Un vocabolario italiano dei tipi: le fonti

| fonte | cosa contiene | licenza | stato |
|---|---|---|---|
| **LOINC, variante linguistica italiana** (traduzione CNR, in UMLS come LNC-IT-IT) | nomi di esami di laboratorio e di **radiologia** (LOINC/RSNA Radiology Playbook): "TC torace", "RM encefalo" | licenza LOINC (gratuita, con attribuzione; account su loinc.org) | candidata per esami e sigle |
| **ICD-9-CM, versione italiana 2007** (Ministero della Salute, usata per le SDO) | interventi e procedure (volume 3): "asportazione", "resezione", "lisi di aderenze" | pubblicazione del Ministero; licenza di riuso da verificare | candidata per gli interventi |
| **MeSH italiano** (ISS, in UMLS come MSHITA) | descrittori con albero (E = procedure, C = malattie, A = anatomia) | licenza UMLS (gratuita su richiesta, uso non redistribuibile) | buona tassonomia; non nel repo |
| **Wikidata**, etichette italiane | voci con "istanza di procedura medica / malattia / struttura anatomica" | CC0 | aperta, qualità variabile |
| SNOMED CT | completa | l'Italia non è membro: serve la licenza affiliata MLDS | decisione aperta (già nota) |

**Proposta.** Il tipo di una parola italiana va preso dove la fonte ha la corrispondenza esatta (principio "deterministico solo dove c'è la corrispondenza"):

- LOINC IT per gli esami;
- ICD-9-CM IT per gli interventi;
- MeSH IT per il resto, se arriva la licenza UMLS;
- Wikidata come riserva, segnata come tale.

Una parola senza corrispondenza resta sconosciuta e produce "non so". Da questa sessione non si raggiungono loinc.org, Wikidata e Hugging Face; le fonti si scaricano nel workflow su GitHub Actions, come NCIt.

## 3. Una procedura di addestramento per Gemma 4 e Nemotron

`melampo.evaluation.model_profiles` raccoglie in un posto i modelli previsti, con quanto è verificato e quanto viene solo dalle pagine dei fornitori:

| profilo | pesi (Hugging Face) | uso | addestramento LoRA | verificato qui |
|---|---|---|---|---|
| `gemma-4-31b` | `google/gemma-4-31B-it`, Apache 2.0 | 1 GPU da 80 GB | NeMo AutoModel `gemma4_31b_peft.yaml` (esempio NVIDIA su 8 GPU); QLoRA su 1 GPU stimato | ruoli via OpenRouter |
| `nemotron-3-super` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (anche FP8, NVFP4) | 2×80 GB in FP8 | ricetta LoRA NVIDIA, 1 nodo da 8 GPU | non ancora misurato |
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
- [NeMo Megatron-Bridge, Nemotron 3 Super](https://docs.nvidia.com/nemo/megatron-bridge/0.4.2/models/llm/nemotron3-super.html)
- [NeMo AutoModel, LLM coverage](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/llm/index.html)
- [UMLS, LOINC Italian (LNC-IT-IT)](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/LNC-IT-IT/metadata.html)
- [Università di Padova, la classificazione ICD-9-CM](https://sdb.unipd.it/laclassificazioneicd-9-cm)
- Hugging Face API, modelli `nvidia/*Nemotron-3*` e `google/gemma-4*` (13 ottobre 2026)
