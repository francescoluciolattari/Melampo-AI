# Rad-SpRL: 2.084 sedi anatomiche annotate da persone, e gpt-oss-120b in FP8 (12 ottobre 2026)

## I dati

Rad-SpRL (S. Datta e K. Roberts; Mendeley Data, DOI 10.17632/yhb26hfz8n.1) annota i ruoli spaziali di 2.000 referti di radiografia del torace IU X-ray (Open-i). La licenza è **CC BY 4.0**, letta dai metadati del dataset: si possono condividere e modificare i dati citando la fonte. Le etichette derivate stanno nel repository, con l'attribuzione.

Ogni relazione ha tre elementi:

- un **indicatore** spaziale, cioè la preposizione ("in", "at", "within");
- un **Trajector**, cioè il reperto;
- uno o più **Landmark**, cioè la sede anatomica.

In tutto ci sono 2.084 Landmark, scritti in 552 forme diverse, e 80 di questi sono discontinui ("right … lung" in "right and left lung"). Per definizione degli annotatori, ogni Landmark è un luogo del corpo in cui si trova qualcosa. Nel nostro schema il ruolo è quindi `structure`: la struttura è sede di un reperto.

Questi dati misurano due cose che il gold set silver non misura bene:

- i falsi allarmi di ogni lettore, cioè quando chiama "altro" una sede;
- la precisione dei link su sintagmi veri di referti radiologici.

Lo script è `scripts/rad_sprl.py`, con due comandi:

- **`convert`** produce le menzioni, i referti e le righe del gold set. Il ruolo è `structure`. La classe viene da `data/gold/rad_sprl_2026-10-12/classes.txt` oppure, se la stringa non è etichettata, resta `UNLABELLED`.
- **`measure`** fa girare il linker su ogni Landmark e riporta link accettati, giusti, sbagliati, da etichettare, astensioni e ruoli detti.

## Il linker sui Landmark

**Prima delle correzioni.**

- **Accettati:** 319 su 2.084, tutti giusti dopo il controllo a mano stringa per stringa (67 forme diverse).
- **Ruoli diversi da struttura:** uno solo, `procedure_site` su "Stable resection of distal left clavicle". È giusto nel nostro schema: la resezione è una procedura sulla clavicola.

**Le astensioni da recuperare.** Le astensioni sono 1.765. Quasi tutte sono corrette per le nostre convenzioni:

- "lungs", "thoracic spine", "thorax", "mediastinum" e "hemidiaphragm" non sono classi;
- "lung bases", "both lower lobes" e "right upper and lower lobes" nominano più di una struttura.

Una famiglia invece andava recuperata: costole numerate e lobi con una parola che colloca il reperto sulla struttura. Esempi: "posterior right 9th rib", "left 6th posterior rib", "medial right middle lobe", "lateral left clavicle", "anterior T10 vertebral body".

**Correzione universale (`AnatomyLinker.link`, `PLACING_WORDS`).** Le parole che collocano un reperto su una struttura senza cambiare quale struttura sia vengono lette attraverso:

- **le parole:** anterior, posterior, lateral, medial, posterolateral, mid, peripheral, perihilar, retrocardiac, più le forme italiane;
- **solo se** il nome, letto per intero, è sconosciuto;
- **il link** diventa `part_of`, e la traccia lo registra (`placing_words`);
- **mai se** la frase intera è un nome UBERON, per esempio "lateral ventricle", che è un ventricolo del cervello;
- "superior" e "inferior" sono esclusi, perché fanno parte di nomi (vena cava superiore, lobo inferiore).

**Dopo la correzione.**

| | prima | dopo |
|---|---|---|
| accettati | 319 | 361 (+42, tutti giusti al controllo) |
| sbagliati | 0 | 0 |
| limite superiore dell'errore al 95% (Clopper-Pearson) | — | 0,83% |
| ruoli diversi da struttura | 1 (giusto) | 1 |

Gli altri controlli:

- **Gold set silver:** una sola variazione, "prostata periferica", che passa da astensione a prostata `part_of` ed è giusta.
- **Controllo esterno:** nessuna variazione, perché le parole che collocano non sono nelle menzioni proposte.
- **Casi a rischio provati:** lateral ventricle, medial meniscus, lateral malleolus, anterior cruciate ligament e posterior tibial artery restano tutti astenuti.

**Cosa vale questo numero.** Le classi delle 90 stringhe accettate le ho scritte io (Claude), **dopo** aver visto le risposte del linker: è un controllo, non un'etichetta cieca. Inoltre la regola delle parole che collocano è nata su questi stessi dati ed è registrata come post-errore in `provenance.json`. Lo 0,83% è quindi un numero di sviluppo. Non certifica nulla.

## Il reticolo e il lettore sui Landmark

Su ogni Landmark ci si aspetta la lettura "struttura":

| braccio | prima | dopo la correzione dei tipi NCIt |
|---|---|---|
| reticolo dei blocchi | 81,9% | **92,2%** |
| lettore Kintsch | 98,5% | 98,5% |

**Il falso allarme principale del reticolo.** "degenerative changes of the (thoracic) spine" (114 casi), "enlargement of the cardiac silhouette" ed "ectasia of the thoracic aorta" venivano letti come `inherent_location`. La causa sta nei tipi NCIt delle teste:

- "Change" ed "Enlargement" sono classi di *Action* (il senso del verbo), che finivano nel tipo "activity";
- "Ectasia" è un *Pathologic Process* dentro "Biological Process".

**Correzione universale (`scripts/ncit_kinds.py`, `block_memory.json` rigenerato dalla stessa versione NCIt 2024-05-07).**

- **Pathologic Process (C16956) diventa `disease`:** un processo patologico è un reperto della struttura in cui avviene, non una sua funzione.
- **Action (C25404) diventa un tipo a sé, da cui non si legge alcun ruolo:** una parola d'azione non dice che ruolo ha la struttura.

**Effetto sul controllo esterno** (reticolo, righe giudicate):

| | prima | dopo |
|---|---|---|
| link giusti MedMentions di struttura letti come struttura | 701/901 | 723/901 |
| link giusti MedMentions di malattia letti come malattia | 657/758 | 673/758 |
| link giusti CRAFT cambiati | 230 | 211 |
| link giusti MedMentions cambiati | 161 | 141 |
| errori letti (MedMentions) | 8 | 6 |

I due errori persi erano presi per il motivo sbagliato. Il reticolo leggeva un verbo come testa: "brain **controls** vigilance" e "pancreas **helps** in planning".

**Gold set silver:** il reticolo passa dal 79,1% all'80,3% sui ruoli; il lettore resta all'82,3%.

**Falsi allarmi che restano** (non toccati, documentati):

- **"low lung volumes" (69):** è una convenzione nostra (misura di un organo, `inherent_location`); in Rad-SpRL il polmone è una sede.
- **"elevation of the hemidiaphragm" (23):** NCIt mette "elevation" tra le proprietà.
- **Altre teste tipizzate male da NCIt:** "PAIN IN THORACIC SPINE" (25), "appearance of the cardiac silhouette", "DISH", "narrowing". Per separare un reperto da una misura serve un tipo clinico della testa (UMLS o SNOMED), non NCIt da solo.

## gpt-oss-120b in FP8

Decisione di Frank: gpt-oss-120b si usa in FP8.

**Cosa cambia nel codice.**

- **Opzione `--quantization`** in `role_probe` e `phrase_probe` e nei due workflow, con predefinito `openai/gpt-oss-120b=fp8`.
- **Su OpenRouter** la precisione è un filtro rigido: `provider: {"quantizations": ["fp8"], "allow_fallbacks": false}`, che il client `OpenRouterChat` ora sa inviare.
- **Domanda di prova:** prima del run, una per modello. Se nessun fornitore serve FP8 il run si ferma con un messaggio chiaro.
- **Nel report** il nome del modello porta la precisione: `openai/gpt-oss-120b@fp8`.

**I pesi.** I pesi ufficiali sono MXFP4 (4 bit). L'FP8 è una conversione di terzi: su Hugging Face ci sono, per esempio, `amd/gpt-oss-120b-w-mxfp4-a-fp8` (pesi MXFP4, attivazioni FP8) e conversioni complete FP8 di singoli autori. In locale serve quindi un checkpoint FP8 scelto, fissato allo sha e verificato.

**L'hardware.** Circa 120 GB di pesi: 2 GPU da 80 GB oppure 1 H200 da 141 GB.

## Il workflow `role-probe`

Ora aggiunge un campione a seme fisso di Landmark Rad-SpRL. L'input è `rad_sprl`: 300 per default, 2084 per prenderli tutti. Rad-SpRL compare come strato `radsprl_en`, accanto agli item del gold set silver.

## Fonti

- Rad-SpRL, Mendeley Data, DOI 10.17632/yhb26hfz8n.1, CC BY 4.0.
- [OpenRouter, provider routing (quantizations, allow_fallbacks)](https://openrouter.ai/docs/features/provider-routing)
- Hugging Face API, ricerca "gpt-oss-120b-FP8" (12 ottobre 2026).
