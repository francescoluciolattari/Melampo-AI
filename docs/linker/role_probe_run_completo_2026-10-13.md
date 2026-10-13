# Role probe, run completo (13 ottobre 2026)

Il run è completo: 963 item (gold set silver e 300 Landmark di Rad-SpRL), 859 anatomici, cinque strati (E3C inglese e italiano, IU X-ray, PARROT italiano, Rad-SpRL). Modelli: gpt-oss-120b in FP8, Gemma 4 31B, Nemotron 3 Ultra. Mistral Small 4 è stato tolto dopo il primo run. Un solo item è rimasto senza risposta di Gemma. Il run è durato circa 5.000 secondi, con le risposte del run precedente riusate (`resume_run`).

È un set di sviluppo (silver). Le regole di accordo qui sotto le ho scelte guardando questi stessi dati, quindi i numeri sono ottimistici.

## Un modello da solo

| braccio | accuratezza | errori "sede o non sede" | "struttura" detta a torto |
|---|---|---|---|
| reticolo | 85,0% | 0 | 89 |
| lettore Kintsch | 88,1% | 3 | 88 |
| Nemotron 3 Ultra | **94,3%** | 0 | 4 |
| Gemma 4 31B | 91,1% | 7 | 4 |
| gpt-oss-120b FP8 | 87,8% | **43** | 4 |

- **Dare le letture del lettore al modello non migliora** (Nemotron 94,5%, Gemma 90,9%, gpt-oss 87,7% con 46 errori sede/non sede).
- **gpt-oss** sbaglia soprattutto su IU X-ray (45/76 giuste, contro 74/78 di Nemotron) e Rad-SpRL. Chiama "altro" o "non è una sede del corpo" parole come *cardiac silhouette*, *cardiomediastinal silhouette* e *goiter*. È una convenzione radiologica (la silhouette cardiaca è la sede), che il modello non conosce. Per i ruoli, da solo, non va bene.
- Le menzioni NOT_ANATOMY (lesioni, liquidi, cellule) sono lette come struttura 87 volte su 104 dal lettore e dal reticolo, 14 (Gemma), 23 (gpt-oss), 27 (Nemotron) dai modelli da soli.

## Due letture indipendenti che concordano, altrimenti a una persona

Il report calcolava l'accordo tra il lettore e il modello *che aveva visto* il lettore: sbagliato, perché il modello tende a ripetere la lettura del lettore. Il programma ora confronta il lettore con la risposta del modello da solo (`exact->agree:<modello>`) e con tutti i modelli insieme (`exact->agree_all`). I numeri qui sotto vengono dalle stesse risposte, ricalcolate fuori dal programma (il codice nuovo dà gli stessi conti).

| regola | a una persona | accuratezza sulle risposte | errori |
|---|---|---|---|
| lettore e Gemma | 20,5% | **99,27%** | 5 |
| lettore e Nemotron | 17,2% | **99,30%** | 5 |
| lettore, Gemma e Nemotron | 22,8% | **99,40%** | 4 |
| lettore con Gemma o con Nemotron | 14,9% | 99,18% | 6 |
| Gemma e Nemotron (senza lettore) | 8,7% | 95,5% | 35 |
| maggioranza di 3 modelli | 1,9% | 94,4% | 47 |
| come sopra, più "procedure_site" se tre modelli concordano | 12,7% | 98,0% | 15 |

Il lettore da solo conta: senza di lui i modelli ripetono gli stessi errori di convenzione (la silhouette, il torace). Con lui l'errore scende sotto l'1%.

**I 4 errori di "lettore, Gemma e Nemotron":**

- *uterus*, in "uterus size": il gold dice struttura, loro `inherent_location`, che è difendibile;
- *abdominal wall*, in "abdominal wall incisions": loro `procedure_site`, che è difendibile;
- *abdominal cavity*: la frase parla di "pelvic cavity"; il gold è dubbio;
- *un albero tracheobronchiale* (broncoscopia): il gold dice sede di procedura, loro struttura.

Almeno tre sono dubbi sull'etichetta, non sul linker.

**Cosa va a una persona (196 su 859):** 107 strutture, 84 sedi di procedura, 5 altre. Le sedi di procedura sono quasi tutte rimandate perché il lettore le legge come strutture: il modello le sa leggere (80 su 92), ma il lettore no.

## Conclusione operativa

Per lo stadio 6, il minimo è: **lettore e Gemma 4 31B concordano, altrimenti una persona.**

- È in una sola GPU, con licenza Apache 2.0.
- Costa un 20% di item a una persona.
- Aggiungere Nemotron non cambia quasi nulla (99,40% contro 99,27%) e richiede 8 GPU.
- gpt-oss-120b non entra nello stadio 6 per i ruoli.

Lavoro aperto: se il lavoro della persona sulle sedi di procedura (84) si può ridurre con un lettore migliore delle procedure, o con un LoRA su Gemma 4 31B (punto 2, serve una GPU). Questi numeri vengono da un set di sviluppo: una conferma richiede un gold set a doppio cieco.

## Gli errori residui, uno per uno (analisi della sera)

Lettore e Gemma sbagliano 5 item, lettore e Nemotron 5. Quattro sono in comune. Ciascuno ne ha uno suo: Gemma *dell'appendice*, Nemotron *piede torto congenito*. I due modelli sbagliano quanto l'altro; la differenza è che Nemotron concorda più spesso col lettore. Nemotron accetta giusti 47 item che Gemma manda a una persona, Gemma 19 che Nemotron manda a una persona. Limite superiore dell'errore al 95%: 1,5% per tutti e due. Su 700 item non si distinguono.

**Tipo A: l'etichetta del gold set è sbagliata (2, in comune).** Le etichette silver le ho scritte io, prima del run.

- *uterus* in "The uterus size was 6cm × 4cm × 3cm": per la convenzione del README ("heart size" ha ruolo `il`), il ruolo è `inherent_location`. Lettore e modelli hanno ragione.
- *abdominal wall* in "the abdominal wall incisions were located…": l'incisione è una procedura sulla parete, quindi il ruolo è `procedure_site`. Lettore e modelli hanno ragione.

**Tipo B: la menzione ha due ruoli e lo schema ne ammette uno (3).** La procedura e un reperto ("nella norma", "no findings", "appaiono normali") cadono sulla stessa struttura. La procedura sta fuori dal sintagma: in un verbo di un'altra proposizione ("the pelvic cavity **was explored** and no specific findings were found in the abdominal cavity"), nel soggetto di una relativa ("una **broncoscopia** che mostrava un albero tracheobronchiale nella norma"), oltre una coordinazione ("**liberazione** del cieco e dell'appendice che appaiono normali"). Il lettore legge il blocco anatomico da solo ("dell appendice") e non arriva alla procedura. Il modello vede il predicato più vicino, il reperto, e risponde struttura. Per *dell'appendice* Nemotron e gpt-oss risalgono alla "liberazione" e Gemma no: è l'errore in più di Gemma. Struttura o sede di procedura sono entrambe vere; in due casi su tre il gold scelto da me è dubbio.

**Tipo C: la parola anatomica sta dentro il nome di una malattia (1, Nemotron).** "intervento chirurgico di **piede torto congenito**". "Piede torto congenito" è una malattia: il piede vi compare come pezzo del nome (`inside_a_name`), oppure è la sede dell'intervento (`procedure_site`, il gold). Il lettore legge un blocco anatomico. Nemotron dice struttura, Gemma e gpt-oss sede di procedura.

**Nessuno dei cinque è un errore di sede contro non-sede.** Nessuno è un errore di addestramento del modello: due sono etichette mie da correggere, tre sono menzioni con due ruoli veri, uno è il confine tra nome di malattia e parte del corpo. Le correzioni utili sono nello schema, non nei pesi:

- consentire due ruoli (sede di procedura **e** sede di un reperto), o una regola di precedenza scritta;
- far vedere al lettore la procedura che regge la frase anche fuori dal sintagma (verbo, soggetto della relativa, coordinazione);
- correggere le due etichette, registrandolo come correzione fatta dopo aver visto le risposte.

## Tipo B e tipo C rivisti: perché il lettore sbaglia (seconda analisi)

Frank ha chiesto tre cose. Perché il lettore sbaglia, se il problema è la divisione dei sintagmi. Perché, se io vedo l'errore, non lo vedono il lettore o il modello. Se il tipo C non sia piuttosto un problema di direzione della lettura, dal largo allo stretto.

### Cosa fa davvero il lettore

Ho fatto girare reticolo e lettore sulle frasi, anche con varianti:

| frase | blocco letto | lettura |
|---|---|---|
| "liberazione del cieco e dell'appendice" | `dell appendice` | struttura 1,00 |
| "Liberazione dell'appendice che appare normale" | `liberazione dell appendice` | struttura 1,00 |
| "Asportazione dell'appendice" | `asportazione dell appendice` | struttura 1,00 |
| "intervento chirurgico del piede" | `intervento chirurgico del piede` | struttura 1,00 |
| "La TC del torace mostrava…" | `tc del torace` | struttura 1,00 |

**Il sintagma viene ampliato** ("asportazione dell appendice", "tc del torace"). **Il lettore non sa che cosa sia la testa:** "asportazione", "liberazione", "intervento", "ecografia", "biopsia", "TC" non hanno un tipo nella memoria dei blocchi, che viene da NCIt e quindi è in inglese. Quando la testa è sconosciuta, il blocco prende il tipo della parola che conosce, quella anatomica, e la lettura esce "struttura" con certezza piena. **Il dubbio non emerge mai.** Per questo, unendo i blocchi, il lettore non si accorge dell'errore.

**Due difetti, misurati.**

1. **Grammatica.** Nella sonda (e in `build_ci_resources.py`) il reticolo è costruito senza grammatica, quindi legge l'italiano con la testa a destra, come l'inglese. La grammatica italiana esiste (`grammar.py`, testa a sinistra) ma non viene passata.
2. **Vocabolario.** Mancano le parole italiane di procedura, e anche alcune inglesi ("radiograph", "sternotomy").

Prova sugli 859 item anatomici, solo lettore:

| variante | accuratezza | sedi di procedura giuste | "struttura" detta a torto |
|---|---|---|---|
| come nel run | 87,9% | 6/92 | 88 |
| con la grammatica della lingua | 86,7% | 7/92 | 81 |
| grammatica + suffissi di procedura (-scopia, -grafia, -tomia, -ectomia, -stomia, -plastica, -centesi, biopsia) e sigle d'esame (TC, TAC, RM, RMN, RX, ECO, PET, CT, MRI) | **90,9%** | **44/92** | **51** |

La grammatica da sola peggiora, e il motivo conta: con la testa a sinistra la parola *precedente* tipizza il blocco, e parole come "su" o "EO" hanno tipi NCIt casuali ("chemical"). Ne escono errori nuovi: "su linfonodi regionali" letto come nome di sostanza. Il "sintagma" del reticolo è una finestra di parole, ciascuna con un tipo. Non è una struttura sintattica: il reticolo non sa quale parola regge quale.

### Lettura dal largo allo stretto

Frank propone di leggere dal largo allo stretto: prima la frase, poi chi regge chi, poi la parola. È quello che fa un analizzatore a dipendenze. Ho provato spaCy `it_core_news_md` (MIT):

| menzione | catena dei reggenti |
|---|---|
| appendice | appendice →conj→ cieco →nmod→ **liberazione** |
| piede (torto congenito) | piede →nmod→ **intervento** →obj→ esegue |
| torace | torace →nmod→ **TC** →nsubj→ mostrava |
| addome | addome →nmod→ **ecografia** |
| albero tracheobronchiale | albero →obj→ **mostrava** →relcl→ broncoscopia |
| membrane basali | membrane →obl→ **evidenziava** |

Il reggente dice il ruolo: un nome di procedura che regge la sede porta a `procedure_site`; un verbo di reperto ("mostrava", "evidenziava", "found") porta a struttura, come per la silhouette cardiaca. La coordinazione eredita il reggente del primo membro: è il caso dell'appendice. Restano necessarie le parole italiane: un reggente sconosciuto deve dare *nessuna lettura*, mai "struttura".

### Gli errori riclassificati

| menzione | prima | dopo l'analisi |
|---|---|---|
| uterus (size) | A | A, etichetta sbagliata |
| abdominal wall (incisions) | A | A, etichetta sbagliata |
| abdominal cavity | B | **A**: "no findings were found in the abdominal cavity", cioè sede di un reperto (assente). Il gold "procedure_site" è sbagliato |
| albero tracheobronchiale | B | **A**: "la broncoscopia mostrava un albero nella norma" ha la stessa forma di "the radiograph showed enlargement of the cardiac silhouette", che il gold chiama struttura. Ho applicato male la mia convenzione |
| dell'appendice (Gemma) | B | **errore vero**: il reggente "liberazione" è oltre una coordinazione. Il lettore non lo conosce; Gemma si lascia guidare dalla relativa "che appaiono normali" |
| piede torto congenito (Nemotron) | C | **errore vero, stesso meccanismo**: il reggente "intervento" è sconosciuto al lettore. Nemotron risponde "struttura" (il piede come sede della malattia), ma il reggente è l'intervento. Frank ha ragione: che un nome di malattia contenga una parola anatomica non è il problema |

Gli errori veri sono 2 su 6 e hanno la stessa causa: il reggente della menzione sta fuori dalla portata del lettore (vocabolario italiano assente, nessuna sintassi), e il modello sceglie il predicato più vicino.

### Perché io lo vedo e loro no

Io leggo la sintassi (chi regge chi) e conosco le parole italiane. Il lettore non ha nessuna delle due. Il modello le ha entrambe, ma non conosce le nostre convenzioni (il run era con `examples=0`) e tende al predicato più vicino. Rimedi coerenti con il principio "deterministico solo dove c'è la corrispondenza":

1. un analizzatore a dipendenze che dia al lettore il reggente della menzione, seguendo la coordinazione;
2. un vocabolario italiano dei tipi delle teste (procedure, esami, dispositivi) da una fonte citabile, più la morfologia; un reggente sconosciuto non dà nessuna lettura;
3. al modello, la catena dei reggenti e gli esempi delle nostre convenzioni.

### Nemotron 3 Super

Nei benchmark generali è sotto Gemma 4 31B (indice AA 12,8 contro 14,7; GPQA 80,0% contro 85,7%; IFBench 71,5% contro 75,6%). Sul nostro compito non l'abbiamo misurato. Architettura: 120B, 12B attivi, Mamba-Transformer ibrido MoE, NVIDIA Open License. **È addestrabile:** NeMo Megatron-Bridge ha ricette per il fine-tuning completo e per LoRA, lanciate su un nodo da 8 GPU. Un QLoRA su una sola GPU, con l'architettura ibrida Mamba, non è documentato.
