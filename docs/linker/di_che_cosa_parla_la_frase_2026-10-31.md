# Di che cosa parla la frase: la struttura come riferimento (31 ottobre 2026)

**Il problema.** Nei due campioni HPO, 21 reperti inglesi su 292 erano attribuiti alla struttura sbagliata.
- **Il linker aveva collegato bene la struttura**: "linfonodi ingranditi lungo l'arco aortico" nomina davvero
  l'arco.
- **L'interprete creava il reperto a ogni struttura nominata** in una frase che descrive qualcosa, senza chiedere
  di che cosa la frase parla.
- **Il lettore non veniva consultato**, perché il lessico decideva prima.
- **Lo schema dei ruoli non aveva la lettura giusta.** Per la domanda del lettore, del modello e della guida dei
  radiologi l'arco era "struttura", ed era vero.

**La decisione di Frank: opzione A.** Il metodo è quello del lettore umano, non una regola deterministica. Il
lettore di Kintsch e il modello leggono la frase ciascuno per conto suo, e devono concordare su ciò che cambia
l'esito.

## Che cosa è stato costruito

| Pezzo | Che cosa fa | Dove |
|---|---|---|
| Ruolo `landmark` (riferimento) | la struttura colloca un'altra cosa e la frase non dice nulla di lei; può stare con altri ruoli (`structure+landmark`) | lettore, modello, gold set, guida |
| Relazione spaziale | dalle parole prima della menzione: contenimento ("in", "of", "nel", "a carico di") contro vicinanza o esterno ("near", "along", "between", "laterally to", "vicino al", "lungo", "tra"), più direzione ("into") e oggetto di un verbo. Le parole sono gli indicatori spaziali dei referti (Rad-SpRL), raggruppati per tipo di relazione (il calcolo delle connessioni fra regioni, RCC) | `memory/spatial_relation.py` |
| Il lettore, secondo stadio | se il primo stadio legge "struttura", una seconda rete pesa le prove lessiche (la parola è una struttura del corpo: sostengono la struttura, e il riferimento a metà peso) contro la relazione (vicinanza: quanto tutte le prove lessicali). Le letture del primo stadio (procedura, nome, misura) non vengono mai toccate | `ci_reader.CIReader.read` |
| Il modello | la stessa domanda con il metodo del lettore: il passo 3 chiede che cosa colloca la struttura, se è solo un riferimento (campo `placed`). Non vede la risposta del lettore | `llm_reading` |
| La decisione | lettore e modello concordano che il reperto è della struttura → resta; concordano che non lo è → nessun reperto lì, la frase va alla persona con ciò che colloca; si contraddicono, o uno non sa → il reperto resta e va alla persona. **Il lettore da solo non toglie mai un reperto.** Due modelli contano come un flusso | `memory/finding_owner.py`, `imaging/interpreter.py`, `ReportReaders` |
| Il gold set | `l` e i ruoli multipli con `+` nel silver; `check` accetta `landmark` e più ruoli; la guida e il protocollo dei radiologi hanno la riga nuova e la seconda domanda per il dubbio | `data/gold/silver_2026-10-12`, `docs/gold_set_*` |

Il link non cambia: "arco aortico" resta collegato all'aorta. Per la misura "sede o no" il riferimento conta come
sede, quindi i numeri del role probe restano confrontabili. I campioni HPO si ricostruiscono identici (verificato),
perché il loro campionamento non usa la nuova decisione.

## Misure locali (lettore da solo)

**Silver set**, 576 righe con un ruolo (E3C e PARROT; IU X-ray non è raggiungibile da qui):

| | senza relazione | con relazione |
|---|---|---|
| ruoli giusti | 418 | 422 |
| errori sede o no | 92 | 92 |
| riferimenti del gold trovati | 0 su 12 | 6 su 12 |
| "riferimento" letto a torto | — | 1 ("placche lungo l'asse femoro-popliteo": sono nella parete) |

**I due campioni HPO** (la lista dei reperti di un'altra struttura viene dalle note delle etichette, scritte prima;
`data/gold/finding_owner_2026-10-31/elsewhere.txt`):

| | di un'altra struttura | alla persona | ancora attribuiti alla struttura | reperti giusti alla persona | reperti giusti tolti |
|---|---|---|---|---|---|
| sviluppo, inglese | 9 | 2 | 7 | 2 | 0 |
| sviluppo, italiano | 3 | 0 | 3 | 0 | 0 |
| verifica, inglese | 12 | 7 | 5 | 2 | 0 |
| verifica, italiano | 2 | 2 | 0 | 0 | 0 |

Il lettore da solo prende i casi in cui la relazione è nelle parole ("lungo", "tra", "vicino", "nell'area di",
"laterale al"). Gli altri chiedono di capire la frase:
- "gli ureteri sono dilatati fino al loro sbocco in vescica": la vescica è il punto d'arrivo;
- "ispessimento della pleura parietale nel lobo inferiore": la pleura è il soggetto;
- "ipertrofia del lobo medio" della prostata, letta come il lobo medio del polmone: è un errore del linker;
- "the liver is enlarged (right lobe…)": anche questo è del linker.

Questo è il lavoro del modello, ed è ciò che misura il workflow.

## Criteri del run con i modelli (fissati prima)

Workflow `owner-probe`, con Gemma 4 31B e Nemotron 3 Ultra.
1. Dei 26 reperti di un'altra struttura, nessuno resta attribuito alla struttura: ognuno è tolto (lettore e
   modello d'accordo) o va alla persona.
2. Fra i reperti giusti, nessuno tolto a torto, o al più due, dichiarati.
3. Il role probe, rilanciato con il prompt nuovo, non peggiora su "sede o no": inglese con Nemotron allo 0,6%.

Le risposte in cache del role probe non valgono più: il prompt è cambiato.

## Limiti

- Gli indicatori spaziali sono una lista. È conoscenza di lingua, pesata come un nodo, non una decisione, ma
  resta incompleta: "dorsally to" c'è, "paravertebralmente" no.
- "Lungo" può voler dire dentro una struttura tubolare ("placche lungo l'arteria"). Lo distingue il modello, non
  il lettore.
- Due dei 26 casi sono errori del linker (lobo medio della prostata, lobo del fegato). La decisione sul reperto li
  manda alla persona solo se il modello li legge. La correzione vera è nel linker: il soggetto della frase come
  previsione.
- Le etichette sono di un solo etichettatore (Claude). La lista dei reperti di un'altra struttura va rivista da un
  radiologo.

## Il run con i modelli (31 ottobre, sera)

**Owner probe** (Gemma 4 31B e Nemotron 3 Ultra con il lettore), sui 26 reperti di un'altra struttura e sui 358
reperti giusti:

| | tolti (lettore e modelli d'accordo) | alla persona | ancora attribuiti alla struttura |
|---|---|---|---|
| reperti di un'altra struttura (26) | 2 | 13 | **11** |
| reperti giusti (358) | **0** | 7 | 351 |

Con il lettore da solo: 0, 11 e 15 per i reperti di un'altra struttura; 0, 4 e 354 per quelli giusti.

**Esito rispetto ai criteri fissati prima.**
- **Criterio 1, non passa**: 11 reperti di un'altra struttura restano attribuiti alla struttura, contro 26 prima.
- **Criterio 2, passa**: nessun reperto giusto viene tolto.
- **Criterio 3, passa per Gemma.** Il role probe è stato lanciato con i modelli predefiniti del suo workflow
  (gpt-oss-120b, Gemma 4 31B, Mistral Small 4), quindi il confronto con Nemotron non c'è.
  - Gemma, in inglese, manda alla persona lo 0,6% degli item con la domanda semplice e lo 0,8% con il metodo,
    contro l'1,9% del 14 ottobre.
  - Gli item su cui lettore e modello sono d'accordo con l'esito sbagliato restano 0 per tutti i modelli.

**Perché ne restano 11.**
1. **I modelli danno quasi sempre due ruoli.** Rispondono `["landmark", "structure"]` oppure
   `["structure", "landmark"]`, e la regola "riferimento solo se *structure* non c'è" li legge come "il reperto è
   qui".
   - Prendere solo il primo ruolo è stato provato in esplorazione: restano 7 errori, ma 3 reperti giusti
     vengono tolti.
   - Basta un "landmark" fra i ruoli: restano 2 errori, ma 145 reperti giusti vanno alla persona.
   - Il problema è la domanda: dai ruoli si ricava l'esito, invece di chiederlo.
2. **Il modello non sa a quale struttura il linker ha collegato la parola.** "Ipertrofia del lobo medio" della
   prostata: il modello dice "struttura", e ha ragione, ma il linker l'ha collegata al lobo medio del polmone. Lo
   stesso vale per il lobo del fegato. Sono 3 degli 11.
3. **Alcune etichette sono discutibili** e vanno date a un radiologo:
   - gli ureteri dilatati "fino allo sbocco in vescica";
   - la prostata che "sporge nella vescica";
   - l'oggetto "alla biforcazione" del bronco del lobo;
   - "ombra dell'arteria succlavia" in diagnosi differenziale.

**Il gold del role probe.** Lo strato Rad-SpRL (300 item) ha il ruolo "structure" per costruzione. Con il ruolo
nuovo una parte di quelle righe è un riferimento: "opacità fra la 7ª e l'8ª costa", "fluido lungo la scissura",
"palpabile a 2 cm dall'arcata costale". Su quello strato il lettore e Gemma dicono "riferimento" in 23 item, 7
volte tutti e due. Quelle righe vanno rilette con lo schema nuovo prima di contare questi casi come errori.

**Proposta per il passo successivo**, da fissare prima del run:
- chiedere al modello l'esito direttamente, nominando la struttura collegata dal linker: "il reperto è di
  <struttura collegata>? sì / no / anche". I ruoli restano come lettura;
- rileggere lo strato Rad-SpRL con lo schema nuovo;
- far rivedere a un radiologo le 26 etichette.

## Perché il radiologo vede gli 11 casi e noi no (1 novembre 2026)

Ho riletto gli 11 casi uno per uno. Per ciascuno ho guardato il referto intero (PARROT), la classe scelta dal linker
e le risposte complete dei due modelli: frase, reggente, predicato, letture, ruoli.

| Caso | Che cosa vede il radiologo | Che cosa facciamo noi |
|---|---|---|
| "Minimal hypertrophy of the **middle lobe**, indenting the bladder floor" (RM prostatica, sezione "TZ:"), in inglese e in italiano | è il lobo medio della **prostata**: l'esame è una RM della prostata, la frase sta nella sezione della zona di transizione, e il polmone non è nemmeno nel campo di vista | il linker lo collega al lobo medio del polmone (fra le classi non ci sono i lobi della prostata). Il modello risponde bene: Gemma scrive "this specific part of the prostate". Ma nessuno gli ha detto a che cosa il linker ha collegato la parola |
| "The liver is slightly enlarged (**right lobe** in the mid-clavicular line 146 mm)" (ecografia dell'addome) | il lobo destro **del fegato**: il fegato è il soggetto della frase | il linker compone "right lobe … mid" nel lobo medio del polmone destro. Il modello risponde "struttura" pensando al fegato (Nemotron: "an anatomical part of the liver"), quindi risponde a un'altra domanda |
| "The ureters are dilated … up to their entry into the **bladder**", in inglese e in italiano | il reperto è degli **ureteri**: la vescica dice solo fin dove arriva la dilatazione. Della vescica il referto parla in un'altra frase ("The bladder is empty… walls diffusely thickened") | i modelli dicono "structure" e "landmark": la vescica è "il punto d'arrivo" del reperto |
| "hypertrophy of the transitional zone partially protruding into the **bladder**" (due referti) | il reperto è della **prostata**; anche qui la vescica ha la sua frase ("Bladder is moderately distended, with slightly thickened walls") | i modelli dicono tutti e due i ruoli. Nemotron scrive "the bladder itself is not described as having any intrinsic pathology" e mette comunque "structure" per primo |
| "thickening of the **parietal pleura** in the lower lobe" | la pleura parietale non fa parte del polmone: riveste la parete toracica, e "nel lobo" vuol dire "all'altezza del lobo" | i modelli dicono "structure", perché il reperto è "IN" il lobo. È la *nostra* definizione a dirglielo: "a finding IN or OF the structure is structure" |
| "a foreign body lodged at the bifurcation of the left main bronchus and the left **upper lobe** bronchus" | il reperto è un **corpo estraneo nel bronco**, non un'anomalia del lobo | il linker ha preso "left upper lobe" dentro il nome del bronco. Tutti e due i modelli rispondono "landmark", poi "structure", e la regola legge "qui" |
| "Partial opacity of the left apex (differential diagnosis: … **shadow of the left subclavian artery**)" | in radiografia l'"ombra dell'arteria" è l'anatomia normale che imita una lesione, ed è un'**ipotesi** di diagnosi differenziale | "differential diagnosis" non è fra le espressioni di ipotesi; "shadow of" viene letto come un reperto dell'arteria |
| "immagine adiposa **dorsalmente al** III medio del corpo sternale" | sta **dietro** lo sterno, nel mediastino | il lettore legge "del" dopo "III medio" come contenimento ("terzo" non è fra le parole di parte); i modelli dicono "landmark" e poi "structure", e la regola legge "qui" |

**Che cosa fa il radiologo che noi non facciamo.**

1. **Parte dal reperto, non dalla struttura.** Prima si chiede che cosa c'è di anomalo (gli ureteri dilatati, la
   zona di transizione ipertrofica, la pleura ispessita, un corpo estraneo) e solo dopo dove sta e di che organo
   fa parte. Noi chiediamo il contrario: "che ruolo ha *vescica*?". Così una struttura che compare nella frase
   sembra sempre coinvolta.
2. **Sa che cosa appartiene a che cosa.** La pleura parietale non è del polmone. Il bronco lobare non è il lobo. Il
   lobo medio in una RM della prostata è della prostata. È conoscenza anatomica (UBERON ha le relazioni *parte
   di*), e nella nostra lettura non entra.
3. **Legge il referto intero.** Conosce l'esame e il suo campo di vista (il polmone non è in una RM pelvica), la
   sezione ("TZ:"), il soggetto della frase ("The liver…"), e sa che ogni organo ha la sua frase: della vescica il
   referto parla altrove. Noi leggiamo una frase isolata.
4. **Distingue l'ipotesi dal reperto.** Una diagnosi differenziale non afferma nulla. "Ombra di" in radiografia è
   anatomia normale.

**Che cosa sbagliamo noi, oltre ai limiti dei lettori.**
- **La nostra definizione ha spinto i modelli.** Ho scritto che "landmark" vale "anche" quando ciò che è collocato
  sta nella struttura, e che un reperto "IN" la struttura è "structure". Quindi "sporge *nella* vescica" e
  "ispessimento della pleura *nel* lobo" sono diventati "structure", come la definizione chiedeva.
- **Al modello non diciamo a quale struttura il linker ha collegato la parola**, quindi non può accorgersi di un
  errore del linker.
- **Dai ruoli ricaviamo l'esito, invece di chiederlo**, e un "structure" in seconda posizione basta a dire "qui".

**Che cosa ne segue, caso per caso.**
- Partire dal reperto, e chiedere l'esito nominando la struttura collegata, risolve i 7 casi di ureteri, prostata,
  pleura, bronco e sterno.
- Il campo di vista dell'esame e il soggetto della frase, come previsione del linker, risolvono i 3 casi del lobo.
- La diagnosi differenziale fra le espressioni di ipotesi risolve l'arteria succlavia.

Sono ipotesi tratte da questi 11 casi, quindi vanno misurate su un campione che non li contiene.

## La lettura nuova: dal reperto, con il referto e l'anatomia (1 novembre 2026, fissata prima del run)

Frank ha detto di procedere. Che cosa cambia:

**La domanda al modello** (`llm_reading.owner_prompt`). Si chiede l'esito, nell'ordine del radiologo:
1. che cosa la frase descrive;
2. a quale struttura appartiene;
3. come sta la struttura collegata rispetto a quella: la stessa, una sua parte, solo un riferimento, una struttura
   diversa da quella che le parole nominano, o solo un'ipotesi;
4. sì, no o non so.

Il modello conosce l'esame, la sezione del referto e la struttura a cui il linker ha collegato le parole. Non
vede la risposta del lettore. La definizione di `landmark` è riscritta senza il "anche", che spingeva i modelli
a rispondere "structure".

**Il lettore** porta la conoscenza del radiologo, ogni pezzo come prova contro l'attribuzione
(`finding_owner`):
- **la relazione spaziale e il secondo stadio**, come prima;
- **l'organo**: il segmento della frase prima della menzione, o la testa che la segue, nomina una struttura
  (lessico del linker e UBERON) che appartiene a un altro organo. Un esempio sono gli ureteri e la vescica. Una
  struttura che UBERON non mette in nessun organo, come un corpo vertebrale o una parete, non conta;
- **il campo di vista**: una struttura fuori dall'area dell'esame, letta dal nome dell'esame (`exam_area`). È il
  caso del polmone in una RM della prostata;
- **la diagnosi differenziale**: la struttura è nominata solo fra le spiegazioni possibili. Vale nella parentesi
  della differenziale o fino alla fine della frase. "Diagnosi differenziale" è entrata anche fra le espressioni
  di ipotesi del lettore (`assertion`).

La decisione resta quella di prima:
- il reperto viene tolto solo se il lettore e i modelli sono d'accordo;
- se si contraddicono, decide una persona;
- il lettore da solo non toglie mai un reperto.

**Sui due campioni già visti**, con il solo lettore, i reperti di un'altra struttura mandati alla persona passano
da 11 a 17 su 26. I reperti giusti mandati alla persona restano 4 su 358. È una misura di sviluppo.

**Il campione nuovo** (`data/gold/finding_owner_2026-11-01/labels.txt`): 209 reperti di referti PARROT mai usati
(seme 20261101), 200 inglesi e 9 italiani, 111 referti, 55 strutture. Le etichette sono state scritte prima di
qualunque lettura: 24 `elsewhere` e 185 `here`; quelle con "?" sono per un radiologo. Gli italiani sono pochi
perché il lettore trova pochi reperti nei referti italiani di PARROT che non sono già stati usati.

**Criteri sul campione nuovo**, con il lettore e i due modelli (Gemma 4 31B e Nemotron 3 Ultra):
1. nessuno dei 24 reperti di un'altra struttura resta attribuito alla struttura;
2. al massimo 2 dei 185 reperti giusti vengono tolti;
3. al massimo il 10% dei reperti giusti (18) va alla persona.

Il role probe va rilanciato, perché la definizione di `landmark` è cambiata.

**Il lettore da solo sul campione nuovo** (misurato dopo aver congelato le etichette e i criteri, senza
correzioni successive):

| | reperti di un'altra struttura | alla persona | ancora attribuiti | reperti giusti alla persona |
|---|---|---|---|---|
| inglese | 23 | 8 | 15 | 2 su 176 |
| italiano | 1 | 0 | 1 | 0 su 8 |

Sui casi già visti il lettore da solo ne prendeva 17 su 26; su quelli nuovi ne prende 8 su 24. La conoscenza
aggiunta oggi pesa meno su casi che non ha visto, quindi gran parte del lavoro resta ai modelli: lo dirà il run.

Il campione di sviluppo HPO si ricostruisce con gli stessi item. Cambia solo la certezza di un item
(parrot-2288, ora `possible`), per via della "diagnosi differenziale" fra le espressioni di ipotesi.

## Il run con i modelli sul campione nuovo (1 novembre 2026)

**Owner probe, lettore con Gemma 4 31B e Nemotron 3 Ultra.**

| | reperti di un'altra struttura: tolti / alla persona / **ancora attribuiti** | reperti giusti: **tolti** / alla persona |
|---|---|---|
| **campione nuovo**, inglese (23 + 176) | 5 / 17 / **1** | **0** / 19 |
| campione nuovo, italiano (1 + 8) | 0 / 1 / 0 | 0 / 0 |
| campione HPO di sviluppo, già visto (12 + 188) | 5 / 7 / 0 | 0 / 13 |
| campione HPO di verifica, già visto (14 + 170) | 9 / 5 / 0 | **2** / 11 |

**Esito rispetto ai criteri fissati prima, sul campione nuovo.**
- **Criterio 1, non passa per un caso.** Resta attribuito "In the mesentery of the small intestine, …
  lymph nodes": i modelli dicono che il mesentere è una parte dell'intestino. L'etichetta è mia e va fatta
  rivedere.
- **Criterio 2, passa:** nessun reperto giusto viene tolto.
- **Criterio 3, non passa per un caso:** vanno alla persona 19 reperti giusti su 184, il 10,3% contro il 10%
  ammesso.

Rispetto a prima il cambio è grande: sui casi nuovi i reperti di un'altra struttura attribuiti in silenzio sono 1
su 24, mentre sui casi già visti erano 11 su 26.

**I 19 reperti giusti mandati alla persona.**
- In 17 casi Gemma e Nemotron non sono d'accordo, o uno dei due non sa, e la regola li tratta come un flusso
  solo.
- In 2 casi il lettore legge un riferimento: "wall thickening *in the area of* the descending colon" e
  "osteolytic foci *in the projection of* the femoral head". Qui i modelli hanno ragione.
- Diversi di questi casi dipendono dalla mia etichetta, e la lettura dei modelli è difendibile:
  - versamento dell'**articolazione** dell'anca, mentre la classe del linker è l'**osso** dell'anca;
  - "L3 over L4": la vertebra L4 fa da riferimento;
  - "history of thyroid cancer surgery", dove il reperto è il dolore spinale;
  - il rene destro nominato solo come confronto ("the left kidney is longer").
- I 2 reperti tolti nel campione di verifica sono anch'essi casi discutibili: "blood … extending around the right
  kidney" e "a shadow in the projection of the second rib". Lì lettore e modelli concordano che si tratta di un
  riferimento, contro la mia etichetta.

**Il role probe** (gpt-oss-120b, Gemma 4, Mistral Small 4).
- **"Sede o no" non cambia:** gli item su cui lettore e modello sono d'accordo con l'esito sbagliato sono 0 per
  tutti i modelli, e Gemma manda alla persona lo 0,2–0,8% degli item.
- **I ruoli fini peggiorano.** Gemma passa dall'89% al 77% di ruoli giusti con la domanda semplice, e dal 91% all'80%
  con il metodo.
  - La causa è un mio errore: riscrivendo la definizione di `landmark` ho tolto anche la frase "un reperto IN o
    DI la struttura è structure". I modelli leggono ora come riferimento anche "opacità *nel* lobo superiore" e
    "granuloma *nel* lobo inferiore".
  - La frase è rimessa, senza l'"anche" che l'aveva causata.
- **Una parte dei 125–142 "riferimento" è vera**, ed è un difetto del gold dello strato Rad-SpRL: "lucency under
  the hemidiaphragm", "density overlying the first rib", "linfonodi all'ilo epatico". Quelle righe vanno ancora
  rilette con lo schema nuovo.

**Proposta.** Tenere la lettura: sui casi nuovi non toglie nessun reperto giusto. Prima di un altro run:
1. un radiologo rivede le etichette discusse sopra e le 24 + 26 dei due elenchi;
2. lo strato Rad-SpRL del role probe va riletto;
3. il role probe va rilanciato con la definizione corretta.

## Perché i modelli non sono d'accordo, e perché sbaglia il lettore (1 novembre 2026)

Ho riletto i 19 reperti giusti mandati alla persona e il caso mancato, con le risposte complete dei due modelli
(reperto, struttura che lo porta, relazione, risposta, motivo).

| Che cosa succede | Casi | Esempio | Che cosa dicono i modelli |
|---|---|---|---|
| **La frase non ha un reperto**: un titolo o un'intestazione | 5 | "BRAIN CT SCAN", "Abdominal Aorta:", "At L4-L5," | tutti e due scrivono "finding: none"; poi Gemma risponde "uncertain" e Nemotron "yes". La domanda non ha senso, perché non c'è niente da attribuire |
| **La struttura è coinvolta, ma non porta il reperto**: il reperto si estende fino a lei, la comprime, o sta al livello che lei forma | 5 | "lesion … extending down to the cervical cord"; "mass effect on the pons, midbrain"; "uncovertebral joints at C5/C6"; "Level L5-S1: no foraminal stenosis" | Gemma dice "part", sì; Nemotron "reference", no. Fra le relazioni che offriamo manca "coinvolta" |
| **Che cosa vuol dire la classe del linker** | 2 | versamento e artrosi dell'articolazione dell'anca, collegati a `hip_left` | abbiamo mostrato al modello il nome "left innominate bone", l'osso dell'anca. Nemotron: "il reperto è dell'articolazione, non dell'osso", quindi no. Gemma: "l'articolazione comprende l'acetabolo", quindi sì |
| **Una tabella di misure molto lunga** | 1 | "LA / LAA" in mezzo ai numeri dell'ecocardiogramma | Nemotron prende come reperto una frase sulla valvola mitrale, molto più avanti |
| **Conoscenza anatomica sbagliata di un modello** | 1 | "the cardiac silhouette shows the shadow of an implanted aortic valve" | Gemma: "la valvola aortica fa parte dell'aorta, non del cuore" (sbagliato) |
| **I due modelli dicono no e la mia etichetta dice sì** | 3 | "L3 over L4"; "history of thyroid cancer surgery … spinal pain"; "left kidney longer (… right kidney 10.5 cm)" | tutti e due: il reperto è di L3, del dolore spinale, del rene sinistro; la struttura marcata è solo riferimento, storia o confronto |
| **Mancato** | 1 | "In the mesentery of the small intestine, … lymph nodes" | tutti e due: il mesentere fa parte dell'intestino |

**Gli ultimi tre casi della tabella vengono dalla domanda.** Chiediamo "il reperto è di questa struttura?", e il
modello sceglie il reperto principale: lo scivolamento di L3, il dolore, il rene sinistro più lungo. Ma la frase dice
anche qualcosa della struttura marcata:
- la tiroide è stata operata;
- il rene destro misura 10,5 cm;
- L4 è la vertebra sotto lo scivolamento.

Il radiologo, per ogni struttura nominata, si chiede che cosa la frase dice di lei. La domanda va ancorata alla
struttura collegata, su tutto ciò che la frase afferma.

**Perché sbaglia il lettore (2 casi).**
- "mild **wall** thickening *in the area of* the descending colon". "In the area of" è fra gli indicatori di
  vicinanza, ma ciò che la frase colloca è una *parete*, cioè una parte della struttura stessa. Quando la cosa
  collocata è una parte (parete, segmento, ansa), la relazione è di contenimento.
- "*In the projection of* the head of the right femur, osteolytic foci". In una radiografia "in proiezione di"
  vuol dire sovrapposto nell'immagine a due dimensioni: non dice né dentro né fuori. Va trattato come un indizio
  debole, non come vicinanza.

**Correzioni proposte**, tutte generali:
1. la domanda ancorata alla struttura collegata: "che cosa dice la frase di <struttura>? Uno dei reperti, una
   misura, la storia, un'affermazione normale le appartiene?";
2. una relazione in più, "coinvolta": il reperto si estende fino a lei, la comprime, o sta al livello o alla
   giunzione che lei forma. Vale come sì;
3. senza reperto, cioè in un titolo o in un'intestazione, non c'è niente da attribuire: il reperto resta, senza
   persona;
4. la classe mostrata come la intende il linker: il nome che le parole hanno trovato e il significato della
   classe ("left hip: la classe è l'osso dell'anca; il testo nomina l'anca");
5. nelle tabelle lunghe, una finestra attorno alla menzione;
6. nel lettore, la parte collocata come contenimento e "in proiezione di" come indizio debole.

Il campione nuovo ora è visto. Queste correzioni vanno misurate su un altro campione di referti PARROT inglesi mai
usati. Restano le etichette da far rivedere a un radiologo.
