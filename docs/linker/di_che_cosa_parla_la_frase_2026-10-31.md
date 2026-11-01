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
