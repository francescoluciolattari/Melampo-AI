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
