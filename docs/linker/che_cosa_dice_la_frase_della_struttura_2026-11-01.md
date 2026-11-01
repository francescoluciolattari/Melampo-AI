# Che cosa dice la frase di questa struttura (1 novembre 2026, fissato prima del run)

## Perché

Alla domanda «il reperto è di questa struttura?» il modello sceglie il reperto principale della frase e
perde ciò che la frase dice della struttura marcata:
- "history of thyroid cancer surgery … spinal pain": il dolore, non la tiroide operata;
- "the left kidney is longer … right kidney 10.5 cm": il rene sinistro più lungo, non i 10,5 cm del destro;
- "L3 over L4": lo scivolamento di L3, non L4 come livello sotto.

Il radiologo si chiede, per ogni struttura nominata, che cosa la frase dica di lei. Questa è la domanda nuova.

## Come funziona

**Il modello** (`llm_reading.statement_prompt`) riceve la frase con le parole marcate, l'esame, la sezione e la
classe del linker come la intende l'atlante: il nome usuale e gli altri nomi. Ad esempio "left hip bone (also
called: left os coxae, left innominate bone)": così il modello sa che l'anca dell'atlante è l'osso. Risponde con:
- **il pezzo** della frase che dice qualcosa della struttura, copiato dal testo;
- **il tipo**, il primo che vale in quest'ordine: reperto suo · coinvolta · storia · normale · solo una misura ·
  riferimento · struttura sbagliata · ipotesi · niente (un titolo, un'intestazione, il nome dell'esame) · non so;
- per riferimento e struttura sbagliata, **di che cosa** è il reperto.

"Coinvolta" è la relazione che mancava: il reperto è di un'altra struttura ma arriva a lei. Si estende fino a
lei o dentro di lei, la comprime, la restringe, la sposta, oppure sta al livello o all'articolazione che lei forma.

**Il lettore** (`finding_owner`) controlla e porta ciò che sa, ma non decide il tipo:
- il pezzo deve essere nella frase (`llm_reading.in_sentence`, maiuscole e spazi a parte): se non c'è, quella
  risposta vale "non so";
- il tipo deve essere coerente con ciò che il lettore sa del pezzo (`type_check`). Le contraddizioni sono quattro:
  - una storia ("history of", "status post", "operato", "pregresso") letta come reperto, misura o normale;
  - un reperto il cui pezzo è solo numeri, unità e parole di misura;
  - una misura il cui pezzo dice qualcosa di anormale non negato ("dilated", "enlarged", "ispessito");
  - "normale" con qualcosa di anormale non negato;
- le prove di prima, che la struttura colloca soltanto un'altra cosa: il riferimento letto dal lettore Kintsch,
  la diagnosi differenziale, la struttura fuori dall'esame, un altro organo descritto;
- nelle tabelle lunghe (frasi di oltre 320 caratteri) lettore e modello leggono una finestra attorno alle parole,
  tagliata ai separatori.

**La decisione** segue la stessa regola di prima (opzione b):
- i due modelli contano come un solo flusso e devono essere d'accordo sul **gruppo** del tipo. Il gruppo è ciò
  che l'interprete attacca: reperto e coinvolta attaccano entrambi un reperto, quindi sono d'accordo fra loro;
  i riferimenti, le strutture sbagliate e le ipotesi formano il gruppo "niente a questa struttura". Un "niente"
  (un titolo) lascia il posto al tipo dell'altro modello;
- se i modelli non sono d'accordo, o uno non sa, decide una persona e il reperto resta com'è;
- "niente a questa struttura" vale solo se anche il lettore ha una prova in quel senso. Altrimenti decide una
  persona: né il lettore da solo né i modelli da soli tolgono un reperto;
- un reperto, una misura, la storia o il normale si accettano se il lettore non ha prove contrarie e non
  contraddice il tipo; altrimenti decide una persona.

**L'interprete** attacca alla struttura solo quel pezzo, con il suo tipo:
- **reperto o coinvolta**: un reperto fatto dal pezzo e non dalla frase intera (`TextSide.statement = "finding"`);
- **misura**: un reperto che è solo una misura, senza carattere né grado. Non gli si cerca un termine HPO;
- **storia o normale**: non è un reperto attuale e va in `statements` (la lettura del caso lo restituisce in
  `statements`);
- **niente**: il reperto resta come oggi;
- **riferimento, struttura sbagliata o ipotesi**: niente a questa struttura, come oggi, con la domanda per la
  persona.

**Il lettore, altre due correzioni** (`spatial_relation`):
- quando la cosa collocata è una parte della struttura (parete, lume, ansa, segmento), "in the area of" indica
  contenimento: "wall thickening in the area of the descending colon" è la parete del colon;
- "in the projection of", "projected over", "overlying" e "in proiezione di" formano una relazione a parte,
  `projection`, con un peso debole (0,25 invece di 1). In una radiografia dicono che le cose sono sovrapposte,
  non dentro né fuori.

**Non cambia l'ingestione.** La lettura nuova si accende solo con `MELAMPO_OWNER_QUESTION=statement`, e
diventerà quella di default solo se passa i criteri qui sotto.

## Il campione e le etichette (scritte prima di ogni lettura)

`data/gold/statement_2026-11-01/labels.txt`:
- 200 item inglesi da 111 referti PARROT mai usati (seed 20261102). Sono esclusi i due campioni HPO e il
  campione "di chi è" del 1° novembre. Non è rimasto nessun referto italiano inutilizzato;
- le etichette per tipo, mie, di un solo etichettatore: 121 reperto, 9 coinvolta, 22 normale, 19 misura,
  14 niente, 12 riferimento, 2 struttura sbagliata, 1 ipotesi, 0 storia;
- 20 etichette sono segnate "?", da far vedere a un radiologo.

**Limite: nel campione non c'è nessuna storia.** Il tipo "storia" si misura solo sui due casi visti (tiroide
e rene), che sono di sviluppo. Va cercato in un altro campione.

Il lettore da solo è stato fatto girare sul campione **dopo** aver scritto le etichette, solo per controllare
che lo script funzioni. Non è stato cambiato nulla dopo averlo visto.

## Criteri, fissati prima del run (campione `statement`, lettore + Gemma 4 + Nemotron 3 Ultra)

1. **Nessun reperto attribuito alla struttura sbagliata in silenzio:** dei 15 item del gruppo "niente a questa
   struttura", nessuno viene attaccato senza passare da una persona.
2. **Nessun reperto giusto tolto:** dei 185 item da tenere, nessuno messo da parte (al massimo due, dichiarati).
3. **Alla persona al massimo il 10%** dei 185 item da tenere (18).
4. **Il tipo giusto:** fra gli item da tenere accettati senza persona, quelli attaccati con un gruppo diverso
   da quello dell'etichetta (reperto, misura, normale, storia; "niente" escluso da entrambi i lati) non superano
   il 5%.
5. **Il pezzo è nella frase** per ogni item accettato. È vero per costruzione; il run riporta quante risposte dei
   modelli non lo rispettano.

Gli item segnati "?" contano come etichettati, ma sono elencati a parte (`for_a_radiologist`). Il campione
"di chi è" del 1° novembre (`fresh`, già visto) gira con le due domande per confronto: è sviluppo, non criterio.
Il role probe va rilanciato per la correzione del lettore: "sede o no" deve restare a 0 item su cui lettore e
modello concordano con l'esito sbagliato.

**Se un criterio non passa**, la lettura nuova resta spenta e lo dico.

## Come si lancia

Workflow `owner-probe`: input `sets` = `statement,fresh` (default), modelli come prima. Le risposte stanno in
`owner_probe.json`: per ogni riga i campi `statement` (verdetto, gruppo, pezzo, risposte dei modelli) e
`reader+models` (la domanda di prima). La tabella per tipo è in `by_set.statement.statement_types`.
