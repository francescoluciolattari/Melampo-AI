# Affermazione per affermazione (2 novembre 2026, fissato prima del run)

## Perché

Con la lettura per tipi, 9 degli 11 disaccordi veri fra Gemma e Nemotron venivano da come avevamo fatto la
domanda: si chiedeva **un'etichetta sola da un elenco ordinato**. Appena la frase dice due cose ("hyperechoic *and*
not enlarged") o una cosa a metà fra due etichette ("*likely* calcified cartilages"), i modelli scelgono in posti
diversi. Gli altri 2 non erano errori del modello: uno era un nostro errore nel contesto, l'altro un'incongruenza
del referto.

## Che cosa cambia

**Il modello non sceglie più.** `llm_reading.claims_prompt` gli chiede che cosa la frase afferma della struttura,
affermazione per affermazione, ciascuna con le parole copiate dalla frase:
- che cosa osserva di diverso dal solito: per ogni osservazione le parole, di chi è (lei o un'altra struttura
  nominata nella frase), le parole della relazione e se il reperto arriva a lei;
- che cosa dice normale o assente; le misure; i fatti del passato;
- il grado di certezza e le alternative elencate;
- i dispositivi; ciò che non si può valutare.

Il modello deve ripetere anche le parole marcate e dire se nominano davvero la struttura collegata.

**Detto, non dedotto.** Ogni affermazione deve avere le sue parole (`read_claims`):
- le parole che non sono nella frase cadono;
- un pezzo accorciato con "…" vale se ogni parte è nella frase, nell'ordine (`llm_reading.copied`);
- se il modello ripete parole marcate diverse da quelle marcate, la risposta non vale (in quel caso ha letto
  altre parole);
- se dice che l'osservazione è di un'altra struttura, questa deve essere nominata nella frase.

**I confini spariscono.** Lettore e modelli sono d'accordo quando attaccherebbero la stessa cosa: un reperto
attuale alla struttura, niente alla struttura perché ciò che si osserva è di un'altra, oppure solo le altre
affermazioni (normale, misura, storia, dispositivo, non valutabile). Non serve più che abbiano scelto la stessa
etichetta.
- **"Hyperechoic and not enlarged"** dà un reperto e un'affermazione normale, attaccati tutti e due.
- **"Likely calcified cartilages"** dà un reperto con il suo grado di certezza.
- **Una diagnosi differenziale** dà un reperto possibile, con le sue alternative.
- **Si attaccano le affermazioni che entrambi i modelli fanno** (`agreed_claims`).

**Il lettore controlla con ciò che sa** (`what_is_said`), senza liste scritte da me:
- **che cosa è un'anomalia lo dice l'ontologia.**
  - Le *qualità* sono le parole delle qualità PATO di cui sono fatti i fenotipi HPO ("enlarged", "dilated",
    "thickened"). Contano solo dove HPO le usa per quella struttura o per le sue parti: "thin" è un'anomalia della
    pelle in HPO, non della parete della vescica.
  - Le *entità* sono le teste dei nomi HPO di molti fenotipi ("cyst", "polyp", "hernia", "stone"), tolte le teste
    anatomiche di UBERON ("wall", "space", "fluid" del liquor) e gli attributi di PATO ("size", "volume").
  - Restano i caratteri radiologici sopra o sotto il tessuto ("hypodense", "hyperechoic"). "Anechoic" e
    "isodense" da soli non sono anomalie.
- **la grammatica del sintagma** (`grammar.Grammar`): una qualità conta dovunque sia nel sintagma; un'entità solo
  se ne è la testa. In "intracranial fluid spaces" la testa è "spaces"; in "free fluid" la testa è "fluid".
- **la negazione e il passato**, dal rilevatore di asserzioni che usiamo già.
- **la coordinazione** (`conjuncts`): in "the stomach and the duodenum" e in "in the liver … and in the
  pancreas" le strutture sono congiunti e condividono ciò che la frase dice. Nessuna porta il reperto dell'altra.
  Questo vale anche per il lettore della domanda di prima.
- **le parole della relazione** copiate dal modello, lette con la relazione spaziale: "between" e "adjacent" sono
  vicinanza, "extends into" è direzione, "in" e "of" sono contenimento.

**Le contraddizioni** (`FindingOwner._contradiction`) mandano a una persona e non decidono mai:
- un reperto "suo" quando il lettore legge un riferimento, un altro organo o una struttura fuori dall'esame;
- un reperto che è solo una misura (un numero con un'unità vera: "S1" non è una misura), o solo un fatto
  passato;
- un reperto "che arriva a lei" quando le parole della relazione dicono vicinanza;
- parole "normali" o di misura che affermano un'anomalia (l'alone ipodenso detto normale).

**"Niente a questa struttura"** vale solo se il lettore ha una prova in quel senso, oppure se le parole della
relazione copiate dal modello dicono vicinanza. Altrimenti decide una persona.

**I due casi di Nemotron:**
- **valvola mitrale.** L'esame e la sezione erano presi da una tabella di misure ("Ao: 33 mm" come sezione).
  Adesso `report_context` divide le frasi anche al punto e virgola; una frase con tre numeri o più è una tabella,
  e un'etichetta seguita da un numero è una misura, non una sezione. Sulla stessa frase l'esame e la sezione ora
  sono vuoti;
- **femore.** L'esame è del lato sinistro e la frase parla del destro. Il reperto resta al femore destro, perché
  la frase lo dice, ma va a una persona con la domanda «l'esame è di un lato, la frase dell'altro: è un errore di
  lato, o l'esame comprende anche questo lato?» (`laterality_conflict`, tipo di domanda `laterality`). Al
  modello l'esame si dà come contesto: «la frase prevale».

**L'interprete** (`interpret_exam`) attacca tutte le affermazioni concordate:
- il reperto, fatto dalle sue parole e con la certezza che la frase gli dà (`possible`; `hypothetical` se ci
  sono alternative);
- una misura senza reperto diventa un reperto che è solo una misura;
- normale, storia, dispositivo e non valutabile vanno in `statements`.

Le risposte della maschera hanno anche **dispositivo** e **non valutabile**. Le definizioni stanno in un posto
solo: la guida per i radiologi (`docs/gold_set_guida_radiologi.md`, sezione 5).

**Nell'ingestione la lettura resta spenta** finché non passa i criteri: `MELAMPO_OWNER_QUESTION=claims`, con
`MELAMPO_HPO` e `MELAMPO_PATO` per ciò che il lettore sa delle anomalie.

## Le altre soluzioni, verificate sul campione già visto (sviluppo, non criterio)

| Soluzione | Verifica |
|---|---|
| Pezzo con "…" | dei 14 pezzi che prima non si trovavano nella frase, 9 ora valgono. I 5 che restano saltano parole senza "…" o le riordinano, quindi non sono copiati e restano fuori |
| Coordinazione | i 4 casi "liver … and pancreas", "stomach and duodenum" (2) e "liver and stomach parenchyma" non danno più `describes:`. Restano: "right upper lobe bronchus" e "ureters … into the bladder" |
| "S1" letto come misura | non succede più: una misura vuole un numero con un'unità |
| "Intracranial fluid spaces" | non è più letto come anomalia: la testa del sintagma è "spaces" |
| Esame e sezione dalla tabella | sulla frase della valvola mitrale esame e sezione sono vuoti |
| Lateralità | "CT of the distal femoral shaft on the left side" con "right femur" dà la domanda alla persona |
| Alone ipodenso detto normale | il lettore lo contraddice ("hypodense") |

**Limiti noti del lettore.** Non riconosce "nodule" né "free fluid", perché sono anche nomi di strutture in
UBERON. Il rilevatore di negazione estende "not enlarged … with cysts" anche alle cisti. Su queste frasi il
lettore controlla meno, ma non decide di più.

## Il campione e le etichette (scritte prima di ogni lettura)

`data/gold/claims_2026-11-02/labels.txt`:
- 200 item inglesi da 102 referti PARROT mai usati (seed 20261103), esclusi tutti i campioni precedenti;
- le affermazioni, mie, di un solo etichettatore: finding 117, involved 23, normal 31, measure 31, history 5,
  device 2, not_assessable 5, nothing 6, aside 6;
- che cosa si attacca: un reperto in 140 item, solo le altre affermazioni in 54, niente alla struttura in 6;
- 22 etichette sono segnate "?".

**Limite: solo 6 item da non attribuire.** Il criterio 1 qui ha poca forza. Il campione `statement`, già visto, ne
ha 15 e gira come sviluppo.

## Criteri, fissati prima del run (campione `claims`, lettore + Gemma 4 + Nemotron 3 Ultra)

1. **Nessun reperto attribuito alla struttura sbagliata in silenzio:** dei 6 item da non attribuire, nessuno
   viene attaccato senza passare da una persona.
2. **Nessun reperto giusto tolto:** dei 140 item con un reperto, nessuno messo da parte (al massimo due,
   dichiarati).
3. **Alla persona al massimo il 10%** dei 194 item che non sono da non attribuire, cioè 19, contando anche le
   domande di lateralità.
4. **Che cosa si attacca è giusto:** fra gli item accettati senza persona, quelli in cui si attacca un reperto dove
   l'etichetta dice solo altre affermazioni, o il contrario, non superano il 5%.
5. **Riportati, non criteri:** le affermazioni attaccate che l'etichetta non ha, quelle dell'etichetta non
   attaccate, le parole cadute perché non copiate, gli item "?".

Se un criterio non passa, la lettura resta spenta e lo dico. Il campione `statement` gira con le stesse domande,
come sviluppo e per confronto con la lettura per tipi.

## Come si lancia

Workflow `owner-probe`: input `sets` = `claims,statement` (default), modelli come prima. Scarica HPO e PATO. La
tabella è in `by_set.claims.claims_kinds`; ogni riga ha il campo `claims` (verdetto, affermazioni concordate,
risposte dei modelli, `review`).

## Il run (2 novembre 2026, pomeriggio): non vale per i criteri

**Molte risposte sono andate perse.** Le chiamate a OpenRouter sono fallite dopo i tentativi (`EncoderError`):
- sul campione `claims`: Nemotron 47 risposte su 200 nella lettura per affermazioni, Gemma nessuna;
- sul campione `statement`, che girava dopo: Nemotron circa il 93%, Gemma circa il 31%.

Il numero degli errori cresce mentre il run va avanti, quindi è probabile che sia finito il credito o che il
servizio limiti le chiamate. Il messaggio non era registrato, solo il tipo; ora lo è (`error_text`). Una risposta
persa vale "non so" e manda la frase alla persona. **Il campione `statement` di questo run è da buttare**, e i
numeri del campione `claims` si possono leggere solo sugli item a cui hanno risposto tutti e due i modelli.

**Sul campione `claims`, sui 153 item con le due risposte:**

| | da non attribuire (5): attribuiti | da tenere (148): tolti / alla persona |
|---|---|---|
| domanda vecchia (167 item con le due risposte) | 0 | **2** / 15 |
| affermazione per affermazione | **2** | 0 / **31 (21%)** |

- **Criterio 1, non passa:** 2 item attribuiti. "Fluid in the right pleural cavity reaching the seventh right
  rib": tutti e due i modelli dicono che arriva alla costola, mentre su una radiografia è il livello. L'altro è la
  calcificazione al livello del legamento sul bordo di L5, un'etichetta segnata "?".
- **Criterio 2, passa.**
- **Criterio 3, non passa:** 21% alla persona.
- **Criterio 4, passa:** 4,3%.

**La causa principale (20 dei 31): i modelli chiamano "altra struttura" una parte della struttura.** Il ponte
del cervello, la parete dell'arco aortico, il collo del femore, la valvola mitrale del cuore, l'ileo terminale
dell'intestino tenue: alla domanda «le parole nominano la struttura collegata?» un modello risponde "no", perché il
nome è diverso. Ma se un nome indica la struttura o una sua parte, lo dice l'anatomia, ed è conoscenza del lettore.
Ora il lettore legge con UBERON (`part_of`, `is_a`) se ciò che il modello nomina è la struttura, una sua parte o il
tutto di cui è parte (`FindingOwner.same_structure`). **Riletto sulle stesse risposte, già viste, quindi come
sviluppo e non come criterio:** alla persona 18 su 149 (12,1%), nessun reperto tolto, 4,6% attaccati in modo
diverso.

**Che cosa resta (i 18):**
- 3 casi "the wall of the arch": il lettore non sa che l'arco è dell'aorta;
- 4 casi in cui Nemotron mette fra le osservazioni "homogeneous", "not enlarged", "smooth wall";
- 2 casi in cui un modello risponde "no" alla domanda «nominano la struttura?» pensando ai reperti della frase,
  non alle parole (L5 nell'ernia);
- 1 caso in cui il lettore legge una struttura della frase precedente (la riga di PARROT contiene due frasi);
- 2 dell'anca: l'articolazione e l'osso della classe;
- gli altri sono disaccordi veri.

## Le correzioni e il secondo campione (2 novembre 2026, sera, fissati prima del run)

**Le correzioni**, tutte generali:
- **anatomia del lettore.** Se ciò che il modello nomina è la struttura, una sua parte o il tutto di cui è parte,
  lo legge il lettore con UBERON (`FindingOwner.same_structure`), anche i nomi brevi come "pons". Se il modello
  indica "di chi è" con una parte (parete, lume, segmento) e il lettore non trova nessun'altra struttura in quelle
  parole ("the wall of the arch"), è una parte di questa struttura;
- **la domanda «nominano la struttura?»** ora dice: «rispondi solo sulle parole marcate, non su ciò che la frase
  descrive; sì se nominano la struttura o una sua parte»;
- **"up to", "reaching" e "fino a" non sono una regola del lettore.** Nella prima stesura li avevo fissati come
  "limite, non contatto". Era un confine deciso prima di leggere, e sbagliato: "a lesion extending up to the
  cervical cord" la tocca, mentre "fluid reaching the seventh rib" su una radiografia dà un'altezza. Il lettore non
  sa leggere queste parole e quindi non contraddice. Le legge il modello, nella frase: la domanda gli chiede se la
  struttura stessa è toccata (infiltrata, compressa, ristretta, spostata, deformata) o se il reperto sta al livello
  o nell'articolazione che lei forma; "no" se la frase dice solo dove sta o fin dove arriva. Se i modelli dicono
  "no" e il lettore non ha prove, decide una persona;
- **un'assenza fra le osservazioni.** Il lettore legge la negazione: "not enlarged" passa fra le affermazioni
  normali;
- **la frase della menzione.** Chi porta il reperto si cerca solo nella frase della menzione (ci si ferma anche a
  ".", "?" e "!");
- **il run conta solo se ha le risposte.** Se va persa più del 5% delle risposte di una lettura, il run non conta
  per i criteri (`valid_for_criteria`, `lost_answers` con i messaggi d'errore);
- **meno chiamate.** Il workflow ha l'input `arms` (default `claims`): la domanda vecchia e la lettura per tipi
  sono già misurate e girano solo se richieste.

**Il secondo campione** (`data/gold/claims_2026-11-02b/labels.txt`):
- 200 item inglesi da 107 referti PARROT mai usati (seed 20261104), con etichette scritte prima di ogni lettura;
- che cosa si attacca: un reperto in 134 item, solo le altre affermazioni in 49, niente alla struttura in 17;
- 18 etichette sono segnate "?".

**Criteri, gli stessi**, sul campione `claims2` (lettore + Gemma 4 + Nemotron 3 Ultra, solo la lettura per
affermazioni):
1. dei 17 item da non attribuire, nessuno attaccato senza passare da una persona;
2. dei 134 item con un reperto, nessuno messo da parte (al massimo due, dichiarati);
3. alla persona al massimo il 10% dei 183 item da tenere (18), contando anche le domande di lateralità;
4. fra gli accettati, al massimo il 5% con un reperto dove l'etichetta non lo ha, o il contrario;
5. il run vale solo se si perde al massimo il 5% delle risposte.
