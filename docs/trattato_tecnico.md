# Melampo — Trattato tecnico

## Un motore diagnostico che riconosce, fonda e sa quando non sapere

**Versione:** settembre 2026
**Repository:** `francescoluciolattari/Melampo-AI`
**Stato:** 415 test, 15 revisioni integrate, 50 file, 11.643 righe

---

## Indice

1. [Il problema](#1-il-problema)
2. [La tesi architetturale](#2-la-tesi-architetturale)
3. [I principi emersi](#3-i-principi-emersi)
4. [Gli strati del sistema](#4-gli-strati-del-sistema)
5. [Il ciclo diagnostico](#5-il-ciclo-diagnostico)
6. [L'intuizione come meccanismo](#6-lintuizione-come-meccanismo)
7. [Il ciclo di apprendimento](#7-il-ciclo-di-apprendimento)
8. [Governo dell'incertezza](#8-governo-dellincertezza)
9. [Inventario dei moduli](#9-inventario-dei-moduli)
10. [Ciò che è deciso e ciò che è misurato](#10-ciò-che-è-deciso-e-ciò-che-è-misurato)
11. [Evoluzione: le correzioni che hanno formato il sistema](#11-evoluzione-le-correzioni-che-hanno-formato-il-sistema)
12. [Ciò che resta](#12-ciò-che-resta)

---

## 1. Il problema

Un sistema di supporto diagnostico deve fare tre cose che si ostacolano a vicenda.

**Riconoscere.** La medicina è troppo vasta perché le sue connessioni siano scritte a mano. Solo un modello addestrato su molta letteratura e molti casi può riconoscere che un certo insieme di reperti evoca una certa condizione.

**Rendere conto.** Un clinico non può usare una conclusione che non può interrogare. "Polmonite, 0.83" senza il perché non è utilizzabile: il medico non può verificare se il sistema si è appoggiato a un reperto che lui sa essere un artefatto.

**Sapere quando non sa.** L'errore diagnostico dominante non è l'ignoranza ma la **chiusura prematura** — fermarsi alla prima spiegazione plausibile. I fattori cognitivi sono coinvolti in circa il 75% degli errori diagnostici, e la maggior parte non riguarda deficit di conoscenza.

Un modello linguistico da solo fa bene la prima e male le altre due. Un sistema a regole fa bene la seconda e male la prima. Melampo è un tentativo di farle tutte e tre assegnandole a componenti diversi.

---

## 2. La tesi architetturale

> **Il modello riconosce. Il grafo fonda. Il calibratore decide se il risultato può uscire.**

Non è una gerarchia ma una divisione di competenze, e discende dal modello del ragionamento diagnostico che la letteratura descrive: riconoscimento rapido, analisi lenta, e un **calibratore** dove le uscite di entrambi si incontrano.

| Componente | Decide | Perché lì |
|---|---|---|
| **Modello addestrato** | Quali diagnosi, in quale ordine | Il riconoscimento di pattern su molti casi appresi è ciò che un modello fa bene e una regola non può fare |
| **Grafo dei concetti** | Su cosa poggia ciascun elemento | La provenienza vive qui e da nessun'altra parte |
| **Calibratore deterministico** | Se il risultato esce, se serve un accertamento, se va escalato | La metacognizione è dove il clinico umano è più fragile: la fiducia correla poco con l'accuratezza |

Il modello **decide** — produce la diagnosi più probabile e il suo ordine. Ciò che non può fare è decidere **senza rendere conto**: ogni voce che mette nel differenziale deve avere un percorso nel grafo, altrimenti è un'ipotesi da vagliare e non una diagnosi proposta.

Il grafo non limita il modello: lo **classifica**. E la classificazione è ciò che rende ispezionabile una conoscenza che nei pesi non può spiegarsi.

Rispetto all'originale umano l'architettura ha un vantaggio su un punto: **il calibratore non si sovraconfida**.

---

## 3. I principi emersi

Questi principi non erano nel progetto iniziale. Sono emersi correggendo errori concreti, e ciascuno ha lasciato una traccia nel codice.

### 3.1 Direzione, non grandezza

Un reperto è un **punto di ingresso**: il grafo si percorre a partire da lì. Un'ipotesi è una **destinazione**: si conferma o si esclude, e non genera discendenza finché non è confermata.

Questa distinzione decide dove va ogni cosa. Registrare la familiarità come reperto con punteggio 0.3 non attenua l'errore, lo traveste: la traversata è binaria, e il grafo verrebbe percorso *dalla* condizione di un parente verso le sue complicanze in un paziente che non ce l'ha. I percorsi sarebbero corretti su una premessa falsa — l'errore più difficile da notare, perché a valle nulla sembra sbagliato.

### 3.2 Gli errori restrittivi si annunciano, quelli permissivi si zittiscono

Un arco mancante genera falsi allarmi finché qualcuno se ne occupa: è auto-correttivo, produce da solo la coda di lavoro. Un arco sbagliato in più **smette** di generare segnalazioni proprio sulle relazioni che avrebbe dovuto rifiutare: si nasconde riuscendo.

Da qui la regola che governa ogni aggiunta alla base di conoscenza: **la direzione che può nuocere richiede una persona, quella che può solo infastidire può essere automatica**.

### 3.3 Il silenzio non è negazione

Un grafo che non contiene una relazione non sta affermando che la relazione non esiste. Un singolo numero non distingue "documentato come raro" da "nessuno ha guardato", e dopo la moltiplicazione lungo un percorso entrambi si leggono come quasi certa negazione.

Da qui gli **intervalli** su ogni arco. E la verifica sui dati reali ha confermato che non era un formalismo imposto: HPO pubblica le frequenze già come *range*, e sarebbe stato il valore puntuale il passaggio con perdita.

### 3.4 La copertura si misura, non si assume

Il recupero ricorsivo elimina la compressione lossy *a priori* — chunking fisso, taglio top-k — ma non elimina la perdita: quello che il modello non interroga non viene mai visto, e una query non formulata non lascia traccia. La perdita si sposta e cambia natura, da sistematica e ispezionabile a stocastica e invisibile.

Quindi la copertura è una variabile strumentata a runtime, non una proprietà garantita dall'architettura.

### 3.5 Un intervallo, tre letture

| Consumatore | Legge | Perché |
|---|---|---|
| Percorso diagnostico | limite inferiore | ciò che l'evidenza garantisce |
| Generazione di ipotesi | limite superiore | ciò che potrebbe essere vero |
| Manutenzione del grafo | ampiezza | dove non sappiamo, quindi dove indagare |

L'esplorazione è ottimista per costruzione, la decisione prudente per costruzione. Stesso grafo, proiezioni opposte, nessuna seconda struttura dati.

### 3.6 Verificare per tentativo, non per ispezione

Ispezionare uno schema dimostra come lo store è configurato, non che un candidato sia irraggiungibile. Il criterio di uscita per l'isolamento è un **test negativo** che tenta il recupero attraverso il percorso evidenziale e si aspetta il fallimento — incluso il caso in cui il candidato avrebbe il punteggio più alto, così l'ordinamento non è la cosa che lo tiene fuori.

### 3.7 Una guardia mai chiamata non separa niente

Due guardie di isolamento sono esistite per settimane nel repository, esercitate solo dai test e invocate **zero volte** nel codice di produzione. La separazione era una convenzione, e le convenzioni reggono finché qualcuno non se ne dimentica.

Da qui il principio: le proprietà di sicurezza vanno scritte come **tabelle di stato per livello**, non come prosa. "La separazione è strutturale" non è verificabile; tre righe con il loro stato lo sono, e si nota subito quando una resta aperta.

### 3.8 Fallire rumorosamente

Ricorre in ogni strato del sistema, perché il difetto più pericoloso non è quello che rompe ma quello che assomiglia a un altro problema.

| Difetto | Come si presentava | Come si presenta ora |
|---|---|---|
| Concetti non risolvibili | "il grafo è troppo rado" | `resolution_gap`, con l'elenco dei termini |
| Copertura insufficiente | assenza di ipotesi | domande esplicite alla base di conoscenza |
| Nessun esame discriminante | lista vuota | referto di lacuna con le ipotesi non mappate |
| Metriche assenti nel gate | permesso implicito | gate chiuso |
| Ipotesi sintetica sul percorso evidenziale | passava | eccezione con la rotta corretta |

### 3.9 Le esortazioni non funzionano, le strutture sì

Le strategie di riduzione dei bias hanno poche probabilità di successo, mentre correggere i deficit di conoscenza porta al successo del ragionamento. Dire a un clinico "attento all'ancoraggio" non lo rende meno ancorato.

Tradotto: un prompt che chiede al modello di considerare le alternative è un'esortazione e non serve. Un meccanismo che **enumera** le alternative indipendentemente dalla volontà del modello è una struttura e funziona. È la giustificazione del Dream Engine, e non è un'analogia: è l'applicazione diretta del risultato.

---

## 4. Gli strati del sistema

```
   TESTO CLINICO  (italiano o inglese)
        │
   ┌────▼─────────────────────────────────────────┐
   │ ESTRAZIONE SEMANTICA                          │
   │  risoluzione concetti → identificatore        │
   │  modificatori → attributi, mai nodi           │
   │  asserzione → intervallo + stato              │
   │  barriera → solo reperti attuali del paziente │
   └────┬─────────────────────────────────────────┘
        │ reperti          ┌─────────────────────┐
        ├─────────────────►│ CANALE FAMILIARITÀ   │
        │                  │ screening + prior    │
        │                  └─────────────────────┘
   ┌────▼─────────────────────────────────────────┐
   │ RECUPERO DUAL-PATH                            │
   │  one-shot → intuizione | ricorsivo → analisi  │
   │  divergenza = segnale di conflitto            │
   └────┬─────────────────────────────────────────┘
        │
   ┌────▼──────────────┐      ┌────────────────────┐
   │ MODELLO           │      │ DREAM ENGINE        │
   │ presenta il caso  │      │ enumera percorsi    │
   └────┬──────────────┘      └────────┬───────────┘
        │ discorso clinico              │ ipotesi
   ┌────▼──────────────────────────────▼───────────┐
   │ LETTORE DEL DISCORSO → copione                 │
   │ VERIFICATORE → fondato / mediato / non supportato│
   └────┬─────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────┐
   │ CALIBRATORE                                   │
   │  esce? · serve un esame? · escalation?        │
   └───────────────────────────────────────────────┘
```

### 4.1 Estrazione semantica

Il testo clinico diventa concetti indirizzabili. Quattro passaggi, ciascuno con una decisione non ovvia.

**Risoluzione.** L'indice ontologico conta 23.800 termini e 42.045 forme superficiali, costruito in 0,4 secondi — contro i sedici termini scritti a mano che c'erano prima. Il matching è esatto e deterministico: una corrispondenza approssimata che risolve silenziosamente il concetto sbagliato è peggio di nessuna corrispondenza, perché un reperto irrisolto si annuncia mentre uno mal risolto si propaga in un percorso, un'ipotesi e una provenienza che sembrano tutti ben formati. L'ambiguità viene **segnalata, non rotta**: due termini condividono il sinonimo "Coughing", e il risolutore lo dice invece di scegliere.

**Modificatori.** `Bilateral` e `Progressive` non sono reperti. Ammessi come concetti si collegherebbero a migliaia di condizioni, rendendo privo di informazione ogni percorso che li attraversa. La distinzione è **già nell'ontologia** — HPO ha 357 termini sotto `Clinical modifier` — quindi è una consultazione della gerarchia, non un giudizio del modello. La fusione in termini pre-coordinati resta un raffinamento opzionale: su ~20.000 termini esistono solo 52 forme `Bilateral X` e 58 `Progressive X`.

**Asserzione.** Il rilevamento distingue polarità, certezza, esperiente, temporalità e fonte. L'uscita è **intervallo e stato**, mai uno scalare: uno scalare farebbe collassare "cercato e assente", "nessuno ha guardato" e "domanda aperta" sullo stesso zero. La fonte è un asse separato perché l'oggettività non sovrascrive uniformemente la soggettività — un clinico può smentire "non ho aritmia" perché l'ECG è osservabile, non può smentire "non ho dolore".

**Barriera.** Un unico punto dove i reperti si assemblano, che ammette solo ciò che è attuale, affermato e del paziente. Ogni rifiuto nomina la propria destinazione: negato → esclusione documentata, ipotetico → domanda aperta, di un parente → canale familiarità.

Due liste sopravvivono, perché rispondono a domande diverse. `clinical_entities` tiene **ogni** menzione, incluse le negate: un chunk che dice "denies fever" deve essere recuperabile cercando "fever", perché la negazione è ciò che chi legge vuole trovare. `patient_findings` tiene solo ciò che ha superato la barriera, ed è quella lista — mai le menzioni — ad alimentare il grafo.

### 4.2 Multilinguismo

L'identificatore è il perno. `HP:0002240` è indipendente dalla lingua, quindi `Hepatomegaly` ed `Epatomegalia` risolvono sullo stesso termine e attraversano lo stesso grafo.

Non è solo comodità: ontologie, terminologie e la letteratura che fornisce le verosimiglianze sono prevalentemente inglesi, quindi tradurre il grafo significherebbe mantenere **due basi di conoscenza** invece di due vocabolari.

| Strato | Lingua |
|---|---|
| Ingresso — testo clinico | Entrambe |
| Conoscenza — grafo, archi, verosimiglianze | Inglese, canonica |
| Uscita — al clinico | Entrambe, resa dagli identificatori |

**Le due lingue non sono simmetriche**, e il numero conta: l'inglese ha 42.045 forme curate, l'italiano copre 3.488 termini — circa il 15% — di cui **524 ufficiali** e 2.964 prodotti da traduzione automatica e marcati preview. Lo stato della traduzione viaggia con la corrispondenza, così un chiamante può richiedere solo corrispondenze curate dove un errore costa più di un'assenza.

Le cue dell'asserzione sono **selezionate per documento, mai fuse**: l'italiano *ma* è un terminatore mentre l'inglese *ma* non è nulla, l'italiano *non* è negazione mentre l'inglese *non* compare dentro le parole.

### 4.3 Il grafo dei concetti

Archi a intervallo con provenienza. Importati da HPO: **285.598 archi in 2,5 secondi**, con una distribuzione che ha confermato il design meglio di qualunque argomento teorico.

| Stato | Archi | Quota |
|---|---|---|
| incerto positivo | 121.626 | 42,6% |
| lacuna | 69.205 | 24,2% |
| negazione debole | 56.954 | 19,9% |
| documentato | 28.109 | 9,8% |
| esclusione documentata | 9.704 | 3,4% |

Solo il 9,8% degli archi reali è stretto e affidabile. Quasi due terzi portano ampiezza epistemica significativa, che un peso singolo avrebbe appiattito in valori dall'aria sicura. E la **negazione debole copre 56.954 relazioni**: sotto uno schema a quattro stati sarebbero state forzate su "esclusione documentata", affermando che cinquantasettemila relazioni erano state escluse quando l'evidenza dice solo che sono debolmente supportate.

Le frazioni osservate diventano intervalli di Wilson, quindi l'ampiezza segue la numerosità: un'osservazione su un paziente resta larga, quarantacinque su cinquanta si stringe.

**Il grafo non è uniforme**, e la disomogeneità è sistematica: HPO annota malattie **rare**, quindi è denso dove la genetica rara è informativa — 65.061 annotazioni sul sistema nervoso — e rado dove vive la patologia comune acquisita: 58 sulla cavità toracica. Serve una seconda fonte, ed è un buco che va misurato per area, non assunto.

### 4.4 Recupero dual-path

Due strategie in parallelo, riconciliate. Convengono perché **falliscono in direzioni opposte**: il one-shot per omissione — il chunk giusto fuori dai top-k — il ricorsivo per overreach, affermando oltre l'evidenza. Uno sotto-dichiara, l'altro sovra-dichiara.

Il premio non è la copertura combinata ma che **la divergenza è un estimatore empirico di incertezza**, per un sistema la cui architettura di sicurezza è costruita sull'incertezza.

| Esito | Disposizione |
|---|---|
| Trovato da entrambi | Confermato |
| Solo one-shot | Ammesso; il percorso veloce ci è arrivato prima |
| Solo ricorsivo, con offset verificabili | Guadagno di recall |
| Solo ricorsivo, senza offset | Scartato come probabile overreach |
| Si contraddicono | Segnale di conflitto → escalation o astensione |

E l'asimmetria cognitiva che il progetto aveva documentato ma non implementato: **RAG all'intuizione, RLM al differenziale**. Due motori con caratteri opposti erano alimentati dalla stessa modalità di recupero.

---

## 5. Il ciclo diagnostico

### 5.1 Il modello presenta, la stanza decide

Il modello non viene addestrato a emettere uno schema rigido. Ragiona in linguaggio clinico, e quel linguaggio non è rumore da comprimere in un formato.

> *"Il quadro è più compatibile con scompenso cardiaco, anche se la febbre non torna. Escluderei una polmonite prima di decidere."*

In quella frase ci sono un'ipotesi principale con un grado di impegno, un reperto discordante segnalato, un candidato da escludere e una richiesta implicita di accertamento. Uno schema JSON perderebbe quasi tutto.

Il modello viene quindi **letto come un collega che presenta un caso**, con la stessa macchina semantica che legge i referti. I connettivi diagnostici portano l'impegno di chi parla e diventano ordinamento: *most consistent with* precede *consider*, che precede *rule out*. Un connettivo lega solo il concetto che lo segue immediatamente, così "compatibile con X, anche se la Y" non trascina l'impegno di X su Y.

Il copione è **l'uscita di quella lettura**. La disposizione rispecchia dove le decisioni cliniche vengono davvero prese: chi presenta non decide, decide la stanza dopo aver ascoltato, interrogato e verificato.

### 5.2 Il copione e la sua verifica

La ricerca sull'expertise descrive la conoscenza clinica organizzata in **illness scripts** con tre componenti: condizioni abilitanti, guasto, conseguenze cliniche. Il novizio ragiona dal guasto alle conseguenze lentamente; l'esperto riconosce il copione dalle conseguenze e risale al guasto solo quando non torna.

Il verificatore classifica ogni elemento:

| Verdetto | Significato |
|---|---|
| `grounded_in_case` | Un reperto ammesso lo afferma |
| `knowledge_mediated` | Nessun reperto lo afferma, un percorso nel grafo lo sostiene — ipotesi ammissibile con il percorso come provenienza |
| `unsupported` | Né il caso né il grafo — l'unico dei tre che è un difetto |

Il verdetto a tre valori è essenziale: un'affermazione assente dal caso **non è automaticamente sbagliata**, perché gran parte dell'inferenza clinica collega reperti a condizioni attraverso conoscenza esterna al caso. Un giudice che rifiutasse ogni relazione non scritta nel referto rifiuterebbe anche il ragionamento clinico corretto.

Il fondamento usa solo i reperti ammessi dalla barriera, mai ogni menzione: una menzione negata non è un'osservazione e non deve fondare nulla.

### 5.3 Rilevamento dell'overreach

La metrica di faithfulness classica è sovrapposizione di termini, e per costruzione non vede il fallimento che il recupero ricorsivo produce davvero. Se la conclusione dice che il reperto A è causato dalla condizione B, con A in un frammento e B in un altro e nessun frammento che li colleghi, **ogni termine è supportato e ogni citazione è reale**. La parte non supportata è la *relazione*, e nessuna ispezione per singolo elemento può vederla.

Il giudice ispeziona le relazioni: guarda un frammento per volta e chiede se ne esiste almeno uno che contenga entrambi i lati. Mescolare le evidenze in un unico sacco di parole distrugge precisamente l'informazione che serve.

### 5.4 Esami discriminanti

Il differenziale esce con l'ordinamento; quando nessuna ipotesi è sufficientemente probabile, il selettore indica **quale osservazione risolverebbe di più l'incertezza**, calcolando il guadagno di informazione atteso sul grafo.

Tre proprietà che corrispondono a come ragiona un clinico: un esame legato ugualmente a tutte le ipotesi vale zero; uno legato a una sola di due ipotesi equiprobabili vale un bit intero; e quando un'ipotesi già domina tutti i punteggi crollano — non c'è più molto da imparare, quindi il sistema smette di proporre lavoro.

Un arco mancante dà l'intervallo pieno, non un valore basso. Altrimenti le lacune si travestono da potere diagnostico: sui numeri reali un esame documentato su entrambi i lati vale 0,296 bit, lo stesso esame con il secondo lato **mancante** ne vale 0,758. Ordinando per guadagno **garantito** invece che puntuale, l'inversione sparisce.

Informazione e carico restano separati. Fonderli farebbe la scelta lo stesso, ma di nascosto e da chi ha scritto la formula — che non ha visto il paziente.

---

## 6. L'intuizione come meccanismo

Il Dream Engine è il componente che risponde all'errore dominante. Non è un'aggiunta esotica: è il rimedio strutturale alla chiusura prematura, reso meccanico invece che lasciato alla volontà del modello — che tende allo stesso errore, documentato come *search satisficing*.

### 6.1 Le ipotesi si trovano, non si scrivono

Un candidato è un **percorso** nel grafo che collega un reperto osservato a una condizione che il caso non ha sollevato. Il percorso è la sua provenienza.

La differenza da un modello generativo è categorica e non di grado: un percorso esiste nel grafo o non esiste, quindi un'ipotesi non può essere fluente e infondata insieme.

**La cascata.** L'intersezione fra le cause di un reperto e le conseguenze di un altro non è una coincidenza di vocabolario: è un meccanismo candidato. Nominarlo converte un'affermazione non supportata di causalità in una tesi su un percorso specifico, che può essere esaminata e respinta.

Sui dati reali: `Hepatomegaly` e `Splenomegaly` si connettono attraverso `Gaucher disease` con intervallo [0.716, 0.886] e zero lacune, su un grafo di 115.875 archi.

### 6.2 La speculazione è un gradiente

Fra i percorsi che esistono, **novità e sostegno sono riportati separatamente**. Un percorso breve su archi ben attestati descrive una connessione che qualunque clinico solleverebbe; uno lungo su archi deboli descrive qualcosa raramente considerato — novità alta, sostegno basso, tracciabile in entrambi i casi. Fonderli in un punteggio è ciò che fa sembrare forte un'ipotesi speculativa.

Nessun percorso non è speculazione: è fabbricazione, e viene scartata.

### 6.3 Due canali, non uno

| | Enumerazione dei meccanismi | Screening da familiarità |
|---|---|---|
| Domanda | Cos'altro potrebbe spiegare questi reperti? | Cos'altro potrebbe avere questo paziente, indipendentemente dal motivo della visita? |
| Gate | Indeterminazione diagnostica + densità locale | Trasmissione, esordio raggiungibile, valutazione pregressa |
| Consegna | Differenziale, come ipotesi di esclusione | Lista screening, fuori dal differenziale |

Il canale screening **non passa dal gate di indeterminazione**, deliberatamente. Quel gate esiste perché un'alternativa esplicativa è informativa solo quando il differenziale è piatto; una considerazione di screening è valida proprio quando il differenziale ha già converso, e passarla dallo stesso gate la perderebbe nei casi chiari — quelli in cui un paziente esce senza che nessuno abbia guardato.

### 6.4 Il grafo rado, e il cambio di registro

Dove la copertura è sottile ogni percorso ignoto raggiunge limite superiore 1.0 e l'ordinamento degenera: una catena di tre ignoti supera una di due archi attestati, e l'ordine lo decidono i criteri di spareggio. Il fallimento non è rumore, è **perdita dell'ordinamento**, ed emettere i primi tre di un ordine arbitrario è peggio che non emettere nulla.

Aspettare un grafo completo non è l'alternativa: un grafo clinico non raggiunge mai la completezza, e si creerebbe uno stallo, perché la coda di completamento si alimenta dell'uso.

Quindi la copertura si valuta **localmente, per caso**, e un vicinato rado **cambia registro** invece di tacere: emette domande dirette alla base di conoscenza. Il ramo è utile dal primo giorno, puntato sulla conoscenza invece che sul paziente.

Due vincoli strutturali tengono onesta la traversata rada: **al massimo una lacuna per percorso** — due ignoti concatenati non sono un'inferenza più debole ma un oggetto diverso, il secondo condizionato alla verità del primo — e **corroborazione da due reperti indipendenti**, che sopravvive alla radità perché un percorso spurio da un reperto è facile mentre due convergenti sulla stessa condizione non lo è.

### 6.5 Le congetture diventano conoscenza

Il ramo compie salti: connessioni che nessun arco dichiara, raggiunte attraverso un meccanismo condiviso o una catena. La ricerca sull'expertise descrive così l'intuizione clinica — riconoscimento di pattern su conoscenza incapsulata, un legame le cui metà sono documentate mentre l'intero non lo è. Non ortodosso, non falso.

Il registro delle congetture fa sì che un salto che si rivela vero **diventi conoscenza che il grafo non aveva**: un nuovo arco, con i casi che l'hanno confermato.

| Passo | Condizione |
|---|---|
| Registrare | **Libero.** Ogni salto è un candidato |
| Testare | Solo contro una conferma **indipendente** dello stesso caso |
| Promuovere | Solo dopo abbastanza conferme, con limite inferiore positivo |

L'arco promosso porta l'intervallo calcolato dalle conferme: tre conferme sono visibilmente meno certe di trenta, e un salto che continua a fallire non viene promosso.

**Una congettura non è attraversabile prima della promozione.** Altrimenti il ramo raggiungerebbe il salto successivo attraverso il precedente, e una catena di connessioni non verificate crescerebbe senza che nessuna evidenza entri da nessuna parte.

Questo è il ciclo che il progetto chiamava *dall'attenzione all'intuizione*, con la traccia cartacea.

---

## 7. Il ciclo di apprendimento

### 7.1 Due velocità

| Cadenza | Cosa cambia | Rivalidazione |
|---|---|---|
| **Giornaliera** | Ricalibrazione degli intervalli del grafo dai casi confermati | No |
| **Trimestrale** | Adattamento del modello su letteratura nuova e casi | Sì, sotto piano di controllo del cambiamento |

Le due cadenze rispecchiano come apprende un clinico: i casi di ieri aggiornano subito la stima di frequenza, la letteratura nuova ristruttura i copioni più lentamente.

La differenza sostanziale non è il nome ma **dove si deposita ciò che si è appreso**. Nel riaddestramento finisce nei pesi, e un peso non dice da quale caso viene. Nella ricalibrazione finisce negli intervalli degli archi, e ogni arco porta i casi che lo hanno aggiornato.

### 7.2 Provenienza dell'etichetta

Un sistema addestrato su casi con diagnosi **documentata** — quella che l'équipe ha raggiunto e registrato, istologia, dimissione, esito — prende l'etichetta dal processo clinico, non dal proprio output.

Il registro delle conferme registra la **fonte** dell'etichetta, perché non pesano tutte uguale: l'istologia vale più di una diagnosi di dimissione non confermata. Il suo ruolo è pesare.

Resta una modalità di fallimento reale: se il suggerimento del sistema **influenza** la diagnosi documentata, il sistema impara dalle proprie proposte. È l'automation bias, ed è silenzioso perché ogni singolo passo sembra corretto. La difesa non è tecnica ma procedurale — una revisione indipendente conta solo se il revisore non ha visto il suggerimento, e uno stato di cecità non registrato viene trattato come non in cieco, perché leggere il suggerimento e concordare è la modalità di fallimento, non una conferma più debole.

`independence_rate` è osservabile nel tempo: se scende, il sistema si sta facendo confermare sempre più dal proprio accordo con sé stesso.

### 7.3 Il modello di resa

Ciò che il ramo dream apprende non è una diagnosi ma **la resa di una forma**: dato un percorso di due salti su archi ben attestati, corroborato da tre reperti, quante volte quel pattern si è rivelato rilevante?

Tassi empirici con intervalli di Wilson invece di una rete addestrata, per tre ragioni e la terza è quella che pesa: il risultato è ispezionabile, un bucket con tre osservazioni resta visibilmente largo invece di fingere conoscenza, e un tasso su esiti contati si spiega a un revisore in una frase.

Tre proprietà lo tengono onesto:

- una forma **mai osservata restituisce l'intervallo pieno**, quindi non viene né promossa né soppressa senza prove;
- l'ordinamento legge il **limite superiore**, quindi una forma non misurata sta in alto — un pattern che nessuno ha misurato è un motivo per guardare, non per nascondere;
- la soppressione richiede una stima **consolidata**, mai una manciata: tacere su prova sottile nasconderebbe i pattern rari che il ramo esiste per far emergere.

Un'ipotesi la cui pratica non ha conferma non insegna nulla: l'assenza di conferma non è prova che l'ipotesi fosse sbagliata, e contarla come fallimento addestrerebbe il modello sul silenzio.

---

## 8. Governo dell'incertezza

### 8.1 Il registro dei claim falsificabili

Un decision record contiene affermazioni di tre tipi, e assegnarne uno solo li rappresenta male tutti.

| Tipo | Natura | Cosa lo confuterebbe |
|---|---|---|
| **CONSTRAINT** | Un confine architetturale. Vale perché il progetto lo ha scelto | Niente. Cambiare idea è un ridisegno, non una confutazione |
| **PLAN** | Una decisione di sequenza, vincolante finché non accade l'evento nominato | Un'informazione nuova, non un esperimento |
| **CLAIM** | Una previsione empirica | Un'osservazione. Finché non è misurata non è decisa |

Un CLAIM scritto come "accettato" è l'errore che l'intera governance esiste per prevenire. I claim escono quindi dal documento ed entrano in un registro dove ciascuno porta l'osservazione che lo confuterebbe, risolverli richiede evidenza, e alcuni sono **bloccanti**: la strategia non può uscire dall'uso di ricerca finché restano aperti.

`blocking_open` è un intero leggibile da codice. Oggi vale **3**.

Un claim può essere **condizionale a una rotta**: prendere una strada non abbassa l'asticella, sostituisce un insieme di domande aperte con un altro. Scegliendo le API esterne, la sufficienza del modello on-premise diventa dormiente e la ricostruzione per aggregazione diventa viva — e il conteggio resta tre sotto entrambe.

### 8.2 Astensione ed escalation

Il calibratore decide se il risultato può uscire. È la componente su cui gli umani falliscono di più — la fiducia correla scarsamente con l'accuratezza — ed è quella che una regola esegue meglio, perché non si sovraconfida.

Alimentata da: divergenza dual-path, copertura misurata, rapporto di fondamento del copione, densità locale del grafo, `conflict_load` e `convergence_index`.

### 8.3 Perimetro deterministico

Restano fuori dal perimetro del modello e non negoziabili: `safety/rails.py`, `training/promotion_policy.py`, `evaluation/model_release_gate.py`, `reasoning/findings_boundary.py`.

Non perché il determinismo sia migliore in assoluto, ma perché il lavoro di quei componenti non è diagnosticare: è arbitrare e cancellare. Generare ipotesi, pesare evidenze, ordinare un differenziale sono compiti per componenti apprese. Decidere se il risultato può uscire è una regola fissa — come in un sistema di volo, dove ci sono componenti adattive ma la protezione dell'inviluppo è immutabile.

Con un'avvertenza registrata: **determinismo senza calibrazione è arbitrarietà sicura di sé**. Le soglie vanno stimate su dati e i loro effetti-scalino esaminati, altrimenti si difende l'auditabilità di un numero che nessuno ha validato.

---

## 9. Inventario dei moduli

### Memoria e conoscenza

| Modulo | Funzione |
|---|---|
| `memory/context_environment.py` | Primitive di navigazione tipizzate; offset obbligatori; ledger di copertura |
| `memory/retrieval_contract.py` | Contratto condiviso, basi di copertura, validatore dei fallimenti silenziosi |
| `memory/concept_paths.py` | Traversata limitata, intervalli, meccanismi condivisi, densità locale |
| `memory/concept_resolution.py` | Indice ontologico, risoluzione deterministica, ruoli dalla gerarchia, multilingua |
| `memory/ontology_import.py` | Annotazioni HPO come archi a intervallo, frequenze pubblicate preservate |
| `memory/graph_coverage.py` | Copertura contro un set di riferimento; guardia sull'interpretabilità |
| `memory/assertion.py` | Polarità, certezza, esperiente, temporalità, fonte → intervallo e stato |

### Ragionamento

| Modulo | Funzione |
|---|---|
| `reasoning/retrieval_reconciliation.py` | Matrice dual-path, segnale di conflitto empirico |
| `reasoning/findings_boundary.py` | Barriera applicata; ogni rifiuto nomina la rotta |
| `reasoning/illness_script.py` | Frame del copione, verificatore, integrazione delle ipotesi |
| `reasoning/clinical_discourse.py` | Lettura del ragionamento del modello come presentazione clinica |
| `reasoning/discriminating_tests.py` | Guadagno di informazione atteso; garantito, non puntuale |
| `reasoning/family_history.py` | Screening e modificatori del prior, mai reperti |

### Ipotesi e apprendimento

| Modulo | Funzione |
|---|---|
| `training/hypothesis_channel.py` | Gate di indeterminazione, involucro di esclusione |
| `training/mechanism_enumeration.py` | Ipotesi per enumerazione; cambio di registro sul grafo rado |
| `training/conjecture_ledger.py` | Salti registrati liberamente, promossi solo su conferma indipendente |
| `training/hypothesis_yield.py` | Resa per forma, appresa da conferme indipendenti |
| `governance/confirmation_registry.py` | Ammette solo conferme indipendenti dall'output |

### Valutazione

| Modulo | Funzione |
|---|---|
| `evaluation/grounding_judge.py` | Overreach strutturale; relazioni, non termini |
| `evaluation/dual_path_ab.py` | A/B appaiato, bootstrap deterministico |
| `evaluation/falsification_program.py` | Registro dei claim con criteri di confutazione |

---

## 10. Ciò che è deciso e ciò che è misurato

### 10.1 Vincoli architetturali

1. L'orchestratore è l'unica autorità diagnostica finale, ed è codice deterministico.
2. Rails, promozione e release gate restano fuori dal perimetro del modello.
3. Il substrato di memoria non viene sostituito: le relazioni tipizzate danno le affordance di navigazione.
4. L'ingestione ha due layer: sorgente immutabile e derivato ri-derivabile.
5. Il canale ipotesi è isolato strutturalmente, non per flag.
6. La familiarità non entra mai fra i reperti del paziente.
7. I modificatori non generano mai nodi.
8. L'estrazione produce intervalli e stati, mai scalari.
9. Le conferme che alimentano l'apprendimento portano la propria fonte.
10. Una congettura non è attraversabile prima della promozione.

### 10.2 Piani con trigger nominato

| Piano | Trigger di revisione |
|---|---|
| Root model: API su sintetico → open-weight on-prem | PHI nell'ambiente |
| API esterne con oracolo a predicati | Esito del vaglio legale |
| Modello diagnostico Meditron con DoRA | Divario misurato rispetto al baseline |
| Un solo adattatore, sul formato | Misura che ne mostri l'insufficienza |
| Gate di complessità sul recupero | Costo misurato del ricorsivo su lookup semplici |

### 10.3 Claim aperti

| Claim | Bloccante |
|---|---|
| `rlm.dual_path_beats_single_path` | **Sì** |
| `rlm.disagreement_is_informative` | **Sì** |
| `privacy.predicate_budget_prevents_reconstruction` | **Sì** |
| `rlm.open_weight_root_is_sufficient` | Dormiente (rotta non attiva) |
| `rlm.coverage_predicts_grounding` | No |
| `rlm.dream_hypotheses_add_value` | No |

### 10.4 Misure reali

| Misura | Valore |
|---|---|
| Archi HPO importati | 285.598 in 2,5 s |
| Archi documentati e stretti | 9,8% |
| Archi con ampiezza epistemica | 62,5% |
| Indice ontologico | 23.800 termini, 42.045 forme, 0,4 s |
| Copertura italiana | 3.488 termini (~15%), 524 ufficiali |
| Densità per area | da 65.061 a 58 annotazioni |
| Modificatori clinici | 357 termini, mai nodi |
| Test | 415 |

---

## 11. Evoluzione: le correzioni che hanno formato il sistema

Il sistema attuale è il risultato di errori corretti. Ognuno ha lasciato un principio.

| # | Formulazione iniziale | Correzione | Principio emerso |
|---|---|---|---|
| 1 | "RLM al posto di RAG" | Il substrato di memoria resta; cambia la strategia di recupero | Struttura, non ampiezza di contesto |
| 2 | Routing esclusivo | Dual-path: il disaccordo è un segnale | Modalità di fallimento opposte si compongono |
| 3 | Relazione non nel caso = errore | Tre verdetti | Inferenza e fabbricazione sono strutturalmente diverse |
| 4 | Peso singolo sugli archi | Intervalli | Il silenzio non è negazione |
| 5 | Quattro stati dell'arco | Due dimensioni continue | Le etichette non compongono; i numeri sì |
| 6 | Enumerazione filtrata per sostegno | Filtro per limite superiore | L'esplorazione legge il tetto, la decisione il pavimento |
| 7 | Isolamento "strutturale" dichiarato | Tabella di stato per livello | Una guardia mai chiamata non separa |
| 8 | Punteggio scalare per l'asserzione | Intervallo e stato | Uno scalare fa collassare quattro stati epistemici |
| 9 | Familiarità con punteggio ridotto | Canale separato | Direzione, non grandezza |
| 10 | Fine-tuning a schema rigido | Lettura del discorso clinico | Il linguaggio clinico porta più di uno schema |
| 11 | "Il modello non decide" | Il modello decide il contenuto, il calibratore lo stato | Due decisioni diverse, due componenti |
| 12 | Circolarità sovrastimata | La fonte dell'etichetta pesa, non esclude | Il rischio è l'influenza, non l'apprendimento |

Quattro di queste correzioni sono partite da obiezioni del committente, non da un'analisi interna. La #5 in particolare — la richiesta di quattro stati invece di due — ha portato alla scoperta che il 62,5% degli archi reali porta ampiezza epistemica, e che la negazione debole da sola copre 56.954 relazioni.

---

## 12. Ciò che resta

### 12.1 Prerequisiti

| # | Cosa | Natura |
|---|---|---|
| A1 | Equivalente ricorsivo del grounding score | Ingegneria |
| A2 | Corpus longitudinale con aghi congiuntivi | Ingegneria + curatela |
| A3.1 | Set di riferimento per la copertura | **Curatela clinica** |
| A3.2 | Fonti oltre il fenotipo: LOINC, ATC, ECTO | Meccanica |
| A4 | Rotazione delle credenziali | Igiene |

### 12.2 Blocchi

| Blocco | Contenuto |
|---|---|
| B | Pilota Dream Engine: enumeratore nel trainer, valutazione clinica su casi densi |
| C | Motore RLM: loop root, sandbox, traiettorie nell'audit store |
| D | Dual-path diagnostico: gate di complessità, riconciliazione in pipeline |
| E | Consolidamento: layer derivato, giudice semantico, transizione on-prem |
| F | Oracolo a predicati e budget di divulgazione |
| G | Vocabolario di staging e ciclo di completamento del grafo |
| L | Strato letteratura: estrazione di relazioni con citazione obbligatoria |

### 12.3 Vincoli d'ordine

- **A1 prima di D5.** Un A/B senza equivalente ricorsivo del grounding confronta grandezze non commensurabili.
- **A3.1 prima di B4.** Senza set di riferimento non si misura la copertura, quindi non si selezionano i casi densi, quindi la valutazione clinica misura il grafo credendo di misurare le ipotesi.
- **Intervalli prima delle importazioni ontologiche.** Altrimenti si importano migliaia di archi con valori puntuali inventati.

### 12.4 Decisioni che non sono ingegneristiche

1. Chi cura il set di riferimento della copertura — una settimana di tempo clinico che sblocca l'intera valutazione.
2. Centro di calcolo per lo stadio letterario, se la misura del divario lo giustifica.
3. Ampiezza del lessico italiano curato, dato che HPO copre il 15%.
4. Licenze del corpus: abstract PubMed per il grafo, PMC OA per il modello, accordi editoriali solo su lacuna misurata.

---

## Nota conclusiva

Il sistema non è un modello che diagnostica né un grafo che diagnostica. È **un modello che riconosce, un grafo che fonda e un calibratore che non si sovraconfida** — la struttura a tre componenti che la scienza cognitiva descrive nel diagnosta esperto, con il terzo più affidabile della sua versione umana.

La scelta ricorrente, in ogni strato, è stata far fallire il sistema **rumorosamente**: un reperto irrisolto si annuncia, una lacuna del grafo si dichiara, un'ipotesi senza percorso viene respinta con la sua rotta. È il criterio che ha guidato più decisioni di qualunque altro, e la ragione è semplice: il difetto pericoloso non è quello che rompe, è quello che assomiglia a un altro problema.

---

*Documentazione di riferimento nel repository: `rlm_on_memory_decision_record.md`, `semantic_extraction_decision_record.md`, `hypothesis_stream_decision_record.md`, `diagnostic_engine_decision_record.md`.*
