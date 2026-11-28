# Conformità e rischio

*[English](../en/COMPLIANCE_RISK.md) · Italiano · aggiornato al 25 novembre 2026*

Oggi Melampo è software di ricerca, ma è costruito come se dovesse diventare un dispositivo medico: classe IIa
dell'MDR, classe C della IEC 62304, sistema di IA ad alto rischio per l'AI Act. Dei ventuno pericoli
registrati, venti hanno un controllo verificato da test; uno è aperto.

## 1. Classificazione

| Quadro | Classe | Perché |
|---|---|---|
| MDR (UE 2017/745), regola 11 | almeno **IIa** | software che dà informazioni usate per prendere decisioni diagnostiche; IIb se un risultato sbagliato potesse causare un grave peggioramento della salute |
| IEC 62304 | **C**, provvisoria | un reperto sbagliato o mancato può contribuire a un danno grave. La classe B richiederebbe che la revisione del medico sia un controllo del rischio esterno al software, documentato e verificato; finché la maschera di revisione (costruita il 27 novembre 2026) non è validata, il progetto adotta la classe più severa |
| AI Act (UE 2024/1689) | **alto rischio**, tramite l'allegato I (dispositivi medici di classe IIa e superiori) | gli obblighi per l'IA nei dispositivi medici si applicano dal 2 agosto 2028 dopo il regolamento (UE) 2026/1744, secondo una fonte secondaria: da verificare su EUR-Lex |
| GDPR, art. 9 | dati sanitari | pseudonimizzazione alla fonte, trattamento locale, archivi cifrati (ARCHITECTURE §5) |

Che cosa chiede la classe C in più della B, e come il progetto lo soddisfa: progettazione di dettaglio di ogni
unità software (docstring dei moduli e ARCHITECTURE), verifica delle unità (pytest, circa 200 file), criteri di
accettazione di ogni unità scritti prima della misura (VALIDATION §1).

## 2. Pericoli, controlli, prove

Il registro è [`docs/registers/risks.yml`](../registers/risks.yml); `scripts/docs_check.py check` fallisce quando un
controllo nomina un test che non esiste. Gravità e probabilità sono prime stime (1–5), da rivedere con un medico.

| Id | Pericolo | Controllo | Stato |
|---|---|---|---|
| R01 | reperto attaccato alla struttura sbagliata | lettore e modelli devono essere d'accordo per un reperto certo; altrimenti in dubbio con ogni lettura; una persona solo per una discordanza del documento; le parti lette con UBERON | parziale: 1 su 24 attribuito in silenzio nell'ultimo campione nuovo valido |
| R02 | reperto giusto tolto | né il lettore né i modelli da soli mettono da parte un reperto | controllato: 0 tolti sui campioni nuovi |
| R03 | il modello afferma ciò che la frase non dice | le affermazioni devono essere parole copiate; parole marcate diverse invalidano la risposta | controllato |
| R04 | reperto assente o negato letto come presente | rilevazione delle asserzioni; le osservazioni negate passano fra le normali | controllato |
| R05 | fatto passato letto come attuale | la storia attaccata come storia | controllato |
| R06 | referto di un paziente mandato fuori dall'ospedale | il router rifiuta i modelli esterni; la procedura unica di lettura legge il testo di un paziente solo con interpreti su questa macchina; le probe richiedono corpora pubblici | controllato |
| R07 | testo non pseudonimizzato nelle revisioni o nell'addestramento | i record di revisione lo rifiutano | controllato |
| R08 | apprendimento da etichette non confermate | base clinica obbligatoria; una seconda persona conferma le revisioni | controllato |
| R09 | record di un altro paziente uniti al caso | identificativi esatti; i record estranei non vengono mai presi | controllato |
| R10 | ipotesi promossa senza conferma | numero minimo di conferme; la promozione chiede una persona | controllato |
| R11 | il memory retriever riporta prove da una memoria vuota | — | **aperto**: da correggere con un'astensione esplicita |
| R12 | esame e frase nominano lati diversi | una domanda sul lato a una persona | controllato |
| R13 | misure fatte su dati usati per tarare | split congelati, campioni nuovi, regola delle risposte perse | controllato |
| R14 | il modello non risponde | una risposta persa vale «non so» e va a una persona | controllato |
| R15 | stato normale o fisiologico letto come reperto | i modelli confrontano ogni aspetto con quello abituale; il lettore controlla con HPO e PATO | parziale: 4 su 169 nel run claims2, prima del controllo |
| R16 | misura giudicata con un valore ricordato | solo una riga di riferimento citata, ricalcolata dal lettore | parziale: 21 righe su 29 verificate; la procedura unica non confronta ancora la misura di una scheda con le righe |
| R17 | il quesito dell'impegnativa letto come reperto attuale | si leggono le sezioni, le conclusioni e la frase precedente; una contraddizione va alla persona | parziale: 4 su 99 in claims4, prima del controllo |
| R18 | la procedura unica legge i documenti del paziente prima di essere misurata | una scheda non concordata è un reperto affermato con incertezza; letture appaiate solo nella stessa frase e da ogni interprete; il lettore precedente tenuto per misurare il cambiamento | parziale: il banco della procedura unica è da fare |
| R19 | un codice dato prima di confrontare le letture | nessun codice sulla scheda; si codifica solo la lettura concordata, per scelta di ogni interprete | controllato |
| R20 | la frase di un paziente entra nella conoscenza condivisa | solo le pubblicazioni scrivono il tesauro; il testo di un paziente solo su questa macchina | controllato |
| R21 | un peso di un profilo da un numero letto male | un solo valore per misura; pesa solo se letto da entrambi gli interpreti; i giudici verificano numeri e direzione | controllato |
| R22 | un documento di un altro paziente, o non medico, nel caso | gli identificativi scritti in ogni documento confrontati con quelli del paziente, su questa macchina; il documento di un altro paziente tenuto fuori e la sua domanda a una persona; un documento di nessun tipo riconosciuto a una persona | parziale: un documento senza identificativi non si può controllare (è usato salvo che una persona dica no nella maschera di revisione); la maschera non è validata |
| R23 | un risultato deciso, trattenuto o promosso su una misura mai validata | la coerenza tra aree e le sue metriche tolte (28 novembre 2026); il percorso e l'escalation si fondano sulle discordanze del caso, ciascuna con la sua frase, che fanno escalation e non trattengono mai un risultato; la parte rapida dell'intuizione è l'ordine di rank_differential; la catena nexus non promuove nulla; rischio e incertezza pesati a mano e i loro cancelli (0,35, 0,65, 0,75) tolti: regole esplicite (discordanze, un documento non letto, una gravità dichiarata alta fanno escalation; nessuna prova, nessuna provenienza sono segnalazioni sul risultato) | parziale: la parte laterale dell'intuition engine non è scritta; la catena nexus conserva i suoi numeri non misurati, chiusa |

## 3. Software di terze parti e modelli esterni

- **Pacchetti Python:** [`docs/registers/soup.md`](../registers/soup.md), generato da `uv.lock` (41 dipendenze
  dirette, 207 pacchetti nel lock). La CI fallisce se non è aggiornato.
- **Modelli, ontologie, corpora:** [`melampo-assets.yaml`](../../melampo-assets.yaml), con licenza, revisione
  fissata e stato (`blocked_licence` dove l'uso non è permesso).
- **Licenze che limitano l'uso:** PARROT e i pesi di VoxTell sono CC BY-NC-SA 4.0 (non commerciale, condivisione
  allo stesso modo); i pesi di VL-JEPA sono solo per ricerca (licenza FAIR); i modelli NVIDIA sono solo per i
  benchmark fino a una revisione della licenza; i termini di Gemma per questo uso medico sono da verificare; Muse Spark Open chiede la dicitura "Built with Muse
  Spark Open", i suoi termini valgono per i modelli che ne derivano (anche gli adattatori LoRA) e chi lo usa non
  deve essere minorenne (ARCHITECTURE, "Lettori locali"). Una
  versione commerciale dovrebbe sostituire ciascuno di questi.
- **La licenza di Melampo:** GNU AGPLv3 con condizioni di attribuzione ai sensi della sezione 7(b) (vedi
  `LICENSE` e il README alla radice). Un decision record del 2 ottobre 2026 registrava CC BY-NC 4.0 e i punti
  legali di quel cambio; quale licenza valga lo decide la revisione legale (voce aperta).

## 4. Dove si soddisfa ogni requisito

| Requisito | Documento |
|---|---|
| Uso previsto, utilizzatori, controindicazioni (MDR allegato II §1) | [README](README.md) |
| Progettazione, architettura, flusso dei dati, supervisione umana (MDR allegato II §3; AI Act allegato IV §2–3) | [ARCHITECTURE](ARCHITECTURE.md) |
| Dati di addestramento, validazione e test; metriche; verifica (MDR allegato II §6; AI Act allegato IV §2) | [VALIDATION](VALIDATION.md) |
| Gestione del rischio (ISO 14971); SOUP (IEC 62304 §5.3, §7.1) | questo file, i registri |
| Controllo delle modifiche, risoluzione dei problemi, configurazione (IEC 62304 §6, §8, §9; AI Act allegato IV §7) | [CONTRIBUTING](CONTRIBUTING.md) |
| Segnalazione delle vulnerabilità (MDR allegato I §17.2, §17.4) | [SECURITY](../SECURITY.md) |
| Sorveglianza post-commercializzazione; valutazione clinica | non iniziate: appartengono alla fase di certificazione |

## 5. Piano delle modifiche predeterminate

Dichiarato il 27 novembre 2026 (decisione del proprietario), perché un modello addestrato di nuovo sia una modifica
prevista e valutata con il dispositivo, non un dispositivo nuovo (AI Act art. 43(4); MDCG 2025-6, sezione sulla
modifica sostanziale; la proposta COM(2025) 1023 darebbe a questi piani una base nel MDR). Nulla è ancora stato
addestrato.

**Che cosa può cambiare.** (a) L'abilità di lettura di ogni interprete della procedura unica, con un adattatore
(LoRA) di quel modello; (b) i pesi e la calibrazione della classifica delle diagnosi. La conoscenza (sinonimi,
equivalenze, i concetti su cui gli interpreti hanno concordato) non si addestra: resta nel tesauro, per ogni modello.

**Da quali dati.** (a) Le conferme di lettura (`review_feedback.ReadingConfirmation`): il dubbio di una scheda
risolto nella maschera di revisione con una risposta chiusa, confermata da una seconda persona e decisa da una terza
quando la risposta non è quella di nessun interprete; la frase tenuta com'è scritta e de-identificata
(`data/deidentify.py`: nomi, codici, contatti, indirizzi, luoghi, date spostate di un numero fisso di giorni per caso,
età oltre 89), e ammessa all'addestramento solo se entrambi gli interpreti leggono la frase de-identificata nella
scheda confermata. (b) Le diagnosi confermate ammesse da `governance/confirmation_registry.py`: una conferma da una
fonte diversa dal sistema (istologia, decorso, revisione indipendente); una diagnosi mancata perché un termine era
letto male va in (a), un peso sbagliato in (b).

**Come.** Un archivio, una procedura, un adattatore per modello: le stesse righe (la direttiva unica sulla frase
de-identificata, la scheda confermata come risposta) addestrano ogni modello con la stessa ricetta (rango, passi,
seme), su un computer dell'ospedale; l'adattatore non ne esce. I metodi che spostano un adattatore da un modello a
un altro (Trans-LoRA, Cross-LoRA, CAST, PorTAL) possono solo avviare un addestramento: oggi sono provati tra
architetture simili, non tra un ibrido Mamba MoE e un transformer denso, né su compiti clinici, e tengono nel caso
migliore l'85-95% di un adattatore addestrato sul modello. **Da ricontrollare entro marzo 2027: PorTAL
(adattamento portabile del compito, luglio 2026) e CAST (proiezione delle attivazioni), i più vicini a un solo
addestramento per ogni modello.** Le diagnosi si aggiornano prima con la ricalibrazione; si rifà il modello solo se
la ricalibrazione non raggiunge le stesse prestazioni.

**Validazione prima dell'uso.** Un caso su cinque (fissato dalla sua chiave) misura soltanto e non addestra mai. Un
adattatore o una ricalibrazione si accetta solo se, su quella parte, gli errori restano sotto il 2% senza errori nuovi
rispetto alla versione in uso, i dubbi sono contati, e (per un adattatore) il modello non restituisce le frasi su cui
è stato addestrato (prova di memorizzazione). Ogni versione accettata è registrata con le impronte dei suoi dati, del
modello di base e della ricetta; una versione che non passa non si usa, e resta quella in uso.

## 6. Voci aperte

1. Correggere R11 (il memory retriever si astiene esplicitamente).
2. Misurare la procedura unica di lettura sui banchi dei referti (R18), con criteri fissati prima del run.
3. Il gold set dei radiologi: etichette, revisione delle etichette di un solo etichettatore.
4. Validare la maschera di revisione (costruita il 27 novembre 2026), che fa della revisione del medico un controllo
   del rischio esterno e verificato.
5. Revisione legale della licenza di Melampo e delle condizioni di attribuzione.
6. Verificare le date dell'AI Act su EUR-Lex; decidere quando parte un sistema di gestione della qualità
   (ISO 13485).
7. Entro marzo 2027: ricontrollare i metodi che spostano un adattatore tra modelli (PorTAL, CAST), sezione 5.
8. Dopo la valutazione di dicembre 2026 su casi pubblicati: valutare la rimozione totale del percorso diagnostico di
   ottobre (`DifferentialEngine`, supporto/contraddizione, critico, catena nexus) con tutti i riferimenti e i file;
   il risultato viene da `case_diagnosis` dal 28 novembre 2026, il percorso di ottobre tenuto solo a confronto.
