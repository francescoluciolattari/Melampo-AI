# Conformità e rischio

*[English](../en/COMPLIANCE_RISK.md) · Italiano · aggiornato al 4 novembre 2026*

Oggi Melampo è software di ricerca, ma è costruito come se dovesse diventare un dispositivo medico: classe IIa
dell'MDR, classe C della IEC 62304, sistema di IA ad alto rischio per l'AI Act. Dei diciassette pericoli
registrati, sedici hanno un controllo verificato da test; uno è aperto.

## 1. Classificazione

| Quadro | Classe | Perché |
|---|---|---|
| MDR (UE 2017/745), regola 11 | almeno **IIa** | software che dà informazioni usate per prendere decisioni diagnostiche; IIb se un risultato sbagliato potesse causare un grave peggioramento della salute |
| IEC 62304 | **C**, provvisoria | un reperto sbagliato o mancato può contribuire a un danno grave. La classe B richiederebbe che la revisione del medico sia un controllo del rischio esterno al software, documentato e verificato; finché la maschera di revisione non esiste ed è validata, il progetto adotta la classe più severa |
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
| R06 | referto di un paziente mandato fuori dall'ospedale | il router rifiuta i modelli esterni; le probe richiedono corpora pubblici | controllato |
| R07 | testo non pseudonimizzato nelle revisioni o nell'addestramento | i record di revisione lo rifiutano | controllato |
| R08 | apprendimento da etichette non confermate | base clinica obbligatoria; una seconda persona conferma le revisioni | controllato |
| R09 | record di un altro paziente uniti al caso | identificativi esatti; i record estranei non vengono mai presi | controllato |
| R10 | ipotesi promossa senza conferma | numero minimo di conferme; la promozione chiede una persona | controllato |
| R11 | il memory retriever riporta prove da una memoria vuota | — | **aperto**: da correggere con un'astensione esplicita |
| R12 | esame e frase nominano lati diversi | una domanda sul lato a una persona | controllato |
| R13 | misure fatte su dati usati per tarare | split congelati, campioni nuovi, regola delle risposte perse | controllato |
| R14 | il modello non risponde | una risposta persa vale «non so» e va a una persona | controllato |
| R15 | stato normale o fisiologico letto come reperto | i modelli confrontano ogni aspetto con quello abituale; il lettore controlla con HPO e PATO | parziale: 4 su 169 nel run claims2, prima del controllo |
| R16 | misura giudicata con un valore ricordato | solo una riga di riferimento citata, ricalcolata dal lettore | parziale: le righe sono da verificare |
| R17 | il quesito dell'impegnativa letto come reperto attuale | si leggono le sezioni, le conclusioni e la frase precedente; una contraddizione va alla persona | parziale: 4 su 99 in claims4, prima del controllo |

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

## 5. Voci aperte

1. Correggere R11 (il memory retriever si astiene esplicitamente).
2. Il gold set dei radiologi: etichette, revisione delle etichette di un solo etichettatore.
3. La maschera di immissione e revisione, che fa della revisione del medico un controllo del rischio esterno e
   verificato.
4. Revisione legale della licenza di Melampo e delle condizioni di attribuzione.
5. Verificare le date dell'AI Act su EUR-Lex; decidere quando parte un sistema di gestione della qualità
   (ISO 13485).
