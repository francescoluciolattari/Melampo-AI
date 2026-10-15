# Classificazioni italiane: CIPI e ICD-9-CM aggiornati ogni settimana (14 ottobre 2026)

## Perché

Il lettore Kintsch non ha un vocabolario italiano. Per lui "asportazione", "liberazione", "intervento", "TC" non hanno un tipo, e la struttura che reggono la legge come una struttura semplice. I titoli delle procedure scritti dai codificatori nella classificazione italiana sono la fonte naturale dei tipi delle parole italiane.

La **CIPI** è la Classificazione italiana delle procedure e degli interventi chirurgici (DM Salute 23 ottobre 2025, in vigore dal 1° gennaio 2027). Il Ministero la pubblica con le tavole di transcodifica dalla **ICD-9-CM**, la classificazione in uso fino ad allora. Licenza per la CIPI confermata da Frank il 14 ottobre 2026. La ICD-10-IM (diagnosi) non si scarica finché la sua licenza non è verificata.

## Il workflow `classifications-it`

Gira ogni lunedì e si può lanciare a mano.

1. **Trova i file** nelle pagine che li elencano: le notizie NSIS-CLASS di reteclassificazioni.it e la pagina CIPI del Ministero. I link al visualizzatore Office vengono risolti nel file vero; oggi sono `6.TAB_CIPI_V.GAMMA_2.0.xlsx` e `7.TRANSCOD_ICD-9-CM-CIPI_V.GAMMA_2.0.xlsx`.
2. **Li scarica**, riprovando se il sito risponde con una pagina invece del file, e calcola lo SHA-256 di ciascuno.
3. **Li confronta** con `data/classifications_it/manifest.json`: un file nuovo o cambiato vuol dire `changed=true`.
4. **Costruisce il dataset** (`data/classifications_it/`):
   - `codes.jsonl`: un codice per riga (sistema, codice, titolo italiano, versione);
   - `transcoding.jsonl`: i legami ICD-9-CM → CIPI;
   - `unified.jsonl`: ogni codice CIPI con i suoi codici e titoli ICD-9-CM;
   - `italian_heads.json`: la parola testa di ogni titolo di procedura (in italiano la testa è la prima parola) con il numero di titoli che la usano. Diventa "procedura" solo se regge almeno 3 titoli, cioè una corrispondenza, non un'ipotesi da un titolo solo;
   - `manifest.json`: file, URL, versione, SHA-256, colonne trovate, conteggi.
5. **Carica il dataset in FalkorDB** (embedded) per verificare che si carichi: grafo `classifications_it`, nodi `Code`, archi `MAPS_TO` da ICD-9-CM a CIPI.
6. **Se qualcosa è cambiato, apre una pull request** (`chore/classifications-it`): propone, non scrive mai su main, come il workflow degli aggiornamenti HPO.

**Le intestazioni delle tabelle si riconoscono dalle parole** ("codice", "descrizione", "ICD-9-CM", "CIPI"), non dalla posizione: tra una versione e l'altra l'ordine delle colonne può cambiare. Il primo run scriverà nel manifest le intestazioni trovate, da controllare.

**Il primo run (15 ottobre 2026).** Il sito del Ministero controlla il browser prima di dare un file: il workflow lo scarica con un Chromium headless (Playwright) e funziona (`fetched_by: browser`). Le intestazioni vere hanno mostrato un errore di lettura: "Descrizione CIPI" e "Codifica ICD9CM" contengono entrambe "cod", e il riconoscimento guardava "cod" prima di "descr". Ora la descrizione si riconosce per prima, e la tabella CIPI (file 6) e la transcodifica (file 7) si leggono ciascuna con le sue colonne. Conteggi dai file veri (versione Gamma 2.0):

| | |
|---|---|
| codici CIPI | 15.164 (con sezione, tipo, inclusioni, esclusioni, note, validità) |
| codici ICD-9-CM | 4.036 |
| legami ICD-9-CM → CIPI | 22.222 |
| parole testa italiane con un tipo | 756 |

I due file Excel sono nel repository (`data/classifications_it/source/`), licenza CIPI confermata da Frank: se il sito blocca il download, il workflow usa quella copia.

**La pull request automatica non si è aperta.** Il ramo `chore/classifications-it` è stato spinto, ma l'ultimo passo si ferma: di solito è l'impostazione del repository *Settings → Actions → General → Workflow permissions → "Allow GitHub Actions to create and approve pull requests"*, spenta. Va accesa una volta; vale anche per `radiology-kinds`.

**I codici ICD-9-CM da HL7.** I CodeSystem ICD-9-CM di HL7 (terminology.hl7.org) contengono solo i metadati, senza codici né titoli, e sono in inglese. I titoli ICD-9-CM in italiano arrivano dalla tavola di transcodifica del Ministero.

## Le parole testa italiane nel lettore: misurato, non tenuto (15 ottobre 2026)

Ho dato `italian_heads.json` al vocabolario dei blocchi del lettore, con la stessa regola di RadLex (riempie solo le parole che NCIt non conosce o tiene per "concettuali"), e ho rimisurato sugli item del run completo:

| | solo NCIt | NCIt + teste CIPI/ICD-9-CM |
|---|---|---|
| lettore italiano giusto | 297/375 | 293/375 |
| sedi di procedura italiane viste dal lettore | 0/72 | 0/72 |
| Nemotron, a una persona (it) | 7 | 7 |
| Nemotron, d'accordo senza ruolo comune (it) | 60 | 64 |
| inglese | invariato | invariato |

Nessun guadagno e un piccolo peggioramento, quindi quel collegamento (parole CIPI come riempitivo del lettore inglese) fu tolto. Poi, lo stesso giorno, le parole italiane sono tornate **insieme alla grammatica italiana** (testa a sinistra) e alle parole grammaticali che non danno mai un tipo: italiano 297→336/375, sedi di procedura 0→54/72. Vedi `lettura_italiana_e_t11_2026-10-15.md`.

**Perché non aiuta.** Il reticolo compone i blocchi italiani come quelli inglesi: il tipo del blocco viene dall'ultima parola. In "asportazione del rene" la testa è la prima parola, quindi sapere che "asportazione" è una procedura non cambia niente; cambia solo qualche parola isolata, e in peggio. Il vocabolario serve solo se il lettore compone l'italiano con la testa a sinistra.

**Dove può servire, da misurare:**
- nel reticolo, una composizione italiana con la testa a sinistra ("nome + di/del + nome"), che decide dal contesto e non da una lista;
- nel prompt del metodo (`llm_method`) come esempi di parole che reggono;
- come dati di addestramento LoRA (titoli di procedura con la sede).

## Due domande di Frank sul lettore

**"T11 level": il lettore non sa o non capisce?** Non capisce, perché gli manca il pezzo che conta.

- "T11" (undicesima vertebra toracica) non è nel suo vocabolario come anatomia: il linker ha uno stadio dei livelli vertebrali, il lettore no.
- Il reticolo compone il blocco "approximately t11 level" e prende l'avverbio "approximately" come predicato.
- Restano tre indizi deboli: il blocco, il predicato e il contesto del documento. Tra struttura e nome di altro finisce 0,501 contro 0,499, cioè un pareggio. Il "non so" è onesto, ma nasce da un'evidenza povera, non da un dubbio vero sul testo.

**"CT brain": perché il ruolo lo vede solo il modello?** Perché il lettore non sa che cosa sia "CT".

- In NCIt "ct" è sigla di geni, proteine e procedure insieme, quindi nessun tipo supera la soglia.
- Il reticolo legge "follow up ct brain" come un composto inglese e prende il tipo dall'ultima parola, "brain", cioè anatomia.
- Il lettore non vede mai un esame; il modello sa che CT è una tomografia computerizzata.
- Il rimedio è lo stesso dell'italiano: un vocabolario che abbia la sigla con il suo tipo (RadLex: CT e MRI sono modalità d'esame), più il reggente "scritto prima della menzione" già nel parser.

## Fonti

- [reteclassificazioni.it, Progetto NSIS-CLASS: documentazione](https://www.reteclassificazioni.it/portal_main.php?portal_view=news_dettaglio&id=116)
- [Ministero della Salute, la classificazione delle procedure CIPI](https://www.salute.gov.it/new/it/tema/assistenza-ospedaliera-sdo/la-classificazione-delle-procedure-cipi/)
- [DM Salute 23 ottobre 2025](https://www.mefop.it/cms/doc/29766/dm-salute-23-ottobre-2025.pdf)
- [HL7 Terminology, ICD-9-CM (procedure codes)](https://terminology.hl7.org/6.5.0/CodeSystem-ICD-9CM-procedurecodes.html)
