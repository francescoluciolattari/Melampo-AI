# Il lettore narrativo e l'interprete d'immagine nell'ingestione (30 ottobre 2026)

## Prima

L'orchestratore dei referti indicava `narrative_reader` e `imaging` tra gli interpreti di un documento, ma nessuno li eseguiva. Il linker, il lettore Kintsch, VoxTell e l'atlante TotalSegmentator erano misurati nelle sonde (role-probe, Kaggle) e non giravano mai su un caso.

## Ora (`data/report_reading.py`, chiamato da `ClinicalIngestionPipeline.from_payload`)

1. **Che cosa si legge.** La nota del medico (il `report_text` com'era prima degli allegati) e il testo di ogni allegato. La nota passa dall'orchestratore come gli allegati; gli allegati hanno già il loro smistamento. Un testo è letto se l'orchestratore gli assegna il lettore narrativo, oppure se lo riconosce come referto di imaging. Un referto di laboratorio puro non viene letto.
2. **Lettore narrativo.** Trova i nomi anatomici noti (lessico e tabella delle parti, prima i più lunghi) e il linker li collega nella loro frase, leggendo il referto intero (`ReportState`).
   - Un link accettato diventa una riga di sede (struttura, ruoli, qualificatori: sede, carattere, certezza, negazione).
   - Un'astensione tiene il suo motivo e va nell'elenco `to_a_person`.
3. **Interprete d'immagine** (`interpret_exam`), sui soli referti di imaging. Una sede diventa un reperto quando la frase la descrive: con una misura, un carattere, un grado o un'incertezza. "Fegato nei limiti" resta una sede e non diventa un reperto.
   - Il lato del testo c'è sempre.
   - Il lato dell'immagine c'è solo se il referto è abbinato al suo esame **senza ambiguità**: un solo referto di imaging e una sola serie DICOM usabile come volume (TC, RM o PET) fra gli allegati, e i segmentatori configurati. VoxTell isola la lesione descritta dalla frase, TotalSegmentator dà l'atlante del paziente, `image_evidence` controlla organo, misura e carattere.
   - Altrimenti ogni reperto dice perché non c'è immagine: `no_volume_in_the_attachments`, `exam_not_paired:<n>_imaging_reports,<m>_volumes`, `segmenters_not_configured`, `volume_unreadable:...`, `segmenter_unavailable:...`.
4. **Dove va il risultato.** In `CaseContext.provenance["report_reading"]`, serializzabile in JSON e senza pixel: testi letti, sedi, astensioni, reperti (`Finding.as_row`) e l'elenco di ciò che deve vedere una persona. Se la lettura fallisce, il motivo è scritto e il caso viene ingerito lo stesso. **La diagnosi non usa ancora questi reperti**: entrano come provenienza del caso, non come `findings` del grafo. Collegarli al grafo (sedi e reperti come nodi, come i fenotipi del laboratorio) è il passo successivo, da misurare.

## Configurazione (ambiente)

| Variabile | Effetto | Default |
|---|---|---|
| `MELAMPO_NARRATIVE_READER=off` | non legge i referti | acceso: il linker si costruisce da `data/linking` in circa un secondo, una volta sola |
| `MELAMPO_UBERON` | percorso di `uberon.obo`: grafo anatomico, controllo dei vicini, proposte di ripiego al padre | assente |
| `MELAMPO_CI_RESOURCES` | cartella di `build_ci_resources.py`: lettore Kintsch in modalità "review" (un disaccordo diventa un'astensione) | assente |
| `MELAMPO_LOCAL_CHAT_ENDPOINT`, `MELAMPO_LOCAL_CHAT_MODELS` | modelli **locali** per lo stadio di scelta del linker; un host non locale è rifiutato | assenti |
| `MELAMPO_IMAGING_SEGMENTERS=voxtell` | VoxTell e TotalSegmentator (`MELAMPO_IMAGING_DEVICE`: cuda o cpu) | assenti: reperti solo sul testo |

`ClinicalIngestionPipeline(report_readers=..., read_reports=False)` permette di passare lettori propri (nei test) oppure di lasciare fuori la lettura.

## Limiti

- **L'abbinamento referto-esame è prudente.** Con due referti di imaging e una serie, o con due serie, non si abbina nulla. La regola per abbinare per data, modalità e distretto va scritta e misurata. Un referto non deve mai essere letto su un esame che non descrive.
- **Il lettore Kintsch e i modelli locali** sono attivi solo se configurati. Senza, il linker è quello deterministico: accetta i nomi esatti e si astiene sul resto.
- **I qualificatori di sede possono sbagliare l'incertezza.** In "Ipodensità di 8 mm nel lobo epatico destro, verosimilmente cisti" la certezza esce "factual", perché "verosimilmente" si riferisce a "cisti" e non alla sede. Il reperto resta, ma l'incertezza della diagnosi non è letta qui.
- **Il lettore narrativo legge anche le note cliniche e gli istologici** (l'orchestratore gli assegna anche quelli), ma l'interprete d'immagine gira solo sui referti di imaging.

## Test

`tests/test_report_reading.py`:

- nota letta, con sedi e reperti sul testo;
- caso senza linker;
- referto di laboratorio non letto;
- serie DICOM reale abbinata, con lesione isolata e misurata (12 mm, `agree`);
- abbinamento rifiutato quando è ambiguo;
- esame senza segmentatori;
- modello remoto rifiutato.
