# Conferma e unione dei casi in sospeso (30 ottobre 2026)

## I tre difetti, com'erano

1. **Una conferma veniva diagnosticata come un caso nuovo.** Con `confirmed_diagnosis` il router dei casi rispondeva `confirm_and_train`, ma la pipeline rifaceva tutta la diagnosi, metteva in coda un nuovo record in sospeso e non registrava la conferma.
2. **L'unione perdeva tutto tranne il testo.** Il caso in sospeso conservava solo `report_text`: findings del medico e fenotipi del laboratorio dei passaggi precedenti andavano persi, e il grafo ripartiva dai findings dell'ultimo invio.
3. **La lettura dei referti vedeva solo l'aggiornamento.** Il testo unito veniva scritto nel caso dopo l'ingestione.

## Le decisioni di Frank

1. **Base clinica della conferma: opzione C.** Arriva nel payload (`confirmation_source`: `histopathology`, `clinical_outcome`, `independent_review` con `reviewer_blinded_to_suggestion`, `reference_standard`) oppure dalla maschera del medico, che chiama la stessa funzione. Senza base il caso si chiude e si conserva, ma non entra nell'apprendimento finché un medico non la dichiara.
2. **Conferma e dati nuovi nello stesso invio: opzione A.**
   - Si chiude il caso confrontando la conferma con la proposta fatta *prima*.
   - I dati nuovi si leggono senza rifare la diagnosi.
   - Un reperto in una sede che la diagnosi non nomina va al medico.
   - **I nuovi referti restano nel flusso di apprendimento.**

## Com'è ora

**`confirm_and_train`** (`ClinicalInferencePipeline._close_confirmed_case`):

- Nessuna diagnosi e nessun record nuovo in coda. Il risultato riporta:
  - `status`;
  - l'esito della chiusura (confronto con la proposta, casi chiusi);
  - `learning` (`admitted`, `rejection`, `awaiting_clinical_basis`);
  - dove è conservato il caso (`encrypted` con `DB_PASSWORD`, altrimenti `this_process_only`);
  - `to_the_physician`.
- I dati arrivati con la conferma (testo, allegati, laboratorio, lettura dei referti) vengono letti dall'ingestione e conservati nel record chiuso, sotto `after_confirmation`. Il record li separa sempre da ciò che il sistema aveva proposto, perché sono scritti conoscendo la risposta.
- **Reperti per il medico.** La diagnosi confermata viene letta dallo stesso linker ("adenocarcinoma of the right upper lobe" → quel lobo). Un reperto in una struttura di un'altra famiglia è un possibile problema nuovo (`site_not_named_by_the_diagnosis`). Se la diagnosi non nomina una sede che il linker sa collocare ("Sarcoidosi", "adenocarcinoma polmonare": il polmone intero non è una classe), ogni reperto nuovo va al medico con quel motivo. Aprire un caso nuovo resta una decisione del medico.
- `case_confirmation.cases_awaiting_source` elenca i casi chiusi senza base. `declare_confirmation_source` completa la base e li ripropone al registro. Una base già registrata non si riscrive, e una revisione che ha visto la proposta resta non indipendente.
- Senza `DB_PASSWORD` il caso chiuso è tenuto in memoria (`InMemoryConfirmedCaseStore`, con anonimizzazione) e il risultato lo dice.

**Apprendimento** (`training_extraction.extract_and_purge`):

- Un record con referti dopo la conferma non viene cancellato dopo l'estrazione delle coppie DPO. Viene marcato `dpo_extracted` e tenuto, e le sue coppie non si ricostruiscono una seconda volta.
- I record riscritti non vengono anonimizzati due volte (`ConfirmedCaseStore.update`). Un HMAC di un HMAC non corrisponderebbe più agli altri record del paziente.

**`merge_and_rerun`** (`_merged_payload`, prima dell'ingestione):

- Il testo unito è il referto da cui si legge il caso.
- La nota del medico è unita allo stesso modo.
- I findings sono l'unione: prima quelli precedenti, senza ripetizioni.
- Il caso in sospeso conserva ora `findings` e `physician_note`.

**Lettura dei referti.** Ogni testo letto dà i suoi reperti, non solo i referti di imaging: una lettera o un istologico che descrive "una massa di 32 mm nel lobo superiore destro" afferma un reperto. Il lato immagine resta solo per i referti di imaging abbinati al loro esame.

## Limiti

- Il confronto di famiglia (sinistra e destra dello stesso organo, segmenti epatici) è grossolano: un nodulo nel lobo inferiore destro, con una diagnosi sul lobo superiore destro, va al medico. È voluto: in caso di dubbio decide il medico.
- La maschera del medico non esiste ancora. Oggi la base arriva solo dal payload o da `declare_confirmation_source`.
- L'orchestratore ha classificato "TC del torace. Nel lobo superiore destro massa…" come nota clinica, non come referto di imaging: gli indizi d'imaging di un testo breve sono deboli. Con un solo referto e un solo volume, l'immagine di quel referto non verrebbe cercata. Va misurato sui referti veri.

## Test

`tests/test_confirmation_and_merge.py`, sulla pipeline vera:

- la conferma chiude il caso senza diagnosi né nuovo record;
- un caso senza base resta in attesa; una base dichiarata dopo lo ammette; una revisione non cieca resta esclusa;
- il nuovo referto resta nel record e il reperto nel fegato va al medico;
- con una diagnosi senza sede, ogni reperto va al medico;
- l'unione tiene i findings precedenti e legge il caso intero;
- l'estrazione DPO tiene il record con i referti successivi alla conferma e non ricostruisce le coppie.
