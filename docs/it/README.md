# Melampo: che cos'è e a che cosa serve

*[English](../en/README.md) · Italiano · aggiornato al 2 novembre 2026*

Melampo legge un caso clinico (la nota del medico, i referti, le tabelle di laboratorio, le immagini) e propone
una diagnosi differenziale strutturata, con le prove a favore e contro ogni ipotesi e un «non posso dirlo»
esplicito quando le prove non reggono. È un **sistema di ricerca**: non è un dispositivo medico e non dà diagnosi
ai pazienti.

## Uso previsto

| | |
|---|---|
| **Che cosa fa** | Legge documenti clinici in italiano e in inglese; collega ciò che dicono all'anatomia (classi di TotalSegmentator ancorate a UBERON) e ai fenotipi (HPO); costruisce il grafo del caso; ordina le ipotesi con le loro prove; manda a una persona ogni lettura che non può confermare. |
| **Per chi** | Ricercatori e medici che lavorano al progetto, su corpora pubblici e su casi pseudonimizzati secondo le regole dei dati del progetto. |
| **Non per** | Diagnosi o cura di un paziente; uso senza un medico che legga ogni risultato; invio di referti di pazienti a servizi fuori dall'ospedale. |
| **Stato regolatorio** | Software di ricerca, senza marcatura CE. Progettato verso la classe IIa dell'MDR e la classe C della IEC 62304 (vedi [COMPLIANCE_RISK](COMPLIANCE_RISK.md)). |

## A che punto siamo

Verificato sul codice di `main`; la mappa completa è in [ARCHITECTURE](ARCHITECTURE.md).

| Parte | Stato | Nota |
|---|---|---|
| Acquisizione dei documenti (PDF, DOCX, immagini, DICOM) e orchestratore dei referti | funziona | il modello dice solo il tipo di documento; indizi deterministici decidono insieme a lui |
| Linker anatomico con il lettore Kintsch e i modelli locali | funziona | i referti dei pazienti li leggono solo modelli locali |
| «Di chi è il reperto?» (finding owner) | funziona, domanda v1 | 1 reperto su 24 di un'altra struttura attribuito in silenzio; 0 su 184 reperti giusti tolti |
| Lettura affermazione per affermazione | costruita, **spenta** | i primi run non hanno passato i criteri; secondo campione pronto |
| Dal reperto al termine HPO | solo nelle prove | 90% di risposte sicure contro una soglia del 95%: non entra nel grafo |
| Grafo dei concetti (FalkorDB) e ramo nexus | funziona | 636.733 archi verificati |
| Conferma e unione dei casi | funziona | una conferma senza base clinica resta fuori dall'addestramento |
| Memory retriever | **difetto noto** | con la memoria vuota inventa tre prove |
| Addestramento (DPO, LoRA) | mai eseguito | nessuna GPU, ancora nessuna etichetta confermata |
| Maschera di immissione e revisione del medico | da costruire | le domande che mostrerà sono pronte |

## Mappa del repository

```text
src/melampo/        il sistema
  data/             acquisizione, orchestratore dei referti, lettura dei referti
  memory/           linker, lettore Kintsch, finding owner, affermazioni, ontologie
  imaging/          interprete d'immagine, segmentatori, reperti
  reasoning/        pipeline, aree funzionali, intuizione, differenziale, orchestratore
  orchestration/    router dei modelli, runtime degli specialisti
  training/         archivi delle conferme, coppie di preferenza, scheduler nexus
  evaluation/       benchmark, gold set, lettura con LLM, feedback delle revisioni
scripts/            probe e benchmark (lanciati dai workflow)
tests/              circa 200 file di test (pytest)
data/               ontologie e riferimenti (pin DVC), etichette gold (data/gold/)
.github/workflows/  CI e probe, tutti ad avvio manuale tranne la CI
docs/en, docs/it    questa documentazione, in inglese e in italiano
docs/registers/     registro dei rischi e software di terze parti (SOUP)
```

## Avvio rapido

```bash
uv sync --extra dev --extra data      # l'ambiente esatto di uv.lock (Python 3.12+)
uv run dvc update -R data             # dati di riferimento, verificati contro i loro pin
uv run pytest -q                      # tutta la suite
```

I test che richiedono FalkorDB, campioni DICOM o la rete falliscono senza di essi; la CI fornisce ciò che serve.

## La documentazione

| File | Risponde a |
|---|---|
| [README](README.md) | che cos'è, per chi, a che punto è |
| [ARCHITECTURE](ARCHITECTURE.md) | com'è fatto e dove vanno i dati |
| [VALIDATION](VALIDATION.md) | come sappiamo che funziona: dati, gold set, probe, criteri, risultati |
| [COMPLIANCE_RISK](COMPLIANCE_RISK.md) | che cosa può andare storto, come lo si ferma, quali norme valgono |
| [CONTRIBUTING](CONTRIBUTING.md) | come cambiarlo senza rompere niente |
| [RADIOLOGIST_GUIDE](RADIOLOGIST_GUIDE.md) | come i radiologi etichettano il gold set |

I documenti precedenti al 2 novembre 2026 (diario delle ricerche, decision record, trattato) restano invariati nel
tag git `docs-archive-2026-11`: `git show docs-archive-2026-11:docs/<file>`.
