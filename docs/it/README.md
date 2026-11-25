# Melampo: che cos'è e a che cosa serve

*[English](../en/README.md) · Italiano · aggiornato al 25 novembre 2026*

Melampo legge un caso clinico (la nota del medico, i referti, le tabelle di laboratorio, le immagini) e propone
una diagnosi differenziale strutturata, con le prove a favore e contro ogni ipotesi e un «per ora non posso
dirlo» esplicito quando le prove non reggono e servono altri accertamenti. È un **sistema di ricerca**: non è un
dispositivo medico e non dà diagnosi ai pazienti.

## Uso previsto

| | |
|---|---|
| **Che cosa fa** | Legge documenti clinici in italiano e in inglese; descrive ogni reperto per che cosa si osserva, dove, con quale metodo e con quale interpretazione, e lo codifica dopo (HPO, UBERON, il tesauro); costruisce il grafo del caso; rivede la letteratura medica per i profili delle malattie; ordina le ipotesi con le loro prove; manda a una persona ogni lettura che non può confermare. |
| **Per chi** | Ricercatori e medici che lavorano al progetto, su corpora pubblici e su casi pseudonimizzati secondo le regole dei dati del progetto. |
| **Non per** | Diagnosi o cura di un paziente; uso senza un medico che legga ogni risultato; invio di referti di pazienti a servizi fuori dall'ospedale. |
| **Stato regolatorio** | Software di ricerca, senza marcatura CE. Progettato verso la classe IIa dell'MDR e la classe C della IEC 62304 (vedi [COMPLIANCE_RISK](COMPLIANCE_RISK.md)). |

## A che punto siamo

Verificato sul codice di `main`; la mappa completa è in [ARCHITECTURE](ARCHITECTURE.md).

| Parte | Stato | Nota |
|---|---|---|
| Acquisizione dei documenti (PDF, DOCX, immagini, DICOM) e orchestratore dei referti | funziona | il modello dice solo il tipo di documento; indizi deterministici decidono insieme a lui |
| Una sola procedura di lettura, per i documenti del caso e per le pubblicazioni | **funziona, non ancora misurata** | predefinita per i referti dal 25 novembre 2026; una scheda, i due interpreti confrontati campo per campo, i codici solo dopo; il testo di un paziente solo su questa macchina; il suo banco è la prossima misura |
| Il lettore precedente (linker anatomico, lettore Kintsch, finding owner, affermazioni) | tenuto per misurare il cambiamento | `MELAMPO_REPORT_READER=anatomy`; `ctrate9`: 0 errori, 3 dubbi (1,0%); il lato immagine e il riassunto del patologo sono ancora solo qui |
| Codici dopo l'accordo (HPO, il tesauro) | funziona | si codifica solo la lettura concordata; il tesauro lo scrivono solo le pubblicazioni |
| Grafo del caso e percorso diagnostico | funziona | quattro stati di ogni manifestazione, rapporti di verosimiglianza, candidate solo dal percorso, esami da proporre |
| Profili delle malattie: revisione sistematica per malattia | funziona, primi run | appendicite acuta: 79 manifestazioni, 6 con un'accuratezza misurata; il run con la procedura unica è il prossimo |
| Conoscenza versionata (GitHub Release, FalkorDBLite) | funziona | una Release solo quando il contenuto cambia; rivista ogni lunedì |
| Libreria dei valori di riferimento | parziale | 21 righe su 29 verificate; la procedura unica non la confronta ancora |
| Ramo dei farmaci | funziona con la cache locale | i farmaci di un paziente si leggono solo dalla cache su questa macchina |
| Grafo dei concetti (FalkorDB) e ramo nexus | funziona | 636.733 archi verificati |
| Conferma e unione dei casi | funziona | una conferma senza base clinica resta fuori dall'addestramento |
| Memory retriever | **difetto noto** | con la memoria vuota inventa tre prove |
| Addestramento (DPO, LoRA) | mai eseguito | nessuna GPU, ancora nessuna etichetta confermata |
| Maschera di immissione e revisione del medico | da costruire | le domande che mostrerà sono pronte |

## Mappa del repository

```text
src/melampo/        il sistema
  data/             acquisizione, orchestratore dei referti, lettura dei referti
  reading/          la procedura unica di lettura: scheda, confronto, codici dopo, tesauro
  clinical/         grafo del caso, stati delle manifestazioni, ramo dei farmaci
  memory/           ontologie, profili delle malattie, grafo della conoscenza; il lettore precedente (linker, Kintsch, affermazioni)
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
