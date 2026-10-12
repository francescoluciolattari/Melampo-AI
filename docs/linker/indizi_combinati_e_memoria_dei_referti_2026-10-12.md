# Indizi combinati, sequenze di tre parole e memoria che impara dai referti (12 ottobre 2026)

Seguito di `collocazioni_cornici_blocchi_2026-10-10.md`. Frank ha chiesto i passi 2 (sequenze più lunghe delle coppie) e 3 (tutti gli indizi di un lettore insieme), e se si trovano referti veri anonimizzati per simulare l'apprendimento del lettore di Kintsch.

**Nessun numero qui è un test.** Misure sullo sviluppo (MedMentions dev); lo split congelato non è stato letto.

## 1. Passi 2 e 3 sugli span di MedMentions

`scripts/chunk_probe.py`. Stessa domanda di prima: due parole vicine sono dentro lo stesso span annotato o a cavallo di un confine?

Questa volta ogni coppia riceve **tutti** gli indizi che ha un lettore del dominio, e la misura è fatta su **tutte** le 125.895 coppie, non solo su quelle già viste:

- statistica delle sequenze: npmi della coppia; P(y | w x) e P(x | y z) dalle **terne ordinate** (passo 2);
- memoria dei nomi (nomi di strutture e di altre cose dentro i quali cade la coppia);
- il tipo NCIt di ciascuna parola;
- la valenza: la seconda parola è una testa di procedura con un posto sito o dispositivo;
- la grammatica: parole vuote, numeri, forme verbali.

**Come si combinano.** Regressione logistica con i valori di default di scikit-learn, nulla regolato. I documenti di addestramento sono divisi a metà: i conteggi vengono dalla metà A (più CRAFT e definizioni NCIt), la regressione si adatta sugli span della metà B, e i numeri si leggono sui documenti dev.

| indizi | AUC | precisione "dentro" | richiamo | F1 | span letti come un'unità |
|---|---|---|---|---|---|
| solo statistica delle sequenze | 0,749 | 0,74 | 0,26 | 0,38 | 21,9 % |
| solo memoria dei nomi | 0,512 | 0,94 | 0,02 | 0,05 | 1,8 % |
| solo tipi delle parole | 0,631 | 0,66 | 0,07 | 0,13 | 4,9 % |
| solo valenza | 0,503 | 0,85 | 0,01 | 0,01 | 0,5 % |
| solo grammatica | 0,803 | 0,65 | 0,03 | 0,05 | 1,9 % |
| **tutti gli indizi** | **0,869** | 0,75 | 0,44 | **0,56** | **37,6 %** |
| tutti tranne la statistica | 0,828 | 0,62 | 0,35 | 0,44 | 26,1 % |
| tutti, coppie ma senza terne | 0,859 | 0,75 | 0,41 | 0,53 | 34,7 % |

21.219 span di più parole.

**Lettura.**
- **Passo 2.** Le terne aggiungono poco ma in modo netto: +0,010 di AUC, +0,023 di F1, +2,9 punti di span letti interi.
- **Passo 3.** Combinare serve. Su tutte le coppie l'AUC sale da 0,75 (statistica sola) a 0,87, e gli span letti interi da 22 % a 38 %. Senza la statistica delle sequenze si scende a 0,83: i conteggi portano conoscenza che i nomi e i tipi non hanno.
- **Gli indizi di conoscenza** (nomi, valenza) sono molto precisi quando ci sono (0,85-0,94), ma ci sono raramente: la memoria è piccola. È di nuovo il punto di Kintsch.
- **Perché non 1.** Le etichette sono una convenzione: MedMentions segna il concetto UMLS più specifico, senza annidamenti, e l'accordo fra annotatori non è pubblicato. Il tetto vero si misura solo con due lettori ciechi (gold set).

## 2. Memoria che impara leggendo referti

`scripts/report_memory_probe.py`. Si simula un medico in formazione che legge referti. Per ogni corpus si tiene da parte il 10 % dei referti; la memoria legge il 0, 10, 25, 50 e 100 % del resto, sopra la memoria di base (abstract, CRAFT, definizioni NCIt: 5,7 M parole). Su quelli mai letti si conta:
- quante coppie e terne di parole la memoria conosce (viste almeno 3 volte);
- quante parole cadono in unità familiari.

Nessuna etichetta: dice quanto di un referto nuovo la memoria ha già incontrato, non se il lettore ha ragione.

| corpus | referti letti | parole lette | coppie note | terne note | parole in unità familiari |
|---|---|---|---|---|---|
| IU X-ray (torace, reali, anonimizzati) | 0 | 0 | 47 % | 15 % | 23 % |
| | 353 | 12.989 | 78 % | 53 % | 55 % |
| | 883 | 32.370 | 83 % | 60 % | 61 % |
| | 1.767 | 65.809 | 87 % | 65 % | 64 % |
| | 3.535 | 135.604 | **90 %** | **71 %** | **67 %** |
| E3C italiano (casi clinici, non referti) | 0 | 0 | 3 % | 0,3 % | 3 % |
| | 234 | 70.982 | 34 % | 8 % | 38 % |

**Lettura.**
- Leggere referti insegna molto e in fretta. Dopo 353 referti di torace la memoria conosce il 78 % delle coppie di un referto nuovo; 5,7 milioni di parole di abstract ne davano il 47 %. I referti si ripetono: il registro conta più della quantità.
- L'italiano parte da zero: la memoria di base è inglese. 234 casi clinici portano al 34 %. Servono referti italiani veri.
- **Limiti.** IU X-ray è un solo ospedale, solo radiografie del torace, molto ripetitivo. Imparare il torace non insegna l'addome. "Conosciuto" non vuol dire "letto bene".
- I conteggi dei referti non migliorano gli span di MedMentions (AUC 0,869 con e senza): sono un altro registro. Per misurare il guadagno sui referti servono span annotati su referti (sotto: RadGraph, o il gold set).

## 3. Referti veri anonimizzati: che cosa si può avere

Verificato il 12 ottobre 2026 sulle pagine dei dataset.

| fonte | che cosa | come si accede | licenza | per Melampo |
|---|---|---|---|---|
| IU X-ray (Open-i, NLM) | 3.927 referti di torace, inglese, reali | libero (già scaricato) | pubblico NLM | sviluppo; un solo ospedale e una sola modalità |
| **MIMIC-IV-Note** | **2.321.355 referti radiologici** di 237.427 pazienti, tutte le modalità; 331.794 lettere di dimissione | credenziali PhysioNet, corso CITI "Data or Specimens Only Research", firma del DUA | PhysioNet Credentialed Health Data License 1.5.0 | la memoria più grande possibile in inglese; gratuito, qualche giorno per le credenziali |
| **RadGraph-XL** | 2.300 referti annotati a mano, span di anatomia e osservazioni e relazioni (modify, located_at, suggestive_of); TC torace, TC addome-pelvi, RM encefalo, RX torace | come sopra (pubblica solo la parte MIMIC) | come sopra | il gold di span e ruoli su referti veri che ci manca per misurare i confini |
| MIMIC-CXR 2.1.0 | 227.835 studi di torace | come sopra | come sopra | sottoinsieme di MIMIC per il torace |
| CT-RATE | TC torace, 25.692 volumi (21.304 pazienti) con referto, inglese tradotto dal turco | gated su Hugging Face | CC BY-NC-SA 4.0, solo ricerca, vietata la redistribuzione | sviluppo; non per il prodotto |
| MR-RATE | RM encefalo e colonna, 98.334 studi con referto (83.425 pazienti) | gated su Hugging Face | CC BY-NC-SA 4.0 | sviluppo; non per il prodotto |
| CheXpert Plus, ReXGradient-160K | torace | registrazione / licenza | da leggere | alternativa a MIMIC-CXR |
| Italiano | nessun corpus pubblico di referti reali (PARROT è fittizio, CC BY-NC-SA) | | | servono referti pseudonimizzati da una struttura |

**Vincolo decisivo per MIMIC e RadGraph.** Il DUA di PhysioNet vieta di condividere i dati con terzi, e PhysioNet precisa che questo vale anche per l'invio a servizi LLM online (API di OpenAI e simili). Il nostro lettore, le collocazioni e le cornici girano in locale o su GitHub Actions senza modelli esterni: si possono usare. **Il braccio LLM di `phrase-probe` (OpenRouter) no.** Inoltre i dati non possono finire in un repository o in un artefatto pubblico: le credenziali sono personali, e i file restano sulla macchina di chi le ha. I conteggi derivati, cioè numeri e non testo, vanno valutati rispetto al DUA prima di qualunque condivisione.

**Che cosa farei.**
1. **Subito, senza iscrizioni.** Workflow `report-memory`: IU X-ray, PARROT inglese e italiano e, con `HF_TOKEN` e i termini accettati, CT-RATE e MR-RATE. Curva di apprendimento su TC e RM, non solo torace.
2. **Credenziali PhysioNet** (Frank o chi del team, gratis, qualche giorno). Con MIMIC-IV-Note la memoria del lettore legge 2,3 milioni di referti di tutte le modalità. Con RadGraph-XL si misurano i confini e i ruoli su referti veri, non su abstract. Il lavoro va fatto in locale, sulla macchina di chi ha le credenziali: preparo lo script che legge i file scaricati e produce solo numeri.
3. **Italiano.** I referti pseudonimizzati della struttura (almeno 300 frasi, meglio migliaia) restano l'unica via. Il meccanismo è lo stesso: la memoria impara il registro leggendo.

## 4. Come si ripete

```
python scripts/chunk_probe.py --medmentions ext/mm --craft ext/craft --ncit ncit.obo [--extra reports_colloc.json.gz]
python scripts/report_memory_probe.py --reports iuxray.jsonl [...] --base base_colloc.json.gz --save reports_colloc.json.gz
python scripts/fetch_public_reports.py hf ibrahimhamamci/CT-RATE --out ct_rate.jsonl   # HF_TOKEN
```
Workflow `report-memory` (manuale; `HF_TOKEN` per i dataset gated). Test: `tests/test_collocations_frames.py` (13).

Fonti: [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note), [RadGraph-XL](https://physionet.org/content/radgraph-xl/1.0.0/), [PhysioNet: uso responsabile con servizi online](https://physionet.org/news/post/gpt-responsible-use), [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), [MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE), [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0).
