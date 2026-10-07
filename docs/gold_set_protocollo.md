# Gold set del linker anatomico: protocollo

Obiettivo: poter scrivere, con una prova, "errore ≤1% sui link accettati, confidenza 95%". Il gold set non lo scrive il software né chi ha scritto il lessico: sono referti veri, etichettati da due radiologi.

## Cosa serve e perché

| Requisito | Motivo |
|---|---|
| Referti reali, pseudonimizzati alla fonte (nomi, date di nascita, identificativi) | Il testo vero ha gli errori e le abitudini che il lessico non ha previsto. Nessun referto grezzo lascia l'ospedale (GDPR). |
| Più radiologi autori e, se possibile, più strutture | Un solo autore rende i campioni correlati. |
| Una menzione per referto | Due menzioni dello stesso referto non sono indipendenti e gonfiano il campione. |
| Due radiologi che non hanno scritto il lessico, schede cieche | Chi vede prima il suggerimento del sistema tende a seguirlo (Dratsch 2023: da ~80% a <20% di accuratezza con un suggerimento sbagliato). |
| Un terzo revisore sui disaccordi | Il rumore delle etichette deve stare molto sotto l'1%. |
| Insieme congelato: nessuno lo guarda durante lo sviluppo | Altrimenti si aggiusta il sistema sul test e la certificazione non vale. |

## Quanti casi

Link accettati necessari per certificare ≤1% al 95% (esatto, Clopper-Pearson): 299 con 0 errori, 628 con ≤2, 1.049 con ≤5, 1.941 con ≤12. Con errore vero dello 0,5% (meglio dell'obiettivo) il test da 299/0 si supera solo nel 22% dei tentativi, quello da 1.049/≤5 nel 57%, quello da 2.000/≤12 nel 79%. **Obiettivo: 2.000–2.300 menzioni** (circa 2.000 accettate). `python scripts/gold_set.py size` ripete il calcolo. Per garanzie per strato (italiano, inglese, con lato) ogni strato richiede il suo numero.

## Passi

1. **Preparare `reports.jsonl`**, una riga per referto: `{"report_id": "...", "text": "<testo pseudonimizzato>", "language": "it|en" (facoltativo), "site": "..."}`.
2. **Campionare**: `python scripts/gold_set.py sample reports.jsonl --out gold_study --n 2300`. Produce `annotator_A.csv`, `annotator_B.csv` (stesso contenuto, ordine diverso), `valid_structures.txt`, `items.jsonl`. Le schede non contengono nulla del sistema. Con il workflow `public-reports` i valori `n` e `cap` si scrivono come numeri nudi (vedi `linker_operativo_github_actions.md`).
3. **Etichettare** (ogni radiologo da solo, senza vedere l'altro né il sistema), vedi le regole sotto.
4. **Controllare le schede**: `python scripts/gold_set.py check annotator_A.csv`.
5. **Accordo e coda**: `python scripts/gold_set.py agree annotator_A.csv annotator_B.csv --out adjudication.csv`. Stampa accordo e kappa e scrive i casi in disaccordo. Il terzo revisore compila `structure` e `relation` in `adjudication.csv`.
6. **Etichette finali**: `python scripts/gold_set.py merge annotator_A.csv annotator_B.csv adjudication.csv --out gold.jsonl`. I casi senza decisione restano fuori e vengono contati.
7. **Valutare**: `python scripts/gold_set.py evaluate gold.jsonl --uberon <uberon-basic.obo> --out gold_report.json`. Il verdetto dice cosa è certificato e a quali condizioni. Per la misura con i due LLM si usa lo stesso gold.jsonl nel workflow di benchmark.
8. **Congelare** `gold.jsonl` e il commit del sistema valutato. Dopo ogni cambio di lessico, tabella, modelli o prompt si ripete su un insieme nuovo o su una parte non ancora vista.

## Regole di etichettatura (per i radiologi)

Per ogni riga si legge la frase e si compila:
- **structure**: l'id della struttura dall'elenco `valid_structures.txt`, oppure
  - `NONE_IN_CLASSES`: struttura anatomica vera ma non fra le classi (utero, appendice, ovaio…);
  - `NOT_ANATOMY`: non è una struttura ("T2" sequenza RM, "LM" tronco comune coronarico, stadio cT3…);
  - `AMBIGUOUS`: il testo non dice di quale struttura si tratta (lato mancante, "iliaco", sigla ambigua).
- **relation**: `equal` (la menzione è la struttura o un suo sinonimo), `part_of` (è una parte: sigma → colon, testa femorale → femore), `contour_of` (è il profilo della struttura su radiografia: profilo cardiaco), `approx` (sinonimo d'uso non strettamente identico: emibacino ≈ osso iliaco).
- **side_in_text**: lato scritto nel testo (dx/sn/bil/none). Il lato si legge dal testo, non si deduce dal resto del referto.
- **note**: facoltativa, solo per dubbi.

Regole:
- Si etichetta la struttura *nominata*, non la patologia né la presenza: "assenza del rene destro" → rene destro (l'assenza è polarità, un attributo a parte).
- Un lume o uno spazio non è l'organo: "loggia renale", "lume esofageo", "ilo epatico" → `NONE_IN_CLASSES` (o `NOT_ANATOMY` se non è una struttura).
- Parete, parenchima, corpo di un organo → l'organo con `part_of`.
- Una struttura nominata solo come modificatore di una misura o di un esame ("frequenza cardiaca", "heart rate", "funzione epatica", "liver function tests", "ormone tiroideo", "thyroid-stimulating hormone") → `NOT_ANATOMY`. (Regola proposta il 7 ottobre dopo la lettura di 600 menzioni di case report e approvata da Frank lo stesso giorno come regola del progetto; i due radiologi la applicano dalla sessione di allineamento e, se la contestano, la si rivede prima di congelare le schede.) Il criterio è il quadro del testo, non la parola vicina: una struttura nominata dentro risultati di laboratorio o segni vitali ("Thyroid, parathyroid, and vitamin D assay were normal") è un esame, non un reperto; la stessa parola nei reperti di un referto di immagini nomina la struttura. "Biopsia epatica", "insufficienza cardiaca", "RM encefalo" nominano la struttura.
- "Anca" / "hip" senza "osso" è la regione dell'anca o l'articolazione coxo-femorale, non l'osso dell'anca (in UBERON *hip* è una regione, UBERON:0001464; la classe TotalSegmentator *hip* è l'osso coxale). Si etichetta la classe più vicina (`hip_left` / `hip_right`) con `relation` = `approx`; "osso dell'anca", "osso iliaco", "os coxae" → `equal`. (Decisione del 7 ottobre, da verificare con i radiologi.) La stessa regola vale per ogni nome che perde la parola-testa: "paravertebrali" senza "muscoli" è una regione, "left innominate" senza "bone" può essere la vena o l'arteria.
- Lato non scritto in una struttura pari → `AMBIGUOUS`, anche se il resto del referto lo lascia intuire.
- Se non si è sicuri, `AMBIGUOUS`: l'astensione del sistema su questi casi è corretta e va misurata come tale.

## Cosa NON è un gold set

I file `heldout_*` e `heldout2_*` in `data/linking/` sono scritti dall'autore del lessico: servono per i test di regressione, non per certificare. I referti di prova nei test del codice sono sintetici e servono solo a provare gli script.

## Sorveglianza dopo l'avvio

Un campione casuale e cieco del 2–5% dei link accettati va rivisto da un radiologo che codifica prima di vedere la scelta del sistema. Alimenta un limite di Clopper-Pearson aggiornato nel tempo e intercetta le derive (nuovo ospedale, nuovo modello).
