# Role probe, primo run (13 ottobre 2026)

## Cosa è successo

Il run (4 modelli: gpt-oss-120b in FP8, Gemma 4 31B, Mistral Small 4, Nemotron 3 Ultra) è **incompleto**. Dall'item 442 in poi (tutti gli item di IU X-ray, PARROT e Rad-SpRL, 521 su 963) ogni chiamata a ogni modello ha risposto **HTTP 403**. Non è un errore dei modelli: è un limite della chiave o del credito OpenRouter (da verificare). Il programma contava quei 403 come `uncertain`, e questo gonfiava la tabella: incertezza del 60% e accuratezza dei modelli calcolata solo su E3C, mentre il reticolo e il lettore erano calcolati su cinque strati. La tabella del report non è confrontabile così com'è.

**Correzioni nel codice.**

- Una chiamata rifiutata è `unreachable`: non è una risposta e non è `uncertain`. Non entra nei punteggi.
- Il report dà anche i punteggi sugli **stessi item** per tutti i bracci (`common`).
- Dopo 12 item di fila rifiutati da tutti i modelli il run si ferma, con un messaggio, e le risposte restano nella cache.
- Nuovo input `resume_run`: l'id di un run precedente. Le sue risposte (artefatto `role-probe`) sono riusate, senza rifare le chiamate né pagarle di nuovo.

## Risultati validi: 396 item che tutti i modelli hanno letto (E3C), 315 anatomici

Item anatomici, solo E3C (inglese e italiano). Set di sviluppo, non di certificazione.

| braccio | incerti | accuratezza sulle risposte | errori | "struttura" detta a torto |
|---|---|---|---|---|
| reticolo | 0% | 72,1% | 88 | 75 |
| lettore Kintsch | 0% | 73,3% | 84 | 75 |
| Nemotron 3 Ultra | 1% | **91,7%** | 26 | 4 |
| Gemma 4 31B | 4% | 89,1% | 33 | 4 |
| gpt-oss-120b FP8 | 0% | 88,5% | 36 | 4 |
| Mistral Small 4 | 6% | 80,1% | 59 | 20 |
| Nemotron + lettore | 1% | 92,0% | 25 | 7 |
| Gemma + lettore | 4% | 89,1% | 33 | 4 |
| gpt-oss + lettore | 0% | 87,9% | 38 | 13 |
| **lettore e Gemma d'accordo** | 35% | **97,1%** | 6 | 4 |
| lettore e Nemotron d'accordo | 28% | 95,2% | 11 | 6 |
| lettore e gpt-oss d'accordo | 27% | 93,9% | 14 | 12 |

**Lettura.**

1. **Il modello che legge il sintagma batte di molto il lettore** (88–92% contro 73%) sugli item di E3C. Dove il lettore sbagliava di più, cioè le **sedi di procedura** (78 item): il lettore ne legge 6 giuste su 92, Nemotron e Gemma 80 su 92, gpt-oss 78. Questo era il punto aperto delle 88 sedi di procedura italiane.
2. **Dare l'uscita del lettore al modello non aiuta**: la precisione resta uguale o peggiora (gpt-oss: "struttura" detta a torto da 4 a 13). Il modello si fida di una lettura "struttura" sbagliata del lettore.
3. **Quando lettore e modello concordano** l'errore scende al 3–6%, ma il 27–35% degli item va a una persona. È la strada coerente con ≥99% di precisione sui link accettati, e il costo è il lavoro della persona.
4. **Mistral Small 4** è chiaramente peggiore (80%, 20 "struttura" a torto). Non è adatto come lettore principale.
5. **Gemma 4 31B** (una GPU, Apache 2.0) sta quasi al livello di Nemotron 3 Ultra (8 GPU) e davanti a gpt-oss-120b. Come base per il punto 2 (LoRA) è la candidata.

**Menzioni NOT_ANATOMY** (lesioni, liquidi, cellule; 81 nei 396 item). Il lettore e il reticolo le leggono come struttura 67–68 volte su 81. I modelli 12–33 volte: Gemma 12, gpt-oss 18, Nemotron 20, Mistral 18. Con il lettore nel prompt peggiora (Gemma 17, Nemotron 33, gpt-oss 26).

## Limiti

- Solo E3C (396 item): niente IU X-ray, PARROT, Rad-SpRL per i modelli.
- Il gold set silver è di sviluppo.
- Il run durava circa 11.000 secondi per 442 item, con 4 modelli e 4 lavoratori in parallelo.

## Da fare

Ripetere il run con `resume_run` = id di questo run, dopo aver controllato credito e limite della chiave OpenRouter. Le 396 risposte E3C sono già pagate.
