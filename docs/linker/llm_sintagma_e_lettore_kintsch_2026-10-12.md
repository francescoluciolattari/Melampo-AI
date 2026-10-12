# Gli LLM leggono il sintagma; test dei punti 1 e 2 (12 ottobre 2026)

Questo documento risponde a quattro domande:

1. Dov'era l'errore nei 9 casi in cui i due modelli concordavano?
2. Perché un modello leggeva la sola parola?
3. Quale modello locale usiamo per lo stadio 6?
4. Come si provano i due modi di istruire (punto 1) o addestrare (punto 2) un LLM con il lettore Kintsch?

## 1. I 9 casi concordi: nessun errore

Le espressioni sono "brain weight", "brain volume", "spleen sterol contents" e "brain plasticity". Ciascuna nomina una misura o un processo della struttura: nel nostro schema il ruolo è `inherent_location`. I due modelli le hanno lette bene.

Comparivano come "falsi allarmi" per un motivo diverso. La metrica del 9 ottobre contava ogni risposta "non è un sito" come un allarme su un link giusto, anche quando l'etichetta del corpus non dice nulla sul ruolo. Il difetto stava nella misura, non nei modelli. È già corretto nel documento del 12 ottobre sulle proposte.

## 2. Perché un modello leggeva la sola parola

Il dato viene dal run "informed" del 10 ottobre, su 331 casi letti da ciascun modello:

| modello | risposte | sola parola | sola parola con un'altra parola subito dopo |
|---|---|---|---|
| nemotron-3-super | 331 | 269 (81%) | 83 |
| gemma-3-27b | 331 | 129 (39%) | 19 |

La differenza tra i modelli non è casuale: dipende dalla domanda che facevamo. Le cause sono tre, tutte nostre.

1. **Un segno con due significati.** Negli esempi annotati del prompt, `[[ ]]` racchiudeva il *concetto intero*, cioè la risposta (per esempio "[[brain tumors]]"). Nella frase da leggere racchiudeva solo la *parola* ("[[brain]] weights"). Gli esempi insegnavano quindi che "quello che sta tra `[[ ]]` è il concetto". Nemotron, il più letterale dei due, ha seguito questa regola quasi sempre.
2. **Esempi sbilanciati.** Erano le frasi annotate di MedMentions più simili che contengono la stessa parola, e lì la parola è spesso annotata da sola ("the [[brain]] RAS"). Il risultato era un insegnamento implicito: la risposta corta è quella giusta.
3. **Nessun controllo della risposta.** La risposta "brain / anatomical structure / body_site: no" si contraddice da sola, eppure veniva accettata. La domanda ("il concetto la cui frase contiene la parola") è soddisfatta anche dalla parola stessa.

### La correzione

Il modulo nuovo è `src/melampo/evaluation/llm_reading.py`. Lo usano `phrase_probe`, `role_probe` e `lora_roles`. La correzione vale per ogni frase e non è legata a casi particolari.

- **Un solo segno, `<w> </w>`, con un solo significato: "la parola di cui chiedo".** Vale anche negli esempi, dove la risposta compare solo dopo la freccia (`... <w>brain</w> tumors ... -> phrase «brain tumors»`). La regola 1 del prompt lo dice esplicitamente: il segno non è mai la risposta e non segna mai i limiti del sintagma.
- **La regola sul sintagma.** Il sintagma va dal primo modificatore al nome testa, insieme ai complementi della testa ("brain weights", "inferior vena cava filter placement"). La sola parola è la risposta solo se nella frase nulla la continua.
- **Esempi bilanciati.** Tra gli esempi ce n'è almeno uno con un sintagma più lungo della parola e almeno uno uguale alla parola, quando entrambi esistono.
- **Una risposta strutturata.** Il modello restituisce il sintagma copiato dalla frase, la testa, il tipo e il ruolo della parola nel nostro schema, oppure `uncertain`. "Sito sì/no" non viene più chiesto a parte: si ricava dal ruolo, quindi non può contraddirlo.
- **Un controllo deterministico.** Il sintagma deve essere un pezzo contiguo della frase che copre la parola segnata, la testa deve essere una parola del sintagma e il ruolo deve appartenere allo schema.
- **Una sola domanda in più, esplicita.** Scatta se la risposta si ferma sulla parola mentre la frase prosegue senza punteggiatura né parole funzionali ("brain → weights"). Il modello riceve: "«brain» è seguito da «weights». Se queste parole appartengono allo stesso sintagma dà il sintagma intero, altrimenti ripete la risposta". Dopo questa domanda la risposta resta quella data. Una risposta malformata riceve una sola richiesta di correzione.
- **Tutto viene contato.** Il report riporta, per ogni modello, le risposte non valide, quelle date con la sola parola, le domande ripetute, le risposte cambiate dalla domanda e le risposte incerte.

## 3. Lo stadio 6: modelli a pesi aperti, eseguibili in locale

Ho verificato le opzioni ad oggi (fonti in fondo). Non esiste un successore di gpt-oss: i più recenti tra i modelli aperti sono DeepSeek V4, GLM-5.x, Kimi K2.6/K3 e Mistral Large 3. Sono molto più grandi, perché servono da 4 a 8 GPU H200, e per un server locale d'ospedale sono sproporzionati. La scelta è quindi questa:

| uso | modello | licenza | hardware locale |
|---|---|---|---|
| stadio 6, modello A | `openai/gpt-oss-120b` (117 miliardi di parametri, 5,1 attivi; HF sha `b5c939de`), **in FP8** (decisione di Frank, 12 ottobre) | Apache 2.0 | FP8: circa 120 GB di pesi, quindi 2 GPU da 80 GB o 1 H200 da 141 GB |
| stadio 6, modello B (famiglia diversa, così i due errori sono indipendenti) | Mistral Small 4, `mistralai/Mistral-Small-4-119B-2603` (119 miliardi, 6,5 attivi) | Apache 2.0 | 1 GPU da 80 GB con la versione NVFP4 o 4 bit |
| punto 2, base per LoRA | `openai/gpt-oss-20b` (stessa famiglia del modello A; OpenAI pubblica la ricetta LoRA con TRL) | Apache 2.0 | 1 GPU da 40–80 GB |
| alternativa densa | `Qwen/Qwen3.8-27B` (agosto 2026) | Apache 2.0 | 1 GPU da 48–80 GB; su HF è un modello immagine+testo |

Regole d'uso:

- **gpt-oss-120b in FP8.** I pesi ufficiali sono MXFP4 (4 bit), quindi FP8 è una conversione. Su OpenRouter la precisione è un filtro rigido, senza ripiego su altre: `provider.quantizations = ["fp8"]` con `allow_fallbacks = false`. L'opzione è `--quantization openai/gpt-oss-120b=fp8`, predefinita in `role_probe`, in `phrase_probe` e nei due workflow. Prima del run c'è una domanda di prova: se nessun fornitore serve FP8, il run si ferma con un messaggio chiaro invece di trasformare ogni risposta in una risposta non valida. Il nome del modello nel report porta la precisione (`openai/gpt-oss-120b@fp8`). In locale serve un checkpoint FP8 fissato allo sha: su Hugging Face esistono solo conversioni di terzi (per esempio quelle AMD a pesi MXFP4 e attivazioni FP8). Prima di adottarne una va scelta e verificata, perché non sono di OpenAI.

- **I testi pubblici** (E3C, PARROT, IU X-ray, CRAFT, MedMentions) possono passare da OpenRouter con gli stessi pesi, per misurare.
- **I referti dei pazienti** non escono mai: vanno a un server locale compatibile con OpenAI (vLLM). `role_probe.py` rifiuta qualunque indirizzo non locale se non c'è `--public-data`.
- **Temperatura 0 e risposte in cache** per (modello, domanda). Una nuova esecuzione non chiede nulla due volte.
- **Riproducibilità.** Su OpenRouter il fornitore può cambiare la quantizzazione. La riproducibilità vera si ha in locale, con i pesi fissati allo sha.

## 4. Punto 1: istruire l'LLM con ciò che il lettore ha costruito

Lo script è `scripts/role_probe.py` e il workflow è `role-probe`. Confronta quattro bracci sugli stessi item del gold set silver (663 item con un ruolo):

- **`lattice`**: il reticolo dei blocchi da solo.
- **`reader`**: il lettore Kintsch da solo. Costruisce sei letture, le integra e dà `underspecified` se il margine è sotto 0,10.
- **`llm`**: ciascun modello con la domanda corretta, senza aiuti.
- **`llm+reader`**: la stessa domanda più ciò che il lettore ha costruito, cioè il blocco e il perché del reticolo, le sei letture con la loro quota dopo l'integrazione e i nodi di evidenza. Il prompt avverte che il lettore può sbagliare e che il modello lo deve controllare sulla frase.

Le misure sono:

- accuratezza del ruolo sugli item a cui il braccio risponde;
- quota di risposte incerte;
- errori critici (sito contro non-sito);
- "ha detto struttura a torto";
- intervallo bootstrap per referto;
- risultati per strato.

I ruoli si confrontano sugli item anatomici. Le menzioni che il gold set chiama NOT_ANATOMY (lesioni, fluidi, cellule) sono una convenzione nostra e si contano a parte.

### La base misurata oggi (lettore e reticolo, eseguiti qui)

Su 559 item anatomici:

| braccio | accuratezza del ruolo | incerti | ha detto "struttura" a torto | errore per referto [IC 95%] |
|---|---|---|---|---|
| lattice | 79,1% | 0% | 89 | 20,9% [15,8–26,3] |
| reader | 82,1% | 0,2% | 88 | 17,9% [12,8–22,8] |

Per strato (risposte giuste del reader sul totale):

| strato | giuste |
|---|---|
| PARROT IT | 113/115 |
| IU X-ray | 72/78 |
| E3C EN | 90/106 |
| E3C IT | 183/259 |

Quasi tutto l'errore sta in un punto preciso: le sedi di procedura in italiano. 83 dei 92 `procedure_site` sono letti come struttura, e 69 di questi sono di E3C IT ("intervento chirurgico di…", "biopsia del…"). Il motivo è che le cornici e le definizioni NCIt sono in inglese. Nessun braccio dice "incerto" su questi casi.

Delle 104 menzioni NOT_ANATOMY, il lettore ne legge 87 come struttura. È l'atteso: "blood", "CSF" e "left ovarian cyst" non sono ruoli, sono tipi di entità.

È qui che l'LLM può aiutare, e qui va misurato. Le risposte dei bracci `llm` e `llm+reader` si ottengono con il workflow, perché dalla sandbox OpenRouter non è raggiungibile.

## 5. Punto 2: addestrare (LoRA) un modello aperto sui ruoli

Lo script è `scripts/lora_roles.py`, con tre comandi:

- **`prepare`**: trasforma gli item nella domanda del solo ruolo, con o senza le evidenze del lettore, e la risposta `{"role": ...}`. Divide in 5 fold raggruppati per referto: un referto non è mai sia in addestramento sia in test. È già eseguito qui: 663 item, con 13–21 sedi di procedura per fold.
- **`train`**: QLoRA su un fold, cioè la base a 4 bit con adattatori r=16 su tutte le proiezioni lineari, 2 epoche, seme fisso. Si addestra sui quattro fold rimanenti.
- **`evaluate`**: lo stesso modello con l'adattatore e senza (zero-shot) sul fold tenuto da parte. Usa le stesse misure di `role_probe`.

Si chiede il solo ruolo perché il gold set dà il ruolo di ogni item ma non sempre il sintagma intero. Insegnare un sintagma che non abbiamo vorrebbe dire insegnare di nuovo la parola nuda.

**Limiti, detti prima.**

- `train` ed `evaluate` richiedono una GPU (1 da 40–80 GB, circa un'ora per fold con gpt-oss-20b). I runner gratuiti di GitHub non ce l'hanno.
- Il gold set ha un solo etichettatore (Claude) e il 70% degli item è "struttura". Un modello addestrato qui impara *queste* etichette e queste convenzioni.
- Il test dice se il ruolo si impara da poche centinaia di esempi e se le evidenze del lettore aiutano. Non certifica nulla.
- Per certificare servono il gold set dei radiologi e, con le credenziali PhysioNet, RadGraph-XL (2.300 referti, da addestrare solo sulla macchina della persona accreditata).

## Cosa lanciare

1. **Phrase probe con la domanda corretta e gli stessi due modelli**, per isolare l'effetto della correzione. Workflow `phrase-probe` con `llm = sample`, `llm_mode = informed` e `llm_models` vuoto. Nel report la riga "LLM answers per model" mostra quante risposte con la sola parola restano.
2. **Role probe**, workflow `role-probe` con i valori predefiniti (gpt-oss-120b e Mistral Small 4, tutti i bracci, 0 esempi). Poi una seconda esecuzione con `examples = 4`.
3. **LoRA**: su una GPU (un server dell'ospedale, una GPU a noleggio o HF Jobs), `prepare`, poi `train` ed `evaluate` per i fold da 0 a 4.

## Fonti

- [Spheron: GPT-OSS 120B vs GLM-5.1 vs DeepSeek V4](https://www.spheron.network/blog/open-weight-frontier-model-showdown-2026/)
- [Kingy AI: best open-weight models 2026](https://kingy.ai/news/best-open-weight-ai-models-in-2026-glm-5-2-vs-deepseek-v4-vs-kimi-k2-6-vs-qwen-vs-mistral/)
- [OpenAI: introducing gpt-oss](https://openai.com/blog/introducing-gpt-oss)
- [OpenRouter: openai/gpt-oss-120b](https://openrouter.ai/openai/gpt-oss-120b)
- [OpenRouter: Mistral Small 4](https://openrouter.ai/mistralai/mistral-small-2603)
- [OpenRouter: Qwen3.8 27B](https://openrouter.ai/qwen/qwen3.8-27b)
- Hugging Face API: `openai/gpt-oss-120b`, `Qwen/Qwen3.8-27B`, `mistralai/Mistral-Small-4-119B-2603`
