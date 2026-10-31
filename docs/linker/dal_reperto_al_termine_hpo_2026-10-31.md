# Dal reperto del referto al termine HPO: candidati, metodo deterministico, scelta del modello (31 ottobre 2026)

## Il problema

La lettura dei referti dà, per ogni reperto, la **sede** (classe TotalSegmentator, ancorata a UBERON), i
qualificatori e la frase. Il grafo diagnostico vuole un **termine HPO**. Per esempio "Nodulo solido di 12 mm a
margini spiculati nel lobo superiore destro" deve diventare *Solid pulmonary nodule*. Il "che cosa" e la sede vanno
composti: il referto scrive "nodulo nel lobo superiore destro", non "nodulo polmonare solido".

## Che cosa dicono le ontologie (verificato sui file)

- **HPO 2026-09-01.** Solo 4.170 termini su 19.894 hanno una definizione logica (`intersection_of`), e nessuno
  dei 34 termini "nodule" ne ha una (cisti: 10 su 189). Le definizioni logiche da sole **non bastano**: la proposta
  del 30 ottobre è corretta qui.
- **HPO è molto preciso sui reperti d'imaging**: *Pulmonary nodule* ha come figli *Solid*, *Subsolid*,
  *Part-solid*, *Pure ground-glass*, *Calcified*, *Cavitating*, *Perifissural pulmonary nodule*. I qualificatori
  che leggiamo già servono a scegliere il figlio giusto.
- **HPO in italiano** (tabella babelon del progetto di traduzione): 3.488 etichette (17,5%), di cui 524 OFFICIAL e
  2.964 CANDIDATE, molte tradotte da DeepL. C'è "Nodulo polmonare" (ufficiale), non "Nodulo polmonare solido". Su
  questo campione, dei 107 termini etichettati 41 hanno un'etichetta italiana (10 ufficiali) e 66 no.
- **RadLex in italiano**: nessuna versione pubblica. L'unico lavoro trovato (Padova 2011, OpenEye) è un lessico
  interno, non distribuito.

## Il generatore dei candidati (`melampo.memory.hpo_candidates`)

**Classi della sede.** Sono le ancore UBERON della classe TotalSegmentator. Se l'ancora non è un organo, si sale
(`part_of`/`is_a`, in ampiezza, al massimo 3 passi) fino al primo livello che contiene un organo (sottoinsiemi
UBERON `organ_slim`/`major_organ`) e si tengono tutte le classi incontrate. Esempio: "lumbar vertebra 4" diventa
"lumbar vertebra", "vertebra" e così via. Una classe TotalSegmentator senza ancora prende quella del prefisso
(kidney_cyst_right → kidney).

**Tre vie.** Nessuna basta da sola:
1. la definizione logica nomina una classe della sede;
2. il nome o un sinonimo (esatto, laico o correlato) contiene un nome della sede: etichetta, sinonimo esatto o
   **aggettivo relazionale** di UBERON (lung → "pulmonary"), quindi *Pulmonary nodule*;
3. i discendenti dei termini trovati con 1 e 2.

**Shortlist.** I 15 candidati che condividono più parole con la frase (radici leggere; prima il numero di parole
condivise, poi la quota), più i termini generali dell'organo ("Abnormal adrenal morphology") e il più generale fra
quelli trovati per nome o per definizione.

**Metodo A, deterministico.** Sceglie il candidato più specifico il cui nome è fatto tutto di parole della frase e
di nomi della sede, con almeno una parola che non sia un nome della sede. Due termini di questo tipo, nessuno dei
due sotto l'altro, danno un'astensione. Per una frase italiana usa le etichette italiane di HPO.

## Prima misura, senza modelli (150 reperti inglesi + 50 italiani)

| Misura (inglese, 107 reperti con termine) | Valore |
|---|---|
| Etichetta tra i candidati della sede | **99/107 (93%)**; candidati per sede: mediana 284 |
| Etichetta nella shortlist (15 + generali) | 75/107 (70%); con 20 termini 79; con il nome giusto aggiunto alla frase 99 |
| A: risposte | 49/150; giuste o più generali 29 (59%); **troppo specifiche 6**; sbagliate 11; termine dove serviva NONE 3 |
| Riferimento "primo della shortlist" | 41 esatti su 150; 42 termini dove serviva NONE |
| A in italiano | 2 risposte su 50 (entrambe esatte): le etichette italiane coprono troppo poco |

**Come leggerla.**
- Il generatore di candidati funziona: 93% di copertura, e gli 8 termini mancanti sono casi limite. Due esempi:
  l'emorragia gastrointestinale vista dallo stomaco, la stenosi bronchiale vista dal lobo.
- Il metodo deterministico **non basta**: risponde a un terzo dei reperti e, quando risponde, è giusto solo nel
  59% dei casi.
- **Il collo di bottiglia è la shortlist (70%)**: i termini mancati non hanno parole in comune con la frase
  ("enlarged" e *Hepatomegaly*, "atrophy of the brain" e *Cerebral atrophy*). Per questo nel metodo B il modello
  prima **nomina** il reperto nello stile dell'ontologia, e la shortlist si costruisce dalla frase più quel nome.
  Con il nome giusto la copertura sale a 99/107, che è il limite superiore. Il nome è solo una chiave di ricerca:
  la risposta è sempre un termine reale dei candidati della sede.

## Il metodo B, la prova (`scripts/hpo_probe.py`, workflow `hpo-probe`)

1. **Frase italiana**: il modello la traduce in inglese.
2. **Nome**: il modello nomina l'anomalia nello stile HPO ("none" se la frase non ne afferma una).
3. **Scelta**: il modello sceglie fra la shortlist costruita dalla frase e dal nome, leggendo la frase originale.
   Le opzioni sono lettere, più "none of these".

Il braccio **C** accetta solo quando A e B concordano; altrimenti l'item va a una persona.

**Modelli.** Gemma 4 31B (`google/gemma-4-31b-it`) e Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b`),
come per il linker. Altri id si passano dall'input `models` del workflow.

**Criteri fissati prima della corsa**, sui reperti inglesi:
- B vale la pena se, fra le risposte, giuste o più generali sono **almeno l'80%**, le **troppo specifiche al
  massimo il 5%**, e se batte sia A sia "primo della shortlist" negli esatti sul totale;
- C deve avere giuste o più generali **almeno il 95%** delle risposte accettate;
- in italiano B deve stare entro 10 punti da B sugli stessi referti in inglese.

Se questi criteri non tengono, nessun reperto entra nel grafo come termine HPO. I reperti restano nella lettura
del referto, come oggi.

## Primo run con i modelli (31 ottobre, mattina)

Corsa `hpo-probe` su 200 item (150 inglesi, 50 italiani), Gemma 4 31B e Nemotron 3 Ultra, etichette della prima
versione.

| Inglese, 150 item | Risposte | Esatti | Più generali | **Troppo specifici** | Sbagliati | Termine dove serviva NONE | Sicuri sulle risposte |
|---|---|---|---|---|---|---|---|
| A (deterministico) | 49 | 24 | 5 | 6 | 11 | 3 | 59% |
| Primo della shortlist | 148 | 41 | 13 | 14 | 38 | 42 | 37% |
| **B Gemma 4 31B** | 149 | **86** | 11 | 10 | 22 | 13 | 65% |
| **B Nemotron 3 Ultra** | 150 | **82** | 10 | 10 | 22 | 20 | 61% |
| C (A e Gemma d'accordo) | 28 | 22 | 1 | 1 | 4 | 0 | 82% |
| Gemma e Nemotron d'accordo | 121 | 73 | 9 | 9 | 15 | 11 | 68% |

In italiano, su 50 item, B è esatto in 24 casi (Gemma) e 27 (Nemotron), sicuro nel 56% delle risposte; A risponde
2 volte.

**Esito rispetto ai criteri fissati prima: B non passa.** È molto meglio di A e del riferimento (57% di esatti sul
totale contro 16% e 27%), ma è sicuro solo nel 61–65% delle risposte (soglia 80%) e troppo specifico nel 6,7%
(soglia 5%). C è sicuro all'82% (soglia 95%). L'italiano sta entro 10 punti dall'inglese: quel criterio passa.
Nessun reperto entra nel grafo come termine HPO.

**Da dove vengono gli errori.**

1. **Il termine giusto non è nella shortlist** (31 item su 107). Lì i modelli non rispondono "nessuno": scelgono un
   termine generale (più generale: sicuro) o uno vicino ma sbagliato (18 su 31). Il nome proposto dal modello ha
   portato pochissimo (shortlist da 75 a 76–77), perché la ricerca è per parole: "Adrenal nodule" non incontra
   *Adrenal mass*, "Renal mass" non incontra *Renal neoplasm*.
2. **Un termine dove non c'è un reperto** (13–20 item): tabelle di misure dell'ecocardiogramma, un quesito
   clinico, frasi spezzate.
3. **Frasi spezzate dalla lettura: un difetto vero, corretto.** Il punto decimale veniva letto come fine della
   frase ("Cyst measuring 1.1 cm in the middle lobe" → "1 cm in the middle lobe"). Riguardava 27 item del
   campione e, fuori dalla prova, la portata della negazione e i reperti di tutti i referti. Ora
   (`assertion._clause_bounds`) un punto fra due cifre non chiude la frase. Con la correzione il campione è
   cambiato: 27 coppie hanno una frase o una struttura diversa e sono state etichettate di nuovo.
4. **Errori miei nelle etichette.**
   - Tre sono chiari: HPO ha il termine esatto e io non l'avevo trovato (*Gallbladder biliary sludging*,
     *Herniation of intervertebral nuclei*, *Short femoral neck*). Li ho corretti dopo aver visto le risposte dei
     modelli, quindi la correzione pende verso di loro: è segnata riga per riga in `labels.txt`.
   - Altri casi sono discutibili e li ho lasciati come erano. Esempi: *Complex renal cyst* per una cisti con un
     setto; *Abnormal renal artery morphology* per un'arteria renale che nasce dal falso lume; *Vertebral
     compression fracture* contro *Compression-fractured thoracic vertebra*, due termini paralleli di HPO.
   - Con le tre correzioni, Gemma è sicuro nel 69% delle risposte, con il 4% di troppo specifici.

**Che cosa è cambiato per il secondo run.**
- Frasi: il punto decimale (sopra).
- Radici: la "e" finale e "-ed" ("infiltrate" e *Pulmonary infiltrates*, "fracture" e *fractured*). Sul campione
  ricostruito il termine giusto è fra i candidati in 106 casi su 114 e nella shortlist in 84 su 114.
- Prompt: misure, normalità, tecnica e quesiti sono indicati esplicitamente come "nessuno". Se il modello ha già
  detto che non c'è un'anomalia, la sua scelta è "nessuno".
- Nuovo braccio `models_agree`: Gemma e Nemotron scelgono lo stesso termine, altrimenti il caso va a una persona.

**Il limite che resta, e la proposta.** La ricerca per parole non supera il 74% di presenza nella shortlist. Il
passo successivo che propongo è una **discesa nell'albero**:
1. il modello parte dal termine generale dell'organo ("Abnormal adrenal morphology");
2. sceglie fra i suoi figli diretti, poi fra i figli del figlio scelto;
3. si ferma dove la frase non dice di più.

Ogni passo è una scelta vincolata fra pochi termini reali. Non serve la sovrapposizione di parole: *Adrenal mass*
è un figlio di *Abnormal adrenal morphology*. Lo propongo dopo il secondo run, che misura quanto valgono le
correzioni di oggi da sole.

## Secondo run (31 ottobre, dopo le correzioni del primo)

Corsa dal commit 88a42c1: stessi 200 item, etichette rifatte sul campione corretto, stessi modelli.

| Inglese, 150 item | Risposte | Esatti | Più generali | **Troppo specifici** | Sbagliati | Termine dove serviva NONE | NONE dove serviva un termine | Sicuri sulle risposte |
|---|---|---|---|---|---|---|---|---|
| A (deterministico) | 57 | 32 | 4 | 5 | 13 | 3 | 0 | 63% |
| Primo della shortlist | 149 | 47 | 15 | 16 | 35 | 36 | 0 | 42% |
| **B Gemma 4 31B** | 150 | **98** | 13 | 9 | 19 | 4 | 7 | **74%** |
| **B Nemotron 3 Ultra** | 150 | **94** | 10 | 10 | 21 | 11 | 4 | 69% |
| C (A e Gemma d'accordo) | 34 | 29 | 0 | 2 | 3 | 0 | 0 | 85% |
| Gemma e Nemotron d'accordo | 118 | 86 | 7 | 6 | 13 | 2 | 4 | 79% |

In italiano, su 50 item, Gemma è esatto in 25 casi e Nemotron in 28; le risposte sicure sono il 58% e il 60%. Se
Gemma e Nemotron devono essere d'accordo, le risposte sicure restano il 59%.

**Esito rispetto ai criteri fissati prima: B non passa, ma migliora.**
- **Sicurezza:** Gemma passa dal 65% al 74% di risposte sicure (soglia 80%).
- **Specificità:** i troppo specifici sono il 6% (soglia 5%).
- **C:** sicuro all'85% (soglia 95%).
- **Italiano:** ora è 16 punti sotto l'inglese (massimo 10), quindi anche questo criterio, che prima passava, non
  passa più.

Nessun reperto entra nel grafo come termine HPO.

**Dove sta il problema: diviso per la presenza del termine giusto nella shortlist (Gemma).**

| | Item | Sicuri | Troppo specifici |
|---|---|---|---|
| Inglese, il termine giusto è nella shortlist | 87 | **82%** | 9% |
| Inglese, il termine giusto non c'è | 27 | **30%** | 4% |
| Inglese, nessun termine atteso | 36 | 89% | — |
| Italiano, il termine giusto è nella shortlist | 31 | 71% | 6% |
| Italiano, il termine giusto non c'è | 13 | 15% | — |

1. **Quando il termine giusto è fra le opzioni, la scelta del modello quasi supera la soglia** (82%, e 87% quando i
   due modelli sono d'accordo). Il limite è la **shortlist**, che contiene il termine giusto solo nel 76% dei casi.
   Fuori dalla shortlist i modelli non dicono "nessuno": prendono il vicino.
   - "Cerebral hemisphere mass lesion" diventa *Abnormal temporal lobe morphology*.
   - "Renal neoplasm" diventa *Abnormal renal morphology*.
   - "Bladder trabeculation" diventa *Urinary bladder wall hypertrophy*.
   - "Multiple rib fractures" diventa *Fractured rib*.
2. **I troppo specifici sono quasi tutti un'interpretazione al posto del reperto**: *Abdominal aortic aneurysm* per
   un'ectasia di 3,2 cm, *Unilateral renal hypoplasia* per "rene sinistro più piccolo", *Prostatitis*, *Hepatic
   hemangioma*. In alcuni casi è discutibile se a sbagliare sia il modello o l'etichetta (*Decreased renal
   parenchymal thickness* per "parenchima assottigliato" è forse più giusto della mia *Abnormal renal morphology*).
   Non ho cambiato le etichette dopo aver visto le risposte, perché le farebbe pendere verso i modelli.
3. **Due termini HPO paralleli contano come errore**: *Vertebral compression fracture* e *Compression-fractured
   thoracic vertebra*. È un difetto della misura, da decidere prima del prossimo run e non dopo.
4. **L'italiano perde soprattutto nella shortlist.** A parità di presenza del termine giusto, l'italiano è vicino
   all'inglese: Nemotron è sicuro all'81% quando il termine c'è.

**Il campione è ormai consumato.** Lo abbiamo guardato due volte e il secondo run ha corretto ciò che il primo aveva
mostrato. Un terzo giro di correzioni misurato su questi 200 item direbbe soprattutto quanto ci siamo adattati a
loro. La prossima modifica va misurata su **un campione nuovo**: un altro seme, strutture non viste, etichette
scritte prima di qualunque risposta dei modelli. Meglio ancora se un secondo etichettatore che non ha visto le
risposte rivede almeno i casi dubbi.

**Proposta per il terzo passo**, da fissare prima del run:
- **Discesa nell'albero** dell'organo, accanto alla shortlist per parole: il modello parte dal termine generale e
  scende fra i figli. Attacca il 24% di casi in cui oggi il termine giusto non è fra le opzioni.
- **Regola "detto, non dedotto"**: il termine deve nominare ciò che il referto descrive, non la diagnosi che se ne
  potrebbe ricavare. È una regola generale, non legata ai casi di questo campione.
- **Equivalenza dichiarata prima**: due termini con la stessa definizione logica, o che si implicano a vicenda, si
  contano come lo stesso termine.
- **Criteri invariati**: 80% di risposte sicure, 5% di troppo specifici, 95% per C, italiano entro 10 punti.

## Ripetizione del secondo run (31 ottobre, pomeriggio)

Frank ha rilanciato il workflow da 5842e09, che cambia solo la nota: stesso codice, stessi item. È una misura della
variabilità dei modelli.

| Inglese | Secondo run | Ripetizione |
|---|---|---|
| B Gemma, sicuri sulle risposte | 74,0% | 74,7% |
| B Nemotron | 69,3% | 68,0% |
| Gemma e Nemotron d'accordo | 78,8% | 77,0% |
| C Gemma | 85,3% | 87,9% |

- **Le risposte sono stabili:** fra i due run cambiano 6 risposte di Gemma su 200 e 5 di Nemotron. Il margine dalle
  soglie non è rumore.
- **Ripetere la domanda non aiuta:** sui 115 item inglesi in cui le quattro risposte coincidono (due modelli per due
  run), le risposte sicure sono il 79%.
- **Gli errori dei modelli sono correlati, come il linker aveva già mostrato:** in 33 item le quattro risposte
  coincidono e sono sbagliate. In 17 di questi il termine giusto non era fra le opzioni. Un secondo modello, o un
  secondo tentativo, non è una seconda prova. Serve un meccanismo di natura diversa.

## Il terzo passo: come legge il radiologo, con la precisione della macchina (fissato prima del run)

Richiesta di Frank: il passo successivo va costruito con lo stesso metodo del linker anatomico
(`architettura_parallela_predittiva_linker_2026-10-06.md`). Flussi paralleli di natura diversa che simulano la
lettura umana, una scelta vincolata in cui il modello è preciso, una convergenza verificabile, l'astensione verso
una persona, e la persona da cui il sistema impara.

| Passo della lettura umana | Che cosa fa il sistema | Dove |
|---|---|---|
| Costruzione esaustiva (Kintsch): tutte le letture possibili, anche sbagliate | I candidati della sede per tre vie (definizione logica, nome, discendenti) | `Phenotypes.candidates` |
| Riconoscimento dalla memoria, per parole | La shortlist; il metodo A | `shortlist`, `deterministic` |
| **Dal genere alla specie**: il radiologo sa che è "un'anomalia del polmone", poi "un reperto di imaging", poi "un nodulo", poi "solido" | **Discesa nell'albero:** dalle cime dei rami della sede (*Abnormal lung morphology*, *Abnormal pulmonary thoracic imaging finding*…), una scelta per passo fra i figli che sono candidati alla sede, fino a "fermati qui" | `descent_roots`, `children_at_site`, `descend` |
| Scelta fra alternative note (Sistema 2) | I modelli scelgono lettere fra termini veri, mai codici | `choice_prompt`, `descent_prompt` |
| **"Detto, non dedotto"**: il radiologo non scrive "aneurisma" per un'ectasia di 3,2 cm | **Verifica deterministica:** un termine è affermato solo se la frase contiene tutte le parole distintive di un suo nome, oppure il nome della sua qualità PATO ("enlarged" per *increased size*). Altrimenti si risale al più vicino antenato affermato: il ripiego al padre del linker | `support`, `stated`; PATO da `pato.obo` |
| Convergenza di meccanismi diversi; i due LLM contano **uno** | **Braccio I:** le scelte dei modelli (shortlist e discesa, entrambi i modelli) danno il termine comune più specifico. La frase decide fin dove salire | `integrate` |
| Conflitto → deliberazione o domanda | "Nessuno" contro un termine, nessun termine comune alla sede, nulla di affermato → **a una persona** | `integrate`, `--review` |
| **Apprendimento**: l'esperto impara dai casi rivisti | La coda per il radiologo (`hpo_review.jsonl`): frase, sede, termini proposti, motivo. La scelta del radiologo diventa una frase etichettata, cioè la memoria che la lettura deterministica consulterà per prima (come i nomi curati del linker). Nessuna regola nasce dal caso singolo e nulla entra nella memoria prima della revisione | coda pronta; la memoria si costruisce con le prime revisioni |

**Che cosa promette la verifica, misurata sul campione di sviluppo.** Questa misura è *di sviluppo*: l'idea è nata
guardando questi item, quindi i numeri pendono a favore.
- Applicata alle risposte già date nel secondo run, la verifica porta i troppo specifici di Gemma dal 6,0% al 2,1%.
  Le risposte sicure salgono dal 74% al 79%, ma gli esatti scendono da 98 a 88: una parte delle risposte giuste
  usava parole che la frase non contiene.
- Il braccio integrato, senza discesa, arriva all'82% di risposte sicure, con il 2,3% di troppo specifici, e
  risponde in 129 item su 150.
- Senza modelli, la discesa può raggiungere il termine giusto in 106 item su 114, cioè sempre quando il termine è
  fra i candidati. Partendo solo dal termine generale dell'organo erano 80: i noduli polmonari stanno sotto *Abnormal
  pulmonary thoracic imaging finding*, non sotto *Abnormal lung morphology*.

**Il campione nuovo** (`data/gold/hpo_heldout_2026-10-31`):
- seme 2026103102, nessun referto del campione di sviluppo;
- 184 item (142 inglesi, 42 italiani), 47 strutture, 48 `NONE`;
- etichette scritte **prima di qualunque metodo**, cercando in tutta HPO;
- 28 etichette segnate "?" per un radiologo.

Il generatore, senza modelli, trova il termine giusto fra i candidati in 92 casi su 94 e nella shortlist in 67 su
94. La discesa può raggiungerlo in 92 su 94.

Nel costruirlo è emerso un difetto anche del campione di sviluppo: l'appaiamento fra italiano e inglese è per
struttura, e a volte la frase italiana scelta non è la traduzione di quella inglese.
- Nel campione nuovo succede in 9 coppie su 21; queste hanno ora un'etichetta propria.
- Nel campione di sviluppo succede in 5 coppie su 50, con un'etichetta sola per coppia. Escluse quelle, l'italiano
  di Gemma passa dal 56% al 60% di risposte sicure. Spiega una parte piccola del distacco dall'inglese.

**Regole dichiarate prima del run.**
- **Termini equivalenti:** si contano come lo stesso termine se hanno la stessa definizione logica, oppure se le
  loro etichette hanno le stesse parole in altro ordine o forma.
- **Più reperti nella frase:** un'etichetta può elencare più termini, uno per reperto, e la risposta si confronta
  con il più vicino.
- **Etichetta propria** per l'item italiano non appaiato.

**Bracci e criteri del terzo run**, invariati come chiesto:
- **B** (scelta dei modelli dalla shortlist): almeno 80% di risposte sicure, al massimo 5% di troppo specifici,
  meglio di A e del riferimento negli esatti;
- **C** (A e B d'accordo): almeno 95% di risposte sicure;
- **italiano:** B entro 10 punti dall'inglese;
- **I** (integrato): è il braccio che scriverebbe nel grafo, quindi deve stare alla soglia di C, almeno 95% di
  risposte sicure e al massimo 5% di troppo specifici, con la copertura dichiarata;
- **S** (B più la verifica) e **D** (discesa) sono riportati per capire, non decidono.

Se I non passa, nessun reperto entra nel grafo. Gli item che vanno alla persona diventano la prima coda per il
radiologo.

## Aggiornamenti dei modelli (verificati il 31 ottobre)

- **Gemma**: la famiglia 4 (aprile 2026: E2B, E4B, 26B MoE, 31B denso; giugno: 12B) non ha un successore. Il 31B
  denso resta il più adatto. MedGemma è ancora su base Gemma 3: non lo considero superiore per questo compito.
- **Nemotron**: dopo Nemotron 3 Ultra (550B, giugno) è uscito solo Nemotron 3.5 Lightning (30B, agosto, per agenti
  veloci): più piccolo, non superiore. Nemotron 4 è annunciato con Mistral ma non ancora rilasciato.
- **Da valutare come terzo modello**:
  - **Mistral Large 4**: anteprima del 6 ottobre, circa 1T di parametri con 52B attivi, pesi aperti promessi entro
    fine ottobre con licenza non ancora pubblicata. Europeo, rilevante per l'MDR.
  - **Mistral Medium 3.5**: pesi aperti, licenza MIT modificata.

  Non ho verificato gli id su OpenRouter: quando i pesi di Large 4 saranno pubblici basta aggiungerlo a `models`.
- Le classifiche dei modelli aperti in testa sono tutte di laboratori cinesi; restiamo con le famiglie non cinesi,
  come deciso.

## Fonti

- HPO release 2026-09-01 (`hp-full.obo`), UBERON (`uberon.obo`), tabella italiana `hp-it.babelon.tsv` del progetto
  hpo-translations: conteggi calcolati sui file.
- [HPO, pagina dei download](https://human-phenotype-ontology.github.io/downloads.html)
- [Stramare et al. 2011, lessico RadLex inglese–italiano in OpenEye](https://research.unipd.it/handle/11577/2478309)
- [Nemotron (Wikipedia)](https://en.wikipedia.org/wiki/Nemotron), [Gemma (Wikipedia)](https://en.wikipedia.org/wiki/Gemma_(language_model))
- [Guida ai modelli Mistral 2026](https://aicomparison.ai/mistral-models-guide/)
- [Classifica dei modelli aperti, BenchLM](https://benchlm.ai/best/open-source)
