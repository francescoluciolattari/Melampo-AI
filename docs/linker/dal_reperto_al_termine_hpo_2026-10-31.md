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
