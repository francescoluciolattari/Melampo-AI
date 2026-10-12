# Deterministico solo dove il vocabolario ha la corrispondenza (12 ottobre 2026)

## La critica di Frank

"Degenerative changes of the spine" veniva letto come proprietà della colonna perché NCIt classifica "Change" come azione. La prima correzione è stata un'altra regola deterministica sui tipi NCIt. Frank ha obiettato che il problema è proprio il determinismo. La lettura dovrebbe essere deterministica solo quando la corrispondenza c'è davvero. Negli altri casi dovrebbero decidere il lettore Kintsch, con NCIt come supporto, e un LLM, leggendo il sintagma intero.

## La misura: due livelli di lettura del reticolo

Ogni lettura del reticolo ora dice da dove viene (`Reading.exact`):

- **esatta**: il tipo viene da un nome che il vocabolario contiene così come è scritto, di due parole o più ("brain weight", "liver transplantation");
- **inferita**: il tipo è composto dal tipo di una sola parola testa ("changes" diventa un'azione NCIt, "volumes" una proprietà).

Precisione dei ruoli letti dal reticolo, con la memoria dei blocchi di prima:

| dati | letture inferite giuste | letture esatte giuste | lettore Kintsch, sugli stessi item delle inferite |
|---|---|---|---|
| Rad-SpRL, 2.084 Landmark | **0/377 (0%)** | nessuna | 353/377 (93,6%) |
| gold set silver, item anatomici | 7/35 (20%) | nessuna | 23/35 (65,7%) |
| MedMentions, link con ruolo nell'etichetta | 150/478 (31,4%) | 10/12 (83,3%) | 384/478 (80,3%) |

Frank aveva ragione. Un ruolo dedotto da una sola parola testa è sbagliato quasi sempre: 0%, 20% e 31%. Sugli stessi item il lettore, che integra l'evidenza invece di applicare una regola, ha ragione nell'81–94% dei casi. Solo la corrispondenza esatta merita di decidere da sola.

Lettore e reticolo a confronto, su tutti gli item:

| dati | reticolo | lettore | cascata "esatta, altrimenti lettore" |
|---|---|---|---|
| Rad-SpRL | 81,9% | 98,5% | 98,5% |
| gold set silver | 79,1% | 82,1% | 82,1% |
| MedMentions | 80,6% | 93,0% | 93,0% |

Con la memoria corretta del pomeriggio (processo patologico come malattia, azione come tipo a sé), il lettore cambia pochissimo: Rad-SpRL 98,5% → 98,5%, silver 82,1% → 82,3%, MedMentions 93,0% → 93,6%. La correzione dei tipi aiutava soprattutto il reticolo, che non deve decidere comunque. Resta come evidenza più pulita, non come regola che decide.

## Cosa cambia nel linker

**`_another_method_decides`.** Prima il reticolo prendeva il posto del lettore per *qualunque* lettura di ruolo, anche se inferita da una sola parola testa. Ora lo fa solo quando la lettura è esatta. Nel caso contrario il lettore legge, e, se è acceso, lo stadio 6 rilegge con l'LLM. La traccia `blocks` dice se la lettura era esatta o inferita.

**`role_probe`, bracci composti.**

| braccio | cosa fa |
|---|---|
| `exact->reader` | il reticolo se esatto, altrimenti il lettore |
| `exact->llm+reader:<modello>` | il reticolo se esatto, altrimenti il modello che vede le letture del lettore e il sintagma |
| `exact->agree:<modello>` | il reticolo se esatto, altrimenti lettore e modello quando concordano; quando non concordano, `uncertain`, cioè decide una persona |

**Le letture inferite** restano nel prompt come evidenza ("composed from the kind of one head word"), con l'avvertenza che possono essere sbagliate.

## Dove resta il problema

- **Sedi di procedura in italiano** (gold set silver): 88 non sono lette né dal reticolo né dal lettore. Lì serve l'LLM con il sintagma. Il workflow `role-probe` lo misura.
- **Falsi ruoli del lettore:** 32 su Rad-SpRL ("lung volumes", "upper right abdomen") e circa 55 su MedMentions.

## Modelli migliori di gpt-oss-120b, non cinesi

I punteggi vengono dalle pagine OpenRouter, che riportano Artificial Analysis e Vals AI (12 ottobre 2026), con la modalità ragionamento dove esiste. Sono benchmark generali, non sul nostro compito.

| modello | origine, licenza | parametri (attivi) | indice AA | GPQA Diamond | IFBench (seguire istruzioni) | hardware locale in FP8 |
|---|---|---|---|---|---|---|
| gpt-oss-120b | OpenAI (USA), Apache 2.0 | 117B (5,1B) | 11,6 | 78,2% | 69,0% | 2×80 GB |
| **Gemma 4 31B** (`google/gemma-4-31B-it`, marzo 2026) | Google (USA), Apache 2.0 | 31B densi | **14,7** | **85,7%** | **75,6%** | **1 GPU** (circa 31 GB) |
| **Nemotron 3 Ultra** (giugno 2026) | NVIDIA (USA), NVIDIA Open Model License | 550B (55B) | **22,9** | **86,7%** | **81,4%** | 8 GPU (circa 550 GB) |
| Nemotron 3 Super | NVIDIA (USA), NVIDIA Open Model License | 120B (12B) | 12,8 | 80,0% | 71,5% | 2×80 GB |
| Mistral Small 4 | Mistral (Francia), Apache 2.0 | 119B (6B) | 11,3 | 76,9% | 48,2% | 2×80 GB |
| Mistral Large 3 (dic. 2025) | Mistral (Francia), Apache 2.0 | 675B (41B) | 9,3 | 68,0% | 36,2% | 8 GPU |

**Lettura della tabella.**

- **Gemma 4 31B** batte gpt-oss-120b su tutti e tre gli indici, sta su una sola GPU ed è anche una base densa adatta a LoRA (punto 2).
- **Nemotron 3 Ultra** è il migliore in assoluto, ma servono otto GPU.
- **Mistral Small 4** segue le istruzioni molto peggio (IFBench 48%). Resta utile come terza famiglia europea, ma non come giudice principale.

Il workflow `role-probe` prova ora per default tre famiglie non cinesi: gpt-oss-120b in FP8, Gemma 4 31B e Mistral Small 4. Nemotron 3 Ultra si aggiunge nell'input `models` (`nvidia/nemotron-3-ultra-550b-a55b`). La scelta dello stadio 6 la decide il nostro compito, misurato su silver e Rad-SpRL, non i benchmark generali.

## Fonti

- [OpenRouter, gpt-oss-120b](https://openrouter.ai/openai/gpt-oss-120b)
- [OpenRouter, Gemma 4 31B](https://openrouter.ai/google/gemma-4-31b-it)
- [OpenRouter, Nemotron 3 Ultra](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b)
- [OpenRouter, Nemotron 3 Super](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b)
- [OpenRouter, Mistral Small 4](https://openrouter.ai/mistralai/mistral-small-2603)
- [OpenRouter, Mistral Large 3](https://openrouter.ai/mistralai/mistral-large-2512)
- [xigh/open-weight-models](https://github.com/xigh/open-weight-models)
- [Wikipedia, Nemotron](https://en.wikipedia.org/wiki/Nemotron)
- Hugging Face API, `google/gemma-4-*` (date di creazione)
