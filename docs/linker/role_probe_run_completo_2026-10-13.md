# Role probe, run completo (13 ottobre 2026)

Il run è completo: 963 item (gold set silver e 300 Landmark di Rad-SpRL), 859 anatomici, cinque strati (E3C inglese e italiano, IU X-ray, PARROT italiano, Rad-SpRL). Modelli: gpt-oss-120b in FP8, Gemma 4 31B, Nemotron 3 Ultra. Mistral Small 4 è stato tolto dopo il primo run. Un solo item è rimasto senza risposta di Gemma. Il run è durato circa 5.000 secondi, con le risposte del run precedente riusate (`resume_run`).

È un set di sviluppo (silver). Le regole di accordo qui sotto le ho scelte guardando questi stessi dati, quindi i numeri sono ottimistici.

## Un modello da solo

| braccio | accuratezza | errori "sede o non sede" | "struttura" detta a torto |
|---|---|---|---|
| reticolo | 85,0% | 0 | 89 |
| lettore Kintsch | 88,1% | 3 | 88 |
| Nemotron 3 Ultra | **94,3%** | 0 | 4 |
| Gemma 4 31B | 91,1% | 7 | 4 |
| gpt-oss-120b FP8 | 87,8% | **43** | 4 |

- **Dare le letture del lettore al modello non migliora** (Nemotron 94,5%, Gemma 90,9%, gpt-oss 87,7% con 46 errori sede/non sede).
- **gpt-oss** sbaglia soprattutto su IU X-ray (45/76 giuste, contro 74/78 di Nemotron) e Rad-SpRL. Chiama "altro" o "non è una sede del corpo" parole come *cardiac silhouette*, *cardiomediastinal silhouette* e *goiter*. È una convenzione radiologica (la silhouette cardiaca è la sede), che il modello non conosce. Per i ruoli, da solo, non va bene.
- Le menzioni NOT_ANATOMY (lesioni, liquidi, cellule) sono lette come struttura 87 volte su 104 dal lettore e dal reticolo, 14 (Gemma), 23 (gpt-oss), 27 (Nemotron) dai modelli da soli.

## Due letture indipendenti che concordano, altrimenti a una persona

Il report calcolava l'accordo tra il lettore e il modello *che aveva visto* il lettore: sbagliato, perché il modello tende a ripetere la lettura del lettore. Il programma ora confronta il lettore con la risposta del modello da solo (`exact->agree:<modello>`) e con tutti i modelli insieme (`exact->agree_all`). I numeri qui sotto vengono dalle stesse risposte, ricalcolate fuori dal programma (il codice nuovo dà gli stessi conti).

| regola | a una persona | accuratezza sulle risposte | errori |
|---|---|---|---|
| lettore e Gemma | 20,5% | **99,27%** | 5 |
| lettore, Gemma e Nemotron | 22,8% | **99,40%** | 4 |
| lettore con Gemma o con Nemotron | 14,9% | 99,18% | 6 |
| Gemma e Nemotron (senza lettore) | 8,7% | 95,5% | 35 |
| maggioranza di 3 modelli | 1,9% | 94,4% | 47 |
| come sopra, più "procedure_site" se tre modelli concordano | 12,7% | 98,0% | 15 |

Il lettore da solo conta: senza di lui i modelli ripetono gli stessi errori di convenzione (la silhouette, il torace). Con lui l'errore scende sotto l'1%.

**I 4 errori di "lettore, Gemma e Nemotron":**

- *uterus*, in "uterus size": il gold dice struttura, loro `inherent_location`, che è difendibile;
- *abdominal wall*, in "abdominal wall incisions": loro `procedure_site`, che è difendibile;
- *abdominal cavity*: la frase parla di "pelvic cavity"; il gold è dubbio;
- *un albero tracheobronchiale* (broncoscopia): il gold dice sede di procedura, loro struttura.

Almeno tre sono dubbi sull'etichetta, non sul linker.

**Cosa va a una persona (196 su 859):** 107 strutture, 84 sedi di procedura, 5 altre. Le sedi di procedura sono quasi tutte rimandate perché il lettore le legge come strutture: il modello le sa leggere (80 su 92), ma il lettore no.

## Conclusione operativa

Per lo stadio 6, il minimo è: **lettore e Gemma 4 31B concordano, altrimenti una persona.**

- È in una sola GPU, con licenza Apache 2.0.
- Costa un 20% di item a una persona.
- Aggiungere Nemotron non cambia quasi nulla (99,40% contro 99,27%) e richiede 8 GPU.
- gpt-oss-120b non entra nello stadio 6 per i ruoli.

Lavoro aperto: se il lavoro della persona sulle sedi di procedura (84) si può ridurre con un lettore migliore delle procedure, o con un LoRA su Gemma 4 31B (punto 2, serve una GPU). Questi numeri vengono da un set di sviluppo: una conferma richiede un gold set a doppio cieco.
