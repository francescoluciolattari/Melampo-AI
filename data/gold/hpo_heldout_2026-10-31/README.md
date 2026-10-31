# Dal reperto al termine HPO: etichette di verifica ("held-out"), 31 ottobre 2026

Campione nuovo per il terzo run. Nessuno dei suoi referti è nel campione di sviluppo
(`../hpo_silver_2026-10-31`).

Come il campione di sviluppo, le etichette sono di **un solo etichettatore (Claude)**, non di radiologi, e non
certificano nulla. A differenza di quel campione, sono state scritte **prima di qualunque metodo**: nessun modello,
nessuna risposta del metodo A, nessuna shortlist. L'etichettatore ha visto solo la frase e ha cercato in tutta HPO
per nome. Nessun parametro del generatore, della discesa o dell'integrazione è stato scelto su questi item.

Nel repository ci sono solo le etichette (`labels.txt`): PARROT è CC BY-NC-SA 4.0. Gli item si ricostruiscono così:

```
python scripts/hpo_probe.py sample --parrot PARROT/data/PARROT_v1_0.jsonl --seed 2026103102 \
    --exclude data/gold/hpo_silver_2026-10-31/labels.txt --out hpo_items_heldout.jsonl
```

**Composizione:** 184 item (142 inglesi, 42 italiani), 142 coppie, 47 strutture; 48 etichette `NONE`.
- I referti italiani di PARROT finiscono qui: restano 21 coppie, non 25.

**Novità dichiarate prima del run:**
- Un'etichetta può elencare più termini, uno per ogni reperto che la frase afferma. Una risposta si confronta con
  il termine più vicino.
- Un item italiano può avere un'etichetta propria. L'appaiamento fra italiano e inglese è per struttura, e in 9
  coppie su 21 la frase italiana scelta non è la traduzione di quella inglese. Lo stesso accade in 5 coppie su 50
  del campione di sviluppo, dove però l'etichetta era una sola per coppia.
- Le 28 righe che finiscono con "?" sono da far rivedere a un radiologo.
