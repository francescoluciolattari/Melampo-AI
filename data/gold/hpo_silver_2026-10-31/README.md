# Dal reperto al termine HPO: etichette di sviluppo ("silver"), 31 ottobre 2026

Etichette di **un solo etichettatore (Claude)**, non di radiologi. Servono a sviluppare e a misurare in modo
ripetibile, **non certificano** nulla. Ogni riga dà il termine HPO (release 2026-09-01) più specifico che la
frase sostiene sul reperto alla struttura dell'item, oppure `NONE`.

Le etichette sono state scritte dopo aver visto la frase e la shortlist del generatore di candidati (non la
risposta del metodo deterministico), **prima** di interrogare qualunque modello. Due parametri del generatore
(`ORGAN_SEARCH_DEPTH`, `WHOLE_DEPTH`) e l'ordine della shortlist sono stati scelti guardando queste etichette:
sul generatore questo set è di sviluppo, non di verifica.

Nel repository ci sono solo le etichette (`labels.txt`, formato `pair|etichetta|nota`): PARROT è CC BY-NC-SA 4.0.
Gli item si ricostruiscono dallo stesso seme:

```
git clone --depth 1 https://github.com/PARROT-reports/PARROT_v1.0.git PARROT
python scripts/hpo_probe.py sample --parrot PARROT/data/PARROT_v1_0.jsonl --out hpo_items.jsonl
```

Composizione: 150 reperti, cioè 150 coppie con etichetta.
- 100 vengono da referti non italiani, letti nella traduzione inglese.
- 50 vengono da referti italiani: ognuno è letto sia nell'originale sia nella traduzione inglese, e le due
  versioni condividono l'etichetta.
- Al massimo 6 reperti per struttura e un reperto per referto.
- 107 etichette sono un termine HPO, 43 sono `NONE`.

`NONE` vuol dire una di queste cose: frase normale; solo misure; tecnica d'esame; titolo; quesito clinico; frase
troncata dalla lettura; reperto riferito a un'altra struttura rispetto a quella collegata dal linker. In
quest'ultimo caso l'item rivela un errore di sede del linker, non del passaggio a HPO: sono 11 su 150.

Convenzioni da far rivedere ai radiologi:
- in HPO mancano alcuni termini, e si è scelto il più vicino più generale: per la diverticolite "Colonic
  diverticula"; per il fango biliare "Abnormal gallbladder morphology"; per l'infarto renale "Abnormal renal
  morphology"; per la prostata ingrandita "Abnormal prostate morphology" (non c'è prostatomegalia, e l'ipertrofia
  prostatica benigna non è scritta);
- "suggestivo di" non vale come affermato: si etichetta ciò che è osservato (pareti vescicali ispessite, non
  cistite);
- una massa solida e captante è una neoplasia dell'organo; una lesione non caratterizzata non è una cisti.
