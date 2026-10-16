# Primo probe su Kaggle e seconda lettura delle immagini (16 ottobre 2026)

## Il run

VoxTell e TotalSegmentator su 5 casi pubblici di MSD Task06 Lung (CC BY-SA 4.0), Kaggle, GPU T4, prompt italiano "tumore polmonare" e inglese "lung tumor".

| caso | Dice it / en | misurato (mm) | riferimento (mm) | stato |
|---|---|---|---|---|
| lung_001 | 0,79 / 0,78 | 15,7 | 17,2 | agree |
| lung_003 | 0,94 / 0,94 | 19,0 / 19,3 | 19,3 | agree |
| lung_004 | 0,85 / 0,85 | 16,3 | 17,9 | agree |
| lung_005 | 0,94 / 0,94 | 32,0 / 32,1 | 32,0 | agree |
| lung_006 | 0,86 / 0,86 | 137,1 / 138,1 | 26,7 | agree (errore) |

Cosa abbiamo imparato:
- Il prompt italiano funziona quanto quello inglese. VoxTell codifica il testo con Qwen3-Embedding-4B, che è multilingue.
- Quattro diametri su cinque cadono entro 1,6 mm dal riferimento.
- Il caso lung_006 è stato dichiarato "agree" con un diametro 5 volte quello vero. Il controllo verificava solo la posizione nell'organo, non la misura. La causa più probabile è un pezzo staccato della regione sulla stessa fetta, che entra nel diametro massimo senza pesare sul Dice. La conferma arriverà dal prossimo run, che stampa le parti.
- Tempi: 72-200 s per chiamata. La CLI ricarica il modello a ogni chiamata. Il prompt italiano, che non è tra gli embedding precalcolati in inglese, carica anche il codificatore del testo.
- Non è stato verificato se MSD Lung sia tra i dati di addestramento di VoxTell (appendice B del paper, arXiv 2511.11450). Finché non lo è, questi Dice non sono numeri da citare.

## Cosa cambia

1. **Misura sulla parte principale.** I diametri si misurano sul pezzo più grande della regione (`lesion_measures.split_main_part`). I pezzi staccati vengono riportati con volume, quota e distanza. Un pezzo di almeno un decimo del principale va a una persona, perché potrebbe essere una seconda lesione.
2. **Seconda lettura deterministica** (`imaging/threshold_check.py`), il metodo del software volumetrico dei tomografi:
   - parte dalla regione di VoxTell, usata come "click";
   - tiene i voxel ≥ -400 HU connessi alla regione, entro 6 mm;
   - un'apertura morfologica stacca i vasi sottili.
   Le due letture concordano se Dice ≥ 0,70 e se i diametri differiscono di non più di max(2 mm, 15%). Se la soglia arriva al bordo della zona di ricerca (parete toracica, mediastino), la lettura non vale. Si applica solo a TC del polmone. Una lesione a vetro smerigliato non viene confermata e va a una persona, ed è il comportamento voluto.
3. **Verifica con il referto.** La dimensione scritta nel referto (letta da `sizes_in_text` sulla clausola collegata dal linker) deve coincidere entro max(3 mm, 30%). Ogni reperto che l'immagine non conferma va a una persona, con i motivi (`ImageSide.review`, `needs_review` nel grafo):
   - fuori dalla struttura nominata;
   - non trovato nell'immagine;
   - più regioni;
   - parti staccate;
   - misura diversa dal referto;
   - seconda lettura in disaccordo;
   - struttura non verificata;
   - misura che niente di indipendente conferma (nessuna misura scritta e nessuna seconda lettura).
4. Le misure non confermate non entrano nella crescita tra esami: si usa la misura del referto. Non generano il vettore di forma.
5. VoxTell si usa tramite la sua API Python: il modello si carica una volta e i prompt si codificano una volta. Se l'API fallisce si torna alla CLI, e il motivo resta nelle note.
6. Il probe scrive la frase di un referto sintetico con la misura di riferimento (MSD non ha referti; la riga è segnata come sintetica). Stampa:
   - il Dice della regione, della parte principale e della lettura a soglia;
   - il diametro della parte principale e quello della regione intera;
   - i motivi per cui un reperto va a una persona.
   I casi sono 15.

## Precisione: cosa si può chiedere

Il Dice contro un contorno manuale ha un tetto: due radiologi che contornano lo stesso tumore non coincidono. Anche il diametro di un nodulo varia di circa 1,5-2 mm tra lettori. L'obiettivo del 99% non riguarda il Dice. Riguarda i reperti accettati: una misura viene accettata solo quando due fonti indipendenti (il referto e/o la lettura a soglia) la confermano, altrimenti va a una persona. La precisione si ottiene astenendosi, non chiedendo a un solo modello di non sbagliare mai.

Restano da fare: referti veri accoppiati a immagini (es. CT-RATE, da verificare la licenza), lesioni del fegato e del pancreas (MSD Task03/07) dove la soglia non si applica, ecografia.
