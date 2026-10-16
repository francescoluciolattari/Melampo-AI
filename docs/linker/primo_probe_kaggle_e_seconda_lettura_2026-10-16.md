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

## Secondo probe (sera del 16 ottobre) e revisione

Risultati su 4 casi (lung_006 è andato in esaurimento di memoria salvando le maschere di una serie di 567 fette):
- **Parte principale.** Nel caso lung_001 VoxTell ha trovato anche una seconda regione: 26% del volume, a 59 mm. Misurata solo la parte principale, il Dice sale da 0,79 a 0,90 e il diametro coincide con il referto (15,7 contro 16 mm).
- **Dimensioni.** Tutte coerenti con il referto sintetico: 15,7/16, 19,0/19, 16,3/18, 32,0/32 mm.
- **Soglia fissa -400 HU, da buttare.** In tutti e quattro i casi è arrivata al bordo della zona di ricerca, scorrendo lungo i vasi e la pleura. Ha dato 25, 32,8 e 44,6 mm contro 16, 18 e 32. Solo in lung_003 coincideva (18,7 mm).

Cambiamenti decisi con Frank:
1. **Quando un reperto va a una persona.** Solo se la lesione del referto non viene identificata (non trovata, più regioni comparabili, fuori dalla struttura, struttura non verificata), oppure se l'immagine differisce dal referto (misura, densità o composizione chiaramente opposte alle parole del referto). Non vanno più a una persona una lesione a vetro smerigliato, una misura confermata da una sola lettura (`measure_support`) o un'altra regione trovata (`other_regions`): sono attributi.
2. **Seconda lettura con soglia a metà strada** (`threshold_check.py`), come la volumetria PET:
   - la soglia sta a metà tra il nucleo della lesione e il tessuto intorno, dal lato della lesione;
   - la ricerca è limitata all'organo, così la parete toracica resta fuori;
   - si applica dove la lesione si distingue dal rumore (rapporto contrasto/rumore ≥ 2,5): TC di qualunque organo, RM e PET;
   - un nodulo a vetro smerigliato viene letto con la sua soglia (circa -725 HU).
3. **Profilo di densità per ogni regione** (`imaging/density.py`):
   - distribuzione (percentili, entropia);
   - in TC, quota in fasce di HU: aria, vetro smerigliato, grasso, liquido, tessuti molli, alta densità, calcio;
   - profilo dal bordo al centro, per profondità relativa, e fuori a 0-2, 2-5 e 5-10 mm;
   - gradiente sul bordo (nitidezza del margine) e asimmetria tra ottanti;
   - densità rispetto al tessuto intorno (ipo/iso/iper);
   - nel polmone aerato, componente solida (quota e diametro, come in Lung-RADS) e composizione solid/part_solid/ground_glass;
   - indicatori di calcio, grasso (valutato all'interno, lontano dal bordo, per l'effetto di volume parziale) e cavitazione.
   Il profilo è anche un vettore (`density_hu`) per ricerca e confronto. In RM ed ecografia restano solo valori relativi: le intensità non sono una scala assoluta.
4. **Parole del referto sulla composizione** (`site_qualifiers`): solido, parzialmente solido/subsolido, vetro smerigliato, calcifico, adiposo, cistico/fluido, cavitato, e le forme negate ("non calcifico"). `compare_with_report` segnala solo le contraddizioni nette.
5. **Probe:**
   - volumi in float32;
   - TotalSegmentator con un solo thread di salvataggio;
   - un caso che fallisce viene registrato e gli altri proseguono;
   - probe.json viene scritto dopo ogni caso;
   - i due prompt girano in una sola esecuzione;
   - la frase sintetica non dice più "solida", che i dati non dicono.

Le soglie (fasce di HU, -300 HU per la parte solida, rapporto 0,8 per "solida") sono valori di partenza da calibrare su dati annotati.
