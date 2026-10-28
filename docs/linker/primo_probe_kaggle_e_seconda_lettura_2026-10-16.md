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

## Terzo probe: 15 casi (notte del 16 ottobre)

Tutti completati; lung_006 corretto (25,2 mm contro 26,7). Prompt italiano e inglese danno gli stessi risultati.

| | |
|---|---|
| accettati | 14 su 15 |
| a una persona | lung_014: VoxTell trova 7-10 pezzi e la parte principale non è il tumore (Dice 0,00). Lo stato "ambiguous" lo ha fermato ed era giusto |
| Dice della parte principale | 0,71-0,94, mediana circa 0,89 |
| errore di diametro sui casi accettati | entro 2 mm in 9 casi |

Errori, letti con il criterio stretto (max 2 mm, 10%):
- **Tumori parzialmente solidi** (riferimento con alone a vetro smerigliato): lung_016 (-3,0 mm), lung_018 (-4,3), lung_020 (-3,7), lung_023 (-5,1). VoxTell segmenta la parte solida e lascia fuori il vetro smerigliato, quindi anche la composizione risultava "solid" invece di "part_solid".
- **lung_022:** 60,5 mm contro 77,6, accettato perché la tolleranza del 30% era troppo larga.
- **"Contiene grasso" falso** su 14 tumori su 15: rumore della TC e miscela aria-tessuto nello stesso voxel (effetto di volume parziale).
- **"Calcio"** in lung_004: con il contrasto potrebbe essere un vaso opacizzato.
- **Seconda lettura a metà strada:** d'accordo solo su lung_003 e lung_018. Negli altri casi scorre ancora lungo i vasi dentro il lobo (+6..+15 mm).

Correzioni:
1. **Tolleranza sulla misura** del referto ridotta a max(3 mm, 15%). lung_022 (17 mm) e lung_023 (5 mm) andrebbero a una persona.
2. **Estensione a vetro smerigliato** (`threshold_check.ground_glass_extent`): nel polmone aerato, la lesione con la sua parte a vetro smerigliato, con soglia di 150 HU (e almeno 3 volte il rumore) sopra il polmone intorno. Tra le due letture dell'immagine si tiene quella che misura ciò che misura il referto (`measured_on`). La densità e la composizione si calcolano sulla lesione intera.
3. **Grasso:** solo un focolaio, di almeno 10 mm³, nei valori mediati su 3×3×3 voxel tra -120 e -40 HU, a 2 mm o più dal bordo e lontano da tessuto aerato o a vetro smerigliato.
4. **Alta densità:** un focolaio sopra 200 HU si chiama `high_density_focus`. Diventa `contains_calcium` solo se si sa che l'esame è senza contrasto (`contrast="unenhanced"`).
5. **Seconda lettura:** apertura nel piano di 2 mm, così i vasi fino a circa 4 mm si staccano.
6. **Probe:**
   - riporta l'errore rispetto al contorno di riferimento e se è entro il criterio stretto;
   - riepiloga quanti casi accettati sono davvero giusti e quanti di quelli mandati a una persona erano davvero sbagliati;
   - i casi salgono a 30.

## Quarto probe (17 ottobre, parziale: 10 casi su 30 nell'output ricevuto)

- **Accettati:** 7 con il prompt italiano. Fuori dal criterio stretto: lung_020 (-3,8 mm) e lung_016 (-2,96 mm, per 0,03 mm). In entrambi è la parte a vetro smerigliato che manca.
- **A una persona:**
  - lung_023: misura 5,1 mm più corta. Errore vero, giusto fermarlo;
  - lung_026: dato "fuori dal polmone" (40% nella maschera) ma misurato bene (-1,3 mm). È un falso allarme: TotalSegmentator segmenta il parenchima, e il tumore che lo sostituisce manca dalla maschera del lobo.
- **Estensione a vetro smerigliato:** non ha aggiunto nulla su nessun caso. L'apertura cancellava l'alone, spesso pochi millimetri; la soglia di 150 HU sopra il polmone lasciava fuori un alone a -725 HU.
- **Grasso:** nessun falso positivo dopo la correzione. Il "focolaio ad alta densità" resta su lung_004 e lung_026, ed è corretto non chiamarlo calcio.
- **Seconda lettura:** d'accordo in 3 casi su 10. Resta un supporto in più, non decide nulla.

Correzioni:
1. **Alone a vetro smerigliato** fatto crescere voxel per voxel dalla regione, solo attraverso la fascia del vetro smerigliato, fino a 10 mm:
   - fascia: valori mediati su 3×3×3 voxel, almeno 100 HU sopra il polmone (e 3 volte il rumore del parenchima), sotto -300 HU;
   - il voxel stesso deve essere fuori dal rumore del polmone;
   - nessuna apertura. I vasi sono più densi della fascia e la crescita non li segue.
2. **Controllo dell'organo:** se meno della metà della regione è nella maschera ma almeno il 20% la tocca, conta la sagoma convessa di ogni lobo, fetta per fetta. Se l'80% ci sta dentro, lo stato è "agree" con la nota `inside_the_outline_of_the_structure`.
3. **Fase del contrasto dal DICOM** (`dicom_volume.contrast_phase`): ContrastBolusAgent, altrimenti le parole di SeriesDescription e ProtocolName ("senza mdc", "basale", "portale", "arteriosa"). Un tag assente vale "unknown", non "senza contrasto".

## Quinto probe (17 ottobre): 30 casi completi

Riepilogo del notebook (prompt italiano; inglese tra parentesi):
- accettati 20 (20), di cui entro max(2 mm, 10%) dal contorno di riferimento 16 (17): precisione circa 80%;
- a una persona 10 (10), di cui davvero sbagliati 10 (9): nessun falso allarme in italiano;
- lung_026, prima dato fuori dal polmone, ora è accettato grazie al controllo sulla sagoma del lobo (-1,4 mm).

Accettati sbagliati: lung_016 (-3,0 mm, fuori per 0,03), lung_020 (-3,8), lung_027 (-8,1 mm su 57), lung_044 (-5,0).

Cosa mostrano gli errori:
- **Sempre per difetto.** Quasi tutti gli errori sono misure più corte del riferimento, e crescono con la dimensione: oltre 5 cm VoxTell sottostima del 15-20% (lung_022 -17, lung_028 -14,6, lung_031 -13,9, lung_033 -13 mm). Questi casi vanno correttamente a una persona.
- **Alone ancora mancante.** Nei tumori soprattutto a vetro smerigliato (lung_042, lung_043, lung_044) l'estensione non c'è o è stata scartata. Il prossimo run stampa perché: soglia, rumore, voxel aggiunti, bordo raggiunto.
- **Regressione corretta.** La densità calcolata sull'estensione ha fatto leggere "part_solid" due tumori solidi (lung_003, lung_004), perché l'estensione era cresciuta nel polmone intorno. Ora la densità è quella della regione misurata.

Correzioni:
1. **Tolleranza** con il referto: max(3 mm, 10%), che è anche l'ordine della variabilità tra radiologi. Ricalcolata sui numeri di questo run (arrotondati): circa 17 accettati, di cui 2 fuori dal criterio stretto (precisione circa 88%), 13 a una persona, nessun falso allarme.
2. **Densità** calcolata sulla regione misurata.
3. **Probe:** dettagli dell'estensione a vetro smerigliato.

Limite: il referto sintetico riporta esattamente la misura di riferimento. Con referti veri la misura del radiologo ha la sua variabilità, quindi la tolleranza va riverificata.

## Misure corte che crescono con la dimensione: cause note e verifica (17 ottobre)

**Cosa dice la letteratura:**
- **Regressione verso la media.** Le reti di segmentazione tendono a sovrastimare i volumi piccoli e a sottostimare quelli grandi; il punto di svolta è vicino al volume medio dei dati di addestramento. Vale sia con la loss Dice sia con la cross-entropy (Bertels et al., analisi teorica e sperimentale del bias di volume; arXiv 2211.04161).
  - Il taglio a 0,5 delle probabilità porta a sotto-segmentare i bordi incerti.
  - La somma delle probabilità (volume "morbido") è meno distorta.
  - Una ricalibrazione lineare stimata su dati separati toglie gran parte del bias.
- **Stesso effetto in clinica.** Un ensemble 3D U-Net per tumori polmonari (Radiology, gennaio 2025; 1.504 TC) dichiara come limite la tendenza a sottostimare il volume, con risultati peggiori sui tumori molto grandi.
- **VoxTell:** addestrato con Dice + cross-entropy, binarizza a logit > 0 (probabilità 0,5) e non ricampiona l'immagine. Il README suggerisce 1,5 mm se la spaziatura è insolita. Il modello lavora a patch di dimensione fissa in voxel: a 0,7 mm un tumore di 8 cm occupa più di 100 voxel per lato.
- **Altri modelli:**
  - nnInteractive (MIC-DKFZ 2025, pesi CC BY-NC-SA 4.0): accetta una segmentazione iniziale e un box 2D. È un secondo modello diverso, utile per rifinire il contorno.
  - LesionLocator (CVPR 2025, codice GPL-3.0, licenza dei pesi da verificare su Zenodo): il suo dataset include MSD Lung, quindi su questi casi non sarebbe una prova pulita.

**Verifica nel probe** (`--diagnose-size`, solo prompt italiano; nessuna di queste misure decide):
- volume rispetto al riferimento;
- quota del riferimento mancante, e dove sta: entro 2 mm dal bordo (bordo messo dentro) o più lontano (parti intere non trovate), con la sua densità;
- taglio delle probabilità di VoxTell a 0,35 e 0,2, e volume morbido;
- VoxTell sulla serie ricampionata a 1,5 mm;
- parti separate da meno di 5 mm unite alla principale;
- nnInteractive che rifinisce la regione partendo da un box.

Il notebook stampa l'errore medio di ciascuna variante per fascia di dimensione (sotto 30 mm, 30-50 mm, sopra 50 mm). La correzione si sceglie solo dopo aver visto quale variante toglie il bias, e va poi verificata su casi diversi da quelli usati per sceglierla.

### Risultato del sesto probe (17 ottobre, 21 casi su 30 completati): l'ipotesi del bias di volume non regge

- **Il volume è giusto.** Rapporto di volume mediano 1,03 sotto i 30 mm, 1,02 tra 30 e 50, 0,97 sopra i 50. Il diametro assiale invece è corto: errore medio -1,5 mm, -1,6 mm, -11,3 mm. Quindi non è una sottostima del volume (regressione verso la media), ma una regione di forma diversa dal riferimento.
- **Le correzioni della letteratura non aiutano.** Taglio a 0,35 o 0,2: al più 1-2 mm di recupero sui grandi (031: -14,2 → -12,4; 033: -17,4 → -17,2). Ricampionamento a 1,5 mm: peggiora quasi sempre (033: -46 mm). Unire le parti vicine: nessun effetto. Volume morbido ≈ volume del taglio.
- **nnInteractive non è affidabile:** sbaglia in entrambe le direzioni (+25,6 mm su 031, +14,4 su 022, -3,6 su 027, +0,2 su 018). 3 casi hanno dato "ndarray is not C-contiguous": corretto con `ascontiguousarray`.
- **Dove sta la parte mancante** sopra i 50 mm: non un bordo sottile (entro 2 mm dal bordo solo 38-57% contro 80-100% dei casi piccoli), mancante 15-28% e in eccesso 10-30%, a 5-8 mm dalla regione, a densità -577…-762 HU (polmone aerato o vetro smerigliato). Il riferimento include zone meno dense che VoxTell lascia fuori e ne aggiunge altre altrove.
- **Sotto i 50 mm** il metodo regge: tra i 12 casi accettati, con prompt inglese 12/12 entro tolleranza, con l'italiano 11/12 (lung_016, parzialmente solido, -3,0 mm contro tolleranza di 3 mm: al limite). I 9 inviati a una persona sono tutti davvero fuori (lung_014 ha segmentato un'altra struttura, Dice 0 sulla parte principale).
- **Seconda lettura a soglia:** d'accordo in 9 casi su 21. Conferma, non decide.
- **Prossimo passo:** il probe ora scrive la "forma": estensioni lungo gli assi principali del riferimento e della regione, quota del mancante agli estremi dell'asse lungo, blob mancanti e loro densità, sfericità. Dirà se il mancante è una coda (spicule, estensione lungo la pleura) o una zona separata.

### Settimo probe (18 ottobre, 15 casi completati): la forma, e la tolleranza scende a 2 mm

- **Sono due fenomeni diversi, non un bias unico.**
  1. *Bordo ruvido* (casi piccoli e medi): il riferimento ha sfericità 0,34-0,69, la regione di VoxTell 0,65-0,85. Il mancante è fatto di molti blob minuscoli (24-320), per il 35-68% agli estremi dell'asse lungo: punte e spicole che VoxTell arrotonda. Costa 0,1-1,1 mm sul diametro maggiore (1-4 mm nei casi 018, 004). È dentro la variabilità tra osservatori: non va corretto.
  2. *Pezzo coerente mancante* (022, 023, 020, 016, tutti parzialmente solidi o molto grandi): un unico blob che vale il 51-93% del mancante, con volume sotto il riferimento (0,85-0,91; 022: Dice 0,71, asse lungo 51 mm contro 64). VoxTell non ha segmentato una parte della lesione. Questi sono i casi che il confronto col referto manda a una persona, ed è il comportamento voluto.
- **Conclusione:** nessuna ricalibrazione globale è supportata dai dati (taglio, scala e nnInteractive non hanno aiutato nel sesto probe). La protezione è il confronto con la misura del referto, non una correzione della misura.
- **Una modifica:** la tolleranza sulla misura del referto passa da max(3 mm, 10%) a max(2 mm, 10%), cioè al limite "stretto" che il probe già usa per giudicare. Motivo: il caso 016 era accettato a 2,7 mm dal referto ma era a 3,0 mm dal riferimento. Con 2 mm sarebbe andato a una persona. Su questi 15 casi i 10 accettati sono tutti entro 1,6 mm.
- Questo run ha completato 15 casi su 30 (il precedente 21): i casi grandi e le serie lunghe sono quelli che mancano. Il campione dei grandi accettati resta quasi vuoto, quindi la precisione sopra 50 mm non è ancora misurata.
- **Perché i run si fermano a metà (log del settimo probe):** il log finisce dentro una barra di avanzamento, senza traceback, dopo il caso 025 (il sedicesimo): il processo è stato ucciso dal sistema, quasi certamente per memoria. VoxTell, TotalSegmentator e nnInteractive tengono memoria tra un caso e l'altro. Rimedio: `--isolate`, un processo per caso (la memoria torna al sistema ogni volta; un caso ucciso diventa una riga con il codice di uscita). Costo: il modello VoxTell si ricarica a ogni caso.

### Ottavo probe (18 ottobre): 30 casi su 30, prima volta completo (Kaggle in background, un processo per caso)

- **Accettati 16, a una persona 14.** Precisione sugli accettati: prompt inglese 16/16 entro max(2 mm, 10%) dal riferimento; italiano 15/16 (lung_016: 3,0 mm contro un limite di 2,9, per l'arrotondamento del referto sintetico 29,0 mm contro 29,26 mm reali; il confronto col referto lo passa a 2,7 mm). Non è ancora il 99%: 16 casi accettati sono pochi.
- **Inviati a una persona: 14 su 30 (47%).** Quasi tutti davvero fuori: casi grandi (022, 028, 031, 033, 027, 036, 042: 8-17 mm corti), parzialmente solidi (018, 020, 023, 043, 044: 3-5,5 mm corti) e il 037 (15 mm, volume 0,78). Falsi allarmi: solo 018 (4,3 mm contro un limite di 4,3). Il 014 (struttura sbagliata, Dice 0) è giustamente inviato.
- **Accettato sopra i 50 mm:** 009 (60 mm, -0,9) e 041 (64 mm, -5,6 mm, dentro il 10%).
- **La quota inviata a una persona (47%) è lontana dal <1% voluto.** Il tetto non è la precisione ma la copertura: le lesioni parzialmente solide e quelle grandi vanno quasi sempre a una persona. Prossimo lavoro: l'alone di vetro smerigliato nelle parzialmente solide (la crescita si ferma con la nota `reaches_the_search_zone_edge` e non viene usata).
- `probe.json` non era tra i file scaricabili da Kaggle; le righe per caso sono state ricostruite dal log.

### Alone di vetro smerigliato: taglio per profilo di densità (18 ottobre; SCARTATO dopo il nono probe, vedi sotto)

- **Dai dati dell'ottavo probe:** nelle lesioni parzialmente solide (016, 018, 020, 023, 037, 042, 043, 044) la crescita dell'alone si spingeva fino al bordo dei 10 mm (da 147 a 7544 voxel aggiunti) e il Dice con l'alone (0,66-0,83) era peggiore di quello della regione di VoxTell (0,75-0,89), mentre al riferimento mancavano solo 1,5-2,7 mm per lato. La banda "più densa del polmone" si attacca a tutto ciò che ha densità simile, non solo all'alone.
- **Correzione:** l'alone finisce dove il profilo di densità lascia il suo plateau, a metà tra il plateau (mediana dei primi 2 mm attorno alla regione, sui valori grezzi) e il polmone: la stessa regola "a metà strada" della seconda lettura. Se la densità non scende mai, non è un alone (nota `reaches_the_search_zone_edge`, estensione non usata); se accanto alla regione c'è solo polmone normale, nessun alone (`no_halo`).
- **Verifica prevista:** al nono probe, sulle stesse lesioni parzialmente solide, `with_ground_glass_axial_mm` deve avvicinarsi al riferimento (errore tra 1 e 2 mm, criterio di Frank del 18 ottobre: 2-3 mm sono troppi) senza peggiorare i casi solidi. Se non succede, la correzione si scarta. Solo test sintetici finora.

### Zone anatomiche, posizione e margini come dati (18 ottobre)

Principio (Frank): nessuno strumento fatto per un tipo di lesione. La lesione si trova dalla descrizione del referto, in qualunque zona anatomica sia; il sistema individua le zone dettagliate coinvolte, isola la lesione e converte caratteristiche (densità, margini) e morfologia in dati per il ramo del grafo. Le lesioni grandi vanno a una persona solo se la misura è sbagliata; la correzione va cercata nell'immagine, mai nella misura del referto (renderebbe il controllo circolare). Allargare la tolleranza non è una strada: sulle 29 lesioni isolate dell'ottavo probe, a 10% il referto di un'altra lesione passa il confronto di sola dimensione nel 9,9% dei casi, a 25% nel 25%.

- **Tutte le strutture coinvolte** (`imaging/involvement.py`): per ogni struttura della segmentazione dell'intero corpo (TotalSegmentator, task `total`, una sola esecuzione multi-etichetta per esame) la quota della lesione dentro (`lesion_share`), la quota della sua superficie esterna (0-2 mm) che vi affaccia (`surround_share`: un tumore che ha sostituito il tessuto non è nella maschera dell'organo, ma l'organo lo circonda), la distanza, la relazione (`inside` / `crosses` il confine / `contact`) e la quota della struttura occupata. Nel grafo: `(:Finding)-[:INVOLVES {relation, lesion_share, surround_share, distance_mm, structure_share}]->(:Structure)`. Sono attributi, non motivi per una persona: un referto non elenca ogni contatto.
- **Posizione nella struttura ospite:** distanza dalla superficie (entro 2 mm: sulla superficie, subpleurica o sottocapsulare), profondità relativa del baricentro e zona geometrica (periferica / intermedia / centrale; per il polmone il terzo interno è verso l'ilo solo approssimativamente). La maschera multi-etichetta resta una sola mappa (`StructureLabels`): cento maschere separate di una TC intera occuperebbero circa 8 GB.
- **Margini** (`imaging/margins.py`): nettezza come larghezza della transizione tra lesione e tessuto intorno (contrasto / gradiente al bordo; netti ≤ 3 mm, sfumati ≥ 5 mm, zona grigia in mezzo); contorno con due misure morfologiche in mm, valide a qualunque risoluzione: quota tolta da un'apertura di raggio 1,5 mm e numero di spicole (parti tolte lunghe almeno 3 mm), concavità riempite da una chiusura proporzionale alla lesione. Categorie: spiculato, lobulato, regolare, irregolare. Sotto 8 mm (o 6 voxel) il contorno non si giudica.
- **Dal referto:** nuove parole lette da `site_qualifiers` come `margin:<tipo>` (spiculato, lobulato, sfumato / mal definito, netti / ben definiti / circoscritti, contorni regolari / irregolari; gli aggettivi ambigui solo accanto a un nome di margine: "forma irregolare" non è un margine) e `position:surface` (subpleurico, sottocapsulare; non "centrale" o "periferico", ambigui).
- **Confronto col referto:** a una persona solo per opposti chiari, fuori dalla zona grigia (spiculata senza alcuna spicola; regolare con spicole evidenti; netti con transizione oltre 7,5 mm; sfumati sotto 2,4 mm; subpleurica a 6 mm o più dalla superficie). Soglie iniziali, da tarare su dati annotati: LIDC-IDRI ha per ogni nodulo i giudizi di quattro radiologi su margine, lobulazione e spiculazione (licenza da verificare).
- **Probe:** opzione `--anatomy` (nel notebook): per ogni caso le zone, la posizione e i margini sulla regione trovata e, accanto, sul contorno di riferimento, per vedere se la lettura dell'immagine trova le stesse zone del riferimento. Il comando `output` di `kaggle_api.py` ora segue tutte le pagine dell'elenco dei file (prima `probe.json` mancava).
- **Limiti:** solo test sintetici; le soglie non sono tarate; con fette spesse il bordo lungo z si sfuma e le spicole si perdono (lo spessore di fetta è registrato con i numeri).

### Nono probe (19 ottobre): l'alone non funziona e viene scartato; zone e margini

**Alone: criterio non raggiunto, correzione scartata (commit 8f4087e annullato).** Criterio di Frank: sulle 8 parzialmente solide (016, 018, 020, 023, 037, 042, 043, 044) errore tra 1 e 2 mm, senza peggiorare le solide.
- Sulle 8: nessun cambiamento. Tutte finiscono in `no_halo` (la densità appena fuori dalla regione è già sotto la soglia a metà strada: 31-116 HU sopra il polmone, nessun plateau). Errori invariati: -3,0 / -4,3 / -3,8 / -5,1 / -3,2 / -10,7 / -5,5 / -5,0 mm.
- Sulle solide: peggiora. Il taglio ha accorciato l'estensione che prima si usava in 029 (+0,7 → -3,2 mm), 034 (-3,4 → -6,0), 041 (-5,6 → -7,2), 045 (-1,9 → -5,3), che da accettati sono andati a una persona: accettati 16 → 12 (italiano 11/12 entro il limite stretto, inglese 12/12).
- Perché: il mancante delle parzialmente solide non è un alone attaccato alla regione. Sono punte alle estremità dell'asse lungo (34-58% del mancante), per lo più vetro smerigliato molto lieve (mediana tra -700 e -816 HU, cioè 60-180 HU sopra il polmone) che il riferimento include e la regola "a metà strada" no. È una differenza di convenzione sul bordo del vetro smerigliato lieve, non un errore di crescita.
- Circolarità da tenere d'occhio: `image_evidence` sceglie tra la regione del segmentatore e quella con alone la lettura più vicina alla dimensione del referto. Allarga di fatto la finestra di accettazione (due letture invece di una) e rende la verifica parzialmente circolare. Da decidere se sostituirla con una regola che usa solo l'immagine.

**Zone, posizione e margini (prima volta su dati reali, 30 casi, regione di VoxTell contro contorno di riferimento).**
- Struttura ospite: uguale in 29/30 (l'unico diverso è il 014, già segnalato: VoxTell ha segmentato un'altra lesione). Quote nei lobi coincidenti entro 0,01-0,05, anche per i lobi attraversati (lung_041: 0,56 + 0,38 contro 0,57 + 0,37) e per le strutture a contatto (aorta in 028: 0,25 contro 0,24).
- Zona (periferica / intermedia / centrale): uguale in 24/30; le differenze sono tutte tra categorie adiacenti, cioè al confine dei terzi. Distanza dalla superficie in genere entro 1-2 mm; fuori linea 042 (12,7 contro 0,0 mm: la regione trovata è corta e sta più dentro).
- Nettezza: uguale in 29/30 (nette / sfumate / nessun giudizio).
- **Contorno: uguale solo in 5/30, e non è un risultato attendibile.** Il riferimento risulta quasi sempre "spiculato" (10-150 spicole), la regione di VoxTell quasi sempre "irregolare" (0-3 spicole). Il contorno manuale è irregolare alla scala del voxel, la regione di VoxTell è liscia: la misura legge la scabrosità della maschera, non necessariamente le spicole vere. Non c'è una verità sulle spicole in MSD, e non si può dire quale delle due letture sia giusta. Finché non è tarata su LIDC-IDRI (giudizi di quattro radiologi su margine, lobulazione, spiculazione), la categoria del contorno si registra ma non può mandare un caso a una persona.

### Dopo il nono probe (19 ottobre): tre modifiche

Principio ribadito da Frank: nessuno strumento specifico per un tipo di lesione. La lesione si individua dalla descrizione del referto; il sistema trova le zone anatomiche dettagliate coinvolte, la isola in qualunque zona sia, e converte caratteristiche (densità, margini) e morfologia in dati per il ramo del grafo. Le parti legate a un organo (alone di vetro smerigliato, composizione solida/parzialmente solida del polmone) restano dati registrati, non scelgono né correggono la misura.

1. **La misura non dipende più dal referto.** Fino al nono probe, tra la regione del segmentatore e quella con alone si teneva la lettura più vicina alla dimensione scritta nel referto: una scelta fatta con la risposta in mano, che dà due finestre di accettazione invece di una. Sulle 29 lesioni isolate (regione dal nono probe, estensione dall'ottavo) la quota di referti di altre lesioni che passano il solo controllo dimensionale sale dal 9,6% al 12,6%, a fronte di 4 referti giusti in più su 29 (16 invece di 12). Ora la misura è sempre la regione del segmentatore; l'estensione con vetro smerigliato è solo un attributo (`with_ground_glass_axial_mm`). Costo atteso: 029, 034, 041 e 045 vanno a una persona (erano accettati solo grazie alla scelta). Gli accettati sul campione scendono a 12/30, con 11/12 (italiano) e 12/12 (inglese) entro il limite stretto: la copertura reale è questa. La dimensione sola, comunque, non basta a discriminare i referti (circa 10% di falsi passaggi tra lesioni diverse): la protezione viene dal controllo congiunto con sede, struttura e carattere.
2. **Il contorno (spiculato, lobulato, regolare, irregolare) è registrato ma non può più mandare un caso a una persona** (`CONTOUR_COMPARED = False` in `imaging/margins.py`) finché non è tarato su spiculazione annotata (LIDC-IDRI). La nettezza del margine (29/30 concordi tra regione e riferimento) resta confrontata col referto.
3. **Overlay per i casi fuori tolleranza** (`--overlays DIR`, `--overlays-all` per tutti): un PNG per caso con la fetta assiale e quella coronale della sezione maggiore, contorno del riferimento in verde, della regione trovata in rosso (giallo dove coincidono), toni di grigio dal 1° al 99° percentile attorno alla lesione. Scritto senza librerie (niente matplotlib). Nel notebook finisce in `/kaggle/working/overlays` e `kaggle_api.py output` lo scarica. Serve a vedere, su 022, 028, 031, 033 (8-17 mm corti), se il riferimento include polmone aerato o vetro smerigliato che VoxTell salta, oppure se è VoxTell a fermarsi prima.

### Decimo probe (20 ottobre): stesse misure del nono; gli overlay dicono dove sta la differenza

- **Numeri:** 30/30 casi, accettati 12 (italiano 11/12 entro il limite stretto, inglese 12/12), 18 a una persona. Misure identiche al nono probe: la rimozione della scelta "lettura più vicina al referto" non cambia nulla perché già non agiva più. 18 overlay generati (casi fuori tolleranza stretta), nessun errore di generazione.
- **Il mancante delle lesioni grandi non è tumore solido non visto.** Nelle lesioni sopra ~55 mm (033, 022, 028, 031, 036, 042) i voxel del riferimento che VoxTell non include hanno densità mediana da -577 a -762 HU (polmone aerato o vetro smerigliato, non tessuto molle); il volume della regione di VoxTell è pari o poco sotto a quello del riferimento (rapporto 0,92-1,03) ma con forma diversa: il 15-33% del riferimento manca e il 6-30% della regione è in eccesso. La densità mediana del riferimento è più bassa di quella della regione (022: -368 contro -116 HU; 031: -347 contro -272; 036: -134 contro -73) e correla con l'errore di misura (r = 0,61 sui 29 casi): il riferimento include una corona di densità polmonare attorno alla massa.
- **Negli overlay** (letti dalle miniature, 18 immagini arrivate senza nome file): dove la lesione è solida e ben delimitata i due contorni coincidono quasi del tutto; nei casi con errore grande il contorno del riferimento corre qualche millimetro oltre quello di VoxTell su polmone aerato, pleura o parete, mentre VoxTell segue il margine del tessuto molle. Un'immagine mostra solo il contorno di VoxTell (nessun riferimento nella fetta): è il caso in cui ha segmentato un'altra lesione (014).
- **Cosa non si può dire:** chi ha ragione. I referti sono sintetici e uguali al riferimento, quindi un "errore" è la differenza tra due convenzioni di bordo, non tra la misura e un radiologo. Un radiologo che scrive "massa di X cm" misura quasi sempre il tessuto molle sulla finestra polmonare, che è più vicino alla regione di VoxTell. Non si corregge la misura finché non c'è una misura umana indipendente.
- **Prossimo passo:** far misurare a un radiologo il diametro maggiore assiale sulle stesse fette (gli overlay sono pronti) e chiedere quale dei due contorni corrisponde alla lesione che misurerebbe. Se la misura umana è vicina a VoxTell, il confronto col referto reale sarà passato; se è vicina al riferimento, il problema è davvero di VoxTell. Registrare nel grafo sia l'estensione del tessuto molle (`solid_axial_mm`) sia quella totale come attributi distinti resta possibile; scegliere tra i due con le parole del referto ("a vetro smerigliato", "parzialmente solida"), mai col numero.

### Leggere come un radiologo (20 ottobre): finestra, parti della lesione, assi

Domanda di Frank: come discrimina un radiologo, come interpreta un'immagine? Traduzione in passi uguali per ogni organo (`imaging/radiologist_reading.py`), registrati accanto alla regione del segmentatore, senza ancora decidere nulla:

1. **Finestra dall'organo che contiene la lesione.** In un lobo polmonare la finestra polmonare (livello -600 HU), altrove quella dei tessuti molli (livello 40 HU). Il bordo è dove la densità attraversa il livello: il grigio medio, dove l'occhio mette il margine (Lung-RADS misura i noduli in finestra polmonare). Se il livello non sta tra la lesione e ciò che la circonda, il bordo va a metà strada tra le due.
2. **Cosa non è lesione anche se ha la stessa densità.** Ogni altra struttura nominata dalla segmentazione dell'intero corpo (vasi, cuore, parete, ossa, altri organi) è esclusa, salvo quelle in cui la lesione già entra (almeno il 5% della lesione). I vasi sottili del polmone, che la segmentazione non nomina, sono tagliati da un'apertura nel piano di 2 mm.
3. **Le parti, nel polmone aerato.** `solid`: la parte connessa più densa di -300 HU. `total`: la parte solida più il vetro smerigliato continuo (più denso del polmone attorno di max(100 HU, 3 volte il rumore)). Lung-RADS dà due misure al nodulo parzialmente solido: totale e componente solida.
4. **Le misure come le prende lui.** Diametro assiale maggiore, asse corto perpendicolare sulla stessa fetta, media dei due (Lung-RADS), per ogni lettura. Asse corto e media entrano anche nel ramo del grafo di ogni lesione (`short_axial_mm`, `mean_axial_mm`).

Nel probe (`--anatomy`) ogni caso riporta le tre letture con Dice e misure contro il contorno di riferimento, e una tabella per fascia di dimensione: dirà quale bordo segue il riferimento di MSD (quello di VoxTell, quello della finestra o quello con il vetro smerigliato). Non serve a tarare nulla sul riferimento: la taratura del bordo "da radiologo" va fatta sui contorni di quattro radiologi per nodulo di LIDC-IDRI.

### Rilettura su CPU (20 ottobre): la GPU solo quando cambia la segmentazione

La quota GPU settimanale di Kaggle (30 ore) è finita il 20 ottobre, prima del giro con le letture da radiologo. Quelle letture, come zone e margini, partono dalla regione di VoxTell e dalle strutture di TotalSegmentator: non serve rifare la segmentazione per provarle.

- `--save-masks DIR`: il giro con GPU salva per ogni caso un file `npz` con la regione di VoxTell per ogni prompt e la mappa delle strutture dell'intero corpo (o le maschere degli organi nominati).
- `--from-masks DIR`: nessun segmentatore, nessuna GPU; rilegge quei file e rifà tutta l'analisi. Un test controlla che la rilettura dia la stessa riga del giro originale (misure, Dice, stato, motivi, zone, letture da radiologo).
- Kaggle: il notebook con GPU salva in `/kaggle/working/masks`; il nuovo notebook `melampo-imaging-replay` (solo CPU) monta l'output dell'ultimo giro con GPU come dati in ingresso e lancia la rilettura. Da Termux: `python kaggle_api.py push replay`, `status replay`, `output replay` (risultati in `output_replay`). Il `push` ora avvisa quando Kaggle rifiuta il giro (come per la quota esaurita), invece di lasciare credere che sia partito.
- Serve un solo giro con GPU (al rinnovo della quota) per avere le maschere; da lì ogni cambiamento dell'analisi si prova su CPU.

### Undicesimo probe (25 ottobre): le letture "a soglia da radiologo" falliscono; VoxTell resta il bordo migliore

Giro con GPU dopo il rinnovo della quota, 30/30 casi, maschere salvate per tutti (`--save-masks`): d'ora in poi le prove dell'analisi si fanno su CPU. Accettati 12 (italiano 11/12 entro il limite stretto), come nel nono e decimo.

Le tre letture (finestra, solida, totale) contro il contorno di riferimento, diametro assiale maggiore:

| lettura | entro max(2 mm, 10%) | errore mediano < 30 mm | 30-50 mm | > 50 mm | Dice mediano |
|---|---|---|---|---|---|
| VoxTell (regione) | 12/30 | -1,6 | -4,6 | -12,4 | 0,85-0,89 |
| finestra (-600 HU) | 3/30 | +9,9 | +15,5 | +14,1 | 0,34-0,56 |
| solida (-300 HU) | 1/27 | -3,4 | +6,0 | +16,1 | 0,28-0,56 |
| totale (con vetro smerigliato) | 2/25 | -0,6 | +11,9 | -55,9 | 0,00-0,51 |

- **Perché falliscono.** Quasi tutte le letture arrivano al bordo della zona (10 mm): la soglia scorre lungo vasi più spessi di 4 mm, pleura e tessuti della parete che la segmentazione non nomina (+15/+26 mm su 004, 006, 015, 029, 041). Accanto al mediastino o alla parete il "tessuto intorno" è tessuto molle e la finestra polmonare non sta tra lesione e intorno (009, 014, 028, 033, 036: si ripiega su metà strada). Sulle masse grandi la lettura totale prende un'altra componente (Dice 0: 026, 027, 031, 041).
- **Cosa insegna.** È la stessa lezione della soglia fissa -400 HU del secondo probe: una regola di densità non sa quale oggetto sta misurando. Il radiologo separa la lesione da ciò che le sta attaccato con la conoscenza dell'oggetto (continuità, forma, anatomia, contrasto), ed è proprio ciò che VoxTell ha imparato: la sua regione è molto più vicina al contorno del riferimento (Dice 0,85 contro 0,3-0,5). Le letture restano registrate come diagnostica, non decidono nulla.
- **Overlay di 022 (con nome file, ora):** non è solo una corona in più. VoxTell include una parte in alto che il riferimento esclude e lascia fuori una parte bassa verso l'ilo che il riferimento include: due scelte diverse su dove finisce la massa e cominciano ilo e polmone alterato, che un radiologo decide anche col contrasto e il contesto. Su 031 i due contorni coincidono quasi ovunque; la differenza è in pochi punti alle estremità.
- **Prossimi passi proposti:** (1) rifinire il bordo di VoxTell solo localmente, in una banda di 3 mm attorno al suo contorno, al livello della finestra (nessuna fuga possibile per costruzione), provato su CPU con le maschere salvate e con il criterio di Frank (1-2 mm sulle parzialmente solide, nessun peggioramento sulle solide); (2) LIDC-IDRI: VoxTell contro i contorni di quattro radiologi per nodulo, per sapere se la sua misura sta dentro la variabilità umana (serve un giro con GPU).

### L'atlante anatomico del paziente (25 ottobre)

Proposta di Frank: per leggere come un radiologo serve un atlante anatomico di tutto il corpo, che dica quali sono gli organi e quindi cosa è lesione, utilizzabile per TC e RM, meglio in DICOM.

**Cosa c'è e cosa manca (TotalSegmentator 2.18).** L'atlante è quello del paziente stesso: ogni voxel dell'esame riceve un'etichetta. Non è completo.
- TC, compito `total` (117 strutture): cervello e cranio (un'unica struttura per il cervello), collo (trachea, esofago, tiroide, carotidi, vasi succlavi), torace (lobi polmonari, cuore, aorta, vene polmonari, vena cava superiore, coste, sterno, clavicole, scapole), addome e pelvi (organi, intestino, vasi iliaci, vescica, prostata, anche), colonna (C1-S1, midollo), arti (solo omero, femore, scapola). Muscoli: solo glutei, autoctoni e ileopsoas.
- Compiti aperti aggiunti (Apache-2.0): `lung_vessels` (arterie, vene, vie aeree e loro parete dentro il polmone), `pleural_pericard_effusion` (versamenti), `abdominal_muscles` (muscoli del tronco tra T4 e L4: pettorale, dentato, gran dorsale, trapezio, retti, obliqui, erettori), `trunk_cavities` (cavità toracica, addominale, mediastino, pericardio: contenitori, non tessuti).
- Mancano: linfonodi (in nessun compito), muscoli intercostali e della parete fuori da T4-L4, grasso e muscolo come tessuti (`tissue_types`, con licenza: gratuita per la ricerca, commerciale su richiesta), muscoli degli arti e strutture fini dell'encefalo (con licenza), peritoneo, nervi, vasi piccoli fuori dal polmone.
- RM, `total_mr` (50 strutture): polmoni interi senza lobi né vasi, cuore e grandi vasi, organi addominali, vertebre come un'unica struttura, pochi muscoli; `body_mr` dà solo tronco e arti. In RM l'atlante è molto più povero e la distinzione tra tessuti viene dalle sequenze (T1, T2, soppressione del grasso, contrasto, diffusione).
- Formato: TotalSegmentator scrive anche DICOM SEG (`--output_type dicom_seg`) e RTSTRUCT, leggibili da un PACS.

**Implementato.**
- `TotalSegmentatorSegmenter.all_structures`: compito principale più i compiti aperti in un'unica mappa (il compito più fine prevale: i lobi di `total` contengono i vasi che `lung_vessels` nomina; la classe ausiliaria `lung_pleural` è scartata) e le cavità a parte come contenitori (`regions`). Un compito che fallisce viene saltato e scritto nelle note. In RM solo `total_mr`.
- La lettura da radiologo usa l'atlante: esclude oltre la regione di VoxTell ogni struttura che non contiene la lesione (contiene = nome dell'ospite o almeno il 20% della lesione: un vaso che la attraversa ne copre anche il 13% e non per questo la contiene); tiene la lesione nella sua cavità; esclude i piani adiposi (grasso omogeneo lontano dal polmone aerato, perché il bordo tra tumore e aria passa per i valori del grasso); misura il tessuto attorno sul parenchima della struttura ospite e non più su un anello misto (che dava rumore di 240-520 HU e la lettura totale sopra la densità del tumore). Nuova lettura `window_band`: lo stesso bordo, ma solo entro 3 mm da quello di VoxTell.
- Le cavità si salvano con le maschere e tornano nella rilettura su CPU.
- Il kit di Kaggle (`kaggle_api.py`) è ora anche nel repository, in `tools/kaggle_kit/`.
- Test sintetici: vaso di 6 mm nominato, parete non nominata fuori dalla cavità, grasso mediastinico, massa di 32 mm con anello misto, banda.

**Da verificare su dati reali:** un giro di rilettura su CPU con le maschere dell'undicesimo probe (solo `total`, niente cavità) dice già quanto pesano piani adiposi, parenchima e banda; il giro con GPU con i compiti aggiunti (circa 1-2 minuti di GPU in più per caso per ciascun compito) dice quanto pesa l'atlante completo. Criterio: 1-2 mm sulle parzialmente solide, nessun peggioramento sulle solide.

### Prima rilettura su CPU con l'atlante (25 ottobre): la banda raddrizza le masse grandi, sbaglia le altre

Rilettura delle maschere dell'undicesimo probe (30/30, nessuna GPU). Le maschere salvate hanno solo il compito principale: niente vasi del polmone, niente muscoli, niente cavità. Hanno agito piani adiposi, parenchima per il tessuto attorno e banda.

| lettura (asse lungo) | entro max(2 mm, 10%) | mediana < 30 mm | 30-50 mm | > 50 mm |
|---|---|---|---|---|
| VoxTell | 12/30 | -1,6 | -4,6 | -12,4 |
| finestra (10 mm) | 2/30 | +7,0 | +15,5 | +14,6 |
| banda 3 mm a -600 HU (`window_band`) | 13/30 | -0,2 | -1,1 | -2,6 |
| solida | 2/30 | -3,4 | +6,0 | +14,6 |
| totale | 2/23 | +8,0 | +14,2 | +11,2 |

- **Il tessuto attorno ora è il parenchima:** rumore 36-194 HU invece di 240-550; la lettura totale non crolla più a un frammento (dove la sua soglia finirebbe sopra la parte solida, viene saltata e scritto perché).
- **La banda a -600 HU** porta le masse solide grandi vicino al riferimento: 028 -14,6 → -2,6 mm, 027 -8,1 → -1,6, 041 -7,2 → +1,3, 034 -6,0 → -3,5, 045 -5,3 → -1,9, 022 -17,1 → -8,0, 031 -14,2 → -8,3. Ma va oltre sui noduli solidi piccoli e medi dove VoxTell era già giusto (004 +4,3, 005 +4,2, 009 +6,4, 010 +4,4, 025 +5,0: arriva al bordo della banda, quasi certamente lungo vasi e pleura che queste maschere non nominano) e accorcia le lesioni a vetro smerigliato (016 -18,8, 042 -20,7, 043 -12,8: il grigio a -600 HU è più chiaro del vetro smerigliato stesso).
- Con il criterio di Frank (1-2 mm sulle parzialmente solide, nessun peggioramento sulle solide) la banda così com'è non passa.
- **Cosa si cambia, e come si evita di adattarsi ai 30 casi.** (1) Nuova lettura `edge`: il bordo dipende da cosa è fatta la lesione, come per il radiologo: lesione solida (mediana della regione ≥ -300 HU) al grigio della finestra polmonare; vetro smerigliato a metà tra la lesione e il polmone; sempre entro la banda. (2) Il giro con GPU con l'atlante completo (vasi, muscoli, cavità) dirà quanto dell'eccesso sui noduli piccoli viene dai vasi e dalla pleura. (3) Regola scelta su questi 30 casi, da verificare senza toccarla sui 33 casi di MSD Task06 Lung mai usati (dal 31° al 63°): `kaggle_api.py push holdout` e `push replay holdout`.

### Dodicesimo probe (26 ottobre): l'atlante completo ferma le fuoriuscite; la banda arriva a 1-2 mm sui solidi

Giro con GPU, atlante completo (110 strutture con vasi e vie aeree del polmone, versamenti, muscoli del tronco; cavità toracica, addominale, mediastino, pericardio). 29 casi su 30: lung_006 (567 fette) ucciso per memoria (-9). Accettati 11 (italiano 10/11 entro il limite stretto).

- **Le fuoriuscite si fermano.** Lettura a finestra entro 10 mm: entro tolleranza 9/29 (prima 2/30); 004 +26,0 → +9,1 mm, 005 +15,3 → +1,3, 010 +15,6 → +1,2, 015 +22,3 → +1,0, 045 +18,1 → 0,0.
- **Banda di 3 mm a -600 HU (`window_band`), per gruppo** (errore mediano assoluto del diametro maggiore, entro tolleranza):

| gruppo | VoxTell | banda |
|---|---|---|
| solide < 30 mm (7) | 0,6 mm, 6/7 | 1,0 mm, 6/7 |
| solide 30-50 mm (5) | 1,4 mm, 3/5 | 1,8 mm, 5/5 |
| solide > 50 mm (9) | 12,4 mm, 1/9 | 10,8 mm, 2/9 |
| parzialmente solide (8) | 4,7 mm, 1/8 | 8,4 mm, 1/8 |

  Sui solidi fino a 50 mm la banda sta tra 1 e 2 mm e porta dentro 034 (-6,0 → -4,2), 045 (-5,3 → -1,9), 029 (-3,2 → +1,0); esce 025 (+0,3 → +3,0). Totale entro tolleranza 14/29 contro 11/29.
- **Masse grandi:** il miglioramento della prima rilettura era in parte una fuoriuscita lungo vasi non ancora nominati: con l'atlante 028 -14,6 → -9,5 (nella rilettura -2,6), 022 → -12,9 (rilettura -8,0). Restano 027 -1,6 e 041 -0,5.
- **Parzialmente solide e lesioni a soglia frammentata:** la banda le accorcia (016 -18,8, 042 -25,2, 043 -12,8; anche 031, cavitata, -27,5): a soglia la lesione si spezza in pezzi e la lettura teneva un pezzo solo.
- **Correzioni.** (1) Le letture in banda raffinano solo verso l'esterno: la regione di VoxTell resta intera e la banda aggiunge ciò che il bordo del radiologo aggiunge, mai toglie. (2) Le serie più lunghe di 400 fette si segmentano a parti (`force_split` di TotalSegmentator), per la memoria.
- **Regola candidata, da verificare senza toccarla sui 33 casi tenuti da parte:** misura con la banda per le lesioni solide; per le parzialmente solide la regione di VoxTell. Scelta su questi 30 casi, quindi vale solo se tiene sugli altri 33.

### Rilettura su CPU della banda che non toglie (27 ottobre) e regola fissata prima della verifica

Maschere del dodicesimo probe (atlante completo), 29 casi. La banda ora tiene intera la regione di VoxTell.

| gruppo | VoxTell (errore mediano assoluto, entro tolleranza) | banda 3 mm |
|---|---|---|
| solide < 30 mm (7) | 0,6 mm, 6/7 | 1,0 mm, 6/7 |
| solide 30-50 mm (5) | 1,4 mm, 3/5 | 1,5 mm, 5/5 |
| solide > 50 mm (9) | 12,4 mm, 1/9 | 9,1 mm, 2/9 |
| parzialmente solide (8) | 4,7 mm, 1/8 | 4,5 mm, 2/8 |

- Entro tolleranza 15/29 contro 11/29; spariti gli accorciamenti (016, 042, 031 tornano al valore di VoxTell). Sui solidi fino a 50 mm la banda sta tra 1 e 2 mm (004 +1,0, 029 +2,0, 034 -3,6, 045 -1,5; esce solo 025, +3,0).
- **Parzialmente solide:** la banda a metà strada non aggiunge nulla (016 -3,0, 020 -3,8, 023 -5,1, 042 -10,7, 043 -5,5, 044 -4,8). Il riferimento include vetro smerigliato tenue, 60-180 HU sopra il polmone, sotto quella soglia.
- **Masse grandi:** restano 8-13 mm corte (022, 028, 033, 036): più di quanto una banda di 3 mm per lato possa restituire.

**Modifiche.** Per il vetro smerigliato il bordo è dove la lesione smette di distinguersi dal polmone: max(100 HU, 3 volte il rumore) sopra il parenchima, entro la banda. Lettura `edge_wide` (banda di 6 mm) solo esplorativa per le masse grandi: non sarà mai la misura senza una verifica a parte.

**Regola fissata ora, prima di vedere i 33 casi tenuti da parte** (`push holdout`), e da non cambiare dopo averli visti: la misura è la lettura `edge` (banda di 3 mm che non toglie nulla alla regione di VoxTell; bordo al grigio della finestra polmonare per le lesioni solide, al limite di visibilità sul polmone per il vetro smerigliato). Criterio di successo sui 33: errore mediano tra 1 e 2 mm sulle lesioni solide fino a 50 mm e sulle parzialmente solide, nessun peggioramento rispetto a VoxTell sulle altre; entro tolleranza almeno quanto VoxTell. Se non succede, la regola si scarta.

### Rilettura del 27 ottobre (sera): il bordo del vetro smerigliato cambia poco

29 casi, maschere con l'atlante. Errore mediano assoluto del diametro maggiore (entro tolleranza):

| gruppo | VoxTell | banda -600 | `edge` (la regola fissata) | `edge_wide` 6 mm |
|---|---|---|---|---|
| solide < 30 mm (7) | 0,6 (6) | 1,0 (6) | 0,6 (6) | 3,2 (3) |
| solide 30-50 mm (5) | 1,4 (3) | 1,5 (5) | 1,5 (5) | 1,5 (4) |
| solide > 50 mm (9) | 12,4 (1) | 9,1 (2) | 9,1 (2) | 9,1 (4) |
| parzialmente solide (8) | 4,6 (1) | 4,6 (2) | 4,4 (2) | 4,7 (1) |

- Entro tolleranza: VoxTell 11, `edge` 15, `edge_wide` 12 su 29.
- Il nuovo bordo del vetro smerigliato sposta poco: 020 -3,8 → -3,5, 043 -5,5 → -4,5, 038 -1,2 → -0,6; gli altri restano. Il vetro smerigliato tenue che il riferimento include sta dentro il rumore del polmone voxel per voxel (60-180 HU sopra il parenchima contro un rumore di 40-70 HU): una soglia sul singolo voxel non lo separa. L'occhio del radiologo lo vede perché integra su un'area; la prova possibile è una soglia sull'immagine mediata, da tenere esplorativa.
- La banda di 6 mm recupera alcune masse grandi (036 -0,3, 022 -6,5) ma va oltre su quasi tutte le lesioni piccole (+4,7…+6,5): resta solo esplorativa.
- Con il criterio di Frank la regola fissata va bene sulle solide fino a 50 mm (0,6 e 1,5 mm), non sulle parzialmente solide (4,4 mm) né sulle masse grandi. La verifica sui 33 casi tenuti da parte dirà se il risultato sulle solide regge su casi mai visti.

### Verifica sui 33 casi tenuti da parte (28 ottobre): la regola migliora VoxTell ma non raggiunge 1-2 mm

Giro con GPU sui casi lung_046-lung_096 (mai usati per scegliere nulla), atlante completo, 32 casi su 33 (lung_047 ucciso per memoria anche con la segmentazione a parti). Gruppi: "non solide" = composizione del riferimento parzialmente solida o vetro smerigliato. Errore mediano assoluto del diametro maggiore (entro tolleranza):

| gruppo | VoxTell | `edge` (regola fissata) |
|---|---|---|
| solide < 30 mm (6) | 3,1 (1) | 2,5 (2) |
| solide 30-50 mm (9) | 4,1 (3) | 4,1 (3) |
| solide > 50 mm (6) | 6,1 (3) | 2,7 (5) |
| non solide (11) | 3,1 (3) | 2,4 (5) |

**Esito rispetto ai criteri fissati il 27 ottobre:**
- errore mediano tra 1 e 2 mm sulle solide fino a 50 mm: **no** (2,5 e 4,1 mm);
- tra 1 e 2 mm sulle parzialmente solide: **no** (2,4 mm);
- nessun peggioramento rispetto a VoxTell: **sì** (la banda non toglie mai, migliora o lascia uguale in ogni caso);
- entro tolleranza almeno quanto VoxTell: **sì** (15 contro 10 su 32).

Per la regola stabilita prima del giro, la lettura `edge` non diventa la misura. Resta una lettura registrata che sui casi nuovi migliora VoxTell senza mai peggiorarlo (soprattutto sulle masse grandi, 5/6 entro tolleranza): se usarla come misura pur senza raggiungere 1-2 mm è una decisione di Frank.

**Cosa insegna la verifica.**
- I 30 casi di lavoro erano favorevoli a VoxTell: sui solidi fino a 50 mm il suo errore mediano era 0,6-1,4 mm, sui casi nuovi 3,1-4,1 mm. Una regola scelta su 30 casi non si trasferisce intera: la verifica su casi mai visti era necessaria.
- Sui solidi 30-50 mm dei casi nuovi la banda non aggiunge nulla (VoxTell corto di 3-13 mm): il bordo del riferimento sta sotto il grigio della finestra o oltre i 3 mm.
- **Precisione degli accettati:** 10 accettati, tutti entro il limite stretto sulla misura; ma uno, lung_096, è un nodulo diverso (Dice 0) nel lobo superiore destro, della stessa misura (12,0 mm) del nodulo di riferimento nel lobo inferiore destro. Il referto sintetico diceva solo "Formazione polmonare di 1,2 cm": la misura da sola lo ha fatto passare. Precisione vera 9/10.
- **Correzione:** con `--report-site` (ora nei notebook) il referto sintetico nomina il lobo del riferimento, come fa un referto vero ("Formazione nel lobo inferiore destro di 1,2 cm"), e il controllo si fa su quel lobo: un nodulo in un altro lobo va a una persona. Test sintetico con due noduli uguali in due lobi.
- **Prossimo passo per il bordo:** LIDC-IDRI, con i contorni di quattro radiologi per nodulo, per sapere quanto variano i radiologi stessi e se 1-2 mm è raggiungibile contro un solo contorno di riferimento.
