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
