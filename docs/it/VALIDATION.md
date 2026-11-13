# Validazione e provenienza dei dati

*[English](../en/VALIDATION.md) · Italiano · aggiornato al 4 novembre 2026*

Melampo afferma soltanto ciò che le sue probe hanno misurato, su campioni su cui nessuno ha tarato il sistema e
con criteri scritti prima del run. Oggi niente è certificato: il gold set dei radiologi non è ancora etichettato,
e le etichette usate finora sono di un solo etichettatore (Claude) e aspettano la revisione di un radiologo.

## 1. Metodo

1. **Un campione nuovo per ogni misura che conta.** I referti già visti durante lo sviluppo non contano più:
   ogni campione si estrae con il suo seed ed esclude tutti i referti dei campioni precedenti
   (`scripts/owner_probe.py sample --seed …`).
2. **Le etichette prima di ogni lettura.** Chi etichetta vede la frase, le parole marcate, la classe collegata e
   l'esame, mai le risposte dei lettori. Le etichette in dubbio finiscono con "?" e vanno a un radiologo.
3. **I criteri prima del run**, in questo file o nella pull request che aggiunge la probe.
4. **Una modifica che non passa il suo criterio resta spenta**, e il risultato si scrive qui. Un run che perde
   più del 5% delle risposte dei modelli (`lost_answers`) non conta.
5. **Nessuna taratura sul riferimento.** Un campione usato per trovare un errore diventa dato di sviluppo; la
   correzione si misura sul campione nuovo successivo.
6. **Due lettori che devono essere d'accordo.** Un lettore deterministico e i modelli linguistici; due modelli
   contano come un solo flusso; una persona decide sulle contraddizioni. L'accordo si misura su ciò che cambia
   l'esito.

## 2. Provenienza dei dati

I testi dei corpora pubblici non si committano mai: solo le etichette, i seed e gli script che estraggono i
campioni. Licenza e stato di ogni risorsa esterna (ontologie, corpora, pesi dei modelli) sono in
[`melampo-assets.yaml`](../../melampo-assets.yaml); i pacchetti Python sono in [`uv.lock`](../../uv.lock) e
nell'[elenco SOUP](../registers/soup.md) generato.

| Fonte | Usata per | Nel repository |
|---|---|---|
| PARROT v1.0 (referti radiologici in più lingue, con traduzione inglese; CC BY-NC-SA 4.0) | campioni del finding owner, dei tipi e delle affermazioni; campioni HPO | etichette in `data/gold/*_2026-1*` |
| E3C (casi clinici italiani e inglesi) | gold set silver, role probe | etichette in `data/gold/silver_2026-10-12` |
| IU X-ray e Rad-SpRL (ruoli spaziali, CC BY 4.0) | role probe, ruolo di riferimento | etichette in `data/gold/rad_sprl_2026-10-12` |
| CRAFT, MedMentions | controllo esterno del linker | lo split di test di MedMentions è congelato: `data/linking/frozen_test_split.json` |
| MSD Task06 Lung, LIDC-IDRI | probe di imaging su Kaggle | notebook in `notebooks/` |
| HPO, MONDO, DOID, LOINC, CIPI, ICD-9-CM; UBERON, PATO | lettura, linking, grafo | pin DVC in `data/*.dvc`; UBERON e PATO scaricati dai workflow all'ultima release |

**Campioni della domanda «di chi è il reperto»** (tutti item inglesi di PARROT, 200 ciascuno salvo dove detto):

| Campione | Seed | Esclude | Etichette | Stato |
|---|---|---|---|---|
| HPO di sviluppo | default | — | `data/gold/hpo_silver_2026-10-31` | visto |
| HPO di verifica | 2026103102 | HPO di sviluppo | `data/gold/hpo_heldout_2026-10-31` | visto |
| owner (209 item, 9 italiani) | 20261101 | entrambi gli HPO | `data/gold/finding_owner_2026-11-01` | visto |
| statement | 20261102 | i precedenti | `data/gold/statement_2026-11-01` | visto |
| claims | 20261103 | i precedenti | `data/gold/claims_2026-11-02` | visto |
| claims2 | 20261104 | i precedenti | `data/gold/claims_2026-11-02b` | visto |
| claims3 (174 item) | 20261105 | i precedenti | `data/gold/claims_2026-11-03` | visto |
| claims4 (121 item, fino a 16 per struttura) | 20261106 | i precedenti | `data/gold/claims_2026-11-04` | visto |
| ctrate (200 item, TC torace di CT-RATE) | 20261107 | — (un altro corpus) | `data/gold/ctrate` | **nuovo, etichettato con la convenzione, non ancora lanciato** |

## 3. Il gold set dei radiologi

Il certificato a cui punta il progetto: «errore ≤ 1% sui link accettati, confidenza 95%» (Clopper-Pearson: 299 link
accettati con 0 errori, 628 con ≤ 2, 1.049 con ≤ 5, 1.941 con ≤ 12). Obiettivo: 2.000–2.300 menzioni da referti
veri pseudonimizzati, una menzione per referto, due radiologi che etichettano da soli e in cieco, un terzo che
decide i disaccordi, l'insieme congelato una volta etichettato. Come si etichetta è nella
[guida per i radiologi](RADIOLOGIST_GUIDE.md); i comandi sono `scripts/gold_set.py sample | check | agree | merge
| evaluate`. Dopo l'avvio, un campione cieco del 2–5% dei link accettati tiene aggiornato il limite.

### La convenzione di etichettatura (6 novembre 2026)

Finché non etichettano i radiologi, i campioni sono etichettati da un solo etichettatore (Claude). I casi dubbi si
decidono con regole scritte, ciascuna con la sua fonte, non con il giudizio del momento; le regole si fissano prima
di etichettare un campione e non si correggono mai guardando le risposte dei modelli.

| Caso | Regola | Fonte |
|---|---|---|
| una variante (milza accessoria, lobo di Riedel) | una variante che l'HPO tiene come fenotipo di una struttura è un reperto di quella struttura | HPO (es. HP:0001747) |
| mesentere, pleura parietale, bronco lobare | una struttura è parte di un'altra solo se UBERON lo dice (`part_of`) | UBERON |
| ciò che dicono l'impegnativa o il quesito clinico | un'indicazione, mai un reperto attuale | le sezioni del referto (LOINC 18785-6) |
| un'embolizzazione, una resezione | una procedura: storia | — |
| una valvola impiantata, un tubo, uno stent | un dispositivo, anche quando «se ne vede l'ombra» | — |
| «a livello di», «in proiezione di» | collocano, salvo che la frase dica che la struttura è toccata | indicatori spaziali di Rad-SpRL |
| «potrebbe», «non si esclude», una diagnosi differenziale (7 novembre 2026) | il dubbio del referto: il reperto con l'affermazione `uncertain` (`finding+uncertain`), tenuto nel percorso diagnostico | il principio del dubbio |
| «compatibile con», «consistent with», «compatible with» (11 novembre 2026, fino ad allora letti come certi) | un'incertezza di livello alto, oltre il 90%: `finding+uncertain_high`; ogni livello più basso («probabilmente», «sospetto per», «possibile», «?», «in favore di», «potrebbe») è `uncertain` | Panicek e Hricak, AJR 2016;207:2-3 |
| «aspecifico», «non caratterizzabile», «per quanto osservabile», «in favore del cuore», «lieve» (11 novembre 2026) | non un'incertezza: ciò che si vede è affermato, la sua natura è aperta (non determinata), il limite dell'esame, un'attribuzione, un grado | la frase afferma ciò che si vede |
| alternative unite da «o», una condizione («se presente»), «da valutare per», «considerato» (11 novembre 2026) | un'incertezza, sotto il livello alto | BioScope (Vincze et al. 2008): «o» indizio di speculazione, «non si esclude» una speculazione, non una negazione; ConText (Harkema et al. 2009): ipotetico |
| un approssimatore («circa», «-simile», «lieve»), una fonte («secondo», «valutato insieme all'ecografia, si è compreso che»), un cambiamento nel tempo («stabile», «nuovo», «ridotto») (11 novembre 2026) | non un'incertezza: gli approssimatori cambiano ciò che si dice, non l'impegno su di esso; una fonte dice da dove viene la conoscenza; il tempo è a parte | Prince, Frader e Bosk 1982 (approssimatori: adattatori, arrotondatori; scudi: di plausibilità, di attribuzione); Hyland 1996 (attenuatori di attributo e di affidabilità); FactBank (Saurí e Pustejovsky: la fattualità è l'impegno di una fonte); ConText (temporalità) |

Un'etichetta che le regole lasciano ancora aperta è segnata "?". **I criteri si calcolano sugli item senza "?"**
(`by_set.<campione>.criteria.without_doubt`); gli item "?" sono riportati a parte, con il loro esito
(`criteria.all`, `claims_kinds.for_a_radiologist`).

### Il principio del dubbio (7 novembre 2026)

Ciò che non è certo non è una domanda per una persona, che non ne sa più di quanto dica il referto: resta nel
percorso diagnostico come **reperto in dubbio** (verdetto `doubt`), con ogni lettura accanto, mai messo da parte.
Altri referti, l'anamnesi o un esame nuovo possono chiuderlo in seguito; dentro il referto, le conclusioni o la
conoscenza non ambigua del lettore (HPO, PATO) possono chiudere un dubbio di lettura. Un reperto è certo quando lo
sostengono due fonti di meccanismo diverso: i due modelli d'accordo (un solo flusso) e il lettore che non li
contraddice, oppure la lettura di un modello confermata dalle conclusioni o dal lettore. Alla persona va solo una
**discordanza del documento**: il contesto e la frase (un reperto letto nell'impegnativa), la frase e le
conclusioni, un «suo» che rimanda a un altro organo, una misura e il suo riferimento citato. Il dubbio che il
referto stesso dichiara resta con il reperto (`clinical`); la nostra lettura non lo chiude mai.

I tipi di dubbio: `reading` (i modelli leggono diversamente), `state` (un modello non sa se differisce dal
consueto), `site` (una lettura lo dà a un'altra struttura: resta dove l'ha messo il linker), `one_reading` (una
sola risposta), `reader` (il lettore legge diversamente dai modelli), `clinical`.

## 4. Risultati

Dal più recente. Il dettaglio di ogni run è nel documento indicato nell'ultima colonna, nel tag
`docs-archive-2026-11` (percorso `docs/linker/…`) o nella pull request.

| Data | Domanda | Risultato | Decisione | Dettaglio |
|---|---|---|---|---|
| 2026-11-11 | Affermazione per affermazione, campione `ctrate5` (300 item, etichettati prima; owner-probe_20) | valido. Sui 293 item senza dubbio: (1) 0; (2) 0; (3) 0%; (4) 0,72% (2: «in the old thorax CT … were observed» e «a patient with gallbladder carcinoma» letti come reperti di oggi); (6) 0% senza la lettura giusta, 0 dei 2 dubbi chiusi sbagliati; (7) 0,68% (2: una menzione tagliata dal linker, «left kidney (cyst»; «suspicious density consistent with» letto come alto); (8) 5,8% come etichettato, 4,8% (14) dopo la correzione di 3 etichette (l'etichettatore aveva letto una frase tagliata, senza il suo «(metastasis?)», e non aveva visto che una frattura patologica è fra le lesioni «consistent with metastasis»). Per lettura: un modello 1 (0,34%: una fonte, «it was understood that», letta come livello), l'altro 13 (4,4%: «nonspecific», «cannot be characterized», «as far as can be observed» messi sia al loro posto sia nell'incertezza, 10; «stable» letto come probabile, 2; «ground-glass-like», 1); l'unione 14, e non ha tenuto nessuna incertezza che un modello da solo perdesse | la domanda distingue l'incertezza come la letteratura sugli attenuatori: solo gli scudi (di plausibilità), non approssimatori, fonti, tempo, natura aperta o limite dell'esame; una parola in un posto solo, e nel codice le parole che una risposta mette fra le non determinate o i limiti non sono la sua incertezza (non contraddizione). La lettura dell'incertezza (ciascun modello, o la loro unione: `MELAMPO_UNCERTAINTY_FROM`) si sceglie su `ctrate6` con una regola fissata prima; la lettura per affermazioni resta spenta | `ctrate5_risultati_2026-11-11` |
| 2026-11-11 | Affermazione per affermazione, campione `ctrate4` (300 item, etichettati prima; owner-probe_18) | valido. Sui 293 item senza dubbio: (1) 0; (2) 0; (3) 1,37% alla persona (4: 3 discordanze del referto con il valore di riferimento, come vuole il principio, 1 menzione spezzata dal campionatore); (4) 0,37%; (6) 0,34% senza la lettura giusta, 1 dei 4 dubbi chiusi sbagliato (43 mm letto come reperto: la frase dà solo la misura, il riferimento la mette sopra la norma); (7) l'incertezza persa in 1 item (0,34%; 11 su ctrate3). Fuori dai criteri: l'incertezza data a 62 dei 250 reperti che il referto afferma (24,8%; 40 da entrambi i modelli): «nonspecific», «cannot be characterized» (31), «consistent with», «compatible with» (16), i limiti dell'esame (4), «in favor of the heart» (4), un grado o una qualità (7). Ciò che non è incertezza non aveva un posto suo, e «nel dubbio, è incerto» ve lo spingeva | ogni incertezza con il suo livello sulla scala di Panicek e Hricak; ciò che si vede ma non è identificato (non determinato) e i limiti dell'esame in posti propri; attribuzione e grado restano in ciò che si vede; tolto «nel dubbio». «Consistent with» è un'incertezza di livello alto (deciso l'11 novembre 2026). Criterio (8): l'incertezza data dove il referto afferma, sotto il 2%. Misurato su `ctrate5`; la lettura per affermazioni resta spenta | `ctrate4_risultati_2026-11-11` |
| 2026-11-10 | Affermazione per affermazione, campione `ctrate3` (300 item, etichettati prima; owner-probe_16) | valido (0 su 600 risposte perse). Sui 287 item senza dubbio: (1) 0 «a parte» attaccati come certi (limite al 95% 1,04%); (2) 0 reperti messi da parte; (3) 0% alla persona (limite 1,04%); (4) 0% delle letture certe attaccate diversamente (limite 1,12%); (6) 7,7% in dubbio, nessun dubbio senza la lettura dell'etichetta, 0 dei 3 dubbi chiusi sbagliati. La domanda sul proprietario sugli stessi item: 11 reperti alla persona, 1 «a parte» mancato. Fuori dai criteri: l'incertezza del referto persa in 11 dei 43 item etichettati incerti (3,8% degli item): «una lesione ipodensa (cisti?)» dice una lesione affermata e una cisti ipotetica; la domanda chiedeva se «ciò che è osservato» è solo probabile (non lo è), un'ipotesi non stava in nessuna delle due liste, e una lista restava solo se tutti i modelli la riempivano. In 5 i due modelli leggono il dubbio in liste diverse, in 4 lo legge un modello, in 2 nessuno (l'interpretazione copiata in ciò che si vede); una volta un'ipotesi è diventata un fatto («Hydatid cyst … ?») | l'incertezza è una sola categoria, ogni voce con le parole a cui si riferisce; ciò che si vede è separato da ciò che la frase dice che è; ciò che un modello legge come incerto resta incerto; anche ciò che si dice di un'altra persona e ciò che il paziente riferisce. Criterio (7): l'incertezza del referto persa sotto il 2% degli item. Misurato su `ctrate4`; fino ad allora la lettura per affermazioni resta spenta | `ctrate3_risultati_2026-11-09` |
| 2026-11-09 | Misura del router con i modelli (router-probe_2; 270 testi etichettati prima) | un interprete mancato: soli indizi 20,4%; Gemma 4 1,1% (limite al 95% 2,9%), Nemotron 3 Ultra 0,4% (limite 1,7%); nessuna risposta inutilizzabile. Gli errori sono referti di EEG e di polisonnografia, che le etichette chiamano note cliniche e i modelli referti di imaging (non parte il lettore delle tabelle, che non avrebbe nulla da leggere) o, una volta, altro (non parte il lettore dei referti). Nel 16-17% delle decisioni resta annotato che modello e indizi differiscono (una nota nello smistamento, non una domanda a una persona) | passano entrambi: il modello locale del router **acceso** quando il server locale del lettore è configurato (`MELAMPO_REPORT_ROUTER=cues` lo spegne). Poi (stesso giorno, rigiocato dalle risposte del run): un testo che un modello chiama «altro» si legge comunque come narrativo, e i modelli si interrogano insieme unendo i tipi: Nemotron da solo 0 mancati (limite al 95% 1,1%), i due uniti 1 (il lettore delle tabelle su un'impressione di EEG, senza nulla da leggere), Gemma da solo 3. Senza `MELAMPO_ROUTER_MODEL` si interrogano tutti i modelli locali | questa pull request |
| 2026-11-09 | Affermazione per affermazione con il principio del dubbio, campione `ctrate2` (300 item, etichettati prima; owner-probe_13) | valido (0 su 600 risposte perse). Sui 294 item senza dubbio: (1) 2 «a parte» attaccati come certi (non passa: 0); (2) 0 reperti messi da parte; (3) 1,7% alla persona (limite al 95% 3,5%); (4) 0% delle letture certe attaccate diversamente (limite 1,1%); (5) 0 risposte perse; (6) 4,4% in dubbio, nessun dubbio senza la lettura dell'etichetta, 1 dei 7 dubbi chiusi sbagliato (non passa: sotto il 2%). La chiusura sbagliata e uno dei due «a parte» erano lo stesso item: le conclusioni hanno chiuso un dubbio su di chi è il reperto (un versamento pleurico letto a livello di un lobo). Due dei cinque alla persona erano errori di un modello su una misura, non discordanze | le conclusioni chiudono un dubbio sull'esserci di un reperto, mai su di chi è; un errore di un modello su una misura (una riga non data, la misura di una lesione contro quella dell'organo) è un dubbio; solo il giudizio della frase contro il riferimento citato è una discordanza. Rigiocato sulle risposte conservate: 1 «a parte» attaccato come certo (i due modelli leggono come dello sterno l'aria nel grasso sottocutaneo davanti allo sterno), 0,7% alla persona, 0% attaccato diversamente, 6,8% in dubbio, 0 dei 2 dubbi chiusi sbagliati. La regola dei luoghi data ai modelli: dei 14 «a parte», 11 tenuti in dubbio con la lettura giusta, 1 messo da parte. Ormai visto: lo misura un terzo campione CT-RATE | questa pull request |
| 2026-11-08 | Misura del router, i soli indizi (etichette scritte prima, 270 testi) | 20,4% dei testi con un interprete mancato (limite superiore al 95% 24,8%): il lettore delle tabelle per 46 note cliniche che gli indizi leggono come soli referti di imaging (citano una RM o una radiografia), il lettore di anatomia patologica per 9 degli 11 testi di anatomia patologica; 14% con un interprete aggiunto | il riferimento con cui si misura il modello locale; segue il run con i modelli | questa pull request |
| 2026-11-08 | Perché si attaccano come certi item «a parte», e perché la lettura che guida un dubbio è spesso sbagliata (run già visti) | Degli 11 «a parte» attaccati come certi su claims2/3/4 (con quelli "?"): 4 sono milze accessorie, reperti della milza secondo la convenzione del 6 novembre, etichettati prima di essa (non rietichettati: i campioni sono visti); 7 sono collocati da parole di luogo («a livello di», «alla base di», «in sede di», «proiettato su») o detti di un altro soggetto («la sua origine dall'aorta» del tripode celiaco): la convenzione che li colloca era data all'etichettatore, non ai modelli. Dei dubbi, le risposte PARROT sono state date con le domande di prima di «differs» e della misura a parte (stati normali e misure elencati come osservati); su CT-RATE, con le domande attuali, la lettura che guida è giusta in 5 su 6; nei dubbi del lettore la sua lettura era spesso quella giusta. La lettura dell'etichetta è fra quelle che un dubbio tiene in tutti gli item tranne 1 su 448 (0,2%) | ai modelli si dà la regola della convenzione sulle parole di luogo e sulle varianti (`CLAIM_STEPS`, misurata su `ctrate2`); un pezzo che è solo una misura è una misura; la lettura del lettore resta nel dubbio e nessuna guida; il criterio 6 conta come errore un dubbio senza la lettura dell'etichetta (sotto il 2%) | questa pull request |
| 2026-11-07 | Il principio del dubbio, su run già visti (le decisioni rifatte sulle risposte conservate) | CT-RATE (199 item): 0 alla persona (limite superiore al 95% 1,5%), 1,0% delle letture certe attaccate diversamente (limite 3,2%), 0 «a parte» attaccati come certi, 0 reperti messi da parte, 3,0% in dubbio, 24 dubbi del referto tenuti con il loro reperto. PARROT claims2/3/4: alla persona 1,1%, 0,6%, 4,6% (era 17,5%, 29,5%, 26,7%), letture certe attaccate diversamente 1,9%, 1,6%, 2,1%, «a parte» attaccati come certi 2, 2, 2 (i due modelli d'accordo nell'errore; su claims4 due andavano alla persona solo perché l'impegnativa era letta fino alla fine del referto), in dubbio 9,9%, 17,6%, 4,6%, la lettura che guida un dubbio sbagliata in 13 su 18, 21 su 28, 5 su 5. Nessun criterio risulta sotto il 2% con confidenza al 95% su questi campioni | principio del dubbio **acceso** nella lettura per affermazioni (che resta spenta nell'ingestione); una sezione prima dei reperti è certa solo fino alla sua riga o alla prima frase. Campione `ctrate2` (300 item) con i criteri sotto il 2% | questa pull request |
| 2026-11-07 | Affermazione per affermazione, campione `ctrate` (TC torace, CT-RATE) | valido (0 su 400 risposte perse); per la prima volta tutti e cinque i criteri soddisfatti: 0 su 1 attribuiti in silenzio, 0 reperti messi da parte, 16 su 198 alla persona (8,1%), 1,1% attaccati diversamente. Dei 16: 7 disaccordi fra i modelli, 5 dei quali annidati (un modello legge un pezzo che l'altro tralascia); 9 da controlli del lettore che richiamavano un senso estraneo al referto (il setto-mascellare degli anfibi per "corpus", le arterie embrionali degli archi per "aortic arch") o leggevano due sedi parallele come una | sensi secondo il contesto: il lettore non richiama sensi di altre specie e quelli dell'embrione solo in un referto sullo sviluppo (corretto il 9 novembre 2026: escludeva ogni struttura embrionale), la conclusione e "its" contano solo quando nominano un altro organo, "its" solo con un pronome, le sedi parallele tengono ciascuna il proprio reperto; il lettore come terzo lettore del pezzo su cui le letture annidate differiscono (**proposta**, cambia "i modelli devono concordare"). Sullo stesso run, già visto: 5 su 198 alla persona (2,5%), nient'altro cambiato. (Corretto lo stesso giorno: le letture per affermazioni di claims2/3/4 non si possono rigiocare dalla cache, perché i loro prompt sono cambiati da allora; sono rigiocate dalle risposte conservate nella riga successiva.) Da misurare su un secondo campione CT-RATE | questa pull request |
| 2026-11-05 | Affermazione per affermazione, campione `claims4` | valido (0 risposte perse). 3 su 7 attribuiti in silenzio, tutti milze accessorie, che l'HPO tiene come fenotipo della milza (HP:0001747): molto probabilmente etichette mie; 0 su 74 reperti messi da parte; 15 su 114 alla persona (13,2%); 7,1% attaccati diversamente, 4 dei quali frasi dell'impegnativa lette come reperti attuali. 38 misure confrontate con un riferimento citato, 4 mandate alla persona dal ricalcolo del lettore, tutte a ragione | **spenta**; il referto letto come documento (sezioni, conclusioni, frase precedente), indicazione e procedura come affermazioni, i dispositivi come dispositivi, la dimensione di una lesione mai confrontata con il riferimento dell'organo, le parole di collocazione ambigue lette solo dai modelli; la convenzione di etichettatura, criteri sugli item senza "?" | questa pull request |
| 2026-11-04 | Affermazione per affermazione, campione `claims3` | valido (0 risposte perse). 4 su 17 attribuiti in silenzio (criterio 1 mancato), 0 su 110 reperti messi da parte, 33 su 157 alla persona (21%, criterio 3 mancato), 2,4% attaccati diversamente, nessuno uno stato normale (claims2: 4 su 5). Alla persona: 13 misure e tabelle, confrontate dai modelli con valori ricordati; 2 fratture costali lette come assenze (la negazione letta sull'ultima parola); 1 dall'estensione del controllo del lettore; 5 etichette "?"; il resto letture vecchie del lettore e disaccordi dei modelli | **spenta**; negazione letta sulla testa dell'osservazione; la biblioteca di riferimento: una misura si confronta solo con un valore citato, ricalcolato dal lettore | questa pull request |
| 2026-11-03 | Affermazione per affermazione, campione `claims2` | valido (0 risposte perse). 5 su 17 attribuiti in silenzio (criterio 1 mancato), 0 su 134 reperti messi da parte, 13 su 182 alla persona (7,1%), 3,0% attaccati diversamente. Dei 5: due in cui il modello diceva «è suo» mentre le sue stesse parole collocavano soltanto il reperto («beneath», «in the topography of»); due in cui la domanda diceva ai modelli che «at the level of» raggiunge la vertebra; un'etichetta "?" («lucency at the base of the heart - pneumothorax?») | **spenta**; correzioni: un modello le cui parole collocano soltanto ciò che dice proprio della struttura si contraddice e va alla persona, «beneath», «underneath» e «in the topography of» letti come parole che collocano, la domanda non dice più che un livello raggiunge la struttura. Sullo stesso run, già visto: 3 su 17 in silenzio, 0 tolti, 13 alla persona. Campione `claims3` | questa pull request |
| 2026-11-02 | Affermazione per affermazione, campione `claims` | run non valido: credito finito (Nemotron ha perso 47 risposte su 200). Sui 153 item con le due risposte: 2 su 5 attribuiti in silenzio, 0 reperti giusti tolti, 21% alla persona, 4,3% attaccati diversamente | **spenta**; correzioni (parti lette con UBERON, domanda sulle parole marcate); campione `claims2` | `affermazione_per_affermazione_2026-11-02` |
| 2026-11-02 | Role probe dopo la correzione del riferimento | «sede o no»: 0 item con lettore e modello d'accordo su un esito sbagliato; ruoli fini di Gemma 86,0% (semplice) e 87,6% (metodo), erano 89,4% e 91,5% | tenuta; lo strato Rad-SpRL va rietichettato | `che_cosa_dice_la_frase_della_struttura_2026-11-01` |
| 2026-11-02 | Tipi di affermazione, campione `statement` | 3 su 15 attribuiti in silenzio, 0 tolti, 20% alla persona; quando accettato il tipo era sempre giusto (0 su 137) | **spenta** | `che_cosa_dice_la_frase_della_struttura_2026-11-01` |
| 2026-11-01 | Finding owner v1, campione nuovo | 1 su 24 attribuito in silenzio (erano 11 su 26 sui casi visti), 0 su 184 reperti giusti tolti, 19 alla persona (10,3%, criterio 10%) | tenuto acceso nell'acquisizione; criteri 1 e 3 mancati di un caso ciascuno | `di_che_cosa_parla_la_frase_2026-10-31` |
| 2026-10-31 | Reperto → termine HPO, campione di verifica | braccio integrato 90,1% sicuro in inglese, 90% in italiano; soglia 95% | non entra nel grafo | `dal_reperto_al_termine_hpo_2026-10-31` |
| 2026-10-30 | Tolleranza delle misure | presa dalla variabilità fra radiologi (LIDC), al posto di max(2 mm, 10%) | in uso | `tolleranza_dai_radiologi_2026-10-30` |
| 2026-10-16 | Imaging su Kaggle (VoxTell + TotalSegmentator, polmone MSD, 5 casi) | primo run completo; seconda lettura delle immagini | base dell'interprete d'immagine | `primo_probe_kaggle_e_seconda_lettura_2026-10-16` |
| 2026-10-14 | Disaccordi in inglese (role probe) | 10% (Nemotron 3 Ultra) e 13% (Gemma 4) dei 484 item alla persona | opzione b: accordo su ciò che cambia l'esito | `disaccordi_inglesi_2026-10-14` |
| 2026-10-13 | Role probe, run completo | 963 item, cinque strati, tre modelli | metodo tenuto | `role_probe_run_completo_2026-10-13` |
| 2026-10-08 | Controllo esterno del linker | CRAFT 1.352 giudicati, 13 errori (99,04%); MedMentions 768 giudicati, 51 errori | analisi degli errori | `analisi_errori_controllo_esterno_2026-10-08` |

## 5. Prossima misura: un campione di TC torace da CT-RATE, e PhysioNet

In PARROT non resta un campione vario (62 item da 42 referti, soprattutto encefalo e aorta). Le prossime misure:

- **CT-RATE** (TC torace, CC BY-NC-SA 4.0, pubblico, utilizzabile con OpenRouter per la ricerca): un campione estratto
  per seed ed etichettato con la convenzione prima di ogni lettura, con gli stessi cinque criteri calcolati sugli
  item senza "?". Copre solo il torace: misura le correzioni, non il sistema intero.

  Il campione (7 novembre 2026): 200 item da 101 referti, 182 con un reperto, 17 da tenere senza reperto, 1 da non
  attribuire, 1 dubbio. Criteri, fissati prima del run, sui 199 item senza dubbio: (1) l'item da non attribuire
  non viene attaccato in silenzio; (2) nessun reperto messo da parte, al massimo due dichiarati; (3) alla persona
  al massimo il 10% degli item da tenere; (4) al massimo il 5% degli item accettati attaccati diversamente
  dall'etichetta; (5) al massimo il 5% di risposte perse. Con un solo item da non attribuire il criterio 1 qui
  dice quasi nulla: il campione misura soprattutto che cosa fa la lettura ai reperti veri (2, 3, 4).
- **CT-RATE, secondo campione** (`ctrate2`, 7 novembre 2026): 300 item, seed 20261108, da referti non usati dal
  primo campione, al massimo 24 per struttura, etichettati con la convenzione (compreso il dubbio del referto) prima
  di ogni lettura. Criteri, fissati prima di etichettare, sugli item senza "?", ogni quota di errore sotto il
  **2%**, la frequenza degli errori clinicamente significativi dei referti stessi (dettatura 2,3%, riconoscimento
  vocale 6–9,6% in du Toit et al., BMC Medical Imaging 2015), ciascuna con il suo limite superiore unilaterale al
  95% (Clopper-Pearson): (1) nessun item da mettere da parte attaccato come certo; (2) nessun reperto messo da
  parte; (3) al massimo il 2% degli item alla persona, ciascuno una discordanza del documento; (4) al massimo il 2%
  delle letture certe attaccate diversamente dall'etichetta, compresi i dubbi chiusi; (5) al massimo il 2% di
  risposte perse; (6) al massimo il 10% degli item in dubbio (un limite di utilità, non di errore), un dubbio che
  non tiene fra le sue letture quella dell'etichetta contato come errore (sotto il 2% degli item), i dubbi chiusi
  sbagliati sotto il 2% dei chiusi. Con 300 item,
  0 errori limitano la quota all'1,0%, 2 errori al 2,1%.
- **CT-RATE, terzo campione** (`ctrate3`, 9 novembre 2026): 300 item, seed 20261109, da referti non usati dai
  primi due, etichettati con la convenzione prima di ogni lettura. Letti due volte sugli stessi item: la lettura
  per affermazioni (spenta) e la domanda sul proprietario (accesa oggi), per confrontarle con gli stessi criteri.
  Proposti prima del run: (1) item attaccati come certi alla struttura sbagliata al massimo l'1% (l'errore più
  grave: il nostro si somma a quello del referto); le altre quote di errore al massimo il 2%; ogni quota con il
  suo limite superiore al 95%; le discordanze del referto intercettate riportate a parte. Frank decide sui
  risultati e sui tipi di errore.
- **CT-RATE, quarto campione** (`ctrate4`, 10 novembre 2026): 300 item, seed 20261110, da referti non usati dai
  primi tre, etichettati con la convenzione prima di ogni lettura. Criteri fissati prima: (1) al massimo l'1%; da
  (2) a (6) come per `ctrate2`, ogni quota di errore sotto il 2% con il suo limite superiore al 95%; (7)
  l'incertezza del referto persa (un item etichettato incerto le cui affermazioni non portano incertezza) sotto il
  2% degli item, riportata anche come quota degli item etichettati incerti. Se passano, la lettura per
  affermazioni diventa la domanda dell'ingestione.
- **CT-RATE, quinto campione** (`ctrate5`, 11 novembre 2026): 300 item, seed 20261111, da referti non usati dai
  primi quattro, etichettati prima di ogni lettura con la convenzione dell'11 novembre («consistent with»
  un'incertezza di livello alto). Criteri fissati prima: (1) al massimo l'1%; da (2) a (7) come per `ctrate4`; (8)
  l'incertezza data dove il referto afferma (nessuna incertezza nell'etichetta, o «consistent with» letto sotto il
  livello alto) sotto il 2% degli item, con il suo limite superiore al 95%.
- **Contro gli errori di etichetta, il manuale, la memoria e gli adattatori** (11 novembre 2026).
  - Etichette: scritte dalla frase intera, mai tagliata (`data/reading/labelling_convention.md`); un secondo
    etichettatore indipendente (un modello che non è nessuno dei lettori, di default Mistral Large) etichetta lo
    stesso campione prima di ogni lettura, dalla sola frase (dal 12 novembre 2026 non gli si dà né convenzione, né
    manuale, né esempi, né liste di parole: solo che cosa vuol dire ogni etichetta, in parole semplici) (`scripts/second_labeller.py`, input `second_labeller` del workflow);
    le sue etichette si aprono solo dopo che quelle del primo sono state committate; si riporta l'accordo (kappa
    di Cohen) e ogni disaccordo si rilegge sulla frase intera e si decide con le fonti, scrivendo il motivo di
    ogni cambiamento, prima che i lettori leggano (doppia annotazione con arbitrato, come è stato fatto BioScope).
    Un'etichetta corretta dopo un run porta il suo motivo, e il risultato si riporta sia come etichettato sia
    come corretto.
  - Il manuale (`data/reading/manual.md`): ciò che le fonti dicono del leggere una frase, dato con ogni domanda,
    versionato con la sua impronta (registrata nel resoconto del probe).
  - La memoria (`data/reading/memory.jsonl`, `scripts/reading_memory.py`): esempi verificati, i più simili dati
    con ogni domanda. Un esempio entra solo quando l'etichetta (senza dubbio, con la convenzione in vigore) e la
    lettura di ogni modello concordano su ciò che la frase afferma e sulla sua incertezza; non contiene testo di
    un corpus chiuso; un esempio si può togliere con il suo motivo (`data/reading/removed.jsonl`). Un campione
    misurato non dà mai esempi a sé stesso.
  - Gli adattatori (`scripts/reader_training.py`, ogni due o tre mesi, ricordati dal workflow
    `reader-training-due`): addestrati sulle GPU locali solo sulla memoria; la regola, fissata prima che esista
    un adattatore: su un campione nuovo il probe gira senza e con l'adattatore, con lo stesso manuale e la stessa
    memoria; l'adattatore entra se ogni criterio passa con esso e i suoi errori (in numero) sono meno; altrimenti
    resta fuori, e il motivo si scrive in `data/reading/training_log.yml`.
- **Che cos'è una classe dell'atlante, e i campioni estratti come allora** (11 novembre 2026). Una classe
  dell'atlante è una struttura se è ancorata a UBERON; una senza ancora UBERON il cui nome, senza il lato, è
  un'anomalia fenotipica di HPO è un reperto, e la sua sede è la struttura di cui HPO dice che l'anomalia è
  (`src/melampo/memory/class_kinds.py`: la stessa lettura per ogni classe, nessun elenco; oggi le cisti renali di
  TotalSegmentator, del rene dal loro lato; i segmenti epatici restano strutture). Il cercatore di nomi non
  attraversa più una parentesi o un punto e virgola («left kidney (cyst?)»). I campioni etichettati prima del 12
  novembre 2026 si estraggono con il cercatore di quel giorno (`MELAMPO_FINDER=2026-11-11` nei workflow che li
  estraggono): sul campione `claims` di PARROT il cercatore nuovo cambierebbe 3 item su 260. Ogni item
  etichettato dei campioni CT-RATE ha la sua impronta (`data/gold/<campione>/fingerprints.txt`): un item estratto
  di nuovo che non corrisponde resta fuori dai criteri e dalla memoria, e lo si dice. Da `ctrate6` i campioni si
  estraggono con il cercatore nuovo.
- **CT-RATE, sesto campione** (`ctrate6`, 11 novembre 2026): 300 item, seed 20261112, da referti non usati dai
  primi cinque, etichettati prima di ogni lettura con la convenzione dell'11 novembre (l'incertezza come la
  distingue la letteratura sugli attenuatori), la frase mai letta tagliata. Criteri fissati prima: (1) al massimo
  l'1%; da (2) a (8) ogni quota di errore sotto il 2% con il suo limite superiore al 95%. La lettura
  dell'incertezza si sceglie con una regola fissata prima: ciascun modello da solo e la loro unione si misurano
  sullo stesso run (criteri 7 e 8 per lettura); si usa quella con meno errori di 7 e 8 insieme, se entrambi
  restano sotto il 2%; a parità, quella che perde meno. Se nessuna passa, la lettura per affermazioni resta spenta.
- **CT-RATE, settimo campione** (`ctrate7`, 13 novembre 2026): 300 item, seed 20261113, da referti non usati dai
  primi sei, etichettati prima di ogni lettura con la convenzione precisata il 13 novembre (un'incertezza va alla
  struttura solo se verte su ciò che se ne dice, una struttura che colloca non ne porta; «at the level of» una
  parte della stessa struttura è in essa; «considered to be compatible with» è di livello alto). La domanda: ogni
  parola che qualifica con ciò che fa prima di collocarla; la misura dice se è della struttura o di ciò che vi si
  vede; una parte è della struttura solo se UBERON gliela dà. Lettori: Gemma 4 31B, Nemotron 3 Ultra senza
  e con ragionamento, Muse Spark Open con ragionamento, in un solo run. Regola fissata prima (Frank,
  13 novembre 2026: la coppia migliore dei tre modelli, due qualsiasi): ogni coppia di due modelli diversi (Gemma
  con Nemotron, con Nemotron con ragionamento, con Muse; Nemotron, senza o con ragionamento, con Muse) si misura
  rigiocando le risposte dello stesso run con la sola coppia (`owner_probe.py --replay`, nello stesso run del
  workflow); per ogni coppia la lettura dell'incertezza si sceglie come su `ctrate6` (ciascuno dei suoi modelli o
  la loro unione); si usa la coppia che supera tutti i criteri (1 al massimo l'1%, da 2 a 8 sotto il 2%) con meno
  errori da 1 a 8 insieme; a parità, meno errori del criterio 1, poi il costo minore. Ogni errore si legge per la sua
  causa, per categoria, prima di scrivere la scelta. Se nessuna coppia passa, la lettura per affermazioni resta
  spenta.
- **CT-RATE, ottavo campione** (`ctrate8`, 13 novembre 2026): 300 item, seed 20261114, dai referti di training
  di CT-RATE (id `ctrate-train-N`; i referti di validazione lasciati dai primi sette hanno solo 8 strutture),
  etichettati prima di ogni lettura con la convenzione del 13 novembre. Da `ctrate7` (owner-probe_29: Gemma con
  Muse 0 errori e 24 dubbi, Nemotron con ragionamento e Muse 0 errori veri e 10 dubbi): la negazione è una
  funzione propria fra le parole che qualificano; il lettore legge le parole normali di un modello dove stanno
  nella frase («tumoral wall thickening» copiato da «no significant tumoral wall thickening» è sotto la sua
  negazione). Lettori: Gemma 4 31B, Nemotron 3 Ultra con ragionamento, Muse Spark Open con ragionamento, in un
  solo run; ogni coppia misurata sulle stesse risposte (`--replay`). Regola fissata prima (Frank, 13 novembre
  2026: si pesano anche i dubbi): conta solo una coppia che supera tutti i criteri (1 al massimo l'1%, da 2 a 8
  sotto il 2%); fra queste, meno errori da 1 a 8, senza contare come errore del criterio 1 una struttura che
  colloca a cui non si attacca nulla; poi meno dubbi; poi il costo minore. L'incertezza di ogni coppia si legge
  come su `ctrate6`.
- **CT-RATE, nono campione** (`ctrate9`, 13 novembre 2026): 300 item, seed 20261115, dai referti di training
  non usati da `ctrate8`, etichettati prima di ogni lettura. Da `ctrate8` (owner-probe_31: Nemotron con
  ragionamento e Muse l'unica coppia che passa, 2 errori del criterio 1 e 15 dubbi, 11 non fondati sul referto),
  otto correzioni universali: il lettore legge in HPO di quale struttura è un fenotipo (una milza accessoria
  «anterior to the spleen» è della milza), legge una frase nuova dopo una congiunzione come coordinata propria, e
  concorda con i modelli quando la sua lettura della relazione colloca la struttura a cui loro danno il reperto; i
  modelli dicono quando una struttura colloca soltanto (`places`); letture che non dicono nulla della struttura,
  con il lettore che legge un altro proprietario, la mettono da parte; una misura che il modello dice della
  struttura non è presa per quella del reperto; un modello le cui parole non sono nella frase è interrogato una
  seconda volta; una misura che la frase giudica («above normal») e una variante del piano consueto sono reperti.
  Lettori, coppie e regola come per `ctrate8`.
- **Misura del router** (`scripts/router_probe.py`, workflow `router-probe`, 8 novembre 2026): il modello locale
  aiuta l'orchestratore a dire i tipi di un testo? Testi pubblici di tipo noto: MTSamples (6 per specialità,
  inglese) e PARROT (20 italiani, 10 di altre lingue), seed 20261108, etichettati (`data/gold/router/labels.txt`)
  prima di ogni lettura. Ogni modello da solo contro gli indizi da soli. Criteri, fissati prima delle etichette:
  con il modello, al massimo il 2% dei testi con un interprete mancato (un tipo dell'etichetta i cui interpreti non
  sono eseguiti), con il suo limite superiore al 95%; al massimo il 2% di risposte del modello inutilizzabili. Allora
  `MELAMPO_REPORT_ROUTER=model` si può accendere; altrimenti resta spento. Gli interpreti aggiunti che nessun tipo
  richiede sono riportati (un costo, non un errore).
- **PhysioNet** (accesso con credenziali richiesto): RadGraph-XL, etichette di radiologi su RX torace, TC torace, TC
  addome-pelvi e RM encefalo; MIMIC-IV-Note per la varietà; RadGraph per l'accordo fra due radiologi; RadCoref per i
  rimandi fra frasi. Si lancia solo in un ambiente protetto, con modelli nostri o un servizio verificato con
  PhysioNet, mai sui runner di GitHub, mai con OpenRouter finché PhysioNet non è d'accordo; nel repository tornano
  solo numeri aggregati.

## 6. Limiti noti

- Tutte le etichette finora sono di un solo etichettatore; le etichette "?" e gli elenchi dei reperti di un'altra
  struttura aspettano un radiologo.
- Il lettore non riconosce come anomalie "nodule" e "free fluid" (in UBERON nominano anche strutture); la
  negazione di "not enlarged … with cysts" arriva fino alle cisti.
- Non è rimasto nessun referto italiano di PARROT inutilizzato: per l'italiano serve un campione proprio.
- Lo strato Rad-SpRL della role probe è stato etichettato prima che esistesse il ruolo di riferimento.
