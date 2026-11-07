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

Un'etichetta che le regole lasciano ancora aperta è segnata "?". **I criteri si calcolano sugli item senza "?"**
(`by_set.<campione>.criteria.without_doubt`); gli item "?" sono riportati a parte, con il loro esito
(`criteria.all`, `claims_kinds.for_a_radiologist`).

## 4. Risultati

Dal più recente. Il dettaglio di ogni run è nel documento indicato nell'ultima colonna, nel tag
`docs-archive-2026-11` (percorso `docs/linker/…`) o nella pull request.

| Data | Domanda | Risultato | Decisione | Dettaglio |
|---|---|---|---|---|
| 2026-11-07 | Affermazione per affermazione, campione `ctrate` (TC torace, CT-RATE) | valido (0 su 400 risposte perse); per la prima volta tutti e cinque i criteri soddisfatti: 0 su 1 attribuiti in silenzio, 0 reperti messi da parte, 16 su 198 alla persona (8,1%), 1,1% attaccati diversamente. Dei 16: 7 disaccordi fra i modelli, 5 dei quali annidati (un modello legge un pezzo che l'altro tralascia); 9 da controlli del lettore che richiamavano un senso estraneo al referto (il setto-mascellare degli anfibi per "corpus", le arterie embrionali degli archi per "aortic arch") o leggevano due sedi parallele come una | sensi secondo il contesto: il lettore richiama solo sensi del corpo di una persona adulta, la conclusione e "its" contano solo quando nominano un altro organo, "its" solo con un pronome, le sedi parallele tengono ciascuna il proprio reperto; il lettore come terzo lettore del pezzo su cui le letture annidate differiscono (**proposta**, cambia "i modelli devono concordare"). Sullo stesso run, già visto: 5 su 198 alla persona (2,5%), nient'altro cambiato, nessun cambiamento su claims2/3/4. Da misurare su un secondo campione CT-RATE | questa pull request |
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
