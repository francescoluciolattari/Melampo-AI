# Proposte migliorative: che cosa è stato adottato, P0 e P1 rivalutati sui dati, gold set di sviluppo (12 ottobre 2026)

Riferimento: `proposte_migliorative_auc_risoluzione_errori_2026-10-11.md` (documento esterno, valutato oggi). Frank ha chiesto di implementare i quattro punti utili, di rivalutare P0 (recall dei candidati) e P1 (reranker generativo) e di costruire un gold set con dati anonimizzati trovati in rete.

**Nessun numero qui è un test né una certificazione.** Il gold set di sviluppo è etichettato da un solo etichettatore (Claude), che è anche chi ha scritto il linker.

## 1. Che cosa è stato adottato (codice e test)

| proposta | dove | che cosa fa |
|---|---|---|
| livello dell'errore | `gold_set.error_level`, `ERROR_LEVELS`; `evaluate` → `error_levels`, ogni errore ha `level` | span, type, sense, concept (`concept:side` se cambia solo il lato), role, relation, assertion, export_convention; un livello primario per errore; l'aggiudicatore può forzarlo (`error_level` nella riga) |
| calibrazione e copertura-rischio | `selective_calibration`: `brier`, `expected_calibration_error`, `reliability`, `coverage_risk` (con limite superiore esatto e numero di link per punto), `risk_upper_bound`, `grouped_bootstrap` | `evaluate` riporta la curva copertura-rischio sul punteggio di convergenza e l'intervallo dell'errore ricampionando referti interi |
| stati di granularità | `LinkResult.match` | `exact`, `approximate` (approx, part_of, contour_of), `parent_candidate` (padre del grafo proposto o applicato: mai promosso a esatto), `abstain` |
| provenienza e regole scritte dopo un errore | `data/linking/provenance.json`, `evaluation/provenance.py`; `evaluate` → `post_error_items` | gli item toccati da una regola nata da un errore sono riportati a parte; 8 voci registrate (GB, paraspinal, hip, sigle tra parentesi, senso nel discorso, finestra delle cornici, plurali; la soglia npmi è registrata come non post-errore) |

Test: `tests/test_evaluation_levels_calibration.py` (8).

## 2. P0 rivalutato: la recall dei candidati conta, e il rimedio è morfologico

Avevo scritto che un concetto giusto che manca fra i candidati è raro. **I dati dicono il contrario.**

**CRAFT.** Annotazioni UBERON che corrispondono a una classe; i nomi senza lato di organi pari sono contati a parte.

| | prima | con i plurali |
|---|---|---|
| collegate alla classe giusta | 1.101 | **1.262** |
| proposte, linker astenuto | 309 | 311 |
| **mai proposte** | **237 (14 %)** | **74 (4,5 %)** |
| collegate a un'altra classe | 3 | 3 |

Le mancate erano soprattutto plurali ("kidneys", "livers", "brains", "hearts", "femurs", "spleens", "prostates"). Il linker non conosceva nessun plurale.

**Correzione** (`word_senses.plural_keys`, `Lexicon.plural_of`):
- il plurale dell'ultima parola di ogni nome del lessico: inglese regolare e irregolare (femora, bronchi, vertebrae, pelves), italiano per i nomi di una parola (reni, ossa);
- solo dove il plurale non è già un'altra voce;
- un plurale di una forma ambigua passa dal profilo dei sensi come il singolare. Al primo giro "axes" era diventato C2 quattro volte; ora no.

**Effetto sul controllo esterno.**
- CRAFT: link giusti 1.315 → 1.476, errori 11 → 14. I tre nuovi errori sono dello stesso tipo dei singolari già noti: "swim bladders" (pesce), "right middle lobes" (granularità), "dorsal aortae" (embrione).
- MedMentions: link giusti 710 → 758, errori invariati (22).

Restano mai proposti: organi pari senza lato (voluto: "kidney" da solo non dice quale classe), aggettivi ("renal", "adrenal", "colonic"), sigle di figure ("Ao", "Tr").

**Gold set di sviluppo (§4).** Il linker collega 81 dei 163 item di classe. Dei mancati, 64 sono "nome sconosciuto":
- parti senza voce: atrio, ventricoli, corpo calloso, zona periferica della prostata, piramide basale, dotto di Wirsung;
- sinonimi d'uso: "cardiomediastinal silhouette", "left base", "ilo";
- span con aggettivi o articoli: E3C annota "a normal liver", "fegato aumentato di volume".

**Conclusione rivista.**
- **Il punto della proposta era giusto: senza candidato nessun giudice recupera.**
- Il rimedio adatto a un dispositivo deterministico è la conoscenza esplicita: morfologia (fatto), parti dal grafo `part_of` di UBERON e nomi italiani delle parti (il prossimo passo), sinonimi d'uso radiologici dal gold set.
- Il recupero denso o ibrido va bene solo come proponente per la coda di revisione, misurato con la recall@K sul gold set. Mai come decisore.

## 3. P1 rivalutato: il reranker generativo resta un proponente

Dati già misurati nel progetto:
- il segmentatore LLM (`phrase-probe llm=sample`, 9 ottobre) trova 10 errori su 22 ma segnala il **29 % dei link giusti**, e azzecca il ruolo in 3 casi su 12;
- i due modelli concordano sul tipo nel 79 % dei casi e mancano gli stessi errori (errori correlati, come in letteratura).

Un decisore con il 29 % di falsi allarmi sui link giusti non è accettabile. Due vincoli restano:
- il contratto d'uso di PhysioNet vieta l'invio di MIMIC/RadGraph a servizi esterni;
- il dispositivo dev'essere deterministico e ispezionabile.

**Conclusione invariata, ora con i numeri:** LLM sì per proporre segmentazioni, riformulazioni o candidati alla revisione, mai come decisore. La priorità alta della proposta per P1 non regge sui nostri dati.

## 4. Il gold set di sviluppo ("silver")

**Fonti trovate e usate** (tutte senza credenziali):

| fonte | che cosa | perché | licenza |
|---|---|---|---|
| E3C layer 1 (FBK) | casi clinici IT (86) ed EN (84) con **span BODYPART annotati a mano** | gli span vengono da persone, non dal nostro lessico: il campione non dipende da ciò che il linker conosce | CC BY-NC |
| PARROT v1.0 | 285 referti italiani **fittizi scritti da radiologi** (10 autori; TC, RM, RX, eco) | italiano radiologico vero di forma | CC BY-NC-SA 4.0 |
| IU X-ray (Open-i) | 3.927 referti reali anonimizzati di RX torace | inglese radiologico reale | NLM, pubblico |

Fonti trovate ma non scaricabili da qui:
- **Rad-SpRL**: 2.000 referti IU X-ray con ruoli spaziali annotati, fra cui Landmark, cioè la sede anatomica (Mendeley Data, DOI 10.17632/yhb26hfz8n.1). Il sito non è raggiungibile da questo ambiente e la licenza non è indicata nella pagina; si scarica da GitHub Actions.
- **RadGraph-XL** e **MIMIC**: servono le credenziali PhysioNet.

**Costruzione** (`scripts/build_silver_gold.py`, `scripts/assemble_silver_gold.py`, etichette in `data/gold/silver_2026-10-12/labels`):
- E3C IT: tutti i 328 span; E3C EN: 150 su 968, seme fisso;
- PARROT: 100 frasi, una per referto, stratificate per modalità;
- IU X-ray: 100 frasi, una per referto;
- per PARROT e IU X-ray ho annotato **ogni** espressione anatomica della frase, non solo quelle che il linker conosce;
- schema: quello della guida dei radiologi (struttura, relazione, ruolo, span);
- **le etichette sono state scritte prima di far girare il linker su questi testi.**

Totale **714 item**:

| strato | item | di classe | fuori classe (N) | non anatomia (X) | ambigui (A) |
|---|---|---|---|---|---|
| E3C IT | 328 | 87 | 173 | 53 | 15 |
| E3C EN | 150 | 19 | 87 | 40 | 4 |
| PARROT IT | 134 | 39 | 77 | 2 | 16 |
| IU X-ray EN | 102 | 18 | 60 | 9 | 15 |

**Il linker sul gold set di sviluppo** (`scripts/gold_set.py evaluate`):
- accettati 82: corretti 81, 1 di relazione ("vena porta": io `part_of`, il linker `equal`; è una convenzione); 2 accettati fuori classe ("TC cranio" → skull; io ho letto "cranio" come la testa);
- **nessun collegamento** su 104 item non anatomici (fluidi, cellule, lesioni, nomi) e 50 ambigui: zero errori critici;
- errore sugli accettati 1,2 %, intervallo bootstrap per referto 0-3,9 % (54 referti); limite esatto al 95 %: 5,65 %. Per certificare ≤1 % con un errore servono 473 link accettati;
- livelli: 11 di ruolo (soprattutto `procedure_site` letto come struttura: "intervento al cuore", "RM encefalo"), 1 di relazione;
- stati: 66 exact, 18 approximate, 630 abstain;
- copertura: 50 % degli item di classe (81/163). I mancati sono soprattutto nomi sconosciuti (§2).

**Che cosa non è.**
- Un solo etichettatore, che è anche chi ha costruito il linker: errori e convenzioni correlati, nessun accordo misurabile. Le etichette non sono mai state confrontate con un secondo lettore.
- Le fonti non sono referti italiani reali: PARROT è fittizio, E3C sono casi clinici pubblicati, IU X-ray è un solo ospedale e solo il torace.
- 163 item di classe sono meno dei 299 che servono per certificare ≤1 % anche con zero errori.
- Licenze non commerciali: si usa per sviluppare e misurare. I testi non sono nel repository; si ricostruiscono dalle fonti (README nella cartella).

**Che cosa serve per farlo diventare un gold.**
1. Due radiologi rietichettano in cieco gli stessi item, partendo dalle schede vuote, non dalle mie etichette.
2. Si misura l'accordo, il terzo aggiudica, e le mie etichette diventano un terzo lettore da confrontare.
3. Si scarica Rad-SpRL da Actions e lo si allinea al nostro schema (Landmark → struttura, con ruolo).

## 5. Prossimi passi proposti

1. Parti: candidati `part_of` da UBERON per le classi, con nomi italiani rivisti (atrio, ventricoli, corpo calloso, zona periferica…).
2. Sinonimi radiologici d'uso dal gold set (silhouette, base polmonare, ilo), marcati come aggiunti dopo l'errore.
3. Rad-SpRL tramite Actions; credenziali PhysioNet per RadGraph-XL e MIMIC.
4. Radiologi sul gold set di sviluppo (schede vuote) per misurare l'accordo con le mie etichette.

Fonti: [E3C](https://github.com/hltfbk/E3C-Corpus) ed [E3C annotation](https://e3c.fbk.eu/annotation), [PARROT v1.0](https://github.com/PARROT-reports/PARROT_v1.0), [Rad-SpRL su Mendeley Data](https://data.mendeley.com/datasets/yhb26hfz8n/1), [RadGraph-XL](https://physionet.org/content/radgraph-xl/1.0.0/), [ANGEL (ACL 2025)](https://aclanthology.org/2025.findings-acl.558/).
