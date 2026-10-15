# I qualificatori della sede nel grafo: precisione, grado, certezza (15 ottobre 2026)

## Perché

Frank: "approximately" non dice se T11 è una vertebra, e il lettore non deve farlo votare su questo; ma non è
vuoto. "Approximately T11" è T11 con un margine; "mild wedging" non è "marked wedging"; "lievemente iperdensa"
dice un grado e una densità. Il grafo deve tenerne conto.

## Che cosa fa (`site_qualifiers.py`, `report_graph.py`)

Ogni collegamento accettato porta tre attributi (`LinkResult.qualifiers`), ciascuno con **le parole del referto
e la loro posizione**:

| attributo | valori | da dove |
|---|---|---|
| sede | `as_named`, `approximate` ("approximately", "near", "adjacent to", "circa", "in prossimità di"), `range` ("T11-T12", "da T6 a T8", "tra L4 e L5", "between the third and fourth"), `region` ("region of", "in sede", "regione") | parole subito prima della menzione, nella stessa frase |
| grado | `low` / `moderate` / `high` ("mild", "lieve", "modesto" / "moderate", "discreto" / "marked", "severe", "marcato", "cospicuo") | prima il tratto tra virgole della menzione (`phrase`), poi la proposizione (`clause`): lo scope è scritto |
| carattere | `density`, `signal`, `echogenicity`, `perfusion`, `metabolism` con direzione `high`/`low`/`equal`/`absent` ("hypodense", "ipointenso", "anecogena", "opacity", "lucency", "addensamento") | dalla forma della parola (prefisso iper/ipo/iso/an + qualità), non da un elenco di reperti |
| certezza | factual / possible / hypothetical, affirmed / negated, current / historical | il rilevatore ConText che esisteva già (`assertion.py`) |

Regole di portata, uguali per inglese e italiano:
- un'approssimazione vale solo se tra lei e la menzione ci sono solo articoli, "of/di", "level/livello" o una
  distanza ("a circa 5 cm dall'arcata"); "around the contusion in the right lobe" è intorno alla contusione;
- un nome di regione vale se regge la menzione ("region of", "in sede para-ilare") o la segue ("hilar region");
- un grado o un carattere nel tratto tra virgole della menzione ha la precedenza; altrimenti si prende quello
  della proposizione e lo si dice (`degree_scope: clause`).

**Descrivono, non decidono**: nessun collegamento cambia per un qualificatore.

## Nel grafo

`(:Report {id})-[:SITE {...}]->(:Structure {cid})` in FalkorDB (embedded o server): menzione, offset, ruoli,
relazione, i tre attributi con le parole e lo scope, e `applies_to`: `site` o `exam_findings` quando la
struttura è la sede di un esame ("TC del torace": il grado e il carattere della frase sono dei reperti
dell'esame, non del torace). Provato con un FalkorDB embedded reale nei test.

## Misura (859 frasi del role-probe, 484 inglesi e 375 italiane)

| | inglese | italiano |
|---|---|---|
| sede non "as named" | 10 (7 approssimata, 2 intervallo, 1 regione) | 7 (4 intervallo, 2 regione, 1 approssimata) |
| grado trovato | 109 (100 nel tratto della menzione) | 46 |
| carattere trovato | 60 (56 nel tratto) | 28 (11 nel tratto) |
| certezza non fattuale | 22 | 11 |
| negati | 75 | 51 |

Controllo a mano (giudice: Claude, da confermare dai radiologi):
- **sede**: dopo le correzioni di portata, 15 su 17 giuste; restano "areas of pulmonary consolidation" (regione
  presa per l'aggettivo) e un refuso del testo ("bolle aree").
- **grado e carattere nel tratto della menzione**: inglese circa 19 su 20, italiano circa 15 su 20. Gli errori
  italiani sono frasi lunghe in cui il tratto tra virgole contiene due reperti ("medio gluteo sx e iperdensità
  sottocutanea nella regione glutea") o la sede di un esame ("l'ecografia dell'addome evidenzia iperecogenicità
  della corticale renale": ora `applies_to: exam_findings`).
- **grado nella proposizione** (`clause`): meno affidabile, per questo è segnato.

## Corretto nel rilevatore di certezza esistente

Le parole dell'esperienziatore, dell'oggettività e della soggettività erano cercate come sottostringhe: "zio"
dentro "spa**zio**" faceva di "riduzione dello spazio discale tra L4 e L5" un reperto **di un familiare**,
quindi escluso dai reperti del paziente (`findings_boundary`). Ora sono cercate come parole intere.

## Limiti e prossimi passi

- Il grado è della proposizione o del tratto, non ancora del singolo reperto: legarlo al reperto richiede di
  sapere quale parola della frase è il reperto (il lettore lo sa per la menzione, non per gli altri nomi).
- Le parole di grado e di approssimazione sono un vocabolario chiuso scritto a mano; RadLex ha descrittori di
  gravità e di certezza che possono sostituirlo quando il workflow `radiology-kinds` avrà girato.
- La misura vera richiede i giudizi dei radiologi sul gold set (colonna da aggiungere al protocollo).
