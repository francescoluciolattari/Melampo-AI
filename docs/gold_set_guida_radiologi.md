# Gold set del linker anatomico: guida per i radiologi

Versione del 10 ottobre 2026. Due radiologi etichettano **da soli**, in cieco, le stesse menzioni; un terzo decide i disaccordi. Il sistema non è mai mostrato: se un suggerimento sbagliato precede la vostra lettura, la influenza (Dratsch 2023). Il protocollo completo è in `gold_set_protocollo.md`.

## Che cosa si fa

Ogni riga della scheda (`annotator_A.csv` o `annotator_B.csv`, si apre con Excel) è una **menzione** in una frase. Si leggono `sentence` e `mention` e si compilano **quattro colonne**: `structure`, `relation`, `role`, `span_text`. La colonna `side_in_text` e `note` sono facoltative. Non si modifica nessun'altra colonna.

## 1. `structure`: che cosa è nominato

L'id della struttura dall'elenco `valid_structures.txt`, oppure:

| valore | quando |
|---|---|
| `NONE_IN_CLASSES` | struttura anatomica vera, ma non nell'elenco (utero, appendice, ovaio) |
| `NOT_ANATOMY` | la parola non nomina qui una struttura (sequenza "T2", "LM" tronco comune, "frequenza cardiaca", "American Heart Association") |
| `AMBIGUOUS` | il testo non permette di dire quale struttura (lato mancante in una struttura pari, "iliaco", sigla ambigua) |

## 2. `relation` (solo se `structure` è una struttura)

`equal` (è la struttura o un sinonimo) · `part_of` (è una parte: sigma → colon) · `contour_of` (profilo su radiografia) · `approx` (sinonimo d'uso non identico: emibacino ≈ osso iliaco). Per `NOT_ANATOMY` e `AMBIGUOUS` la colonna resta vuota.

## 3. `role`: che cosa fa la parola nella frase (novità)

Il ruolo dipende da `structure`.

**Se è una struttura** (un id o `NONE_IN_CLASSES`):

| `role` | quando | esempio |
|---|---|---|
| `structure` | la frase parla della struttura stessa | "fegato ingrossato", "the liver is enlarged" |
| `procedure_site` | la struttura è il sito di una procedura | "biopsia epatica", "liver donation", "RM encefalo" |
| `device_site` | è il sito di un dispositivo | "stent coronarico", "pacemaker atriale" |
| `source_of` | è il luogo da cui vengono una sostanza o cellule | "liver extract", "cellule di milza" in un esperimento |

**Se è `NOT_ANATOMY`**:

| `role` | quando | esempio |
|---|---|---|
| `inherent_location` | la struttura è ciò di cui si parla una misura, una funzione o un processo | "frequenza cardiaca", "funzione epatica", "heart development" |
| `inside_a_name` | la parola è un pezzo del nome di un'altra cosa (istituzione, scala, molecola, dispositivo) | "Dallas Heart Study", "thyroid peroxidase", "brain phantom" |
| `not_a_body_site` | la parola è parte di una cosa che non è un corpo | "heart of Maroilles cheese" (cuore del formaggio) |
| `other` | nessuno dei tre; spiegare in `note` | |

**Se è `AMBIGUOUS`** la colonna `role` resta vuota.

Nel dubbio fra `structure` e un altro ruolo: si chiede se **il lettore del referto penserebbe a quella parte del corpo del paziente**. "Biopsia epatica" nomina il fegato (`procedure_site`); "frequenza cardiaca" no (`inherent_location`).

## 4. `span_text`: il nome per intero (novità)

Di solito vuota: la menzione è già il nome intero. Si compila **solo** quando la menzione è una parte di un nome più lungo che la contiene, scrivendo il nome per intero com'è nel testo: menzione "Heart" → `span_text` "Dallas Heart Study"; menzione "liver" → "liver fatty acid binding protein". Deve contenere la menzione (lo controlla `check`). Se non si è sicuri dove finisce il nome, lasciare vuoto e scrivere in `note`.

## Regole che non cambiano

Le regole del protocollo valgono tutte: si etichetta la struttura *nominata*, non la patologia; il lato si legge dal testo e non dal resto del referto; "anca" senza "osso" è una regione; un lume non è l'organo. **Se non si è sicuri: `AMBIGUOUS`**. L'astensione del sistema su quei casi è corretta e viene misurata come tale.

## Prima della raccolta: la sessione di allineamento

Una volta, prima delle 2.000 menzioni, i due radiologi fanno insieme il giro di prova (`alignment_session/`, 40 frasi da corpora pubblici):

1. ognuno compila la propria scheda da solo (circa 40 minuti);
2. il terzo revisore (o chi coordina) lancia `python scripts/gold_set.py agree annotator_A.csv annotator_B.csv`: legge accordo, kappa e confusioni di ruolo;
3. si discute **ogni** disaccordo e ogni caso in cui i radiologi hanno letto diversamente dagli annotatori del corpus (`reference_for_the_discussion.csv`, da aprire solo dopo aver compilato);
4. dove la regola non basta si scrive la decisione in `gold_set_protocollo.md` **prima** di cominciare la raccolta vera.

Il giro non fa parte dello studio e non conta per la certificazione. Se il kappa di `structure` o di `role` resta sotto 0,8 dopo la discussione, si scrive la regola mancante e si ripete con 40 frasi nuove.

## Cosa si consegna

Le due schede compilate, mai modificate dopo la consegna. Nessun referto lascia l'ospedale; il testo è pseudonimizzato alla fonte.
