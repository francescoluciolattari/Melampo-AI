# Gold set: guida per i radiologi

*[English](../en/RADIOLOGIST_GUIDE.md) · Italiano · versione del 10 ottobre 2026, ruolo di riferimento e ruoli
multipli dal 31 ottobre, affermazioni dal 2 novembre 2026*

Due radiologi etichettano **da soli e in cieco** le stesse menzioni; un terzo decide i disaccordi. Il sistema non
è mai mostrato: un suggerimento sbagliato visto prima della lettura la influenza (Dratsch et al., *Radiology*
2023: da circa l'80% a meno del 20% di accuratezza).

## 1. Che cosa si fa

Ogni riga della scheda (`annotator_A.csv` o `annotator_B.csv`, si apre con Excel) è una **menzione** in una frase.
Si leggono `sentence` e `mention` e si compilano **quattro colonne**: `structure`, `relation`, `role`, `span_text`;
anche `claims` quando la scheda la contiene. `side_in_text` e `note` sono facoltative. Non si modifica nessun'altra
colonna.

| Colonna | Che cosa scrivere |
|---|---|
| `structure` | l'id da `valid_structures.txt`, oppure `NONE_IN_CLASSES` (struttura vera ma non nell'elenco: utero, appendice, ovaio), `NOT_ANATOMY` (la parola qui non nomina una struttura: la sequenza "T2", "frequenza cardiaca", "American Heart Association"), `AMBIGUOUS` (il testo non dice quale: lato mancante in una struttura pari, "iliaco", una sigla ambigua) |
| `relation` | solo per una struttura: `equal` (la struttura o un sinonimo), `part_of` (una parte: sigma → colon, testa femorale → femore), `contour_of` (il profilo su una radiografia), `approx` (sinonimo d'uso non identico: emibacino ≈ osso iliaco) |
| `side_in_text` | il lato scritto nel testo (destro, sinistro, bilaterale, nessuno); mai dedotto dal resto del referto |
| `note` | facoltativa, per i dubbi |

## 2. `role`: che cosa fa la parola nella frase

**Per una struttura** (un id o `NONE_IN_CLASSES`):

| `role` | Quando | Esempio |
|---|---|---|
| `structure` | la frase parla della struttura stessa | "fegato ingrossato" |
| `landmark` | la struttura serve solo a collocare un'altra cosa: un linfonodo, un liquido, una massa o un altro organo vicino, lungo, attorno, tra, fuori o verso di lei | "linfonodi lungo l'arco aortico", "liquido tra le anse del tenue" |
| `procedure_site` | il sito di una procedura | "biopsia epatica", "RM encefalo" |
| `device_site` | il sito di un dispositivo | "stent coronarico", "pacemaker atriale" |
| `source_of` | il luogo da cui vengono una sostanza o cellule | "estratto di fegato" |

**Per `NOT_ANATOMY`:**

| `role` | Quando | Esempio |
|---|---|---|
| `inherent_location` | la struttura è ciò di cui si parla una misura, una funzione o un processo | "frequenza cardiaca", "funzione epatica" |
| `inside_a_name` | la parola è un pezzo del nome di un'altra cosa | "Dallas Heart Study", "tireoperossidasi" |
| `not_a_body_site` | parte di una cosa che non è un corpo | "cuore del formaggio Maroilles" |
| `other` | nessuno dei tre; spiegarlo in `note` | |

Per `AMBIGUOUS` la colonna resta vuota.

**Nel dubbio fra `structure` e un altro ruolo:** il lettore del referto penserebbe a quella parte del corpo del
paziente? **Nel dubbio fra `structure` e `landmark`:** la frase dice qualcosa di questa struttura, o la usa per
dire dove sta un'altra cosa? Il criterio è di che cosa sono la misura e il carattere della frase, non la
preposizione: "placche lungo l'asse femoro-popliteo" sono nella parete dell'arteria (`structure`).

**Più ruoli insieme** si scrivono uniti da `+`, il principale per primo: in "ispessimento della parete del colon
in prossimità della flessura splenica", la menzione "flessura splenica" è `structure+landmark`.

## 3. `span_text`: il nome per intero

Di solito vuota. Si compila **solo** quando la menzione è una parte di un nome più lungo che la contiene,
scrivendo il nome intero com'è nel testo: "Heart" → "Dallas Heart Study". Se non si è sicuri dove finisce il nome,
lasciare vuoto e scriverlo in `note`.

## 4. `claims`: che cosa la frase afferma della struttura

Si scrive tutto ciò che la frase afferma **di lei**, con le sue parole, senza aggiungere ciò che si sa. Non si
sceglie un'etichetta sola: si segnano tutte le affermazioni presenti, unite da `+`.

| Affermazione | Quando | Esempio |
|---|---|---|
| `finding` | la frase osserva qualcosa di diverso dal solito in lei o in una sua parte, anche lieve o solo probabile | "pancreas **hyperechoic**", "**likely** calcified cartilages" |
| `involved` | un reperto di un'altra struttura arriva a lei: si estende in lei, la comprime, la restringe, o sta al livello o nell'articolazione che lei forma | "disc herniation **at the L5-S1 level**", "the mass **molds** the pancreas" |
| `normal` | la frase dice che è normale, o che un'anomalia è assente | "**not enlarged**", "**no stones**" |
| `measure` | una sua misura | "measures **10.5 cm**" |
| `history` | un fatto del suo passato | "**history of** thyroid surgery" |
| `device` | un dispositivo in lei | "**intubation tube** in the trachea" |
| `not_assessable` | non si può valutare | "the wall **is not assessable**" |
| `aside` | ciò che si osserva è di un'altra struttura, e questa serve solo a dire dove sta | "nodes **along** the aorta", "a collection **between** the flank and the psoas" |
| `nothing` | è nominata senza che se ne affermi nulla: un titolo, il nome dell'esame | "THYROID ULTRASOUND" |

**Detto, non dedotto.** Un valore che si sa normale ("IMT 0,5 mm") è solo una misura se la frase non dice
"normale". "Hyperechoic and not enlarged" sono due affermazioni: `finding+normal`. **Vicino non è contatto:**
"between", "adjacent", "along" dicono dove sta (`aside`); "extends into", "infiltrates", "compresses" dicono che
arriva a lei (`involved`).

## 5. Regole che valgono sempre

- Si etichetta la struttura **nominata**, non la malattia né la sua presenza: "assenza del rene destro" → rene
  destro.
- Un lume o uno spazio non è l'organo: "loggia renale", "lume esofageo" → `NONE_IN_CLASSES`.
- Parete, parenchima, corpo di un organo → l'organo, con `part_of`.
- Una struttura nominata solo come modificatore di una misura o di un esame ("frequenza cardiaca", "funzione
  epatica", "ormone tireostimolante") → `NOT_ANATOMY`; la stessa parola fra i reperti di un referto di immagini
  nomina la struttura.
- "Anca" senza "osso" è la regione o l'articolazione dell'anca: si etichetta la classe più vicina (`hip_left`,
  `hip_right`) con `approx`; "osso dell'anca", "osso iliaco", "os coxae" → `equal`.
- Lato non scritto in una struttura pari → `AMBIGUOUS`, anche se il resto del referto lo fa intuire. Se il lato sta
  solo nel nome dell'esame, scrivere "lato dall'esame" in `note`.
- Una parola dentro un nome proprio (istituzione, studio, scala, molecola, dispositivo) → `NOT_ANATOMY`.
- **Se non si è sicuri: `AMBIGUOUS`.** L'astensione del sistema su quei casi è corretta e viene misurata come tale.

## 6. La procedura

1. **Sessione di allineamento**, una volta, prima delle 2.000 menzioni: 40 frasi da corpora pubblici
   (`python scripts/gold_set.py alignment … --n 40`). Ogni radiologo compila la scheda da solo (circa 40 minuti);
   poi accordo e kappa (`gold_set.py agree`), si discute ogni disaccordo, la lettura di riferimento si apre solo
   dopo aver compilato. Se il kappa di `structure` o di `role` resta sotto 0,8, si scrive la regola mancante e si
   ripete con 40 frasi nuove. La sessione non fa parte dello studio.
2. **Etichettatura:** ogni radiologo da solo, senza vedere l'altro né il sistema. Controllare la scheda con
   `gold_set.py check`.
3. **Accordo e decisione dei disaccordi:** `gold_set.py agree` scrive i disaccordi; il terzo revisore li decide;
   `gold_set.py merge` scrive le etichette finali.
4. **Congelamento:** le etichette e il commit del sistema valutato si congelano; nessuno li guarda durante lo
   sviluppo.

## 7. Che cosa si consegna

Le due schede compilate, mai modificate dopo la consegna. Nessun referto lascia l'ospedale; il testo è
pseudonimizzato alla fonte.
