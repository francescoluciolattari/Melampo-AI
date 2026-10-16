# L'orchestratore dei referti: prima il tipo, poi l'interprete (16 ottobre 2026)

## Com'era

Nessun orchestratore. Ogni testo passava dalla grammatica dei valori di laboratorio, e il tipo si deduceva
**dopo**, contando le righe lette (≥ 2 righe numeriche = "laboratorio"). Conseguenze misurate su esempi:
- un istologico con due righe di immunoistochimica risultava "laboratorio" e le sue frasi diventavano "righe non lette";
- "Ki-67 20%", "ER 90%" (percentuale attaccata al numero) e "HER2 2+" (punteggio) non erano letti, e in un
  documento non di laboratorio **sparivano senza avviso**.

## Com'è ora (`data/report_router.py`)

1. **Il modello dice solo il tipo.** Una domanda, un elenco chiuso: laboratorio, elettroforesi, microbiologia,
   anatomia patologica, referto di imaging, nota clinica, altro. Più tipi insieme sono ammessi (una lettera di
   dimissione con una tabella di esami e un referto TC), con le righe dove stanno. Non legge valori, non
   interpreta, non diagnostica. È un modello **locale** (lo stesso endpoint di Gemma 4 / Nemotron): un endpoint
   non locale è rifiutato (`local_chat`), perché i referti sono dati di pazienti (GDPR).
2. **Gli indizi deterministici dicono lo stesso a modo loro**: righe di laboratorio lette, frazioni proteiche,
   antibiogramma, intestazioni dell'istologico (macroscopia, microscopia, immunoistochimica), parole di un
   referto di imaging (tecnica, reperto, conclusioni, TC, RM, ecografia), immagini allegate.
3. **Lo smistamento è l'unione; il disaccordo si registra.** Si attiva ogni interprete chiesto da una delle due
   parti: un interprete deterministico che non trova nulla non costa nulla, uno non attivato perde quello che
   solo lui leggerebbe. Se modello e indizi si contraddicono, o il modello non risponde, la decisione è marcata
   `needs_review` e gli indizi smistano da soli. **Il modello non è mai l'unico cancello.**

Interpreti: `lab_table` (grammatica dei valori, antibiogrammi, punteggi), `electrophoresis_trace` (oggi
registrato come presente e non letto), `pathology` (da costruire: sezioni, punteggi, classificazioni; per ora
l'immunoistochimica passa da `lab_table` e la diagnosi dal lettore), `narrative_reader` (lettore e linker),
`imaging` (DICOM e immagini).

La decisione viaggia con ogni allegato (`ProcessedAttachment.routing`, nel riepilogo del caso). La grammatica
riceve dall'orchestratore se il documento è "solo laboratorio": in un documento con parti in prosa si
segnalano come non lette solo le righe impaginate come tabella, mai le frasi.

## Corretto nella grammatica dei valori
- "20%" attaccato: letto come 20 con unità %.
- Punteggi "0/1+/2+/3+" e "+/++/+++": letti come valore semiquantitativo (`score:2+`), mai confrontati con un intervallo.
- Una riga impaginata come tabella (colonne, oppure "Grading: G2", "pT: 1b") che nessuno legge è sempre segnalata.

## Elettroforesi: il tracciato si perde?

Sì, in parte, e va detto. La tabella dà le percentuali e i g/dL delle frazioni; il tracciato mostra la **forma**:
- una **componente monoclonale piccola** può stare dentro una frazione gamma (o beta) di valore normale: le
  percentuali non la mostrano, il picco stretto sì. È il caso clinicamente più importante (MGUS, mieloma);
- il **ponte beta-gamma** (cirrosi), la doppia banda (bisalbuminemia, alfa-1), le bande oligoclonali.

Di solito il laboratorio lo scrive nel commento ("presenza di componente monoclonale in zona gamma, 0,8 g/dL")
e il commento ora è letto come testo. Quando il commento manca, l'informazione è solo nel tracciato. Per questo
l'orchestratore, davanti a un'elettroforesi, annota "tracciato presente e non letto" e, se il PDF contiene
l'immagine, quante sono: il caso va visto da una persona. Due strade per leggerlo, da misurare: nei PDF
digitali il tracciato è spesso disegnato come vettori e si può digitalizzare in modo deterministico (curva →
picchi); nelle scansioni serve un modello di immagine, da validare prima di fidarsi.

## Prossimi passi
- Interprete dell'anatomia patologica: sezioni, grading/staging, TNM, margini, recettori, diagnosi al linker.
- Lettura deterministica del tracciato vettoriale dell'elettroforesi.
- Collegamento degli esami a LOINC (variante italiana).
- Prova su referti veri dei laboratori che invieranno i casi.
