# Disambiguazione dal contesto e dalla lingua: cosa dicono neurobiologia, pedagogia e IA, e cosa ne abbiamo costruito (6 ottobre 2026)

Richiesta di Frank: una soluzione universale, in cui contesto e lingua sono i discriminanti in ogni caso (il "GB" cistifellea / globuli bianchi è solo il primo esempio), da inserire nei nove passi, simulando la comprensione contestuale umana e aggiungendo la precisione dei modelli di IA.

## Che cosa è stato davvero riletto, e che cosa no
- **Letto in questa sessione** (pagine aperte e consultate): la rassegna di Rodd & Rodd sull'ambiguità lessicale; la sintesi di Kintsch (2005) sul modello costruzione-integrazione; lo studio JAMIA 2024 sugli LLM e le sigle cliniche; PLACID (2026); l'articolo su Nature Communications 2022 sulle abbreviazioni cliniche; McCarthy et al. (2005) sulle priorità di senso specifiche del dominio.
- **Dai documenti già nel Project**: `comprensione_umana_linking_clinico_2026-10-06.md` (N400/P600, Mosè, cingolato, metacomprensione).
- **Non riletto in profondità: la pedagogia.** Quello che ne uso (teoria degli schemi, il ruolo del contesto nella comprensione, il vocabolario appreso dal contesto) viene dalla mia conoscenza generale, non da una rilettura dei trattati in questa sessione. Va considerato un'ipotesi di lavoro finché non è verificato sulle fonti. Chiedere "tutti i trattati" non è fattibile in una sessione: se serve, si può fare una rassegna dedicata e circoscritta.
- Le cifre di prestazione degli LLM qui sotto vengono dalle fonti lette e **non sono state riverificate** al momento di scrivere questo documento: vanno controllate sull'originale prima di citarle altrove.

## Neurobiologia e psicolinguistica: che cosa dicono, che cosa ne segue
| Risultato | Conseguenza di progetto | Dove |
|---|---|---|
| Accesso riordinato: tutti i significati si attivano; frequenza relativa e contesto li riordinano; contesto forte elimina il significato subordinato (Duffy, Morris & Rayner 1988; Rodd et al. 2002/2005) | Nessun significato è "il primo": ogni senso della forma raccoglie prove e compete | `word_senses.Sense.score` |
| Omonimia (significati senza relazione) e polisemia (significati collegati) si elaborano diversamente (Rodd) | GB (omonimo) e "midollo" (spinale/osseo, collegati ma distinti) stanno nello stesso inventario, ma con classi consentite diverse | campo `classes` |
| Costruzione-integrazione (Kintsch): costruzione permissiva, integrazione per soddisfacimento di vincoli; lo schema è un vincolo fra gli altri | Cue di tipo diverso (parola, schema di numeri/unità, lingua, contesto ampio) sommati; nessuna regola singola decide | `judge()` |
| Controllo semantico nella corteccia frontale inferiore sinistra quando i significati competono | L'astensione è lo stato normale della competizione non risolta, con motivo `sense_conflict:<senso>` | passo 8 |
| Priorità di senso specifiche del dominio (McCarthy et al. 2005): la frequenza dei sensi cambia col dominio | Per un referto radiologico, il senso dominante non è quello del dizionario generale; gli indizi vanno tarati sul dominio e sulla lingua | inventario + gold set |
| N400 (recupero) distinto da P600 (integrazione) | Recupero e integrazione restano punteggi separati | passo 4 |

## IA clinica: che cosa dicono
- La disambiguazione delle sigle cliniche è un problema studiato: con modelli grandi le prestazioni sull'inglese sono alte, ma calano in altre lingue e i modelli sono sovra-confidenti (JAMIA 2024; Nature Communications 2022; PLACID 2026 per i confronti). **Conseguenza:** il senso non si affida al solo LLM e non si usa la sua confidenza; il LLM resta nel passo 6 su candidati già filtrati.
- Modelli dedicati alla disambiguazione (T5 e simili) raggiungono accuratezze molto alte su benchmark inglesi in dominio: non c'è evidenza che valgano su referti italiani né su sigle fuori inventario.
- La precisione dei modelli entra dove misura bene (recupero, proposta, scelta tra candidati già compatibili); il senso, dove l'errore è silenzioso e costoso, è deciso da vincoli espliciti e verificabili, con astensione quando mancano.

## Che cosa è stato costruito
- `src/melampo/memory/word_senses.py` e `data/linking/word_senses.json`: un meccanismo unico, i fatti nei dati. Sei forme di partenza: GB, LM, ponte, ileo, digiuno, midollo.
- Regole: ogni senso raccoglie prove (forti 3, deboli 1, schemi di numeri/unità forti, lingua +1/−2, contesto ampio a metà peso); il senso anatomico passa con ≥2 punti e ≥1 di margine; altrimenti astensione con il senso concorrente nominato.
- Sostituisce `AMBIGUOUS_ABBREVIATIONS`, `gb_reading` e `CONTEXT_ABBREVIATIONS` (scritti a mano, per parola).
- Test: `tests/test_word_senses.py` (meccanismo su una forma inventata che non esiste nei dati, lingua come discriminante, contesto a metà peso, inventario spedito, integrazione nel linker, classi consentite) più i test GB e LM preesistenti: 337 test verdi sulle suite del linker, gold set, corpora, parti e bench.

## Onestà sui limiti
1. **Misurato:** i test passano; i casi E3C dove è nato l'errore ora si comportano come previsto; le suite esistenti non peggiorano.
2. **Dedotto, non misurato:** che i pesi e le soglie siano quelli giusti; che le sei voci coprano i casi reali; che la stima della lingua regga su referti veri (è un'euristica a parole-spia).
3. **Non coperto:** le forme non elencate (altre sigle, "base", "corpo", "seno", "L5" come dermatomero...), negazione, distanza sintattica, referto intero come contesto (il parametro c'è, ma nessun chiamante lo usa ancora), flessioni non elencate.
4. **Prossimi passi:** scansione UMLS per trovare sistematicamente le forme con più tipi semantici; passare il referto intero come `context`; tarare pesi e soglie sul gold set; misurare errori di senso sul gold set come categoria a sé.
5. **Certificazione:** solo il gold set reale (due radiologi in cieco + terzo) può dire se la precisione sugli accettati è ≥99%.

## Fonti
- Rodd, J. & Rodd, L. — rassegna sull'ambiguità lessicale e l'accesso riordinato (consultata).
- Duffy, S., Morris, R., Rayner, K. (1988), *Lexical ambiguity and fixation times in reading*; Rodd, Gaskell, Marslen-Wilson (2002, 2005) — citati nella rassegna sopra.
- Kintsch, W. (2005), sintesi del modello costruzione-integrazione (consultata).
- McCarthy, D. et al. (2005), *Domain-specific sense distributions and predominant sense acquisition* (H05-1053, consultato).
- JAMIA (2024), LLM e disambiguazione delle sigle cliniche; Nature Communications (2022), abbreviazioni cliniche; PLACID (2026) — consultati; cifre non riverificate qui.
- Aurnhammer et al. (2023), Caucheteux et al. (2023), Woolnough et al. (2021), Rastle et al. (2004), Prinz et al. (2020): dal documento `comprensione_umana_linking_clinico_2026-10-06.md`.
- Pedagogia (teoria degli schemi, contesto e comprensione, vocabolario dal contesto): conoscenza generale, **non riletta in questa sessione**.
