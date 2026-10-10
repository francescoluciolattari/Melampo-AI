# Il lettore Kintsch e la comprensione umana: dove diverge, che cosa è stato corretto (10 ottobre 2026)

Risposta alla domanda di Frank: *"Com'è possibile che il lettore Kintsch non sia utile, se la comprensione umana riesce dove il lettore fallisce? Analizziamo il problema e correggiamolo; cerca nella documentazione dove nasce."*

Fonti nel Project: `perche_il_medico_legge_e_noi_sbagliamo_2026-10-09.md` (§2, §3, §6.1, §6.2, §7), `comprensione_umana_linking_clinico_2026-10-06.md`, `lettura_del_sintagma_cervello_e_modelli_2026-10-09.md`, `analisi_fallimenti_interpretativi_medmentions_2026-10-09.md`. Codice: `ci_reader.py`, `chunk_lattice.py`.

**Nessun numero qui è un test.** Le correzioni sono state scritte dopo aver letto le righe di sviluppo (1.935 link giudicati, 30 errori; lo split di test congelato non è stato toccato). Una scelta (solo "cibo", §4) è stata fatta guardando i dati e lo dico dove cade.

## 1. Risposta breve

1. **La premessa è vera solo in parte.** Su 30 errori di sviluppo, il lettore legge come legge un medico (la parola è una struttura) in 22 casi, più 2 discutibili (phantom); lì l'"errore" è nell'etichetta (granularità, mappatura, etichetta dubbia, nome di un processo che contiene l'organo), non nella lettura. Lo dice già il documento del 9 ottobre (§2: 16 su 33 sono convenzioni, 4 dubbie) e lo dice l'AUC: 0,51-0,63 perché il segnale giusto non c'è nell'etichetta. Questa classificazione è mia: la conferma spetta ai radiologi del gold set.
2. **Dove l'uomo riesce davvero e il lettore no**, le cause sono quattro e stavano nel progetto, non erano ignote: (a) *il senso si fissa nel discorso*: un medico che ha letto "heart of Maroilles cheese" legge il terzo "heart" dello stesso testo come cuore del formaggio; (b) *le regole sono nodi della rete, non filtri a posteriori* (il problema di Linda, Kintsch 1998): il lettore non vedeva la regola "parte di un oggetto" del reticolo; (c) *l'unità si costruisce prima del giudizio e attraverso la sigla definita*: "inferior vena cava (IVC) filter placement" è un nome solo per chi legge, per il nostro reticolo era spezzato dalla parentesi; (d) *il lettore era ridondante con il reticolo*: i suoi nodi "blocco" e "testa" sono il reticolo stesso, quindi dove il reticolo non sa, il lettore non sa.
3. **Ho corretto (a), (b), (c)** in modo universale, con una costante presa dalla fonte (98 %, Gale, Church e Yarowsky 1992). Risultato sullo sviluppo: errori letti diversamente da "struttura" 2 → 6 su 19 (MedMentions); link giusti cambiati 36 → 38 su 590, di cui solo 4 che il reticolo non cambia già (0,7 %). Il reticolo e il lettore insieme agiscono ora su 10 dei 19 errori.
4. **(d) non si corregge con codice**: serve memoria. Restano 9 errori, e nessuno è risolvibile da un lettore senza altra conoscenza (§5).

## 2. Dove nasce il problema, nella documentazione

| Il documento dice | Che cosa implica per il lettore |
|---|---|
| §1 punto 1 (Kintsch 1998 p. 221; 2001 p. 197): dove manca la conoscenza non c'è memoria di lavoro a lungo termine; l'errore di predicazione "indica una mancanza di conoscenza, non un difetto dell'algoritmo". | Nove errori di nome composto (procedura, dispositivo, funzione) dipendono dalla memoria. Il lettore ne ha poca: NCIt non contiene nessuno dei 13 nomi lunghi in errore. |
| §3, riga "Conoscenza formale contro associazione" (Kintsch 1998 §11.2.1): nel problema di Linda la regola formale entra come nodo e rovescia la preferenza. "Le nostre regole deterministiche agiscono come filtri a posteriori, non come nodi della rete." | Il lettore non aveva nodi di regola. La regola "parte di un oggetto" del reticolo (`heart of cheese`) decideva da sola fuori dalla rete. |
| §3, riga "Un senso per discorso" (Gale, Church, Yarowsky 1992): "Il braccio `discourse` è locale." | Il lettore leggeva ogni occorrenza da sola. Due dei tre "heart" del formaggio dicono solo "in the heart" accanto a "rind". |
| §3, riga "Il medico esperto": scopre le incoerenze e riparte da un'altra ipotesi; §3 "Circolo ermeneutico": una sola passata. | Non c'è revisione. Non è stata costruita ora (§5). |
| `lettura_del_sintagma…` §1-§3 (Ding 2016, Nelson 2017): l'unità si costruisce prima del giudizio, e sigle e parentesi non la spezzano. | Il reticolo trattava "(" come interruzione. |
| §6.1 "Che cosa ne ricavo": le parole di contesto sono rumore a questa scala; l'ambito centrato non aggiunge informazione; "la causa non è un segnale che manca, ma che 16 dei 33 errori sono convenzioni di etichetta". | L'AUC bassa è in gran parte un problema di metrica, non di lettore. |
| §6.2: massimo correggibile da una lettura 10 su 19; criterio vecchio coincideva con il massimo. | Dire che il lettore "non serve" confrontandolo con tutti gli errori mette nel conto 9 errori che nessuna lettura può correggere. |

Dal codice (`ci_reader.py`, `construct`), verificato:

- I nodi `block` e `head` vengono da `ChunkLattice.candidates`: stessa memoria, stesse teste del reticolo. Non è un secondo parere indipendente.
- I nodi `traces`/`domain`/`gist` ripetono in parte la frequenza di base ("una struttura è quasi sempre una struttura"): dove sono indipendenti sono pochi (il tipo "cibo" ha 11 esempi nell'addestramento).
- Non c'era nessun nodo di regola e nessuna memoria del documento sui sensi.

## 3. I 30 errori di sviluppo: che cosa direbbe chi legge, che cosa il lettore

Colonne: la lettura di una persona è un mio giudizio sulla frase, non una misura. "Lettore prima / ora" è la decisione del lettore (tutte le prove) con esclusione del documento letto.

| errore (n) | che cosa fa una persona | lettore prima | lettore ora | natura |
|---|---|---|---|---|
| CRAFT: bladder (3), right middle lobe (7) | struttura | struttura | struttura | granularità dell'etichetta |
| CRAFT: aortic arch embrionale (1) | un'altra struttura (arteria dell'arco faringeo) | struttura | struttura | serve la cornice "embrione" (la cornice dell'esame non esiste per gli articoli) |
| MM: aortic arch (2) | struttura | struttura | struttura | mappatura CUI→UBERON |
| MM: heart del formaggio (3) | non un organo | struttura | **non un sito** | senso nel discorso + regola |
| MM: inferior vena cava filter placement (2) | struttura come sito di procedura/dispositivo | 1 su 2 | **2 su 2** | sigla tra parentesi |
| MM: large bowel volumes assessment (1) | struttura, ruolo registrato | ruolo | ruolo | convenzione |
| MM: liver donation, developing brain (2), brain controls, development of pancreas (5) | struttura, con ruolo (funzione, procedura) | struttura | struttura (il reticolo registra il ruolo in 4 su 5) | convenzione: nome di un processo |
| MM: colon (2), brain "retained inside the cranium" (1) | struttura | struttura | struttura | etichetta dubbia |
| MM: heart interlukine-6 (1) | struttura come origine della molecola | struttura | struttura | refuso e coda "-6": la testa non si legge |
| MM: brain phantom (2) | un oggetto che imita il cervello | struttura | struttura | testa "phantom" non agisce (§5) |

## 4. Che cosa è stato corretto

Tutto è universale (nessun caso nominato nel codice) e viene dalle fonti, non dagli errori.

1. **Un senso per discorso** (`Discourse`, `CIReader.discourse`). Il testo viene scandito una volta: dove il reticolo legge una parola-struttura come parte di un oggetto con la regola "di" ("heart of Maroilles cheese"), la parola e la lettura si registrano. Per le altre occorrenze della stessa parola nel documento entra un nodo con peso 0,98 (Gale, Church, Yarowsky 1992: una parola tiene il senso nel discorso il 98 % delle volte) che sostiene la lettura fissata e inibisce quella predefinita ("struttura"). L'occorrenza che ha fissato il senso non sostiene sé stessa (posizione nel documento). Una lettura per difetto non è un'osservazione; un ruolo in un composto ("liver donation") è locale alla frase e non conta.
2. **Nodo di regola** (`construct`, famiglia `rules`). La lettura "parte di un oggetto" del reticolo entra come nodo nella rete, con legame negativo verso "struttura": la regola formale che cambia il senso prevale sul senso che sostituisce (Kintsch 1998, Linda). Un ruolo che la struttura assume ("appearance of the small intestine") non è un senso della parola e non entra: provato, aggiungerlo faceva cambiare lettura a circa 16 link giusti in più ("number of spinal cord", "appearance of small intestine").
3. **Sigle definite trasparenti** (`see_through_abbreviations`). Una sigla di 2-10 caratteri tra parentesi, con almeno due maiuscole (o maiuscola più cifra o trattino), non spezza il sintagma: "inferior vena cava (IVC) filter placement" si legge come senza la sigla; "(n=24)", "(left)", "(B)" restano come sono; la sigla che è la menzione stessa resta. L'offset della menzione si ricalcola. Vale nel reticolo (`read`, `candidates`) e nel lettore.
4. **Cablaggio.** `AnatomyLinker` costruisce per ogni testo il gist, l'ambito e la memoria dei sensi, e passa al lettore la posizione della menzione nel testo (`at`). Il comportamento resta quello dichiarato: il lettore parla solo dove nessun altro metodo decide e non cambia mai un link.
5. **Metrica.** `ci_probe` ora riporta anche *link giusti cambiati solo dal lettore* (quelli che il reticolo lascia fermi): un ruolo che anche il reticolo registra non è un falso allarme del lettore, il prodotto tiene il link e scrive il ruolo.

### Una scelta presa guardando i dati

All'inizio il senso fissato nel discorso valeva per qualunque proprietario di tipo "cibo" o "dispositivo". Scandendo i 373 documenti con righe giudicate sono comparsi 6 sensi fissati: 1 vero (formaggio) e 5 sbagliati (neuron/eminence "form", layer/"bulb", sequence/"development", domain/"forms", cell/"bulb"): le teste *form*, *bulb*, *development* sono tipate "dispositivo" nel lessico delle teste. Un senso sbagliato fissato per tutto il testo avvelena le altre occorrenze (132 di "neuron" in un documento). Quindi **solo il proprietario di tipo "cibo" fissa un senso per il testo** (`DISCOURSE_OWNER_KINDS`); "dispositivo" aspetta un lessico delle teste curato dai radiologi. Questa restrizione è dipendente dai dati di sviluppo, e il caso positivo è un solo documento: tre righe di un solo articolo. **Non vale come prova di generalità.** Vale come dimostrazione che il meccanismo, per il caso per cui è scritto, funziona, e come misura del suo costo (nessun link giusto cambiato).

## 5. Che cosa si misura

Sviluppo, split congelato escluso; il documento letto non è mai nella memoria (leave-one-document-out).

| | MedMentions (19 errori / 590 giusti) | CRAFT (11 / 1.315) |
|---|---|---|
| Lettore prima (bundle `reader-gated`) | errori letti "non struttura" 2; giusti cambiati 36 | 0; 62 |
| Lettore ora | **6**; 38 | 0; 62 |
| …di cui solo il lettore (il reticolo lascia fermo) | errori 2; giusti **4** (0,7 %) | 0; giusti 10 (0,8 %) |
| Senza regole né discorso (solo sigle) | 3; 38 | 0; 62 |
| Senza discorso | 4; 38 | |
| Senza regole | 5; 38 | |
| Flusso del lettore nel linker (dove nessun altro metodo decide) | **2** errori su 19, 3 giusti su 590 | 0 errori, 0 giusti |
| Prima, nel linker | 0 errori, 3 giusti | 0, 0 |
| AUC (1 − quota di struttura) | 0,724 (era 0,595) | 0,365 |
| Reticolo da solo, giusti su cui agisce | 123 (era 121) | 223 |

Lettura. Le sigle trasparenti spostano un errore (la seconda "inferior vena cava filter placement") e due link giusti che prendono un ruolo uguale a quello della stessa frase senza sigla ("medial temporal lobe (MTL) activity" ora è letto come "medial temporal lobe activity"); la regola come nodo risolve il primo "heart" del formaggio; il discorso, gli altri due. I link giusti cambiati restano 38 su 590, perché la maggior parte sono ruoli che anche il reticolo registra: 20 su 27 in MedMentions e 38 su 49 in CRAFT coincidono con il ruolo del reticolo. Le cifre del criterio preregistrato ("12 su 22 con al più il 3 %") non passano ancora, per la ragione già detta: i giusti "cambiati" sono quasi tutti ruoli; il nuovo conteggio "solo lettore" dice 0,7-0,8 %.

Il reticolo e il lettore insieme agiscono sui seguenti 10 dei 19 errori di MedMentions: inferior vena cava ×2, heart (formaggio) ×3, liver donation, developing brain (prima frase), large bowel, brain controls, development of pancreas. Il numero 10 coincide con il massimo del §6.2, ma non l'ho confrontato riga per riga con quella classificazione.

## 6. Che cosa resta, e di chi è

I 9 errori che nessun codice di lettura risolve:

- 2 × *aortic arch* (mappatura), 3 etichette dubbie (colon ×2, "brain retained inside the cranium"): servono i radiologi e l'allineamento delle convenzioni.
- *developing brain* (seconda frase): la prima frase aveva "propofol" come testa; in questa la testa è un participio. Il filtro grammaticale è spento di default nel lettore.
- *heart interlukine-6*: refuso e "-6" (il tokenizzatore taglia al trattino). Una tolleranza ai refusi sulle teste è un'idea universale ma da misurare con un lessico di prova; non l'ho fatta.
- *brain phantom* ×2: servirebbe fissare per il testo il senso "oggetto" della parola, cioè un proprietario di tipo "dispositivo" (*phantom*). È la ragione per cui un lessico curato delle teste vale più di qualunque costante.
- *aortic arch* embrionale (CRAFT): serve la cornice (organismo, età), che negli articoli non è un'intestazione d'esame.

Resta valido ciò che il documento del 9 ottobre dice a monte: la memoria è la causa prima. Il lettore non può avere la memoria di un medico con 4.392 abstract e 146.096 definizioni; la sola leva su cui possiamo agire prima del gold set è il **lessico delle teste curato** (inglese e italiano) e uno **spazio semantico grande** (PubMed, referti). Dopo, il gold set dà il segnale pulito (span e ruolo) per misurare.

## 7. Limiti

- Il caso positivo del discorso è un solo documento. Il 98 % è il numero di Gale e colleghi per parole ambigue in testi, non la precisione della nostra regola: la nostra regola ha 1 caso vero e 0 falsi sui link giudicati, e 5 falsi su 6 sensi fissati se si scandiscono tutte le parole del testo (§4).
- Il 98 % di Gale, Church e Yarowsky è dalla mia conoscenza, non riverificato nel testo in questa sessione.
- Solo inglese. Per l'italiano non c'è lessico delle teste né memoria dei blocchi (vedi `supporto_italiano_piano_2026-10-10.md`).
- Le tracce scritte per il lettore non applicano la trasparenza delle sigle quando registrano i segnali dall'addestramento (differenza piccola, non misurata).
- `load_reader` rilegge una memoria senza la tabella per documento (voluto: un documento di produzione non è nell'addestramento). Per misurare un documento di addestramento con il lettore salvato si legge con la sua stessa annotazione in memoria: le misure si fanno col lettore costruito in memoria, come fa `ci_probe`.
- La classificazione umana/lettore della §3 è mia e non è stata verificata da un radiologo.

## 8. Come si ripete

```
ci-probe  (workflow)  oppure
python scripts/ci_probe.py --rows external_check.rows.json --craft ext/craft --medmentions ext/mm \
    --ncit ncit.obo --space space_open.npz --out ci_probe.json --markdown ci_probe.md
```
Test: `tests/test_ci_reader.py` (rule node, discourse, abbreviations: 5 test nuovi).
