# Supporto italiano del reticolo dei blocchi e del lettore: stato e piano

10 ottobre 2026. Frank ha chiesto di preparare il supporto italiano. Qui: che cosa c'è, che cosa è stato aggiunto oggi, che cosa manca e in che ordine si fa. Il linker legge già referti italiani (lessico, tabella delle parti, ruoli, discorso); ciò che è solo inglese è il **reticolo dei blocchi** (`chunk_lattice.py`) e il **lettore** (`ci_reader.py`), perché le loro memorie vengono da NCIt, MedMentions e CRAFT.

## 1. Che cosa cambia in italiano

| inglese | italiano | conseguenza |
|---|---|---|
| composti a testa destra: "heart rate" è un *rate* | testa a sinistra: "frequenza cardiaca" è una *frequenza*; il modificatore è un aggettivo relazionale ("cardiaca" = del cuore) o un complemento ("frequenza del cuore") | la parola che tipizza la frase è la **prima**, non l'ultima |
| "of the heart" | "di / del / della / dei / degli / delle / dell'" | altre espressioni regolari per il costrutto "di" |
| i verbi si riconoscono spesso dalla desinenza (-ed, -ly) | -ato, -ito, -uto sono anche nomi ("tessuto", "tratto", "stato") | senza un analizzatore grammaticale si filtrano solo parole chiuse e avverbi in -mente |
| il modificatore è un nome ("brain weight") | il modificatore è spesso un aggettivo ("epatico", "renale", "polmonare") che il lessico anatomico già conosce | la menzione da collegare può essere l'aggettivo |

## 2. Che cosa è stato aggiunto oggi (spento di default)

* `src/melampo/memory/grammar.py`: `Grammar.for_language("it")` con parole chiuse, avverbi in -mente, espressioni di "di/del" prima e dopo, `head_side = "left"`. Quella inglese ha le stesse parti con filtro di verbi e avverbi.
* `ChunkLattice(memory, grammar=Grammar.for_language("it"))`: il blocco è tipizzato dalla **prima** parola, la testa di una frase nominale dopo "di" è la sua prima parola, la testa prima di "del" è la prima della frase che precede, e un blocco che finisce con la menzione ma comincia con un'altra parola non è più "struttura per forza" ma prende il tipo della testa.
* Test su una memoria giocattolo (`tests/test_grammar_and_head_typing.py`): "La frequenza cardiaca è normale" → ruolo `inherent_location`; "Il prelievo del fegato" → `procedure_site`; la stessa frase letta con testa a destra non dà il ruolo.

Questo prova che il **meccanismo** funziona. Non prova nulla sull'italiano reale: nessuna memoria di blocchi italiana esiste ancora, quindi un reticolo italiano oggi non legge nulla (resta sul collegamento).

## 3. Che cosa manca, e di chi è

| manca | perché | chi / come |
|---|---|---|
| **gold set italiano** (span e ruolo) | senza non si può misurare nulla in italiano; i corpora pubblici sono inglesi | radiologi; è il pacchetto di `gold_set_guida_radiologi.md`, con referti italiani pseudonimizzati |
| **memoria di blocchi italiana** (`block_memory_it.json`): parola → tipo | le teste italiane ("frequenza", "funzione", "biopsia", "stent") non sono in NCIt in italiano | (a) UMLS: `UmlsHeadTyper` può leggere una fonte italiana (MeSH italiano, `MSHITA`) se la licenza UMLS dell'utente la copre: **da verificare**, non l'ho verificato; (b) una lista breve curata dai radiologi, che non siano gli stessi che scrivono il lessico |
| **nomi lunghi italiani** ("cuore artificiale", "pancreas artificiale") | `longer_names.json` viene da NCIt e da Protein Ontology in inglese | stessa lista curata; la traduzione automatica dei nomi inglesi propone, i radiologi decidono |
| **tracce e spazio semantico italiani** | il lettore impara da MedMentions (inglese) | si potranno scrivere dal gold set italiano e da referti non annotati; oggi non c'è nulla |
| **analizzatore grammaticale italiano** | la forma da sola non distingue nomi e participi | opzionale: `Grammar.verbal` è il punto dove collegare `it_core_news` (spaCy) o Stanza |

## 4. Proposta di partenza per la lista curata

Da validare dai radiologi: **non è nel repo come dato** perché chi scrive il lessico non può esserne anche il validatore (protocollo del gold set).

* procedura (ruolo `procedure_site`): biopsia, resezione, trapianto, prelievo, ablazione, drenaggio, asportazione, impianto, angioplastica, embolizzazione, intervento
* misura o funzione (ruolo `inherent_location`): frequenza, funzione, volume, diametro, spessore, dimensione, sviluppo, perfusione, flusso, gittata, pressione, tono
* dispositivo (ruolo `device_site`): stent, protesi, pacemaker, catetere, filtro, shunt, clip, drenaggio
* nome di altra cosa (ruolo `inside_a_name`): enzima, ormone, antigene, recettore, peptide, proteina, scala, indice, studio, registro

## 5. Ordine di lavoro

1. Referti italiani pseudonimizzati per il gold set (almeno 300 frasi per la prima misura) e giro di allineamento.
2. I radiologi correggono e completano la lista curata; la si scrive come `block_memory_it.json` con tipo e fonte per voce.
3. Si misura il reticolo italiano sul gold set: ruoli giusti e link giusti cambiati, con gli stessi criteri dell'inglese.
4. Solo se passa, il linker sceglie il reticolo dalla lingua della frase (`language_of`), come già sceglie il lessico.

Fino al punto 3 il reticolo italiano resta spento e il linker italiano si comporta come prima.
