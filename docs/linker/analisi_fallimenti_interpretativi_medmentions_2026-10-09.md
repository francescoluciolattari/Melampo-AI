# Perché il linker sbaglia su MedMentions: dove fallisce il flusso e dove il cervello umano fa la differenza — 2026-10-09

Domanda di Frank: capire gli errori di interpretazione su MedMentions, analizzando con un modello dove il flusso fallisce, con che cosa si capisce quali strategie e modelli funzionano e quali no, e dove il cervello umano fa la differenza nel comprendere contesto e termini.

Stato misurato (commit fissati, verify spento, regola del progetto `by_project_rule`): **MedMentions 22 errori su 732 link giudicati (96,99%)**, CRAFT 11 su 1.326 (99,17%). Dopo F1 con NCIt tipizzato (§4) sono 3 errori in meno e 0 link giusti persi; i referti reali non cambiano.

## 1. Che cosa uso per capire (gli strumenti, e che cosa dicono)

| Strumento | A che cosa serve | Limite |
|---|---|---|
| Controllo esterno (`external_check.py`) con regola del progetto | conta accordi ed errori, riga per riga, su etichette di altri | l'oro di MedMentions etichetta il concetto UMLS più specifico del sintagma, senza sovrapposizioni, non la sede anatomica: una parte degli "errori" è convenzione |
| Confronto per riga prima/dopo (insieme di accettati per chiave corpus-documento-menzione-frase) | dice quali errori una regola ferma e quali link giusti perde | richiede commit fissati |
| Segnali con AUC e punti di lavoro (`head_probe`) | misura se un segnale (testa sintattica, attenzione, dominio) separa errori da link giusti | solo ordine relativo, non certifica |
| `verify` con due modelli, prompt generico | prova se un modello vede l'errore leggendo la frase | vedi §3 |
| Lettore cieco indipendente (un modello senza il nostro codice, 122 schede: 22 errori + 100 link giusti) | misura che cosa si capisce dal solo testo, senza conoscere l'etichetta | è ancora un modello, non un radiologo |
| Profilo dei flussi dell'evidenza (nome, cieco, discorso, conflitti) | vede se l'errore ha un profilo diverso dai link giusti | nessuna differenza (§2) |
| Tipo semantico dell'oro | dice che cosa l'annotatore stava etichettando | mostra la convenzione, non il giusto |

## 2. Dove il flusso fallisce: gli errori hanno lo stesso profilo dei link giusti

Il profilo dei flussi è identico: 15 errori su 22 hanno esattamente il supporto `(`name`, `blind`, `discourse`)`, lo stesso dei link giusti; nessun errore ha conflitti. Il nome dice "heart", la lettura cieca dice "cuore", il discorso (altre strutture nel documento) sostiene. **Non c'è un flusso di evidenza che "si accorge" dell'errore**, perché tutti i flussi leggono la parola e non il sintagma intero. È la stessa cosa che si vede nei segnali sintattici (AUC 0,48–0,58): il fallimento sta prima, nel modo di costruire la menzione, non nella fiducia.

Gli errori rimasti per causa e per stadio che dovrebbe fermarli:

| Menzione (frase) | Che cosa è in realtà | Stadio che dovrebbe fermarlo | Fermabile in generale? |
|---|---|---|---|
| inferior vena cava filter placement ×2 | dispositivo + procedura | lettura del sintagma (testa) | sì, con un vocabolario di dispositivi (non NCIt: non è lì) |
| donor-specific spleen cells transfusion | trasfusione di cellule | testa: "transfusion" | sì con un elenco di procedure da ontologia |
| Liver Donation | donazione | testa: "donation" | idem |
| heart interlukine-6 | refuso di "interleukin" | ortografia | no (refuso) |
| simulated colon microbiome | microbioma | testa | sì con vocabolario di microbiologia |
| brain phantoms | oggetto di prova | testa | sì (phantom) |
| brain white matter property | granularità | oro | no (convenzione) |
| heart/Maroilles cheese ×3 | formaggio | dominio del documento | no nel dominio clinico (fuori dominio) |
| aortic arch ×2 | mappatura UMLS → UBERON | oro | no (mappatura) |
| colon ×2 | rumore dell'oro | oro | no |
| developing brain ×2 | stadio di sviluppo | convenzione | decisione dei radiologi |
| large bowel, parenchyma | granularità | oro | decisione dei radiologi |
| brain controls (verbo), development of pancreas helps (verbo) | processo non preso | parser | sì con un parser, non con le regole attuali |

Totale: circa 7 composti ancora aperti, 3 formaggio, 2 mappatura, 2+ rumore dell'oro, 4 granularità, 2 "developing", 2 processo con verbo, 1 refuso.

## 3. Che cosa ha funzionato e che cosa no

**Ha funzionato (misurato, errori fermati senza perdere link giusti o con perdita registrata):**
- Trattino e sigle definite nel testo (Schwartz–Hearst), regola con testa non sede: F2/F3.
- Sinonimi UBERON come nomi più lunghi, senso innesto/donazione: F4.
- Processo e funzione dopo o prima della struttura, con teste da Gene Ontology: F6 (MedMentions 38 → 25).
- Nomi tipizzati da NCIt (proteine, geni, sostanze chimiche, strumenti di valutazione): F1 (MedMentions 25 → 22). I tipi vengono dall'ontologia, non da elenchi di parole.

**Non ha funzionato (e perché):**
- `verify` con due modelli e prompt generico: CRAFT 0 errori fermati su 11, MedMentions 2 su 38, ma ferma anche link giusti. Il modello legge la frase senza la convenzione di etichettatura e senza il sintagma intero.
- Segnali sintattici della testa (AUC 0,48–0,58): un composto semplice ("prostate symptom score") è sintatticamente un sintagma nominale come "prostate gland".
- Veto per dominio del documento: 25 link su 28 in CRAFT sono di dominio "adulto", niente separa il formaggio.
- Regola globale "testa non sede": segnala solo 32 link giudicati (30 giusti, 2 errori).

**Lettore cieco indipendente** (modello senza accesso al codice, 122 schede): 99 link giusti su 100 marcati corretti (1 messo in dubbio); degli 22 errori ne segnala 6 (fantocci, filtro della vena cava ×2, formaggio ×3) e ne dà per buoni 16. I 16 che il lettore dà per buoni sono quasi tutti convenzione dell'oro o richiedono conoscenza che la frase non contiene. Conclusione: **un secondo modello da solo non basta**; l'errore che resta non si vede dal solo testo senza la convenzione di etichettatura.

## 4. F1 con NCIt tipizzato (questa consegna)

`scripts/build_longer_names.py` costruisce `data/linking/longer_names.json` dalla NCIt (release 2024-05-07) con questi tipi: strumento di valutazione 449, sostanza chimica 489, proteina 927, gene 571 (2.436 nomi). Esclusi le domande e le risposte di questionario, i codici CDISC e i nomi con parola anatomica solo come qualificatore. Il linker (`LongerNames.containing`) cerca in finestre fino a 9 parole che contengono la menzione, vince il nome più lungo, e pone il veto `mention_is_inside_the_name_of_another_thing:<tipo>:<nome>`.

Misura: MedMentions 25 → 22 errori (prostate symptom score ×2, Liver Fatty Acid Binding Protein Deficiency), 0 link giusti persi; CRAFT invariato; referti reali invariati; bench identico.

**Protein Ontology.** Dalla sessione non si scarica: il file va costruito dal workflow `longer-names` su GitHub (indirizzo di default non verificato). Il builder la usa già (`--pr`), ma `longer_names.json` attuale contiene solo NCIt + UBERON.

## 5. Dove il cervello umano fa la differenza (rivisto dopo le domande di Frank)

**1. Legge il sintagma intero come un concetto.** È il compito di F1, e non è una facoltà fuori portata: è riconoscimento di concetti con tipo (§6 elenca i modelli che lo fanno). Quello che mancava al linker è un flusso che legga il sintagma e dica che tipo di cosa è.

**2. Entità con nome, tema del documento, verbi: è contesto.** Frank ha ragione, e il linker ha già un flusso "discorso", ma oggi legge solo le *altre strutture anatomiche* del documento. Non legge che "Maroilles", "rind", "MRS agar", "MALDI-TOF" sono entità di un altro tipo (formaggio, terreno di coltura, tecnica di laboratorio). Il contesto ha tre parti diverse: (a) il tipo delle altre entità nella frase e nel documento, (b) il tema del documento, (c) la struttura della frase (verbi e relazioni: "how the brain *controls*"). Il profilo lessicale del documento (F5) ha fallito sul formaggio perché misurava adulto/pediatrico, non il tema; il contesto tipizzato da entità è un'altra misura e non è ancora provata. Nella NCIt "Cheese" esiste come classe (C178207), quindi i tipi delle parole vicine si possono leggere da un'ontologia.

**3. I refusi: sì, un modello li corregge** (un LLM o un modello di linguaggio mascherato; esiste lavoro pubblicato sulla correzione dei refusi nei testi medici con modelli mascherati, BMC Bioinformatics 2022: ho letto solo il titolo, l'articolo non era raggiungibile). Due precisazioni: (i) nel caso concreto la correzione non basterebbe: scritto bene, "heart interleukin-6" resta un caso in cui l'oro etichetta tutto il sintagma come misura di laboratorio (tipo T059); il refuso non è la causa dell'errore; (ii) in un dispositivo medico una correzione silenziosa è un'inferenza: va registrata come ipotesi con il testo originale accanto. Il modo più verificabile è la distanza di edit dal vocabolario ("interlukine" dista poche modifiche da "interleukin"), con un modello solo per scegliere tra più candidati.

**4. Lo scopo dell'etichetta, spiegato.** Ogni corpus risponde a una domanda diversa, e la stessa frase riceve etichette diverse.
- *MedMentions* serve a riconoscere concetti UMLS in un abstract. Le istruzioni agli annotatori (Mohan e Li) sono: "annotate the most specific concept for each mention, without any overlaps". In "heart interleukin-6 levels" il concetto più specifico è la misura; "heart" non riceve etichetta propria. In "heart development" l'etichetta va al processo.
- *CRAFT* (parte anatomica) serve a trovare ogni menzione di struttura anatomica in UBERON, anche annidata: la guida dice che si annota il termine incluso quando la sua parola centrale è diversa da quella del sintagma che lo contiene ([Gold-standard ontology-based anatomical annotation in the CRAFT Corpus](https://pmc.ncbi.nlm.nih.gov/articles/PMC7243923)). In "heart development" la parola centrale è "development", quindi "heart" riceve l'etichetta dell'organo.
- *Il nostro scopo* è dire in quale struttura sta un reperto di un referto. Una parola anatomica che qualifica un processo, una misura, un dispositivo o una proteina non dice dove sta il reperto; dice di cosa si parla. Per questo F6 non fa il link e registra la struttura con un ruolo (`inherent_location`); il progetto ha anche il ruolo `procedure site`.

Conseguenza che finora non avevo detto: **una parte dei composti "ancora aperti" non è un errore per il prodotto se il ruolo è registrato**: "inferior vena cava filter placement" o "Liver Donation" hanno la struttura come sede della procedura. L'errore c'è solo se quel link entra in un campo "sede del reperto". Il controllo esterno oggi conta il link come errore perché l'oro etichetta la procedura; per il prodotto la misura giusta è "il ruolo registrato è giusto?", e questo non l'ho ancora misurato (§8, punto 1). Il cervello umano fa la differenza qui perché conosce lo scopo; la macchina lo riceve solo se è scritto nei dati e nel gold set dei radiologi.

## 6. Modelli che leggono il sintagma intero: che cosa esiste e che cosa vale

Premessa: i modelli di *entity linking* (SapBERT, KRISSBERT, arboEL) ricevono la menzione già delimitata e scelgono il concetto; su MedMentions con menzioni d'oro arboEL ha richiamo@1 0,69 e SciSpacy 0,58 ([BELB](https://arxiv.org/pdf/2308.11537)); KRISSBERT dichiara circa 58,3% di accuratezza top-1 ([scheda del modello](https://huggingface.co/microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL), licenza MIT). Non decidono il confine del sintagma e sono lontani dal 99%: non sono la risposta a F1. Servono modelli che *delimitano e tipizzano* il sintagma:

| Famiglia | Modello | Come legge il sintagma | Limiti |
|---|---|---|---|
| A. Riconoscitore di intervalli con tipo, zero-shot | GLiNER-BioMed ([Ihor/gliner-biomed-bi-small-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-small-v1.0), Apache-2.0, [articolo](https://arxiv.org/abs/2504.00676)) | i tipi si danno in linguaggio naturale ("anatomical structure", "medical device", "procedure", "protein", "assessment scale", "food"); restituisce l'intervallo con tipo e punteggio | solo inglese; F1 medio 56,9 su 8 corpora (scheda): è un segnale, non un giudice; l'articolo non dice nulla sugli intervalli annidati |
| B. Recupero del sintagma contro le ontologie | codificatore tipo SapBERT/KRISSBERT sui gruppi di parole che contengono la menzione, contro NCIt + UBERON | generalizza `longer_names` alle forme non elencate ("spleen cells transfusion" ≈ trasfusione di cellule) | rischio di falsi veti su "liver biopsy"; serve una soglia calibrata |
| C. LLM come segmentatore | un modello a cui si chiede di dividere la frase in concetti, ciascuno con un tipo di un elenco chiuso, e di dire quale contiene la parola | è l'operazione che fa l'umano | mai provato in questa forma: il `verify` provato chiedeva "il link è giusto?", non "segmenta" |
| D. Tipo della testa da ontologia | ultime parole dei nomi NCIt per classe (procedura, dispositivo…), come per `process_heads` da GO | deterministico e verificabile | copre solo teste frequenti; il confine con i casi veri (liver biopsy) è il ruolo (§5.4) |

PubTator 3.0 ([articolo](https://arxiv.org/pdf/2401.11048)) copre gene, malattia, sostanza chimica, variante, specie e linea cellulare (tipi da verificare sull'articolo, che non ho letto per intero): serve per proteine e sostanze, già coperte da NCIt e Protein Ontology; non per dispositivi e procedure.

**Esperimento proposto: `phrase-probe`** (come `head-probe`, lo lancia Frank da GitHub, perché Hugging Face non è raggiungibile dalla mia sessione): sui 22 errori e sui link giusti di MedMentions e CRAFT misura A, B, C e D come AUC e punti di lavoro (errori fermati per link giusti persi), tenendo fuori dal calcolo i composti con ruolo registrato. Aspettative oneste: nessuno di questi modelli, preso da solo, arriva al 99%; la strada del progetto resta flussi indipendenti che si confrontano, con astensione al conflitto. Con 22 errori l'intervallo di confidenza è largo: il risultato va ricontrollato su una parte tenuta ferma (split di test congelato).

## 7. Che cosa significa per l'obiettivo del 99% su MedMentions

- Il tetto realistico senza aggiudicazione indipendente è circa il 98%: restano 7 composti aperti, ma 11 casi su 22 sono gold-side o fuori dominio (formaggio, mappatura UMLS, rumore, granularità).
- **Un 99% su un campione non è un 99% certificato**: con 735 link e 7 errori il limite superiore al 95% è circa 1,9%.
- Il numero che conta per il prodotto è il gold set dei radiologi (due letture cieche + aggiudicatore); con il metodo attuale servono almeno 299 link senza errori per una certificazione al 99% a livello 95%.
- Per non sovradattare, **congelare lo split ufficiale test di MedMentions** (trng 456 errori/17, dev 153/2, test 126/6) e sviluppare solo su trng+dev.

## 8. Prossimi passi

1. Misurare MedMentions anche con la regola del ruolo (il ruolo registrato è giusto?), non solo con l'etichetta di link.
2. Costruire e lanciare `phrase-probe` (§6): GLiNER-BioMed, recupero del sintagma, LLM segmentatore, tipo della testa.
3. Lanciare `longer-names` su GitHub per aggiungere Protein Ontology e sostituire `data/linking/longer_names.json` con l'artefatto.
4. Aggiungere un vocabolario di dispositivi e procedure da ontologia (da NCIt, dove le classi di dispositivo e procedura esistono, oppure da una fonte con licenza che Frank indichi; non ancora verificato).
5. Decisione dei radiologi su "developing brain" e granularità (F7).
6. Congelare lo split di test di MedMentions.
