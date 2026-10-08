# Perché il linker sbaglia su MedMentions: dove fallisce il flusso e dove il cervello umano fa la differenza — 2026-10-09

Domanda di Frank: capire gli errori di interpretazione su MedMentions, analizzando con un modello dove il flusso fallisce, con che cosa si capisce quali strategie e modelli funzionano e quali no, e dove il cervello umano fa la differenza nel comprendere contesto e termini.

Stato misurato (commit fissati, verify spento, regola del progetto `by_project_rule`): **MedMentions 22 errori su 732 link giudicati (96,99%)**, CRAFT 11 su 1.326 (99,17%). Dopo F1 con NCIt tipizzato (§4) sono 3 errori in meno e 0 link giusti persi; i referti reali non cambiano.

## 1. Che cosa uso per capire (gli strumenti, e che cosa dicono)

| Strumento | A che cosa serve | Limite |
|---|---|---|
| Controllo esterno (`external_check.py`) con regola del progetto | conta accordi ed errori, riga per riga, su etichette di altri | l'oro di MedMentions etichetta il concetto UMLS più lungo, non la sede anatomica: una parte degli "errori" è convenzione |
| Confronto per riga prima/dopo (insieme di accettati per chiave corpus-documento-menzione-frase) | dice quali errori una regola ferma e quali link giusti perde | richiede commit fissati |
| Segnali con AUC e punti di lavoro (`head_probe`) | misura se un segnale (testa sintattica, attenzione, dominio) separa errori da link giusti | solo ordine relativo, non certifica |
| `verify` con due modelli, prompt generico | prova se un modello vede l'errore leggendo la frase | vedi §3 |
| Lettore cieco indipendente (un modello senza il nostro codice, 122 schede: 22 errori + 100 link giusti) | misura che cosa si capisce dal solo testo, senza conoscere l'etichetta | è ancora un modello, non un radiologo |
| Profilo dei flussi dell'evidenza (nome, cieco, discorso, conflitti) | vede se l'errore ha un profilo diverso dai link giusti | nessuna differenza (§2) |
| Tipo semantico dell'oro | dice che cosa l'annotatore stava etichettando | mostra la convenzione, non il giusto |

## 2. Dove il flusso fallisce: gli errori hanno lo stesso profilo dei link giusti

Il profilo dei flussi è identico: 15 errori su 22 hanno esattamente il supporto `('name', 'blind', 'discourse')`, lo stesso dei link giusti; nessun errore ha conflitti. Il nome dice "heart", la lettura cieca dice "cuore", il discorso (altre strutture nel documento) sostiene. **Non c'è un flusso di evidenza che "si accorge" dell'errore**, perché tutti i flussi leggono la parola e non il sintagma intero. È la stessa cosa che si vede nei segnali sintattici (AUC 0,48–0,58): il fallimento sta prima, nel modo di costruire la menzione, non nella fiducia.

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

## 5. Dove il cervello umano fa la differenza

1. **Legge il sintagma come un concetto intero.** Davanti a "inferior vena cava filter placement" un radiologo vede un'unica procedura; il linker vede quattro parole e una struttura. L'oro di MedMentions fa lo stesso: etichetta il concetto UMLS più lungo.
2. **Conosce le entità con nome e i loro tipi** (proteina, test, dispositivo). Noi lo sostituiamo con le ontologie: funziona dove l'ontologia copre (NCIt per proteine, test), non dove manca (dispositivi, procedure rare, microbioma).
3. **Legge l'argomento del documento** ("Maroilles", "rind", "LAB agar" → latteria). Noi non lo facciamo e il prodotto clinico non ne ha bisogno: il tipo di documento si decide a monte.
4. **Usa la sintassi e i verbi** ("how the brain controls" = verbo). Con le regole di adiacenza non lo facciamo; serve un parser, e il parser da solo non separa i casi (AUC ~0,55).
5. **Corregge i refusi** ("interlukine").
6. **Conosce lo scopo dell'etichetta.** Un annotatore di MedMentions etichetta il processo, uno di CRAFT l'organo. Il cervello umano sa quale scopo serve; la macchina deve essere istruita (qui lo scopo è la sede anatomica del reperto, e vince la scelta fatta in F6).

## 6. Che cosa significa per l'obiettivo del 99% su MedMentions

- Il tetto realistico senza aggiudicazione indipendente è circa il 98%: restano 7 composti aperti, ma 11 casi su 22 sono gold-side o fuori dominio (formaggio, mappatura UMLS, rumore, granularità).
- **Un 99% su un campione non è un 99% certificato**: con 735 link e 7 errori il limite superiore al 95% è circa 1,9%.
- Il numero che conta per il prodotto è il gold set dei radiologi (due letture cieche + aggiudicatore); con il metodo attuale servono almeno 299 link senza errori per una certificazione al 99% a livello 95%.
- Per non sovradattare, **congelare lo split ufficiale test di MedMentions** (trng 456 errori/17, dev 153/2, test 126/6) e sviluppare solo su trng+dev.

## 7. Prossimi passi

1. Lanciare `longer-names` su GitHub per aggiungere Protein Ontology e sostituire `data/linking/longer_names.json` con l'artefatto.
2. Aggiungere un vocabolario di dispositivi e procedure da ontologia (da NCIt, dove le classi di dispositivo e procedura esistono, oppure da una fonte con licenza che Frank indichi; non ancora verificato).
3. Decisione dei radiologi su "developing brain" e granularità (F7).
4. Congelare lo split di test di MedMentions.
