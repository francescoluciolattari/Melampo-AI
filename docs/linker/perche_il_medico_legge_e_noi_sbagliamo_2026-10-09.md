# Perché il medico legge un quadro clinico senza problemi e noi sbagliamo: contesto, scomposizione, modo di lettura, o un altro modello?

9 ottobre 2026. Risposta alla domanda di Frank dopo il run `phrase-probe` con il segmentatore LLM (`docs/linker/lettura_del_sintagma_cervello_e_modelli_2026-10-09.md` §8.5). Qui "quadro" è letto come "quadro clinico". Per le fonti sul cervello e sull'esperto vale anche `comprensione_umana_linking_clinico_2026-10-06.md`, che non ripeto.

## 1. Risposta breve

1. **Non è il contesto, non è la scomposizione e non è l'LLM: è prima di tutto la memoria.** L'esperto non scompone: riconosce blocchi già conosciuti. Dei 33 errori misurati, 12 sono nomi composti che esistono in un vocabolario molto più grande del nostro (UMLS), 13 sono differenze di vocabolario o di granularità fra il nostro identificatore e quello dell'etichetta, 3 sono la nostra memoria troppo grossolana. Solo 2 chiedono un contesto fuori dalla frase.
2. **Il contesto serve, ma meno di quanto sembra, e in una forma precisa:** l'aspettativa del caso (esame, area, scopo) e il discorso intero. Per la frase attorno alla menzione il flusso di oggi lo usa già; per il documento intero no.
3. **Il modo di lettura è diverso, e qui c'è una differenza vera.** Il medico legge con uno scopo, in due direzioni (dall'insieme alla parte e ritorno) e rivede la prima ipotesi. Il nostro flusso parte dalla menzione e va verso il contesto, una volta, e non rivede.
4. **Un LLM diverso, da solo, non cambia nulla.** I due modelli provati concordano nel 79 % dei casi e mancano gli stessi errori. Quello che cambia il risultato è ciò che si dà al modello: definizioni da fonte curata, esempi già annotati dalla stessa convenzione, e un arbitro deterministico. È un esperimento da fare, con i criteri fissati prima (§6).
5. **Metà degli "errori" non sono errori di lettura.** Sono disaccordi con la convenzione di un corpus. Un medico, dato lo stesso testo e un altro vocabolario, "sbaglierebbe" nello stesso modo.

## 2. Che cosa ho misurato: perché sbagliamo (33 errori del controllo esterno)

Classificazione fatta a mano da me su tutti i 33 errori giudicati (22 MedMentions, 11 CRAFT), con frase, identificatore nostro, identificatore dell'etichetta e tipo semantico UMLS. Non ho accesso ai nomi dei CUI (licenza UMLS), quindi alcune assegnazioni sono inferenze dal tipo semantico e dal fatto che lo stesso CUI compare in frasi diverse. Un secondo lettore (un radiologo) dovrebbe rifarla.

| causa | n | casi | che cosa dovrebbe fare il lettore |
|---|---|---|---|
| **Nome composto più lungo, presente in UMLS** | 12 | "inferior vena cava filter placement" ×2, "heart of Maroilles cheese", "mucosa of the colon", "donor-specific spleen cells", "Living-Related Liver Donation", "developing brain" ×2, "development of pancreas", "heart interleukin-6", "brain white matter property", "Canine brain phantoms" | conoscere il nome composto. Non è una questione di grammatica |
| **Vocabolario o granularità dell'etichetta diversi dal nostro** | 13 | CRAFT: "bladder" ×3 (nostro `urinary_bladder`, etichetta UBERON:0018707), "right middle lobe" ×7 (nostro `lung_middle_lobe_right`, etichetta UBERON:0009912). MedMentions: "large bowel" (nostro colon), "colon" ×2 con CUI C3888384 di tipo T204 (forma di specie) | tradurre fra vocabolari. Non è comprensione |
| **La nostra memoria è troppo grossolana** | 3 | "aortic arch" ×3, risolto da noi come `aorta` (relazione parte-di) | sapere che l'arco aortico è un concetto a sé |
| **Serve il documento, non la frase** | 2 | "in the heart" ×2: la frase non dice che si parla di formaggio, il titolo o la frase precedente sì | leggere il discorso |
| **Etichetta anomala** | 3 | "the brain retained inside the cranium" etichettato come cranio (C0037303); "How the brain controls vigilance" (C0596948, T041); "brain parenchyma" nella frase del fantoccio (C0933845) | nessuno: è rumore o convenzione dell'annotatore |

Conclusioni dalla tabella.

- **Lettura vera (nome composto + documento): 14 su 33 (42 %).** Il resto (19 su 33) è vocabolario, memoria o rumore dell'etichetta.
- **Gli annotatori di MedMentions non leggevano come un medico: cercavano.** L'articolo dice che gli annotatori professionisti hanno cercato a mano i termini nel Metathesaurus UMLS 2017AA, scegliendo il concetto più specifico con la migliore corrispondenza e senza menzioni sovrapposte ([Mohan & Li, MedMentions](https://arxiv.org/pdf/1902.09476)). La verità di terreno è quindi "il nome più lungo che UMLS conosce", non "la parte del corpo di cui si parla". Questo spiega perché 12 errori hanno uno span più lungo di quello che NCIt contiene. L'articolo non dà un accordo fra annotatori, solo il 97,3 % fra revisori e annotatori su 469 concetti di otto riassunti.
- Per il prodotto la convenzione bersaglio è RadGraph e le classi di TotalSegmentator (`lettura_del_sintagma...` §7.2), non MedMentions né CRAFT. Una parte delle 16 voci di vocabolario non sarebbe un errore con il bersaglio giusto.

## 3. Che cosa fa il cervello e che cosa fa il nostro flusso

| meccanismo | fonte | il medico | il nostro flusso | che cosa dice la misura |
|---|---|---|---|---|
| **Memoria di lavoro a lungo termine** | Ericsson & Kintsch 1995, Psychological Review 102 (sintesi: [jimdavies.org](https://www.jimdavies.org/summaries/ericsson1995.html)) | l'esperto codifica il testo in una struttura di recupero nella memoria a lungo termine, già organizzata per categorie del campo (nella diagnosi medica i fatti sono richiamati per categorie significative). Non ha più capacità: ha più memoria utile | la memoria è NCIt + UBERON + 6.705 nomi lunghi; pochi nomi composti di procedura, dispositivo o misura (e quasi nessuno in italiano); nessuna struttura di recupero per tipo di esame | 12 errori su 33 sono assenza di un nome composto |
| **Previsione multilivello guidata dallo scopo** | Kuperberg & Jaeger 2016, Language, Cognition and Neuroscience 31: [pdf](https://kuperberg.mgh.harvard.edu/wp-content/uploads/kuperbergjaeger_lcn_15.pdf) | il contesto pre-attiva il significato a più livelli, nella misura in cui serve allo scopo del lettore; più candidati in parallelo, ciascuno con un grado di credenza | lo stato del referto (esame, area, lato) pre-attiva e vincola; non c'è un grado di credenza per candidato oltre il voto dei flussi | lo scopo "trovare la sede di una struttura" è fissato e cieco al tipo di cosa descritta |
| **Modello della situazione** | Zwaan & Radvansky 1998, Psychological Bulletin: [pdf](https://sites.ualberta.ca/~dmiall/Cognitive/Readings/Zwaan_Radvansky_1998.pdf) | il lettore costruisce una rappresentazione di ciò che il testo descrive (spazio, tempo, causa, intenzione, protagonisti), non della frase; l'integrazione è immediata | il flusso `discourse` esiste ma è locale; nessun modello del documento | 2 errori su 33 (formaggio) |
| **Un senso per discorso** | Gale, Church & Yarowsky 1992, DARPA Speech and Natural Language Workshop: [pdf](https://preview.aclanthology.org/fix_video/H92-1044.pdf) | una parola ripetuta nello stesso brano di solito ha lo stesso senso | non usato | applicabile ai 2 "heart" del formaggio |
| **Circolo ermeneutico** | [Stanford Encyclopedia of Philosophy, Hermeneutics](https://plato.stanford.edu/entries/hermeneutics/) | si capisce il testo intero per capire le parti e viceversa; le ipotesi iniziali ("pregiudizi" per Gadamer) sono proiezione e superamento ripetuti di interpretazioni inadeguate | una sola passata dalla menzione al contesto; la prima lettura vince | il reticolo non rivede: sceglie il cammino di costo minimo e si ferma |
| **Colpo d'occhio globale** | [Evans et al., Cognitive Research 2021](https://link.springer.com/article/10.1186/s41235-021-00339-5) | i radiologi valutano le immagini anomale come più anomale delle normali a esposizione breve e illimitata (sopra il caso); il segnale è debole e più tempo non lo migliora in modo affidabile | non c'è uno stadio di "di che cosa parla questo referto" prima della lettura locale se non lo stato del referto | il gist orienta, non decide |

Due precisazioni di onestà.

- Il colpo d'occhio dei radiologi è **sopra il caso ma debole**. Non lo uso come prova che il gist da solo basti.
- Ericsson e Kintsch parlano di esperti in generale e citano la diagnosi medica fra gli esempi. Che i radiologi usino una struttura di recupero per i nomi composti dei referti è un'inferenza, non una misura.

## 4. Le tre domande di Frank

**È il contesto che fa la differenza?** In parte, e di due tipi diversi. (1) L'aspettativa del caso (esame, area, scopo): c'è già nello stadio 1 e nei flussi di frame. (2) Il discorso intero: manca. Pesa su 2 errori su 33 in questi corpora, ma nei referti reali conta di più (lato ereditato dalla frase precedente, "la lesione", stesso termine ripetuto). Va misurato sui referti, non sugli articoli.

**Il metodo di scomposizione del testo?** Il cervello non scompone prima di conoscere: riconosce blocchi già in memoria e compone solo ciò che non conosce (Ericsson & Kintsch; Christiansen & Chater, già nel §3 del documento sul sintagma). Il reticolo fa lo stesso come disegno, ma la sua memoria è piccola. Migliorare l'algoritmo di segmentazione senza ingrandire e curare la memoria sposta pochi errori: lo dice anche il segmentatore LLM, che trova i nomi composti ma li trova perché li conosce (§8.5).

**La modalità di lettura?** Qui c'è la differenza più grande ed è quella che non abbiamo ancora provato. Il medico legge in due direzioni, con uno scopo, e rivede. Noi leggiamo dalla menzione verso l'esterno, una volta, con uno scopo unico, e accettiamo la prima lettura stabile. Il circolo ermeneutico e la previsione guidata dall'utilità (Kuperberg & Jaeger) dicono la stessa cosa: capire è proporre, controllare e superare ipotesi, non estrarre.

## 5. Serve un LLM diverso o un modello nuovo?

Quello che sappiamo:

- I due modelli provati danno lo stesso tipo nel 79 % dei casi (263 su 331) e mancano gli stessi errori (nessuno dei due vede l'anatomia semplice dell'arco aortico, del colon o del cervello in via di sviluppo). Un terzo LLM generico produrrebbe con alta probabilità lo stesso schema. L'errore congiunto di LLM diversi è alto: circa il 60 % delle volte scelgono la stessa risposta sbagliata ([Kim et al., ICML 2025](https://arxiv.org/html/2506.07962v1), nel documento di ottobre 6).
- Che cosa cambia l'esito, secondo la letteratura:
  - Definizioni da fonte curata: aggiungere definizioni UMLS ai prompt ha dato un miglioramento relativo medio del 15 % per GPT-4 su sei set biomedici di NER; le definizioni generate dallo stesso GPT-4 hanno dato poco o niente, e su CDR meno del livello senza definizioni ([arXiv 2404.00152](https://arxiv.org/html/2404.00152v2)). La conoscenza deve venire da fuori, non dal modello.
  - Esempi annotati con la stessa convenzione: nel compito francese EvalLLM 2025 (NER, 40 documenti di addestramento) GPT-4.1 con un riassunto della guida di annotazione e 10 esempi scelti per somiglianza ha ottenuto micro-F1 75,79 contro 65,22 di GLiNER fine-tuned con verifica ([arXiv 2510.03577](https://arxiv.org/html/2510.03577v1)). Con pochi dati annotati vince il prompt con esempi.
  - Molti dati annotati: un lavoro del 2022 su NER e relazioni biomediche trova che GPT-3 in-context resta sotto un piccolo modello fine-tuned, anche con selezione dinamica degli esempi ([Gutiérrez et al., Findings of EMNLP 2022](https://preview.aclanthology.org/author-url/2022.findings-emnlp.329/)). È un modello più vecchio, quindi il confronto va rifatto.
- Quindi la scelta dipende da quanto gold avremo. Oggi poco: il modello giusto è **un LLM che legge con in mano le definizioni e gli esempi di convenzione recuperati dalla nostra memoria, controllato da un arbitro deterministico**. Quando avremo il gold set dei radiologi (1.500-2.000 menzioni) si confronta con un piccolo modello fine-tuned. Nessun modello nuovo "universale" è indicato dalla ricerca.

## 6. Proposta: lettura a due direzioni, con memoria e casi, controllata

Disegno (non realizzato in questa consegna):

1. **Memoria prima** (ingrandire e curare): nomi composti di procedura, dispositivo, misura e tessuto, in inglese e in italiano, da una fonte con licenza chiara (UMLS è la fonte da cui vengono le etichette di MedMentions; la licenza per un dispositivo medico va verificata). Con il tipo e il ruolo approvati da radiologi.
2. **Traduzione fra vocabolari** come passaggio deterministico fra i nostri identificatori e quelli del corpus o del bersaglio (RadGraph, TotalSegmentator): toglie dalla metrica i 13 errori di vocabolario e permette di vedere quelli veri.
3. **Gist e scopo prima della menzione**: un passaggio che legge titolo, impressione e frase e dice "di che cosa parla" (strutture, procedure, dispositivi, alimenti), usato come aspettativa; il documento intero per il senso unico per discorso.
4. **Lettura a due direzioni con revisione**: la menzione propone, il contesto controlla, l'arbitro (reticolo e regole) accetta, ripiega o si astiene; se c'è conflitto si ripropone con il candidato successivo, non si sceglie il primo stabile.
5. **LLM come lettore con strumenti**, non come giudice: gli si danno definizioni dalla memoria e N esempi annotati simili (da uno split mai usato per misurare); risponde con un blocco e un ruolo; un altro meccanismo lo controlla.

Esperimenti da fare, in ordine di costo, con i criteri fissati qui prima di guardare i risultati.

| # | esperimento | costo | criterio di riuscita |
|---|---|---|---|
| E1 | ripetere il controllo esterno confrontando a livello di concetto (sinonimi, genitori) invece che di identificatore | gratis | si vede quanti dei 13 errori di vocabolario spariscono; non è un miglioramento del linker, è una correzione della misura |
| E2 | `phrase-probe` con i due LLM che ricevono definizioni NCIt dei candidati e 5 esempi MedMentions dello split di sviluppo (quelli del test escluse) | circa 670 chiamate | almeno 12 errori MedMentions su 22 trovati con al più 3 % di link giusti segnalati, oppure ruolo giusto in almeno il 70 % dei casi (gli stessi criteri del run precedente) |
| E3 | flusso di documento: stesso termine nello stesso referto, senso unico per discorso | gratis, ma servono referti: iu-xray e E3C | non peggiora nessun link giusto; trova almeno i casi di lato ereditato |
| E4 | memoria curata di nomi composti con approvazione dei radiologi | richiede radiologi | errori con span lungo scendono sotto un terzo, senza perdere link giusti |

Il risultato di E2 decide se serve un modello più grande: se anche con definizioni e esempi i due LLM non superano il criterio, la risposta non è un LLM migliore ma più memoria (E4) e un gold set.

## 7. Limiti di quanto scritto

- 33 errori, di cui 22 in MedMentions; la classificazione è mia e non è stata verificata da un secondo lettore.
- Articoli, non referti: il peso del contesto di documento nei referti reali è sconosciuto.
- Alcune fonti sono sintesi (Ericsson & Kintsch è letto dalla sintesi di jimdavies.org, non dall'originale; un articolo di Drew et al. 2013 non è stato leggibile e non è usato).
- L'evidenza sul confronto fra LLM e modelli piccoli è su NER, non su entity linking, e su tipi di entità diversi dai nostri.
- Il termine "senso unico per discorso" è citato come ipotesi: la fonte letta non dà un tasso generale.

## Fonti

- Mohan S, Li D. MedMentions: a large biomedical corpus annotated with UMLS concepts. arXiv 1902.09476. https://arxiv.org/pdf/1902.09476
- Ericsson KA, Kintsch W. Long-term working memory. Psychological Review 102(2), 1995. Sintesi: https://www.jimdavies.org/summaries/ericsson1995.html
- Kuperberg GR, Jaeger TF. What do we mean by prediction in language comprehension? Language, Cognition and Neuroscience 31, 2016. https://kuperberg.mgh.harvard.edu/wp-content/uploads/kuperbergjaeger_lcn_15.pdf
- Zwaan RA, Radvansky GA. Situation models in language comprehension and memory. Psychological Bulletin 123, 1998. https://sites.ualberta.ca/~dmiall/Cognitive/Readings/Zwaan_Radvansky_1998.pdf
- Gale W, Church K, Yarowsky D. One sense per discourse. DARPA Speech and Natural Language Workshop 1992. https://preview.aclanthology.org/fix_video/H92-1044.pdf
- Stanford Encyclopedia of Philosophy, Hermeneutics. https://plato.stanford.edu/entries/hermeneutics/
- Evans et al. Cognitive Research: Principles and Implications 2021. https://link.springer.com/article/10.1186/s41235-021-00339-5
- On-the-fly definition augmentation of LLMs for biomedical NER. arXiv 2404.00152. https://arxiv.org/html/2404.00152v2
- Sistema francese per la sfida EvalLLM 2025 (NER biomedico e eventi sanitari). arXiv 2510.03577. https://arxiv.org/html/2510.03577v1
- Gutiérrez BJ et al. Thinking about GPT-3 in-context learning for biomedical IE? Think again. Findings of EMNLP 2022. https://preview.aclanthology.org/author-url/2022.findings-emnlp.329/
- Kim E et al. Correlated errors in large language models. ICML 2025. https://arxiv.org/html/2506.07962v1
