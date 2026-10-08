# Analisi degli errori del controllo esterno (CRAFT, MedMentions) — 2026-10-08

Corsa completa rifatta il 2026-10-08 su CRAFT v5.1.0 e MedMentions (commit pinnati), linker del branch `feat/linker-nine` (c72e932), senza chiamate ai modelli (`verify` spento). Numeri: CRAFT 1.352 giudicati, 13 errori (99,04%); MedMentions 768 giudicati, 51 errori (93,4% con la regola del progetto). Tutti i 64 casi sono elencati sotto.

## 1. Sintesi per causa

**CRAFT (13)**: 7 `right middle lobe` e 3 `bladder`: l'oro usa l'etichetta generica ("anatomical lobe", "bladder organ"), il linker è più specifico (lobo medio del polmone destro, vescica urinaria). Non è una struttura sbagliata ma una differenza di granularità dell'oro. 3 `aortic arch`: in UBERON "aortic arch" è sinonimo EXACT di *pharyngeal arch artery* (embrione) e RELATED di *arch of aorta* (adulto); nel testo erano archi embrionali ("fourth aortic arch artery defects", "aortic arch arteries"), il linker ha scelto l'aorta adulta. Questi 2-3 sono errori veri.

**MedMentions (51)**, secondo la mia lettura:

| Causa | Casi |
|---|---|
| A nome composto (molecola, score, procedura, asse, abbreviazione) | 16 |
| B1 processo/funzione dell'organo | 16 |
| B2 misura/campo (non anatomia) | 2 |
| C senso "tessuto per innesto" | 4 |
| D dominio non medico (formaggio) | 3 |
| F granularità (testo nomina una parte o un insieme) | 4 |
| G rumore/convenzione dell'oro | 4 |
| H arco aortico: mappatura UMLS→UBERON dell'oro | 2 |

Lettura onesta: 25 sono mancati veri (A 16 + B2 2 + C 4 + D 3); 16 sono la decisione aperta su processo/funzione (B1); 4 sono granularità; 4 sono rumore dell'oro; 2 sono un limite della mappatura UMLS→UBERON (il linker ha ragione). Intervallo di precisione su 768: 96,7% (solo i 25 veri) – 94,1% (veri + B1 + F). La cifra 93,4% dello script conta tutto.

## 2. Perché i meccanismi attuali non li fermano

Riprodotti uno a uno sul linker: tutti `accepted` con `blind=support`, `name`, `discourse`. Tre lacune generali:
1. La testa del composto è cercata solo nella parola adiacente (`_head`). "liver **fatty acid binding** protein", "brain **natriuretic** peptide", "prostate **acid** phosphatase", "inferior vena cava **filter placement**", "prostate **symptom** score" hanno la testa 2-4 parole dopo.
2. Il trattino non conta: "gut-brain", "gut-liver", "pro-brain" sono parole unite, ma `locate` vede il trattino come confine di parola.
3. Nessun controllo sulle abbreviazioni definite nel testo: "TWIK-related spinal cord K(+) (TRESK) channel", "drug effects on the nervous system (DENS)", "N-terminal pro-brain natriuretic peptide [NT-proBNP]", "liver fatty acid binding protein (L-FABP)", "donor-specific spleen cell transfusion (DST)", "auditory brainstem response (ABR)": il nome dell'organo sta dentro l'espansione di un'abbreviazione che il testo stesso definisce.

Il profilo di convergenza (`support`, `conflicts`, `blind`) è identico a quello dei link giusti, come già detto: il lettore cieco legge le stesse lettere.

## 3. Correzioni proposte (universali)

| # | Meccanismo | Copre (indici MedMentions / CRAFT) |
|---|---|---|
| F1 | Testa del composto entro 4 parole nello stesso sintagma (stop a punteggiatura, preposizione, verbo), con liste `attribute_heads`/`procedure_heads` ampliate (protein, peptide, score, filter, placement, transfusion, axis, model, cytokine/IL-n, tolleranza ai refusi) | 0,1,3,4,5,14,16,19,20,27,28,42 |
| F2 | Mention unita da trattino a un'altra parola (non numero, non lato, non nome noto) = parte di un composto | 8,16,23 |
| F3 | Abbreviazione definita nel testo: se il nome sta dentro l'espansione di "forma lunga (SIGLA)" e l'espansione non coincide con il nome, la mention è parte di un altro nome | 9,18,20,23,28,41 |
| F4 | Senso "tessuto per innesto/donazione": autologous, homologous, allogeneic, irradiated, donor(-specific), donation, graft, transplantation | 44-47, 25 |
| F5 | Archi embrionali: "aortic arch artery/arteries", "fourth aortic arch", "PAA", contesto di sviluppo → senso *pharyngeal arch artery*, altrimenti astensione | CRAFT 3 aortic arch |
| F6 | Testa di processo/funzione (development, developing, circuits, connectivity, dynamics, arousal, state, response, research) per direzione: `response` solo a destra | 2,6,15,21,22,26,29,30,31,32,34,35,37,38,39,41,43 (**dipende dalla decisione B1**) |
| F7 | Relazione `part_of`/`broader` invece di `equal` per "large bowel", "brain white matter", "brain parenchyma" (il testo nomina una parte o un insieme) | 36,48,49,50 |
| — | Dominio non medico ("heart of Maroilles cheese"), rumore dell'oro, mappatura UMLS: nessuna regola. Lo vede solo `verify` o va escluso in aggiudicazione | 10,11,12 / 13,17,24,33 / 7,40 |

## 4. Decisioni di Frank
1. Processo/funzione (B1, 16 casi): astenersi (consigliato per `radiology_report`: "heart development" non è un reperto e non costa richiamo) oppure collegare con relazione `subject_of_process` che il resto della pipeline non usa come sede.
2. Granularità (CRAFT 10 + F 4): contare `coarser_label` come errore o separare la metrica in "struttura sbagliata" e "più specifico del testo". Consiglio di separarle, tenendo la regola severa come riga a parte.
3. Procedere con F1-F5 e F7 in un unico bundle, misurando prima/dopo su CRAFT, MedMentions e sul bench, con il vincolo che nessun link giusto già accettato venga perso senza essere elencato.

## 5. Elenco completo

### CRAFT
| Mention | Oro | Nostro | Contesto |
|---|---|---|---|
| bladder | bladder organ | urinary_bladder | …ed lines) adjacent to seminal vesicles (SV) coinciding with displacement of the bladder (Bl), features typically associated with massive prostate tumors (right).… |
| Bladder | bladder organ | urinary_bladder | …erence between the prostates of Ptenpc1 or Pten+/+ mice (at 12 mo; arrowheads). Bladder (Bl) and seminal vesicles (SV) are indicated.… |
| right middle lobe | anatomical lobe | lung_middle_lobe_right | …ng an accessory lobe on the right side and had underdevelopment of the anterior right middle lobe (Figure 1A).… |
| right middle lobe | anatomical lobe | lung_middle_lobe_right | …diffuse hypoplasia and specific loss of the accessory lobe and a portion of the right middle lobe.… |
| right middle lobe | anatomical lobe | lung_middle_lobe_right | …ment of Fog2 expression in the mesenchyme surrounding the accessory bud and the right middle lobe bud, which are the lobes that do not develop normally in Fog2 mutant mice (Figu… |
| right middle lobe | anatomical lobe | lung_middle_lobe_right | …ent that results in specific loss of the accessory lobe and partial loss of the right middle lobe.… |
| right middle lobe | anatomical lobe | lung_middle_lobe_right | …blished (E12.5), it is more focally expressed in the mesenchyme surrounding the right middle lobe and accessory buds as these lobes form.… |
| right middle lobe | anatomical lobe | lung_middle_lobe_right | …This matches the phenotype of right middle lobe and accessory lobe loss, and suggests that Fog2 has a specific patterning role … |
| right middle lobe | anatomical lobe | lung_middle_lobe_right | …ht) lacks the development of the accessory lobe and the anterior portion of the right middle lobe (marked with arrows on the control sample on the left).… |
| bladder | bladder organ | urinary_bladder | …The bladder is also dilated.… |
| aortic arch | pharyngeal arch artery | aorta | …tion of the smooth muscle cell layer of endothelial structures derived from the aortic arch arteries [1-3].… |
| aortic arch | pharyngeal arch artery 4 | aorta | …Mice deficient in TGF-β2 display fourth aortic arch artery defects [7], while neural crest cell specific abrogation of TGF-β type I… |
| aortic arch | pharyngeal arch artery | aorta | …asks possible defects in derivatives of the 4th PAAs, i.e., interruption of the aortic arch.… |

### MedMentions
| # | Mention | Etichetta dell'oro | Nostro | Causa | Contesto |
|---|---|---|---|---|---|
| 0 | prostate | International prostate symptom score | prostate | A nome composto (molecola, score, procedura, asse, abbreviazione) | …International prostate symptom score, international index of erectile function-5 scores, maximal and a… |
| 1 | prostate | international prostate symptom score | prostate | A nome composto (molecola, score, procedura, asse, abbreviazione) | …Mean international prostate symptom score in patients with prostatitis was numerically but not significantl… |
| 2 | brain | brain interaction processes | brain | B1 processo/funzione dell'organo | …ms, whose characterization is of importance for a complete understanding of the brain interaction processes.… |
| 3 | inferior vena cava | inferior vena cava filter placement | inferior_vena_cava | A nome composto (molecola, score, procedura, asse, abbreviazione) | …The next frontier of office -based inferior vena cava filter placement There is an increasing number of procedures that traditionally… |
| 4 | inferior vena cava | inferior vena cava (IVC) filter placement | inferior_vena_cava | A nome composto (molecola, score, procedura, asse, abbreviazione) | …We chose to evaluate the feasibility, safety of inferior vena cava (IVC) filter placement in the office-based setting.… |
| 5 | IVC | inferior vena cava (IVC) filter placement | inferior_vena_cava | A nome composto (molecola, score, procedura, asse, abbreviazione) | …We chose to evaluate the feasibility, safety of inferior vena cava (IVC) filter placement in the office-based setting.… |
| 6 | brain | brain development | brain | B1 processo/funzione dell'organo | …In addition to its classic roles in brain development, retinoic acid (RA) has recently been shown to regulate excitatory … |
| 7 | aortic arch | aortic arch | aorta | H arco aortico: mappatura UMLS→UBERON dell'oro | …Social environment did not influence innervation in NZWs (aortic arch: p = .078, thoracic aorta: p = .34) or WHHLs (arch: p = .97, thoracic: p = .61)… |
| 8 | brain | gut-brain axis | brain | A nome composto (molecola, score, procedura, asse, abbreviazione) | …Targeting the ecology within: The role of the gut-brain axis and human microbiota in drug addiction Despite major advances in our under… |
| 9 | spinal cord | TWIK-related spinal cord K(+) (TRESK) channel | spinal_cord | A nome composto (molecola, score, procedura, asse, abbreviazione) | …TREK-2 and TRESK currents TWIK-related K(+) channel-2 (TREK-2) and TWIK-related spinal cord K(+) (TRESK) channel are members of two-pore domain K(+) channel family.… |
| 10 | heart | heart | heart | D dominio non medico (formaggio) | …Samples from rind and heart of Maroilles cheese were used, the LAB were selected on MRS agar at 30°C and 19… |
| 11 | heart | heart | heart | D dominio non medico (formaggio) | …ed: 105 strains from Maroilles made with raw milk (38 on the rind and 67 in the heart) and 92 strains from Maroilles made with pasteurized milk (39 on the rind and 5… |
| 12 | heart | heart | heart | D dominio non medico (formaggio) | …ed: 105 strains from Maroilles made with raw milk (38 on the rind and 67 in the heart) and 92 strains from Maroilles made with pasteurized milk (39 on the rind and 5… |
| 13 | colon | colon | colon | G rumore/convenzione dell'oro | …Opening the operatory specimen, the mucosa of the colon appeared totally ischemic, whilst the serosa was normal.… |
| 14 | prostate | prostate acid phosphatase | prostate | A nome composto (molecola, score, procedura, asse, abbreviazione) | …An immunohistochemical evaluation of prostatic markers (prostate-specific antigen [PSA], prostate-specific membrane antigen [PSMA], prostate aci… |
| 15 | brain | brain circuits | brain | B1 processo/funzione dell'organo | …om diverse pathophysiologies that affect the structure and function of specific brain circuits.… |
| 16 | liver | gut-liver axis model | liver | A nome composto (molecola, score, procedura, asse, abbreviazione) | …Gut Microbiota and Alcoholic Liver Disease The gut-liver axis model has often explained liver disease physiopathol… |
| 17 | brain | brain retained inside the cranium | brain | G rumore/convenzione dell'oro | …on adult rat hippocampal volume and shape using ex vivo structural MRI with the brain retained inside the cranium to prevent distortions due to dissection, followed … |
| 18 | DENS | drug effects on the nervous system" (DENS) scale | vertebrae_C2 | A nome composto (molecola, score, procedura, asse, abbreviazione) | …scale for primates, motor tasks, and the " drug effects on the nervous system" (DENS) scale.… |
| 19 | spleen | donor-specific spleen cells transfusion | spleen | A nome composto (molecola, score, procedura, asse, abbreviazione) | …Anti-CD45RB and donor-specific spleen cells transfusion inhibition allograft skin rejection mediated by memory T cell… |
| 20 | spleen | Donor-specific spleen cell transfusion | spleen | A nome composto (molecola, score, procedura, asse, abbreviazione) | …Donor-specific spleen cell transfusion (DST) alone also failed to induce the tolerance in the pre-sen… |
| 21 | heart | heart development | heart | B1 processo/funzione dell'organo | …gnalling through Slit and Netrin pathways plays a role in cell migration during heart development.… |
| 22 | heart | heart development | heart | B1 processo/funzione dell'organo | …we show that another Slit and Netrin receptor, Dscam1, the role of which during heart development was previously unknown, is required for both normal migration of ca… |
| 23 | brain | N-terminal pro-brain natriuretic peptide | brain | A nome composto (molecola, score, procedura, asse, abbreviazione) | …tides (N-terminal pro-atrial natriuretic peptide [NT-proANP] and N-terminal pro-brain natriuretic peptide [NT-proBNP]), cardiac and skeletal troponins (cTnI, cTnT, a… |
| 24 | colon | colon | colon | G rumore/convenzione dell'oro | …We divided the colon into 4 regions and compared PET/CT results for each region with colonoscopy and… |
| 25 | liver | liver living donors | liver | A nome composto (molecola, score, procedura, asse, abbreviazione) | …Complications and Near-Miss Events After Hepatectomy for Living-Related Liver Donation: An Italian Single Center Report of One Hundred Cases BACKGROUND In he… |
| 26 | heart | autonomic control of the heart | heart | B1 processo/funzione dell'organo | …ise and sertraline might exert positive effects on the autonomic control of the heart among older patients with major depression.… |
| 27 | Liver | Liver Fatty Acid Binding Protein | liver | A nome composto (molecola, score, procedura, asse, abbreviazione) | …Liver Fatty Acid Binding Protein Deficiency Provokes Oxidative Stress, Inflammation, … |
| 28 | liver | liver fatty acid binding protein | liver | A nome composto (molecola, score, procedura, asse, abbreviazione) | … results of this study demonstrated that PZA decreased the expression levels of liver fatty acid binding protein (L-FABP) and its target gene, peroxisome proliferato… |
| 29 | brain | developing brain | brain | B1 processo/funzione dell'organo | … inhibitor of TNF-a, prevents propofol -induced neurotoxicity in the developing brain Propofol can induce acute neuronal apoptosis, neuronal loss or long-term cognit… |
| 30 | brain | developing brain | brain | B1 processo/funzione dell'organo | … have demonstrated that propofol can increase the TNF-α level in the developing brain, but there is a lack of direct evidence to show whether TNF-α is partially or f… |
| 31 | gallbladder | contractibility of gallbladder | gallbladder | B1 processo/funzione dell'organo | …Emodin can enhance the contractibility of gallbladder and alleviate cholestasis by regulating plasma CCK levels, [Ca(2+)]i in cholecy… |
| 32 | brain | brain circuits | brain | B1 processo/funzione dell'organo | …However, the relative influence of cognitive and emotional brain circuits to the feeding circuitry in the hypothalamus and hindbrain remains unc… |
| 33 | colon | colon | colon | G rumore/convenzione dell'oro | …Mucosa - associated biohydrogenating microbes protect the simulated colon microbiome from stress associated with high concentrations of poly-unsaturated … |
| 34 | brain | brain connectivity patterns | brain | B1 processo/funzione dell'organo | …However, little is known about the effects of this sedation on the brain connectivity patterns in the damaged brain essential for differential diagnosis… |
| 35 | brain | brain arousal | brain | B1 processo/funzione dell'organo | …Nonetheless, given the known importance of the thalamus in brain arousal, its disruption could well reflect the diminished movement obtained in … |
| 36 | large bowel | large bowel | colon | F granularità (testo nomina una parte o un insieme) | …11 IBS with mixed bowel habit (IBS-M) underwent whole-gut transit and small and large bowel volumes assessment with MRI scans from t=0 to t=360 min.… |
| 37 | brain | brain controls | brain | B1 processo/funzione dell'organo | …Hypocretins and Arousal How the brain controls vigilance state transitions remains to be fully understood.… |
| 38 | brain | brain state dynamics | brain | B1 processo/funzione dell'organo | …sms by which such a relatively small population of neurons controls fundamental brain state dynamics.… |
| 39 | pancreas | development of pancreas | pancreas | B1 processo/funzione dell'organo | …The knowledge of development of pancreas helps in planning new therapeutic interventions in the treatment of various con… |
| 40 | aortic arch | aortic arch | aorta | H arco aortico: mappatura UMLS→UBERON dell'oro | …First, we compared atherosclerosis in the aortic arch of age-matched (24 weeks) C57BL/6J control (n = 10), LDL-receptor deficient (n … |
| 41 | brainstem | auditory brainstem response | brain | B2 misura/campo (non anatomia) | … chronic experiments on immature rabbits by recording of short-latency auditory brainstem response (ABR) and distortion product ot… |
| 42 | heart | levels of heart interlukine-6 | heart | B1 processo/funzione dell'organo | …Higher levels of heart interlukine-6 (IL-6) and tumor necrosis factor-α (TNF-α) were observed in LPS g… |
| 43 | brain | brain research | brain | B2 misura/campo (non anatomia) | …The potential of this type of multimodal setup for brain research is demonstrated by our preliminary studies on human, showing effects o… |
| 44 | Costal Cartilage | Irradiated Homologous Costal Cartilage | costal_cartilages | C senso "tessuto per innesto" | …Autologous vs Irradiated Homologous Costal Cartilage as Graft Material in Rhinoplasty Studies comparing surgical results of rhinopla… |
| 45 | costal cartilage | autologous costal cartilage | costal_cartilages | C senso "tessuto per innesto" | …Autologous vs Irradiated Homologous Costal Cartilage as Graft Material in Rhinoplasty Studies comparing surgical results of rhinopla… |
| 46 | costal cartilage | irradiated homologous costal cartilage | costal_cartilages | C senso "tessuto per innesto" | …Autologous vs Irradiated Homologous Costal Cartilage as Graft Material in Rhinoplasty Studies comparing surgical results of rhinopla… |
| 47 | costal cartilage | Autologous costal cartilage | costal_cartilages | C senso "tessuto per innesto" | …Autologous costal cartilage also had better histologic properties than IHCC did, suggesting it as an ideal … |
| 48 | brain | brain white matter property | brain | F granularità (testo nomina una parte o un insieme) | …ssion scores were also associated with decreased fractional anisotropy (FA) - a brain white matter property - within the forceps minor and the left superior temporal… |
| 49 | brain | brain parenchyma | brain | F granularità (testo nomina una parte o un insieme) | …Canine brain phantoms were fabricated from osteological skull specimens, agarose brain paren… |
| 50 | brain | brain parenchyma | brain | F granularità (testo nomina una parte o un insieme) | …ull, agarose, and cheese components approximated the in vivo features of skull, brain parenchyma, and contrast-enhancing tumors of meningeal and glial origin, respec… |

## 6. Stato dopo le correzioni (2026-10-08, stesso giorno)

Implementati F2, F3, F4, F5 (come profilo di dominio del documento: **evidenza, non veto**) e, in forma generale, la testa del sintagma nominale (parte di F1) con i sinonimi lunghi di UBERON. F1 come esperimento (parser, attenzione di un encoder) è nel workflow `head-probe`, da lanciare a mano. F6 resta aperto; F7: nessuna modifica al codice.

| | Prima | Dopo |
|---|---|---|
| CRAFT (1.349 giudicati) | 13 errori (99,04%) | 11 errori (99,18%) |
| MedMentions (755 giudicati) | 51 errori (93,36%) | 38 errori (94,97%) |
| Link fermati dalle nuove regole | – | 15 (tutti errori) |
| Link giusti persi | – | 1 (`pancreas` in "islet-to-pancreas volume ratios") |
| Bench interno (held-out IT/EN, linking) | – | identico |
| Test | – | 628 passati |

Cosa fa ciascuna regola, uguale per ogni struttura:
- **Testa del sintagma (F1 parziale)**: la testa è l'ultima parola del sintagma a destra della menzione (composti inglesi con testa a destra), tolte le nominalizzazioni (expression, level, concentration…). Il veto scatta solo se la testa è un nome "non sede" (molecola, scala, via, dispositivo, risposta…); "brain natriuretic peptide", "prostate acid phosphatase", "prostate symptom score" si fermano, "liver gene expression" no.
- **F2 trattino**: parola unita a un'altra da trattino ("gut-brain axis"); contano solo se esiste una testa non-sede. Prefissi di posizione/lato ("intra-hepatic", "right-sided") non contano.
- **F3 sigle definite**: algoritmo di Schwartz e Hearst (2003) per "forma lunga (SIGLA)" e "SIGLA (forma lunga)"; se la menzione è una parte della forma lunga e la testa è non-sede, astensione. "congenital heart disease (CHD)" e "spinal cord injury (SCI) model" restano collegati.
- **F4 materiale d'innesto**: qualificatore (autologous, homologous, donor-specific…) subito prima della menzione e sintagma che finisce con graft/transplant/cell/transfusion…
- **Sinonimi UBERON**: "aortic arch artery" è un nome lungo di un'altra struttura (arco faringeo): la menzione è dentro un nome più lungo.
- **F5 dominio del documento**: parole di stadio (embryo, fetal, E12.5…), in frase o ≥3 nel documento; se il nome è condiviso tra una struttura in sviluppo e una adulta, il link resta ma con conflitto `name_shared_with_a_developing_structure_in_a_developmental_text`. Un veto a livello di documento è stato scartato: nei dati CRAFT 25 casi su 28 in testi embrionali sono giustamente adulti.

Errori rimasti (49: 11 CRAFT, 38 MedMentions): in gran parte processo/funzione (F6, dipende dalla convenzione: CRAFT etichetta l'organo in "heart development", MedMentions il processo), granularità, rumore dell'oro, dominio non medico (formaggio), e pochi composti ("inferior vena cava filter placement").

F7 (relazione `part_of`/`broader`): nessun cambio. `parenchyma` è già collegato all'organo per convenzione di progetto; `bladder` e `right middle lobe` sono più specifici dell'oro, non sbagliati; "large bowel" → `colon` è una scelta di lessico da confermare o segnare come approssimata.
