# Immagini, tracciati, LOINC e referti nel tempo (16 ottobre 2026)

## Verifiche

### D21: Nemotron Ultra o Gemma 4 come encoder?
No, così come sono. Sono modelli che *generano* testo; i loro stati interni non sono addestrati per essere
confrontati. LLM2Vec (2024) lo misura su MTEB: un modello generativo usato senza adattamento fa 42–44, adattato con
addestramento contrastivo 55–57. Le stesse famiglie hanno però le loro versioni per i vettori:
- **EmbeddingGemma 2** (Google, 6 ottobre 2026): dall'architettura Gemma 4, **Apache 2.0**, oltre 100 lingue,
  multimodale, 270M per il testo, vettori fino a 768. Aggiunto al test degli encoder (`embeddinggemma-2`).
- **Nemotron-3-Embed-1B** (NVIDIA, luglio 2026): licenza OpenMDW, "ready for commercial use", italiano tra le 34
  lingue valutate, 2048 dimensioni. Già nel test (4 ottobre: alla pari con Qwen3-8B, lateralità 1,00).
- Chi ha risolto il problema non affida tutto al vettore: filtra con i campi strutturati (struttura, lato, misura,
  negazione) e ordina con un encoder dedicato. Il nostro test lo aveva già mostrato (misure e negazione falliscono
  con tutti gli encoder).

Da fare: rilanciare `encoder-bench` con EmbeddingGemma 2 e Nemotron-3-Embed-1B accanto a Qwen3-8B e Gemini (come
riferimento), poi la scelta del titolare.

### Dove far girare i test (dati pubblici soltanto)
| | costo | GPU | note |
|---|---|---|---|
| **Kaggle** | gratis | T4 / P100, circa 30 h/settimana, sessioni di 12 h | primo posto per i test |
| **Modal** | 0 fissi, 30 $/mese di credito, poi al secondo | T4 ~0,59 $/h, L4 ~0,80 $/h | a consumo, nessun abbonamento |
| RunPod | credito prepagato, al secondo | A40 ~0,44 $/h, 4090 ~0,74 $/h | alternativa |
| Google Cloud | prova senza GPU | quota GPU solo con account a pagamento | dopo |
| OpenRouter | — | — | solo modelli linguistici: non esegue VoxTell né TotalSegmentator |

Un dato pseudonimizzato resta personale (GDPR): su questi servizi solo dataset pubblici.

### Licenze per modalità (da accettare prima dell'uso)
| Modalità | Modello | Pesi | Da accettare | Test | Prodotto |
|---|---|---|---|---|---|
| TC/RM/PET | VoxTell | CC BY-NC-SA 4.0 | nulla, download automatico | sì | non commerciale: da chiudere col legale |
| TC/RM | TotalSegmentator | Apache-2.0 (compiti aperti) | licenza gratuita non commerciale per alcuni compiti | sì | sì (compiti aperti) |
| TC | VISTA3D / NV-Segment-CT | NVIDIA Open Model License | termini NVIDIA | sì | da verificare ("not for clinical usage") |
| TC/RM/RX | MedGemma 1.5 | termini HAI-DEF | i termini HAI-DEF | sì | "research and commercial use" con i termini |
| Eco/RX | BiomedParse | CC BY-NC-SA 4.0, gated | condividere il contatto, accettare le condizioni | sì | no (escluso come dispositivo medico) |
| Eco | SonoBase (sett. 2026) | dichiarati rilasciati, licenza non indicata | — | da verificare | da verificare |
| Eco | UltraSAM3 | licenza non indicata | — | da verificare | da verificare |
| Eco | MedSAM2 | solo ricerca ed educazione | — | sì | no |
| RX | RAD-DINO | MIT (solo encoder) | — | sì | dichiarato solo ricerca |
| MOC | nessun modello: BMD, T-score, Z-score sono numeri del referto | — | — | — | — |

Dataset verificati: TotalSegmentator TC (1228 TC, CC BY 4.0), TotalSegmentator RM (616 RM, CC BY-NC-SA 2.0),
Medical Segmentation Decathlon (tumori di fegato, polmone, pancreas, colon, CC BY-SA 4.0). Per l'ecografia ogni
dataset ha la sua licenza; per BUSI è pubblicata una lettera sui problemi di qualità (usare la versione curata).

## Costruito

### Immagini (`src/melampo/imaging/`)
- `lesion_measures.py`: dalla maschera e dalla geometria DICOM, centro in mm, volume, diametro massimo e assiale
  (RECIST), assi principali, superficie, sfericità, allungamento, valori HU; il bordo come punti in mm (una mesh
  con scikit-image). Su una sfera sintetica di 10 mm su griglia anisotropa: volume entro 6%, superficie entro 12%,
  sfericità 0,97.
- `segmenters.py`: VoxTell (CLI, testo libero → regione), TotalSegmentator (organi, per il controllo), maschere
  date; conversione LPS ↔ RAS/NIfTI; un segmentatore assente lascia il reperto sul testo con il motivo.
- `findings.py`: il nodo **Reperto** = testo (struttura del linker, ruoli, qualificatori, misure scritte, frase
  sorgente) + immagine (maschera, misure, forma, esito: concorde / discordante / non trovato / ambiguo / non
  verificato / senza immagine) + vettori (forma; testo quando D21 sarà decisa). Se discordante, la parte visiva è
  messa da parte. FalkorDB: Paziente → Studio → Reperto → Struttura.
- `interpreter.py`: dal referto e dall'esame ai reperti; una frase diventa reperto solo se *descrive* qualcosa
  (misura, carattere, grado, dubbio); una sede nominata e basta resta un arco SITE; una negata non è un reperto.
- `scripts/imaging_probe.py`: la misura vera su dati pubblici (Dice, esiti, misure, tempi), per Kaggle.

### Referti nel tempo (`imaging/longitudinal.py`)
Due esami della stessa persona: stessa struttura (obbligatoria), stessa sede (distanza in mm se c'è lo stesso
frame of reference, altrimenti posizione relativa dentro l'organo), crescita plausibile, stesso carattere.
Abbinamento uno a uno, i pari merito vanno a una persona. Arco **SEGUE** con giorni, variazione del diametro,
rapporto dei volumi, tempo di raddoppio (Schwartz) e confrontabilità delle acquisizioni (QIBA).

### Tracciati (`src/melampo/data/traces.py`)
- PDF digitale: le curve vettoriali (poppler `pdftocairo -svg`, già richiesto) → numeri.
- Scansione: l'immagine del tracciato si vettorializza (griglie e assi tolti, il tratto più lungo tenuto, per ogni
  colonna il punto più alto) → stesso metodo.
- Elettroforesi: zone (albumina, α1, α2, β, γ), picchi con larghezza, **picco stretto in zona gamma/beta descritto
  e mandato a una persona**, confronto con le percentuali della tabella. Su curve sintetiche: zone entro 1,5 punti
  (vettore) e 2 punti (immagine); picco stretto trovato in entrambi.
- Altri tracciati (ECG, spirometria, CTG): estratti come serie, non interpretati; ciascuno avrà la sua analisi.

### LOINC (`src/melampo/data/loinc.py`, `scripts/build_loinc_index.py`)
Componente (nome inglese e variante italiana), proprietà dall'unità, materiale dalla sezione: un codice solo se un
solo termine attivo combacia; più candidati → a una persona; nessuno → niente. Serve scaricare LOINC da loinc.org
(account gratuito, accettare la licenza) e costruire l'indice; l'indice non è nel repository finché non sono
verificate le condizioni di ridistribuzione.

Il riepilogo di ogni allegato porta solo conteggi e tipi (regola di privacy già nei test): le letture restano
sull'oggetto.

## Come lanciare la misura delle immagini su Kaggle
1. Nuovo notebook, acceleratore GPU, Internet attivo (Kaggle chiede la verifica del telefono).
2. Dati: un dataset pubblico con maschere di lesione, per esempio Medical Segmentation Decathlon Task03 Liver o
   Task06 Lung (CC BY-SA 4.0).
3. `pip install voxtell TotalSegmentator nibabel`, poi `git clone` del repository.
4. Un `cases.json` con immagine, maschera di riferimento, struttura e frase; poi
   `python scripts/imaging_probe.py --cases cases.json --out probe.json --device cuda`.
5. Si scaricano `probe.json` e il log e li si porta qui.
