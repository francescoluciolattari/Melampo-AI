# L'interprete dei referti di anatomia patologica (16 ottobre 2026)

Quando l'orchestratore (`report_router`) riconosce un istologico o un citologico, il testo passa a
`data/pathology_report.py`. Legge in modo deterministico quello che il patologo scrive in forma strutturata,
con le grammatiche dei sistemi pubblicati. Non indovina e non completa: un pT2 non diventa uno stadio.

## Che cosa legge

| | esempi |
|---|---|
| sezioni | materiale, notizie cliniche, macroscopia, microscopia, immunoistochimica, molecolare, diagnosi, commento (IT/EN) |
| TNM | pT2 pN1a(sn), ypT1b, pT3 pN2a, T2 N0 M0, edizione AJCC/UICC |
| grado e sistemi | G1-G4, Gleason 4+3=7, ISUP / Grade Group, Nottingham 2 (3+2+2), FIGO, Ishak (grading-staging), METAVIR A2F2, Breslow, Clark, R0/R1/R2 |
| margini | indenni/liberi/negativi o coinvolti/positivi, con la distanza minima se scritta |
| linfonodi | 1/2, 5/18, "0 of 12 lymph nodes" |
| invasione | linfovascolare, perineurale: presente / assente |
| dimensioni | "dimensioni massime 2,1 cm" |
| marcatori | ER/RE, PgR/RP, HER2 (score IHC e FISH/ISH separati), Ki-67, PD-L1 (TPS/CPS), MLH1/PMS2/MSH2/MSH6 (conservata/persa), CK7/CK20, CDX2, TTF-1, p40, GATA3 e altri |
| molecolare | EGFR, KRAS, BRAF, ALK, ... : wild type / mutato (con la variante), variante testata e non trovata ("BRAF V600E: wild type"), riarrangiato, amplificato; MSI/MSS |
| diagnosis | il testo della diagnosi con le sue posizioni, per il lettore e il linker (struttura, ruolo, qualificatori) |

Ogni elemento conserva le parole stampate e le posizioni nel testo. Le righe di una sezione strutturata
(immunoistochimica, molecolare) che nessuna grammatica legge sono restituite come residuo: nulla sparisce.

Regole che evitano i falsi positivi, uguali per tutti i referti:
- "T2" senza prefisso (p, c, y) è una categoria TNM solo con la sua N ("T2 N0"): da solo è una pesatura RM
  ("T2 pesate") o una vertebra ("T11");
- le sigle corte dei marcatori (ER, RE, PR, AR) contano solo con un valore o uno stato accanto;
- R0/R1/R2 solo nella diagnosi o nel commento.

## Misura

Cinque referti scritti da me (mammella, prostata in inglese, colon con MMR, biopsia epatica, polmone con
PD-L1 ed EGFR): tutti gli elementi letti, nessun residuo, nessun falso positivo sui testi di controllo (RM
"T1 e T2 pesate", vertebra T11, "il re"). **Non è una misura su referti veri**: le formulazioni reali variano
per laboratorio, e la prova che conta sono i primi istologici veri, con il residuo a mostrare le lacune.

## Nel flusso

L'allegato riconosciuto come anatomia patologica porta `pathology` nel riepilogo del caso (sezioni, elementi,
diagnosi, residuo). L'immunoistochimica impaginata come tabella è letta anche dalla grammatica dei valori, che
però non la confronta mai con un intervallo. La diagnosi passa al lettore e al linker come il resto del testo.

## Prossimi passi
- Il tracciato dell'elettroforesi: nei PDF digitali la curva vettoriale diventa numeri e picchi; nelle scansioni
  si individua l'immagine del tracciato, la si vettorializza e si applica lo stesso metodo (Frank, 16 ottobre).
- LOINC per gli esami (variante italiana).
