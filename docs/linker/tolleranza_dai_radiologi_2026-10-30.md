# La tolleranza presa dai radiologi (30 ottobre 2026)

## La domanda

Quando la misura di Melampo è "diversa dal referto" al punto da chiamare una persona? Fino a oggi la regola era max(2 mm, 10%), scelta a mano al sesto probe. Ora la misuriamo sui radiologi stessi: due radiologi che misurano lo stesso nodulo differiscono già. Una differenza che rientra in quella fra due radiologi non è un errore di Melampo.

## Il metodo

Si usano i limiti di accordo al 95% di Bland-Altman fra due lettori: 1,96 × √2 × la deviazione standard dei quattro diametri dentro lo stesso nodulo. È il metodo della metrologia medica, lo stesso dei profili QIBA.

Fonte: i 40 noduli LIDC-IDRI del set "misura". Ognuno ha quattro radiologi, misurati tutti con il nostro codice (`lesion_measures`). La fascia di dimensione è la media dei lettori. Script: `scripts/reader_agreement.py`. Dati: `data/lidc/reader_agreement_misura_2026-10-30.json`. L'intervallo fra parentesi è un bootstrap sui noduli (seme fisso, 2.000 ripetizioni).

| Fascia | Noduli | Limiti di accordo 95% | Differenza fra due lettori: mediana | 90° percentile | 95° percentile |
|---|---|---|---|---|---|
| sotto 10 mm | 19 | **3,0 mm** [2,4–3,5] | 0,9 | 2,8 | 3,1 |
| 10–20 mm | 10 | **6,8 mm** [3,6–10,0] | 1,4 | 5,2 | 9,8 |
| 20 mm e oltre | 11 | **12,2 mm** [7,8–16,0] | 2,8 | 11,7 | 14,4 |

La tolleranza nuova (`size_tolerance.READERS`) è questo limite più 0,5 mm per l'arrotondamento del referto, che scrive i millimetri interi o i centimetri con un decimale.

I numeri del 29 ottobre (0,6 / 1,3 / 2,7 mm di mediana, 2,0 / 3,1 / 8,0 mm al 90° percentile) misuravano un'altra cosa: un lettore contro la media degli altri tre. Quella media toglie metà della variabilità. Il referto, invece, è un lettore solo, e il confronto giusto è fra due lettori.

## Quanti casi vanno a una persona

Ogni radiologo scrive a turno il "referto", con il suo diametro arrotondato al millimetro.

| Chi misura | Fascia | Tolleranza attuale max(2 mm, 10%) | Limiti di accordo dei radiologi |
|---|---|---|---|
| un altro radiologo (la media degli altri tre) | tutte | 44/160 (28%) | 3/160 (2%) |
| | sotto 10 mm | 12/76 | 0/76 |
| | 10–20 mm | 12/40 | 0/40 |
| | 20 mm e oltre | 20/44 | 3/44 |
| Melampo, regola congelata del 29 ottobre | tutte | 48/132 (36%) | 6/132 (5%) |
| | sotto 10 mm | 8/60 | 0/60 |
| | 10–20 mm | 19/36 | 4/36 |
| | 20 mm e oltre | 21/36 | 2/36 |
| VoxTell da solo | tutte | 43/132 (33%) | 7/132 (5%) |

A questi si aggiungono i 7 noduli su 40 che VoxTell non ha trovato, che vanno sempre a una persona.

## Come si leggono

1. **La tolleranza fissa era sbagliata per la persona, non per Melampo.** Mandava a revisione un quarto delle misure dei radiologi stessi.
2. **I limiti sono larghi per i noduli grandi: 12 mm oltre i 20 mm.** Non è rumore di misura. In un nodulo irregolare due radiologi prendono assi diversi: uno include uno spicolo o un vaso in una fetta, l'altro no. In LIDC-IDRI-0130 due lettori misurano 13 mm e due 23–24 mm, con un Dice fra loro di 0,85: la forma disegnata è quasi la stessa, il diametro maggiore no. Anche diametro medio e volume non fanno meglio (95° percentile relativo 54% e 109% oltre i 20 mm).
3. **Quindi oltre i 20 mm il controllo sulla misura scopre solo gli errori grossolani.** Che la lesione sia quella giusta lo dicono la sede (il lobo nominato dal referto, l'atlante) e l'accordo di due letture indipendenti, non la misura.
4. **Il campione è piccolo.** 10–11 noduli per fascia sopra i 10 mm, con intervalli larghi (7,8–16,0 mm oltre i 20 mm). Il margine non si può separare: oltre i 20 mm tutti i noduli hanno un margine irregolare secondo i lettori.

## Cosa cambia nel codice

- `melampo.imaging.size_tolerance`: le politiche `fixed` (quella di prima) e `readers`, con `tolerance_mm` e `consistent`.
- `findings.image_evidence(size_policy=...)`: **il default resta `fixed`.** La politica cambia solo se si chiama `a una persona`; la misura resta sempre la regione del segmentatore.
- `lidc_probe` registra il controllo con tutte e due le politiche (`size_check_by_policy`).
- Il notebook LIDC lancia `reader_agreement.py` alla fine.

## La verifica, fissata ora

Il giro `push lidc verifica` (38 noduli mai visti) dirà due cose.

1. Se i limiti dei 38 noduli cadono dentro gli intervalli bootstrap della tabella.
2. Quanti casi vanno a una persona con `readers`. Il criterio: meno del 10% delle misure dei radiologi, e nessuna lettura di Melampo accettata con un Dice sul consenso sotto 0,5 (la misura entro tolleranza ma su un'altra forma).

Se le due condizioni reggono, `readers` diventa il default, con la tabella stimata sui 78 noduli insieme. Se non reggono, resta `fixed` e si riparla.
