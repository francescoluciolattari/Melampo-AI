# Gold set di sviluppo ("silver"), 12 ottobre 2026

Etichette di **un solo etichettatore (Claude)**, non di radiologi. Servono a sviluppare e misurare in modo
ripetibile; **non certificano** nulla (vedi `docs/linker/valutazione_proposte_e_gold_set_silver_2026-10-12.md`).

Contenuto: solo le etichette (`labels/*.txt`); i testi non sono nel repository (licenze non commerciali).
Si ricostruiscono dalle fonti:

```
git clone --depth 1 https://github.com/hltfbk/E3C-Corpus.git E3C            # CC BY-NC
git clone --depth 1 https://github.com/PARROT-reports/PARROT_v1.0.git PARROT  # CC BY-NC-SA 4.0
python scripts/fetch_public_reports.py iuxray --out iuxray.jsonl              # IU X-ray (Open-i, NLM)
python scripts/build_silver_gold.py --e3c E3C --parrot PARROT/data/PARROT_v1_0.jsonl --iuxray iuxray.jsonl --out silver
python scripts/assemble_silver_gold.py --items silver --labels data/gold/silver_2026-10-12/labels --out silver/gold.jsonl
python scripts/gold_set.py evaluate silver/gold.jsonl --reports silver/reports.jsonl --uberon data/linking/uberon-basic.obo
```

Strati: `e3c_it` (328 span BODYPART annotati a mano da E3C, tutti), `e3c_en` (150 su 968, seme fisso),
`parrot_it` (100 frasi, una per referto, per modalità; 134 menzioni), `iuxray_en` (100 frasi, una per referto;
102 menzioni). Le etichette sono state scritte prima di far girare il linker su questi testi.

Codici: struttura `N` NONE_IN_CLASSES, `X` NOT_ANATOMY, `A` AMBIGUOUS, altrimenti l'id della classe;
relazione `e` equal, `po` part_of, `co` contour_of, `ap` approx; ruolo `s` structure, `p` procedure_site,
`d` device_site, `l` landmark, `o` source_of, `il` inherent_location, `in` inside_a_name, `nb` not_a_body_site, `ot` other.

Convenzioni scelte (da far rivedere ai radiologi): regioni del corpo (torace, addome, arti) `N`; fluidi,
cellule e lesioni `X/ot`; polmone intero o "i polmoni" `N` (non c'è una classe del polmone intero);
strutture pari senza lato o coordinate (due strutture in una menzione) `A`; parenchima di un organo `po`;
"cardiomediastinal silhouette" / "ombra cardiaca" `heart co`; "heart size" `X/il`; "TC cranio" `N` (la testa).

**31 ottobre 2026: il ruolo `landmark`.** Le righe sono state rilette per il nuovo ruolo (la struttura serve
solo a collocare un'altra cosa). Ne sono cambiate 12, segnate nella nota con "[landmark, relabelled 31 Oct
2026]": 8 sono diventate `l` e 4 `s+l`. Più ruoli insieme si scrivono uniti da `+`, il principale per primo.
Le righe di IU X-ray (`iu.txt`) non sono state rilette, perché i testi non sono raggiungibili da qui.
