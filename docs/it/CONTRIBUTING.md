# Contribuire: come cambiare Melampo senza rompere niente

*[English](../en/CONTRIBUTING.md) · Italiano · aggiornato al 2 novembre 2026*

Ogni modifica passa da una pull request con i suoi test; una modifica al modo in cui il sistema legge o decide
richiede anche una probe su un campione nuovo, con i criteri scritti prima del run.

## 1. Che cosa nessuna modifica può fare

- dichiarare una validità clinica senza una misura in [VALIDATION](VALIDATION.md);
- mandare testi di pazienti a un servizio fuori dall'ospedale, o aggiungere una chiamata di rete non configurata
  esplicitamente;
- lasciare che un lettore da solo, o i modelli da soli, tolgano o promuovano un reperto, un'ipotesi o
  un'etichetta;
- aggirare l'astensione, le domande per una persona, la provenienza o la traccia di audit;
- aggiungere una regola deterministica che decide un caso: il lettore controlla con ontologie e grammatica, i
  modelli leggono, una persona decide sulle contraddizioni;
- tarare un componente sul campione che lo misura;
- committare dati di pazienti, testi di corpora non commerciali, token o password.

## 2. Branch, pull request, commit

| Passo | Regola |
|---|---|
| Branch | da `main`, con nome `feat/…`, `fix/…`, `docs/…`; un argomento per branch |
| Messaggio di commit | prima riga: che cosa è cambiato; corpo: perché, e il criterio o la misura che lo sostiene. È la traccia di audit: dire chi ha deciso quando la decisione era del responsabile |
| Pull request | descrizione con che cosa cambia, i test e, per una modifica della lettura, la probe e i suoi criteri; la unisce il responsabile dopo che la CI è passata |
| Sessioni senza diritti di push | consegnano un git bundle; il responsabile lo applica (`git fetch <bundle> HEAD`, `git cherry-pick`) e fa il push |
| Tag | `docs-archive-2026-11` conserva la documentazione precedente al 2 novembre 2026 |

## 3. Test e CI

```bash
uv run pytest -q                              # tutta la suite
uv run pytest -q tests/test_claims_reading.py # un'area
uv run python scripts/docs_check.py check     # registri, link, le due lingue
uv run python scripts/docs_check.py soup      # dopo aver cambiato le dipendenze
```

`ci.yml` esegue la suite e controlla i pin DVC dei dati di riferimento a ogni push e pull request;
`docs-check.yml` controlla la documentazione. Ogni altro workflow si avvia a mano o a orario, perché chiama
modelli a pagamento o scarica dati grandi.

## 4. Aggiungere o cambiare una lettura

1. Scrivere la modifica e i suoi test.
2. Estrarre un campione nuovo (`scripts/owner_probe.py sample --seed <seed nuovo>`): esclude tutti i referti dei
   campioni precedenti.
3. Etichettarlo **prima** che qualsiasi lettore o modello giri su di esso; segnare con "?" le etichette in dubbio.
4. Scrivere i criteri in VALIDATION §5 e nella pull request.
5. Tenere la modifica spenta nell'acquisizione (con una variabile d'ambiente) finché la probe non passa.
6. Lanciare il workflow; un run che perde più del 5% delle risposte non conta.
7. Scrivere il risultato in VALIDATION §4, passato o no. Se non passa, la modifica resta spenta.

## 5. Workflow

| Workflow | Che cosa fa | Avvio |
|---|---|---|
| `ci` | test, pin DVC | push, pull request |
| `docs-check` | registri, link, inglese e italiano allineati | push, pull request |
| `owner-probe` | di chi è il reperto: lettore, modelli, affermazioni (input `sets`, `arms`, `models`) | manuale |
| `role-probe` | il ruolo di una parola: lettore e modelli | manuale |
| `hpo-probe` | dal reperto al termine HPO | manuale |
| `external-check` | il linker su CRAFT e MedMentions | manuale |
| `public-reports` | corpora pubblici di referti per il campione del gold set | manuale |
| `ci-probe`, `lattice-probe`, `phrase-probe`, `head-probe`, `verify-probe` | letture del sintagma e i loro controlli | manuale |
| `linking-bench`, `encoder-bench`, `focused-comparison-bench`, `root-model-bench`, `vetting-bench` | benchmark di modelli ed encoder (chiamate a pagamento) | manuale |
| `collocations`, `longer-names`, `report-memory`, `rad-sprl`, `symptom-coverage` | dati per il lettore e il linker | manuale |
| `classifications-it`, `radiology-kinds`, `data-and-dependency-updates` | aggiornamenti settimanali (CIPI, ICD-9-CM, RadLex, HPO, dipendenze) | a orario |
| `pin-model-revisions` | fissa i pesi dei modelli a un commit e a un hash esatti | manuale |

Segreti usati: `OPENROUTER_API_KEY` (solo corpora pubblici), `HF_TOKEN`, `UMLS_API_KEY`, `DB_PASSWORD`. Stanno nei
segreti del repository e nell'ambiente, mai nel codice o nei documenti.

## 6. Problemi e modifiche

Un difetto si apre come issue con la sua riproduzione e, se tocca un pericolo, l'id del rischio di
[`docs/registers/risks.yml`](../registers/risks.yml). La correzione nomina l'issue nel suo commit; un pericolo nuovo
entra nel registro con il suo controllo e il test che lo verifica.
