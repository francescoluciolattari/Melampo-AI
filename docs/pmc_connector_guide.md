# PMC Case Report Connector — Guida operativa

`src/melampo/connectors/pmc_case_reports.py` recupera case report da PMC e li
converte in `EvaluationCase` per il protocollo B4 (`docs/final_treatise_decision_record.md`,
sezione sulla valutazione del Dream Engine).

## Dove va eseguito

**Non nell'ambiente di esecuzione di Claude.** L'accesso di rete di Claude è
allowlisted (GitHub, PyPI, npm e pochi altri) e NCBI non è nell'elenco.
Eseguire questo script da una macchina con accesso Internet normale — laptop,
server, notebook Jupyter.

## Prerequisiti

- Python 3.11+, nessuna libreria oltre la libreria standard.
- Una email di contatto (NCBI la richiede; non è autenticazione). Il progetto
  ha un default già configurato: `francesco.lucio.lattari@gmail.com`.
- Facoltativo ma consigliato: una API key NCBI, gratuita, da
  https://www.ncbi.nlm.nih.gov/account/settings/ — alza il limite di frequenza
  da 3 a 10 richieste al secondo.

## Configurazione della API key — non va mai scritta nel codice

La chiave vive **solo** in due posti: il secret GitHub `NCBI_API_KEY` del
repository, e — quando esegui lo script in locale — una variabile d'ambiente
con lo stesso nome sulla tua macchina. Non va mai incollata in un file
tracciato dal repository, nemmeno temporaneamente: un secret committato per
errore resta nella cronologia Git anche dopo essere stato rimosso dall'ultima
versione, e va considerato compromesso.

**In locale, prima di eseguire lo script:**

```bash
export NCBI_API_KEY="la-tua-chiave"     # bash/zsh, sessione corrente
```

oppure su Windows PowerShell:

```powershell
$env:NCBI_API_KEY = "la-tua-chiave"
```

**In un workflow GitHub Actions**, il secret è già configurato nel repository;
va solo esposto come variabile d'ambiente al passo che esegue lo script:

```yaml
- name: Run PMC fetch
  env:
    NCBI_API_KEY: ${{ secrets.NCBI_API_KEY }}
  run: python scripts/fetch_pmc_cases.py
```

## Uso minimo

`FetchConfig.from_environment()` legge la chiave dalla variabile d'ambiente e
usa l'email di default del progetto — non serve passare nulla a mano:

```python
from melampo.connectors.pmc_case_reports import FetchConfig, PmcCaseReportFetcher, LICENSE_COMMERCIAL

config = FetchConfig.from_environment(license_group=LICENSE_COMMERCIAL)
fetcher = PmcCaseReportFetcher(config=config)
```

Se `NCBI_API_KEY` non è impostata, il recupero funziona comunque, solo più
lentamente (limite di frequenza senza chiave). Non è un errore bloccante.

Per usare un'email o una chiave diverse da quelle di default:

```python
config = FetchConfig(
    email="altro.indirizzo@dominio.it",
    api_key="chiave-passata-esplicitamente",  # solo se non usi from_environment
    license_group=LICENSE_COMMERCIAL,
)
fetcher = PmcCaseReportFetcher(config=config)

pmcids = fetcher.search_case_reports(
    query_terms=["cardiology", "differential diagnosis"],
    max_results=200,
)
fetcher.fetch_articles(pmcids)

report = fetcher.to_evaluation_cases()
print(report.as_dict())          # quanti casi caricati, quanti respinti e perché

cases = report.cases             # lista di EvaluationCase pronta per dream_capture_benchmark
```

## Sulla licenza

`license_group` è obbligatorio e vincolante:

| Valore | Uso |
|---|---|
| `LICENSE_COMMERCIAL` | CC0, CC BY, CC BY-SA, CC BY-ND — uso commerciale consentito |
| `LICENSE_NONCOMMERCIAL` | CC BY-NC e varianti — solo uso non commerciale |
| *(nessun terzo valore)* | `oa_other` non è mai recuperabile: nessuna licenza leggibile a macchina non può essere classificata |

Per un percorso commerciale, usare **sempre** `LICENSE_COMMERCIAL`. La licenza
viene letta dai metadati di ogni singolo articolo, non assunta dal filtro di
ricerca: un articolo con licenza diversa da quella configurata viene scartato
e registrato in `fetcher.skipped`, mai incluso per errore.

## Rispetto del limite di frequenza

Il connettore rispetta automaticamente il limite pubblicato da NCBI (3
richieste/secondo senza chiave, 10 con chiave). Su centinaia di articoli il
recupero richiede quindi qualche minuto — è normale, non un malfunzionamento.

## Cosa succede agli articoli problematici

Ogni articolo che fallisce — XML malformato, corpo vuoto, licenza diversa,
diagnosi non identificabile, presentazione che rivela già la diagnosi — finisce
in `fetcher.skipped` con una ragione, non interrompe l'esecuzione. Controllare
`fetcher.report()` dopo il recupero per vedere quanti articoli sono stati
scartati e perché.

## Dopo il recupero

Le `EvaluationCase` prodotte alimentano direttamente
`melampo.evaluation.dream_capture_benchmark`, seguendo il protocollo B4 già
implementato (Fase 1: misura di cattura senza clinico).

## Estrazione della diagnosi: attenzione

`_extract_diagnosis` prende la prima frase dopo l'intestazione di sezione
(*"Final diagnosis:"*, *"Diagnosis:"*, *"Conclusion:"*). È una euristica
grezza, pensata come punto di partenza. **Prima di usare i casi per una
valutazione che conta, far rivedere un campione da una persona** — è lo stesso
principio già applicato ovunque nel sistema: un'etichetta prodotta da una
macchina non è un'etichetta confermata finché qualcuno non l'ha verificata.
