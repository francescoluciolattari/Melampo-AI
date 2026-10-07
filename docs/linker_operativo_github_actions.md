# Linker anatomico: come si lanciano i workflow (GitHub Actions)

Guida operativa per `public-reports` (referti pubblici e fogli ciechi) e `linking-bench` (misura del linker con i due modelli). Vale dal commit del ramo `feat/linker-streams` in poi.

## Regola generale

In **Actions** si sceglie il workflow nella colonna a sinistra, poi **Run workflow** in alto a destra della lista dei run. Il menu **Use workflow from** in cima al modulo sceglie il ramo, e **il modulo mostra solo le opzioni del file di workflow di quel ramo**. Se un'opzione manca (per esempio `verify`), il ramo scelto non contiene la modifica: non è un errore del workflow.

## `public-reports`

| Campo | Cosa scrivere | Note |
|---|---|---|
| Use workflow from | `main` (dopo il merge) o `feat/linker-streams` | il loader MultiCaRe corretto sta solo dopo il merge |
| source | `multicare` (o `parrot-it`, `iuxray`, `e3c-it`; i `discover-*` stampano solo le colonne) | |
| n | **solo il numero**, per esempio `600` | vuoto = tutte le menzioni proposte |
| cap | **solo il numero**, per esempio `30` | vuoto = nessun tetto per menzione scritta; serve dove un nome domina (IU X-ray: "heart") |

I campi si chiamano già `n` e `cap`: scrivere `n=600` era un errore (lo script rispondeva `invalid int value: 'n=600'`). Dal commit di questa guida il workflow e lo script accettano anche `n=600` e `cap=30`, ma un valore che non è un numero ferma il run con un messaggio chiaro.

Risultato: l'artifact `public-reports-<source>` con `reports.jsonl` e la cartella `gold_study` (schede cieche `annotator_A.csv`, `annotator_B.csv`, `valid_structures.txt`, `items.jsonl`). `items.jsonl` è per gli sviluppatori e non va agli annotatori.

## `linking-bench`

| Campo | Cosa scrivere |
|---|---|
| Use workflow from | un ramo che contiene `graph` e `verify` (`main` dopo il merge, o `feat/linker-streams`) |
| mode | `linker` per misurare il linker |
| graph | casella da spuntare (acceso di default): controllo dei vicini nel grafo UBERON |
| verify | casella da spuntare (spenta di default): per i link dal solo nome a una forma a rischio, i due modelli leggono la frase e devono dire SÌ tutti e due |

Per misurare l'effetto della verifica si lanciano **due run identici**, uno con `verify` spenta e uno con `verify` accesa (`graph` acceso in entrambi). Serve il secret `OPENROUTER_API_KEY` nel repository (Settings → Secrets and variables → Actions).

Come si riconosce il run: il riepilogo del run in cima riporta ramo, commit, mode, graph e verify. L'artifact si chiama `linking-bench-results-graph-<true|false>-verify-<true|false>`, quindi i due zip hanno nomi diversi. Contiene `linking_results.json` e `linking_summary.md`.

## Se qualcosa non torna

| Sintomo | Causa probabile | Rimedio |
|---|---|---|
| `verify` (o `graph`) non compare nel modulo | ramo senza il bundle | scegliere `feat/linker-streams` oppure fare il merge in `main` |
| `invalid int value: 'n=600'` | scritto il nome del campo insieme al valore | scrivere `600` (con il workflow aggiornato non succede più) |
| `n and cap take a whole number` | valore non numerico | scrivere solo cifre |
| il run `public-reports` per MultiCaRe dura ore | ramo senza il campionamento con arresto anticipato | usare un ramo con il bundle |
| il run con `verify` non chiama i modelli | manca `OPENROUTER_API_KEY` | aggiungere il secret |

## Applicare un bundle (Termux)

```
cd ~/github/Melampo-AI
git status -sb | head -3          # se c'è un rebase in corso: git rebase --abort
git switch main
git fetch ~/<nome-del-bundle>.bundle +feat/linker-streams:refs/heads/feat/linker-streams
git push -f origin feat/linker-streams
git log --oneline -3 feat/linker-streams
```

Il bundle richiede il commit di base `c9cf2be` (merge di `feat/public-reports`). Il merge in `main` si fa poi dalla pagina del ramo su GitHub.
