#!/usr/bin/env python3
"""Come la Kaggle CLI (push / status / output) ma solo con la libreria standard: niente pip, funziona in Termux.
Credenziali: KAGGLE_USERNAME e KAGGLE_API_TOKEN (il token unico), oppure il vecchio ~/.kaggle/kaggle.json.

  python kaggle_api.py push      carica kaggle_imaging_probe.ipynb e lo fa girare sui server Kaggle (GPU, Internet)
  python kaggle_api.py status    running / complete / error
  python kaggle_api.py output    scarica probe.json, probe.log e il resto in ./output
  python kaggle_api.py quota     ore di GPU usate e rimaste

Con "replay" dopo il comando si usa il notebook di rilettura su CPU (nessuna quota GPU): rilegge le regioni
salvate dall'ultimo giro con GPU e rifà solo l'analisi. I risultati vanno in ./output_replay.
  python kaggle_api.py push replay
  python kaggle_api.py status replay
  python kaggle_api.py output replay

Con "holdout" si lavora sui casi tenuti da parte (MSD Task06 Lung dal 31° al 63°, mai usati per scegliere una
regola): un giro con GPU a sé (notebook melampo-imaging-holdout) e la sua rilettura su CPU. Risultati in
./output_holdout e ./output_replay_holdout.
  python kaggle_api.py push holdout            poi status holdout, output holdout
  python kaggle_api.py push replay holdout     poi status replay holdout, output replay holdout

Con "lidc" si lavora su LIDC-IDRI (quattro radiologi per nodulo): "misura" di default, "verifica" per i noduli
tenuti da parte. Risultati in ./output_lidc, ./output_lidc_verifica, ./output_replay_lidc, ...
  python kaggle_api.py push lidc                poi status lidc, output lidc            (GPU)
  python kaggle_api.py push replay lidc         poi status replay lidc, output replay lidc   (solo CPU)
  python kaggle_api.py push lidc verifica       poi status lidc verifica, output lidc verifica
"""
import base64, json, os, sys, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = "melampo-imaging-probe"
NOTEBOOK = os.path.join(HERE, "kaggle_imaging_probe.ipynb")
REPLAY = "replay" in sys.argv[2:]
HOLDOUT = "holdout" in sys.argv[2:]
LIDC = "lidc" in sys.argv[2:]
VERIFICA = LIDC and "verifica" in sys.argv[2:]
if LIDC:
    GPU_SLUG = "melampo-lidc" + ("-verifica" if VERIFICA else "")
    SLUG = "melampo-lidc-replay" + ("-verifica" if VERIFICA else "") if REPLAY else GPU_SLUG
    NOTEBOOK = os.path.join(HERE, "kaggle_lidc.ipynb")
    OUT = "output" + ("_replay" if REPLAY else "") + "_lidc" + ("_verifica" if VERIFICA else "")
else:
    GPU_SLUG = "melampo-imaging-holdout" if HOLDOUT else "melampo-imaging-probe"   # the run whose masks are read back
    SLUG = GPU_SLUG
    if REPLAY:
        SLUG, NOTEBOOK = "melampo-imaging-replay" + ("-holdout" if HOLDOUT else ""), \
            os.path.join(HERE, "kaggle_imaging_replay.ipynb")
    OUT = "output" + ("_replay" if REPLAY else "") + ("_holdout" if HOLDOUT else "")
BASE = "https://www.kaggle.com/api/v1"


def creds():
    """(username, token, 'bearer'|'basic'). Il nuovo token API di Kaggle (una sola stringa) va in
    KAGGLE_API_TOKEN, oppure nel file ~/.kaggle/access_token; il nome utente in KAGGLE_USERNAME.
    Il vecchio formato kaggle.json (username + key) funziona ancora."""
    user = os.environ.get("KAGGLE_USERNAME")
    token = os.environ.get("KAGGLE_API_TOKEN")
    tfile = os.path.expanduser("~/.kaggle/access_token")
    if not token and os.path.exists(tfile):
        token = open(tfile).read().strip()
    if token and user:
        return user, token, "bearer"
    key, path = os.environ.get("KAGGLE_KEY"), os.path.expanduser("~/.kaggle/kaggle.json")
    if not (user and key) and os.path.exists(path):
        data = json.load(open(path))
        user, key = data["username"], data["key"]
    if user and key:
        return user, key, "basic"
    sys.exit("Mancano le credenziali: esporta KAGGLE_USERNAME=<tuo nome utente> e KAGGLE_API_TOKEN=<il token>")


def call(method, url, body=None):
    user, key, mode = creds()
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Authorization", "Bearer " + key if mode == "bearer"
                   else "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode())
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:500]}")


def notebook_text():
    """Come la CLI ufficiale: output svuotati e ogni cella con il sorgente in una sola stringa (il server lo vuole così)."""
    nb = json.load(open(NOTEBOOK))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
        if LIDC and 'MODE = "gpu"' in cell.get("source", ""):
            src = cell["source"]
            if REPLAY:
                src = src.replace('MODE = "gpu"', 'MODE = "replay"')
            if VERIFICA:
                src = src.replace('SET = "misura"', 'SET = "verifica"')
            cell["source"] = src
        if HOLDOUT and "CASES = slice(0, 30)" in cell.get("source", ""):
            cell["source"] = cell["source"].replace("CASES = slice(0, 30)", "CASES = slice(30, 63)")
            cell["_holdout"] = True
    if HOLDOUT and not any(c.pop("_holdout", False) for c in nb.get("cells", [])):
        sys.exit("Il notebook non ha la riga CASES = slice(0, 30): kit vecchio, estrai di nuovo kaggle_kit.tar.gz")
    return json.dumps(nb)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    user = creds()[0]
    if cmd == "push":
        body = {
            "slug": f"{user}/{SLUG}", "newTitle": SLUG, "text": notebook_text(),
            "language": "python", "kernelType": "notebook", "isPrivate": True, "enableGpu": not REPLAY,
            "enableInternet": True, "datasetDataSources": [], "competitionDataSources": [],
            # la rilettura monta l'output dell'ultimo giro con GPU (le maschere salvate) in /kaggle/input
            "kernelDataSources": [f"{user}/{GPU_SLUG}"] if REPLAY else [],
            "modelDataSources": [], "categoryIds": []}
        if not REPLAY:
            body["machineShape"] = "NvidiaTeslaT4"
        out = call("POST", f"{BASE}/kernels/push", body)
        print(json.dumps(out, indent=1))
        if out.get("error"):
            print("\nATTENZIONE: il giro NON è partito. Lo stato resta quello del giro precedente.")
    elif cmd == "quota":
        print(json.dumps(call("GET", f"{BASE}/kernels/quota"), indent=1))
    elif cmd == "status":
        q = urllib.parse.urlencode({"userName": user, "kernelSlug": SLUG})
        print(json.dumps(call("GET", f"{BASE}/kernels/status?{q}"), indent=1))
    elif cmd == "output":
        # l'elenco dei file arriva a pagine: si seguono tutte (prima mancava probe.json, sulla seconda pagina)
        files, token, out = [], "", {}
        for _ in range(50):
            params = {"userName": user, "kernelSlug": SLUG, "pageSize": 200}
            if token:
                params["pageToken"] = token
            page = call("GET", f"{BASE}/kernels/output?{urllib.parse.urlencode(params)}")
            out = out or page
            files += page.get("files", [])
            token = page.get("nextPageToken") or ""
            if not token:
                break
        os.makedirs(OUT, exist_ok=True)
        everything = "all" in sys.argv[2:]     # "output all" scarica anche i .nii.gz e le maschere
        names = [os.path.basename(f["fileName"]) for f in files]
        print(f"{len(files)} file nell'output:", ", ".join(n for n in names if not n.endswith(".nii.gz")))
        for f in files:
            name = os.path.basename(f["fileName"])
            if not everything and name.endswith((".nii.gz", ".nii", ".tar", ".tar.gz", ".tgz", ".zip", ".npz")):
                print("saltato (dati, non servono):", name)
                continue
            urllib.request.urlretrieve(f["url"], os.path.join(OUT, name))
            print("scaricato", name)
        if out.get("log"):
            open(os.path.join(OUT, "log.json"), "w").write(out["log"])
    else:
        sys.exit(__doc__)


main()
