#!/usr/bin/env python
"""The mask of the sources index (8 November 2026): a page on this computer to see the sources, add one, edit one,
and accept or reject the documents the weekly search found.

    python scripts/sources_mask.py [--port 8765]      # then open http://localhost:8765

It listens on this computer only (127.0.0.1), writes ``data/reference/sources.yml`` through the same checks as
the command line (``memory/sources_index.py``), and needs nothing beyond Python. It is the first of Melampo's
masks: the entry of reports and of the data for a diagnosis will follow, with the same look and the same rules.
"""

from __future__ import annotations

import argparse
import html
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory import sources_index as si

LABELS = {
    "kind": {"database": "banca dati", "journal": "rivista", "guideline": "linea guida", "textbook": "trattato",
             "open_textbook": "trattato aperto", "reference_site": "sito di riferimento",
             "terminology": "terminologia", "document": "documento"},
    "access": {"open": "aperto", "free_registration": "registrazione gratuita", "subscription": "abbonamento",
               "institutional": "accesso istituzionale", "public_domain": "pubblico dominio",
               "purchase": "acquisto"},
    "status": {"in_use": "in uso", "available": "disponibile", "to_register": "da registrare",
               "not_reachable": "non raggiungibile", "candidate": "candidato", "rejected": "scartato"},
}

STYLE = """
:root{--bg:#fafaf7;--fg:#1d2329;--muted:#5d6670;--line:#d9ddd4;--accent:#1565c0;--ok:#2e7d32;--warn:#b26a00;
--bad:#b3261e;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#15181b;--fg:#e8eaed;--muted:#a0a7ae;--line:#33393f;--accent:#7fb2f0;
--ok:#7bc47f;--warn:#f0b357;--bad:#f28b82;--card:#1d2125}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line)}h1{margin:0;font-size:20px}
header p{margin:4px 0 0;color:var(--muted)}main{padding:16px 24px;display:grid;gap:20px;max-width:1300px}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.bar a,button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;
padding:5px 10px;text-decoration:none;font:inherit;cursor:pointer}.bar a.on{border-color:var(--accent);color:var(--accent)}
table{border-collapse:collapse;width:100%;background:var(--card)}th,td{border-bottom:1px solid var(--line);
padding:7px 8px;text-align:left;vertical-align:top}th{font-size:13px;color:var(--muted);font-weight:600}
td small{color:var(--muted)}.pill{white-space:nowrap;font-size:12px;border-radius:10px;padding:1px 8px;border:1px solid var(--line)}
.in_use{color:var(--ok)}.to_register{color:var(--warn)}.candidate{color:var(--accent)}.rejected,.not_reachable{color:var(--bad)}
form.edit{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;display:grid;
grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
label{display:grid;gap:4px;font-size:13px;color:var(--muted)}input,select,textarea{font:inherit;color:var(--fg);
background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px}textarea{min-height:64px}
.wide{grid-column:1/-1}.msg{padding:10px 12px;border-radius:6px;border:1px solid var(--line)}
.msg.bad{border-color:var(--bad);color:var(--bad)}.msg.ok{border-color:var(--ok);color:var(--ok)}
.inline{display:inline}
"""


def _opt(name: str, current: str) -> str:
    return "".join(f'<option value="{v}"{" selected" if v == current else ""}>{html.escape(t)}</option>'
                   for v, t in LABELS[name].items())


def page(index: si.SourcesIndex, query: dict[str, list[str]], message: str = "", bad: bool = False,
         draft: si.Source | None = None) -> str:
    show = (query.get("status") or [""])[0]
    edit_id = (query.get("edit") or [""])[0]
    current = draft or index.by_id().get(edit_id) or si.Source(id="", kind="journal", title="", url="",
                                                                 access="open", status="available")
    counts = {k: sum(s.status == k for s in index.sources) for k in LABELS["status"]}
    bar = f'<a href="/" class="{"on" if not show else ""}">tutte ({len(index.sources)})</a>'
    bar += "".join(f'<a href="/?status={k}" class="{"on" if show == k else ""}">{t} ({counts[k]})</a>'
                   for k, t in LABELS["status"].items() if counts[k])
    rows = []
    for s in index.sources:
        if show and s.status != show:
            continue
        accept = ""
        if s.status == "candidate":
            accept = (f'<form method="post" action="/status" class="inline"><input type="hidden" name="id" '
                      f'value="{html.escape(s.id)}"><button name="status" value="available">accetta</button> '
                      f'<button name="status" value="rejected">scarta</button></form>')
        rows.append(
            f"<tr><td><b>{html.escape(s.title)}</b><br><small>{html.escape(s.publisher)}"
            f"{' · ' + html.escape(s.identifier) if s.identifier else ''}</small></td>"
            f"<td>{LABELS['kind'].get(s.kind, s.kind)}</td><td>{LABELS['access'].get(s.access, s.access)}"
            f"<br><small>{html.escape(s.licence)}</small></td>"
            f"<td><span class='pill {s.status}'>{LABELS['status'].get(s.status, s.status)}</span></td>"
            f"<td><small>{html.escape(s.how_to_reach or s.note)}</small></td>"
            f"<td><a href='{html.escape(s.url)}' target='_blank' rel='noopener'>apri</a> · "
            f"<a href='/?edit={html.escape(s.id)}#modulo'>modifica</a> {accept}</td></tr>")
    msg = f'<div class="msg {"bad" if bad else "ok"}">{html.escape(message)}</div>' if message else ""
    c = current
    fields = ", ".join(c.fields)
    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Indice delle fonti</title>
<style>{STYLE}</style></head><body>
<header><h1>Indice delle fonti di Melampo</h1><p>Letteratura, trattati, ricerche e documentazione consolidata: che
cosa è, come si raggiunge, con quale licenza. La ricerca settimanale aggiunge i documenti come candidati.</p></header>
<main>{msg}<div class="bar">{bar}</div>
<table><thead><tr><th>Fonte</th><th>Tipo</th><th>Accesso e licenza</th><th>Stato</th><th>Come raggiungerla</th>
<th></th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<form class="edit" id="modulo" method="post" action="/save">
<label>Identificativo (minuscole, cifre, - . _)<input name="id" required value="{html.escape(c.id)}"></label>
<label>Titolo<input name="title" required value="{html.escape(c.title)}"></label>
<label>Editore o autore<input name="publisher" value="{html.escape(c.publisher)}"></label>
<label>Indirizzo (URL)<input name="url" type="url" required value="{html.escape(c.url)}"></label>
<label>Tipo<select name="kind">{_opt('kind', c.kind)}</select></label>
<label>Accesso<select name="access">{_opt('access', c.access)}</select></label>
<label>Stato<select name="status">{_opt('status', c.status)}</select></label>
<label>Licenza<input name="licence" value="{html.escape(c.licence)}"></label>
<label>Ambiti (separati da virgola: {', '.join(si.FIELDS)})<input name="fields" value="{html.escape(fields)}"></label>
<label>Di quale fonte (solo per un documento)<input name="of" value="{html.escape(c.of)}"></label>
<label>PMID, PMCID, DOI o ISBN<input name="identifier" value="{html.escape(c.identifier)}"></label>
<label class="wide">Come raggiungerla (obbligatorio se chiusa: abbonamento, accesso istituzionale, acquisto)
<textarea name="how_to_reach">{html.escape(c.how_to_reach)}</textarea></label>
<label class="wide">Nota<textarea name="note">{html.escape(c.note)}</textarea></label>
<div class="wide"><button type="submit">Salva</button> <a href="/">nuova fonte</a></div>
</form></main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    index_path: Path = si.SOURCES

    def _send(self, body: str, code: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _form(self) -> dict[str, str]:
        n = int(self.headers.get("Content-Length") or 0)
        q = parse_qs(self.rfile.read(n).decode("utf-8"), keep_blank_values=True)
        return {k: v[0].strip() for k, v in q.items()}

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path != "/":
            self._send("non trovato", 404)
            return
        self._send(page(si.SourcesIndex.load(self.index_path), parse_qs(u.query)))

    def do_POST(self) -> None:
        index = si.SourcesIndex.load(self.index_path)
        f = self._form()
        if self.path == "/status":
            s = index.by_id().get(f.get("id", ""))
            if s is None or f.get("status") not in LABELS["status"]:
                self._send(page(index, {}, "fonte o stato sconosciuto", True), 400)
                return
            s.status = f["status"]
            index.save()
            self._send(page(index, {"status": ["candidate"]}, f"{s.id}: {LABELS['status'][s.status]}"))
            return
        old = index.by_id().get(f.get("id", ""))
        source = si.Source(id=f.get("id", ""), kind=f.get("kind", ""), title=f.get("title", ""),
                           url=f.get("url", ""), access=f.get("access", ""), status=f.get("status", ""),
                           publisher=f.get("publisher", ""), licence=f.get("licence", ""),
                           fields=[x.strip() for x in f.get("fields", "").split(",") if x.strip()],
                           how_to_reach=f.get("how_to_reach", ""), note=f.get("note", ""), of=f.get("of", ""),
                           identifier=f.get("identifier", ""), added=old.added if old else "",
                           added_by=old.added_by if old else "manual", last_checked=old.last_checked if old else "")
        problems = index.put(source)
        if problems:
            self._send(page(index, {}, "non salvato: " + "; ".join(problems), True, draft=source), 400)
            return
        self._send(page(si.SourcesIndex.load(self.index_path), {}, f"salvato: {source.id}"))

    def log_message(self, *args) -> None:  # quiet
        return


def serve(port: int, index: Path) -> None:
    Handler.index_path = index
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"indice delle fonti: http://localhost:{port}  (Ctrl+C per chiudere)")
    server.serve_forever()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--index", default=str(si.SOURCES))
    args = ap.parse_args(argv)
    serve(args.port, Path(args.index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
