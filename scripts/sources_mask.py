#!/usr/bin/env python
"""The mask of the sources index (8 November 2026): a page on this computer to see the sources, add one, edit one,
and accept or reject the documents the weekly search found.

Since 18 November 2026 it also says which sources the disease profiles read: each source is switched on or off
for the profiles with one button, given its connector (how it is read) and its number of documents per disease;
a source is removed; a document is added by its link (PMID, PMCID, DOI or address) for the diseases it names; a
PDF is added from this computer: an open one (CC0, CC BY, CC BY-SA) goes to ``data/reference/documents`` and is
versioned, any other to ``data/private/pdfs``, which is never committed and is read only on this computer.

    python scripts/sources_mask.py [--port 8765]      # then open http://localhost:8765

It listens on this computer only (127.0.0.1), writes ``data/reference/sources.yml`` through the same checks as
the command line (``memory/sources_index.py``), and needs nothing beyond Python. It is the first of Melampo's
masks: the entry of reports and of the data for a diagnosis will follow, with the same look and the same rules.
"""

from __future__ import annotations

import argparse
import html
import re
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
             "terminology": "terminologia", "library": "raccolta", "document": "documento"},
    "access": {"open": "aperto", "free_registration": "registrazione gratuita", "subscription": "abbonamento",
               "institutional": "accesso istituzionale", "public_domain": "pubblico dominio",
               "purchase": "acquisto"},
    "status": {"in_use": "in uso", "available": "disponibile", "to_register": "da registrare",
               "not_reachable": "non raggiungibile", "candidate": "candidato", "rejected": "scartato"},
    "connector": {"": "—", "europe_pmc_reviews": "Europe PMC: review e linee guida aperte",
                  "pubmed_abstracts": "PubMed: abstract (solo ricerca)",
                  "ncbi_bookshelf": "NCBI Bookshelf: capitoli (StatPearls)",
                  "documents": "documenti aggiunti col link", "pdf_folder": "cartella di PDF",
                  "wikipedia_index": "Wikipedia come indice delle pubblicazioni citate"},
}
OPEN_PDFS = ROOT / "data" / "reference" / "documents"
PRIVATE_PDFS = ROOT / "data" / "private" / "pdfs"
MAX_UPLOAD = 80 * 1024 * 1024

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
.inline{display:inline}.on-p{color:var(--ok);font-weight:600}.off-p{color:var(--muted)}
fieldset{border:1px solid var(--line);border-radius:8px;padding:12px}legend{color:var(--muted);font-size:13px}
.check{display:flex;gap:6px;align-items:center;color:var(--fg)}
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
    in_profiles = sum(1 for s in index.sources if s.profiles and s.kind != "document")
    bar = f'<a href="/" class="{"on" if not show else ""}">tutte ({len(index.sources)})</a>'
    bar += (f'<a href="/?status=profiles" class="{"on" if show == "profiles" else ""}">lette per i profili '
            f'({in_profiles})</a>')
    bar += "".join(f'<a href="/?status={k}" class="{"on" if show == k else ""}">{t} ({counts[k]})</a>'
                   for k, t in LABELS["status"].items() if counts[k])
    rows = []
    for s in index.sources:
        if show == "profiles":
            if not (s.profiles and s.kind != "document"):
                continue
        elif show and s.status != show:
            continue
        accept = ""
        if s.status == "candidate":
            accept = (f'<form method="post" action="/status" class="inline"><input type="hidden" name="id" '
                      f'value="{html.escape(s.id)}"><button name="status" value="available">accetta</button> '
                      f'<button name="status" value="rejected">scarta</button></form>')
        sid = html.escape(s.id)
        if s.kind == "document":
            profiles = ("<small>" + html.escape(", ".join(s.diseases) or "per titolo") + "</small>"
                        if s.of in _readers_of_documents(index) else "")
        else:
            connector = html.escape(LABELS["connector"].get(s.connector, s.connector))
            label = (f"<span class='on-p'>sì</span><br><small>{connector}"
                     f"{' · ' + str(s.max_documents) + ' doc.' if s.max_documents else ''}"
                     f"{' · solo ricerca' if s.research_only else ''}</small>") if s.profiles else \
                "<span class='off-p'>no</span>"
            switch = (f'<form method="post" action="/profiles" class="inline"><input type="hidden" name="id" '
                      f'value="{sid}"><button name="profiles" value="{"0" if s.profiles else "1"}">'
                      f'{"disattiva" if s.profiles else "attiva"}</button></form>') if s.connector else \
                "<br><small>scegli prima come si legge (modifica)</small>"
            profiles = label + "<br>" + switch
        remove = (f'<form method="post" action="/remove" class="inline" onsubmit="return confirm(\'Rimuovere '
                  f'{sid} dall&#39;indice?\')"><input type="hidden" name="id" value="{sid}">'
                  f'<button>rimuovi</button></form>')
        rows.append(
            f"<tr><td><b>{html.escape(s.title)}</b><br><small>{html.escape(s.publisher)}"
            f"{' · ' + html.escape(s.identifier) if s.identifier else ''}</small></td>"
            f"<td>{LABELS['kind'].get(s.kind, s.kind)}</td><td>{LABELS['access'].get(s.access, s.access)}"
            f"<br><small>{html.escape(s.licence)}</small></td>"
            f"<td><span class='pill {s.status}'>{LABELS['status'].get(s.status, s.status)}</span></td>"
            f"<td>{profiles}</td>"
            f"<td><small>{html.escape(s.how_to_reach or s.note)}</small></td>"
            f"<td><a href='{html.escape(s.url)}' target='_blank' rel='noopener'>apri</a> · "
            f"<a href='/?edit={sid}#modulo'>modifica</a> {accept} {remove}</td></tr>")
    msg = f'<div class="msg {"bad" if bad else "ok"}">{html.escape(message)}</div>' if message else ""
    c = current
    fields = ", ".join(c.fields)
    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Indice delle fonti</title>
<style>{STYLE}</style></head><body>
<header><h1>Indice delle fonti di Melampo</h1><p>Letteratura, trattati, ricerche e documentazione consolidata: che
cosa è, come si raggiunge, con quale licenza. La ricerca settimanale aggiunge i documenti come candidati.</p></header>
<main>{msg}<div class="bar">{bar}</div>
<table><thead><tr><th>Fonte</th><th>Tipo</th><th>Accesso e licenza</th><th>Stato</th><th>Profili delle malattie</th>
<th>Come raggiungerla</th><th></th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{_add_forms(index)}
<h2 style="font-size:16px;margin:8px 0 0">Aggiungi o modifica una fonte</h2>
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
<fieldset class="wide"><legend>Lettura per i profili delle malattie</legend>
<label class="check"><input type="checkbox" name="profiles" value="1"{" checked" if c.profiles else ""}>
leggi questa fonte per i profili</label>
<label>Come si legge (connettore)<select name="connector">{_opt('connector', c.connector)}</select></label>
<label>Documenti per malattia (0 = predefinito)<input name="max_documents" type="number" min="0" max="100"
value="{c.max_documents or 0}"></label>
<label class="check"><input type="checkbox" name="research_only" value="1"{" checked" if c.research_only else ""}>
licenza solo per ricerca (letta solo con --research)</label>
<label>Cartella o file (per i PDF)<input name="path" value="{html.escape(c.path)}"></label>
<label>Restringi la ricerca (query)<input name="query" value="{html.escape(c.query)}"></label>
<label>Malattie (solo per un documento, separate da virgola)<input name="diseases"
value="{html.escape(', '.join(c.diseases))}"></label></fieldset>
<div class="wide"><button type="submit">Salva</button> <a href="/">nuova fonte</a></div>
</form></main></body></html>"""


def _readers_of_documents(index: si.SourcesIndex) -> set[str]:
    return {s.id for s in index.sources if s.connector == "documents"}


def _folders(index: si.SourcesIndex) -> list[si.Source]:
    return [s for s in index.sources if s.connector == "pdf_folder"]


def _add_forms(index: si.SourcesIndex) -> str:
    """The two quick ways to add a source to the profiles: a document by its link, a PDF from this computer."""
    of = "".join(f'<option value="{html.escape(x)}">{html.escape(x)}</option>'
                 for x in sorted(_readers_of_documents(index)))
    folders = "".join(f'<option value="{html.escape(s.id)}">{html.escape(s.title)} ({html.escape(s.path)})</option>'
                      for s in _folders(index))
    return f"""<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px">
<form class="edit" method="post" action="/link" style="grid-template-columns:1fr">
<b>Aggiungi un documento col suo link</b>
<label>PMID, PMCID, DOI o indirizzo<input name="link" required placeholder="PMC5187920, 27999274, 10.3390/..., https://..."></label>
<label>Titolo<input name="title" required></label>
<label>Malattie per cui leggerlo (separate da virgola; vuoto = quelle nominate nel titolo)<input name="diseases"></label>
<label>Licenza<input name="licence" placeholder="CC BY 4.0"></label>
<label>Raccolta<select name="of">{of}</select></label>
<div><button type="submit">Aggiungi</button></div></form>
<form class="edit" method="post" action="/pdf" enctype="multipart/form-data" style="grid-template-columns:1fr">
<b>Aggiungi un PDF da questo computer</b>
<label>File PDF<input type="file" name="file" accept="application/pdf,.pdf" required></label>
<label>Titolo<input name="title" required></label>
<label>PMID, DOI o ISBN e pagine<input name="identifier"></label>
<label>Anno<input name="year"></label>
<label>Malattie (separate da virgola; vuoto = quelle nominate nel titolo)<input name="diseases"></label>
<label>Licenza<input name="licence" placeholder="CC BY 4.0, oppure: abbonamento personale"></label>
<label>Cartella<select name="folder">{folders}</select></label>
<small>Un PDF con licenza aperta (CC0, CC BY, CC BY-SA) va nella cartella del repository e viene versionato;
qualunque altro va nella cartella privata, che non viene mai inviata a GitHub e si legge solo su questo computer.</small>
<div><button type="submit">Carica</button></div></form></div>"""


def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name or "document.pdf").stem).strip("._")[:80] or "document"
    return stem + ".pdf"


def save_pdf(index: si.SourcesIndex, form: dict[str, str], data: bytes, filename: str) -> tuple[str, bool]:
    """Write an uploaded PDF and its citation beside it. An open licence (CC0, CC BY, CC BY-SA) goes to a versioned
    folder, any other to a research-only one (never committed), whatever folder was chosen: a closed PDF can never
    end up in the public repository by mistake. Returns the message and whether it failed."""
    if not data.startswith(b"%PDF"):
        return "non salvato: il file non è un PDF", True
    title = form.get("title", "").strip()
    if not title:
        return "non salvato: il titolo è obbligatorio", True
    from melampo.memory import disease_profiles as dp

    licence = form.get("licence", "").strip()
    open_ = dp.open_for_use(licence)
    chosen = index.by_id().get(form.get("folder", ""))
    fits = [s for s in _folders(index) if s.research_only != open_]
    folder_source = chosen if chosen in fits else (fits[0] if fits else None)
    folder = Path(folder_source.path) if folder_source else (OPEN_PDFS if open_ else PRIVATE_PDFS)
    folder = folder if folder.is_absolute() else ROOT / folder
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_name(filename or title)
    base, n = target.stem, 1
    while target.exists():
        target = folder / f"{base}-{n}.pdf"
        n += 1
    target.write_bytes(data)
    import yaml

    meta = {"title": title, "identifier": form.get("identifier", "").strip(), "year": form.get("year", "").strip(),
            "licence": licence, "diseases": [d.strip() for d in form.get("diseases", "").split(",") if d.strip()]}
    target.with_suffix(".yml").write_text(yaml.safe_dump({k: v for k, v in meta.items() if v}, allow_unicode=True,
                                                         sort_keys=False), "utf-8")
    where = "cartella aperta del repository" if open_ else "cartella privata (mai inviata a GitHub)"
    return f"salvato: {target.name} nella {where}", False


def multipart(content_type: str, body: bytes) -> tuple[dict[str, str], bytes, str]:
    """The fields, the file and its name of a multipart form (Python's email parser: nothing to install)."""
    from email.parser import BytesParser
    from email.policy import HTTP

    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    fields, data, filename = {}, b"", ""
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition") or ""
        if part.get_filename():
            data, filename = part.get_payload(decode=True) or b"", part.get_filename()
        elif name:
            fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace").strip()
    return fields, data, filename


def document_from_link(form: dict[str, str]) -> si.Source:
    """A document added by its link: a PMCID, a PMID, a DOI or an address, with the address a person opens."""
    link = form.get("link", "").strip()
    ident, url = "", link
    if re.fullmatch(r"(?i)pmc\d+", link):
        ident, url = link.upper(), f"https://europepmc.org/article/PMC/{link.upper()}"
    elif re.fullmatch(r"\d{5,9}", link):
        ident, url = link, f"https://pubmed.ncbi.nlm.nih.gov/{link}/"
    elif link.lower().removeprefix("doi:").startswith("10."):
        ident = link.lower().removeprefix("doi:")
        url = f"https://doi.org/{ident}"
    sid = "doc-" + (re.sub(r"[^a-z0-9]+", "-", (ident or link).lower()).strip("-")[:70] or "link")
    return si.Source(id=sid, kind="document", title=form.get("title", ""), url=url, access="open",
                     status="available", licence=form.get("licence", ""), of=form.get("of", ""), identifier=ident,
                     added_by="manual",
                     diseases=[d.strip() for d in form.get("diseases", "").split(",") if d.strip()])


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
        if self.path == "/pdf":
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                self._send(page(index, {}, "non salvato: il file supera 80 MB", True), 413)
                return
            fields, data, filename = multipart(self.headers.get("Content-Type", ""), self.rfile.read(n))
            message, bad = save_pdf(index, fields, data, filename)
            self._send(page(index, {}, message, bad), 400 if bad else 200)
            return
        f = self._form()
        if self.path == "/profiles":
            s = index.by_id().get(f.get("id", ""))
            if s is None:
                self._send(page(index, {}, "fonte sconosciuta", True), 400)
                return
            s.profiles = f.get("profiles") == "1"
            problems = s.problems()
            if problems:
                self._send(page(index, {}, "non cambiato: " + "; ".join(problems), True), 400)
                return
            index.save()
            self._send(page(index, {}, f"{s.id}: {'letta' if s.profiles else 'non più letta'} per i profili"))
            return
        if self.path == "/remove":
            problems = index.remove(f.get("id", ""))
            message = "non rimossa: " + "; ".join(problems) if problems else f"rimossa: {f.get('id')}"
            self._send(page(si.SourcesIndex.load(self.index_path), {}, message, bool(problems)),
                       400 if problems else 200)
            return
        if self.path == "/link":
            source = document_from_link(f)
            problems = index.put(source)
            message = "non salvato: " + "; ".join(problems) if problems else f"aggiunto: {source.id}"
            self._send(page(si.SourcesIndex.load(self.index_path), {}, message, bool(problems)),
                       400 if problems else 200)
            return

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
        extra = si.source_from({"profiles": f.get("profiles", ""), "research_only": f.get("research_only", ""),
                                "max_documents": f.get("max_documents", "0"), "diseases": f.get("diseases", "")})
        source.profiles, source.research_only = extra.profiles, extra.research_only
        source.max_documents, source.diseases = extra.max_documents, extra.diseases
        source.connector, source.path, source.query = f.get("connector", ""), f.get("path", ""), f.get("query", "")
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
