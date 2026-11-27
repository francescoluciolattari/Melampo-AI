#!/usr/bin/env python
"""The review mask (27 November 2026): a page on this computer to open a case, add its documents, say which of
them are used, read the case, and answer the questions the reading sends to a person.

    python scripts/review_mask.py [--port 8766]      # then open http://localhost:8766

The documents (``data/case_review.py``, the owner's rules):

* a document whose identifiers are another patient's is **discarded**, and no one can change it;
* a document without identifiers, or with the patient's, is part of the patient's file: the mask asks whether to
  use it, with **yes** given;
* a document with one identifier of the patient and one of another: the mask asks whose it is;
* a document of no recognised kind: the mask asks what it is (with **no** given when the local model too
  recognised none).

The doubts of the cards (a field the two interpreters read differently, a finding only one read, a part the heading
contradicts) and the questions of the reading show their closed answers first; what each interpreter read is shown
only after the answer (a suggestion seen first steers the reading). An answer is a proposal until a second person
confirms it, and a third adjudicates it when it is none of the interpreters' readings; only then does it enter the
archive of the confirmations, de-identified (``review_feedback``, ``data/deidentify.py``). Everything stays on this computer, under
``data/private/cases`` (never committed); it listens on 127.0.0.1 only and needs nothing beyond Python.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))   # mask_common: the masks' one look and one form reader

from mask_common import STYLE, multipart

from melampo.data.case_attachments import DISCARDED, NOT_USED, USE
from melampo.data.case_review import CASES, CHECK_LABEL, CaseFolder

MAX_UPLOAD = 200 * 1024 * 1024
DECISION_LABEL = {USE: "usato", NOT_USED: "non usato", DISCARDED: "scartato"}
COMPARED_LABEL = {"fiscal_code_matches": "codice fiscale uguale", "fiscal_code_differs": "codice fiscale diverso",
                  "birth_date_matches": "data di nascita uguale", "birth_date_differs": "data di nascita diversa",
                  "name_matches": "nome uguale", "name_differs": "nome diverso", "sex_differs": "sesso diverso"}
QUESTION_LABEL = {"use": "Usare questo documento nel caso?", "whose": "Di chi è questo documento?",
                  "what": "Che cosa è questo documento?"}
e = html.escape


def _head(title: str, message: str = "", bad: bool = False) -> str:
    msg = f'<div class="msg {"bad" if bad else "ok"}">{e(message)}</div>' if message else ""
    return (f'<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{e(title)}</title><style>{STYLE}</style></head>'
            f'<body><header><h1>{e(title)}</h1><p>Maschera di revisione di Melampo: tutto resta su questo computer.'
            f'</p></header><main>{msg}')


def home(root: Path, message: str = "", bad: bool = False) -> str:
    cases = CaseFolder.all(root)
    rows = "".join(f'<tr><td><a href="/case?id={quote(c.id)}">{e(c.id)}</a></td><td>{len(c.files())}</td>'
                   f'<td>{"letto" if c.reading() else "da leggere"}</td></tr>' for c in cases)
    table = (f"<table><tr><th>Caso</th><th>Documenti</th><th>Lettura</th></tr>{rows}</table>" if cases
             else "<p>Nessun caso.</p>")
    return _head("Casi", message, bad) + table + """
<form class="edit" method="post" action="/new">
<label>Identificativo del caso (lettere, cifre, - _)<input name="id" required pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,63}"></label>
<label>Cognome<input name="patient_surname"></label>
<label>Nome<input name="patient_name"></label>
<label>Codice fiscale<input name="patient_fiscal_code" maxlength="16"></label>
<label>Data di nascita (gg/mm/aaaa)<input name="patient_birth_date"></label>
<div class="wide"><button type="submit">Apri un nuovo caso</button></div>
</form></main></body></html>"""


def _document_row(case: CaseFolder, r) -> str:
    compared = ", ".join(COMPARED_LABEL.get(c, c) for c in r.compared)
    check = f'{e(CHECK_LABEL.get(r.check, r.check))}{f"<br><small>{e(compared)}</small>" if compared else ""}'
    notes = "".join(f"<br><small>{e(n)}</small>" for n in r.notes)
    if r.locked:
        choice = f'<span class="lock">scartato — non modificabile</span>{notes}'
    else:
        # the answer a person finds marked: theirs, or for "use" the decision given (yes)
        marked = r.answer or ({USE: "yes", NOT_USED: "no"}.get(r.given, "") if r.question == "use" else "")
        options = "".join(
            f'<label class="check"><input type="radio" name="answer" value="{e(code)}" required'
            f'{" checked" if code == marked else ""}>{e(label)}</label>' for code, label, _ in r.options)
        who = (f'<small>risposto da {e(r.by)} il {e(r.at)}</small>' if r.by
               else f'<span class="given">dato: {e(DECISION_LABEL[r.given])}</span>')
        choice = (f'<form method="post" action="/decide"><input type="hidden" name="id" value="{e(case.id)}">'
                  f'<input type="hidden" name="doc" value="{e(r.id)}"><b>{e(QUESTION_LABEL[r.question])}</b>'
                  f'{options}<label>Chi risponde<input name="by" required value="{e(r.by)}"></label>'
                  f'<button type="submit">Salva</button> {who}</form>{notes}')
    klass = "lock" if r.decision == DISCARDED else ("doubt" if r.question in ("whose", "what") else "")
    return (f'<tr><td>{e(r.name)}<br><small>{e(", ".join(r.kinds) or "—")}</small></td><td>{check}</td>'
            f'<td class="{klass}">{e(DECISION_LABEL[r.decision])}</td><td>{choice}</td></tr>')


def _question(case: CaseFolder, q: dict, answered: dict | None) -> str:
    sentence = e(q.get("sentence", ""))
    if answered:
        system = e(json.dumps(q.get("system", {}), ensure_ascii=False, indent=1))
        return (f'<details open><summary><b>{e(q.get("question_it", ""))}</b></summary><p>«{sentence}»</p>'
                f'<p>Risposta: <b>{e(answered["answer"])}</b> {e(answered.get("text", ""))} — {e(answered["by"])}, '
                f'{e(answered["at"])}</p><p><small>Che cosa aveva letto il sistema (mostrato solo dopo la risposta):'
                f'</small></p><pre>{system}</pre></details>')
    options = "".join(f'<label class="check"><input type="radio" name="code" value="{e(o["code"])}" required>'
                      f'{e(o.get("label_it", o["code"]))}</label>' for o in q.get("options", []))
    return (f'<details open><summary><b>{e(q.get("question_it", ""))}</b></summary><p>«{sentence}»</p>'
            f'<form method="post" action="/answer"><input type="hidden" name="id" value="{e(case.id)}">'
            f'<input type="hidden" name="q" value="{e(q["id"])}">{options}'
            f'<label>Struttura nominata (solo se la risposta la chiede)<input name="text"></label>'
            f'<label>Chi risponde<input name="by" required></label><button type="submit">Rispondi</button>'
            f'</form></details>')


def _card_question(case: CaseFolder, q: dict, answered: dict | None) -> str:
    """The doubt of a card: its closed answers; once answered, what each interpreter read, and the confirmation
    of a second person (and of a third when the answer is none of the interpreters')."""
    head = (f'<details open><summary><b>{e(q["name"])}</b> — <small>{e(q["doubt"])}, {e(q["report"])}</small>'
            f'</summary><p>«{e(q["sentence"])}»</p>')
    hidden = f'<input type="hidden" name="id" value="{e(case.id)}"><input type="hidden" name="q" value="{e(q["id"])}">'
    if not answered:
        options = "".join(f'<label class="check"><input type="radio" name="code" value="{e(o["code"])}" required>'
                          f'{e(o["label_it"])}</label>' for o in q["options"])
        return (head + f'<form method="post" action="/card-answer">{hidden}{options}'
                f'<label>Se «altro»: la lettura giusta<input name="text"></label>'
                f'<label>Chi risponde<input name="by" required></label><button type="submit">Rispondi</button>'
                f'</form></details>')
    label = next((o["label_it"] for o in q["options"] if o["code"] == answered["answer"]), answered["answer"])
    system = e(json.dumps(q["system"]["alternatives"], ensure_ascii=False, indent=1))
    body = (f'<p>Risposta: <b>{e(label)}</b> {e(answered.get("text", ""))} — {e(answered["by"])}, '
            f'{e(answered["at"])}</p><p><small>Che cosa aveva letto ogni interprete (mostrato solo dopo la '
            f'risposta):</small></p><pre>{system}</pre>')
    if not answered.get("confirmed_by"):
        body += (f'<form method="post" action="/card-confirm">{hidden}<input type="hidden" name="role" '
                 f'value="confirmed"><label>Conferma di una seconda persona<input name="by" required></label>'
                 f'<button type="submit">Confermo</button></form>')
    else:
        body += f'<p>Confermata da {e(answered["confirmed_by"])}</p>'
        if answered.get("needs_adjudication") and not answered.get("adjudicated_by"):
            body += (f'<p class="doubt">La risposta non è quella di nessun interprete: serve una terza persona.</p>'
                     f'<form method="post" action="/card-confirm">{hidden}<input type="hidden" name="role" '
                     f'value="adjudicated"><label>Terza persona<input name="by" required></label>'
                     f'<button type="submit">Confermo</button></form>')
        elif answered.get("adjudicated_by"):
            body += f'<p>Decisa da una terza persona: {e(answered["adjudicated_by"])}</p>'
    return head + body + "</details>"


def case_page(case: CaseFolder, message: str = "", bad: bool = False) -> str:
    rows = case.rows()
    used = sum(r.decision == USE for r in rows)
    table = ("<table><tr><th>Documento</th><th>Identificativi</th><th>Nel caso</th><th>Revisione</th></tr>"
             + "".join(_document_row(case, r) for r in rows) + "</table>") if rows else "<p>Nessun documento.</p>"
    reading = case.reading()
    if reading is None:
        read = "<p>Il caso non è ancora stato letto.</p>"
    else:
        answers = case.review()["answers"]
        questions = "".join(_question(case, q, answers.get(q["id"])) for q in case.questions())
        cards = case.card_questions()
        answered = case.review().get("cards", {})
        card_part = "".join(_card_question(case, q, answered.get(q["id"])) for q in cards)
        doubts = "".join(f'<tr><td>{e(d.get("report", ""))}</td><td>{e(d.get("mention", ""))}</td>'
                         f'<td>{e(d.get("reason", ""))}</td><td><small>{e(d.get("clause", ""))}</small></td></tr>'
                         for d in case.doubts())
        read = (f"<h2>Dubbi delle schede ({len(cards)})</h2>{card_part or '<p>Nessuno.</p>'}"
                f"<h2>Domande della lettura ({len(case.questions())})</h2>{questions or '<p>Nessuna.</p>'}"
                f"<h2>Altro da rivedere ({len(case.doubts())})</h2>"
                + (f"<table><tr><th>Documento</th><th>Parole</th><th>Motivo</th><th>Frase</th></tr>{doubts}</table>"
                   if doubts else "<p>Nulla.</p>"))
    return _head(f"Caso {case.id}", message, bad) + f"""
<p><a href="/">tutti i casi</a> · {len(rows)} documenti, {used} usati nel caso</p>
<h2>Documenti</h2>{table}
<form class="edit" method="post" action="/upload?id={quote(case.id)}" enctype="multipart/form-data">
<label class="wide">Aggiungi un documento (PDF, immagine, DICOM)<input type="file" name="file" required></label>
<div class="wide"><button type="submit">Aggiungi</button></div></form>
<form class="edit" method="post" action="/read"><input type="hidden" name="id" value="{e(case.id)}">
<label class="wide">Nota del medico (facoltativa)<textarea name="report_text"></textarea></label>
<div class="wide"><button type="submit">Leggi il caso con i documenti usati</button>
<small>con i lettori locali configurati su questo computer</small></div></form>
{read}</main></body></html>"""


def read_case(case: CaseFolder, report_text: str) -> str:
    """The case read with the decisions of the review, the reading kept in the case's folder."""
    from melampo.data.ingestion import ClinicalIngestionPipeline

    context = ClinicalIngestionPipeline().from_payload(case.payload(report_text))
    reading = dict((context.provenance or {}).get("report_reading") or {})
    case.save_reading(reading)
    return f"letto: {len(reading.get('texts', []))} testi, {len(case.questions())} domande, {len(case.doubts())} dubbi"


class Handler(BaseHTTPRequestHandler):
    root: Path = CASES

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

    def _case(self, case_id: str) -> CaseFolder | None:
        return next((c for c in CaseFolder.all(self.root) if c.id == case_id), None)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/":
            self._send(home(self.root))
            return
        case = self._case((parse_qs(u.query).get("id") or [""])[0]) if u.path == "/case" else None
        if case is None:
            self._send("non trovato", 404)
            return
        self._send(case_page(case))

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path == "/upload":
            case = self._case((parse_qs(u.query).get("id") or [""])[0])
            n = int(self.headers.get("Content-Length") or 0)
            if case is None or n > MAX_UPLOAD:
                self._send("caso sconosciuto o file oltre 200 MB", 400)
                return
            _, data, filename = multipart(self.headers.get("Content-Type", ""), self.rfile.read(n))
            try:
                case.add(filename, data)
                self._send(case_page(case, f"aggiunto: {filename}"))
            except ValueError as error:
                self._send(case_page(case, f"non aggiunto: {error}", True), 400)
            return
        f = self._form()
        if u.path == "/new":
            try:
                case = CaseFolder.create(f.get("id", ""), f, self.root)
                self._send(case_page(case, "caso aperto"))
            except ValueError as error:
                self._send(home(self.root, f"non aperto: {error}", True), 400)
            return
        case = self._case(f.get("id", ""))
        if case is None:
            self._send("caso sconosciuto", 404)
            return
        try:
            if u.path == "/decide":
                row = case.decide(f.get("doc", ""), f.get("answer", ""), f.get("by", ""))
                self._send(case_page(case, f"{row.name}: {DECISION_LABEL[row.decision]}"))
            elif u.path == "/answer":
                case.answer(f.get("q", ""), f.get("code", ""), f.get("by", ""), f.get("text", ""))
                self._send(case_page(case, "risposta salvata"))
            elif u.path == "/card-answer":
                case.answer_card(f.get("q", ""), f.get("code", ""), f.get("by", ""), f.get("text", ""))
                self._send(case_page(case, "risposta salvata: serve la conferma di una seconda persona"))
            elif u.path == "/card-confirm":
                case.confirm_card(f.get("q", ""), f.get("by", ""), f.get("role", "confirmed"))
                self._send(case_page(case, "conferma salvata"))
            elif u.path == "/read":
                self._send(case_page(case, read_case(case, f.get("report_text", ""))))
            else:
                self._send("non trovato", 404)
        except PermissionError as error:
            self._send(case_page(case, str(error), True), 403)
        except (ValueError, RuntimeError) as error:
            self._send(case_page(case, f"non salvato: {error}", True), 400)

    def log_message(self, *args) -> None:  # quiet
        return


def serve(port: int, root: Path) -> None:
    Handler.root = root
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"maschera di revisione: http://localhost:{port}  (Ctrl+C per chiudere)")
    server.serve_forever()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--cases", default=str(CASES))
    args = ap.parse_args(argv)
    serve(args.port, Path(args.cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
