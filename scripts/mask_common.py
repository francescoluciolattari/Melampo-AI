"""What Melampo's masks share (27 November 2026): one look and one reader of an uploaded form.

The masks (``sources_mask.py``, ``review_mask.py``) are pages on this computer only (127.0.0.1), with nothing to
install beyond Python.
"""

from __future__ import annotations

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
.lock{color:var(--bad);font-weight:600}.given{color:var(--muted);font-size:12px}.doubt{color:var(--warn)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
"""


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


__all__ = ["STYLE", "multipart"]
