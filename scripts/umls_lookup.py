"""UMLS lookups for the probes (``phrase-probe`` E2/E4, ``external-check`` E1), with a disk cache.

Built on ``melampo.connectors.umls.UmlsConnector`` (same base URI, ``apiKey`` query parameter, the
injectable ``transport`` for tests). What this adds is what a measurement over a few thousand
mentions needs and the connector does not do:

* a JSON cache on disk, so a second run (or a run on another branch) asks UTS nothing it already
  asked; the GitHub workflows keep it with ``actions/cache``;
* pacing and retries (UTS answers 429 when called too fast; 5xx happen);
* an explicit "lost" state: a lookup that failed after the retries is recorded as lost and counted
  in the report, never silently turned into "UMLS has nothing".

Four questions are asked, all read-only:

``concept(cui)``      name and semantic types (TUI) of a concept;
``definition(cui)``   one definition, preferring the sources in ``DEFINITION_SOURCES``;
``exact(string)``     concepts with a name written exactly as the string (``searchType=exact``);
``codes(cui, sab)``   the codes a source (FMA, NCI) uses for the concept, via its atoms.

The key comes from ``UMLS_API_KEY``. Without it every lookup answers ``None`` (unavailable), and the
callers switch the arm off and say so in the report.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.connectors.umls import UMLS_BASE, UmlsConfig, UmlsConnector  # noqa: E402

DEFINITION_SOURCES = ("NCI", "MSH", "CSP", "MEDLINEPLUS", "NCI_NCI-GLOSS", "HPO")
_TUI = re.compile(r"/TUI/(T\d{3})")
_RETRY = (429, 500, 502, 503, 504)


class UmlsLookup:
    """Cached, paced UTS lookups. ``transport(url, params) -> dict`` replaces the network in tests."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: Path | None = None,
        transport=None,
        min_interval: float = 0.07,
        retries: int = 5,
        sleep=time.sleep,
    ):
        key = api_key if api_key is not None else os.environ.get("UMLS_API_KEY")
        self.connector = UmlsConnector(UmlsConfig(api_key=key), transport=transport)
        self.available = bool(key)
        self.cache_path = cache_path
        self.cache: dict[str, Any] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text("utf-8"))
        self.min_interval = min_interval
        self.retries = retries
        self.sleep = sleep
        self.lost = 0
        self.asked = 0
        self._lock = threading.Lock()
        self._last = 0.0

    # -- plumbing --------------------------------------------------------------------------------

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, sort_keys=True), "utf-8")

    def _call(self, url: str, params: dict[str, str]) -> dict | None:
        """One GET with pacing and retries; ``{}`` for 404 (nothing there), ``None`` if lost."""
        for attempt in range(self.retries + 1):
            with self._lock:
                wait = self._last + self.min_interval - time.monotonic()
                if wait > 0:
                    self.sleep(wait)
                self._last = time.monotonic()
            try:
                self.asked += 1
                return self.connector._fetch(url, params)
            except HTTPError as error:
                if error.code == 404:
                    return {}
                if error.code not in _RETRY:
                    break
            except (URLError, TimeoutError, OSError, ValueError):
                pass
            self.sleep(min(30.0, 1.5 * 2**attempt))
        self.lost += 1
        return None

    def _cached(self, key: str, compute):
        if key in self.cache:
            return self.cache[key]
        if not self.available:
            return None
        value = compute()
        if value is not None:
            self.cache[key] = value
        return value

    # -- questions -------------------------------------------------------------------------------

    def concept(self, cui: str) -> dict | None:
        """``{"name": str, "types": [TUI...]}``; ``{}`` if UTS does not know the CUI."""

        def compute():
            payload = self._call(f"{UMLS_BASE}/content/current/CUI/{cui}", {})
            if payload is None:
                return None
            result = payload.get("result") or {}
            if not isinstance(result, dict) or not result:
                return {}
            return {"name": str(result.get("name", "")), "types": tuis(result.get("semanticTypes"))}

        return self._cached(f"concept:{cui}", compute)

    def definition(self, cui: str) -> str | None:
        """One definition (preferred sources first), ``""`` if there is none."""

        def compute():
            payload = self._call(f"{UMLS_BASE}/content/current/CUI/{cui}/definitions", {"pageSize": "50"})
            if payload is None:
                return None
            items = payload.get("result") or []
            if not isinstance(items, list):
                return ""
            ranked = sorted(
                (i for i in items if isinstance(i, dict) and i.get("value")),
                key=lambda i: DEFINITION_SOURCES.index(i.get("rootSource"))
                if i.get("rootSource") in DEFINITION_SOURCES
                else len(DEFINITION_SOURCES),
            )
            return _plain(ranked[0]["value"])[:400] if ranked else ""

        return self._cached(f"definition:{cui}", compute)

    def exact(self, string: str) -> list[dict] | None:
        """Concepts with a name written exactly as ``string``: ``[{"cui", "name", "types"}]``."""
        norm = " ".join(string.split()).lower()

        def compute():
            payload = self._call(
                f"{UMLS_BASE}/search/current", {"string": norm, "searchType": "exact", "pageSize": "25"}
            )
            if payload is None:
                return None
            results = (payload.get("result") or {}).get("results") or []
            found = []
            for item in results:
                if not isinstance(item, dict) or item.get("ui") in (None, "", "NONE"):
                    continue
                types = tuis(item.get("semanticTypes"))
                if not types:
                    info = self.concept(item["ui"])
                    if info is None:
                        return None
                    types = info.get("types", [])
                found.append({"cui": item["ui"], "name": str(item.get("name", "")), "types": types})
            return found

        return self._cached(f"exact:{norm}", compute)

    def codes(self, cui: str, sab: str) -> list[str] | None:
        """Codes of source ``sab`` (e.g. ``FMA``, ``NCI``) for the concept."""

        def compute():
            payload = self._call(
                f"{UMLS_BASE}/content/current/CUI/{cui}/atoms", {"sabs": sab, "pageSize": "100"}
            )
            if payload is None:
                return None
            out = []
            for atom in payload.get("result") or []:
                if not isinstance(atom, dict):
                    continue
                code = str(atom.get("code", "")).rstrip("/").rsplit("/", 1)[-1]
                if code and code not in out:
                    out.append(code)
            return out

        return self._cached(f"codes:{sab}:{cui}", compute)


def tuis(semantic_types) -> list[str]:
    """TUIs from the ``semanticTypes`` field, whatever its shape (objects with a uri, or strings)."""
    out = []
    for item in semantic_types or ():
        text = item.get("uri", "") if isinstance(item, dict) else str(item)
        match = _TUI.search(text) or re.fullmatch(r"(T\d{3})", text)
        if match and match.group(1) not in out:
            out.append(match.group(1))
    return out


def _plain(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())
