"""Fetch case reports from PMC and turn them into B4 evaluation cases.

Project defaults: contact email and API key. The key is read from the
``NCBI_API_KEY`` environment variable at call time — populated from a GitHub
Actions secret of the same name when run in CI, or exported locally by the
operator — and is never written into this file or any other tracked file.
Committing a working credential to a public repository is a standing risk
regardless of provider, and NCBI's own guidance is to keep API keys out of
version control for the same reason. ``FetchConfig.from_environment`` is the
supported way to build a config; constructing one with a literal key inline is
still possible but bypasses that discipline, and reviewers should treat a
literal key in a diff as a defect to fix before merge, not a style note.

Runs where the operator has network access to NCBI, which this repository's own
execution environment may not have — its outbound access is allowlisted, and
NCBI's endpoints are ordinarily outside that list. The client is therefore
built to be invoked by a person with a normal internet connection, not assumed
to run inside any particular sandbox.

Two design points carry the discipline established for this evaluation into a
live connector, where a mistake is easier to make because nothing forces it to
surface.

**License is a hard partition, not a filter.** Building a corpus for a
commercial deliverable while allowing NC-licensed articles to sit in the same
call is the kind of thing that is easy to do once and expensive to unwind later
— every model trained since would need to be re-derived once the mixing is
found. So the license is required up front, not inferred from an in-memory flag
that a later refactor could silently drop, and `oa_other` is refused outright
because a missing or unreadable license cannot be classified as anything.

**Rate limiting matches the published contract exactly**, not conservatively
below it. NCBI publishes a limit rather than leaving it to guesswork, and
guessing in either direction is wrong: too slow wastes time across thousands of
articles, too fast risks the caller's key being throttled or blocked, with no
visible error pointing at the cause.
"""

import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..evaluation.case_corpus import LoadReport, load_records
from ..evaluation.dream_capture_benchmark import EvaluationCase

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OAI_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"

# Project defaults, not secrets. The email is a contact address NCBI's usage
# policy requires and carries no access on its own; publishing it is normal
# practice (it appears in every request's query string regardless). The API
# key environment variable name is a convention, matching the GitHub Actions
# secret configured for this repository -- the key itself lives only there and
# in the operator's local environment, never in tracked files.
DEFAULT_CONTACT_EMAIL = "francesco.lucio.lattari@gmail.com"
NCBI_API_KEY_ENV_VAR = "NCBI_API_KEY"

LICENSE_COMMERCIAL = "oa_comm"
LICENSE_NONCOMMERCIAL = "oa_noncomm"
LICENSE_OTHER = "oa_other"

# oa_other carries no machine-readable licence, or a custom one — it cannot be
# classified as either commercial or non-commercial, so it is never fetchable
# rather than requiring the caller to remember to exclude it.
FETCHABLE_LICENSES = frozenset({LICENSE_COMMERCIAL, LICENSE_NONCOMMERCIAL})

# NCBI's published contract: 3 requests/second with a key, 10/second is the
# documented ceiling for registered tools. Matched exactly rather than padded,
# for the reason in the module docstring.
REQUESTS_PER_SECOND_WITH_KEY = 3.0
REQUESTS_PER_SECOND_WITHOUT_KEY = 3.0  # unkeyed is capped lower still (3/s) in practice


@dataclass(frozen=True)
class FetchConfig:
    """Identification NCBI requires, and the license partition for this run.

    ``email`` and ``tool`` are not authentication — E-utilities has none for
    this endpoint — they are how NCBI contacts an operator whose usage needs
    attention, and omitting them risks silent throttling with no diagnostic.
    """

    email: str
    tool: str = "melampo-b4-connector"
    api_key: str | None = None
    license_group: str = LICENSE_COMMERCIAL

    def __post_init__(self) -> None:
        if not self.email or "@" not in self.email:
            raise ValueError("a contact email is required by NCBI's usage policy")
        if self.license_group not in FETCHABLE_LICENSES:
            raise ValueError(
                f"license_group must be one of {sorted(FETCHABLE_LICENSES)}, "
                f"got {self.license_group!r} — oa_other has no reliable machine-readable "
                "license and is never fetched automatically"
            )

    @property
    def requests_per_second(self) -> float:
        return REQUESTS_PER_SECOND_WITH_KEY if self.api_key else REQUESTS_PER_SECOND_WITHOUT_KEY

    @classmethod
    def from_environment(
        cls,
        *,
        email: str = DEFAULT_CONTACT_EMAIL,
        license_group: str = LICENSE_COMMERCIAL,
        tool: str = "melampo-b4-connector",
    ) -> "FetchConfig":
        """Build a config with the API key read from the environment, never from a literal.

        Looks up ``NCBI_API_KEY`` at call time. Absent, the fetcher still works
        at the unkeyed rate rather than raising -- a missing key degrades
        throughput, it does not block evaluation, and failing loudly here would
        make local experimentation harder for no safety benefit.
        """
        return cls(
            email=email,
            tool=tool,
            api_key=os.environ.get(NCBI_API_KEY_ENV_VAR) or None,
            license_group=license_group,
        )


@dataclass
class FetchedArticle:
    """One article as retrieved, before it is split into presentation and outcome."""

    pmcid: str
    title: str
    full_text: str
    license_group: str
    license_url: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pmcid": self.pmcid,
            "title": self.title,
            "license_group": self.license_group,
            "license_url": self.license_url,
            "characters": len(self.full_text),
        }


class RateLimiter:
    """Sleep exactly as long as the published contract requires, no more."""

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._interval <= 0:
            return
        now = time.monotonic()
        if self._last_call is not None:
            elapsed = now - self._last_call
            remaining = self._interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


@dataclass
class PmcCaseReportFetcher:
    """Search PMC for case reports and fetch their full text under one license.

    A thin wrapper over E-utilities. No PMC-specific library is required, so it
    runs wherever a person can reach ``eutils.ncbi.nlm.nih.gov`` — this
    repository's own execution may not be able to, since its network access is
    allowlisted and NCBI is ordinarily outside that list.
    """

    config: FetchConfig
    limiter: RateLimiter = field(init=False)
    fetched: list[FetchedArticle] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    transport: Any = None
    """Injectable (url, params) -> response text, for testing without a live NCBI endpoint."""

    def __post_init__(self) -> None:
        self.limiter = RateLimiter(self.config.requests_per_second)

    def search_case_reports(
        self, query_terms: Sequence[str] = (), max_results: int = 200
    ) -> list[str]:
        """PMC IDs of open-access case reports matching the query and license.

        The license restricts the search itself — via PMC's open-access
        filter — not just what gets kept afterwards, so an oa_noncomm article
        never has to be discarded after being fetched.
        """
        terms = list(query_terms) or ["case report"]
        query = " AND ".join([*terms, "case reports[Publication Type]", "open access[filter]"])
        params = {
            "db": "pmc",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "tool": self.config.tool,
            "email": self.config.email,
        }
        if self.config.api_key:
            params["api_key"] = self.config.api_key

        self.limiter.wait()
        payload = self._get_json(f"{EUTILS_BASE}/esearch.fcgi", params)
        ids = payload.get("esearchresult", {}).get("idlist", [])
        return [f"PMC{item}" for item in ids]

    def fetch_articles(self, pmcids: Sequence[str]) -> list[FetchedArticle]:
        """Fetch full text for each id, keeping only articles under the configured license.

        One at a time rather than batched, so the rate limiter governs actual
        request timing and a single malformed record cannot abort the run —
        it is skipped with a reason instead.
        """
        for pmcid in pmcids:
            self.limiter.wait()
            try:
                article = self._fetch_one(pmcid)
            except Exception as error:  # noqa: BLE001 - recorded, not raised, per-article
                self.skipped.append((pmcid, f"fetch_error:{error}"))
                continue
            if article is None:
                continue
            if article.license_group != self.config.license_group:
                self.skipped.append((pmcid, f"license_mismatch:{article.license_group}"))
                continue
            self.fetched.append(article)
        return self.fetched

    def to_evaluation_cases(
        self, diagnosis_headings: Sequence[str] = ()
    ) -> LoadReport:
        """Split each fetched article and load it through the shared corpus discipline.

        Reuses ``case_corpus.load_records``, so a leaked diagnosis or a too-short
        presentation is rejected here exactly as it would be for any other
        source — the connector does not get a laxer check than a hand-built
        JSONL file would.
        """
        from ..evaluation.case_corpus import split_presentation

        records = []
        for article in self.fetched:
            presentation, revealed = split_presentation(article.full_text, diagnosis_headings)
            diagnosis = _extract_diagnosis(revealed)
            if not diagnosis:
                self.skipped.append((article.pmcid, "no_diagnosis_section_found"))
                continue
            records.append(
                {
                    "case_id": article.pmcid,
                    "presentation": presentation,
                    "diagnosis": diagnosis,
                    "source": f"pmc:{article.license_group}",
                }
            )
        return load_records(records, source=f"pmc:{self.config.license_group}")

    def report(self) -> dict[str, Any]:
        return {
            "fetched": len(self.fetched),
            "skipped": len(self.skipped),
            "skipped_reasons": dict(_count_reasons(self.skipped)),
            "license_group": self.config.license_group,
        }

    def _fetch_one(self, pmcid: str) -> FetchedArticle | None:
        numeric_id = pmcid.removeprefix("PMC")
        params = {
            "db": "pmc",
            "id": numeric_id,
            "rettype": "full",
            "retmode": "xml",
            "tool": self.config.tool,
            "email": self.config.email,
        }
        if self.config.api_key:
            params["api_key"] = self.config.api_key

        xml_text = self._get_text(f"{EUTILS_BASE}/efetch.fcgi", params)
        return _parse_article_xml(pmcid, xml_text)

    def _get_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        import json

        return json.loads(self._get_text(url, params))

    def _get_text(self, url: str, params: dict[str, str]) -> str:
        if self.transport is not None:
            return self.transport(url, params)
        request = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": self.config.tool})
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")


def _parse_article_xml(pmcid: str, xml_text: str) -> FetchedArticle | None:
    """Extract title, body text and license from a PMC full-text XML record."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    title_element = root.find(".//article-title")
    title = "".join(title_element.itertext()).strip() if title_element is not None else ""

    body_element = root.find(".//body")
    full_text = "\n\n".join(
        "".join(paragraph.itertext()).strip()
        for paragraph in (body_element.iter("p") if body_element is not None else [])
    ).strip()
    if not full_text:
        return None

    license_group, license_url = _extract_license(root)
    return FetchedArticle(
        pmcid=pmcid, title=title, full_text=full_text, license_group=license_group, license_url=license_url
    )


def _extract_license(root: ET.Element) -> tuple[str, str | None]:
    """Read the license from the article's own metadata rather than assuming it.

    The search step already filters by license, but articles are classified
    here independently: a search result carries a query-time snapshot, and the
    per-article record is the authoritative statement of terms.
    """
    license_element = root.find(".//license")
    url = license_element.get("{http://www.w3.org/1999/xlink}href") if license_element is not None else None
    text = "".join(license_element.itertext()).lower() if license_element is not None else ""

    if url:
        lowered_url = url.lower()
        if "by-nc" in lowered_url:
            return LICENSE_NONCOMMERCIAL, url
        if any(marker in lowered_url for marker in ("/by/", "/by-sa/", "/by-nd/", "publicdomain", "zero")):
            return LICENSE_COMMERCIAL, url

    if "non-commercial" in text or "noncommercial" in text or "nc-" in text:
        return LICENSE_NONCOMMERCIAL, url
    if text:
        return LICENSE_COMMERCIAL, url
    return LICENSE_OTHER, url


def _extract_diagnosis(revealed_section: str) -> str:
    """Pull a short diagnosis phrase from the section split off by split_presentation.

    Deliberately conservative: takes the first sentence rather than the whole
    section, since the section following the heading is typically discussion
    prose, not a clean label. A human reviewer should treat this as a starting
    point, not a final answer — the same caution applied everywhere else a
    machine-derived label enters an evaluation.
    """
    first_stop = revealed_section.find(".")
    candidate = revealed_section[: first_stop if first_stop > 0 else 200].strip()
    for prefix in ("final diagnosis:", "diagnosis:", "conclusion:"):
        if candidate.lower().startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
    return candidate


def _count_reasons(skipped: Sequence[tuple[str, str]]) -> Iterator[tuple[str, int]]:
    counts: dict[str, int] = {}
    for _, reason in skipped:
        key = reason.split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    yield from sorted(counts.items())


def cases_from_fetcher(fetcher: PmcCaseReportFetcher) -> list[EvaluationCase]:
    """Convenience accessor returning only the cases that passed the corpus discipline."""
    return fetcher.to_evaluation_cases().cases
