import json
import time

import pytest

from melampo.connectors.pmc_case_reports import (
    FETCHABLE_LICENSES,
    LICENSE_COMMERCIAL,
    LICENSE_NONCOMMERCIAL,
    LICENSE_OTHER,
    FetchConfig,
    PmcCaseReportFetcher,
    RateLimiter,
    _extract_diagnosis,
    _parse_article_xml,
)

CASE_XML_TEMPLATE = """<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>{title}</article-title></title-group>
      <permissions>
        <license xlink:href="{license_url}" xmlns:xlink="http://www.w3.org/1999/xlink">
          <license-p>{license_text}</license-p>
        </license>
      </permissions>
    </article-meta>
  </front>
  <body>
    <p>{presentation}</p>
    <p>Final diagnosis: {diagnosis}. The patient was treated accordingly.</p>
  </body>
</article>"""

PRESENTATION = (
    "A 59-year-old woman presented with progressive dyspnea over three weeks, "
    "bilateral pleural effusion on chest radiograph, and marked hepatomegaly on examination."
)


def _xml(title="Case report", license_url="https://creativecommons.org/licenses/by/4.0/",
         license_text="CC BY", presentation=PRESENTATION, diagnosis="cardiac amyloidosis"):
    return CASE_XML_TEMPLATE.format(
        title=title, license_url=license_url, license_text=license_text,
        presentation=presentation, diagnosis=diagnosis,
    )


# --------------------------------------------------------------------------
# License is a hard partition
# --------------------------------------------------------------------------


def test_a_noncommercial_config_cannot_be_constructed_as_commercial_by_accident():
    config = FetchConfig(email="ops@example.org", license_group=LICENSE_NONCOMMERCIAL)
    assert config.license_group == LICENSE_NONCOMMERCIAL


def test_oa_other_is_never_a_fetchable_license():
    """Missing or unreadable license cannot be classified as anything."""
    with pytest.raises(ValueError, match="oa_other"):
        FetchConfig(email="ops@example.org", license_group=LICENSE_OTHER)


def test_an_unrecognised_license_group_is_rejected():
    with pytest.raises(ValueError):
        FetchConfig(email="ops@example.org", license_group="oa_whatever")


def test_only_two_license_groups_are_ever_fetchable():
    assert FETCHABLE_LICENSES == {LICENSE_COMMERCIAL, LICENSE_NONCOMMERCIAL}


def test_email_is_required_by_ncbi_policy():
    with pytest.raises(ValueError):
        FetchConfig(email="")
    with pytest.raises(ValueError):
        FetchConfig(email="not-an-email")


def test_an_article_under_the_wrong_license_is_skipped_not_kept():
    config = FetchConfig(email="ops@example.org", license_group=LICENSE_COMMERCIAL)

    def transport(url, params):
        return _xml(license_url="https://creativecommons.org/licenses/by-nc/4.0/", license_text="CC BY-NC")

    fetcher = PmcCaseReportFetcher(config=config, transport=transport)
    fetcher.fetch_articles(["PMC1"])

    assert fetcher.fetched == []
    assert fetcher.skipped[0] == ("PMC1", "license_mismatch:oa_noncomm")


def test_a_matching_license_is_kept():
    config = FetchConfig(email="ops@example.org", license_group=LICENSE_COMMERCIAL)
    fetcher = PmcCaseReportFetcher(config=config, transport=lambda url, params: _xml())
    fetcher.fetch_articles(["PMC1"])

    assert len(fetcher.fetched) == 1
    assert fetcher.fetched[0].license_group == LICENSE_COMMERCIAL


def test_license_is_read_per_article_not_assumed_from_the_search_filter():
    """A search result is a query-time snapshot; the per-article record is authoritative."""
    config = FetchConfig(email="ops@example.org", license_group=LICENSE_NONCOMMERCIAL)
    fetcher = PmcCaseReportFetcher(
        config=config,
        transport=lambda url, params: _xml(license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/"),
    )
    fetcher.fetch_articles(["PMC1"])
    assert fetcher.fetched[0].license_group == LICENSE_NONCOMMERCIAL


# --------------------------------------------------------------------------
# License extraction from real-shaped XML
# --------------------------------------------------------------------------


def test_cc_by_is_read_as_commercial():
    article = _parse_article_xml("PMC1", _xml(license_url="https://creativecommons.org/licenses/by/4.0/"))
    assert article.license_group == LICENSE_COMMERCIAL


def test_cc_by_nc_is_read_as_noncommercial():
    article = _parse_article_xml("PMC1", _xml(license_url="https://creativecommons.org/licenses/by-nc/4.0/"))
    assert article.license_group == LICENSE_NONCOMMERCIAL


def test_no_license_element_is_read_as_other():
    xml = CASE_XML_TEMPLATE.replace(
        '''<permissions>
        <license xlink:href="{license_url}" xmlns:xlink="http://www.w3.org/1999/xlink">
          <license-p>{license_text}</license-p>
        </license>
      </permissions>''',
        "",
    ).format(title="Case report", presentation=PRESENTATION, diagnosis="cardiac amyloidosis")
    article = _parse_article_xml("PMC1", xml)
    assert article.license_group == LICENSE_OTHER


def test_malformed_xml_yields_no_article_rather_than_raising():
    assert _parse_article_xml("PMC1", "<not><valid") is None


def test_an_article_with_no_body_text_is_dropped():
    xml = """<?xml version="1.0"?>
<article>
  <front><article-meta><title-group><article-title>Empty</article-title></title-group></article-meta></front>
  <body></body>
</article>"""
    article = _parse_article_xml("PMC1", xml)
    assert article is None


# --------------------------------------------------------------------------
# Splitting into evaluation cases reuses the shared discipline
# --------------------------------------------------------------------------


def test_a_fetched_article_becomes_an_evaluation_case():
    config = FetchConfig(email="ops@example.org")
    fetcher = PmcCaseReportFetcher(config=config, transport=lambda url, params: _xml())
    fetcher.fetch_articles(["PMC1"])
    report = fetcher.to_evaluation_cases()

    assert len(report.cases) == 1
    assert "amyloidosis" not in report.cases[0].presentation.lower()
    assert "amyloidosis" in report.cases[0].documented_diagnosis.lower()
    assert report.cases[0].source == "pmc:oa_comm"


def test_leaked_presentations_are_rejected_by_the_same_rule_as_any_other_source():
    """The connector does not get a laxer check than a hand-built file would."""
    config = FetchConfig(email="ops@example.org")
    leaking_presentation = PRESENTATION + " This is a case of cardiac amyloidosis."
    fetcher = PmcCaseReportFetcher(
        config=config, transport=lambda url, params: _xml(presentation=leaking_presentation)
    )
    fetcher.fetch_articles(["PMC1"])
    report = fetcher.to_evaluation_cases()
    assert report.cases == []
    assert any(reason == "presentation_contains_the_diagnosis" for _, reason in report.rejected)


def test_a_diagnosis_prefix_is_stripped():
    assert _extract_diagnosis("Final diagnosis: cardiac amyloidosis. Further details.") == "cardiac amyloidosis"


def test_no_diagnosis_section_is_skipped_with_a_reason():
    config = FetchConfig(email="ops@example.org")
    xml = _xml().replace("Final diagnosis: cardiac amyloidosis. The patient was treated accordingly.", "")
    fetcher = PmcCaseReportFetcher(config=config, transport=lambda url, params: xml)
    fetcher.fetch_articles(["PMC1"])
    fetcher.to_evaluation_cases()
    assert any(reason == "no_diagnosis_section_found" for _, reason in fetcher.skipped)


def test_a_fetch_error_is_recorded_per_article_rather_than_aborting_the_run():
    config = FetchConfig(email="ops@example.org")

    def failing_transport(url, params):
        raise TimeoutError("simulated network failure")

    fetcher = PmcCaseReportFetcher(config=config, transport=failing_transport)
    fetcher.fetch_articles(["PMC1", "PMC2"])
    assert fetcher.fetched == []
    assert len(fetcher.skipped) == 2
    assert all(reason.startswith("fetch_error") for _, reason in fetcher.skipped)


def test_the_report_summarises_fetched_and_skipped():
    config = FetchConfig(email="ops@example.org", license_group=LICENSE_COMMERCIAL)
    fetcher = PmcCaseReportFetcher(
        config=config,
        transport=lambda url, params: _xml(license_url="https://creativecommons.org/licenses/by-nc/4.0/"),
    )
    fetcher.fetch_articles(["PMC1"])
    summary = fetcher.report()
    assert summary["fetched"] == 0
    assert summary["skipped"] == 1
    assert summary["skipped_reasons"]["license_mismatch"] == 1


# --------------------------------------------------------------------------
# Rate limiting matches the published contract
# --------------------------------------------------------------------------


def test_the_limiter_enforces_the_minimum_interval():
    limiter = RateLimiter(requests_per_second=10.0)
    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.2 - 0.02  # two intervals of 0.1s, small tolerance for scheduling


def test_a_zero_rate_never_blocks():
    limiter = RateLimiter(requests_per_second=0.0)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    assert time.monotonic() - start < 0.05


def test_keyed_and_unkeyed_configs_report_a_positive_rate():
    keyed = FetchConfig(email="ops@example.org", api_key="k")
    unkeyed = FetchConfig(email="ops@example.org")
    assert keyed.requests_per_second > 0
    assert unkeyed.requests_per_second > 0


# --------------------------------------------------------------------------
# Search query construction
# --------------------------------------------------------------------------


def test_search_restricts_to_case_reports_and_open_access():
    config = FetchConfig(email="ops@example.org")
    seen = {}

    def transport(url, params):
        seen.update(params)
        return json.dumps({"esearchresult": {"idlist": ["123", "456"]}})

    fetcher = PmcCaseReportFetcher(config=config, transport=transport)
    ids = fetcher.search_case_reports(["cardiac amyloidosis"])

    assert ids == ["PMC123", "PMC456"]
    assert "case reports[Publication Type]" in seen["term"]
    assert "open access[filter]" in seen["term"]
    assert "cardiac amyloidosis" in seen["term"]
    assert seen["email"] == "ops@example.org"
