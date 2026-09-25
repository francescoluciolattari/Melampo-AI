"""Regression tests for two verified ingestion defects that made a laboratory
report unusable before any laboratory-specific work could start:

1. PHI redaction treated any long run of digits and separators as a phone
   number, erasing reference ranges ("12.0 - 16.0") and even values glued
   to the next column ("11.8   10^3/uL" -> "[REDACTED_PHONE]^3/uL").
2. A document's text was rebuilt by joining its chunks, which overlap by
   design, so every overlap appeared twice -- a duplicated laboratory line
   reads as a repeated measurement.

Both fixed in data/document_processing.py; the second also in the two
callers that re-joined chunks (case_attachments.py, dicom_handler.py -- the
latter covered in test_dicom_handler.py).
"""

from collections import Counter

import pytest

from melampo.data.case_attachments import CaseAttachment, process_case_attachments
from melampo.data.document_processing import ClinicalDocumentProcessor

# The exact report that exposed defect 1, plus the range/series shapes a
# laboratory report prints (spaced, unspaced, per-mm3 counts, serial values).
LABORATORY_REPORT = """EMOCROMO
Emoglobina            13.5   g/dL      12.0 - 16.0
Leucociti             11.8   10^3/uL    4.0 - 10.0   H
Piastrine           250000   /mm3    150000 - 450000
Globuli bianchi       7500   /mm3      4000-10000
Creatinina             2.1   mg/dL     0.50 - 1.20   H
PCR                   48     mg/L       0 - 5        H
Sodio 138 mmol/L 135-145
Leucociti 11.8 10.5 9.8
Glicemia 312 250 198 (serie)
Glicemia 95 110 102 98 (serie)"""


@pytest.fixture
def processor():
    return ClinicalDocumentProcessor()


def test_a_laboratory_report_passes_redaction_unchanged(processor):
    redacted, kinds = processor.redact_text(LABORATORY_REPORT)
    assert redacted == LABORATORY_REPORT
    assert kinds == []


@pytest.mark.parametrize(
    "value_text",
    [
        "12.0 - 16.0",
        "11.8   10^3/uL",
        "0.50 - 1.20",
        "4000-10000",
        "150000 - 450000",
        "135-145",
        "11.8 10.5 9.8",
        "312 250 198",
    ],
)
def test_ranges_and_value_series_are_never_taken_for_phone_numbers(processor, value_text):
    assert processor.redact_text(f"Valore {value_text} fine")[0] == f"Valore {value_text} fine"


@pytest.mark.parametrize(
    "phone",
    [
        "+39 06 1234 5678",
        "+39 333 1234567",
        "0039 06 12345678",
        "06 12345678",
        "(06) 1234567",
        "06-12345678",
        "06.1234.5678",
        "0571 123456",
        "333 1234567",
        "3331234567",
        "333.123.4567",
        "333 123 4567",
    ],
)
def test_italian_phone_numbers_are_still_redacted(processor, phone):
    """No phone word in front on purpose: these must be recognised by shape alone."""
    redacted, kinds = processor.redact_text(f"Chiamare il {phone} in orario d'ufficio")
    assert redacted == "Chiamare il [REDACTED_PHONE] in orario d'ufficio"
    assert kinds == ["phone"]


@pytest.mark.parametrize("prefix", ["Tel.", "Tel:", "Telefono", "Cell.", "Fax"])
def test_a_number_after_an_explicit_phone_word_is_redacted_even_without_a_0_or_3_prefix(processor, prefix):
    redacted, _ = processor.redact_text(f"{prefix} 800 123456")
    assert redacted == f"{prefix} [REDACTED_PHONE]"


@pytest.mark.parametrize("date", ["12/09/2026", "12.09.2026", "03.10.2026", "12-09-2026"])
def test_dates_are_redacted_as_dates_with_any_single_separator(processor, date):
    redacted, kinds = processor.redact_text(f"Data prelievo: {date}")
    assert redacted == "Data prelievo: [REDACTED_DATE]"
    assert kinds == ["date"]


def test_a_decimal_range_is_not_mistaken_for_a_date(processor):
    """The date rule requires the same separator on both sides: a mixed form
    would match "4.0-10" inside "4.0-10.0"."""
    assert processor.redact_text("Leucociti 4.0-10.0")[0] == "Leucociti 4.0-10.0"


def test_email_redaction_is_unchanged(processor):
    assert processor.redact_text("Scrivere a lab@ospedale.it")[0] == "Scrivere a [REDACTED_EMAIL]"


def _long_report(lines: int = 120) -> str:
    return "EMOCROMO\n" + "\n".join(f"Parametro{i:03d}  valore {i} unita" for i in range(lines))


def test_a_completed_result_carries_the_whole_redacted_text(processor):
    text = _long_report() + "\nContatto: lab@ospedale.it"
    result = processor.process_document_bytes(text.encode(), prefer_structured_parser=False)
    assert result["status"] == "completed"
    assert result["chunk_count"] > 1
    assert result["text"] == processor.redact_text(text)[0]


def test_attachment_text_is_the_whole_document_with_no_duplicated_lines(processor):
    """The exact scenario that exposed defect 2: ~3,900 characters, default
    chunking -- 12 lines used to come back twice."""
    report = _long_report()
    bundle = process_case_attachments([CaseAttachment(filename="emocromo.txt", data=report.encode())], processor=processor)
    text = bundle.attachments[0].text
    assert text == report
    duplicated = [line for line, count in Counter(text.splitlines()).items() if count > 1]
    assert duplicated == []


def test_chunks_for_memory_still_overlap_as_designed(processor):
    """The fix only stops re-joining chunks into document text; retrieval chunks keep their overlap."""
    result = processor.process_document_bytes(_long_report().encode(), prefer_structured_parser=False)
    first, second = result["documents"][0], result["documents"][1]
    assert first["metadata"]["char_end"] > second["metadata"]["char_start"]
