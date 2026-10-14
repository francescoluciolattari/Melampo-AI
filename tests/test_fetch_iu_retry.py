"""The IU X-ray archive is retried when Open-i answers with an HTML page instead of the file."""

import gzip
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fetch_public_reports as fp  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_an_html_answer_is_retried_then_the_archive_is_taken():
    answers = [b"<!DOCTYPE html><html>maintenance</html>", gzip.compress(b"tar")]
    got = fp.fetch_iu_archive(opener=lambda req, timeout: _Resp(answers.pop(0)), sleep=lambda s: None)
    assert got[:2] == b"\x1f\x8b"


def test_it_stops_with_a_clear_message_after_the_last_attempt():
    with pytest.raises(SystemExit) as stop:
        fp.fetch_iu_archive(attempts=2, opener=lambda req, timeout: _Resp(b"<!html"), sleep=lambda s: None)
    assert "not a gzip file" in str(stop.value)
