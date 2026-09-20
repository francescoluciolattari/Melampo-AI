"""Where a confirmed case's data goes once training has used it -- kept, not deleted.

Corrected directly after an earlier design step assumed deletion: the raw
record is preserved, not discarded, protected two ways rather than
relying on either alone.

**Encrypted at rest**, reusing `memory/encrypted_store.py`'s
`EncryptedJsonlStore` exactly as built for UMLS content -- the same
`DB_PASSWORD` secret, the same PBKDF2-derived key, the same append-only,
line-level Fernet encryption. Not a new mechanism invented for this: the
project already has one, chosen for the same reason (no database server
this project has any other use for), and reusing it is one fewer thing to
audit.

**Anonymised before it is ever written**, on top of encryption, not
instead of it. A name, a surname, or a national identifier (codice
fiscale, or any similarly unique ID) is replaced with an HMAC-SHA256 of
its value, keyed by `DB_PASSWORD` -- not a bare hash: a bare SHA-256 of a
common name is reversible by dictionary lookup regardless of how well the
file around it is encrypted, since names have far less entropy than a
cryptographic key. Keying the HMAC with the same secret that protects the
file means reversing an anonymised name requires the same secret an
attacker would already need to decrypt the record it sits inside --
anonymisation adds a second, independent barrier rather than a
false-feeling extra step defeated by the same compromise that breaks the
first.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memory.encrypted_store import EncryptedJsonlStore

DEFAULT_CONFIRMED_CASE_STORE_PATH = "data/confirmed_cases.jsonl"

# Extend this list, do not silently rely on catching everything by shape --
# an identifying field this list misses is a field that reaches disk
# un-anonymised, and that failure should be visible in a code review of
# this constant, not discovered later.
IDENTIFYING_FIELDS = (
    "name",
    "nome",
    "surname",
    "cognome",
    "full_name",
    "patient_name",
    "codice_fiscale",
    "tax_id",
    "national_id",
    "fiscal_code",
)


def _anonymize_value(value: str, password: str) -> str:
    return hmac.new(password.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def anonymize_identifying_fields(data: dict[str, Any], password: str) -> dict[str, Any]:
    """A shallow copy of `data` with every IDENTIFYING_FIELDS key HMAC-hashed, recursing into nested dicts.

    Case data is nested (case_context, demographics, metadata) rather than
    flat, so this walks the whole structure rather than checking only the
    top level -- a name buried inside `case_context.demographics.name`
    must be caught exactly as one sitting at the top.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = anonymize_identifying_fields(value, password)
        elif key in IDENTIFYING_FIELDS and isinstance(value, str) and value:
            result[key] = _anonymize_value(value, password)
        else:
            result[key] = value
    return result


@dataclass
class ConfirmedCaseStore:
    """Persists a confirmed case's record, encrypted and anonymised, once its training use is served."""

    password: str
    path: Path = Path(DEFAULT_CONFIRMED_CASE_STORE_PATH)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._store = EncryptedJsonlStore(path=self.path, password=self.password)

    def persist(self, record: dict[str, Any]) -> None:
        """Anonymise identifying fields, then encrypt and append -- both steps, always, never one without the other."""
        self._store.append(anonymize_identifying_fields(record, self.password))

    def load(self):
        yield from self._store.load()

    def __len__(self) -> int:
        return len(self._store)
