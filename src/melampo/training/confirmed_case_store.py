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

    def delete(self, case_id: str) -> None:
        """Purge a confirmed case's record once its training use is served -- an event, not a rewrite of the file.

        EncryptedJsonlStore is append-only, the same primitive already
        used for the UMLS cache and (as an event log) NexusCandidateStore
        -- deletion here follows the same tombstone principle rather than
        reading and rewriting the whole file: append a marker, and have
        load() honour the most recent event for a given case_id, in file
        order, exactly as NexusCandidateStore's own event replay already
        does.
        """
        self._store.append({"case_id": case_id, "_event": "deleted"})

    def load(self):
        """Every confirmed case still retained -- the latest event per case_id, deletions honoured, in file order."""
        latest_by_case: dict[str, dict[str, Any] | None] = {}
        for record in self._store.load():
            case_id = record.get("case_id")
            if case_id is None:
                continue
            latest_by_case[case_id] = None if record.get("_event") == "deleted" else record
        for record in latest_by_case.values():
            if record is not None:
                yield record

    def __len__(self) -> int:
        return sum(1 for _ in self.load())
