"""Encrypted-at-rest local storage, for data a license requires be kept secure.

UMLS's own license agreement makes the licensee responsible for protecting
the data it grants access to -- unlike HPO or PubMed, which are public
domain, UMLS content sits under a real contractual obligation. Requested
directly: local storage that is genuinely encrypted, tied to a project
secret (`DB_PASSWORD`), not a database server this project has no other use
for.

**Why this is a file, not a database server.** Every other persistent store
in this project (`graph_store`, `term_history`, `vector_memory`) is a local
file, dependency-free beyond the standard library, chosen because a
clinical-graph project with no existing database infrastructure gains
nothing from standing one up for this alone -- connection pooling,
migrations, and an always-running server process are real operational cost
for a single encrypted file's worth of benefit. `cryptography`'s Fernet
(symmetric, authenticated encryption -- AES-128 in CBC mode with an HMAC,
not just obfuscation) is a well-audited, minimal dependency that keeps this
project's "no server to run" posture intact while genuinely satisfying
"encrypted at rest".

**Key derivation, not a raw password as the key.** `DB_PASSWORD` is a
human-chosen secret, not itself a cryptographic key -- using it directly
would make the encryption only as strong as whatever the person typed.
PBKDF2HMAC with a per-store random salt (saved alongside the ciphertext,
since a salt is not a secret) derives an actual key from it, the standard
construction for turning a password into something safe to encrypt with.
"""

import base64
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

SALT_FILENAME_SUFFIX = ".salt"
KDF_ITERATIONS = 600_000  # OWASP's 2023 minimum recommendation for PBKDF2-HMAC-SHA256


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=KDF_ITERATIONS)
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


@dataclass
class EncryptedJsonlStore:
    """Append-only JSONL storage, encrypted at rest with a password-derived key.

    Same shape and discipline as `LearnedEdgeStore` and `TermHistoryStore` --
    append, load, missing file means empty -- with every line's JSON payload
    encrypted individually before it touches disk, rather than encrypting
    the whole file as one blob. Line-level encryption keeps the append-only
    property genuinely append-only: adding a record is still one write, not
    a decrypt-modify-re-encrypt of the entire store.
    """

    path: Path
    password: str

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._salt = self._load_or_create_salt()
        self._fernet = Fernet(_derive_key(self.password, self._salt))

    @property
    def salt_path(self) -> Path:
        return self.path.with_name(self.path.name + SALT_FILENAME_SUFFIX)

    def _load_or_create_salt(self) -> bytes:
        if self.salt_path.exists():
            return self.salt_path.read_bytes()
        salt = os.urandom(16)
        self.salt_path.parent.mkdir(parents=True, exist_ok=True)
        self.salt_path.write_bytes(salt)
        return salt

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ciphertext = self._fernet.encrypt(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        with self.path.open("ab") as handle:
            handle.write(ciphertext + b"\n")

    def append_many(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            for record in records:
                ciphertext = self._fernet.encrypt(json.dumps(record, ensure_ascii=False).encode("utf-8"))
                handle.write(ciphertext + b"\n")
        return len(records)

    def load(self) -> Iterator[dict[str, Any]]:
        """Decrypt and yield every stored record.

        Raises `WrongPasswordError` immediately on the first line that fails
        to decrypt, rather than silently skipping it: a decryption failure
        here means either a wrong password or corrupted ciphertext, and
        both are situations the caller must be told about explicitly, not
        ones where "keep going and report what parsed" is the safe default
        the other stores in this project use for merely malformed JSON.
        """
        if not self.path.exists():
            return
        for line_number, line in enumerate(self.path.read_bytes().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                plaintext = self._fernet.decrypt(line)
            except InvalidToken as error:
                raise WrongPasswordError(
                    f"could not decrypt {self.path} at line {line_number}: wrong DB_PASSWORD, "
                    "or the file is corrupted"
                ) from error
            yield json.loads(plaintext.decode("utf-8"))

    def __len__(self) -> int:
        return sum(1 for _ in self.load())


class WrongPasswordError(Exception):
    """Raised when a stored record cannot be decrypted with the given password.

    A distinct exception type rather than letting `InvalidToken` propagate
    directly: a caller catching this should be able to do so without
    importing `cryptography` themselves, and the message already explains
    what to check.
    """
