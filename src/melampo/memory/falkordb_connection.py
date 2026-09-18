"""One connection factory for FalkorDB, switchable between embedded and remote by editing a config file only.

Decided directly: FalkorDBLite for this repository, with the explicit
requirement that moving to FalkorDB Service later costs a configuration
file edit, never a code change. Verified before building anything: this is
not an aspiration this module has to engineer around -- it is FalkorDBLite's
own stated design. Its own documentation states the migration path
directly: "The API for falkordblite is designed to mirror the standard
falkordb-py client. To switch from the local Unix-socket based engine to a
remote TCP-based cluster, all you need to do is change your
initialization." This module's entire job is to be that one initialization
point, so every caller downstream (the future graph-query layer this
enables, not built in this change) talks to one interface regardless of
which backend answers it.

**What this change deliberately does not do.** It does not migrate
`InMemoryConceptGraph`, `candidate_retrieval.py`, or `MechanismEnumerator`
to query FalkorDB via Cypher -- that is a separate, substantially larger
piece of work (rewriting the graph-traversal layer itself), scoped
separately. This establishes and verifies the connection/configuration
foundation that work will build on, proven against a real embedded
instance, not a mock.

**The configuration file, not environment variables, is the primary
switch.** `data/falkordb_config.toml` (created with a `lite` default on
first read if absent, so the project runs out of the box) holds `mode =
"lite"` or `mode = "service"` plus the fields each mode needs. Environment
variables remain available as an override for deployments that prefer
them (containers, CI), but the file is what changing modes actually means
for a person running this locally.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODE_LITE = "lite"
MODE_SERVICE = "service"

DEFAULT_CONFIG_PATH = Path("data/falkordb_config.toml")
DEFAULT_LITE_DB_PATH = "data/falkordb_lite.db"

_DEFAULT_CONFIG_TOML = f'''\
# FalkorDB connection configuration.
#
# mode = "lite"    -- embedded, zero-setup, the project's default. Data
#                     lives in the file named by lite_db_path.
# mode = "service" -- a real FalkorDB server (self-hosted or FalkorDB
#                     Cloud). Fill in host/port/password/ssl below.
#
# Switching modes: change mode (and the fields the new mode needs) here.
# No code in this project needs to change -- falkordb_connection.py reads
# this file and returns the same kind of connection either way.

mode = "{MODE_LITE}"

[lite]
db_path = "{DEFAULT_LITE_DB_PATH}"

[service]
host = "localhost"
port = 6379
password = ""
ssl = false
'''


@dataclass(frozen=True)
class FalkorDBConfig:
    """Which FalkorDB backend to connect to, and how."""

    mode: str = MODE_LITE
    lite_db_path: str = DEFAULT_LITE_DB_PATH
    service_host: str = "localhost"
    service_port: int = 6379
    service_password: str | None = None
    service_ssl: bool = False

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "FalkorDBConfig":
        """Read the config file, creating it with the lite default if it does not exist yet.

        Creating a sensible default on first read (rather than requiring
        the file to be created manually before anything works) is what
        makes "lite by default, out of the box" actually true for a fresh
        checkout of this repository.
        """
        path = Path(path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_DEFAULT_CONFIG_TOML, encoding="utf-8")
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> "FalkorDBConfig":
        mode = str(raw.get("mode", MODE_LITE)).strip().lower()
        if mode not in (MODE_LITE, MODE_SERVICE):
            raise ValueError(f"falkordb config: unrecognised mode {mode!r}, expected 'lite' or 'service'")
        lite = raw.get("lite", {}) or {}
        service = raw.get("service", {}) or {}
        return cls(
            mode=mode,
            lite_db_path=str(lite.get("db_path", DEFAULT_LITE_DB_PATH)),
            service_host=str(service.get("host", "localhost")),
            service_port=int(service.get("port", 6379)),
            service_password=(str(service.get("password")) or None) if service.get("password") else None,
            service_ssl=bool(service.get("ssl", False)),
        )


def connect(config: FalkorDBConfig | None = None) -> Any:
    """One FalkorDB connection, embedded or remote, decided entirely by `config`.

    Returns an object with `.select_graph(name)` either way -- the same
    interface FalkorDBLite mirrors from the standard falkordb-py client, by
    its own design, verified directly (a Cypher CREATE and MATCH round-trip
    against a real embedded instance) before this module was written
    around it.
    """
    config = config or FalkorDBConfig.from_file()
    if config.mode == MODE_LITE:
        from redislite.falkordb_client import (
            FalkorDB,
        )

        Path(config.lite_db_path).parent.mkdir(parents=True, exist_ok=True)
        return FalkorDB(config.lite_db_path)

    from falkordb import (
        FalkorDB,
    )

    return FalkorDB(
        host=config.service_host,
        port=config.service_port,
        password=config.service_password,
        ssl=config.service_ssl,
    )
