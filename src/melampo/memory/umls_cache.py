"""Cache UMLS crosswalk results in the encrypted store, so a licensed lookup is made once, not on every query.

Two reasons this cache exists, not one. The practical reason: a live UTS API
call on every normalisation attempt would be slow and would hit NLM's
servers far more than a bounded, cacheable lookup set requires -- crosswalk
results for a given HPO code do not change between UMLS releases within a
project's working lifetime. The license reason: UMLS content is not public
domain the way HPO is, and the license agreement makes the licensee
responsible for protecting it -- caching through `EncryptedJsonlStore`
rather than a plain file is what makes "cache the results" consistent with
that obligation rather than working against it.
"""

from dataclasses import dataclass
from typing import Any

from .encrypted_store import EncryptedJsonlStore


@dataclass
class UmlsCache:
    """A crosswalk cache backed by an encrypted store, keyed by (hpo_id, target_source)."""

    store: EncryptedJsonlStore
    _index: dict[str, list[dict[str, Any]]] | None = None

    def _ensure_loaded(self) -> dict[str, list[dict[str, Any]]]:
        if self._index is None:
            self._index = {}
            for record in self.store.load():
                # Overwrite rather than accumulate: each stored record
                # already is the complete result set for its key, not an
                # incremental item to append to a growing list. Iterating
                # the append-only log in order and overwriting means the
                # most recently cached entry for a key wins, exactly the
                # semantics a cache needs -- a first version of this method
                # appended each record's result list as a single nested
                # element instead, producing a list-of-lists that broke the
                # very first read.
                self._index[record["key"]] = record["result"]
        return self._index

    def _key(self, hpo_id: str, target_source: str | None) -> str:
        return f"{hpo_id}|{target_source or ''}"

    def get(self, hpo_id: str, target_source: str | None = None) -> list[dict[str, Any]] | None:
        """Cached crosswalk results for this HPO code, or None if never cached.

        None (never cached) is distinct from an empty list (cached, and UMLS
        genuinely has no crosswalk for this code) -- the second is a real,
        worth-keeping answer, and treating it the same as "go look it up
        again" would repeat a network call this cache exists to avoid.
        """
        return self._ensure_loaded().get(self._key(hpo_id, target_source))

    def put(self, hpo_id: str, target_source: str | None, results: list[dict[str, Any]]) -> None:
        self.store.append({"key": self._key(hpo_id, target_source), "result": results})
        self._ensure_loaded()[self._key(hpo_id, target_source)] = results


def crosswalk_with_cache(
    connector: Any, cache: UmlsCache, hpo_id: str, *, target_source: str | None = None
) -> list[Any]:
    """Crosswalk an HPO code, serving from the encrypted cache when available.

    A thin function rather than a method on either class, since it composes
    two things (a live connector, a cache) that each have a reason to exist
    independently -- a caller with only a cache (offline, or replaying a
    previous session) or only a connector (a one-off lookup not worth
    persisting) should not have to construct the other.
    """
    from ..connectors.umls import CrosswalkResult

    cached = cache.get(hpo_id, target_source)
    if cached is not None:
        return [CrosswalkResult(**record) for record in cached]

    live = connector.crosswalk_from_hpo(hpo_id, target_source=target_source)
    cache.put(hpo_id, target_source, [item.as_dict() for item in live])
    return live
