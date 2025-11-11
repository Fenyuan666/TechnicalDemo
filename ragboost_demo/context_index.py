"""Context index used to simulate KV cache awareness."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .data import Document
from .retrieval import RetrievedContext


@dataclass
class CacheEntry:
    cache_id: str
    doc_sequence: Tuple[str, ...]
    hits: int = 0

    def longest_prefix_with(self, candidate_sequence: Iterable[str]) -> int:
        length = 0
        for cached_id, incoming_id in zip(self.doc_sequence, candidate_sequence):
            if cached_id != incoming_id:
                break
            length += 1
        return length


class ContextIndex:
    """Tracks cached context sequences and provides ordering/dedup signals."""

    def __init__(self) -> None:
        self._entries: Dict[str, CacheEntry] = {}
        self._doc_to_cache: Dict[str, set[str]] = {}
        self._sequence_counter = itertools.count(1)

    # --- cache bookkeeping -------------------------------------------------
    def update_cache(self, doc_ids: List[str]) -> CacheEntry:
        cache_id = f"cache-{next(self._sequence_counter)}"
        entry = CacheEntry(cache_id=cache_id, doc_sequence=tuple(doc_ids))
        self._entries[cache_id] = entry
        for doc_id in doc_ids:
            self._doc_to_cache.setdefault(doc_id, set()).add(cache_id)
        return entry

    def record_hit(self, cache_id: str) -> None:
        if cache_id in self._entries:
            self._entries[cache_id].hits += 1

    # --- query helpers -----------------------------------------------------
    def best_prefix_match(self, doc_ids: List[str]) -> Optional[Tuple[CacheEntry, int]]:
        best: Optional[Tuple[CacheEntry, int]] = None
        for entry in self._entries.values():
            prefix_len = entry.longest_prefix_with(doc_ids)
            if prefix_len == 0:
                continue
            if not best or prefix_len > best[1]:
                best = (entry, prefix_len)
        return best

    def cached_doc_locations(self, doc_id: str) -> List[str]:
        return sorted(self._doc_to_cache.get(doc_id, []))

    def seen_doc(self, doc_id: str) -> bool:
        return doc_id in self._doc_to_cache


@dataclass
class ContextOrdering:
    """Reorders contexts to maximize cache prefix reuse."""

    index: ContextIndex

    def reorder(self, contexts: List[RetrievedContext]) -> Tuple[List[RetrievedContext], List[str]]:
        if not contexts:
            return contexts, []

        doc_ids = [ctx.doc.doc_id for ctx in contexts]
        best = self.index.best_prefix_match(doc_ids)
        if not best:
            return contexts, []

        entry, prefix_len = best
        reorder_map = {doc_id: idx for idx, doc_id in enumerate(entry.doc_sequence)}

        def order_key(ctx: RetrievedContext) -> Tuple[int, float]:
            desired = reorder_map.get(ctx.doc.doc_id, len(reorder_map) + 1)
            return (desired, -ctx.score)

        reordered = sorted(contexts, key=order_key)
        order_hint = (
            f"Reordered to reuse {prefix_len}/{len(entry.doc_sequence)} cached docs "
            f"from {entry.cache_id}."
        )
        self.index.record_hit(entry.cache_id)
        return reordered, [order_hint]


@dataclass
class ContextDeduplication:
    """Filters duplicate contexts and emits location hints for skipped docs."""

    index: ContextIndex
    max_contexts: int = 4

    def deduplicate(self, contexts: List[RetrievedContext]) -> Tuple[List[RetrievedContext], List[str]]:
        deduped: List[RetrievedContext] = []
        hints: List[str] = []

        for ctx in contexts:
            doc_id = ctx.doc.doc_id
            if self.index.seen_doc(doc_id):
                locations = ", ".join(self.index.cached_doc_locations(doc_id))
                hints.append(f"Doc {doc_id} reused via {locations}.")
                deduped.append(self._pointer_context(ctx, locations))
            else:
                deduped.append(ctx)

            if len(deduped) >= self.max_contexts:
                break

        return deduped, hints

    @staticmethod
    def _pointer_context(ctx: RetrievedContext, locations: str) -> RetrievedContext:
        pointer_doc = Document(
            doc_id=ctx.doc.doc_id,
            title=f"{ctx.doc.title} (cached)",
            content=f"Refer to cached context via {locations}.",
            tags=ctx.doc.tags,
        )
        return RetrievedContext(doc=pointer_doc, score=ctx.score - 0.1)
