"""RAGBoost orchestrator for the demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .context_index import ContextDeduplication, ContextOrdering
from .retrieval import RetrievedContext


@dataclass
class RAGBoost:
    ordering: ContextOrdering
    deduplication: ContextDeduplication

    def enhance(
        self, prompt: str, retrieved: List[RetrievedContext]
    ) -> Tuple[List[RetrievedContext], List[str]]:
        del prompt  # prompt is part of the signature to mirror a real system

        deduped, dedup_hints = self.deduplication.deduplicate(retrieved)
        reordered, order_hints = self.ordering.reorder(deduped)
        hints = dedup_hints + order_hints
        return reordered, hints
