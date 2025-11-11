"""Toy inference engine that simulates KV cache aware generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .context_index import ContextIndex
from .retrieval import RetrievedContext


@dataclass
class InferenceOutput:
    answer: str
    used_doc_ids: List[str]
    cache_id: str
    metrics: Dict[str, float]


class InferenceEngine:
    def __init__(self, index: ContextIndex) -> None:
        self.index = index

    def generate(
        self, prompt: str, contexts: Sequence[RetrievedContext], hints: Sequence[str]
    ) -> InferenceOutput:
        doc_ids = [ctx.doc.doc_id for ctx in contexts]
        prefix_match = self.index.best_prefix_match(doc_ids)
        prefix_overlap = prefix_match[1] if prefix_match else 0

        answer_parts = [f"Question: {prompt}", "Answer synthesis:"]
        for ctx in contexts:
            answer_parts.append(f"- ({ctx.doc.title}) {ctx.doc.content}")
        if hints:
            answer_parts.append("Hints injected: " + "; ".join(hints))
        answer = "\n".join(answer_parts)

        entry = self.index.update_cache(doc_ids)
        metrics = {
            "contexts_used": float(len(contexts)),
            "prefix_overlap": float(prefix_overlap),
            "prefill_tokens_saved": float(prefix_overlap * 128),
        }
        return InferenceOutput(
            answer=answer,
            used_doc_ids=doc_ids,
            cache_id=entry.cache_id,
            metrics=metrics,
        )
