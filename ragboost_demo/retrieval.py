"""Toy retrieval system used by the demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .data import Document


@dataclass
class RetrievedContext:
    doc: Document
    score: float

    def to_payload(self) -> dict:
        return {
            "doc_id": self.doc.doc_id,
            "title": self.doc.title,
            "snippet": self.doc.content,
            "score": round(self.score, 3),
        }


class RetrievalSystem:
    """A minimal retriever that ranks documents by token overlap."""

    def __init__(self, documents: Iterable[Document]):
        self._documents = list(documents)

    def retrieve(self, query: str, limit: int = 4) -> List[RetrievedContext]:
        query_tokens = self._tokenize(query)
        ranked = []
        for doc in self._documents:
            doc_tokens = doc.searchable_text().split()
            overlap = len(query_tokens.intersection(doc_tokens))
            if overlap == 0:
                continue

            semantic_hint = 1.0 if any(tag in query_tokens for tag in doc.tags) else 0.0
            score = overlap + semantic_hint
            ranked.append(RetrievedContext(doc=doc, score=score))

        ranked.sort(key=lambda ctx: ctx.score, reverse=True)
        return ranked[:limit]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = {token.strip(" ,.!?").lower() for token in text.split() if token.strip()}
        # Add single CJK characters so Chinese text without spaces can still match.
        tokens.update({char for char in text if "\u4e00" <= char <= "\u9fff"})
        return {t for t in tokens if t}
