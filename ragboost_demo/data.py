"""Static dataset used by the RAGBoost demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Document:
    """Simple representation of a knowledge base document."""

    doc_id: str
    title: str
    content: str
    tags: tuple[str, ...] = ()

    def searchable_text(self) -> str:
        return " ".join((self.title, self.content, " ".join(self.tags))).lower()


def load_knowledge_base() -> List[Document]:
    """Return a tiny in-memory knowledge base for the demo."""

    return [
        Document(
            doc_id="physics-quantum",
            title="Quantum Mechanics Fundamentals",
            content=(
                "Quantum mechanics studies matter and light on the atomic and "
                "subatomic scale. Wave-particle duality, quantization, and "
                "probability amplitudes describe system behavior. 量子力学 研究 "
                "原子 尺度 的 物质 与 光 并 使用 概率 波 函数 描述 系统 。"
            ),
            tags=("physics", "quantum", "science"),
        ),
        Document(
            doc_id="physics-relativity",
            title="Relativity Basics",
            content=(
                "Einstein's special relativity introduces a constant speed of "
                "light and links space with time. General relativity extends the "
                "idea to describe gravity as curvature of spacetime. 相对论 强调 "
                "时空 的 统一 并 通过 时空 弯曲 来 描述 引力 。"
            ),
            tags=("physics", "relativity"),
        ),
        Document(
            doc_id="ml-rag",
            title="Retrieval Augmented Generation",
            content=(
                "RAG systems retrieve external documents to ground the reasoning "
                "of large language models. They combine a retriever with a generator. "
                "RAG 系统 先 检索 文档 再 结合 大 模型 生成 更 准确 的 答案 。"
            ),
            tags=("machine-learning", "rag"),
        ),
        Document(
            doc_id="systems-kv-cache",
            title="KV Cache Acceleration",
            content=(
                "Key-value caches store transformer attention activations so "
                "repeated prefixes are not recomputed. Cache hits drastically "
                "reduce prefill latency. KV 缓存 记录 上下文 的 注意力 结果 ， "
                "命中 时 可以 跳过 预填 阶段 。"
            ),
            tags=("systems", "optimization"),
        ),
        Document(
            doc_id="ragboost-paper",
            title="RAGBoost Concepts",
            content=(
                "RAGBoost reorders and deduplicates retrieved contexts so they "
                "align with cached prefixes, injecting hints to preserve "
                "original ranking. RAGBoost 会 对 文档 重排 去重 ， 增强 KV 缓存 "
                "命中 率 并 保留 语义 提示 。"
            ),
            tags=("rag", "ragboost"),
        ),
    ]
