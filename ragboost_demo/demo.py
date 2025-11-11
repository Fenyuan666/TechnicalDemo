"""Executable demo that wires all components together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .context_index import ContextDeduplication, ContextIndex, ContextOrdering
from .data import load_knowledge_base
from .inference import InferenceEngine, InferenceOutput
from .ragboost import RAGBoost
from .retrieval import RetrievalSystem, RetrievedContext


@dataclass
class PipelineResult:
    prompt: str
    retrieved: List[RetrievedContext]
    enhanced_contexts: List[RetrievedContext]
    hints: List[str]
    output: InferenceOutput

    def to_report(self) -> Dict:
        return {
            "prompt": self.prompt,
            "retrieved": [ctx.to_payload() for ctx in self.retrieved],
            "enhanced_contexts": [ctx.to_payload() for ctx in self.enhanced_contexts],
            "hints": self.hints,
            "answer": self.output.answer,
            "cache_id": self.output.cache_id,
            "metrics": self.output.metrics,
        }


class RagboostPipeline:
    def __init__(self) -> None:
        docs = load_knowledge_base()
        self.index = ContextIndex()
        self.retriever = RetrievalSystem(docs)
        self.ordering = ContextOrdering(self.index)
        self.dedup = ContextDeduplication(self.index)
        self.ragboost = RAGBoost(ordering=self.ordering, deduplication=self.dedup)
        self.inference = InferenceEngine(self.index)

    def ask(self, prompt: str) -> PipelineResult:
        retrieved = self.retriever.retrieve(prompt)
        enhanced_contexts, hints = self.ragboost.enhance(prompt, retrieved)
        output = self.inference.generate(prompt, enhanced_contexts, hints)
        return PipelineResult(
            prompt=prompt,
            retrieved=retrieved,
            enhanced_contexts=enhanced_contexts,
            hints=hints,
            output=output,
        )


def build_demo_pipeline() -> RagboostPipeline:
    return RagboostPipeline()


def _format_report(report: Dict) -> str:
    lines = [f"Prompt: {report['prompt']}", f"Cache ID: {report['cache_id']}"]
    lines.append("Retrieved contexts:")
    for row in report["retrieved"]:
        lines.append(f"  - {row['doc_id']} (score={row['score']})")
    lines.append("Enhanced contexts:")
    for row in report["enhanced_contexts"]:
        lines.append(f"  - {row['doc_id']} (score={row['score']})")
    if report["hints"]:
        lines.append("Hints:")
        for hint in report["hints"]:
            lines.append(f"  * {hint}")
    lines.append("Metrics:")
    for key, value in report["metrics"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("Answer preview:")
    lines.append(report["answer"])
    return "\n".join(lines)


def main() -> None:
    pipeline = build_demo_pipeline()
    prompts = [
        "请解释量子力学的基本原理，并说明它与相对论的区别。",
        "继续说明量子力学和KV缓存之间的联系，以及RAGBoost如何加速推理。",
    ]
    for idx, prompt in enumerate(prompts, start=1):
        result = pipeline.ask(prompt)
        report = result.to_report()
        print("=" * 80)
        print(f"Turn {idx}")
        print(_format_report(report))


if __name__ == "__main__":
    main()
