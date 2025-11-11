"""Demo implementation of a RAGBoost-style pipeline."""

__all__ = ["build_demo_pipeline"]


def build_demo_pipeline():
    from .demo import build_demo_pipeline as _build

    return _build()
