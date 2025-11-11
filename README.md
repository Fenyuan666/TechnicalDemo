# TechnicalDemo

RAGBOOST demo implementation that mirrors the workflow described in the prompt.  
The repository now contains a lightweight Python pipeline with:

- **Retrieval System** (`ragboost_demo/retrieval.py`) – keyword overlap retriever that returns the initial contexts.
- **RAGBoost module** (`ragboost_demo/ragboost.py`) – orchestrates context deduplication and reordering.
- **Context Index** (`ragboost_demo/context_index.py`) – tracks cached prefixes, exposes context ordering + de-dup hints, and simulates KV reuse stats.
- **Inference Engine** (`ragboost_demo/inference.py`) – mock LLM that consumes the updated contexts, produces an answer, and updates the cache metadata.
- **Demo wiring** (`ragboost_demo/demo.py`) – end-to-end script that showcases two conversation turns, highlighting cache hits, hints, and metrics.

## Run the demo

```bash
python -m ragboost_demo.demo
```

The script prints each stage of the pipeline so you can see:

1. Retrieved vs. enhanced contexts (after ordering & dedup).
2. Injected hints that preserve semantic ordering or point to cached content.
3. Cache IDs, prefix overlap, and simulated prefill token savings.
4. The synthesized answer produced by the toy inference engine.
