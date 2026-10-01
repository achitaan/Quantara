# Research charts and assistant verification — 2026-10-01

Preview: http://127.0.0.1:3100. API: http://127.0.0.1:8200. Existing records are preserved.

## What is visible

- Simulator: sampled live replay valuations, completed equity/benchmark curves, drawdown, series toggles and pointer/keyboard inspection. Full trades and execution assumptions remain available. A historical run completes once; Paper trading processes further bars.
- Models: live mean log rewards in windows of up to 100 training steps, completed episode returns, and held-out policy/buy-and-hold/SMA/momentum/benchmark curves. These are actual recorded observations. Training rewards are explicitly distinguished from untouched test performance. Curves preserve starting capital and endpoints; metrics use the full evaluation history. Benchmarks exclude execution costs. Old checkpoints have no recoverable learning telemetry and must be retrained to record it.
- Assistant: streamed previews, elapsed time, retrieval mode, source count, tool progress, source excerpts with pages, stop control and prompt examples. Preview citations are unchecked until the completed answer replaces the draft. Failed/cancelled streams never become final explanations.

## Changes behind the interface

Ollama streaming preserves validated tool calls and final citation checks. Greetings and document-only questions avoid unnecessary analytical schemas and saved records. Default maximum output is 600 tokens, with a configurable 30-minute keep-alive. A dedicated local chat worker avoids waiting behind two occupied research workers. Job polling and SSE can omit completed result payloads; the UI fetches each newly completed result separately.

The MiniLM encoder loads cached weights offline on CPU and is reused between queries. This avoids repeated network metadata checks and preserves Qwen's GPU allocation. `scripts/cache_embedding_model.py` performs an explicit one-time download for a new installation. FAISS indexes still validate model/revision/dimensions/document hash; text search remains available before indexing. The current demo index has been rebuilt locally.

## Measured outcomes

These are warm measurements on this machine, not service-level guarantees. Cold model loading and longer/tool-based questions still take time.

| Case | Completed reply | Evidence |
| --- | ---: | --- |
| Greeting, first optimized warm call | 1.30 s | First observed preview 1.04 s; 59 prompt tokens |
| Greeting, repeated warm call | 0.65 s | First observed preview 0.27 s |
| Risk tool | 16.14 s | Expected tool selected; valid arguments and citation identifiers |
| Equal-weight optimization | 8.51 s | Expected tool selected; valid arguments and citation identifiers |
| Tax tool | 10.03 s | Expected tool selected; valid arguments; provisional limitations preserved |
| Document query with cached local embeddings | 4.17 s | Retrieved/cited actual page 4; no citation warning |
| Repeated document query | 1.85 s | Local embedding retrieval/context preparation about 0.09 s |
| Real site document query | 4.3 s | Four local passages visible; no browser errors |

The pre-change greeting took 139.54 seconds. That run included different residency/prompt conditions, so it is not a controlled throughput comparison. Actual analytical explanations remain model output: valid citation identifiers do not prove every inference is correct. Numerical calculations stay in Python and their records remain inspectable. No hosted inference fallback was added.

Fresh PPO (2,048 actual steps, 21 reward samples, 13 completed episodes) and DDPG (2,000 steps, 20 samples, 13 episodes) records are saved in the demo workspace. Their held-out charts show the policy alongside simple strategies; model outperformance is not asserted.

## Checks and artifacts

- Backend suite: 69 passed, 9 optional/integration checks skipped. New checks cover partial/incomplete streams, cancellation, citation resolution, minimal greeting context, replay snapshot accounting and the reserved chat worker.
- Additional real neural checks: both PPO and DDPG reproducibility/held-out isolation/checksum tests passed; cached MiniLM/FAISS and DistilBERT verification passed with the local cache selected.
- Five browser workflows passed: end-to-end research, real policy training/interactive charts, streamed preview/source display, mobile navigation/safe rendering and theme persistence.
- Production build, TypeScript/API contract generation, Ruff and whitespace checks pass. The actual desktop server also rendered training curves and a cited RAG answer without JavaScript errors.

Ignored runtime evidence: `assistant-latency-before.json`, `assistant-latency-after.json`, `qwen-optimized-evaluation.json`, `rag-optimized-acceptance.json`, `training-visualization-acceptance.json`, `training-preview.png`, `assistant-rag-preview.png`. Full training rewards/checkpoint provenance live in the database and the checkpoint details panel.
