# Phase 2.18E.1 Graded Utility Benchmark

Generated: 2026-06-06T06:07:20Z
Runtime seconds: 5.924

## Scope

Offline evaluation only. This run did not modify /match, production ranking, embeddings, ANN artifacts, Phase 1 ingestion, or production weights.

## Overall Metrics

| K | old NDCG | evidence NDCG | delta | old mean utility | evidence mean utility | utility delta |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.959167 | 0.972201 | 0.013034 | 0.693762 | 0.697964 | 0.004201 |
| 25 | 0.972844 | 0.976498 | 0.003654 | 0.678957 | 0.677568 | -0.001389 |
| 50 | 0.982000 | 0.987529 | 0.005529 | 0.660302 | 0.662114 | 0.001812 |

## Per-Persona NDCG Delta

| persona | @10 | @25 | @50 |
|---|---:|---:|---:|
| aisha | 0.006753 | -0.007358 | -0.000482 |
| kenji | 0.014473 | 0.007226 | 0.013065 |
| marcus | 0.023934 | 0.012148 | 0.006947 |
| priya | 0.006978 | 0.002601 | 0.002587 |

## Utility Label Boundaries

- Hard-filter failures are gated to zero utility.
- Missing evidence is neutral because sidecar-derived terms only add nonnegative bonus.
- LCA/H-1B is employer historical filing activity only, not job-level sponsorship truth.
- Company size/type is a soft preference signal only.
- LLM overlay is partial/review-only and low weight.

## Go/No-Go

Decision: `go_to_18g0_contract_only`

Safety and evaluation contracts pass, but lift is modest or uneven; keep production integration behind a contract/design gate.

Manual review remains required for top movers before any production integration claim.
