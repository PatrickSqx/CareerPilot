# Phase 2.18E Old-vs-Evidence Benchmark

Generated: 2026-06-06T05:35:53Z

## Validation

- Candidate membership unchanged: True
- D1 rows: 501
- D2 rows: 501
- Duplicate D1 persona/job rows: 0
- Duplicate D2 persona/job rows: 0

## Overall Deltas

| Cutoff | Strict P@K old | Strict P@K evidence | Delta | Strict pass delta | Mean score delta |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.9750 | 0.9750 | +0.0000 | +0.0000 | +0.0340 |
| 25 | 0.9900 | 0.9900 | +0.0000 | +0.0000 | +0.0321 |
| 50 | 0.9750 | 0.9750 | +0.0000 | +0.0000 | +0.0300 |
| 100 | 0.9075 | 0.9100 | +0.0025 | +0.0025 | +0.0278 |
| 200 | 0.5825 | 0.5825 | +0.0000 | +0.0000 | +0.0261 |

## Per Persona Rank Movement

- aisha: rows=89, adjusted=71, moved_up=33, moved_down=28, max_gain=12, max_loss=-13
- kenji: rows=146, adjusted=121, moved_up=54, moved_down=70, max_gain=27, max_loss=-16
- marcus: rows=176, adjusted=153, moved_up=60, moved_down=82, max_gain=17, max_loss=-9
- priya: rows=90, adjusted=88, moved_up=44, moved_down=37, max_gain=15, max_loss=-36

## Boundaries

- Production /match was not modified.
- Production ranking weights were not modified.
- Embeddings, FAISS, and ANN artifacts were not rebuilt.
- Phase 1 ingestion was not modified.
- No live APIs were used.
- Evidence results remain offline dry-run evidence, not a production ranking claim.
