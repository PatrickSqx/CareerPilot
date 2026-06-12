# H-1B/LCA Employer Activity Cache

LCA activity is historical or recent employer filing activity from DOL OFLC disclosure data. It is not confirmed sponsorship for any specific job posting.

- Official source page: https://www.dol.gov/agencies/eta/foreign-labor/performance
- Generated at: 2026-06-05T00:37:37Z
- As-of date: 2026-03-31
- Rows seen: 1,144,306
- Deduped records used: 1,097,777
- Visa classes included: H-1B

## Rolling Windows

- `recent_2q`: 2025-10-01 through 2026-03-31 (2 quarters)
- `recent_3q`: 2025-07-01 through 2026-03-31 (3 quarters)
- `historical_8q`: 2024-04-01 through 2026-03-31 (8 quarters)

## Output Files

- `employer_role_lca_summary.csv`: employer plus role-family rolling-window counts.
- `employer_lca_summary.csv`: employer-level rolling-window counts across all roles.
- `employer_lca_lookup.json`: employer-key lookup for future enrichment or diagnostics.
- `lca_cache_manifest.json`: source files, windows, row counts, and limitations.

## Top Recent Focus-Role Activity

| Employer | Role family | Label | Recent 3Q certified cases | Recent 3Q certified positions |
|---|---|---|---:|---:|
| Qualcomm Technologies, Inc. | software_engineering | recent_lca_activity_high | 338 | 22318 |
| Amazon.com Services LLC | software_engineering | recent_lca_activity_high | 4258 | 19143 |
| CGI Technologies and Solutions Inc. | software_engineering | recent_lca_activity_high | 345 | 9704 |
| NVIDIA Corporation | software_engineering | recent_lca_activity_high | 757 | 7815 |
| Oracle America, Inc. | software_engineering | recent_lca_activity_high | 542 | 7337 |
| Amazon Web Services, Inc. | software_engineering | recent_lca_activity_high | 1136 | 6654 |
| Cisco Systems, Inc. | software_engineering | recent_lca_activity_high | 341 | 6445 |
| Apple Inc. | software_engineering | recent_lca_activity_high | 2144 | 5790 |
| GOLDMAN SACHS SERVICES LLC | software_engineering | recent_lca_activity_high | 355 | 5404 |
| Qualcomm Atheros, Inc. | software_engineering | recent_lca_activity_high | 65 | 4916 |
| Amazon.com Services LLC | data_analytics | recent_lca_activity_high | 686 | 4814 |
| GOLDMAN SACHS & CO. LLC | data_analytics | recent_lca_activity_high | 84 | 4242 |
| Amazon Development Center U.S., Inc. | software_engineering | recent_lca_activity_high | 809 | 4105 |
| Qualcomm Innovation Center, Inc. | software_engineering | recent_lca_activity_high | 54 | 4014 |
| Deloitte Consulting LLP | software_engineering | recent_lca_activity_high | 1784 | 3654 |
| KFORCE INC. | software_engineering | recent_lca_activity_high | 329 | 3320 |
| Qualcomm Technologies, Inc. | ml_ai | recent_lca_activity_high | 40 | 3208 |
| JPMorgan Chase & Co. | software_engineering | recent_lca_activity_high | 1336 | 3179 |
| Microsoft Corporation | software_engineering | recent_lca_activity_high | 2971 | 3139 |
| Meta Platforms, Inc | software_engineering | recent_lca_activity_high | 1831 | 3031 |
