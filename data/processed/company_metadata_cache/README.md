# Company Metadata Cache Outputs

Phase 2.17A writes no-key company metadata cache outputs here.

Generated files:

- `company_metadata_cache_frozen.csv`: offline-safe current company metadata rows. In 2.17A, all rows may remain `metadata_status=unknown` until private enrichment evidence is added.
- `company_metadata_evidence_frozen.jsonl`: redacted append-only evidence export. Private raw provider payloads are not exported.
- `company_metadata_cache_manifest.json`: build counts and boundary flags.
- `apify_bootstrap_plan_dry_run.json`: planned private Apify batches. This is a dry-run plan only and does not prove Apify was called.
- `company_metadata_apify_bootstrap_manifest.json`: Phase 2.17B private Apify bootstrap dry-run/live manifest.
- `company_metadata_apify_bootstrap_batches.jsonl`: Phase 2.17B redacted batch log. In dry-run mode it contains planned inputs; in live mode it contains counts only, not raw provider payloads.
- `company_metadata_local_remap_manifest.json`: no-cost local remap manifest for saved private Apify payloads whose returned item order did not match input order.
- `company_metadata_entity_scope_policy.md`: matching policy for franchise, brand, staffing, branch, and job-board alias cases.

Boundary:

- This cache is a company-level sidecar, not Phase 1 ingestion.
- Phase 2.17A/B does not change ranking behavior.
- Default/offline use does not require an Apify key.
- Unknown company size must stay neutral and must not be treated as small.
- Future ranking should use `usable_employer_size_bucket` / `size_bucket`, not raw brand or parent-company size, when `entity_scope=brand_or_parent`.
- Phase 2.17B live bootstrap requires explicit `--run-live` plus `APIFY_TOKEN`; dry-run mode does not call Apify.
- Future live runs preserve full returned provider batch items under `data/private/company_metadata_apify_raw_batches/` before matching. Those private raw archives are not processed exports.

Latest Phase 2.17B live bootstrap and local remap:

- 2,217 candidate companies processed in 89 live batches.
- Original live run inserted 2,217 evidence rows: 1,645 accepted, 461 no-result, 14 review, and 97 rejected.
- Four batches had provider result order that did not match input order. The local remap reused saved private raw payloads and made no Apify calls.
- Local remap appended 100 evidence rows: 86 accepted and 14 no-result.
- A targeted unknown-only rerun processed 489 remaining unknown companies with a $3 budget cap, preserved 20 private raw batch archives before matching, inserted 489 evidence rows, and added 11 accepted matches.
- Manual review confirmed two exact-name URL-conflict cases as valid company matches and appended two `manual_accepted` evidence rows.
- Cross-batch review confirmed 38 `AdventHealth *` location/unit rows as valid AdventHealth health-system parent context and appended 38 `manual_accepted` evidence rows. These are parent/system-level matches, not specific facility-page matches.
- Current cache status counts after targeted rerun and manual review: 1,779 `matched` and 438 `unknown`.
- Frozen current rows: 2,217; frozen redacted evidence rows: 2,846.
- Processed exports do not contain raw provider payloads or API-token material.
