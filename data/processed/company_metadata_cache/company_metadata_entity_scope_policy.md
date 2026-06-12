# Company Metadata Entity Scope Policy

Phase 2.17 company metadata enrichment must separate the LinkedIn entity that was matched from the employer-size signal that future ranking may safely use.

## Fields

`matched_entity_size_bucket`

- Size bucket for the entity returned by the provider.
- Example: a Taco Bell brand page can be `enterprise_10001_plus`.
- This is evidence about the matched LinkedIn entity only.

`entity_scope`

- `corporate_employer`: the matched entity is likely the same employer shown in the posting.
- `franchise_operator`: the matched entity is a multi-location operator/franchisee, not only the parent brand.
- `brand_or_parent`: the matched entity is a brand, parent, or corporate page for a franchise-heavy chain.
- `single_location`: evidence supports a specific store, branch, or local location entity.
- `staffing_agency`: the matched entity is a staffing, recruiting, or workforce intermediary.
- `job_board_alias`: the input looked like a job-board or source-system alias rather than an employer.
- `unknown`: scope is unresolved.

`usable_employer_size_bucket`

- Ranking-safe employer size.
- This may differ from `matched_entity_size_bucket`.
- For `brand_or_parent`, `staffing_agency`, `job_board_alias`, and `unknown`, keep this as `unknown` unless later evidence proves the matched entity is the actual employer context.

`size_bucket`

- Compatibility field for future ranking reads.
- It should mirror `usable_employer_size_bucket`, not the raw matched LinkedIn entity size.

`size_usage_policy`

- `usable_employer_context`: corporate employer size can be used as a soft company-size signal.
- `usable_franchise_operator_context`: franchise operator size can be used as a soft operator-size signal.
- `usable_single_location_context`: specific-location size can be used, if evidence supports it.
- `brand_context_only`: brand or parent size is stored but must not be used as employer size.
- `staffing_context_only`: staffing agency size is stored but must not be treated as end-client size or sponsorship evidence.
- `job_board_alias_ignore`: job-board/source alias evidence should not drive company-size scoring.
- `unknown_neutral`: no usable size evidence; future ranking must stay neutral.

## Franchise Rule

For franchise-heavy brands such as Taco Bell, Burger King, Popeyes, KFC, Subway, Pizza Hut, Dunkin, McDonald's, Wendy's, and similar chains:

- If the snapshot company is only the brand name and the provider returns the brand or parent company page, set `entity_scope=brand_or_parent`.
- Store the provider's matched size in `matched_entity_size_bucket`.
- Keep `usable_employer_size_bucket=unknown` and `size_bucket=unknown`.
- Do not use the parent or brand size to infer the posting's true employer size.
- Do not use brand size to infer sponsorship friendliness.

If the provider returns a named franchise operator, such as a company explicitly described as a franchisee/operator or DBA entity:

- Set `entity_scope=franchise_operator`.
- Use the operator's size as `usable_employer_size_bucket`.
- Still do not treat size as confirmed sponsorship.

If the company name appears to be a single store code, branch, or local unit:

- Use `entity_scope=single_location` only when evidence supports that exact local entity.
- Otherwise set `entity_scope=unknown` or route to review.

## Staffing Rule

For staffing and recruiting companies, the matched company size may be real, but it may describe the intermediary rather than the final end-client.

- Set `entity_scope=staffing_agency`.
- Store matched size separately.
- Keep `usable_employer_size_bucket=unknown` for default ranking unless a later workflow explicitly wants staffing-agency context.
- Never treat staffing-agency size as confirmed sponsorship evidence for the specific posting.

## Matching And Review Rules

High-confidence automatic use is allowed only when the matched entity appears to be the same corporate employer, franchise operator, or specific location represented by the posting.

Send to review or keep neutral when:

- The only match is a brand or parent page for a franchise-heavy chain.
- The company URL is only a job-board URL such as CareerBuilder or Snagajob.
- The name contains a store number, unit code, or location code but the provider returns only a brand page.
- Multiple plausible operator or parent entities exist.
- Domain evidence conflicts with the returned LinkedIn entity.

Unknown is neutral. It must not be treated as small, sponsor-hostile, or sponsor-friendly.

## Future Ranking Contract

Future Phase 2 ranking may read only the frozen offline cache.

- Read `size_bucket` or `usable_employer_size_bucket`, not `matched_entity_size_bucket`, for company-size scoring.
- Use company size as a soft preference only.
- Do not convert company size into confirmed sponsorship.
- Keep `unknown` neutral.
- Explain brand-only or staffing-only evidence as context, not as employer-size proof.
