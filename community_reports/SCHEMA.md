# Community field-report ledger — schema & moderation lifecycle

`ledger.jsonl` is a **moderated, repo-committed** record of Michigander field
reports about specific utility poles. It is **JSON Lines**: one independent
JSON object per line, no enclosing array, no trailing commas. The deterministic
VISTA pipeline reads this file and folds the moderated rows in as a
**citizen-corroboration overlay** — a priority / corroboration signal that is
reconciled against the model. It is **never** model training input and it
**never** changes `predict_proba`, the model, drivers, tiers, or risk ordering.

## Why an overlay, not a model input

This is public-safety infrastructure data. An open, user-writable channel that
fed the model would be an obvious abuse and poisoning surface (a single bad
actor could move risk scores). Instead:

- The model stays a closed, deterministic function of the audited feature
  stack (NOAA normals + image-derived structure/vegetation features).
- Community reports ride **alongside** the model output as corroboration: they
  raise/clear human attention and let a planner see "the model says HIGH and
  two field reports independently agree" vs. "uncorroborated."
- Every report is **maintainer-reviewed before it appears** (see lifecycle).

## Record fields

Every line is exactly this object (key order is not significant):

| field         | type                          | meaning |
|---------------|-------------------------------|---------|
| `report_id`   | string                        | Stable unique id, e.g. `CR-2026-0001`. The ledger is sorted by this in the payload. |
| `pole_id`     | string \| null                | Fleet pole id (e.g. `P00114`) the report is about, or `null` if the location does not map to a known fleet pole. |
| `lat`         | number                        | WGS84 latitude of the reported location. |
| `lon`         | number                        | WGS84 longitude of the reported location. |
| `county`      | string                        | County name (one of the DTE SE-Michigan counties). |
| `conditions`  | array of strings              | One or more observed conditions. Canonical values: `Leaning pole`, `Vegetation contact`, `Damaged hardware/crossarm`, `Low/down wire`, `Cracked/rotted pole`, `Other`. |
| `severity`    | `"low" \| "medium" \| "urgent"` | Reporter's urgency assessment. |
| `note`        | string                        | Short free-text description. No PII expected or required. |
| `reporter`    | string                        | Optional self-chosen handle. May be empty (`""`). No PII required. |
| `submitted`   | string `YYYY-MM-DD`           | Date the report was submitted (fixed string; no clock is read by the pipeline). |
| `status`      | `"verified" \| "pending" \| "rejected"` | Moderation state (see lifecycle). |
| `source`      | `"resident" \| "lineman" \| "sample"` | Origin channel. `sample` rows are seeded reference examples. |

## Moderation lifecycle

```
  resident/lineman submits  →  in-app export (.jsonl)  →  Pull Request adding
  line(s) to community_reports/ledger.jsonl

        ┌─────────────┐  maintainer review
        │  pending    │ ───────────────────────────┐
        └─────────────┘                             │
              │ corroborated / plausible            │ spam / unsafe / duplicate
              ▼                                      ▼
        ┌─────────────┐                       ┌─────────────┐
        │  verified   │                       │  rejected   │
        └─────────────┘                       └─────────────┘
```

- **pending** — submitted via PR, not yet reviewed. Folded into the overlay as
  an unverified signal (shown, but not counted as corroboration).
- **verified** — a maintainer confirmed the report is plausible and not abuse.
  Folded in and counted as **corroboration** for its pole.
- **rejected** — spam, unsafe, or duplicate. **Excluded entirely** from the
  pipeline (the row may stay in the file as an audit trail; it is dropped on
  ingest).

The pipeline ingests only `status ∈ {verified, pending}` and drops
`rejected`. Ingestion is deterministic: rows are normalized and sorted by
`report_id`, so two pipeline runs over the same ledger produce a
byte-identical `output/app_data.json`.

## Contributing

Use the in-app **"＋ Report a pole"** mode, then
**"⤓ Download community reports (.jsonl)"**. Open a Pull Request that appends
the downloaded line(s) to `community_reports/ledger.jsonl` (keep it valid JSON
Lines — one object per line). A maintainer reviews and flips `status` from
`pending` to `verified` or `rejected` before it appears as corroboration.
