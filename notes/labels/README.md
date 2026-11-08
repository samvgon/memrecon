# Hand labels

Sam's marks from hand checks, read back from the Probe Hand Check page's database and kept here as the
record. One JSON object per line. Never edited after saving; a later round is a new file.

| File | What it holds |
|---|---|
| `2026-11-07-probe-handcheck-v2-partial.jsonl` | Extractor version 2 (E28) on E27's stratified sample: rows marked so far (about a third of the sessions), and missed details per line. `kind: row` has the row as shown (`detail`, `qualifier`, `field`, `who`, `row_kind`, `status`, `attached_to`), `verdict` (ok or wrong) and `note`; `kind: missed` has the line and `note`. Reviewed together on Nov 7, which led to extractor version 3. |
