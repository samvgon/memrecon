# Hand labels

Sam's marks from hand checks, read back from the Probe Hand Check page's database and kept here as the
record. One JSON object per line. Never edited after saving; a later round is a new file.

| File | What it holds |
|---|---|
| `2026-11-07-probe-handcheck-v2-partial.jsonl` | Extractor version 2 (E28) on E27's stratified sample: rows marked so far (about a third of the sessions), and missed details per line. `kind: row` has the row as shown (`detail`, `qualifier`, `field`, `who`, `row_kind`, `status`, `attached_to`), `verdict` (ok or wrong) and `note`; `kind: missed` has the line and `note`. Reviewed together on Nov 7, which led to extractor version 3. |
| `2026-11-07-probe-handcheck-v3.jsonl` | Extractor version 3 (E29) on the same 37 sessions: all 152 rows marked (116 ok, 36 wrong) and 8 lines with missed details. Same fields as version 2, plus `record` (what R is, from the log), `number`, `resolved` and `replaces`. |
| `2026-11-08-probe-handcheck-v4.jsonl` | Extractor version 4 (E30) plus the split pass (E31) on the same 37 sessions: all 183 rows marked (179 ok, 4 wrong; 16 of them parts from the split pass, all ok) and 3 lines with missed details. Same fields as version 3, plus `origin` (`extract` or `split`) and `carried_from` (`v3` when the mark was copied from version 3 because the row was identical in every column the model writes, and Sam left it as it was). |
