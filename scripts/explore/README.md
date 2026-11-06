# Data exploration scripts

Every finding about the data that a decision relies on must come from a script in this folder, not from an ad hoc command typed into a terminal. That way each finding can be re-run, and if we later find we misread a field, we can see exactly which code and which assumption produced the decision.

## Rules

1. **One question per script**, named `E<number>_<short_name>.py` (E01, E02, ...). The docstring states the question, the method and the limits.
2. **Run through the runner**, from the repo root: `uv run python scripts/explore/run.py E01`. It saves `notes/evidence/E01.txt` with a header recording the date, the memrecon commit, the Memora commit and the exit status, followed by the output.
3. **Commit the script and its evidence file together**, before citing it in a decision. The header's code version is only meaningful if the scripts are committed (it says if they are not).
4. **Never edit a script after a decision cites it.** If the question or method changes, write a new script with the next number and say in the log which one supersedes the old one.
5. **Log it** in the decision log: an entry in the evidence register (what it answers, what it showed, which assumptions it uses) and, for every interpretation of a field that it relies on, an entry in the assumptions register. Decisions cite evidence as `E03` and assumptions as `A2`.
6. **Scripts only read the dataset.** Nothing here writes to `data/`.

## Current scripts

| ID | Question |
|---|---|
| E01 | What do `session_type` and `operation` contain, and how many of each? |
| E02 | How many remembered and forgotten items do Memora's evaluation questions list? Does the counter miss differently shaped evidence? |
| E03 | For one project's add, delete and update sessions, what does the dialogue say compared with the log? |
| E04 | How many turns are flagged `share_memory: True`, and how long are they? |
| E05 | How often do flagged turns contain hedging language (loose and tight searches)? |
| E06 | How much of what the log records for adds and updates is spoken in the dialogue, and how much of it is in flagged turns? |
| E07 | Which flagged turns are facts to remember (add, update sessions) and which are instructions to forget (delete sessions)? |
| E08 | How big is Memora's scored rubric (the `evaluation` yes/no sub-questions)? Supersedes E02 for sizing. |
| E09 | How long is each persona's full history (turns, characters, estimated tokens)? |
| E10 | LoCoMo (`data/locomo/locomo10.json`): conversation lengths, question categories, evidence references. Prints the file's sha256 (the data is not a git clone). |
| E11 | LongMemEval (`data/longmemeval/*.json`): question types, history lengths, answer sessions and `has_answer` turns. Prints each file's sha256. Written from the README before the data was seen. |
| E12 | After a fact is introduced, how often do later sessions mention it again (strict string match, user turns, excluding update/delete of the same fact)? Lower bound on natural recall opportunities. |
| E13 | Is file order the same as date order (assumption A6), and are session IDs and turn numbers unique and consecutive? Needed before applying operations in order and pointing memories at source turns. |
| E14 | How many code-gradable probe candidates (numbers, quoted titles, names, hashtags, qualifiers, the record's own value) do the flagged turns of add and update sessions give, per persona, and how many belong to facts never changed later? Sizes the probe pool for the power estimate (D10). Candidates found by pattern, not checked probes. |

`common.py` holds the shared loaders and the version helpers. `../make_split.py` recreates or checks the frozen persona split.
