<!-- markdownlint-disable-file -->
<!--
Probe split pass, version 1 (D49, pending Sam's review). Asked only for an answer-key row whose detail is
at least 6 words long and contains "and" (not numbers, dates or names; only rows that hold, not ones that
remove or finish something). Sam's idea from the hand check of version 3 and 4: an item that joins
separate things ("provide a preliminary financial analysis ... and outline key considerations ...") should
be two probes; code alone cannot tell them from one thing with "and" inside ("account management and
fraud detection modules"), so the judge decides. Each part is copied exactly from the detail (code
checks) and becomes a row attached to the whole, which stays as it is. Harness side, run by the judge
model; never shown to the agent. Never edit a version that has been used in a run: copy it to v2.
-->

=== SYSTEM ===
You check one detail that a user told a personal assistant: is it one thing, or several separate things joined together?

You are given the line the user said, the detail (copied from that line) and what the detail is (its field).

It is one thing when "and" joins words inside a single thing: "account management and fraud detection modules", "technical roadblocks and mitigation strategies", "Training Materials for Operations and Compliance Teams", "raw energy and harder rock sound". It is several things when it joins separate items of a list or separate actions, each of which could be remembered or forgotten on its own: "provide a preliminary financial analysis of the pilot and outline key considerations for the next phase" is two things ("provide a preliminary financial analysis of the pilot", "outline key considerations for the next phase").

If it is one thing, answer with an empty list. If it is several, give each part copied exactly from the detail, word for word, without the joining "and" or comma. Do not reword a part or add words to it.

Answer with exactly one block in this form and nothing else:
<parts>{"parts": []}</parts>

=== USER ===
Line: {line}

Detail: {detail}

Field: {field}
