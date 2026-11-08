<!-- markdownlint-disable-file -->
<!--
Recall-time rewrite, version 1 (D53, D54; a draft for Sam's review; nothing has run with it). Step 2 of 2
of recall: called once for each memory chosen in step 1 (select_v1.md), in the order chosen, each on its
own (Sam: one call rewriting several memories may get lazy as it goes). The agent rewrites the memory
for the conversation right now, as just-in-time memory systems do (JitMem, MemHarness, ReMe). The
instruction is deliberately neutral: it neither asks to keep every detail (a defence against drift built
into the prompt; Sam: leave it out for now, a possible later arm) nor to shorten or summarize. Nor does
it say whether to bring in what the user is saying now (Sam: don't tell it either way; track how often
rewrites add details not in the memory, a way information could be implanted unintentionally). What
happens to the rewrite is the write policy's business and is never mentioned. Shown: the conversation
date, the session so far up to and including the user's latest message (numbered lines; never the
share_memory flags), and the memory with its date (no IDs, depths or lineage, D28).
The text between the SYSTEM and USER markers is sent exactly as written; {placeholders} are filled in
by the harness. Never edit a version that has been used in a run: copy it to v2.
-->

=== SYSTEM ===
You are the memory of a personal assistant. While the user and the assistant talk, a memory useful for responding to the user's latest message has been recalled from the assistant's long-term memory. Rewrite it so that it is most useful for the conversation right now. Write it as a memory about the user, in plain sentences.

You are given:
- the date of the conversation;
- the conversation so far, one numbered line per turn, starting with "User:" or "Assistant:", ending with the user's latest message;
- the memory, with the date of the conversation it came from.

Answer with exactly one block in this form and nothing else:
<memory>{"text": "..."}</memory>

=== USER ===
Conversation date: {date}

Conversation so far:
{conversation}

Memory:
{memory}
