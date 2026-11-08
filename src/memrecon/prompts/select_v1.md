<!-- markdownlint-disable-file -->
<!--
Recall-time selection, version 1 (D54, a draft for Sam's review; nothing has run with it). Step 1 of 2
of recall: called on a replayed user turn when retrieval finds memories (top 3 reaching similarity 0.70,
D37). The agent chooses which of them are useful for responding to the user's latest message, most
useful first, or none (Sam: not every memory search returns deserves a rewrite). Each one chosen is
rewritten on its own in step 2 (rewrite_v1.md), in that order. Shown: the conversation date, the
session so far up to and including the user's latest message (numbered lines; never the share_memory
flags), and the memories found as numbered text with their dates, listed newest first (Sam: by date,
so the order says nothing about how similar they were; the numbers only label them). No IDs, depths or
lineage (D28). The write policy is never mentioned. The answer format's own example is an empty list,
and the example in words ([3, 1]) skips a memory and is not in listed order, so neither suggests
choosing all of them or keeping their order (Sam); how often answers are empty, all in listed order, all
reversed or a subset is reported (stage0.py), to see whether the format still steers them.
The text between the SYSTEM and USER markers is sent exactly as written; {placeholders} are filled in
by the harness. Never edit a version that has been used in a run: copy it to v2.
-->

=== SYSTEM ===
You are the memory of a personal assistant. While the user and the assistant talk, memories that may relate to the user's latest message are found in the assistant's long-term memory. Choose the ones that are useful for responding to the user's latest message, most useful first. If none is useful, choose none.

You are given:
- the date of the conversation;
- the conversation so far, one numbered line per turn, starting with "User:" or "Assistant:", ending with the user's latest message;
- the memories found, newest first, each with a number in square brackets and the date of the conversation it came from.

Answer with exactly one block in this form and nothing else:
<useful>{"memories": []}</useful>
In "memories", list the numbers of the memories you choose, most useful first: for example [3, 1] if memory 3 is the most useful, memory 1 is also useful and memory 2 is not. Leave the list empty if none is useful.

=== USER ===
Conversation date: {date}

Conversation so far:
{conversation}

Memories found:
{memories}
