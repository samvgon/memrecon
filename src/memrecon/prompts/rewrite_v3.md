<!-- markdownlint-disable-file -->
<!--
Recall-time rewrite, version 3 (D56, pending Sam's review). Version 2's anchor did not hold: in the rerun
pilot (results/stage0/control/55dede0b40, financial_analyst, not evidence) 3 of 5 note-arm rewrites came
back word for word as under version 1 ("$9.50 on a coffee" still became "$11.83 on lunch"; one meeting's
notes still became another's): the model takes a recalled memory to describe what the user is talking
about now and updates it to match. Version 3 says the rewrite writes what this memory says, in the form
most useful for the user's latest message, and that the memory may be about something other than what
the user is talking about now. Still nothing about keeping details or about bringing in what the user
says now (D54); if this fails too, Sam will consider adding "keep anything you wouldn't want to forget".
Checked on the same 20-session pilot for whether it does the task, not for how much it drifts.
Version 2's notes follow.
Recall-time rewrite, version 2 (D55). Version 1 with one sentence restored from
D53's draft: "The rewrite is a version of this memory: it is about what the memory is about." In the
20-session Stage 0 pilot (results/stage0/control/70eb37687c, financial_analyst, not evidence), version 1
mostly rewrote the conversation in the memory's place: "$9.50 on a coffee" became "$11.83 on lunch" (the
user's new message), one meeting's notes became another meeting's, a cost-benefit mandate became a
summary of a different project proposal. That replaces a memory rather than adapting it, and under
replace or keep would wipe the store within a few sessions. The sentence anchors the rewrite to its
memory without saying anything about keeping details or about bringing in what the user is saying now
(both still deliberately unsaid, D54). Version 1's notes follow.
Recall-time rewrite, version 1 (D53, D54). Step 2 of 2
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
by the harness. Never edit a version that has been used in a run: copy it to v4.
-->

=== SYSTEM ===
You are the memory of a personal assistant. While the user and the assistant talk, a memory useful for responding to the user's latest message has been recalled from the assistant's long-term memory. Rewrite the memory for the conversation right now: write what this memory says, in the form most useful for responding to the user's latest message. The memory may be about something other than what the user is talking about now; it is still about what it was about. Write it as a memory about the user, in plain sentences.

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
