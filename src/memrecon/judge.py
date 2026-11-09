"""The judge for Memora's yes/no sub-questions (D21, D61).

The judge model (role `judge`, a different model family from the agent, D16) answers one sub-question
about one agent answer, seeing only the answer (as Memora's judge does). Answer read strictly, with the
same retries as the agent's calls (D36). A sub-question is passed when the judge's answer equals Memora's
expected answer ("yes" for memory_presence, "no" for forgetting_absence), and FAMA is computed from the
passes (fama.py).

The Stage 0 check (D21) compares the judge with Sam's hand labels: raw agreement >= 90%, Cohen's kappa
>= 0.7, and a flip rate <= 5% when every item is judged twice (the second time with another request seed,
so the call is made again rather than read from the cache).
"""

from __future__ import annotations

from memrecon.ingest import MAX_ATTEMPTS, UnusableAnswer, answer_format, extract_block, load_prompt
from memrecon.llm import ModelCallError, OutputTruncated

JUDGE_PROMPT = "judge_v1"  # D61


def parse_judgement(reply: str) -> tuple[str, str]:
    data = extract_block(reply, "judgement")
    answer, reason = data.get("answer"), data.get("reason", "")
    if set(data) - {"answer", "reason"} or answer not in ("yes", "no") or not isinstance(reason, str):
        raise UnusableAnswer('The block must be {"reason": "...", "answer": "yes" or "no"}.')
    return answer, reason.strip()


class Judge:
    def __init__(self, client, prompt: str = JUDGE_PROMPT, max_tokens: int = 512, tag: str | None = None,
                 call_settings: dict | None = None) -> None:
        self.client = client
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.tag = tag
        self.call_settings = dict(call_settings or {})

    def judge(self, response: str, question: str, meta: dict | None = None) -> dict:
        """{"answer": "yes"/"no" or None, "reason", "error", "attempts", "cached"}."""
        system, template = load_prompt(self.prompt)
        user = template.format(response=response, question=question)
        reminder = " " + answer_format(system)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        limit, error, cached = self.max_tokens, "", True
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = self.client.chat(messages, max_tokens=limit, tag=self.tag, **self.call_settings,
                                          meta={"step": "judge", "attempt": attempt, **(meta or {})})
                cached = cached and getattr(result, "cached", False)
            except OutputTruncated:
                limit *= 2
                error = f"cut off at {limit // 2} tokens"
                continue
            except ModelCallError as e:
                error = str(e)
                if "reasoning" in error or "unreadable" in error:
                    continue
                return {"answer": None, "reason": "", "error": error, "attempts": attempt, "cached": False}
            try:
                answer, reason = parse_judgement(result.content)
                return {"answer": answer, "reason": reason, "error": "", "attempts": attempt, "cached": cached}
            except UnusableAnswer as e:
                error = str(e)
                messages = messages + [
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": f"That answer could not be used: {e} Answer again, following the instructions exactly.{reminder}"},
                ]
        return {"answer": None, "reason": "", "error": error, "attempts": MAX_ATTEMPTS, "cached": False}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters' yes/no labels on the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in ("yes", "no"))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
