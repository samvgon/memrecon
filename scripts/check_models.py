"""Send one small question to the agent and the judge through the model client.

Local setup (the default), with both servers up (scripts/serve.sh status):

    uv run python scripts/check_models.py

Hosted setup (MEMRECON_MODELS=config/models.openrouter.env in .env): the same
command; it makes two small paid calls, so MEMRECON_ALLOW_PAID=1 must be set.

Checks that each model answers, that thinking is off (the client refuses
reasoning text), and that the raw call is saved under results/raw_calls/.
Calls are cached, so a second run answers from disk; pass --fresh to call
the models again.
"""

import sys

from memrecon.llm import ModelCallError, client

QUESTIONS = {
    "agent": "What is 17 times 3? Answer with the number only.",
    "judge": "Is the sky green? Answer yes or no.",
}


def main() -> int:
    fresh = "--fresh" in sys.argv
    failed = False
    for role, question in QUESTIONS.items():
        c = client(role)
        try:
            r = c.chat([{"role": "user", "content": question}], max_tokens=50, use_cache=not fresh)
        except ModelCallError as e:
            print(f"{role}: FAILED: {e}")
            failed = True
            continue
        source = "cache" if r.cached else r.served_by
        print(f"{role} ({c.ep.model}, from {source}): {r.content!r}")
        print(f"  tokens: {r.usage}  finish: {r.finish_reason}")
        print(f"  raw: {r.raw_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
