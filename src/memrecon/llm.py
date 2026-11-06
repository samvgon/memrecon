"""Client for the models: the agent and the judge (D15–D18, D22).

Code only ever asks for a *role* ("agent", "judge", ...). Which model serves
each role, and where, is set in a models file:

- config/models.env (default): local llama.cpp servers started with
  scripts/serve.sh, the setup used for all of Sam's runs;
- config/models.openrouter.env: the same models hosted on OpenRouter, for
  anyone without the local setup. Select it with MEMRECON_MODELS in .env.

Every call goes through ModelClient.chat, which:

- always turns thinking off, in the way the role's provider expects, and
  refuses a response that still contains reasoning text;
- refuses to call a paid provider unless MEMRECON_ALLOW_PAID=1 is set, and
  stops once recorded spending reaches MEMRECON_BUDGET_USD, if that is set;
- raises OutputTruncated instead of returning an answer cut off at the
  output limit (a cut-off rewrite would look like memory drift);
- appends every call that reaches a model (not cache hits) to a ledger,
  results/raw_calls/ledger.jsonl, with tokens, time and cost;
  `uv run python -m memrecon.llm` prints a spending and token report;
- writes the raw request and the raw response text to disk *before*
  anything parses them, so a parsing bug never forces a rerun;
- caches each call under a hash of the full request (model name, messages and
  every setting), so a resumed or repeated run reuses the saved response
  instead of calling the model again. Cut-off, empty and failed answers are
  saved for inspection but never reused from the cache.

The cache is a resume aid, not proof of reproducibility: models are not
perfectly deterministic, even at temperature 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = REPO_ROOT / "config" / "models.env"

# How each provider is called. "paid" providers need MEMRECON_ALLOW_PAID=1.
PROVIDERS = {
    "llamacpp": {"paid": False},
    "openrouter": {"paid": True},
}

# Request settings the client controls itself; callers may not pass them.
RESERVED = {"chat_template_kwargs", "reasoning", "reasoning_effort", "provider", "model", "messages", "usage"}

# Placeholder key; local servers ignore it. Real keys live in .env (git-ignored), never in code.
LOCAL_KEY = "sk-local"


class ModelCallError(RuntimeError):
    """The model could not be reached, or returned an error or an unreadable response."""


class ThinkingError(ModelCallError):
    """The model returned reasoning text even though thinking was turned off."""


class PaidCallBlocked(ModelCallError):
    """A paid provider was about to be called without MEMRECON_ALLOW_PAID=1."""


class BudgetExceeded(ModelCallError):
    """Recorded spending has reached MEMRECON_BUDGET_USD; no more paid calls."""


class OutputTruncated(ModelCallError):
    """The model stopped at the output limit (max_tokens); the answer is incomplete.

    The partial text is in .text. The call is saved and in the ledger, but is
    not reused from the cache. Raise max_tokens for that step.
    """

    def __init__(self, message: str, text: str, max_tokens: int | None):
        super().__init__(message)
        self.text = text
        self.max_tokens = max_tokens


def read_env_file(path: str | Path) -> dict[str, str]:
    """KEY=VALUE lines of a .env-style file as a dict (comments and blank lines skipped)."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        values[key] = value.strip().strip('"').strip("'")
    return values


def load_dotenv(path: str | Path = REPO_ROOT / ".env") -> None:
    """Read a .env file into os.environ without overriding variables already set. Missing file: no-op."""
    if Path(path).is_file():
        for key, value in read_env_file(path).items():
            os.environ.setdefault(key, value)


def models_file() -> Path:
    """The models file in use: MEMRECON_MODELS (relative to the repo root, or absolute), else config/models.env."""
    chosen = os.environ.get("MEMRECON_MODELS")
    if not chosen:
        return DEFAULT_MODELS
    path = Path(chosen).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _prefix(role: str) -> str:
    return role.upper().replace("-", "_")


FIELDS = ("KIND", "PROVIDER", "FILE", "NAME", "PORT", "CTX", "URL", "HOST", "KEY_ENV")


def roles(config: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Every role in a models file with its settings, e.g. {"agent": {"KIND": "chat", "PORT": "8080", ...}}.

    PROVIDER defaults to llamacpp; settings a role doesn't use are empty strings.
    """
    values = read_env_file(config or models_file())
    out = {}
    for key in values:
        if not key.endswith("_NAME"):
            continue
        prefix = key[: -len("_NAME")]
        settings = {field: values.get(f"{prefix}_{field}", "") for field in FIELDS}
        settings["PROVIDER"] = settings["PROVIDER"] or "llamacpp"
        out[prefix.lower().replace("_", "-")] = settings
    return out


@dataclass(frozen=True)
class Endpoint:
    role: str
    base_url: str
    model: str
    api_key: str
    provider: str = "llamacpp"
    host: str = ""  # OpenRouter only: the one hosting company to use (no fallback to others)
    key_env: str = ""  # name of the variable the key came from, for error messages


def endpoint(role: str, config: str | Path | None = None) -> Endpoint:
    """Endpoint for a chat role in the models file.

    MEMRECON_<ROLE>_URL, _MODEL and _API_KEY in the environment (or .env)
    override the file for this machine.
    """
    config = Path(config) if config else models_file()
    available = roles(config)
    if role not in available:
        raise ValueError(f"unknown role {role!r}; {config.name} has {sorted(available)}")
    s = available[role]
    if s["KIND"] != "chat":
        raise ValueError(f"role {role!r} is {s['KIND'] or 'missing a KIND'}, not a chat model")
    if s["PROVIDER"] not in PROVIDERS:
        raise ValueError(f"role {role!r} has unknown PROVIDER {s['PROVIDER']!r}; expected one of {sorted(PROVIDERS)}")
    if not s["NAME"]:
        raise ValueError(f"{config.name} is missing NAME for {role!r}")
    if s["PROVIDER"] == "llamacpp":
        if not (s["URL"] or s["PORT"]):
            raise ValueError(f"{config.name} needs PORT (or URL) for {role!r}")
        default_url = s["URL"] or f"http://localhost:{s['PORT']}/v1"
    else:
        if not s["URL"]:
            raise ValueError(f"{config.name} needs URL for hosted role {role!r}")
        default_url = s["URL"]

    env = f"MEMRECON_{_prefix(role)}"
    key = os.environ.get(f"{env}_API_KEY") or (os.environ.get(s["KEY_ENV"], "") if s["KEY_ENV"] else LOCAL_KEY)
    return Endpoint(
        role=role,
        base_url=os.environ.get(f"{env}_URL", default_url).rstrip("/"),
        model=os.environ.get(f"{env}_MODEL", s["NAME"]),
        api_key=key,
        provider=s["PROVIDER"],
        host=s["HOST"],
        key_env=s["KEY_ENV"],
    )


@dataclass
class ChatResult:
    content: str
    finish_reason: str | None
    usage: dict
    key: str
    raw_path: Path
    cached: bool
    served_by: str  # the provider, plus the hosting company when OpenRouter reports one


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _atomic_write_json(path: Path, record: dict) -> None:
    """Write to a temp file, flush, fsync, then rename, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cache_key(body: dict) -> str:
    """sha256 of the request body (model name, messages and every setting), canonical JSON."""
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_ledger_lock = threading.Lock()


def ledger_path(raw_dir: str | Path) -> Path:
    return Path(raw_dir) / "ledger.jsonl"


def read_ledger(raw_dir: str | Path = REPO_ROOT / "results" / "raw_calls") -> list[dict]:
    path = ledger_path(raw_dir)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line cut off by a crash
    return rows


def total_spend(raw_dir: str | Path = REPO_ROOT / "results" / "raw_calls") -> float:
    return sum(row.get("cost_usd") or 0.0 for row in read_ledger(raw_dir))


def _append_ledger(raw_dir: Path, row: dict) -> None:
    path = ledger_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _has_text(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)  # e.g. a non-empty reasoning_details list


class ModelClient:
    def __init__(
        self,
        ep: Endpoint,
        raw_dir: str | Path = "results/raw_calls",
        timeout: float = 900.0,
        retries: int = 3,
        backoff: float = 10.0,
    ) -> None:
        self.ep = ep
        self.raw_dir = Path(raw_dir)
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def request_body(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 0,
        **extra,
    ) -> dict:
        reserved = RESERVED & set(extra)
        if reserved:
            raise ValueError(f"{sorted(reserved)} are set by the client (thinking is always off); don't pass them")
        body = {
            "model": self.ep.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            **extra,
        }
        if self.ep.provider == "llamacpp":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.ep.provider == "openrouter":
            body["reasoning"] = {"effort": "none"}
            body["usage"] = {"include": True}  # ask OpenRouter to report the call's cost
            if self.ep.host:
                body["provider"] = {"order": [self.ep.host], "allow_fallbacks": False}
        else:
            raise ValueError(f"unknown provider {self.ep.provider!r}")
        return body

    def path_for(self, key: str) -> Path:
        return self.raw_dir / self.ep.role / key[:2] / f"{key}.json"

    def chat(self, messages: list[dict], *, use_cache: bool = True, tag: str | None = None, **settings) -> ChatResult:
        """One model call. `tag` labels the call in the ledger (e.g. "stage0-control"); it is not part of the cache key."""
        body = self.request_body(messages, **settings)
        key = cache_key(body)
        path = self.path_for(key)

        if use_cache and path.exists():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                result = self._parse(record, key, path, cached=True)
                if result.content:
                    return result
            except (json.JSONDecodeError, ModelCallError):
                pass  # unreadable, failed, cut-off or empty answer on disk: call again and overwrite it

        self._check_allowed()
        record = {
            "key": key,
            "role": self.ep.role,
            "provider": self.ep.provider,
            "host": self.ep.host,
            "base_url": self.ep.base_url,
            "request": body,
            "started_at": _now(),
        }
        status, text, attempts, latency = self._post(body)
        record.update(
            {
                "http_status": status,
                "response_text": text,
                "attempts": attempts,
                "latency_s": round(latency, 3),
                "finished_at": _now(),
            }
        )
        _atomic_write_json(path, record)  # raw response saved before any parsing
        self._log(record, tag)
        return self._parse(record, key, path, cached=False)

    def _log(self, record: dict, tag: str | None) -> None:
        """Append a ledger row for a call that reached the model."""
        usage, finish, served = {}, None, record["provider"]
        try:
            data = json.loads(record["response_text"])
            usage = data.get("usage") or {}
            finish = (data.get("choices") or [{}])[0].get("finish_reason")
            if isinstance(data.get("provider"), str):
                served = f"{served}/{data['provider']}"
        except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
            pass
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)):
            cost = 0.0 if not PROVIDERS[self.ep.provider]["paid"] else None  # None: paid, but no cost reported
        _append_ledger(
            self.raw_dir,
            {
                "at": record["finished_at"],
                "role": self.ep.role,
                "model": self.ep.model,
                "served_by": served,
                "tag": tag or os.environ.get("MEMRECON_TAG", ""),
                "key": record["key"],
                "http_status": record["http_status"],
                "finish_reason": finish,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": cost,
                "latency_s": record["latency_s"],
            },
        )

    def _check_allowed(self) -> None:
        if PROVIDERS[self.ep.provider]["paid"] and os.environ.get("MEMRECON_ALLOW_PAID") != "1":
            raise PaidCallBlocked(
                f"role {self.ep.role!r} uses {self.ep.provider}, which costs money. "
                "Set MEMRECON_ALLOW_PAID=1 in .env to allow paid calls."
            )
        budget = os.environ.get("MEMRECON_BUDGET_USD")
        if budget and PROVIDERS[self.ep.provider]["paid"]:
            spent = total_spend(self.raw_dir)
            if spent >= float(budget):
                raise BudgetExceeded(
                    f"recorded spending ${spent:.4f} has reached MEMRECON_BUDGET_USD=${float(budget):.2f}; "
                    "raise it in .env to continue"
                )
        if not self.ep.api_key:
            raise ModelCallError(
                f"no API key for role {self.ep.role!r}: set {self.ep.key_env or 'its key variable'} in .env"
            )

    def _post(self, body: dict) -> tuple[int | None, str, int, float]:
        """POST with retries for connection errors, 429 (rate limit) and 503 (server loading or busy)."""
        url = f"{self.ep.base_url}/chat/completions"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.ep.api_key}"}
        last_error = ""
        for attempt in range(1, self.retries + 2):
            start = time.monotonic()
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                    return resp.status, text, attempt, time.monotonic() - start
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", errors="replace")
                if e.code not in (429, 503):
                    return e.code, text, attempt, time.monotonic() - start
                last_error = f"HTTP {e.code}: {text[:200]}"
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last_error = f"{type(e).__name__}: {e}"
            if attempt <= self.retries:
                time.sleep(self.backoff * attempt)
        raise ModelCallError(
            f"{self.ep.role} at {self.ep.base_url} not reachable after {self.retries + 1} attempts ({last_error})"
        )

    def _parse(self, record: dict, key: str, path: Path, cached: bool) -> ChatResult:
        status = record.get("http_status")
        if status != 200:
            raise ModelCallError(f"HTTP {status} from {self.ep.role}; raw response saved at {path}")
        try:
            data = json.loads(record["response_text"])
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            raise ModelCallError(f"unreadable response ({e!r}); raw response saved at {path}") from e
        # llama.cpp puts reasoning in reasoning_content; OpenRouter in reasoning / reasoning_details.
        for field in ("reasoning_content", "reasoning", "reasoning_details"):
            if _has_text(message.get(field)):
                raise ThinkingError(
                    f"{self.ep.role} returned reasoning text ({field}) with thinking off; raw response saved at {path}"
                )
        details = (data.get("usage") or {}).get("completion_tokens_details") or {}
        if isinstance(details, dict) and (details.get("reasoning_tokens") or 0) > 0:
            raise ThinkingError(
                f"{self.ep.role} used {details['reasoning_tokens']} reasoning tokens with thinking off "
                f"(the text may be hidden, but the tokens are billed); raw response saved at {path}"
            )
        content = (message.get("content") or "").strip()
        if choice.get("finish_reason") == "length":
            limit = record.get("request", {}).get("max_tokens")
            raise OutputTruncated(
                f"{self.ep.role} stopped at the output limit (max_tokens={limit}); the answer is cut off. "
                f"Raise max_tokens for this step. Raw response saved at {path}",
                text=content,
                max_tokens=limit,
            )
        served_by = record.get("provider", self.ep.provider)
        if isinstance(data.get("provider"), str):
            served_by = f"{served_by}/{data['provider']}"
        return ChatResult(
            content=content,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
            key=key,
            raw_path=path,
            cached=cached,
            served_by=served_by,
        )


def client(role: str, raw_dir: str | Path = REPO_ROOT / "results" / "raw_calls", **kwargs) -> ModelClient:
    """Convenience: read .env, then build a client for a role ("agent", "judge", ...)."""
    load_dotenv()
    return ModelClient(endpoint(role), raw_dir=raw_dir, **kwargs)


def report(raw_dir: str | Path = REPO_ROOT / "results" / "raw_calls") -> str:
    """Calls, tokens and cost from the ledger, by role, model and tag. Cache hits are not counted."""
    rows = read_ledger(raw_dir)
    if not rows:
        return f"No calls recorded yet ({ledger_path(raw_dir)})."
    groups: dict[tuple, dict] = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "usd": 0.0, "unpriced": 0, "failed": 0})
    for row in rows:
        g = groups[(row.get("role", ""), row.get("served_by", ""), row.get("model", ""), row.get("tag", ""))]
        g["calls"] += 1
        g["prompt"] += row.get("prompt_tokens") or 0
        g["completion"] += row.get("completion_tokens") or 0
        if row.get("cost_usd") is None:
            g["unpriced"] += 1
        else:
            g["usd"] += row["cost_usd"]
        if row.get("http_status") != 200 or row.get("finish_reason") == "length":
            g["failed"] += 1
    lines = [f"{'role':<12} {'served by':<24} {'tag':<16} {'calls':>6} {'prompt tok':>11} {'output tok':>11} {'USD':>11}"]
    for (role, served, _model, tag), g in sorted(groups.items()):
        lines.append(
            f"{role:<12} {served:<24} {tag or '-':<16} {g['calls']:>6} {g['prompt']:>11} {g['completion']:>11} {g['usd']:>11.6f}"
            + (f"  ({g['failed']} failed or cut off)" if g["failed"] else "")
            + (f"  ({g['unpriced']} paid calls with no cost reported)" if g["unpriced"] else "")
        )
    lines.append(f"total spend: ${total_spend(raw_dir):.6f}   (ledger: {ledger_path(raw_dir)})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else REPO_ROOT / "results" / "raw_calls"))
