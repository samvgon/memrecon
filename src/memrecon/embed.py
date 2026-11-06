"""Embeddings for retrieval (D35).

The `embed` role in the models file (config/models.env) names the embedding
model and its port. Texts get the model's task prefix before embedding
(nomic-embed-text-v1.5: "search_query:" for what is searched with,
"search_document:" for what is searched), set in the models file as
EMBED_QUERY_PREFIX and EMBED_DOC_PREFIX.

Every vector is cached in results/embeddings.sqlite under a hash of the
model name and the exact prefixed text, so each text is embedded once,
whichever run asks for it. Calls that reach the server are appended to the
ledger like chat calls (role "embed", tokens, outcome). Vectors are
normalised to length 1, so a dot product is the cosine similarity.
Embeddings are never shown to the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path

from memrecon.llm import REPO_ROOT, ModelCallError, _append_ledger, _now, load_dotenv, models_file, roles

DEFAULT_CACHE = REPO_ROOT / "results" / "embeddings.sqlite"
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "raw_calls"


def _normalise(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


class Embedder:
    def __init__(
        self,
        role: str = "embed",
        cache_path: str | Path = DEFAULT_CACHE,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        config: str | Path | None = None,
        base_url: str | None = None,
        batch_size: int = 32,
        timeout: float = 300.0,
        retries: int = 3,
        backoff: float = 5.0,
    ) -> None:
        load_dotenv()
        settings = roles(config or models_file()).get(role)
        if settings is None or settings["KIND"] != "embed":
            raise ValueError(f"no embedding role {role!r} in the models file")
        env = f"MEMRECON_{role.upper().replace('-', '_')}"
        self.role = role
        self.model = os.environ.get(f"{env}_MODEL", settings["NAME"])
        self.base_url = (base_url or os.environ.get(f"{env}_URL") or f"http://localhost:{settings['PORT']}/v1").rstrip("/")
        self.prefixes = {"query": settings["QUERY_PREFIX"], "document": settings["DOC_PREFIX"]}
        self.raw_dir = Path(raw_dir)
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path))
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, model TEXT, dim INTEGER, vec BLOB)")
        self.db.commit()
        self.calls = 0

    def prefixed(self, text: str, kind: str) -> str:
        if kind not in self.prefixes:
            raise ValueError("kind must be 'query' or 'document'")
        p = self.prefixes[kind]
        return f"{p} {text}" if p else text

    def key(self, prefixed_text: str) -> str:
        return hashlib.sha256(f"{self.model}\0{prefixed_text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        """Unit-length vectors for `texts`, in order; cached texts are not sent again."""
        prepared = [self.prefixed(t, kind) for t in texts]
        keys = [self.key(t) for t in prepared]
        found = self._lookup(keys)
        missing = [(k, t) for k, t in dict(zip(keys, prepared)).items() if k not in found]
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i : i + self.batch_size]
            vectors = self._request([t for _, t in chunk])
            self.db.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)",
                [(k, self.model, len(v), array("f", v).tobytes()) for (k, _), v in zip(chunk, vectors)],
            )
            self.db.commit()  # each batch is saved as soon as it arrives
            found.update({k: v for (k, _), v in zip(chunk, vectors)})
        return [found[k] for k in keys]

    def _lookup(self, keys: list[str]) -> dict[str, list[float]]:
        out = {}
        unique = list(dict.fromkeys(keys))
        for i in range(0, len(unique), 500):
            part = unique[i : i + 500]
            rows = self.db.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})", part
            ).fetchall()
            for k, blob in rows:
                a = array("f")
                a.frombytes(blob)
                out[k] = list(a)
        return out

    def _request(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        url = f"{self.base_url}/embeddings"
        last = ""
        for attempt in range(1, self.retries + 2):
            start = time.monotonic()
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                vectors = [None] * len(inputs)
                for item in data["data"]:
                    vectors[item["index"]] = _normalise(item["embedding"])
                if any(v is None for v in vectors):
                    raise ModelCallError("embedding response is missing some inputs")
                self.calls += 1
                self._log(len(inputs), (data.get("usage") or {}).get("prompt_tokens"), "ok", time.monotonic() - start)
                return vectors
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", errors="replace")
                self._log(len(inputs), None, "http_error", time.monotonic() - start)
                if e.code != 503:
                    raise ModelCallError(f"HTTP {e.code} from {self.role}: {text[:300]}") from e
                last = f"HTTP 503: {text[:200]}"
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last = f"{type(e).__name__}: {e}"
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                self._log(len(inputs), None, "unreadable", time.monotonic() - start)
                raise ModelCallError(f"unreadable embedding response ({e!r})") from e
            if attempt <= self.retries:
                time.sleep(self.backoff * attempt)
        raise ModelCallError(f"{self.role} at {self.base_url} not reachable after {self.retries + 1} attempts ({last})")

    def _log(self, n_inputs: int, tokens, outcome: str, latency: float) -> None:
        _append_ledger(
            self.raw_dir,
            {
                "at": _now(),
                "role": self.role,
                "model": self.model,
                "served_by": "llamacpp",
                "tag": os.environ.get("MEMRECON_TAG", ""),
                "http_status": 200 if outcome == "ok" else None,
                "finish_reason": None,
                "outcome": outcome,
                "prompt_tokens": tokens,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "latency_s": round(latency, 3),
                "meta": {"inputs": n_inputs},
            },
        )
