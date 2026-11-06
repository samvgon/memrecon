"""Tests for the embedding client and retrieval (D35), against a fake embedding server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from memrecon.embed import Embedder
from memrecon.llm import ModelCallError, read_ledger
from memrecon.memory import At, MemoryStore, Source
from memrecon.retrieval import Retriever

CONFIG = """EMBED_KIND=embed
EMBED_FILE=~/x.gguf
EMBED_NAME=test-embed
EMBED_PORT=1
EMBED_CTX=8192
EMBED_QUERY_PREFIX=search_query:
EMBED_DOC_PREFIX=search_document:
"""

# A toy embedding: counts of a few topic words, so similarity is predictable.
TOPICS = ["budget", "coffee", "design", "steps"]


def toy_vector(text):
    t = text.lower()
    return [float(t.count(w)) + 0.01 for w in TOPICS]


class FakeEmbedServer:
    def __init__(self, fail_first=0):
        self.requests = []
        self.fail_first = fail_first
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                if fake.fail_first > 0:
                    fake.fail_first -= 1
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"loading")
                    return
                data = {
                    "data": [{"index": i, "embedding": toy_vector(t)} for i, t in enumerate(body["input"])],
                    "usage": {"prompt_tokens": 5 * len(body["input"])},
                }
                out = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    s = FakeEmbedServer()
    yield s
    s.httpd.shutdown()


@pytest.fixture
def embedder(server, tmp_path):
    cfg = tmp_path / "models.env"
    cfg.write_text(CONFIG)
    return Embedder(config=cfg, base_url=server.url, cache_path=tmp_path / "emb.sqlite", raw_dir=tmp_path / "raw", backoff=0)


def test_prefixes_cache_and_ledger(server, embedder):
    v = embedder.embed(["the budget", "coffee time"], "document")
    assert server.requests[0]["input"] == ["search_document: the budget", "search_document: coffee time"]
    assert server.requests[0]["model"] == "test-embed"
    assert abs(sum(x * x for x in v[0]) - 1) < 1e-6  # unit length
    again = embedder.embed(["coffee time", "the budget"], "document")
    assert len(server.requests) == 1  # both cached
    assert again[1] == pytest.approx(v[0])
    embedder.embed(["the budget"], "query")  # a different prefix is a different text
    assert server.requests[1]["input"] == ["search_query: the budget"]
    rows = read_ledger(embedder.raw_dir)
    assert [r["role"] for r in rows] == ["embed", "embed"] and rows[0]["meta"] == {"inputs": 2}


def test_batches(server, embedder):
    embedder.batch_size = 2
    embedder.embed([f"text {i}" for i in range(5)], "document")
    assert [len(r["input"]) for r in server.requests] == [2, 2, 1]


def test_retries_while_loading(tmp_path):
    s = FakeEmbedServer(fail_first=1)
    cfg = tmp_path / "models.env"
    cfg.write_text(CONFIG)
    e = Embedder(config=cfg, base_url=s.url, cache_path=tmp_path / "e.sqlite", raw_dir=tmp_path / "raw", backoff=0)
    assert len(e.embed(["budget"], "document")) == 1
    assert len(s.requests) == 2
    s.httpd.shutdown()


def test_unreachable_raises(tmp_path):
    cfg = tmp_path / "models.env"
    cfg.write_text(CONFIG)
    e = Embedder(config=cfg, base_url="http://127.0.0.1:9/v1", cache_path=tmp_path / "e.sqlite", raw_dir=tmp_path, retries=0, timeout=2)
    with pytest.raises(ModelCallError):
        e.embed(["x"], "document")


def test_needs_an_embed_role(tmp_path):
    cfg = tmp_path / "models.env"
    cfg.write_text("AGENT_KIND=chat\nAGENT_NAME=x\nAGENT_PORT=1\n")
    with pytest.raises(ValueError):
        Embedder(role="agent", config=cfg, cache_path=tmp_path / "e.sqlite")


def test_search_ranks_and_thresholds(embedder):
    r = Retriever(embedder, k=2, threshold=0.5)
    r.add([("m1", "budget budget"), ("m2", "coffee"), ("m3", "design and budget")])
    hits = r.search("what was the budget?")
    assert [mid for mid, _ in hits] == ["m1", "m3"]  # most relevant first
    assert hits[0][1] >= hits[1][1] >= 0.5
    assert r.search("steps today") == []  # nothing reaches the threshold
    assert [m for m, _ in r.search("coffee", k=1)] == ["m2"]


def test_sync_follows_the_store(embedder, server):
    s = MemoryStore("p")
    src = [Source("p", 1, 1)]
    a = s.note("budget for the project", src, At(1, 1))
    b = s.note("loves coffee", src, At(1, 2))
    r = Retriever(embedder, k=5)
    r.sync(s)
    assert len(r) == 2
    s.replace(a, "budget is tight, design too", At(2, 1))
    s.forget([b], At(3, 1))
    r.sync(s)
    assert len(r) == 1
    assert r.search("design")[0][0] == a
    calls = len(server.requests)
    r.sync(s)  # nothing changed: nothing re-embedded
    assert len(server.requests) == calls
