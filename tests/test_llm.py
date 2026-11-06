"""Tests for the model client, against a small fake server (no model needed)."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from memrecon.llm import (
    Endpoint,
    ModelCallError,
    ModelClient,
    ThinkingError,
    DEFAULT_MODELS,
    REPO_ROOT,
    BudgetExceeded,
    OutputTruncated,
    PaidCallBlocked,
    read_ledger,
    report,
    endpoint,
    load_dotenv,
    roles,
)


def ok_response(content="51", reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return 200, json.dumps(
        {
            "choices": [{"finish_reason": "stop", "index": 0, "message": message}],
            "usage": {"prompt_tokens": 27, "completion_tokens": 3, "total_tokens": 30},
        }
    )


class FakeServer:
    """Answers POST /v1/chat/completions with queued (status, body) pairs and records requests."""

    def __init__(self):
        self.requests = []
        self.replies = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                fake.requests.append(json.loads(self.rfile.read(length)))
                status, body = fake.replies.pop(0) if fake.replies else ok_response()
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


@pytest.fixture
def make_client(server, tmp_path):
    def make(**kwargs):
        ep = Endpoint(role="agent", base_url=server.url, model="test-model", api_key="sk-local")
        return ModelClient(ep, raw_dir=tmp_path / "raw", retries=0, backoff=0, **kwargs)

    return make


MSG = [{"role": "user", "content": "What is 17 times 3? Answer with the number only."}]


def test_request_turns_thinking_off_and_uses_model(server, make_client):
    result = make_client().chat(MSG, max_tokens=50)
    sent = server.requests[0]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["model"] == "test-model"
    assert sent["max_tokens"] == 50
    assert sent["temperature"] == 0.0
    assert result.content == "51"
    assert result.usage["completion_tokens"] == 3
    assert result.cached is False


def test_cannot_turn_thinking_on(make_client):
    with pytest.raises(ValueError):
        make_client().chat(MSG, chat_template_kwargs={"enable_thinking": True})


def test_raw_request_and_response_saved(make_client):
    result = make_client().chat(MSG)
    record = json.loads(result.raw_path.read_text())
    assert record["request"]["messages"] == MSG
    assert record["http_status"] == 200
    assert json.loads(record["response_text"])["choices"][0]["message"]["content"] == "51"
    assert result.raw_path.parent.parent.name == "agent"


def test_cache_hit_skips_server(server, make_client):
    client = make_client()
    first = client.chat(MSG)
    second = client.chat(MSG)
    assert len(server.requests) == 1
    assert second.cached is True
    assert second.content == first.content
    assert second.key == first.key


def test_different_settings_are_not_cached_together(server, make_client):
    client = make_client()
    a = client.chat(MSG, seed=0)
    b = client.chat(MSG, seed=1)
    assert a.key != b.key
    assert len(server.requests) == 2


def test_use_cache_false_calls_again(server, make_client):
    client = make_client()
    client.chat(MSG)
    client.chat(MSG, use_cache=False)
    assert len(server.requests) == 2


def test_reasoning_text_is_refused_but_saved(server, make_client):
    server.replies.append(ok_response(content="", reasoning="The user is asking"))
    client = make_client()
    with pytest.raises(ThinkingError):
        client.chat(MSG)
    saved = list((client.raw_dir / "agent").rglob("*.json"))
    assert len(saved) == 1
    assert "The user is asking" in json.loads(saved[0].read_text())["response_text"]


def test_empty_reasoning_field_is_fine(server, make_client):
    server.replies.append(ok_response(content="No.", reasoning=""))
    assert make_client().chat(MSG).content == "No."


def test_unreadable_response_is_saved_before_parsing(server, make_client):
    server.replies.append((200, "this is not json"))
    client = make_client()
    with pytest.raises(ModelCallError):
        client.chat(MSG)
    saved = list((client.raw_dir / "agent").rglob("*.json"))
    assert json.loads(saved[0].read_text())["response_text"] == "this is not json"


def test_failed_call_on_disk_is_retried_not_reused(server, make_client):
    server.replies.append((200, "this is not json"))
    client = make_client()
    with pytest.raises(ModelCallError):
        client.chat(MSG)
    result = client.chat(MSG)  # next reply is a good one
    assert result.content == "51"
    assert result.cached is False
    assert len(server.requests) == 2


def test_http_error_raises_with_raw_saved(server, make_client):
    server.replies.append((500, '{"error": "boom"}'))
    client = make_client()
    with pytest.raises(ModelCallError, match="HTTP 500"):
        client.chat(MSG)
    saved = list((client.raw_dir / "agent").rglob("*.json"))
    assert "boom" in json.loads(saved[0].read_text())["response_text"]


def test_503_is_retried(server, tmp_path):
    server.replies.append((503, '{"error": "loading model"}'))
    ep = Endpoint(role="judge", base_url=server.url, model="m", api_key="k")
    client = ModelClient(ep, raw_dir=tmp_path, retries=1, backoff=0)
    assert client.chat(MSG).content == "51"
    assert len(server.requests) == 2


def test_unreachable_server_raises(tmp_path):
    ep = Endpoint(role="agent", base_url="http://127.0.0.1:9/v1", model="m", api_key="k")
    client = ModelClient(ep, raw_dir=tmp_path, retries=1, backoff=0, timeout=2)
    with pytest.raises(ModelCallError, match="not reachable"):
        client.chat(MSG)
    assert not list(tmp_path.rglob("*.json"))


def _clear_env(monkeypatch, role):
    prefix = "MEMRECON_" + role.upper().replace("-", "_")
    for suffix in ("_URL", "_MODEL", "_API_KEY"):
        monkeypatch.delenv(prefix + suffix, raising=False)
    for var in ("MEMRECON_MODELS", "MEMRECON_ALLOW_PAID", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_endpoint_from_config(monkeypatch):
    _clear_env(monkeypatch, "judge")
    ep = endpoint("judge")
    assert ep.base_url == "http://localhost:8081/v1"
    assert ep.model == "gemma-4-26b-a4b-it-ud-q4_k_m"


def test_env_overrides_config(monkeypatch):
    _clear_env(monkeypatch, "judge")
    monkeypatch.setenv("MEMRECON_JUDGE_URL", "http://example:9999/v1/")
    monkeypatch.setenv("MEMRECON_JUDGE_MODEL", "other-model")
    ep = endpoint("judge")
    assert ep.base_url == "http://example:9999/v1"
    assert ep.model == "other-model"


def test_hyphenated_role(monkeypatch):
    _clear_env(monkeypatch, "agent-small")
    assert endpoint("agent-small").base_url == "http://localhost:8083/v1"
    monkeypatch.setenv("MEMRECON_AGENT_SMALL_URL", "http://elsewhere:1/v1")
    assert endpoint("agent-small").base_url == "http://elsewhere:1/v1"


def test_unknown_and_non_chat_roles_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        endpoint("other")
    with pytest.raises(ValueError, match="not a chat model"):
        endpoint("embed")


def test_swapping_a_model_is_a_config_change(tmp_path, monkeypatch):
    _clear_env(monkeypatch, "agent")
    config = tmp_path / "models.env"
    config.write_text("AGENT_KIND=chat\nAGENT_FILE=~/x.gguf\nAGENT_NAME=new-model\nAGENT_PORT=9000\nAGENT_CTX=4096\n")
    ep = endpoint("agent", config=config)
    assert (ep.model, ep.base_url) == ("new-model", "http://localhost:9000/v1")


def test_repo_config_is_complete():
    # Every role serve.sh can start has all five settings, and ports don't collide.
    found = roles(DEFAULT_MODELS)
    assert {"agent", "judge", "embed"} <= set(found)
    for role, settings in found.items():
        assert settings["PROVIDER"] == "llamacpp", role
        for field in ("KIND", "FILE", "NAME", "PORT", "CTX"):
            assert settings[field], (role, field)
        assert settings["KIND"] in ("chat", "embed"), role
        assert settings["FILE"].endswith(".gguf"), role
    ports = [s["PORT"] for s in found.values()]
    assert len(ports) == len(set(ports))


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nMEMRECON_T1="from-file"\nexport MEMRECON_T2=two\nMEMRECON_T3=three\n')
    monkeypatch.setenv("MEMRECON_T3", "already-set")
    monkeypatch.delenv("MEMRECON_T1", raising=False)
    monkeypatch.delenv("MEMRECON_T2", raising=False)
    load_dotenv(env)
    assert os.environ["MEMRECON_T1"] == "from-file"
    assert os.environ["MEMRECON_T2"] == "two"
    assert os.environ["MEMRECON_T3"] == "already-set"


# --- hosted (OpenRouter) roles -------------------------------------------------

OPENROUTER_CONFIG = REPO_ROOT / "config" / "models.openrouter.env"


def hosted_config(tmp_path, url):
    config = tmp_path / "hosted.env"
    config.write_text(
        "AGENT_KIND=chat\nAGENT_PROVIDER=openrouter\n"
        f"AGENT_URL={url}\nAGENT_NAME=qwen/qwen3.6-35b-a3b\n"
        "AGENT_HOST=deepinfra\nAGENT_KEY_ENV=TEST_OR_KEY\n"
    )
    return config


def hosted_client(tmp_path, url):
    return ModelClient(endpoint("agent", config=hosted_config(tmp_path, url)), raw_dir=tmp_path / "raw", retries=0, backoff=0)


def test_paid_call_blocked_without_opt_in(server, tmp_path, monkeypatch):
    _clear_env(monkeypatch, "agent")
    monkeypatch.setenv("TEST_OR_KEY", "sk-test")
    client = hosted_client(tmp_path, server.url)
    with pytest.raises(PaidCallBlocked):
        client.chat(MSG)
    assert server.requests == []  # nothing was sent


def test_hosted_request_turns_reasoning_off_and_pins_host(server, tmp_path, monkeypatch):
    _clear_env(monkeypatch, "agent")
    monkeypatch.setenv("TEST_OR_KEY", "sk-test")
    monkeypatch.setenv("MEMRECON_ALLOW_PAID", "1")
    server.replies.append((200, json.dumps({
        "provider": "DeepInfra",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "51", "reasoning": None}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2},
    })))
    result = hosted_client(tmp_path, server.url).chat(MSG)
    sent = server.requests[0]
    assert sent["reasoning"] == {"effort": "none"}
    assert sent["provider"] == {"order": ["deepinfra"], "allow_fallbacks": False}
    assert "chat_template_kwargs" not in sent
    assert sent["model"] == "qwen/qwen3.6-35b-a3b"
    assert result.content == "51"
    assert result.served_by == "openrouter/DeepInfra"
    assert json.loads(result.raw_path.read_text())["provider"] == "openrouter"


@pytest.mark.parametrize("field,value", [("reasoning", "Let me think"), ("reasoning_details", [{"type": "reasoning.text", "text": "hm"}])])
def test_hosted_reasoning_text_refused(server, tmp_path, monkeypatch, field, value):
    _clear_env(monkeypatch, "agent")
    monkeypatch.setenv("TEST_OR_KEY", "sk-test")
    monkeypatch.setenv("MEMRECON_ALLOW_PAID", "1")
    message = {"role": "assistant", "content": "51", field: value}
    server.replies.append((200, json.dumps({"choices": [{"finish_reason": "stop", "message": message}]})))
    with pytest.raises(ThinkingError, match=field):
        hosted_client(tmp_path, server.url).chat(MSG)


def test_hosted_missing_key(server, tmp_path, monkeypatch):
    _clear_env(monkeypatch, "agent")
    monkeypatch.delenv("TEST_OR_KEY", raising=False)
    monkeypatch.setenv("MEMRECON_ALLOW_PAID", "1")
    with pytest.raises(ModelCallError, match="TEST_OR_KEY"):
        hosted_client(tmp_path, server.url).chat(MSG)
    assert server.requests == []


def test_cached_hosted_answer_readable_without_opt_in(server, tmp_path, monkeypatch):
    _clear_env(monkeypatch, "agent")
    monkeypatch.setenv("TEST_OR_KEY", "sk-test")
    monkeypatch.setenv("MEMRECON_ALLOW_PAID", "1")
    client = hosted_client(tmp_path, server.url)
    client.chat(MSG)
    monkeypatch.delenv("MEMRECON_ALLOW_PAID")
    assert client.chat(MSG).cached is True  # reading a saved answer costs nothing
    assert len(server.requests) == 1


def test_models_file_selected_by_env(monkeypatch):
    _clear_env(monkeypatch, "agent")
    monkeypatch.setenv("MEMRECON_MODELS", "config/models.openrouter.env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    ep = endpoint("agent")
    assert ep.provider == "openrouter"
    assert ep.base_url == "https://openrouter.ai/api/v1"
    assert ep.api_key == "sk-test"
    monkeypatch.delenv("MEMRECON_MODELS")
    assert endpoint("agent").provider == "llamacpp"


def test_openrouter_config_is_complete():
    found = roles(OPENROUTER_CONFIG)
    assert {"agent", "judge"} <= set(found)
    for role, s in found.items():
        assert s["PROVIDER"] == "openrouter", role
        for field in ("KIND", "URL", "NAME", "HOST", "KEY_ENV"):
            assert s[field], (role, field)
        assert not s["FILE"], role  # a hosted role never names a local file


def test_reserved_settings_rejected(make_client):
    for name in ("reasoning", "provider", "chat_template_kwargs"):
        with pytest.raises(ValueError):
            make_client().chat(MSG, **{name: {}})


# --- cut-off answers, ledger, budget --------------------------------------------


def reply(content="51", finish="stop", usage=None, provider=None):
    data = {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 2}}
    if provider:
        data["provider"] = provider
    return 200, json.dumps(data)


def test_cut_off_answer_raises_and_is_not_reused(server, make_client):
    server.replies.append(reply("The budget is 100,0", finish="length"))
    client = make_client()
    with pytest.raises(OutputTruncated) as err:
        client.chat(MSG, max_tokens=5)
    assert err.value.text == "The budget is 100,0"
    assert err.value.max_tokens == 5
    assert client.chat(MSG, max_tokens=5).content == "51"  # called again, not answered from the cut-off copy
    assert len(server.requests) == 2


def test_empty_answer_is_not_reused(server, make_client):
    server.replies.append(reply(""))
    client = make_client()
    assert client.chat(MSG).content == ""
    assert client.chat(MSG).content == "51"
    assert len(server.requests) == 2


def test_hidden_reasoning_tokens_refused(server, make_client):
    usage = {"prompt_tokens": 10, "completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 38}}
    server.replies.append(reply("51", usage=usage))
    with pytest.raises(ThinkingError, match="38 reasoning tokens"):
        make_client().chat(MSG)


def test_ledger_records_real_calls_only(server, make_client):
    client = make_client()
    client.chat(MSG, tag="stage0-control")
    client.chat(MSG, tag="stage0-control")  # cache hit: free, not in the ledger
    rows = read_ledger(client.raw_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["tag"] == "stage0-control"
    assert row["role"] == "agent" and row["model"] == "test-model"
    assert row["prompt_tokens"] == 27 and row["completion_tokens"] == 3
    assert row["cost_usd"] == 0.0  # local calls cost nothing
    assert "stage0-control" in report(client.raw_dir)


def test_tag_is_not_part_of_the_cache_key(server, make_client):
    client = make_client()
    a = client.chat(MSG, tag="one")
    b = client.chat(MSG, tag="two")
    assert a.key == b.key and b.cached


def test_failed_calls_are_in_the_ledger(server, make_client):
    server.replies.append((500, '{"error": "boom"}'))
    client = make_client()
    with pytest.raises(ModelCallError):
        client.chat(MSG)
    row = read_ledger(client.raw_dir)[0]
    assert (row["http_status"], row["outcome"]) == (500, "http_error")
    assert "rejected" in report(client.raw_dir)


def test_hosted_cost_recorded_and_budget_stops(server, tmp_path, monkeypatch):
    _clear_env(monkeypatch, "agent")
    monkeypatch.setenv("TEST_OR_KEY", "sk-test")
    monkeypatch.setenv("MEMRECON_ALLOW_PAID", "1")
    monkeypatch.setenv("MEMRECON_BUDGET_USD", "0.01")
    usage = {"prompt_tokens": 1000, "completion_tokens": 10, "cost": 0.006}
    server.replies.extend([reply("51", usage=usage, provider="DeepInfra"), reply("52", usage=usage)])
    client = hosted_client(tmp_path, server.url)
    client.chat(MSG)
    assert server.requests[0]["usage"] == {"include": True}
    client.chat(MSG, seed=1)  # spent 0.006 < 0.01: allowed
    with pytest.raises(BudgetExceeded):
        client.chat(MSG, seed=2)  # spent 0.012 >= 0.01: blocked before sending
    assert len(server.requests) == 2
    rows = read_ledger(client.raw_dir)
    assert [r["cost_usd"] for r in rows] == [0.006, 0.006]
    assert rows[0]["served_by"] == "openrouter/DeepInfra"
    assert "0.0120" in report(client.raw_dir)


def test_budget_does_not_block_local_calls(server, make_client, monkeypatch):
    monkeypatch.setenv("MEMRECON_BUDGET_USD", "0")
    assert make_client().chat(MSG).content == "51"


def test_paid_call_without_reported_cost_is_flagged(server, tmp_path, monkeypatch):
    _clear_env(monkeypatch, "agent")
    monkeypatch.setenv("TEST_OR_KEY", "sk-test")
    monkeypatch.setenv("MEMRECON_ALLOW_PAID", "1")
    client = hosted_client(tmp_path, server.url)
    client.chat(MSG)
    assert read_ledger(client.raw_dir)[0]["cost_usd"] is None
    assert "no cost reported" in report(client.raw_dir)


# --- every attempt kept; outcomes in the ledger ---------------------------------


def test_retry_keeps_the_earlier_attempt(server, make_client):
    server.replies.append(reply("The budget is 100,0", finish="length"))
    client = make_client()
    with pytest.raises(OutputTruncated):
        client.chat(MSG)
    result = client.chat(MSG)  # retried; the cut-off attempt must survive
    folder = result.raw_path.parent
    attempt = folder / f"{result.key}.attempt1.json"
    assert attempt.exists()
    assert "100,0" in json.loads(attempt.read_text())["response_text"]
    assert json.loads(result.raw_path.read_text())["response_text"] != json.loads(attempt.read_text())["response_text"]


def test_attempts_are_numbered(server, make_client):
    server.replies.extend([reply("", finish="stop"), reply("", finish="stop")])
    client = make_client()
    for _ in range(3):
        client.chat(MSG)  # empty, empty, then the default good answer
    key = client.chat(MSG).key  # now a cache hit
    folder = client.path_for(key).parent
    assert sorted(p.name for p in folder.glob(f"{key}*")) == [f"{key}.attempt1.json", f"{key}.attempt2.json", f"{key}.json"]


def test_fresh_call_keeps_the_cached_answer_too(server, make_client):
    client = make_client()
    first = client.chat(MSG)
    client.chat(MSG, use_cache=False)
    assert (first.raw_path.parent / f"{first.key}.attempt1.json").exists()


@pytest.mark.parametrize(
    "response,outcome",
    [
        (reply("51"), "ok"),
        (reply(""), "empty"),
        (reply("51", finish="length"), "truncated"),
        (ok_response(content="", reasoning="thinking"), "reasoning"),
        ((200, "not json"), "unreadable"),
        ((500, "{}"), "http_error"),
    ],
)
def test_ledger_records_outcome(server, make_client, response, outcome):
    server.replies.append(response)
    client = make_client()
    try:
        client.chat(MSG)
    except ModelCallError:
        pass
    assert read_ledger(client.raw_dir)[-1]["outcome"] == outcome


def test_meta_goes_into_the_ledger_not_the_cache_key(server, make_client):
    client = make_client()
    a = client.chat(MSG, meta={"memory_tokens": 412, "persona": "creative_designer"})
    b = client.chat(MSG, meta={"memory_tokens": 999})
    assert a.key == b.key and b.cached
    assert read_ledger(client.raw_dir)[0]["meta"] == {"memory_tokens": 412, "persona": "creative_designer"}
