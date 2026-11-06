"""Tests for the run monitor's readers and the memory guard (D31)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from memrecon.memory import At, MemoryStore, Source
from memrecon.monitor import (
    GiB,
    MemoryPressure,
    dir_size,
    gpu_memory,
    latest_summaries,
    ledger_rates,
    parse_meminfo,
    parse_prometheus,
    save_summary,
    store_summary,
    system_memory,
    wait_for_memory,
)

MEMINFO = "MemTotal:       131072000 kB\nMemFree:         1000000 kB\nMemAvailable:    65536000 kB\nHugePages_Total:       0\n"


def test_parse_meminfo_converts_kb_to_bytes():
    info = parse_meminfo(MEMINFO)
    assert info["MemTotal"] == 131072000 * 1024
    assert info["HugePages_Total"] == 0


def test_system_memory(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text(MEMINFO)
    assert system_memory(f) == {"total": 131072000 * 1024, "available": 65536000 * 1024}
    assert system_memory(tmp_path / "missing") is None


def test_gpu_memory_reads_amdgpu_counters(tmp_path):
    dev = tmp_path / "card1" / "device"
    dev.mkdir(parents=True)
    (dev / "mem_info_vram_used").write_text("536870912\n")
    (dev / "mem_info_gtt_used").write_text("42949672960\n")
    (tmp_path / "card0" / "device").mkdir(parents=True)  # a card without counters is skipped
    cards = gpu_memory(tmp_path)
    assert cards == [{"card": "card1", "vram_used": 536870912, "vram_total": None, "gtt_used": 42949672960, "gtt_total": None}]


def _meminfo(tmp_path, available_kb):
    f = tmp_path / "meminfo"
    f.write_text(f"MemTotal: 131072000 kB\nMemAvailable: {available_kb} kB\n")
    return f


def test_guard_passes_when_memory_is_free(tmp_path):
    slept = []
    wait_for_memory(8, meminfo=_meminfo(tmp_path, 20 * 1024 * 1024), sleep=slept.append, say=lambda s: None)
    assert slept == []


def test_guard_waits_then_continues(tmp_path):
    f = _meminfo(tmp_path, 2 * 1024 * 1024)  # 2 GB
    calls = []

    def fake_sleep(s):
        calls.append(s)
        f.write_text("MemTotal: 131072000 kB\nMemAvailable: 20971520 kB\n")  # memory freed: 20 GB

    wait_for_memory(8, poll_s=30, meminfo=f, sleep=fake_sleep, say=lambda s: None)
    assert calls == [30]


def test_guard_gives_up_cleanly(tmp_path):
    with pytest.raises(MemoryPressure, match="2.0 GB available"):
        wait_for_memory(8, poll_s=30, max_wait_s=60, meminfo=_meminfo(tmp_path, 2 * 1024 * 1024), sleep=lambda s: None, say=lambda s: None)


def test_guard_threshold_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMRECON_MIN_FREE_GB", "1")
    wait_for_memory(meminfo=_meminfo(tmp_path, 2 * 1024 * 1024), sleep=lambda s: pytest.fail("should not wait"), say=lambda s: None)


def test_guard_is_a_no_op_without_meminfo(tmp_path):
    wait_for_memory(8, meminfo=tmp_path / "missing", sleep=lambda s: pytest.fail("should not wait"), say=lambda s: None)


def test_parse_prometheus():
    text = (
        "# HELP llamacpp:prompt_tokens_seconds Average prompt throughput\n"
        "# TYPE llamacpp:prompt_tokens_seconds gauge\n"
        "llamacpp:prompt_tokens_seconds 84.12\n"
        'llamacpp:requests_processing{slot="0"} 1\n'
        "llamacpp:kv_cache_usage_ratio 0.125\n"
    )
    m = parse_prometheus(text)
    assert m == {"llamacpp:prompt_tokens_seconds": 84.12, "llamacpp:requests_processing": 1.0, "llamacpp:kv_cache_usage_ratio": 0.125}


def test_ledger_rates_window_and_memory_tokens():
    now = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    def row(minutes_ago, **kw):
        return {"at": (now - timedelta(minutes=minutes_ago)).isoformat(), "role": "agent", **kw}
    rows = [
        row(1, prompt_tokens=3000, completion_tokens=100, outcome="ok", meta={"memory_tokens": 400}),
        row(2, prompt_tokens=1000, completion_tokens=50, outcome="reasoning"),
        row(30, prompt_tokens=99999, completion_tokens=999, outcome="ok"),  # outside the window
        {"at": "not a time", "role": "agent"},
    ]
    r = ledger_rates(rows, now=now.timestamp(), window_s=300)["agent"]
    assert (r["calls"], r["prompt_tokens"], r["output_tokens"], r["memory_tokens"], r["rejected"]) == (2, 4000, 150, 400, 1)
    assert r["prompt_tok_per_s"] == pytest.approx(4000 / 300)


def _store():
    s = MemoryStore("creative_designer")
    a = s.note("Budget for the localisation project: €100,000.", [Source("creative_designer", 89, 4)], At(89, 4))
    s.recall(a)
    child = s.keep(a, "Roughly €100k.", At(95, 2))
    b = s.note("Prefers modernist design.", [Source("creative_designer", 29, 2)], At(29, 2))
    s.forget([b], At(40, 1))
    return s, a, child


def test_store_summary_counts_live_memories_only():
    s, _, _ = _store()
    summary = store_summary(s)
    assert summary["live_memories"] == 2
    assert summary["depths"] == {1: 1, 2: 1}
    assert summary["max_depth"] == 2
    assert summary["retired"] == {"forgotten": 1}
    assert summary["total_recalls"] == 1
    assert summary["live_text_chars"] == len("Budget for the localisation project: €100,000.") + len("Roughly €100k.")


def test_summaries_saved_next_to_snapshots_and_newest_found(tmp_path):
    s, _, _ = _store()
    run = tmp_path / "runs" / "keep" / "creative_designer"
    out1 = save_summary(s, run / "session_0095.json")
    s.note("Another fact.", [Source("creative_designer", 96, 1)], At(96, 1))
    out2 = save_summary(s, run / "session_0096.json")
    import os, time
    os.utime(out1, (time.time() - 100, time.time() - 100))
    assert out2.name == "session_0096.summary.json"
    found = latest_summaries(tmp_path)
    assert len(found) == 1 and found[0][0] == out2 and found[0][1]["live_memories"] == 3


def test_dir_size(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text("12345")
    (tmp_path / "y.json").write_text("123")
    assert dir_size(tmp_path) == (8, 2)
    assert dir_size(tmp_path / "missing") == (0, 0)
