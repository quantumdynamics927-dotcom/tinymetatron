"""
tests/test_api.py
================
ECOSYSTEM pytest suite for the FastAPI surface (contract section 2).

Covers (contract section 5 / test descriptions):
    * GET  /data/stats  -> 200
    * GET  /model/info  -> returns config
    * POST /data/add    -> 200 with added count (private-training mode + key)
    * POST /generate    -> 200 (untrained model is fine) returns
      text + tokens_generated
    * GET  /train/status -> 200 with status fields
    * Deploy-mode gating: /data/add and /train/start return 403 in demo mode
      and when the X-API-Key is missing in private mode.

The API DB path is steered to a temp file via the TMT_DB_PATH env var so the
real metatron.db is never touched (contract rule 0.6). Deploy mode + API key
are read at request time by api.py, so the fixtures set them via monkeypatch.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

# Import the app after the env var is set in the fixture below.
from fastapi.testclient import TestClient


# Texts long enough to clear the quality threshold (>=0.4).
_ADD_TEXTS = [
    "The firewall enforces a zero-trust policy with mfa and tls encryption.",
    "Sparse attention masks reduce the quadratic cost of self-attention.",
    "Mixture of experts routes tokens to specialized feed forward networks.",
    "The transformer model uses attention, embeddings and a softmax over logits.",
]

_ADMIN_HEADERS = {"X-API-Key": "testkey"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Private-training mode client: admin endpoints enabled with a test key."""
    db_path = str(tmp_path / "api_test.db")
    monkeypatch.setenv("TMT_DB_PATH", db_path)
    monkeypatch.setenv("TMT_DEPLOY_MODE", "private-training")
    monkeypatch.setenv("TMT_API_KEY", "testkey")
    import api as api_mod
    with TestClient(api_mod.app) as c:
        yield c


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    """Demo mode client: admin endpoints disabled regardless of key."""
    db_path = str(tmp_path / "api_demo.db")
    monkeypatch.setenv("TMT_DB_PATH", db_path)
    monkeypatch.setenv("TMT_DEPLOY_MODE", "demo")
    monkeypatch.setenv("TMT_API_KEY", "testkey")
    import api as api_mod
    with TestClient(api_mod.app) as c:
        yield c


# ── GET /health -> 200 ────────────────────────────────────────────────────────
def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200, f"status {r.status_code}: {r.text}"
    assert r.json() == {"status": "ok"}


# ── GET /data/stats -> 200, contract keys present ─────────────────────────────
def test_data_stats_ok(client: TestClient) -> None:
    r = client.get("/data/stats")
    assert r.status_code == 200, f"status {r.status_code}: {r.text}"
    body = r.json()
    for k in ("total", "by_domain", "avg_quality", "used_in_training"):
        assert k in body, f"/data/stats missing key {k}"


# ── GET /model/info -> returns config ─────────────────────────────────────────
def test_model_info_returns_config(client: TestClient) -> None:
    r = client.get("/model/info")
    assert r.status_code == 200, f"status {r.status_code}: {r.text}"
    info = r.json()
    assert "config" in info and isinstance(info["config"], dict)
    from config import CONFIG
    assert info["config"].get("vocab_size") == CONFIG["vocab_size"]
    assert "active_checkpoint" in info


# ── POST /data/add -> 200 with added count (private mode + key) ────────────────
def test_data_add_returns_added_count(client: TestClient) -> None:
    r = client.post("/data/add", json={
        "texts": _ADD_TEXTS, "domain": "cybersecurity",
        "quality_threshold": 0.4,
    }, headers=_ADMIN_HEADERS)
    assert r.status_code == 200, f"status {r.status_code}: {r.text}"
    body = r.json()
    assert body["added"] >= 1, f"added>=1, got {body['added']}"
    assert body["added"] + body["rejected"] == len(_ADD_TEXTS)
    assert body["domain"] == "cybersecurity"


# ── POST /generate -> 200, returns text + tokens_generated (untrained ok) ─────
def test_generate_returns_text_and_tokens(client: TestClient) -> None:
    r = client.post("/generate", json={
        "prompt": "firewall ", "max_length": 8, "temperature": 0.7,
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text}"
    gen = r.json()
    assert "text" in gen and isinstance(gen["text"], str)
    assert "tokens_generated" in gen
    assert isinstance(gen["tokens_generated"], int)
    assert gen["tokens_generated"] >= 0
    # step==0 for an untrained model (no checkpoint in a fresh temp DB)
    assert gen.get("step") == 0


# ── /generate input bounds: prompt over 2000 chars -> 422 ─────────────────────
def test_generate_rejects_oversized_prompt(client: TestClient) -> None:
    r = client.post("/generate", json={
        "prompt": "x" * 2001, "max_length": 4, "temperature": 0.7,
    })
    assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text}"


# ── /generate output bound: max_length over 256 -> 422 ────────────────────────
def test_generate_rejects_oversized_max_length(client: TestClient) -> None:
    r = client.post("/generate", json={
        "prompt": "ok", "max_length": 257, "temperature": 0.7,
    })
    assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text}"


# ── GET /train/status -> 200 with status fields ───────────────────────────────
def test_train_status_ok(client: TestClient) -> None:
    r = client.get("/train/status")
    assert r.status_code == 200, f"status {r.status_code}: {r.text}"
    st = r.json()
    for k in ("is_training", "current_step", "total_steps", "current_loss"):
        assert k in st, f"/train/status missing {k}"
    assert st["is_training"] is False


# ── Admin gate: private mode but no key -> 403 ────────────────────────────────
def test_data_add_requires_key(client: TestClient) -> None:
    r = client.post("/data/add", json={
        "texts": _ADD_TEXTS, "domain": "cybersecurity",
        "quality_threshold": 0.4,
    })  # no X-API-Key header
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"


def test_data_add_wrong_key(client: TestClient) -> None:
    r = client.post("/data/add", json={
        "texts": _ADD_TEXTS, "domain": "cybersecurity",
        "quality_threshold": 0.4,
    }, headers={"X-API-Key": "wrong"})
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"


# ── Demo mode: admin endpoints disabled even with a valid key -> 403 ──────────
def test_data_add_forbidden_in_demo(demo_client: TestClient) -> None:
    r = demo_client.post("/data/add", json={
        "texts": _ADD_TEXTS, "domain": "cybersecurity",
        "quality_threshold": 0.4,
    }, headers=_ADMIN_HEADERS)
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"


def test_train_start_forbidden_in_demo(demo_client: TestClient) -> None:
    r = demo_client.post("/train/start", json={
        "steps": 1, "domain": "general", "min_quality": 0.4,
    }, headers=_ADMIN_HEADERS)
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"


# ── Demo mode: read-only endpoints still public ──────────────────────────────
def test_readonly_endpoints_ok_in_demo(demo_client: TestClient) -> None:
    assert demo_client.get("/health").status_code == 200
    assert demo_client.get("/data/stats").status_code == 200
    assert demo_client.get("/model/info").status_code == 200
    assert demo_client.get("/train/status").status_code == 200
    assert demo_client.post("/generate", json={
        "prompt": "hi", "max_length": 4, "temperature": 0.7,
    }).status_code == 200