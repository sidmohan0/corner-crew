import asyncio
import json
from importlib import import_module

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.config import Settings
from app.main import create_app

CLASSIFICATION = {
    "privacy": "public",
    "complexity": "low",
    "high_risk_domain": False,
    "factual_claims": False,
    "code_generation": False,
}
GOOD = {"confidence": 0.95, "passed": True, "reason": "Answers the request."}
BAD = {"confidence": 0.4, "passed": False, "reason": "Uncertain."}


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        openai_api_key="test-openai",
        gemini_api_key="test-gemini",
        anthropic_api_key="test-anthropic",
        local_api_key="local",
        minicpm_api_key="local",
        openai_model="openai-test",
        gemini_model="gemini-test",
        anthropic_model="anthropic-test",
        local_model="local-test",
        minicpm_model="minicpm-test",
        local_base_url="http://127.0.0.1:8080/v1/",
        minicpm_base_url="http://127.0.0.1:8081/v1/",
        local_data_residency="US",
        memory_path=str(tmp_path / "memory.sqlite3"),
        model_metadata={
            p: {
                "model": f"{p}-test",
                "region": "US",
                "confidential_approved": True,
                "input_usd_per_million": 1,
                "output_usd_per_million": 2,
            }
            for p in ["openai", "gemini", "anthropic"]
        },
    )


def install_upstream(monkeypatch, callback=None):
    calls = []

    async def handler(request):
        body = json.loads(request.content)
        stage = body.get("response_format", {}).get("json_schema", {}).get("name", "execute")
        if stage == "execute" and body["messages"][0]["content"].startswith("Evaluate"):
            stage = "evaluate"
        calls.append((stage, body, request))
        content = (
            CLASSIFICATION
            if stage == "classify"
            else GOOD
            if stage in {"evaluate", "verify_independent"}
            else "Hello!"
        )
        if callback:
            override = callback(stage, body, request)
            if asyncio.iscoroutine(override):
                override = await override
            if override is not None:
                content = override
        if isinstance(content, httpx.Response):
            return content
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content)
                            if isinstance(content, dict)
                            else content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            },
        )

    def factory(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr(import_module("app.main"), "AsyncOpenAI", factory)
    return calls


def payload(**kwargs):
    return {"messages": [{"role": "user", "content": "Say hello."}], **kwargs}


def post(settings, body):
    with TestClient(create_app(settings)) as client:
        return client.post("/chat", json=body)


def test_auto_pipeline(monkeypatch, settings):
    calls = install_upstream(monkeypatch)
    response = post(settings, payload())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["provider"] == "minicpm"
    assert data["status"] == "ok"
    assert data["usage"]["model_calls"] == 3
    assert [s for s, _, _ in calls] == ["classify", "execute", "evaluate"]
    assert data["content"] == "Hello!"


@pytest.mark.parametrize(
    "provider,host",
    [
        ("openai", "api.openai.com"),
        ("google", "generativelanguage.googleapis.com"),
        ("anthropic", "api.anthropic.com"),
        ("local", "127.0.0.1"),
        ("minicpm", "127.0.0.1"),
    ],
)
def test_explicit_provider_still_classifies(monkeypatch, settings, provider, host):
    calls = install_upstream(monkeypatch)
    result = post(settings, payload(provider=provider))
    assert result.status_code == 200, result.text
    assert calls[0][0] == "classify"
    assert calls[1][2].url.host == host


@pytest.mark.parametrize(
    "privacy,approved,status",
    [("confidential", True, 200), ("confidential", False, 403), ("restricted", True, 403)],
)
def test_privacy_enforced_before_cloud(monkeypatch, settings, privacy, approved, status):
    settings.model_metadata["openai"]["confidential_approved"] = approved
    calls = install_upstream(
        monkeypatch,
        lambda stage, *_: {**CLASSIFICATION, "privacy": privacy} if stage == "classify" else None,
    )
    result = post(settings, payload(provider="openai"))
    assert result.status_code == status
    if status == 403:
        assert len(calls) == 1


def test_explicit_privacy_cannot_be_downgraded(monkeypatch, settings):
    settings.model_metadata["openai"]["confidential_approved"] = False
    calls = install_upstream(monkeypatch)
    result = post(settings, payload(provider="openai", policy={"privacy": "confidential"}))
    assert result.status_code == 403
    assert len(calls) == 1


def test_unknown_classifier_residency_rejects_without_calls(monkeypatch, settings):
    settings.local_data_residency = None
    calls = install_upstream(monkeypatch)
    result = post(settings, payload(policy={"data_residency": "US"}))
    assert result.status_code == 403
    assert not calls


def test_unknown_cloud_region_and_disallowed_provider(monkeypatch, settings):
    calls = install_upstream(monkeypatch)
    settings.model_metadata["openai"]["region"] = None
    assert (
        post(settings, payload(provider="openai", policy={"data_residency": "US"})).status_code
        == 403
    )
    assert (
        post(settings, payload(provider="openai", policy={"providers": ["local"]})).status_code
        == 403
    )
    assert all(stage == "classify" for stage, _, _ in calls)


def test_remote_classifier_is_not_trusted_as_local(monkeypatch, settings):
    settings.minicpm_base_url = "https://example.com/v1/"
    calls = install_upstream(monkeypatch)
    assert post(settings, payload()).status_code == 503
    assert not calls


def test_malformed_classifier_fails_closed(monkeypatch, settings):
    calls = install_upstream(
        monkeypatch, lambda stage, *_: "not json" if stage == "classify" else None
    )
    assert post(settings, payload(provider="openai")).status_code == 503
    assert len(calls) == 1


@pytest.mark.parametrize(
    "limit,action,status", [(1, "best_effort", 429), (2, "best_effort", 200), (2, "reject", 429)]
)
def test_call_budget_includes_control_calls(monkeypatch, settings, limit, action, status):
    calls = install_upstream(monkeypatch)
    result = post(settings, payload(budget={"max_model_calls": limit, "on_exceeded": action}))
    assert result.status_code == status, result.text
    assert len(calls) == limit
    if status == 200:
        assert result.json()["status"] == "uncertain"
        assert result.json()["confidence"] is None


def test_zero_cost_budget_never_calls_paid_provider(monkeypatch, settings):
    calls = install_upstream(monkeypatch)
    result = post(settings, payload(provider="openai", budget={"max_total_cost_usd": 0}))
    assert result.status_code == 429
    assert len(calls) == 1


def test_unknown_prices_excluded(monkeypatch, settings):
    settings.model_metadata["openai"]["input_usd_per_million"] = None
    calls = install_upstream(monkeypatch)
    assert post(settings, payload(provider="openai")).status_code == 503
    assert len(calls) == 1


def test_wall_deadline_returns_best_answer(monkeypatch, settings):
    async def callback(stage, *_):
        if stage == "evaluate":
            await asyncio.sleep(0.2)

    calls = install_upstream(monkeypatch, callback)
    result = post(settings, payload(budget={"max_wall_time_ms": 80}))
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "uncertain"
    assert result.json()["usage"]["elapsed_ms"] < 180
    assert len(calls) == 3


def test_model_errors_are_sanitized_and_fallback_is_bounded(monkeypatch, settings):
    def callback(stage, body, request):
        if stage == "execute" and body["model"] == "minicpm-test":
            raise httpx.ConnectError("sensitive-secret", request=request)

    calls = install_upstream(monkeypatch, callback)
    result = post(settings, payload())
    assert result.status_code == 200
    assert result.json()["provider"] == "local"
    assert result.json()["usage"]["model_calls"] == 4
    assert len(calls) == 4
    assert "sensitive-secret" not in result.text


def test_low_confidence_escalates(monkeypatch, settings):
    reviews = 0

    def callback(stage, *_):
        nonlocal reviews
        if stage == "evaluate":
            reviews += 1
            return BAD if reviews == 1 else GOOD

    calls = install_upstream(monkeypatch, callback)
    result = post(settings, payload(verification={"mode": "off"}, budget={"max_model_calls": 6}))
    assert result.status_code == 200, result.text
    assert result.json()["provider"] == "local"
    assert any(t["stage"] == "escalate" for t in result.json()["trace"])
    assert len(calls) == 5


def test_three_family_majority(monkeypatch, settings):
    calls = install_upstream(monkeypatch, lambda stage, *_: BAD if stage == "evaluate" else None)
    result = post(
        settings,
        payload(
            compute={"escalate_on": []},
            verification={"strategies": ["consensus"]},
            budget={"max_model_calls": 8},
        ),
    )
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "ok"
    event = next(t for t in result.json()["trace"] if t["stage"] == "consensus_result")
    assert event["total"] == 3
    assert len(calls) == 5


def test_consensus_never_invents_independence(monkeypatch, settings):
    install_upstream(monkeypatch, lambda stage, *_: BAD if stage == "evaluate" else None)
    result = post(
        settings,
        payload(
            policy={"providers": ["local"]},
            compute={"escalate_on": []},
            verification={"strategies": ["consensus"]},
            budget={"max_model_calls": 8},
        ),
    )
    assert result.status_code == 200
    assert result.json()["status"] == "uncertain"
    assert any("Insufficient independent" in w for w in result.json()["uncertainty"])


def test_disagreement_rejects(monkeypatch, settings):
    def callback(stage, body, _):
        if stage == "evaluate":
            return BAD
        if stage == "execute":
            return body["model"]

    install_upstream(monkeypatch, callback)
    result = post(
        settings,
        payload(
            compute={"escalate_on": []},
            verification={"strategies": ["consensus"]},
            consensus={"disagreement": "reject"},
            budget={"max_model_calls": 8},
        ),
    )
    assert result.status_code == 422


def test_json_schema_and_python_syntax_checks(monkeypatch, settings):
    install_upstream(
        monkeypatch, lambda stage, *_: '{"count":"wrong"}' if stage == "execute" else None
    )
    result = post(
        settings,
        payload(
            compute={"escalate_on": []},
            response={
                "format": "json",
                "json_schema": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                },
            },
            verification={"failure_action": "reject"},
        ),
    )
    assert result.status_code == 422
    install_upstream(
        monkeypatch,
        lambda stage, *_: "```python\ndef broken(:\n```" if stage == "execute" else None,
    )
    result = post(
        settings,
        payload(
            compute={"escalate_on": []},
            verification={
                "mode": "always",
                "strategies": ["tool_check"],
                "failure_action": "reject",
            },
        ),
    )
    assert result.status_code == 422


def test_memory_isolated_persistent_and_classified_before_routing(monkeypatch, settings):
    calls = install_upstream(monkeypatch)
    body = payload(
        user_id="alice",
        memory={"write": "auto"},
        messages=[{"role": "user", "content": "My favorite color is teal."}],
    )
    assert post(settings, body).json()["status"] == "ok"
    calls.clear()
    post(settings, payload(user_id="bob", memory={"read": True}))
    assert "teal" not in calls[0][1]["messages"][1]["content"]
    calls.clear()
    post(settings, payload(user_id="alice", memory={"read": True}))
    assert "teal" in calls[0][1]["messages"][1]["content"]


@pytest.mark.parametrize(
    "changes",
    [
        {"budget": {"max_model_calls": 0}},
        {"compute": {"initial_tier": "frontier", "max_tier": "cheap"}},
        {"memory": {"read": True}},
        {"budget": {"typo": True}},
        {"response": {"format": "json", "json_schema": {"$ref": "https://example.com/schema"}}},
    ],
)
def test_invalid_controls_rejected_without_calls(monkeypatch, settings, changes):
    calls = install_upstream(monkeypatch)
    assert post(settings, payload(**changes)).status_code == 422
    assert not calls


def test_catalog_and_docs(monkeypatch, settings):
    install_upstream(monkeypatch)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert len(client.get("/models").json()) == 5
        assert "test-openai" not in client.get("/providers").text
        schema = client.get("/openapi.json").json()["components"]["schemas"]["ChatRequest"][
            "properties"
        ]
        assert {
            "policy",
            "budget",
            "compute",
            "verification",
            "consensus",
            "memory",
            "routing",
            "response",
        } <= schema.keys()


def test_full_requested_envelope_and_memory_privacy(monkeypatch, settings):
    from pathlib import Path

    body = json.loads(Path("examples/adaptive-chat.json").read_text())
    calls = install_upstream(monkeypatch)
    response = post(settings, body)
    assert response.status_code == 200, response.text
    assert response.json()["classification"]["privacy"] == "public"
    privacy_event = next(t for t in response.json()["trace"] if t["stage"] == "privacy_policy")
    assert privacy_event["effective_privacy"] == "confidential"
    settings.model_metadata["openai"]["confidential_approved"] = False
    calls.clear()
    response = post(
        settings, payload(user_id="demo-user", provider="openai", memory={"read": True})
    )
    assert response.status_code == 403
    assert len(calls) == 1


def test_effort_and_remaining_token_caps(monkeypatch, settings):
    calls = install_upstream(monkeypatch)
    result = post(settings, payload(budget={"max_reasoning_tokens": 50}))
    assert result.status_code == 429
    assert calls[0][1]["max_tokens"] == 50
    assert calls[1][1]["max_tokens"] == 20
    assert calls[0][1]["chat_template_kwargs"]["enable_thinking"] is False
    assert len(calls) == 2


def test_budget_cannot_bypass_reject_verification(monkeypatch, settings):
    install_upstream(monkeypatch)
    response = post(
        settings, payload(budget={"max_model_calls": 2}, verification={"failure_action": "reject"})
    )
    assert response.status_code == 422


def test_escalation_reject_on_insufficient_budget(monkeypatch, settings):
    install_upstream(monkeypatch, lambda stage, *_: BAD if stage == "evaluate" else None)
    response = post(
        settings,
        payload(
            budget={"max_model_calls": 3},
            verification={"mode": "off"},
            compute={"on_escalation_failure": "reject"},
        ),
    )
    assert response.status_code == 422


def test_soft_latency_stops_optional_calls(monkeypatch, settings):
    async def callback(stage, *_):
        if stage == "classify":
            await asyncio.sleep(0.03)

    calls = install_upstream(monkeypatch, callback)
    response = post(settings, payload(budget={"max_latency_ms": 10}))
    assert response.status_code == 200
    assert response.json()["status"] == "uncertain"
    assert len(calls) == 2


def test_no_duplicate_family_consensus(monkeypatch, settings):
    settings.model_metadata["openai"]["family"] = "qwen3-vl"
    settings.model_metadata["gemini"]["family"] = "qwen3-vl"
    settings.model_metadata["anthropic"]["family"] = "qwen3-vl"
    calls = install_upstream(monkeypatch, lambda stage, *_: BAD if stage == "evaluate" else None)
    response = post(
        settings,
        payload(
            compute={"escalate_on": []},
            verification={"strategies": ["consensus"]},
            budget={"max_model_calls": 8},
        ),
    )
    assert response.json()["status"] == "uncertain"
    assert len(calls) == 4


def test_consensus_reject_is_not_bypassed_by_best_effort_budget(monkeypatch, settings):
    install_upstream(monkeypatch, lambda stage, *_: BAD if stage == "evaluate" else None)
    response = post(
        settings,
        payload(
            compute={"escalate_on": []},
            verification={"strategies": ["consensus"]},
            consensus={"disagreement": "reject"},
            budget={"max_model_calls": 4},
        ),
    )
    assert response.status_code == 422


def test_invalid_evaluator_returns_uncertainty_not_fake_confidence(monkeypatch, settings):
    install_upstream(monkeypatch, lambda stage, *_: "not JSON" if stage == "evaluate" else None)
    response = post(settings, payload())
    assert response.status_code == 200
    assert response.json()["confidence"] is None
    assert response.json()["status"] == "uncertain"


def test_stream_emits_real_trace_then_result(monkeypatch, settings):
    install_upstream(monkeypatch)
    with TestClient(create_app(settings)) as client:
        response = client.post("/chat/stream", json=payload())
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "trace"
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["content"] == "Hello!"
    stages = [e["event"]["stage"] for e in events if e["type"] == "trace"]
    assert stages.index("classify") < stages.index("classify_complete") < stages.index("execute")
    assert events[-1]["data"]["trace"] == [e["event"] for e in events if e["type"] == "trace"]


def test_stream_policy_rejection_is_terminal(monkeypatch, settings):
    install_upstream(monkeypatch)
    settings.local_data_residency = None
    with TestClient(create_app(settings)) as client:
        response = client.post("/chat/stream", json=payload(policy={"data_residency": "US"}))
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "error"
    assert events[-1]["status"] == 403
    assert not any(e["type"] == "result" for e in events)
