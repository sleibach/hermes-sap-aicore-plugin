from __future__ import annotations

from hermes_sap_aicore.config import AiCoreConfig
from hermes_sap_aicore.proxy import (
    _aicore_url,
    _deployment_from_model,
    _models_payload,
    _orchestration_payload,
    _orchestration_url,
)


def test_deployment_from_model_strips_alias_prefix():
    assert _deployment_from_model("sap-aicore:abc") == "abc"
    assert _deployment_from_model("sap-aicore/abc") == "abc"
    assert _deployment_from_model("aicore:abc") == "abc"
    assert _deployment_from_model("plain-deployment") == "plain-deployment"
    assert _deployment_from_model("sap-aicore-deployment") == ""


def test_aicore_url_normalizes_v2_and_encodes_deployment():
    config = AiCoreConfig(
        client_id="client",
        client_secret="secret",
        auth_url="https://auth.example.test",
        ai_api_url="https://api.example.test/v2",
        resource_group="default",
        deployment_id="deployment",
    )

    assert (
        _aicore_url(config, "deployment with spaces")
        == "https://api.example.test/v2/inference/deployments/deployment%20with%20spaces/chat/completions"
    )

    assert (
        _orchestration_url(config, "deployment with spaces")
        == "https://api.example.test/v2/inference/deployments/deployment%20with%20spaces/v2/completion"
    )


def test_models_payload_uses_configured_models(monkeypatch):
    monkeypatch.setenv("SAP_AICORE_MODELS", "dep-a,dep-b")

    payload = _models_payload()

    assert [item["id"] for item in payload["data"]] == ["dep-a", "dep-b"]


def test_orchestration_payload_maps_openai_request():
    payload = _orchestration_payload(
        {
            "model": "anthropic--claude-4.5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 20,
            "temperature": 0,
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
    )

    prompt_templating = payload["config"]["modules"]["prompt_templating"]
    assert prompt_templating["model"] == {
        "name": "anthropic--claude-4.5-sonnet",
        "params": {"max_tokens": 20, "temperature": 0},
    }
    assert prompt_templating["prompt"]["template"] == [{"role": "user", "content": "Hello"}]
    assert prompt_templating["prompt"]["tools"] == [{"type": "function", "function": {"name": "lookup"}}]
