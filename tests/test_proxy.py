from __future__ import annotations

import hermes_sap_aicore.proxy as proxy
from hermes_sap_aicore.config import AiCoreConfig
from hermes_sap_aicore.proxy import (
    _aicore_url,
    _as_openai_sse,
    _active_mode,
    _deployment_from_model,
    _deployment_model_name,
    _is_chat_model,
    _models_payload,
    _orchestration_messages,
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


def _isolate_env(monkeypatch, tmp_path):
    """Keep _models_payload hermetic: empty HERMES_HOME, no live network."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("SAP_AICORE_MODELS", raising=False)
    monkeypatch.delenv("SAP_AICORE_MODEL_NAME", raising=False)


def test_models_payload_pins_explicit_models(monkeypatch, tmp_path):
    _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SAP_AICORE_MODELS", "dep-a,dep-b")

    payload = _models_payload()

    assert [item["id"] for item in payload["data"]] == ["dep-a", "dep-b"]


def test_models_payload_uses_live_listing(monkeypatch, tmp_path):
    _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(proxy, "_live_models", lambda: ["gpt-5.5", "anthropic--claude-4.5-sonnet"])

    payload = _models_payload()

    assert [item["id"] for item in payload["data"]] == ["gpt-5.5", "anthropic--claude-4.5-sonnet"]


def test_models_payload_falls_back_to_model_name(monkeypatch, tmp_path):
    _isolate_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SAP_AICORE_MODEL_NAME", "anthropic--claude-4.5-sonnet")

    def _boom():
        raise RuntimeError("no network")

    monkeypatch.setattr(proxy, "_live_models", _boom)

    payload = _models_payload()

    assert [item["id"] for item in payload["data"]] == ["anthropic--claude-4.5-sonnet"]


def test_is_chat_model_filters_embeddings_and_rpt():
    assert _is_chat_model("gpt-5.5")
    assert _is_chat_model("anthropic--claude-4.5-sonnet")
    assert not _is_chat_model("text-embedding-3-small")
    assert not _is_chat_model("nvidia--llama-3.2-nv-embedqa-1b")
    assert not _is_chat_model("sap-rpt-1-large")


def test_deployment_model_name_reads_backend_details():
    deployment = {
        "id": "dep1",
        "details": {"resources": {"backendDetails": {"model": {"name": "gpt-5.5", "version": "x"}}}},
    }
    assert _deployment_model_name(deployment) == "gpt-5.5"


def test_active_mode(monkeypatch):
    monkeypatch.setenv("SAP_AICORE_API_MODE", "orchestration")
    assert _active_mode() == "orchestration"
    monkeypatch.setenv("SAP_AICORE_API_MODE", "foundation")
    assert _active_mode() == "foundation"


def test_orchestration_payload_request_model_wins_over_env(monkeypatch):
    monkeypatch.setenv("SAP_AICORE_MODEL_NAME", "anthropic--claude-4.5-sonnet")
    payload = _orchestration_payload(
        {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 20,
            "temperature": 0,
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
    )

    prompt_templating = payload["config"]["modules"]["prompt_templating"]
    assert prompt_templating["model"] == {
        "name": "gpt-5.5",
        "params": {"max_tokens": 20, "temperature": 0},
    }
    assert prompt_templating["prompt"]["template"] == [{"role": "user", "content": "Hello"}]
    assert prompt_templating["prompt"]["tools"] == [{"type": "function", "function": {"name": "lookup"}}]


def test_orchestration_payload_falls_back_to_env_when_model_placeholder(monkeypatch):
    monkeypatch.setenv("SAP_AICORE_MODEL_NAME", "anthropic--claude-4.5-sonnet")
    payload = _orchestration_payload(
        {"model": "sap-aicore-model", "messages": [{"role": "user", "content": "Hi"}]}
    )
    assert payload["config"]["modules"]["prompt_templating"]["model"]["name"] == "anthropic--claude-4.5-sonnet"


def test_orchestration_messages_convert_tool_results_to_system_messages():
    messages = _orchestration_messages(
        [
            {"role": "user", "content": "Where am I?"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {
                "role": "assistant",
                "content": "I'll check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{\"command\":\"pwd\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "/Users/soeren.leibach"},
        ]
    )

    assert len(messages) == 3
    assert "tool_calls" not in messages[1]
    assert messages[-1] == {
        "role": "user",
        "content": "Tool result (call_1):\n/Users/soeren.leibach",
    }


def test_openai_sse_preserves_tool_calls():
    body = b"""{
      "final_result": {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "anthropic--claude-4.7-opus",
        "choices": [{
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I'll inspect the project.",
            "tool_calls": [{
              "id": "call_123",
              "type": "function",
              "function": {
                "name": "terminal",
                "arguments": "{\\"command\\": \\"pwd\\"}"
              }
            }]
          },
          "finish_reason": "tool_calls"
        }]
      }
    }"""

    sse = _as_openai_sse(body).decode("utf-8")

    assert "\"content\": \"I'll inspect the project.\"" in sse
    assert '"tool_calls"' in sse
    assert '"id": "call_123"' in sse
    assert '"name": "terminal"' in sse
    assert '"arguments": "{\\"command\\": \\"pwd\\"}"' in sse
    assert '"finish_reason": "tool_calls"' in sse
