from __future__ import annotations

import json

from hermes_sap_aicore.config import load_config


def test_load_config_from_service_key_file(tmp_path, monkeypatch):
    service_key = {
        "clientid": "client",
        "clientsecret": "secret",
        "url": "https://auth.example.test",
        "serviceurls": {"AI_API_URL": "https://api.example.test"},
    }
    key_path = tmp_path / "key.json"
    key_path.write_text(json.dumps(service_key), encoding="utf-8")

    monkeypatch.setenv("SAP_AICORE_SERVICE_KEY", str(key_path))
    monkeypatch.setenv("SAP_AICORE_DEPLOYMENT_ID", "deployment-1")
    monkeypatch.setenv("SAP_AICORE_RESOURCE_GROUP", "team-a")

    config = load_config()

    assert config.client_id == "client"
    assert config.client_secret == "secret"
    assert config.token_url == "https://auth.example.test/oauth/token"
    assert config.inference_base_url == "https://api.example.test/v2"
    assert config.resource_group == "team-a"
    assert config.deployment_id == "deployment-1"


def test_model_deployment_overrides_env(tmp_path, monkeypatch):
    service_key = {
        "clientid": "client",
        "clientsecret": "secret",
        "url": "https://auth.example.test/oauth/token",
        "serviceurls": {"AI_API_URL": "https://api.example.test/v2"},
    }
    key_path = tmp_path / "key.json"
    key_path.write_text(json.dumps(service_key), encoding="utf-8")

    monkeypatch.setenv("SAP_AICORE_SERVICE_KEY", str(key_path))
    monkeypatch.setenv("SAP_AICORE_DEPLOYMENT_ID", "fallback")

    config = load_config("from-hermes-model")

    assert config.token_url == "https://auth.example.test/oauth/token"
    assert config.inference_base_url == "https://api.example.test/v2"
    assert config.deployment_id == "from-hermes-model"
