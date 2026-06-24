from __future__ import annotations

import pytest

from hermes_sap_aicore.installer import configure_config_yaml, install


def test_install_writes_model_provider_drop_in(tmp_path):
    provider_dir = install(
        tmp_path,
        write_env=True,
        env_values={
            "SAP_AICORE_SERVICE_KEY": "/tmp/key.json",
            "SAP_AICORE_DEPLOYMENT_ID": "deployment",
            "SAP_AICORE_MODEL_NAME": "anthropic--claude-4.5-sonnet",
            "SAP_AICORE_API_MODE": "orchestration",
            "SAP_AICORE_PROXY_KEY": "local",
        },
    )

    assert provider_dir == tmp_path / "plugins" / "model-providers" / "sap-aicore"
    assert (provider_dir / "__init__.py").exists()
    assert "kind: model-provider" in (provider_dir / "plugin.yaml").read_text(encoding="utf-8")

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert 'SAP_AICORE_SERVICE_KEY="/tmp/key.json"' in env_text
    assert 'SAP_AICORE_DEPLOYMENT_ID="deployment"' in env_text
    assert 'SAP_AICORE_MODEL_NAME="anthropic--claude-4.5-sonnet"' in env_text
    assert 'SAP_AICORE_API_MODE="orchestration"' in env_text
    assert 'SAP_AICORE_PROXY_KEY="local"' in env_text


def test_configure_config_yaml_registers_provider(tmp_path):
    yaml = pytest.importorskip("yaml")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"model": {"provider": "sap-aicore", "base_url": "https://openrouter.ai/api/v1"}, "providers": {}}
        ),
        encoding="utf-8",
    )

    assert configure_config_yaml(config_path, proxy_url="http://127.0.0.1:8765/v1") is True

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entry = data["providers"]["sap-aicore"]
    assert entry["base_url"] == "http://127.0.0.1:8765/v1"
    assert entry["key_env"] == "SAP_AICORE_PROXY_KEY"
    assert entry["transport"] == "openai_chat"
    assert data["model"]["base_url"] == "http://127.0.0.1:8765/v1"
    assert data["agent"]["tool_use_enforcement"] == [
        "anthropic--",
        "gpt",
        "codex",
        "gemini",
        "grok",
        "qwen",
        "deepseek",
    ]
    # Existing config is preserved and a backup is written.
    assert data["model"]["provider"] == "sap-aicore"
    assert (tmp_path / "config.yaml.bak").exists()


def test_install_write_config_creates_provider_entry(tmp_path):
    pytest.importorskip("yaml")
    install(
        tmp_path,
        write_env=False,
        env_values={},
        write_config=True,
        proxy_url="http://127.0.0.1:8765/v1",
    )
    import yaml

    data = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert "sap-aicore" in data["providers"]
