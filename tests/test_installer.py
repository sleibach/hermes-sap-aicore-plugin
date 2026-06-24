from __future__ import annotations

from hermes_sap_aicore.installer import install


def test_install_writes_model_provider_drop_in(tmp_path):
    provider_dir = install(
        tmp_path,
        write_env=True,
        env_values={
            "SAP_AICORE_SERVICE_KEY": "/tmp/key.json",
            "SAP_AICORE_DEPLOYMENT_ID": "deployment",
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
    assert 'SAP_AICORE_API_MODE="orchestration"' in env_text
    assert 'SAP_AICORE_PROXY_KEY="local"' in env_text
