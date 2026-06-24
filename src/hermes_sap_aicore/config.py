"""Configuration helpers for SAP AI Core service-key based access."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when the SAP AI Core plugin is not configured correctly."""


_DOTENV_LOADED = False


@dataclass(frozen=True)
class AiCoreConfig:
    client_id: str
    client_secret: str
    auth_url: str
    ai_api_url: str
    resource_group: str
    deployment_id: str
    api_version: str = ""
    client_type: str = "Hermes SAP AI Core Plugin"

    @property
    def token_url(self) -> str:
        value = self.auth_url.rstrip("/")
        if value.endswith("/oauth/token"):
            return value
        return f"{value}/oauth/token"

    @property
    def inference_base_url(self) -> str:
        value = self.ai_api_url.rstrip("/")
        return value if value.endswith("/v2") else f"{value}/v2"


def _first_string(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _unquote_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.replace('\\"', '"').replace("\\\\", "\\")


def _load_hermes_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    hermes_home = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()
    env_path = hermes_home / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = _unquote_env_value(value)


def _load_service_key() -> dict[str, Any]:
    raw_json = os.getenv("SAP_AICORE_SERVICE_KEY_JSON", "").strip()
    if raw_json:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"SAP_AICORE_SERVICE_KEY_JSON is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("SAP_AICORE_SERVICE_KEY_JSON must contain a JSON object")
        return data

    path_value = (
        os.getenv("SAP_AICORE_SERVICE_KEY")
        or os.getenv("SAP_AICORE_SERVICE_KEY_FILE")
        or os.getenv("AICORE_SERVICE_KEY")
        or ""
    ).strip()
    if not path_value:
        return {}

    path = Path(path_value).expanduser()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"SAP AI Core service key file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"SAP AI Core service key file is not valid JSON: {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"SAP AI Core service key file must contain a JSON object: {path}")
    return data


def load_config(model_deployment_id: str = "") -> AiCoreConfig:
    _load_hermes_dotenv()
    service_key = _load_service_key()
    service_urls = service_key.get("serviceurls") if isinstance(service_key.get("serviceurls"), dict) else {}

    client_id = os.getenv("AICORE_CLIENT_ID", "").strip() or _first_string(service_key, "clientid", "client_id")
    client_secret = (
        os.getenv("AICORE_CLIENT_SECRET", "").strip()
        or _first_string(service_key, "clientsecret", "client_secret")
    )
    auth_url = os.getenv("AICORE_AUTH_URL", "").strip() or _first_string(service_key, "url", "auth_url")
    ai_api_url = (
        os.getenv("AICORE_BASE_URL", "").strip()
        or os.getenv("SAP_AICORE_BASE_URL", "").strip()
        or _first_string(service_urls, "AI_API_URL", "AI_API_URL_V2")
        or _first_string(service_key, "AI_API_URL", "ai_api_url")
    )
    resource_group = (
        os.getenv("AICORE_RESOURCE_GROUP", "").strip()
        or os.getenv("SAP_AICORE_RESOURCE_GROUP", "").strip()
        or _first_string(service_key, "resource_group", "resourceGroup")
        or "default"
    )
    deployment_id = (
        model_deployment_id.strip()
        or os.getenv("SAP_AICORE_DEPLOYMENT_ID", "").strip()
        or os.getenv("AICORE_DEPLOYMENT_ID", "").strip()
    )
    api_version = os.getenv("SAP_AICORE_API_VERSION", "").strip()
    client_type = os.getenv("AI_CLIENT_TYPE", "").strip() or "Hermes SAP AI Core Plugin"

    missing = [
        name
        for name, value in {
            "clientid/AICORE_CLIENT_ID": client_id,
            "clientsecret/AICORE_CLIENT_SECRET": client_secret,
            "url/AICORE_AUTH_URL": auth_url,
            "serviceurls.AI_API_URL/AICORE_BASE_URL": ai_api_url,
            "SAP_AICORE_DEPLOYMENT_ID or Hermes model": deployment_id,
        }.items()
        if not value
    ]
    if missing:
        raise ConfigError("Missing SAP AI Core configuration: " + ", ".join(missing))

    return AiCoreConfig(
        client_id=client_id,
        client_secret=client_secret,
        auth_url=auth_url,
        ai_api_url=ai_api_url,
        resource_group=resource_group,
        deployment_id=deployment_id,
        api_version=api_version,
        client_type=client_type,
    )
