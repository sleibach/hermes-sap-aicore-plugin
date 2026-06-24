"""Hermes model-provider registration for SAP AI Core."""

from __future__ import annotations

import os


DEFAULT_PROXY_BASE_URL = "http://127.0.0.1:8765/v1"


def _load_env() -> None:
    try:
        from .config import _load_hermes_dotenv

        _load_hermes_dotenv()
    except Exception:
        pass


def _fallback_models() -> tuple[str, ...]:
    _load_env()
    configured = (
        os.getenv("SAP_AICORE_MODELS", "").strip()
        or os.getenv("SAP_AICORE_MODEL_NAME", "").strip()
    )
    models = [item.strip() for item in configured.replace(";", ",").split(",") if item.strip()]
    return tuple(models or ["sap-aicore-model"])


def register() -> None:
    """Register the SAP AI Core provider with Hermes.

    Hermes model-provider discovery imports user plugin directories in its own
    Python process. Keeping this function tiny avoids loading the proxy modules
    or reading credentials during provider discovery.
    """

    from providers import register_provider
    from providers.base import ProviderProfile

    _load_env()
    profile = ProviderProfile(
        name="sap-aicore",
        aliases=("aicore", "sap-ai-core", "generative-ai-hub", "genaihub"),
        display_name="SAP AI Core",
        description=(
            "SAP AI Core Generative AI Hub via local OpenAI-compatible proxy. "
            "In orchestration mode, use Hermes model id as the foundation model name."
        ),
        signup_url="https://help.sap.com/docs/sap-ai-core",
        env_vars=("SAP_AICORE_PROXY_KEY", "SAP_AICORE_PROXY_BASE_URL"),
        base_url=os.getenv("SAP_AICORE_PROXY_BASE_URL", "").strip() or DEFAULT_PROXY_BASE_URL,
        auth_type="api_key",
        supports_health_check=True,
        supports_vision=True,
        fallback_models=_fallback_models(),
        default_aux_model=(
            os.getenv("SAP_AICORE_AUX_DEPLOYMENT_ID", "").strip()
            or os.getenv("SAP_AICORE_DEPLOYMENT_ID", "").strip()
            or os.getenv("AICORE_DEPLOYMENT_ID", "").strip()
        ),
    )
    register_provider(profile)

