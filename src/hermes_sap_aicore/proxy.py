"""Local OpenAI-compatible proxy for SAP AI Core Generative AI Hub."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import AiCoreConfig, ConfigError, load_config

LOGGER = logging.getLogger("hermes_sap_aicore.proxy")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class TokenCache:
    """Small thread-safe OAuth client-credentials token cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = ""
        self._expires_at = 0.0

    def get(self, config: AiCoreConfig) -> str:
        with self._lock:
            now = time.time()
            if self._token and now < self._expires_at:
                return self._token

            token, expires_in = _request_token(config, use_basic_auth=True)
            self._token = token
            self._expires_at = now + max(60, expires_in - 60)
            return self._token

    def clear(self) -> None:
        with self._lock:
            self._token = ""
            self._expires_at = 0.0


TOKEN_CACHE = TokenCache()


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)


def _request_token(config: AiCoreConfig, *, use_basic_auth: bool) -> tuple[str, int]:
    form: dict[str, str] = {"grant_type": "client_credentials"}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    if use_basic_auth:
        raw = f"{config.client_id}:{config.client_secret}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
    else:
        form["client_id"] = config.client_id
        form["client_secret"] = config.client_secret

    body = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(config.token_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if use_basic_auth and exc.code in {400, 401, 403}:
            LOGGER.debug("Basic-auth token request failed; retrying with credentials in body")
            return _request_token(config, use_basic_auth=False)
        raise RuntimeError(f"Token request failed with HTTP {exc.code}: {_http_error_body(exc)}") from exc

    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise RuntimeError("Token response did not contain access_token")
    expires_in = int(payload.get("expires_in") or 3600)
    return token, expires_in


def _deployment_from_model(model: str) -> str:
    value = (model or "").strip()
    for prefix in ("sap-aicore:", "sap-aicore/", "aicore:", "aicore/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    if value == "sap-aicore-deployment":
        return ""
    return value


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _models_payload() -> dict[str, Any]:
    configured = (
        os.getenv("SAP_AICORE_MODELS", "").strip()
        or os.getenv("SAP_AICORE_DEPLOYMENT_ID", "").strip()
        or os.getenv("AICORE_DEPLOYMENT_ID", "").strip()
    )
    models = [item.strip() for item in configured.replace(";", ",").split(",") if item.strip()]
    if not models:
        models = ["sap-aicore-deployment"]
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "owned_by": "sap-ai-core"} for model in models],
    }


def _aicore_url(config: AiCoreConfig, deployment_id: str) -> str:
    encoded_deployment = urllib.parse.quote(deployment_id, safe="")
    url = f"{config.inference_base_url}/inference/deployments/{encoded_deployment}/chat/completions"
    if config.api_version:
        url += "?" + urllib.parse.urlencode({"api-version": config.api_version})
    return url


def _orchestration_url(config: AiCoreConfig, deployment_id: str) -> str:
    encoded_deployment = urllib.parse.quote(deployment_id, safe="")
    return f"{config.inference_base_url}/inference/deployments/{encoded_deployment}/v2/completion"


def _model_params(payload: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"):
        if key in payload and payload[key] is not None:
            params[key] = payload[key]
    return params


def _orchestration_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model_name = os.getenv("SAP_AICORE_MODEL_NAME", "").strip() or str(payload.get("model") or "").strip()
    if not model_name or model_name == "sap-aicore-deployment":
        raise ConfigError("SAP_AICORE_MODEL_NAME or Hermes model must contain the AI Core foundation model name")

    prompt: dict[str, Any] = {"template": payload.get("messages") or []}
    if payload.get("tools"):
        prompt["tools"] = payload["tools"]
    if payload.get("tool_choice"):
        prompt["tool_choice"] = payload["tool_choice"]
    if payload.get("response_format"):
        prompt["response_format"] = payload["response_format"]

    model: dict[str, Any] = {"name": model_name}
    params = _model_params(payload)
    if params:
        model["params"] = params

    return {
        "config": {
            "modules": {
                "prompt_templating": {
                    "model": model,
                    "prompt": prompt,
                }
            }
        }
    }


def _as_openai_response(body: bytes) -> bytes:
    data = json.loads(body.decode("utf-8"))
    final_result = data.get("final_result")
    if isinstance(final_result, dict):
        return json.dumps(final_result).encode("utf-8")
    return body


def _as_openai_sse(body: bytes) -> bytes:
    response = json.loads(_as_openai_response(body).decode("utf-8"))
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    chunk_base = {
        "id": response.get("id", "chatcmpl-sap-aicore"),
        "object": "chat.completion.chunk",
        "created": response.get("created", int(time.time())),
        "model": response.get("model", ""),
    }
    first = dict(chunk_base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]
    last = dict(chunk_base)
    last["choices"] = [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}]
    return (
        f"data: {json.dumps(first)}\n\n"
        f"data: {json.dumps(last)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def _runtime_config(payload: dict[str, Any]) -> AiCoreConfig:
    model = str(payload.get("model") or "")
    model_deployment_id = ""
    if not (os.getenv("SAP_AICORE_DEPLOYMENT_ID") or os.getenv("AICORE_DEPLOYMENT_ID")):
        model_deployment_id = _deployment_from_model(model)
    return load_config(model_deployment_id)


def _forward_foundation_chat_completion(payload: dict[str, Any]) -> tuple[int, bytes, str]:
    config = _runtime_config(payload)
    token = TOKEN_CACHE.get(config)

    outbound = dict(payload)
    if os.getenv("SAP_AICORE_FORWARD_MODEL", "").strip().lower() not in {"1", "true", "yes"}:
        outbound.pop("model", None)

    body = json.dumps(outbound).encode("utf-8")
    headers = {
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "AI-Resource-Group": config.resource_group,
        "AI-Client-Type": config.client_type,
    }
    request = urllib.request.Request(_aicore_url(config, config.deployment_id), data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("SAP_AICORE_TIMEOUT", "600"))) as response:
            return response.status, response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            TOKEN_CACHE.clear()
        return exc.code, exc.read(), exc.headers.get_content_type() or "application/json"


def _forward_orchestration_completion(payload: dict[str, Any]) -> tuple[int, bytes, str]:
    config = _runtime_config(payload)
    token = TOKEN_CACHE.get(config)
    wants_stream = bool(payload.get("stream"))
    outbound = _orchestration_payload(payload)
    body = json.dumps(outbound).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "AI-Resource-Group": config.resource_group,
        "AI-Client-Type": config.client_type,
    }
    request = urllib.request.Request(
        _orchestration_url(config, config.deployment_id),
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("SAP_AICORE_TIMEOUT", "600"))) as response:
            response_body = response.read()
            if wants_stream:
                return response.status, _as_openai_sse(response_body), "text/event-stream"
            return response.status, _as_openai_response(response_body), "application/json"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            TOKEN_CACHE.clear()
        return exc.code, exc.read(), exc.headers.get_content_type() or "application/json"


def _forward_chat_completion(payload: dict[str, Any]) -> tuple[int, bytes, str]:
    mode = os.getenv("SAP_AICORE_API_MODE", os.getenv("SAP_AICORE_DEPLOYMENT_TYPE", "foundation")).strip().lower()
    if mode in {"orchestration", "orchestration-v2", "orchestration_v2"}:
        return _forward_orchestration_completion(payload)
    return _forward_foundation_chat_completion(payload)


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "HermesSapAiCoreProxy/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path in {"", "/health", "/v1/health"}:
            _json_response(self, 200, {"status": "ok", "provider": "sap-aicore"})
            return
        if path in {"/v1/models", "/models"}:
            _json_response(self, 200, _models_payload())
            return
        _json_response(self, 404, {"error": {"message": f"Unsupported path: {self.path}"}})

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            _json_response(self, 404, {"error": {"message": f"Unsupported path: {self.path}"}})
            return

        try:
            payload = _read_json(self)
            status, body, content_type = _forward_chat_completion(payload)
            _text_response(self, status, body, content_type)
        except ConfigError as exc:
            _json_response(self, 500, {"error": {"message": str(exc), "type": "sap_aicore_config_error"}})
        except Exception as exc:
            LOGGER.exception("Proxy request failed")
            _json_response(self, 502, {"error": {"message": str(exc), "type": "sap_aicore_proxy_error"}})


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), ProxyHandler)
    LOGGER.warning("SAP AI Core Hermes proxy listening on http://%s:%s/v1", host, port)
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SAP AI Core proxy for Hermes Agent.")
    parser.add_argument("--host", default=os.getenv("SAP_AICORE_PROXY_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("SAP_AICORE_PROXY_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--log-level", default=os.getenv("SAP_AICORE_PROXY_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    try:
        run(args.host, args.port)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
