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


def _log_aicore_http_error(kind: str, status: int, body: bytes) -> None:
    raw = body.decode("utf-8", errors="replace")
    message = raw[:500]
    try:
        data = json.loads(raw)
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            request_id = str(error.get("request_id") or "").strip()
            error_message = str(error.get("message") or "").strip()
            message = f"{error_message} request_id={request_id}".strip()
    except Exception:
        pass
    LOGGER.warning("SAP AI Core %s request failed with HTTP %s: %s", kind, status, message)


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


def _active_mode() -> str:
    mode = os.getenv("SAP_AICORE_API_MODE", os.getenv("SAP_AICORE_DEPLOYMENT_TYPE", "foundation")).strip().lower()
    return "orchestration" if mode in {"orchestration", "orchestration-v2", "orchestration_v2"} else "foundation"


def _is_chat_model(name: str) -> bool:
    lowered = (name or "").lower()
    if not lowered:
        return False
    if "embed" in lowered or "embedqa" in lowered:
        return False
    if lowered.startswith("sap-rpt") or "rpt-1" in lowered:
        return False
    return True


def _fetch_deployments(config: AiCoreConfig, token: str) -> list[dict[str, Any]]:
    url = config.inference_base_url + "/lm/deployments?" + urllib.parse.urlencode({"status": "RUNNING"})
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": config.resource_group,
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    resources = data.get("resources") or data.get("data") or []
    return [item for item in resources if isinstance(item, dict)]


def _deployment_model_name(deployment: dict[str, Any]) -> str:
    details = deployment.get("details") if isinstance(deployment.get("details"), dict) else {}
    resources = details.get("resources") if isinstance(details.get("resources"), dict) else {}
    backend = resources.get("backendDetails") or resources.get("backend_details") or {}
    model = backend.get("model") if isinstance(backend, dict) else None
    if isinstance(model, dict):
        return str(model.get("name") or "").strip()
    return ""


def _live_models() -> list[str]:
    """Query SAP AI Core for the catalog the active mode can actually serve."""
    config = load_config(os.getenv("SAP_AICORE_DEPLOYMENT_ID", "") or os.getenv("AICORE_DEPLOYMENT_ID", "") or "list")
    token = TOKEN_CACHE.get(config)
    deployments = _fetch_deployments(config, token)

    if _active_mode() == "orchestration":
        names = sorted(
            {
                name
                for dep in deployments
                if dep.get("scenarioId") in {"foundation-models", None} or _deployment_model_name(dep)
                for name in [_deployment_model_name(dep)]
                if _is_chat_model(name)
            }
        )
        return names

    # Foundation mode routes by deployment id, so the model id IS the deployment id.
    ids = []
    for dep in deployments:
        if dep.get("scenarioId") not in {"foundation-models", None}:
            continue
        if not _is_chat_model(_deployment_model_name(dep)):
            continue
        dep_id = str(dep.get("id") or "").strip()
        if dep_id:
            ids.append(dep_id)
    return sorted(ids)


def _models_payload() -> dict[str, Any]:
    from .config import _load_hermes_dotenv

    _load_hermes_dotenv()
    configured = os.getenv("SAP_AICORE_MODELS", "").strip()
    models = [item.strip() for item in configured.replace(";", ",").split(",") if item.strip()]

    if not models:
        try:
            models = _live_models()
        except Exception as exc:
            LOGGER.warning("Live model listing failed, falling back to configured model: %s", exc)

    if not models:
        fallback = os.getenv("SAP_AICORE_MODEL_NAME", "").strip()
        models = [fallback] if fallback else ["sap-aicore-model"]

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


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _orchestration_messages(messages: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return normalized

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role in {"system", "user", "assistant"}:
            content = message.get("content") or ""
            if role == "assistant" and not _stringify_content(content).strip():
                continue
            sanitized: dict[str, Any] = {
                "role": role,
                "content": content,
            }
            normalized.append(sanitized)
            continue
        if role in {"tool", "function"}:
            tool_name = str(message.get("name") or message.get("tool_call_id") or "tool").strip()
            content = _stringify_content(message.get("content", ""))
            normalized.append(
                {
                    "role": "user",
                    "content": f"Tool result ({tool_name}):\n{content}",
                }
            )

    return normalized


def _orchestration_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # The model picked in Hermes wins; the env default is only a fallback so a
    # bare `hermes` session without an explicit model still works.
    model_name = str(payload.get("model") or "").strip()
    if not model_name or model_name in {"sap-aicore-model", "sap-aicore-deployment"}:
        model_name = os.getenv("SAP_AICORE_MODEL_NAME", "").strip()
    if not model_name or model_name in {"sap-aicore-model", "sap-aicore-deployment"}:
        raise ConfigError("Hermes model or SAP_AICORE_MODEL_NAME must contain the AI Core foundation model name")

    prompt: dict[str, Any] = {"template": _orchestration_messages(payload.get("messages") or [])}
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


def _looks_like_incomplete_tool_intent(content: str) -> bool:
    text = " ".join((content or "").strip().lower().split())
    if not text:
        return False
    starters = (
        "let me ",
        "i'll ",
        "i will ",
        "i need to ",
        "i should ",
        "now let me ",
        "next, i'll ",
        "next i'll ",
    )
    actions = ("check", "inspect", "look", "read", "search", "explore", "find", "open", "analyze")
    return text.startswith(starters) and any(action in text[:160] for action in actions)


def _nudge_incomplete_tool_intent(response: dict[str, Any], request_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if not request_payload or not request_payload.get("tools"):
        return response
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        return response
    choice = choices[0]
    if not isinstance(choice, dict):
        return response
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    if message.get("tool_calls"):
        return response
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    if _looks_like_incomplete_tool_intent(content):
        choice["finish_reason"] = "length"
    return response


def _as_openai_response(body: bytes, request_payload: dict[str, Any] | None = None) -> bytes:
    data = json.loads(body.decode("utf-8"))
    final_result = data.get("final_result")
    if isinstance(final_result, dict):
        return json.dumps(_nudge_incomplete_tool_intent(final_result, request_payload)).encode("utf-8")
    return body


def _as_openai_sse(body: bytes, request_payload: dict[str, Any] | None = None) -> bytes:
    response = json.loads(_as_openai_response(body, request_payload).decode("utf-8"))
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    chunk_base = {
        "id": response.get("id", "chatcmpl-sap-aicore"),
        "object": "chat.completion.chunk",
        "created": response.get("created", int(time.time())),
        "model": response.get("model", ""),
    }
    chunks: list[str] = []
    first = dict(chunk_base)
    first_delta: dict[str, Any] = {"role": "assistant"}
    if content:
        first_delta["content"] = content
    first["choices"] = [{"index": 0, "delta": first_delta, "finish_reason": None}]
    chunks.append(f"data: {json.dumps(first)}\n\n")

    for index, tool_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        tool_chunk = dict(chunk_base)
        tool_chunk["choices"] = [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_call.get("id") or f"call_sap_aicore_{index}",
                            "type": tool_call.get("type") or "function",
                            "function": {
                                "name": function.get("name") or "",
                                "arguments": arguments,
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
        chunks.append(f"data: {json.dumps(tool_chunk)}\n\n")

    last = dict(chunk_base)
    last["choices"] = [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}]
    chunks.append(f"data: {json.dumps(last)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


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
        error_body = exc.read()
        _log_aicore_http_error("foundation", exc.code, error_body)
        return exc.code, error_body, exc.headers.get_content_type() or "application/json"


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
                return response.status, _as_openai_sse(response_body, payload), "text/event-stream"
            return response.status, _as_openai_response(response_body, payload), "application/json"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            TOKEN_CACHE.clear()
        error_body = exc.read()
        _log_aicore_http_error("orchestration", exc.code, error_body)
        return exc.code, error_body, exc.headers.get_content_type() or "application/json"


def _forward_chat_completion(payload: dict[str, Any]) -> tuple[int, bytes, str]:
    if _active_mode() == "orchestration":
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
