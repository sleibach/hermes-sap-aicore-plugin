# Hermes SAP AI Core Plugin

Installable Hermes Agent model-provider support for SAP AI Core Generative AI Hub deployments.

Hermes `v0.17.0` model-provider plugins are discovered from:

```text
$HERMES_HOME/plugins/model-providers/<provider>/
```

They work best when the provider exposes an OpenAI-compatible `/v1/chat/completions` surface. SAP AI Core uses OAuth client-credentials service keys plus `AI-Resource-Group`, so this package provides:

- a Hermes provider profile named `sap-aicore`
- a local OpenAI-compatible proxy at `http://127.0.0.1:8765/v1`
- token exchange and token caching for SAP AI Core service-key JSON files
- model-to-deployment mapping, where the Hermes model id is used as the SAP AI Core deployment id
- orchestration mode, where SAP's orchestration `final_result` is normalized back to OpenAI chat-completions JSON

## Install

From this directory:

```bash
python3 -m pip install -e .
hermes-sap-aicore-install \
  --service-key /path/to/hermes-key.json \
  --deployment-id <SAP_AI_CORE_DEPLOYMENT_ID> \
  --api-mode orchestration \
  --model-name anthropic--claude-4.5-sonnet \
  --resource-group default \
  --write-env
```

The installer writes the drop-in provider to:

```text
~/.hermes/plugins/model-providers/sap-aicore/
```

It also appends missing values to `~/.hermes/.env` when `--write-env` is set.

## Run

Start the proxy in one terminal:

```bash
sap-aicore-hermes-proxy
```

Then run Hermes with the SAP AI Core provider:

```bash
hermes -z "Say hello in one sentence." --provider sap-aicore -m <SAP_AI_CORE_DEPLOYMENT_ID>
```

For orchestration mode, `-m` is the foundation model name unless `SAP_AICORE_MODEL_NAME` is set:

```bash
hermes -z "Say hello in one sentence." --provider sap-aicore -m anthropic--claude-4.5-sonnet
```

You can also set multiple visible model/deployment ids for the Hermes model picker:

```bash
export SAP_AICORE_MODELS="deployment-a,deployment-b"
```

## Configuration

Required:

- `SAP_AICORE_SERVICE_KEY` or `SAP_AICORE_SERVICE_KEY_FILE`: path to the SAP AI Core service-key JSON
- `SAP_AICORE_DEPLOYMENT_ID` or the Hermes `-m` model value: SAP AI Core deployment id
- `SAP_AICORE_PROXY_KEY`: local key Hermes sends to the proxy; it is not forwarded to SAP

Optional:

- `SAP_AICORE_API_MODE`: `foundation` for `/chat/completions`, `orchestration` for `/v2/completion`
- `SAP_AICORE_MODEL_NAME`: foundation model name used by orchestration mode; when unset, Hermes `-m` is used
- `SAP_AICORE_RESOURCE_GROUP`: defaults to `default`
- `SAP_AICORE_PROXY_BASE_URL`: defaults to `http://127.0.0.1:8765/v1`
- `SAP_AICORE_PROXY_HOST`: defaults to `127.0.0.1`
- `SAP_AICORE_PROXY_PORT`: defaults to `8765`
- `SAP_AICORE_API_VERSION`: appended as `api-version=...` when needed by a deployment
- `SAP_AICORE_FORWARD_MODEL=true`: keep the OpenAI `model` field in the forwarded request
- `SAP_AICORE_TIMEOUT`: proxy request timeout in seconds, defaults to `600`

The service key can also be supplied as `SAP_AICORE_SERVICE_KEY_JSON`.

## Design Note

Capability audit:

- BTP layer: SAP AI Core is the correct platform service for governed model access.
- CAP layer: not applicable; this is a Hermes provider plugin, not a CAP service.
- Fiori layer: not applicable.
- Decision: custom Hermes plugin plus local proxy, because Hermes provider profiles are declarative API-key/OpenAI-compatible entries, while SAP AI Core requires OAuth token exchange and an AI Core deployment URL.

## Test

```bash
python3 -m pytest
```
