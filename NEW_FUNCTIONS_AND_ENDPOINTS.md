# New functions and API endpoints (branch `farhan`)

> Paths in this file are relative to the `arachne` superproject root, where this repo is checked out as a submodule. The full cross-repo version lives at `NEW_FUNCTIONS_AND_ENDPOINTS.md` in the superproject.

## llm-utils (`centaurus/fictive/llm-utils`, branch `farhan`)

Package version bumped `0.1.0` -> `0.2.0`. Two themes: (1) token streaming through a uniform `generate_stream` interface, (2) a new OpenAI backend. Nothing existing was removed.

### `llm_utils/pipelines.py`

function/api endpoint: `LLMPipeline.generate_stream`
Base-class streaming hook. Default implementation does not stream: it calls `self.generate(inputs)` and yields a single `done` event, so every pipeline (SGLang, vLLM, transformers, ...) satisfies the interface and callers never need to check for streaming support.
`centaurus/fictive/llm-utils/llm_utils/pipelines.py:125`
Prerequisites: None beyond whatever the concrete pipeline's `generate` needs.
Inputs: `inputs` -- same shape `generate` accepts (a string, a list of message dicts, or a list of message lists for batch).
Outputs: generator of event dicts. Contract shared by all overrides: zero or more `{"type": "delta", "text": str}` / `{"type": "thinking_delta", "text": str}`, then exactly one `{"type": "done", "message": <what generate() would return>}`.

function/api endpoint: `MinimaxPipeline.generate_stream`
Streams a single MiniMax turn token by token via `minimax_chat_completion_stream`. Batched (list-of-lists) inputs fall back to one `done` event.
`centaurus/fictive/llm-utils/llm_utils/pipelines.py:418`
Prerequisites: `MINIMAX_API_KEY` and `MINIMAX_BASE_URL` (env or `.env` at the fixed path used by `request_minimax._read_dotenv`); `anthropic` pip package.
Inputs: `inputs` as above.
Outputs: `delta` / `thinking_delta` events, then `done` with `{"role": "assistant", "thinking": str, "content": str}`.

function/api endpoint: `AnthropicAPIPipeline.generate_stream`
Streams a single Anthropic Messages API turn via `anthropic_messages_completion_stream`. Batched inputs fall back to one `done` event.
`centaurus/fictive/llm-utils/llm_utils/pipelines.py:467`
Prerequisites: `PipelineConfig.token` and `PipelineConfig.base_url` populated (Centaurus fills these from its own env); `anthropic` pip package.
Inputs: `inputs` as above.
Outputs: `delta` / `thinking_delta` events, then `done` with `{"role": "assistant", "content": str}`.

function/api endpoint: `OpenAIPipeline` (class) / `OpenAIPipeline.generate` / `OpenAIPipeline.generate_stream`
New backend selected by `pipeline_type="openai"`. `generate` mirrors the other pipelines (single request or batch, debug logging); `generate_stream` streams a single turn and falls back to `done` for batches.
`centaurus/fictive/llm-utils/llm_utils/pipelines.py:478` (`generate` at 487, `generate_stream` at 516)
Prerequisites: `OPENAI_API_KEY` (required) and `OPENAI_BASE_URL` (optional, default `https://api.openai.com/v1`), from env or the nearest `.env` walking up from CWD; `openai` pip package (not yet listed in llm-utils extras -- install manually).
Inputs: `cfg: PipelineConfig` (uses `model`, `temperature`, `max_new_tokens`, `timeout`, `log_level`); `inputs` as above.
Outputs: `generate` -> `{"role": "assistant", "thinking": str, "content": str}` or a list of those for batches. `generate_stream` -> `thinking_delta` / `delta` events then `done` with the same message shape.

function/api endpoint: `MockPipeline.generate_stream`
Emits the mock reply in 5-word chunks with a 10 ms sleep between them so offline UI work sees realistic streaming. Non-dict (batch) results yield a single `done`.
`centaurus/fictive/llm-utils/llm_utils/pipelines.py:758`
Prerequisites: None (offline). Note: the mock pipeline writes a `llm_utils/.mock_pipeline_cache/` corpus at runtime; this directory is now in `.gitignore`.
Inputs: `inputs` as above.
Outputs: `delta` events then `done` with the full mock message.

function/api endpoint: `pipeline_from_config` (modified)
Now also constructs `OpenAIPipeline` when `cfg.pipeline_type == "openai"`. `PIPELINE_TYPES` literal gained `"openai"`.
`centaurus/fictive/llm-utils/llm_utils/pipelines.py:841`
Prerequisites: as for the selected pipeline.
Inputs: `cfg: PipelineConfig`.
Outputs: an `LLMPipeline` subclass instance.
Note: `llm_utils/cli.py` argparse `--pipeline_type` choices still list only `sglang|vllm|minimax|anthropic`; `openai` and `mock` are reachable through `PipelineConfig` directly but not through the CLI.

### `llm_utils/request_openai.py` (new file)

function/api endpoint: `request_openai.openai_chat_completion`
Sends one Chat Completions request and returns the assistant message. Reasoning-model families (`gpt-5*`, `o1*`, `o3*`, `o4*`) are sent without `temperature`; the token cap is passed as `max_completion_tokens`.
`centaurus/fictive/llm-utils/llm_utils/request_openai.py:147`
Prerequisites: `OPENAI_API_KEY`; optional `OPENAI_BASE_URL`; `openai` pip package.
Inputs: `cfg: RequestConfig` (`model`, `temperature`, `max_new_tokens`, `timeout`), `messages: List[{"role","content"}]`.
Outputs: `{"role": "assistant", "thinking": str, "content": str}` (`thinking` comes from `reasoning_content` when the provider returns it, else `""`).

function/api endpoint: `request_openai.openai_chat_completion_stream`
Streaming variant of the above using `stream=True`.
`centaurus/fictive/llm-utils/llm_utils/request_openai.py:157`
Prerequisites: same as `openai_chat_completion`.
Inputs: `cfg: RequestConfig`, `messages`.
Outputs: generator of `thinking_delta` / `delta` events followed by `done` with the same message shape as the non-streaming call.

function/api endpoint: `request_openai.openai_chat_completion_batch`
Sequentially runs `openai_chat_completion` for each message list.
`centaurus/fictive/llm-utils/llm_utils/request_openai.py:196`
Prerequisites: same as `openai_chat_completion`.
Inputs: `cfg: RequestConfig`, `requests_messages: List[List[message]]`.
Outputs: `List[message]` in input order.

function/api endpoint: `request_openai._get_openai_credentials` / `_dotenv_values` / `_read_dotenv` / `_clean_env_value` / `_client` / `_completion_kwargs` / `_message_from_choice` (private helpers)
Credential loading walks from CWD up to `/` merging every `.env` (nearest wins) so the server launched from `centaurus/` still finds the repo-root `.env`. Raises `ValueError` if `OPENAI_API_KEY` is missing or `OPENAI_BASE_URL` lacks an `http(s)://` scheme; `ImportError` if `openai` is not installed.
`centaurus/fictive/llm-utils/llm_utils/request_openai.py:24-144`
Prerequisites: as above.
Inputs: none / `cfg` / `messages` respectively.
Outputs: `{"api_key", "base_url"}` dict, an `openai.OpenAI` client, a request kwargs dict, or a normalised assistant message.

### `llm_utils/request_anthropic_api.py`

function/api endpoint: `anthropic_messages_completion_stream`
Streams one Messages API request with `client.messages.stream(...)`, forwarding `text_delta` and `thinking_delta` content-block events.
`centaurus/fictive/llm-utils/llm_utils/request_anthropic_api.py:124`
Prerequisites: `cfg.token` (API key) and `cfg.base_url`; `anthropic` pip package.
Inputs: `cfg: RequestConfig`, `messages` (system message extracted and sent as `system=`).
Outputs: `delta` / `thinking_delta` events then `done` with `{"role": "assistant", "content": str}`.

### `llm_utils/request_minimax.py`

function/api endpoint: `minimax_chat_completion_stream`
Streams one MiniMax completion through the Anthropic-compatible endpoint; accumulates both thinking and text.
`centaurus/fictive/llm-utils/llm_utils/request_minimax.py:125`
Prerequisites: `MINIMAX_API_KEY`, `MINIMAX_BASE_URL` (env or `.env`); `anthropic` pip package.
Inputs: `cfg: RequestConfig` (`model`, `max_new_tokens`, `timeout`), `messages`.
Outputs: `delta` / `thinking_delta` events then `done` with `{"role": "assistant", "thinking": str, "content": str}` (both stripped).

### `llm_utils/__init__.py`

Exports added: `anthropic_messages_completion_stream`, `minimax_chat_completion_stream`, `openai_chat_completion`, `openai_chat_completion_batch`, `openai_chat_completion_stream`, `OpenAIPipeline`. `__version__` is now `0.2.0`.

### Other
- `DOCS.md`: documents the streaming contract and the OpenAI helpers (per the repo's `AGENTS.md` documentation policy).
- `.gitignore`: now ignores `llm_utils/.mock_pipeline_cache/`.
