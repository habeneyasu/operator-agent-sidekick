# Browser Sidekick Agent

An **operator-style**, tool-augmented AI agent built with **LangGraph**. The design centers on a **worker / evaluator** loop: the worker drives task execution (including tools once integrated), and the **evaluator** scores the latest result against **success criteria** to decide whether to **retry with feedback**, **finish**, or **pause for human clarification**.

**Current status:** Foundational layers are in place—typed configuration, Pydantic state, a provider-agnostic LLM abstraction (with an OpenAI adapter), centralized prompts, evaluator JSON parsing, **worker and evaluator nodes**, a **tools package**, and **LangGraph wiring** (`compile_sidekick_graph`, optional `ToolNode`, worker ↔ evaluator routing) with unit tests. **Next milestones:** **product entrypoint** (wire `ChatOpenAI` + `create_sidekick_graph_from_settings`), **memory / checkpointing** (`src/memory/`), and a **Gradio** UI (`src/ui/`).

---

## Features (target architecture)

| Area | What you get |
|------|----------------|
| **Self-evaluation** | Worker + evaluator models iterate until success or a controlled stop for user input. |
| **Tooling** | Browser (Playwright), sandboxed files, web search, Python REPL, Wikipedia, notifications—**to be implemented under `src/tools/`**. |
| **Sessions** | Checkpointing and **`thread_id`**-scoped state for long runs—**planned under `src/memory/`**. |
| **Configuration** | **Type-safe** settings from the environment via `Settings` / `get_settings()` in `src/config.py`. |
| **State & prompts** | **`AgentState`**, **`EvaluatorOutput`**, and shared prompt builders in `src/utils/prompts.py`. |

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/config.py` | Typed environment settings (`Settings`, `get_settings()`). |
| `src/state.py` | `AgentState` and `EvaluatorOutput` (Pydantic). |
| `src/llm/` | `BaseLLMClient`, `OpenAIClient`, request/response DTOs. |
| `src/utils/prompts.py` | Worker and evaluator prompt builders (including evaluator JSON instruction). |
| `src/utils/parsing.py` | Parses evaluator LLM text into `EvaluatorOutput` (raw JSON or fenced code block). |
| `src/agents/worker.py` | Worker node: normalises history, builds the LLM request, runs the model, returns updated state. |
| `src/agents/evaluator.py` | Evaluator node: builds evaluator request, parses structured output, updates flags and transcript. |
| `src/agents/graph.py` | `SidekickGraphState`, `compile_sidekick_graph`, `create_sidekick_graph_from_settings`, routers. |
| `src/tools/` | `SidekickTool` (`spec.py`), `ToolRegistry` / `build_tools`, sandbox files, Serper search, Wikipedia, subprocess Python, Pushover, Playwright browser. **LLM / graph wiring still pending.** |
| `src/ui/` | Gradio app (**planned**). |
| `tests/unit/` | Unit tests (**run with `uv`**). |
| `docs/developer.md` | Developer guide, module map, config table, testing matrix, and change log. |

---

## Quick start

### Prerequisites

- **[uv](https://github.com/astral-sh/uv)** (recommended)
- **Python 3.12+**

### Environment

Copy the example env file and add your keys (at minimum **`OPENAI_API_KEY`** when using `OpenAIClient`):

```bash
cp .env.example .env
```

Edit `.env` for your keys and paths. Full variable reference: [Configuration](#configuration) and `.env.example`.

### Run tests

From the repository root:

```bash
uv run --with pytest --with pydantic pytest -q
```

Graph tests also need LangChain / LangGraph packages:

```bash
uv run --with pytest --with pydantic --with langgraph --with langchain-core --with langchain-openai pytest -q tests/unit/
```

Focused suites:

```bash
uv run --with pytest --with pydantic pytest -q tests/unit/test_state.py
uv run --with pytest --with pydantic pytest -q tests/unit/test_llm_base.py tests/unit/test_openai_client.py
uv run --with pytest --with pydantic pytest -q tests/unit/test_prompts.py
uv run --with pytest --with pydantic pytest -q tests/unit/test_worker.py
uv run --with pytest --with pydantic pytest -q tests/unit/test_evaluator.py
uv run --with pytest --with pydantic pytest -q tests/unit/test_tools/
```

### Run the app

The interactive UI will live in **`src/ui/gradio_app.py`**. Until that lands, there is **no** primary runnable entrypoint in this repository.

---

## Configuration

Variables are documented in **`.env.example`** and summarised in **`docs/developer.md`**. Commonly used:

| Variable | Role |
|----------|------|
| `OPENAI_API_KEY` | Credential for `OpenAIClient`. |
| `OPENAI_MODEL_WORKER` / `OPENAI_MODEL_EVALUATOR` | Default model IDs. |
| `LLM_TIMEOUT_SECONDS` | HTTP client timeout for LLM calls. |
| `MAX_AGENT_ITERATIONS` | Safety cap for graph loops (used once `graph.py` exists). |
| `SANDBOX_DIR` / `SESSION_STORE_DIR` | Local sandbox and session storage roots. |
| `BROWSER_HEADLESS` | Headless vs visible browser (for future Playwright tools). |
| `SERPER_API_KEY`, `PUSHOVER_*` | Optional integrations for search and push notifications. |

---

## Architecture

Target control flow once LangGraph and tools are connected:

```mermaid
flowchart LR
  User[User / UI] <--> Sidekick[Sidekick / LangGraph]
  Sidekick --> Worker[Worker LLM]
  Worker -->|tool calls| Tools[Tools]
  Tools --> Worker
  Worker -->|assistant reply| Eval[Evaluator LLM]
  Eval -->|insufficient: feedback| Worker
  Eval -->|success or user input needed| End[END / pause]
```

### Implemented today

- **`AgentState`** – Conversation `messages`, `success_criteria`, evaluator flags, iteration metadata.
- **Worker** (`src/agents/worker.py`) – Builds provider-agnostic chat requests from history + prompts; appends an assistant turn. *Tool calling / `ToolNode` wiring is next now that `src/tools/` ships concrete `SidekickTool` implementations.*
- **Evaluator** (`src/agents/evaluator.py`) – Treats the **last** message as the assistant answer, requests a **JSON** judgement, validates into **`EvaluatorOutput`**, updates flags, and appends an evaluator feedback line to the transcript.
- **Prompts** – Single source of truth in `src/utils/prompts.py`, including the evaluator’s JSON-only instruction.
- **Parsing** – `src/utils/parsing.py` tolerates markdown-fenced JSON and validates with Pydantic.
- **Tools** (`src/tools/`) – Runnable `SidekickTool` set: sandbox I/O, optional web search, Wikipedia, subprocess Python, optional Pushover, optional Playwright (install browsers separately). Composed via `ToolRegistry` / `build_tools` and wired into LangGraph via `sidekick_tool_to_langchain` + `ToolNode` when tools are non-empty.
- **Graph** (`src/agents/graph.py`) – LangGraph state machine (`SidekickGraphState`): worker → optional tools → evaluator → (`END` or worker). Production wiring uses `ChatOpenAI.bind_tools` + `with_structured_output(EvaluatorOutput)` via `create_sidekick_graph_from_settings`.

### Pending integration

- **Runnable app** – Instantiate `ChatOpenAI`, evaluator `with_structured_output(EvaluatorOutput)`, and call `create_sidekick_graph_from_settings` (or `compile_sidekick_graph`) from CLI / Gradio.
- **`src/memory/`** – Durable checkpointing and session lifecycle (`thread_id`, TTL, etc.).
- **`src/ui/`** – Gradio chat, success criteria, go/reset, and wiring to the compiled graph.

---

## Documentation

| Resource | Use it for |
|----------|------------|
| [`docs/developer.md`](docs/developer.md) | Change protocol, module ownership, config contract, testing matrix, PR checklist. **Update it when you ship features.** |

---

## License

See **`LICENSE`** (replace placeholder content when you choose MIT, Apache-2.0, or another license).

---

## Contributing

1. Implement or change behaviour **with tests**.
2. Update **`docs/developer.md`** (change log, config, testing notes as applicable).
3. Verify locally:

   ```bash
   uv run --with pytest --with pydantic pytest -q
   ```

4. Open a pull request with a clear summary and test results.
