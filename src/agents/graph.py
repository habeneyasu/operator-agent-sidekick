"""LangGraph construction: worker, optional ToolNode, evaluator, conditional edges."""

from __future__ import annotations

from datetime import datetime
import time
from typing import Annotated, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel

from src.agents.evaluator import build_evaluator_chat_messages
from src.config import Settings, get_settings
from src.state import AgentState, EvaluatorOutput
from src.tools.browser import BrowserController
from src.tools.spec import SidekickTool
from src.tools.tool_registry import build_tools
from src.utils.prompts import build_worker_system_message
from src.logger import get_logger
from src.metrics import increment_counter, observe_histogram
from src.utils.parsing import parse_evaluator_output
from src.tools.wikipedia import fetch_wikipedia_summary
from pathlib import Path

_log = get_logger("graph")


class SidekickGraphState(TypedDict, total=False):
    """LangGraph state for the sidekick loop (``add_messages`` on ``messages``)."""

    messages: Annotated[list[BaseMessage], add_messages]
    success_criteria: str
    feedback_on_work: str | None
    success_criteria_met: bool
    user_input_needed: bool
    iteration: int
    thread_id: str | None
    force_retrieval: bool


def sidekick_tool_to_langchain(tool: SidekickTool) -> StructuredTool:
    """Wrap a :class:`SidekickTool` as a LangChain ``StructuredTool`` with a single string input.

    We standardize on a single argument named ``input`` to avoid ambiguity across tools.
    """

    class _SingleStringInput(BaseModel):
        input: str

    def _invoke(input: str) -> str:
        t0 = time.perf_counter()
        try:
            result = tool.invoke(input)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            try:
                increment_counter("tool_calls_total", labels={"tool": tool.name, "status": "success"})
                observe_histogram("tool_latency_ms", dt_ms, labels={"tool": tool.name, "status": "success"})
            except Exception:
                pass
            return result
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            try:
                increment_counter("tool_calls_total", labels={"tool": tool.name, "status": "error"})
                observe_histogram("tool_latency_ms", dt_ms, labels={"tool": tool.name, "status": "error"})
            except Exception:
                pass
            raise

    return StructuredTool.from_function(
        coroutine=None,
        func=_invoke,
        name=tool.name,
        description=tool.description,
        args_schema=_SingleStringInput,
    )


def lc_messages_to_agent_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to dicts compatible with :class:`AgentState` / evaluator."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": str(m.content or "")})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": str(m.content)})
    return out


def route_worker(state: SidekickGraphState) -> Literal["tools", "evaluator"]:
    # Guard: if there are no messages yet, go straight to evaluator
    if not state.get("messages"):
        _log.debug("route_worker: no messages -> evaluator")
        return "evaluator"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        _log.debug("route_worker: tool_calls detected -> tools")
        return "tools"
    _log.debug("route_worker: no tool_calls -> evaluator")
    return "evaluator"


def route_evaluator(state: SidekickGraphState) -> Literal["worker", "__end__"]:
    if state.get("success_criteria_met") or state.get("user_input_needed"):
        _log.debug(
            "route_evaluator: end (success=%s user_input=%s)",
            state.get("success_criteria_met"),
            state.get("user_input_needed"),
        )
        return "__end__"
    _log.debug("route_evaluator: continue -> worker")
    return "worker"


def _evaluator_lc_messages(agent_state: AgentState) -> list[BaseMessage]:
    cms = build_evaluator_chat_messages(agent_state)
    lc: list[BaseMessage] = []
    for cm in cms:
        if cm.role == "system":
            lc.append(SystemMessage(content=cm.content))
        elif cm.role == "user":
            lc.append(HumanMessage(content=cm.content))
        else:
            lc.append(HumanMessage(content=cm.content))
    return lc


def _approx_tokens_from_messages(messages: list[BaseMessage]) -> int:
    """Rough token estimate from all message contents (chars/4 heuristic)."""
    total_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    return max(0, total_chars // 4)


def _sanitize_answer_text(text: str) -> str:
    """Remove ReAct scaffolding prefixes, keeping the actual answer content.

    - If 'Answer:' exists, return everything after the first occurrence.
    - Otherwise strip lines that are ONLY a ReAct label (e.g. 'Thought: ...' as a
      standalone prefix line), but keep the content of those lines.
    - If stripping leaves nothing, return the original text unchanged.
    """
    t = text or ""
    lower = t.lower()

    # If there's an explicit Answer: marker, use everything after it
    if "answer:" in lower:
        idx = lower.find("answer:")
        result = t[idx + len("answer:"):].strip()
        if result:
            return result

    # Otherwise remove only the label prefix from ReAct lines, keep the content
    lines = []
    react_prefixes = ("thought:", "action:", "observation:")
    for line in t.splitlines():
        stripped = line.strip()
        lower_stripped = stripped.lower()
        # If the line IS just a label with no content after it, skip it
        is_bare_label = any(lower_stripped == p.rstrip(":") + ":" for p in react_prefixes)
        if is_bare_label:
            continue
        # If the line starts with a label, strip the label but keep the content
        for prefix in react_prefixes:
            if lower_stripped.startswith(prefix):
                content_after = stripped[len(prefix):].strip()
                if content_after:
                    lines.append(content_after)
                break
        else:
            lines.append(line)

    cleaned = "\n".join(lines).strip()
    # Safety: if we stripped everything, return the original
    return cleaned if cleaned else t.strip()


def _format_steps_markdown(text: str) -> str:
    """Convert leading numbered steps to bolded bullet points:
    '1. do X' -> '- **Step 1:** do X'
    """
    if not text:
        return text
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        # Match 'N. ' at line start (after optional leading spaces)
        if stripped and stripped[0].isdigit():
            i = 0
            while i < len(stripped) and stripped[i].isdigit():
                i += 1
            if i < len(stripped) and stripped[i] == ".":
                # Extract step number and remainder (skip dot and optional space)
                step_num = stripped[:i]
                remainder = stripped[i + 1 :].lstrip()
                formatted = f"- **Step {step_num}:** {remainder}" if remainder else f"- **Step {step_num}:**"
                out_lines.append(formatted)
                continue
        out_lines.append(line)
    return "\n".join(out_lines)



def _worker_node_fn(
    state: SidekickGraphState,
    llm_with_tools: Runnable,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    settings = get_settings()
    sys_content = build_worker_system_message(
        success_criteria=state.get("success_criteria") or "",
        feedback_on_work=state.get("feedback_on_work"),
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    sys_msg = SystemMessage(content=sys_content)
    # Extract any retrieved local docs before filtering SystemMessages
    local_docs_content: str = ""
    for m in state.get("messages", []):
        if isinstance(m, SystemMessage):
            c = str(getattr(m, "content", "") or "")
            if "Retrieved (Local Docs):" in c:
                local_docs_content = c
                break

    # Build a history window under character budget — always keep the first user message
    raw_hist = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    char_budget = max(0, settings.history_char_limit)
    if raw_hist:
        # Always anchor the first message (original user request)
        anchor = raw_hist[:1]
        rest = raw_hist[1:]
        hist: list[BaseMessage] = []
        used = sum(len(str(getattr(m, "content", "") or "")) for m in anchor)
        for m in reversed(rest):
            content = str(getattr(m, "content", "") or "")
            length = len(content)
            if used + length > char_budget:
                break
            hist.append(m)
            used += length
        hist.reverse()
        hist = anchor + hist
    else:
        hist = []

    criteria_text = (state.get("success_criteria") or "").lower()
    overview_like = ("overview" in criteria_text) or ("advantages" in criteria_text)

    pre_msgs: list[BaseMessage] = [sys_msg]
    if local_docs_content:
        # Merge docs + instruction into one HumanMessage to avoid consecutive human turns
        instruction = (
            "\n\nUsing the documentation above, write a concise answer:\n"
            "1. One short paragraph (2-3 sentences) on what this project does.\n"
            "2. Three bullet points (starting with '- ') on its concrete advantages.\n"
            "Be specific to this repository. No external URLs."
            if overview_like else ""
        )
        pre_msgs.append(HumanMessage(content=local_docs_content + instruction))
    _log.info("worker: invoking LLM (history=%d)", len(hist))
    response = llm_with_tools.invoke([*pre_msgs, *hist])
    _log.info("worker: raw response type=%s content=%r", type(response).__name__, str(getattr(response, "content", response))[:300])
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    else:
        raw_content = str(response.content or "")
        cleaned = _sanitize_answer_text(raw_content)
        formatted = _format_steps_markdown(cleaned)
        _log.info("worker: after sanitize/format content=%r", formatted[:300])
        # Hard fallback: if sanitization produced nothing, use the raw content
        final_content = formatted if formatted.strip() else raw_content.strip()
        response = AIMessage(content=final_content)
    dt = (time.perf_counter() - t0) * 1000
    _log.info("worker: completed in %.1f ms", dt)
    return {"messages": [response]}


RESEARCH_KEYWORDS = [
    "what is", "define", "definition", "explain", "who is", "history of",
    "latest", "current", "citation", "source", "reference", "facts", "factual",
    # Addis/local examples (customizable)
    "etb exchange rate", "sheger", "ethiopia", "addis ababa", "current time in ethiopia",
    # Repo/overview cues to force retrieval/grounding
    "overview", "readme", "repository", "project advantages", "docs", "documentation",
]


def _intent_detection_node(state: SidekickGraphState) -> dict[str, Any]:
    """Detect if the query is likely factual/definitional and require retrieval."""
    text_blobs: list[str] = []
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            text_blobs.append(str(m.content or "").lower())
    crit = (state.get("success_criteria") or "").lower()
    blob = " ".join(text_blobs + [crit])
    needs = any(k in blob for k in RESEARCH_KEYWORDS)
    if needs:
        return {"force_retrieval": True}
    return {"force_retrieval": False}

def _auto_retrieve_node(state: SidekickGraphState) -> dict[str, Any]:
    """If retrieval is mandatory and tools are unavailable (e.g., Ollama without bind_tools),
    perform a lightweight Wikipedia fetch and attach the observation to the transcript."""
    if not state.get("force_retrieval"):
        return {}
    # Find the latest user query
    user_query = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_query = str(m.content or "").strip()
            if user_query:
                break
    if not user_query:
        return {}
    # If the query looks like an overview of this repo, ground in local docs instead of external web
    lower_q = user_query.lower()
    if any(k in lower_q for k in ["overview", "readme", "advantages", "repository", "project"]):
        try:
            root = Path.cwd()
            readme_p = root / "README.md"
            dev_p = root / "docs" / "developer.md"
            snippets: list[str] = []
            def _section_slice(text: str, needles: list[str], max_len: int = 1500) -> str:
                lower = text.lower()
                starts = [lower.find(n) for n in needles if lower.find(n) != -1]
                if starts:
                    start = min(starts)
                    return text[start : start + max_len]
                return text[:max_len]
            if readme_p.exists():
                txt = readme_p.read_text(encoding="utf-8", errors="ignore")
                snippets.append(
                    "(README.md)\n"
                    + _section_slice(
                        txt,
                        ["## features", "## architecture", "## why this matters", "## business impact"],
                        max_len=1500,
                    )
                )
            if dev_p.exists():
                txt = dev_p.read_text(encoding="utf-8", errors="ignore")
                snippets.append(
                    "(docs/developer.md)\n"
                    + _section_slice(
                        txt,
                        ["## 2) current architecture snapshot", "## 2.1 architecture principles", "## 2.2 system boundaries"],
                        max_len=1000,
                    )
                )
            if snippets:
                content = "Retrieved (Local Docs):\n" + "\n\n".join(snippets)
                return {"messages": [SystemMessage(content=content)]}
        except Exception:
            pass
    # Fallback to Wikipedia summary for general factual queries
    try:
        summary = fetch_wikipedia_summary(user_query)
        slug = user_query.replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{slug}"
        content = f"Retrieved (Wikipedia): {url}\nSummary: {summary}"
        obs = SystemMessage(content=content)
        return {"messages": [obs]}
    except Exception:
        return {}

def route_intent(state: SidekickGraphState) -> Literal["autoretrieve", "worker"]:
    return "autoretrieve" if state.get("force_retrieval") else "worker"

def _finalize_node(state: SidekickGraphState) -> dict[str, Any]:
    """Finalize in straight-line mode: mark success if there's a substantive assistant reply."""
    last_assistant = ""
    has_tool_calls_only = False
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            content = str(getattr(m, "content", "") or "")
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls and not content.strip():
                has_tool_calls_only = True
            else:
                last_assistant = content
            break

    insufficient = (
        "couldn't find sufficient information" in last_assistant.lower()
        or "cannot provide a comprehensive overview" in last_assistant.lower()
    )
    success = bool(last_assistant.strip() and not insufficient and not has_tool_calls_only)

    # Check token cap — straight-line path bypasses the evaluator router so we check here
    approx_tokens = _approx_tokens_from_messages(state.get("messages", []))
    try:
        increment_counter(
            "evaluator_judgements_total",
            labels={"accepted": str(success).lower()},
        )
        if success:
            increment_counter("evaluator_accept_total")
        observe_histogram("tokens_total", float(approx_tokens), labels={})
    except Exception:
        pass

    feedback_line = (
        "Evaluator Feedback on this answer: Accepted."
        if success
        else "Evaluator Feedback on this answer: Rejected: no substantive answer produced."
    )
    return {
        "messages": [SystemMessage(content=feedback_line)],
        "success_criteria_met": success,
        "user_input_needed": False,
        "iteration": state.get("iteration", 0) + 1,
    }
def _evaluator_node_fn(
    state: SidekickGraphState,
    evaluator_llm: Runnable,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    settings = get_settings()
    agent_state = AgentState(
        messages=lc_messages_to_agent_dicts(state["messages"]),
        success_criteria=state.get("success_criteria") or "",
        feedback_on_work=state.get("feedback_on_work"),
        success_criteria_met=state.get("success_criteria_met", False),
        user_input_needed=state.get("user_input_needed", False),
        iteration=state.get("iteration", 0),
        thread_id=state.get("thread_id"),
    )
    lc_eval_full = _evaluator_lc_messages(agent_state)
    # Apply same character budget to evaluator prompt messages
    char_budget = max(0, settings.history_char_limit)
    lc_eval: list[BaseMessage] = []
    used = 0
    for m in reversed(lc_eval_full):
        content = str(getattr(m, "content", "") or "")
        length = len(content)
        if used + length > char_budget:
            break
        lc_eval.append(m)
    lc_eval.reverse()
    _log.info("evaluator: invoking LLM")
    result = evaluator_llm.invoke(lc_eval)
    if not isinstance(result, EvaluatorOutput):
        # Fallback: accept plain-text evaluator outputs by parsing into EvaluatorOutput
        try:
            text = getattr(result, "content", "") if hasattr(result, "content") else str(result)
            result = parse_evaluator_output(text)
        except Exception as exc:
            # Heuristic fallback for non-JSON evaluator outputs (e.g., local Ollama)
            txt = (text or "").strip()
            lower = txt.lower()
            # If we already have a retrieval observation and a non-empty assistant reply, accept.
            has_retrieval_obs = any(
                isinstance(m, SystemMessage) and "Retrieved (Wikipedia):" in str(getattr(m, "content", ""))
                for m in state.get("messages", [])
            )
            has_local_obs = any(
                isinstance(m, SystemMessage) and "Retrieved (Local Docs):" in str(getattr(m, "content", ""))
                for m in state.get("messages", [])
            )
            last_assistant = ""
            for m in reversed(state.get("messages", [])):
                if isinstance(m, AIMessage):
                    last_assistant = str(getattr(m, "content", "") or "")
                    break
            # If criteria demand local docs, reject external-only answers
            criteria_text = (state.get("success_criteria") or "").lower()
            local_only = ("only in local docs" in criteria_text) or ("only in local" in criteria_text)
            # Penalize framework-centric answers for overview tasks
            mentions_framework = ("react " in lower) or ("react:" in lower) or ("chain-of-thought" in lower) or ("framework" in lower)
            overview_like = ("overview" in criteria_text) or ("advantages" in criteria_text)
            if local_only and ("wikipedia" in lower or "http" in lower) and not has_local_obs:
                result = EvaluatorOutput(
                    feedback="Ground in local docs (README.md or docs/developer.md); external sources are not allowed.",
                    success_criteria_met=False,
                    user_input_needed=False,
                )
            elif overview_like and mentions_framework and not has_local_obs:
                result = EvaluatorOutput(
                    feedback="Do not describe frameworks; summarize this repository using README.md / docs/developer.md with one inline citation.",
                    success_criteria_met=False,
                    user_input_needed=False,
                )
            elif (has_retrieval_obs or has_local_obs) and last_assistant:
                result = EvaluatorOutput(
                    feedback="Accepted based on retrieved evidence and a substantive answer.",
                    success_criteria_met=True,
                    user_input_needed=False,
                )
            else:
                contains_source = ("http://" in txt) or ("https://" in txt) or ("wikipedia" in lower) or ("source" in lower)
                asks_question = ("?" in txt) and (("clarify" in lower) or ("would you like" in lower) or ("do you want" in lower))
                mentions_plan_only = ("i will search" in lower) or ("i can browse" in lower) or ("i will look up" in lower)
                # Simple success heuristic: prefer presence of a source or a strong definitional statement
                criteria = criteria_text
                definitional = ("define" in criteria) or ("what is" in criteria) or ("explain" in criteria) or ("overview" in criteria)
                strong_definition = (" is " in lower) or (" refers to " in lower)
                # If local-only, require local obs; else allow web source
                if local_only:
                    success = bool(has_local_obs and last_assistant and not mentions_plan_only)
                else:
                    success = bool(contains_source or (definitional and strong_definition and not mentions_plan_only))
                user_input_needed = bool(asks_question and not success)
                feedback_parts = []
                if mentions_plan_only:
                    feedback_parts.append("Do not narrate plans; call a retrieval tool and include a citation.")
                if definitional and not (contains_source or has_local_obs):
                    feedback_parts.append("Provide at least one citation or ground in local docs as requested.")
                if asks_question and not success:
                    feedback_parts.append("Ask a specific, necessary question only if required to proceed.")
                if not feedback_parts:
                    feedback_parts.append("Improve clarity and ensure the response satisfies the success criteria.")
                result = EvaluatorOutput(
                    feedback=" ".join(feedback_parts),
                    success_criteria_met=success,
                    user_input_needed=user_input_needed,
                )
    feedback_line = f"Evaluator Feedback on this answer: {result.feedback}"
    dt = (time.perf_counter() - t0) * 1000
    _log.info(
        "evaluator: completed in %.1f ms (success=%s user_input=%s)",
        dt,
        result.success_criteria_met,
        result.user_input_needed,
    )
    # Finality metric: evaluator acceptance rate
    try:
        increment_counter(
            "evaluator_judgements_total",
            labels={"accepted": str(bool(result.success_criteria_met)).lower()},
        )
        if result.success_criteria_met:
            increment_counter("evaluator_accept_total")
    except Exception:
        # Metrics must never break the control flow
        pass
    return {
        # Append evaluator feedback as a SystemMessage so UIs don't show it as assistant text
        "messages": [SystemMessage(content=feedback_line)],
        "feedback_on_work": result.feedback,
        "success_criteria_met": result.success_criteria_met,
        "user_input_needed": result.user_input_needed,
        # Increment iteration after each evaluator pass
        "iteration": state.get("iteration", 0) + 1,
    }


def compile_sidekick_graph(
    *,
    llm_worker: Runnable,
    llm_evaluator_structured: Runnable,
    sidekick_tools: list[SidekickTool],
    checkpointer: Any | None = None,
    max_iterations: int | None = None,
    tokens_per_run_limit: int | None = None,
    straight_line: bool | None = None,
):
    """Compile the sidekick graph.

    ``llm_worker`` should be a chat model **with** ``bind_tools`` applied when
    ``sidekick_tools`` is non-empty. Pass a plain model when there are no tools.
    """
    from langgraph.prebuilt import ToolNode

    lc_tool_wrappers = [sidekick_tool_to_langchain(t) for t in sidekick_tools]

    def worker(state: SidekickGraphState) -> dict[str, Any]:
        return _worker_node_fn(state, llm_worker)

    def evaluator(state: SidekickGraphState) -> dict[str, Any]:
        return _evaluator_node_fn(state, llm_evaluator_structured)

    # Router with optional iteration cap (ends when iteration >= max_iterations)
    def _route_evaluator_with_cap(state: SidekickGraphState) -> Literal["worker", "__end__"]:
        if max_iterations is not None and state.get("iteration", 0) >= max_iterations:
            _log.info(
                "iteration cap reached: %s >= %s -> end",
                state.get("iteration", 0),
                max_iterations,
            )
            return "__end__"
        # Financial metric and kill-switch: approximate tokens per run
        # Emit tokens_total when success OR when exceeding soft limit
        try:
            approx_tokens = _approx_tokens_from_messages(state.get("messages", []))
        except Exception:
            approx_tokens = None
        decision = route_evaluator(state)
        if decision == "__end__":
            # Emit tokens_total on natural end
            try:
                if approx_tokens is not None:
                    observe_histogram("tokens_total", float(approx_tokens), labels={})
            except Exception:
                pass
            return decision
        # If still looping, check limit and end if exceeded
        if approx_tokens is not None and tokens_per_run_limit is not None and approx_tokens >= tokens_per_run_limit:
            _log.warning("token cap reached: approx=%s >= limit=%s -> end", approx_tokens, tokens_per_run_limit)
            try:
                increment_counter("token_kills_total")
                observe_histogram("tokens_total", float(approx_tokens), labels={"killed": "true"})
            except Exception:
                pass
            return "__end__"
        # Loop health metric: iterations to success (emit only when evaluator finishes with success)
        if decision == "__end__" and state.get("success_criteria_met"):
            try:
                observe_histogram(
                    "iterations_to_success",
                    float(state.get("iteration", 0)),
                    labels={},
                )
                # Optional: highlight one-shot success (iteration <= 1)
                if state.get("iteration", 0) <= 1:
                    increment_counter("one_shot_success_total")
            except Exception:
                pass
        return decision

    graph_builder = StateGraph(SidekickGraphState)
    graph_builder.add_node("intent", _intent_detection_node)
    graph_builder.add_node("autoretrieve", _auto_retrieve_node)
    graph_builder.add_node("worker", worker)
    if not straight_line:
        graph_builder.add_node("evaluator", evaluator)
    else:
        graph_builder.add_node("finalize", _finalize_node)

    if lc_tool_wrappers:
        graph_builder.add_node("tools", ToolNode(lc_tool_wrappers))
        graph_builder.add_edge(START, "intent")
        graph_builder.add_conditional_edges(
            "intent",
            route_intent,
            {"autoretrieve": "autoretrieve", "worker": "worker"},
        )
        graph_builder.add_edge("autoretrieve", "worker")
        if straight_line:
            # Straight-line: worker → finalize → END (skip evaluator)
            graph_builder.add_edge("worker", "finalize")
            graph_builder.add_edge("finalize", END)
        else:
            graph_builder.add_conditional_edges(
                "worker",
                route_worker,
                {"tools": "tools", "evaluator": "evaluator"},
            )
            graph_builder.add_edge("tools", "worker")
    else:
        graph_builder.add_edge(START, "intent")
        graph_builder.add_conditional_edges(
            "intent",
            route_intent,
            {"autoretrieve": "autoretrieve", "worker": "worker"},
        )
        graph_builder.add_edge("autoretrieve", "worker")
        if straight_line:
            graph_builder.add_edge("worker", "finalize")
            graph_builder.add_edge("finalize", END)
        else:
            graph_builder.add_edge("worker", "evaluator")

    if not straight_line:
        graph_builder.add_conditional_edges(
            "evaluator",
            _route_evaluator_with_cap,
            {"worker": "worker", "__end__": END},
        )

    return graph_builder.compile(checkpointer=checkpointer)


def create_sidekick_graph_from_settings(
    settings: Settings,
    *,
    llm_worker: Runnable,
    llm_evaluator_structured: Runnable,
    checkpointer: Any | None = None,
    browser: BrowserController | None = None,
    include_browser: bool = False,
):
    """Build tools from settings and compile the graph."""
    tools = build_tools(
        settings,
        browser=browser,
        include_browser=include_browser,
    )
    lc_tools = tools
    # Bind tools only if the worker supports it; otherwise, fall back to no tools.
    if lc_tools and hasattr(llm_worker, "bind_tools"):
        try:
            llm_bound = getattr(llm_worker, "bind_tools")(
                [sidekick_tool_to_langchain(t) for t in lc_tools]
            )
        except NotImplementedError:
            # Some providers (e.g., ChatOllama) expose bind_tools but don't implement it.
            # Fall back to no-tools path.
            _log.info("bind_tools not implemented by this model; disabling tools for this run.")
            llm_bound = llm_worker
            lc_tools = []
    else:
        llm_bound = llm_worker
        if lc_tools and not hasattr(llm_worker, "bind_tools"):
            # Disable tools to avoid ToolNode path when worker cannot issue tool calls
            lc_tools = []
    return compile_sidekick_graph(
        llm_worker=llm_bound,
        llm_evaluator_structured=llm_evaluator_structured,
        sidekick_tools=lc_tools,
        checkpointer=checkpointer,
        max_iterations=settings.max_agent_iterations,
        tokens_per_run_limit=settings.tokens_per_run_limit,
        straight_line=True,
    )
