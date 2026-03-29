from __future__ import annotations

import os
import time
from typing import Any, List, Tuple, Dict

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from src.config import get_settings
from src.sidekick import build_graph_with_openai, build_graph_with_openrouter
from src.memory.saver import build_memory_saver
from src.memory.session import new_thread_id, thread_config


def _ensure_llm_key() -> None:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY to run the Gradio app.")


def build_graph_for_ui():
    s = get_settings()
    saver = build_memory_saver()
    if os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_BASE_URL"):
        graph = build_graph_with_openrouter(settings=s, checkpointer=saver)
    else:
        graph = build_graph_with_openai(settings=s, checkpointer=saver)
    return graph


def _messages_to_langchain(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    lc: List[BaseMessage] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            lc.append(HumanMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
    return lc


def _invoke_once(
    graph: Runnable,
    thread_id: str,
    success_criteria: str,
    messages: List[Dict[str, str]],
    user_text: str,
) -> Tuple[List[Dict[str, str]], str]:
    history_msgs = _messages_to_langchain(messages)
    if user_text.strip():
        history_msgs.append(HumanMessage(content=user_text.strip()))
    cfg = thread_config(thread_id)
    state = {
        "messages": history_msgs,
        "success_criteria": success_criteria or "Respond helpfully",
        "feedback_on_work": None,
        "success_criteria_met": False,
        "user_input_needed": False,
        "iteration": 0,
        "thread_id": thread_id,
    }
    # Include tags/metadata for LangSmith tracing (if enabled via env)
    out = graph.invoke(
        state,
        config={**cfg, "tags": ["ui", "sidekick"], "metadata": {"thread_id": thread_id}},
    )
    # Find the latest assistant response (the worker's reply) from the transcript
    assistant_reply = ""
    for m in out["messages"][::-1]:
        if isinstance(m, AIMessage):
            assistant_reply = str(m.content or "")
            break
    updated = [*messages]
    if user_text.strip():
        updated.append({"role": "user", "content": user_text.strip()})
    updated.append({"role": "assistant", "content": assistant_reply or "(no reply)"})
    return updated, ""


def launch() -> None:
    _ensure_llm_key()
    graph = build_graph_for_ui()

    with gr.Blocks(title="Browser Sidekick Agent") as demo:
        gr.Markdown("## Browser Sidekick Agent")
        success = gr.Textbox(label="Success Criteria", value="Provide a concise helpful response")
        thread = gr.State(new_thread_id())
        chatbot = gr.Chatbot(label="Conversation")
        user = gr.Textbox(label="Your message", placeholder="Ask something...")
        send = gr.Button("Send", variant="primary")
        reset = gr.Button("Reset Thread")

        def on_send(chat, text, succ, tid):
            if not text.strip():
                return chat, "", tid
            updated_chat, _ = _invoke_once(graph, tid, succ, chat, text)
            return updated_chat, "", tid

        def on_reset():
            return [], "", new_thread_id()

        send.click(on_send, inputs=[chatbot, user, success, thread], outputs=[chatbot, user, thread])
        reset.click(on_reset, inputs=None, outputs=[chatbot, user, thread])

    demo.queue().launch()


if __name__ == "__main__":
    launch()

