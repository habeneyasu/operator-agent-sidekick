from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

import gradio as gr
import plotly.graph_objects as go
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from src.config import get_settings
from src.memory.saver import build_memory_saver
from src.memory.session import new_thread_id, thread_config
from src.metrics import get_hist_values, get_metrics_summary, observe_histogram
from src.sidekick import (
    build_graph_with_ollama,
    build_graph_with_openai,
    build_graph_with_openrouter,
)

# ── Design tokens ────────────────────────────────────────────────────────────
_COLORS = {
    "green": "#10B981",
    "yellow": "#F59E0B",
    "red": "#EF4444",
    "blue": "#3B82F6",
    "indigo": "#6366F1",
    "surface": "#0F172A",       # card / panel background
    "surface_alt": "#1E293B",   # slightly lighter surface
    "border": "#334155",
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "grid": "#1E293B",
}

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=_COLORS["surface"],
    plot_bgcolor=_COLORS["surface"],
    font=dict(family="Inter, system-ui, sans-serif", color=_COLORS["text_primary"], size=12),
    margin=dict(l=48, r=16, t=40, b=40),
    xaxis=dict(
        gridcolor=_COLORS["border"],
        linecolor=_COLORS["border"],
        tickfont=dict(color=_COLORS["text_secondary"], size=11),
        title_font=dict(color=_COLORS["text_secondary"], size=11),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor=_COLORS["border"],
        linecolor=_COLORS["border"],
        tickfont=dict(color=_COLORS["text_secondary"], size=11),
        title_font=dict(color=_COLORS["text_secondary"], size=11),
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=_COLORS["border"],
        font=dict(color=_COLORS["text_secondary"], size=11),
    ),
    hoverlabel=dict(
        bgcolor=_COLORS["surface_alt"],
        bordercolor=_COLORS["border"],
        font=dict(color=_COLORS["text_primary"], size=12),
    ),
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_llm_key() -> None:
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_ollama = bool(os.getenv("OLLAMA_BASE_URL"))
    if not (has_openai or has_openrouter or has_ollama):
        raise RuntimeError(
            "Set one of: OPENAI_API_KEY, OPENROUTER_API_KEY(+OPENROUTER_BASE_URL), or OLLAMA_BASE_URL."
        )


def build_graph_for_ui():
    s = get_settings()
    saver = build_memory_saver()
    if s.ollama_base_url and (s.ollama_model_worker or s.ollama_model_evaluator):
        return build_graph_with_ollama(settings=s, checkpointer=saver)
    if s.openrouter_api_key and s.openrouter_base_url:
        return build_graph_with_openrouter(settings=s, checkpointer=saver)
    return build_graph_with_openai(settings=s, checkpointer=saver)


def _messages_to_langchain(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in messages:
        role, content = m.get("role"), m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


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
    t0 = time.perf_counter()
    out = graph.invoke(
        state,
        config={**cfg, "tags": ["ui", "sidekick"], "metadata": {"thread_id": thread_id}},
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    try:
        observe_histogram("e2e_latency_ms", dt_ms, labels={})
    except Exception:
        pass
    assistant_reply = ""
    for m in reversed(out["messages"]):
        if isinstance(m, AIMessage):
            content = str(m.content or "").strip()
            if content:
                assistant_reply = content
                break
    # Last-resort: scan for any non-empty AIMessage content
    if not assistant_reply:
        for m in out["messages"]:
            if isinstance(m, AIMessage) and str(m.content or "").strip():
                assistant_reply = str(m.content).strip()
                break
    updated = [*messages]
    if user_text.strip():
        updated.append({"role": "user", "content": user_text.strip()})
    updated.append({"role": "assistant", "content": assistant_reply or "(no reply)"})
    return updated, ""


# ── KPI card HTML ─────────────────────────────────────────────────────────────

def _threshold_color(value: float, good: float, warn: float, lower_is_better: bool = False) -> str:
    if lower_is_better:
        if value <= good:
            return _COLORS["green"]
        if value <= warn:
            return _COLORS["yellow"]
        return _COLORS["red"]
    if value >= good:
        return _COLORS["green"]
    if value >= warn:
        return _COLORS["yellow"]
    return _COLORS["red"]


def _spark_bar(pct: float, color: str) -> str:
    """Thin progress bar used inside KPI cards."""
    return (
        f'<div style="height:3px;border-radius:2px;background:{_COLORS["border"]};margin-top:6px">'
        f'<div style="height:3px;border-radius:2px;width:{min(pct,100):.1f}%;background:{color}"></div>'
        f"</div>"
    )


def _kpi_card(label: str, value: str, sub: str, color: str, bar_pct: float | None = None) -> str:
    bar_html = _spark_bar(bar_pct, color) if bar_pct is not None else ""
    bg = _COLORS["surface_alt"]
    border = _COLORS["border"]
    txt_pri = _COLORS["text_primary"]
    txt_sec = _COLORS["text_secondary"]
    return (
        f'<div style="background:{bg};border:1px solid {border};border-top:3px solid {color};'
        f'border-radius:8px;padding:14px 16px;margin-bottom:10px">'
        f'<div style="font-size:11px;font-weight:600;letter-spacing:.06em;'
        f'text-transform:uppercase;color:{txt_sec};margin-bottom:4px">{label}</div>'
        f'<div style="font-size:26px;font-weight:700;color:{txt_pri};line-height:1">{value}</div>'
        f'<div style="font-size:11px;color:{txt_sec};margin-top:4px">{sub}</div>'
        f"{bar_html}"
        f"</div>"
    )


def _section_header(title: str) -> str:
    return (
        f'<div style="font-size:11px;font-weight:700;letter-spacing:.08em;'
        f"text-transform:uppercase;color:{_COLORS['text_secondary']};"
        f'border-bottom:1px solid {_COLORS["border"]};padding-bottom:6px;margin:16px 0 10px">'
        f"{title}</div>"
    )


def _build_kpi_html(s: Dict[str, Any]) -> str:
    finality = float(s["finality_pct"])
    it_p95 = float(s["iterations_p95"])
    lat_p50 = float(s["e2e_latency_p50_ms"])
    lat_p95 = float(s["e2e_latency_p95_ms"])
    avg_tokens = float(s["avg_tokens_per_run"])

    c_fin = _threshold_color(finality, good=90.0, warn=75.0)
    c_iter = _threshold_color(it_p95, good=2.0, warn=5.0, lower_is_better=True)
    c_lat = _threshold_color(lat_p95, good=3000.0, warn=8000.0, lower_is_better=True)
    c_tok = _threshold_color(avg_tokens, good=0.0, warn=20000.0, lower_is_better=True)

    lat_p50_s = f"{lat_p50/1000:.2f}s" if lat_p50 >= 1000 else f"{lat_p50:.0f}ms"
    lat_p95_s = f"{lat_p95/1000:.2f}s" if lat_p95 >= 1000 else f"{lat_p95:.0f}ms"

    html = (
        f'<div style="font-family:Inter,system-ui,sans-serif;padding:4px 0">'
        + _section_header("Quality")
        + _kpi_card("Acceptance Rate", f"{finality:.1f}%", "evaluator finality", c_fin, bar_pct=finality)
        + _kpi_card("Iterations p95", f"{it_p95:.1f}", "loops to success", c_iter, bar_pct=min(it_p95 / 8 * 100, 100))
        + _section_header("Performance")
        + _kpi_card("Latency p50", lat_p50_s, "median end-to-end", c_lat)
        + _kpi_card("Latency p95", lat_p95_s, "tail end-to-end", c_lat)
        + _section_header("Cost")
        + _kpi_card("Avg Tokens / Run", f"{avg_tokens:,.0f}", "approx. prompt+completion", c_tok)
        + "</div>"
    )
    return html


# ── Chart builders ────────────────────────────────────────────────────────────

def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=_COLORS["text_secondary"]), x=0),
        annotations=[dict(
            text="No data yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=13, color=_COLORS["text_secondary"]),
        )],
        **_PLOTLY_LAYOUT,
    )
    return fig


def _tool_success_chart(tool_rates: Dict[str, float]) -> go.Figure:
    if not tool_rates:
        return _empty_fig("Tool Reliability")

    tools = sorted(tool_rates.keys())
    rates = [tool_rates[t] for t in tools]
    colors = [
        _COLORS["green"] if r >= 90 else _COLORS["yellow"] if r >= 70 else _COLORS["red"]
        for r in rates
    ]

    fig = go.Figure(go.Bar(
        x=tools,
        y=rates,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{r:.0f}%" for r in rates],
        textposition="outside",
        textfont=dict(size=11, color=_COLORS["text_primary"]),
        hovertemplate="<b>%{x}</b><br>Success rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Tool Reliability", font=dict(size=13, color=_COLORS["text_primary"]), x=0),
        yaxis=dict(range=[0, 115], ticksuffix="%", **_PLOTLY_LAYOUT["yaxis"]),
        xaxis=dict(tickangle=-20, **_PLOTLY_LAYOUT["xaxis"]),
        **{k: v for k, v in _PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def _latency_chart(lat_vals: List[float]) -> go.Figure:
    if not lat_vals:
        return _empty_fig("Latency Distribution (ms)")

    fig = go.Figure(go.Histogram(
        x=lat_vals,
        nbinsx=min(20, max(5, len(lat_vals))),
        marker=dict(color=_COLORS["indigo"], line=dict(color=_COLORS["surface"], width=1)),
        hovertemplate="Latency: %{x:.0f}ms<br>Count: %{y}<extra></extra>",
        name="Latency",
    ))

    # p50 / p95 reference lines
    sorted_vals = sorted(lat_vals)
    def _perc(p: float) -> float:
        i = max(0, min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1)))))
        return sorted_vals[i]

    p50, p95 = _perc(50), _perc(95)
    for val, label, color in [(p50, "p50", _COLORS["green"]), (p95, "p95", _COLORS["yellow"])]:
        fig.add_vline(
            x=val,
            line=dict(color=color, width=1.5, dash="dash"),
            annotation=dict(
                text=f"{label}: {val:.0f}ms",
                font=dict(size=10, color=color),
                bgcolor=_COLORS["surface"],
                borderpad=3,
            ),
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(text="Latency Distribution", font=dict(size=13, color=_COLORS["text_primary"]), x=0),
        xaxis=dict(title="ms", **_PLOTLY_LAYOUT["xaxis"]),
        yaxis=dict(title="Requests", **_PLOTLY_LAYOUT["yaxis"]),
        showlegend=False,
        **{k: v for k, v in _PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def _token_trend_chart(tok_vals: List[float]) -> go.Figure:
    if not tok_vals:
        return _empty_fig("Token Usage per Run")

    x = list(range(1, len(tok_vals) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=tok_vals,
        mode="lines+markers",
        line=dict(color=_COLORS["blue"], width=2),
        marker=dict(size=5, color=_COLORS["blue"]),
        fill="tozeroy",
        fillcolor=f"rgba(59,130,246,0.08)",
        hovertemplate="Run %{x}<br>Tokens: %{y:,.0f}<extra></extra>",
        name="Tokens",
    ))
    avg = sum(tok_vals) / len(tok_vals)
    fig.add_hline(
        y=avg,
        line=dict(color=_COLORS["text_secondary"], width=1, dash="dot"),
        annotation=dict(
            text=f"avg {avg:,.0f}",
            font=dict(size=10, color=_COLORS["text_secondary"]),
            bgcolor=_COLORS["surface"],
            borderpad=3,
        ),
        annotation_position="right",
    )
    fig.update_layout(
        title=dict(text="Token Usage per Run", font=dict(size=13, color=_COLORS["text_primary"]), x=0),
        xaxis=dict(title="Run #", **_PLOTLY_LAYOUT["xaxis"]),
        yaxis=dict(title="Tokens", **_PLOTLY_LAYOUT["yaxis"]),
        showlegend=False,
        **{k: v for k, v in _PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ── Render all metrics ────────────────────────────────────────────────────────

def _render_metrics() -> Tuple[str, Any, Any, Any]:
    s = get_metrics_summary()
    kpi_html = _build_kpi_html(s)
    tool_fig = _tool_success_chart(s["tool_success_rates"])
    lat_fig = _latency_chart(get_hist_values("e2e_latency_ms"))
    tok_fig = _token_trend_chart(get_hist_values("tokens_total"))
    return kpi_html, tool_fig, lat_fig, tok_fig


def _empty_metrics() -> Tuple[str, Any, Any, Any]:
    empty_html = (
        f'<div style="font-family:Inter,system-ui,sans-serif;color:{_COLORS["text_secondary"]};'
        f'font-size:13px;padding:12px 0">No data yet — send a message to populate metrics.</div>'
    )
    return (
        empty_html,
        _empty_fig("Tool Reliability"),
        _empty_fig("Latency Distribution"),
        _empty_fig("Token Usage per Run"),
    )


# ── Layout ────────────────────────────────────────────────────────────────────

def launch() -> None:
    _ensure_llm_key()
    graph = build_graph_for_ui()

    css = """
    .gradio-container { background: #0F172A !important; }
    .gr-panel, .gr-box { background: #1E293B !important; border-color: #334155 !important; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="Sidekick — Operator Agent") as demo:

        gr.HTML(
            f'<div style="font-family:Inter,system-ui,sans-serif;padding:20px 0 8px">'
            f'<div style="font-size:20px;font-weight:700;color:{_COLORS["text_primary"]}">Sidekick</div>'
            f'<div style="font-size:13px;color:{_COLORS["text_secondary"]};margin-top:2px">'
            f"Operator-style AI agent · LangGraph · Worker / Evaluator loop</div>"
            f"</div>"
        )

        with gr.Row(equal_height=False):
            # ── Left: chat ──────────────────────────────────────────────────
            with gr.Column(scale=5, min_width=400):
                success = gr.Textbox(
                    label="Success Criteria",
                    placeholder="Optional — leave blank to use default policy",
                    lines=1,
                )
                thread = gr.State(new_thread_id())
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    layout="bubble",
                    buttons=["copy"],
                )
                with gr.Row():
                    user = gr.Textbox(
                        label="",
                        placeholder="Type a message and press Send…",
                        scale=5,
                        container=False,
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=80)
                reset = gr.Button("↺  New Thread", size="sm", variant="secondary")

            # ── Right: metrics ──────────────────────────────────────────────
            with gr.Column(scale=3, min_width=300):
                gr.HTML(
                    f'<div style="font-size:12px;font-weight:700;letter-spacing:.07em;'
                    f'text-transform:uppercase;color:{_COLORS["text_secondary"]};'
                    f'padding:4px 0 10px;border-bottom:1px solid {_COLORS["border"]};margin-bottom:4px">'
                    f"Observability</div>"
                )
                kpi = gr.HTML(value=_empty_metrics()[0])

                with gr.Tabs():
                    with gr.Tab("Tool Reliability"):
                        tool_plot = gr.Plot(show_label=False)
                    with gr.Tab("Latency"):
                        lat_plot = gr.Plot(show_label=False)
                    with gr.Tab("Token Usage"):
                        tok_plot = gr.Plot(show_label=False)

        # ── Event handlers ──────────────────────────────────────────────────
        def on_send(chat, text, succ, tid):
            if not text.strip():
                kpi_html, t_fig, l_fig, tok_fig = _render_metrics()
                return chat, text, tid, gr.update(value=kpi_html), t_fig, l_fig, tok_fig
            updated_chat, _ = _invoke_once(graph, tid, succ, chat, text)
            kpi_html, t_fig, l_fig, tok_fig = _render_metrics()
            return updated_chat, "", tid, gr.update(value=kpi_html), t_fig, l_fig, tok_fig

        def on_reset():
            kpi_html, t_fig, l_fig, tok_fig = _empty_metrics()
            return [], "", new_thread_id(), gr.update(value=kpi_html), t_fig, l_fig, tok_fig

        outputs = [chatbot, user, thread, kpi, tool_plot, lat_plot, tok_plot]
        send.click(on_send, inputs=[chatbot, user, success, thread], outputs=outputs)
        user.submit(on_send, inputs=[chatbot, user, success, thread], outputs=outputs)
        reset.click(on_reset, inputs=None, outputs=outputs)

    demo.queue().launch(css=css, theme=gr.themes.Base())


if __name__ == "__main__":
    launch()
