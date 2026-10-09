# SPDX-License-Identifier: BSD-3-Clause

"""
Gradio-based web UI for OpenEnv environments.

Mounted at /web via gr.mount_gradio_app() from create_web_interface_app() when
ENABLE_WEB_INTERFACE is set. The page follows the loop an agent runs: reset,
take an action, read the result, with the same call in Python next to it.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

import gradio as gr

from .mcp_environment import MCPEnvironment
from .mcp_types import ListToolsAction
from .types import EnvironmentMetadata


def get_gradio_display_title(
    metadata: Optional[EnvironmentMetadata],
    fallback: str = "OpenEnv Environment",
) -> str:
    """Return the title used for the Gradio app (browser tab and Blocks)."""
    name = metadata.name if metadata else fallback
    return f"OpenEnv Agentic Environment: {name}"


def _header_html(metadata: Optional[EnvironmentMetadata], kind: str) -> str:
    name = metadata.name if metadata else "environment"
    title = name.removesuffix("_env").replace("_", " ").title()
    description = metadata.description if metadata else ""
    return (
        f'<div class="oe-header"><span class="oe-eyebrow">{html.escape(name)} · {kind}</span>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>"
    )


def _step_title(number: int, title: str, text: str) -> str:
    return f'<div class="oe-step"><h2>{number}. {title}</h2><p>{text}</p></div>'


def _short(value: Any, limit: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _result_html(data: Dict[str, Any], step_count: int) -> str:
    """The step's output, then reward, done and step."""
    obs = data.get("observation", {}) or {}
    if "result" in obs:
        result = obs["result"]
        main = result.get("data", result) if isinstance(result, dict) else result
        body = f'<pre class="oe-output">{html.escape(_short(main, 2000))}</pre>'
    else:
        rows = "".join(
            f"<div><span>{html.escape(k)}</span><code>{html.escape(_short(v))}</code></div>"
            for k, v in obs.items()
            if k not in ("metadata", "done", "reward") and v not in (None, "", [], {})
        )
        body = f'<div class="oe-fields">{rows}</div>' if rows else ""
    stats = "".join(
        f"<div><span>{label}</span><code>{html.escape(str(value))}</code></div>"
        for label, value in (
            ("reward", data.get("reward")),
            ("done", str(data.get("done", False)).lower()),
            ("step", step_count),
        )
    )
    return f'<div class="oe-result">{body}<div class="oe-stats">{stats}</div></div>'


def _episode_html(entries: List[List[str]]) -> str:
    if not entries:
        items = '<p class="oe-muted">Nothing yet. Reset, then run a step.</p>'
    else:
        items = (
            "<ol>"
            + "".join(
                f"<li><span>{html.escape(badge)}</span><div><code>{html.escape(call)}</code>"
                f"<small>{html.escape(result)}</small></div></li>"
                for badge, call, result in entries
            )
            + "</ol>"
        )
    return f'<div class="oe-episode"><h2>Episode</h2>{items}</div>'


def _list_tools(env: Any) -> List[Any]:
    return list(env.step(ListToolsAction()).tools)


def _input_for(name: str, schema: Dict[str, Any], label_suffix: str = "") -> Any:
    kind = schema.get("type", "string")
    label = f"{name} ({kind}){label_suffix}"
    if "enum" in schema:
        return gr.Dropdown(choices=schema["enum"], label=label)
    if kind == "boolean":
        return gr.Checkbox(label=label)
    if kind in ("integer", "number"):
        return gr.Number(label=label, precision=0 if kind == "integer" else None)
    return gr.Textbox(label=label, lines=1)


def _field_input(field: Dict[str, Any]) -> Any:
    """A form input for one action field."""
    label = f"{field['name']} ({field['type']})"
    default = field.get("default_value")
    if field["type"] == "checkbox":
        return gr.Checkbox(label=label, value=bool(default))
    if field["type"] == "number":
        return gr.Number(label=label, value=default)
    if field["type"] == "select":
        return gr.Dropdown(choices=field.get("choices") or [], label=label)
    return gr.Textbox(
        label=label,
        value=default if isinstance(default, str) else None,
        lines=4 if field["type"] in ("textarea", "tensor") else 1,
    )


def build_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """
    Build a Gradio Blocks app for the OpenEnv web interface.

    Args:
        web_manager: WebInterfaceManager (reset/step_environment, get_state).
        action_fields: Field dicts from _extract_action_fields(action_cls).
        metadata: Environment metadata for README/name.
        is_chat_env: If True, single message textbox; else form from action_fields.
        title: App title (overridden by metadata.name when present; see get_gradio_display_title).
        quick_start_md: Optional Quick Start markdown (class names already replaced).

    Returns:
        gr.Blocks to mount with gr.mount_gradio_app(app, blocks, path="/web").
    """
    env = web_manager.env
    tools = _list_tools(env) if isinstance(env, MCPEnvironment) else []
    is_mcp = bool(tools)
    kind = "MCP environment" if is_mcp else "environment"

    def render(data: Dict[str, Any]):
        """The env's drawing and its one-click actions for this observation."""
        obs = data.get("observation", {}) or {}
        drawing = env.render_web(obs)
        actions = env.web_actions(obs) if not data.get("done") else []
        return (
            gr.update(value=drawing or "", visible=bool(drawing)),
            gr.update(
                choices=[(label, str(i)) for i, (label, _) in enumerate(actions)],
                value=None,
                visible=bool(actions),
            ),
            [action for _, action in actions],
        )

    async def reset_env(entries):
        data = await web_manager.reset_environment()
        entries = [["R", "reset()", "new episode"]]
        return (
            *render(data),
            "",
            json.dumps(data, indent=2),
            _episode_html(entries),
            entries,
        )

    async def run_step(action: Dict[str, Any], call: str, entries):
        if not entries:
            await web_manager.reset_environment()
            entries = [["R", "reset()", "new episode"]]
        try:
            data = await web_manager.step_environment(action)
        except Exception as e:
            raise gr.Error(str(e))
        count = web_manager.episode_state.step_count
        obs = data.get("observation", {}) or {}
        result = obs.get("result", {})
        summary = result.get("data", result) if isinstance(result, dict) else result
        entries = entries + [
            [str(count), call, _short(summary or f"reward {data.get('reward')}")]
        ]
        return (
            *render(data),
            _result_html(data, count),
            json.dumps(data, indent=2),
            _episode_html(entries),
            entries,
        )

    with gr.Blocks(title=get_gradio_display_title(metadata, fallback=title)) as demo:
        entries_state = gr.State([])
        gr.HTML(_header_html(metadata, kind))
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=360):
                with gr.Row(elem_classes="oe-card oe-row", equal_height=True):
                    gr.HTML(
                        _step_title(
                            1,
                            "Start an episode",
                            "<code>reset()</code> returns the first observation.",
                        )
                    )
                    reset_btn = gr.Button(
                        "Reset", variant="secondary", scale=0, min_width=110
                    )

                with gr.Column(elem_classes="oe-card"):
                    if is_mcp:
                        gr.HTML(
                            _step_title(
                                2,
                                "Call a tool",
                                "<code>step()</code> sends a tool call. Pick one:",
                            )
                        )
                    else:
                        gr.HTML(
                            _step_title(
                                2,
                                "Take an action",
                                "<code>step()</code> sends an action.",
                            )
                        )
                    with gr.Row(equal_height=False):
                        visual = gr.HTML(visible=False, elem_classes="oe-visual")
                        with gr.Column(min_width=260):
                            quick = gr.Radio(
                                choices=[],
                                visible=False,
                                label="Legal actions",
                                elem_classes="oe-actions",
                            )
                            if is_mcp:
                                tool_names = [t.name for t in tools]
                                tool_choice = gr.Radio(
                                    choices=[
                                        (
                                            f"{t.name}: {t.description.strip().splitlines()[0]}",
                                            t.name,
                                        )
                                        for t in tools
                                    ],
                                    value=tool_names[0] if tool_names else None,
                                    show_label=False,
                                    elem_classes="oe-tools",
                                )
                                arg_inputs: List[Any] = []
                                arg_keys: List[tuple] = []
                                groups = []
                                for i, tool in enumerate(tools):
                                    with gr.Column(visible=i == 0) as group:
                                        for arg, schema in tool.input_schema.get(
                                            "properties", {}
                                        ).items():
                                            arg_inputs.append(_input_for(arg, schema))
                                            arg_keys.append((tool.name, arg))
                                    groups.append(group)
                                tool_choice.change(
                                    lambda name: [
                                        gr.update(visible=t.name == name) for t in tools
                                    ],
                                    inputs=tool_choice,
                                    outputs=groups,
                                )

                                async def step_fn(entries, name, *values):
                                    arguments = {
                                        arg: value
                                        for (tool, arg), value in zip(arg_keys, values)
                                        if tool == name and value not in (None, "")
                                    }
                                    action = {
                                        "type": "call_tool",
                                        "tool_name": name,
                                        "arguments": arguments,
                                    }
                                    args = ", ".join(
                                        f"{k}={json.dumps(v)}"
                                        for k, v in arguments.items()
                                    )
                                    return await run_step(
                                        action, f"{name}({args})", entries
                                    )

                                step_inputs = [entries_state, tool_choice, *arg_inputs]
                            else:
                                fields = [
                                    f for f in action_fields if f["name"] != "type"
                                ]
                                required = [f for f in fields if f.get("required")]
                                optional = [f for f in fields if not f.get("required")]
                                fields = required + optional
                                field_inputs = [_field_input(f) for f in required]
                                if optional:
                                    with gr.Accordion(
                                        "More fields", open=False, elem_classes="oe-raw"
                                    ):
                                        field_inputs += [
                                            _field_input(f) for f in optional
                                        ]

                                async def step_fn(entries, *values):
                                    action = {}
                                    for field, value in zip(fields, values):
                                        if field["type"] == "checkbox":
                                            action[field["name"]] = bool(value)
                                        elif value not in (None, ""):
                                            action[field["name"]] = value
                                    args = ", ".join(
                                        f"{f['name']}={json.dumps(action[f['name']])}"
                                        for f in fields
                                        if f["name"] in action and f.get("required")
                                    )
                                    return await run_step(
                                        action, f"step({args})", entries
                                    )

                                step_inputs = [entries_state, *field_inputs]

                            step_btn = gr.Button(
                                "Run step", variant="primary", elem_classes="oe-run"
                            )
                    result = gr.HTML()
                    with gr.Accordion("raw JSON", open=False, elem_classes="oe-raw"):
                        raw_json = gr.Code(
                            language="json", interactive=False, show_label=False
                        )

            with gr.Column(scale=2, min_width=320):
                with gr.Column(elem_classes="oe-card"):
                    episode = gr.HTML(_episode_html([]))
                if quick_start_md:
                    with gr.Column(elem_classes="oe-card oe-code"):
                        gr.HTML(
                            '<div class="oe-step"><h2>Same call in Python</h2></div>'
                        )
                        gr.Markdown(
                            quick_start_md.replace(
                                "### Connect to this environment\n\n", ""
                            )
                        )

        if metadata and metadata.readme_content:
            with gr.Accordion("README", open=False, elem_classes="oe-raw"):
                gr.Markdown(metadata.readme_content)

        quick_state = gr.State([])

        async def quick_step(entries, actions, choice):
            action = actions[int(choice)]
            args = ", ".join(f"{k}={json.dumps(v)}" for k, v in action.items())
            return await run_step(action, f"step({args})", entries)

        outputs = [visual, quick, quick_state, result, raw_json, episode, entries_state]
        reset_btn.click(reset_env, inputs=[entries_state], outputs=outputs)
        step_btn.click(step_fn, inputs=step_inputs, outputs=outputs)
        quick.input(
            quick_step, inputs=[entries_state, quick_state, quick], outputs=outputs
        )

    return demo
