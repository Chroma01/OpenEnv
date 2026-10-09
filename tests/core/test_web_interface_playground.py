# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the default web playground at /web and its environment hooks."""

import gradio as gr
import pytest

from openenv.core.env_server.gradio_ui import _result_html, build_gradio_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.web_interface import (
    _extract_action_fields,
    WebInterfaceManager,
)


class MoveAction(Action):
    action_id: int


class BoardObservation(Observation):
    legal_actions: list[int] = []


class TinyEnv(Environment):
    def reset(self, seed=None, episode_id=None, **kwargs):
        return BoardObservation(legal_actions=[0, 1])

    def step(self, action, timeout_s=None, **kwargs):
        return BoardObservation(legal_actions=[0, 1], reward=1.0, done=True)

    @property
    def state(self):
        return State()


def test_environment_hooks_default_to_nothing():
    env = TinyEnv()
    assert env.render_web({"legal_actions": [0, 1]}) is None
    assert env.web_actions({"legal_actions": [0, 1]}) == []


def test_result_shows_tool_output_and_stats():
    data = {"observation": {"result": {"data": "Hello"}}, "reward": 0.0, "done": False}
    html = _result_html(data, step_count=1)
    assert "Hello" in html
    assert "reward" in html and "done" in html and "step" in html


def test_result_lists_observation_fields():
    data = {
        "observation": {"legal_actions": [0, 1], "metadata": {}},
        "reward": 1.0,
        "done": True,
    }
    html = _result_html(data, step_count=2)
    assert "legal_actions" in html
    assert "metadata" not in html


def test_playground_builds_for_a_plain_env():
    manager = WebInterfaceManager(TinyEnv(), MoveAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    assert isinstance(blocks, gr.Blocks)


def test_playground_lists_mcp_tools():
    echo = pytest.importorskip("echo_env.server.echo_environment")
    manager = WebInterfaceManager(
        echo.EchoEnvironment(), CallToolAction, CallToolObservation
    )
    blocks = build_gradio_app(
        manager, _extract_action_fields(CallToolAction), None, False
    )
    radios = [b for b in blocks.blocks.values() if isinstance(b, gr.Radio)]
    tool_names = {value for radio in radios for _, value in radio.choices}
    assert {"echo_message", "echo_with_length"} <= tool_names


def test_catch_offers_legal_moves_and_a_board():
    pytest.importorskip("open_spiel")
    from openspiel_env.server.openspiel_environment import OpenSpielEnvironment

    env = OpenSpielEnvironment(game_name="catch")
    obs = env.reset().model_dump()
    assert env.web_actions(obs) == [
        ("0 · left", {"action_id": 0}),
        ("1 · stay", {"action_id": 1}),
        ("2 · right", {"action_id": 2}),
    ]
    assert 'aria-label="Catch board"' in env.render_web(obs)
