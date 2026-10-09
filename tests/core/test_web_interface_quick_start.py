# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Quick Start markdown shown next to the web interface."""

import importlib
import sys
import textwrap

import pytest

from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import EnvironmentMetadata
from openenv.core.env_server.web_interface import get_quick_start_markdown


@pytest.fixture
def demo_env(tmp_path, monkeypatch):
    """A tiny environment package that exports a client and an action."""
    package = tmp_path / "demo_env"
    package.mkdir()
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """
            from openenv.core.env_client import EnvClient
            from openenv.core.env_server.types import Action

            class DemoAction(Action):
                text: str
                count: int = 1

            class DemoEnv(EnvClient):
                pass
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module("demo_env")
    sys.modules.pop("demo_env", None)


def _metadata(name):
    return EnvironmentMetadata(name=name, description="")


def test_uses_exported_client_and_real_action_fields(demo_env, monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)

    md = get_quick_start_markdown(
        _metadata("demo_env"), demo_env.DemoAction, CallToolObservation
    )

    assert "from demo_env import DemoAction, DemoEnv" in md
    assert 'DemoEnv(base_url="http://localhost:8000")' in md
    assert 'env.step(DemoAction(text="..."))' in md
    assert "pip install" not in md


def test_on_a_space_uses_its_url_and_install_line(demo_env, monkeypatch):
    monkeypatch.setenv("SPACE_ID", "openenv/demo_env")
    monkeypatch.setenv("SPACE_HOST", "openenv-demo-env.hf.space")

    md = get_quick_start_markdown(
        _metadata("demo_env"), demo_env.DemoAction, CallToolObservation
    )

    assert "pip install git+https://huggingface.co/spaces/openenv/demo_env" in md
    assert 'DemoEnv(base_url="https://openenv-demo-env.hf.space")' in md


def test_mcp_env_lists_tools(demo_env, monkeypatch):
    monkeypatch.delenv("SPACE_HOST", raising=False)

    md = get_quick_start_markdown(
        _metadata("demo_env"), CallToolAction, CallToolObservation
    )

    assert "from demo_env import DemoEnv" in md
    assert "env.list_tools()" in md
    assert "CallToolAction(" not in md


def test_env_without_client_package_points_to_readme(monkeypatch):
    monkeypatch.delenv("SPACE_HOST", raising=False)

    md = get_quick_start_markdown(
        _metadata("no_such_env_package"), CallToolAction, CallToolObservation
    )

    assert "README" in md
    assert "```python" not in md
