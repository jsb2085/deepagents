"""Tests for the `dcode models` command group."""

from __future__ import annotations

import argparse
import io
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

from rich.console import Console

from deepagents_code.client.commands.models import run_models_command

if TYPE_CHECKING:
    import pytest


def _run_text(args: argparse.Namespace, *, width: int = 200) -> tuple[int, str]:
    buf = io.StringIO()
    test_console = Console(file=buf, highlight=False, width=width)
    with patch("deepagents_code.config.console", test_console):
        code = run_models_command(args)
    return code, buf.getvalue()


def _patch_catalog(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: dict[str, list[str]],
    gateway_specs: list[str],
    gateway_base_url: str | None = None,
) -> None:
    from deepagents_code.client.commands import models as _models

    monkeypatch.setattr(
        "deepagents_code.model_config.get_available_models",
        lambda: available,
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.get_ti_gateway_model_specs",
        lambda: [*gateway_specs],
    )
    active = gateway_base_url is not None or bool(gateway_specs)
    monkeypatch.setattr(
        _models,
        "_gateway_info",
        lambda _specs: {
            "active": active,
            "auth_mode": "tijwt" if active else "api",
            "base_url": gateway_base_url,
            "probed": bool(gateway_specs),
        },
    )


class TestModelsListText:
    """Tests for `dcode models list` text output."""

    def test_lists_grouped_models(self, monkeypatch) -> None:
        args = argparse.Namespace(models_command="list", output_format="text")
        _patch_catalog(
            monkeypatch,
            available={"openai": ["gpt-4o"], "ollama": ["llama3"]},
            gateway_specs=[],
        )
        code, output = _run_text(args)
        assert code == 0
        assert "2 models available" in output
        assert "openai:gpt-4o" in output
        assert "ollama:llama3" in output

    def test_gateway_section_rendered(self, monkeypatch) -> None:
        args = argparse.Namespace(models_command="list", output_format="text")
        _patch_catalog(
            monkeypatch,
            available={"openai": ["gpt-4o", "claude-x"]},
            gateway_specs=["openai:claude-x"],
            gateway_base_url="https://gateway.example/v1",
        )
        code, output = _run_text(args)
        assert code == 0
        assert "TI gateway models" in output
        assert "openai:claude-x" in output

    def test_gateway_probe_failure_note(self, monkeypatch) -> None:
        from deepagents_code.client.commands import models as _models

        args = argparse.Namespace(models_command="list", output_format="text")
        monkeypatch.setattr(
            "deepagents_code.model_config.get_available_models",
            lambda: {"openai": ["gpt-4o"]},
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.get_ti_gateway_model_specs",
            list,
        )
        monkeypatch.setattr(
            _models,
            "_gateway_info",
            lambda _specs: {
                "active": True,
                "auth_mode": "tijwt",
                "base_url": "https://gateway.example/v1",
                "probed": False,
            },
        )
        code, output = _run_text(args)
        assert code == 0
        assert "returned no models" in output


class TestModelsListJson:
    """Tests for `dcode models list --json` output."""

    def test_json_envelope(self, monkeypatch, capsys) -> None:
        args = argparse.Namespace(models_command="list", output_format="json")
        _patch_catalog(
            monkeypatch,
            available={"openai": ["gpt-4o"]},
            gateway_specs=["openai:claude-x"],
            gateway_base_url="https://gateway.example/v1",
        )
        code = run_models_command(args)
        assert code == 0
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["command"] == "models list"
        assert envelope["data"]["models"] == {"openai": ["openai:gpt-4o"]}
        assert envelope["data"]["gateway_models"] == ["openai:claude-x"]
        assert envelope["data"]["auth_mode"] == "tijwt"


class TestModelsDispatch:
    """Tests for `dcode models` group dispatch."""

    def test_bare_group_shows_help(self) -> None:
        args = argparse.Namespace(models_command=None, output_format="text")
        with patch("deepagents_code.ui.show_models_help") as show_help:
            assert run_models_command(args) == 0
        show_help.assert_called_once_with()
