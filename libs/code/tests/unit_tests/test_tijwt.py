"""Unit tests for the TI JWT (Kerberos ticket) auth toggle."""

from __future__ import annotations

import subprocess
import threading
from typing import Any

import pytest

from deepagents_code import tijwt
from deepagents_code.tijwt import TIJWTError, TIJWTManager, normalize_auth_mode


class TestNormalizeAuthMode:
    """Canonicalization of user-supplied auth-mode values."""

    def test_api_variants(self) -> None:
        assert normalize_auth_mode("api") == "api"
        assert normalize_auth_mode(" API ") == "api"

    @pytest.mark.parametrize(
        "raw", ["tijwt", "TIJWT", "kerberos", "Kerberos", "ti", " TI "]
    )
    def test_tijwt_aliases(self, raw: str) -> None:
        assert normalize_auth_mode(raw) == "tijwt"

    def test_empty_is_none(self) -> None:
        assert normalize_auth_mode(None) is None
        assert normalize_auth_mode("") is None
        assert normalize_auth_mode("   ") is None

    def test_invalid_raises(self) -> None:
        with pytest.raises(TIJWTError):
            normalize_auth_mode("oauth")


class TestTIJWTManager:
    """Caching, refresh, and fetcher-failure behavior."""

    def _manager_with_fake_fetch(
        self,
        tokens: list[str],
        monkeypatch: pytest.MonkeyPatch | None = None,
        *,
        threshold: int = 300,
        ttl: int = 3600,
    ) -> tuple[TIJWTManager, list[int]]:
        calls: list[int] = []

        manager = TIJWTManager(
            refresh_threshold_seconds=threshold,
            token_ttl_seconds=ttl,
            fetch_command=("fake-fetch",),
        )

        def _fake() -> str:
            calls.append(1)
            return tokens[min(len(calls) - 1, len(tokens) - 1)]

        if monkeypatch is not None:
            monkeypatch.setattr(manager, "_fetch_fresh_token", _fake)
        else:
            manager._fetch_fresh_token = _fake  # noqa: SLF001 # test double
        return manager, calls

    def test_caches_token_until_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager, calls = self._manager_with_fake_fetch(["tok1"], monkeypatch)
        assert manager.get_token() == "tok1"
        assert manager.get_token() == "tok1"
        assert len(calls) == 1

    def test_refreshes_when_inside_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager, calls = self._manager_with_fake_fetch(
            ["tok1", "tok2"], monkeypatch, ttl=1000
        )
        assert manager.get_token() == "tok1"
        assert len(calls) == 1
        # Move time to within the 300s refresh window of the 1000s TTL.
        start = manager.expires_at or 0.0
        monkeypatch.setattr(tijwt.time, "time", lambda: start - 1000 + 800)
        assert manager.get_token() == "tok2"
        assert len(calls) == 2

    def test_force_refresh_and_clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager, calls = self._manager_with_fake_fetch(
            ["a", "b", "c"], monkeypatch
        )
        assert manager.get_token() == "a"
        assert manager.force_refresh() == "b"
        assert len(calls) == 2
        manager.clear()
        assert manager.get_token() == "c"
        assert len(calls) == 3

    def test_empty_output_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = TIJWTManager(fetch_command=("fake",))

        class _Result:
            stdout = "   \n"

        monkeypatch.setattr(
            tijwt.subprocess, "run", lambda *args, **kwargs: _Result()
        )
        with pytest.raises(TIJWTError):
            manager.get_token()

    def test_fetcher_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = TIJWTManager(fetch_command=("fake",))

        def _boom(*args: object, **kwargs: object) -> object:
            raise subprocess.CalledProcessError(1, "fake")

        monkeypatch.setattr(tijwt.subprocess, "run", _boom)
        with pytest.raises(TIJWTError):
            manager.get_token()

    def test_concurrent_calls_fetch_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager, calls = self._manager_with_fake_fetch(["tok"], monkeypatch)
        results: list[str] = []

        def _work() -> None:
            results.append(manager.get_token())

        threads = [threading.Thread(target=_work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == ["tok"] * 8
        assert len(calls) == 1


class TestResolveAuthMode:
    """Precedence: CLI flag > env > config.toml > default."""

    def test_default_is_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPAGENTS_CODE_AUTH_MODE", raising=False)
        monkeypatch.delenv("TI_AUTH_MODE", raising=False)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        assert tijwt.resolve_auth_mode() == "api"

    def test_cli_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        assert tijwt.resolve_auth_mode(cli_value="api") == "api"

    def test_env_fallback_ti_auth_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPAGENTS_CODE_AUTH_MODE", raising=False)
        monkeypatch.setenv("TI_AUTH_MODE", "kerberos")
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        assert tijwt.resolve_auth_mode() == "tijwt"

    def test_config_toml_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPAGENTS_CODE_AUTH_MODE", raising=False)
        monkeypatch.delenv("TI_AUTH_MODE", raising=False)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml",
            lambda: {"models": {"auth_mode": "kerberos"}},
        )
        assert tijwt.resolve_auth_mode() == "tijwt"

    def test_invalid_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "oauth")
        with pytest.raises(TIJWTError):
            tijwt.resolve_auth_mode()


class TestModelWiring:
    """TI JWT mode marks providers configured and injects the bearer token."""

    def test_auth_status_configured_in_tijwt_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_OPENAI_API_KEY", raising=False)
        from deepagents_code.model_config import (
            ModelConfig,
            ProviderAuthState,
            get_provider_auth_status,
        )

        monkeypatch.setattr(ModelConfig, "load", classmethod(lambda cls: cls()))
        status = get_provider_auth_status("openai")
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.as_legacy_bool() is True

    def test_provider_kwargs_use_tijwt_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "kerberos")
        monkeypatch.setattr(tijwt, "get_tijwt_token", lambda: "jwt-123")
        from deepagents_code import config as _config
        from deepagents_code.model_config import ModelConfig

        class _FakeConfig:
            def get_kwargs(
                self, provider: str, *, model_name: str | None = None
            ) -> dict[str, Any]:
                return {}

            def get_base_url(self, provider: str) -> str | None:
                return None

            def get_api_key_env(self, provider: str) -> str | None:
                return None

        monkeypatch.setattr(ModelConfig, "load", classmethod(lambda cls: _FakeConfig()))
        monkeypatch.setattr(
            _config, "_read_config_toml_retries", lambda: None
        )
        kwargs = _config._get_provider_kwargs("openai", model_name="gpt-5.5")
        assert kwargs["api_key"] == "jwt-123"

    def test_ollama_uses_authorization_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setattr(tijwt, "get_tijwt_token", lambda: "jwt-ollama")
        from deepagents_code import config as _config
        from deepagents_code.model_config import ModelConfig

        class _FakeConfig:
            def get_kwargs(
                self, provider: str, *, model_name: str | None = None
            ) -> dict[str, Any]:
                return {}

            def get_base_url(self, provider: str) -> str | None:
                return None

            def get_api_key_env(self, provider: str) -> str | None:
                return None

        monkeypatch.setattr(ModelConfig, "load", classmethod(lambda cls: _FakeConfig()))
        monkeypatch.setattr(_config, "_read_config_toml_retries", lambda: None)
        kwargs = _config._get_provider_kwargs("ollama")
        headers = kwargs["client_kwargs"]["headers"]
        assert headers["Authorization"] == "Bearer jwt-ollama"
