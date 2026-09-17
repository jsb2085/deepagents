"""Unit tests for the TI JWT (Kerberos ticket) auth toggle."""

from __future__ import annotations

import subprocess
import threading
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from deepagents_code import tijwt
from deepagents_code.tijwt import TIJWTError, TIJWTManager, normalize_auth_mode

if TYPE_CHECKING:
    from urllib.request import Request as _Request


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


class TestGatewayResolution:
    """Gateway endpoint and team-ID resolution for `tijwt` mode."""

    def test_base_url_defaults_to_gateway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_BASE_URL", raising=False)
        monkeypatch.delenv("TI_BASE_URL", raising=False)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        assert tijwt.resolve_base_url() == tijwt.TI_GATEWAY_DEFAULT_BASE_URL

    def test_base_url_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TI_BASE_URL", "https://gateway.example/v1")
        assert tijwt.resolve_base_url() == "https://gateway.example/v1"

    def test_team_id_prefers_litellm_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_TEAM_ID", raising=False)
        monkeypatch.delenv("TI_TEAM_ID", raising=False)
        monkeypatch.setenv("LITELLM_TEAM_ID", "MY_TEAM")
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        assert tijwt.resolve_team_id() == "MY_TEAM"

    def test_team_id_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_TEAM_ID", raising=False)
        monkeypatch.delenv("TI_TEAM_ID", raising=False)
        monkeypatch.delenv("LITELLM_TEAM_ID", raising=False)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        assert tijwt.resolve_team_id() == tijwt.TI_DEFAULT_TEAM_ID

    def test_team_headers_empty_in_api_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "api")
        assert tijwt.team_headers() == {}


class TestGatewayKwargs:
    """Gateway endpoint + team headers land in model kwargs in `tijwt` mode."""

    def _kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setattr(tijwt, "get_tijwt_token", lambda: "jwt-xyz")
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_BASE_URL", raising=False)
        monkeypatch.delenv("TI_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_TEAM_ID", raising=False)
        monkeypatch.delenv("TI_TEAM_ID", raising=False)
        monkeypatch.delenv("LITELLM_TEAM_ID", raising=False)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        from deepagents_code import config as _config
        from deepagents_code.model_config import ModelConfig

        class _FakeConfig:
            def get_kwargs(
                self, provider: str, *, model_name: str | None = None
            ) -> dict[str, Any]:
                return {}

            def get_base_url(self, provider: str) -> str | None:
                return base_url

            def get_api_key_env(self, provider: str) -> str | None:
                return None

        monkeypatch.setattr(ModelConfig, "load", classmethod(lambda cls: _FakeConfig()))
        monkeypatch.setattr(_config, "_read_config_toml_retries", lambda: None)
        return _config._get_provider_kwargs(provider)

    def test_gateway_base_url_and_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kwargs = self._kwargs(monkeypatch, "openai")
        assert kwargs["base_url"] == tijwt.TI_GATEWAY_DEFAULT_BASE_URL
        team = {tijwt.TI_TEAM_ID_HEADER: tijwt.TI_DEFAULT_TEAM_ID}
        assert kwargs["default_headers"] == team
        assert kwargs["extra_headers"] == team

    def test_explicit_base_url_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs = self._kwargs(
            monkeypatch, "openai", base_url="https://custom.example/v1"
        )
        assert kwargs["base_url"] == "https://custom.example/v1"

    def test_litellm_gets_api_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs = self._kwargs(monkeypatch, "litellm")
        assert kwargs["api_base"] == tijwt.TI_GATEWAY_DEFAULT_BASE_URL

    def test_user_headers_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setattr(tijwt, "get_tijwt_token", lambda: "jwt-xyz")
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_TEAM_ID", raising=False)
        monkeypatch.delenv("TI_TEAM_ID", raising=False)
        monkeypatch.delenv("LITELLM_TEAM_ID", raising=False)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        from deepagents_code import config as _config

        out = _config._apply_tijwt_auth_kwargs(
            "openai",
            {
                "base_url": "https://custom.example/v1",
                "default_headers": {"x-other": "1"},
            },
        )
        assert out["default_headers"]["x-other"] == "1"
        assert (
            out["default_headers"][tijwt.TI_TEAM_ID_HEADER]
            == tijwt.TI_DEFAULT_TEAM_ID
        )
        assert out["base_url"] == "https://custom.example/v1"


class TestVerifySsl:
    """TLS-verification opt-out for the TI gateway (default: verify on)."""

    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_VERIFY_SSL", raising=False)
        monkeypatch.delenv("TI_VERIFY_SSL", raising=False)

    def test_default_verifies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_env(monkeypatch)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        assert tijwt.resolve_verify_ssl() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off"])
    def test_falsy_disables(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("TI_VERIFY_SSL", raw)
        assert tijwt.resolve_verify_ssl() is False

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on"])
    def test_truthy_enables(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("TI_VERIFY_SSL", raw)
        assert tijwt.resolve_verify_ssl() is True

    def test_unrecognized_falls_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TI_VERIFY_SSL", "maybe")
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        assert tijwt.resolve_verify_ssl() is True

    def test_config_toml_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_env(monkeypatch)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml",
            lambda: {"models": {"ti_verify_ssl": False}},
        )
        assert tijwt.resolve_verify_ssl() is False

    def test_verify_on_adds_no_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setattr(tijwt, "get_tijwt_token", lambda: "jwt-xyz")
        self._clear_env(monkeypatch)
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_config_toml", lambda: {}
        )
        from deepagents_code import config as _config

        out = _config._apply_tijwt_auth_kwargs("openai", {})
        assert "http_client" not in out
        assert "http_async_client" not in out

    def test_verify_off_injects_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        httpx = pytest.importorskip("httpx")
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setenv("TI_VERIFY_SSL", "false")
        monkeypatch.setattr(tijwt, "get_tijwt_token", lambda: "jwt-xyz")
        from deepagents_code import config as _config

        out = _config._apply_tijwt_auth_kwargs("openai", {})
        assert isinstance(out["http_client"], httpx.Client)
        assert isinstance(out["http_async_client"], httpx.AsyncClient)
        # `httpx.Client` exposes no public `verify` flag; `verify=False`
        # surfaces as a non-verifying SSL context on the underlying pool.
        import ssl as _ssl

        for client in (out["http_client"], out["http_async_client"]):
            pool = client._transport._pool  # private access, assertion only
            context = pool._ssl_context  # private access, assertion only
            assert context.verify_mode == _ssl.CERT_NONE
            assert context.check_hostname is False

    def test_explicit_clients_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("httpx")
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setenv("TI_VERIFY_SSL", "0")
        monkeypatch.setattr(tijwt, "get_tijwt_token", lambda: "jwt-xyz")
        from deepagents_code import config as _config

        sentinel = object()
        out = _config._apply_tijwt_auth_kwargs(
            "openai", {"http_client": sentinel}
        )
        assert out["http_client"] is sentinel
        assert "http_async_client" in out


class TestListGatewayModels:
    """Live `/v1/models` probe behavior."""

    def _tijwt_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setattr(tijwt, "get_tijwt_token", lambda: "jwt-probe")
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_BASE_URL", raising=False)
        monkeypatch.delenv("TI_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_TI_TEAM_ID", raising=False)
        monkeypatch.delenv("TI_TEAM_ID", raising=False)
        monkeypatch.delenv("LITELLM_TEAM_ID", raising=False)
        monkeypatch.setattr("deepagents_code.config_manifest.load_config_toml", dict)

    def _fake_urlopen(self, body: bytes) -> tuple[Any, dict[str, str | None]]:
        """Build a `urlopen` stub serving `body` plus its recorded request.

        Returns:
            `(fake_urlopen, seen)` where `seen` maps `"url"`,
                `"auth"`, and `"team"` to the observed request values.
        """
        from unittest.mock import MagicMock

        seen: dict[str, str | None] = {}

        def fake_urlopen(request: _Request, **_kwargs: Any) -> MagicMock:
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            seen["team"] = request.get_header("X-litellm-team-id")
            response = MagicMock()
            response.read.return_value = body
            response.__enter__.return_value = response
            return response

        return fake_urlopen, seen

    def test_success_returns_sorted_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        self._tijwt_env(monkeypatch)
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {"id": "claude-opus-4-6", "object": "model"},
                    {"id": "gpt-4o", "object": "model"},
                    {"id": 123, "object": "model"},
                    {"nope": True},
                ],
            }
        ).encode()
        fake_urlopen, seen = self._fake_urlopen(body)
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        assert tijwt.list_gateway_models() == ["claude-opus-4-6", "gpt-4o"]
        assert seen["url"] == f"{tijwt.TI_GATEWAY_DEFAULT_BASE_URL}/v1/models"
        assert seen["auth"] == "Bearer jwt-probe"
        assert seen["team"] == tijwt.TI_DEFAULT_TEAM_ID

    def test_http_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from urllib.error import URLError

        self._tijwt_env(monkeypatch)

        def fake_urlopen(*_args: Any, **_kwargs: Any) -> NoReturn:
            msg = "connection refused"
            raise URLError(msg)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        assert tijwt.list_gateway_models() == []

    def test_token_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "tijwt")
        monkeypatch.setattr("deepagents_code.config_manifest.load_config_toml", dict)

        def boom() -> str:
            msg = "kerberos ticket expired"
            raise TIJWTError(msg)

        monkeypatch.setattr(tijwt, "get_tijwt_token", boom)
        assert tijwt.list_gateway_models() == []

    def test_non_tijwt_mode_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", "api")
        monkeypatch.setattr("deepagents_code.config_manifest.load_config_toml", dict)
        with pytest.raises(TIJWTError):
            tijwt.list_gateway_models()


class TestGatewayModelMerge:
    """Merging live gateway IDs into the model catalog."""

    def _tijwt_env(
        self, monkeypatch: pytest.MonkeyPatch, *, mode: str = "tijwt"
    ) -> None:
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTH_MODE", mode)
        monkeypatch.setattr("deepagents_code.config_manifest.load_config_toml", dict)

    def _stub_gateway_models(
        self, monkeypatch: pytest.MonkeyPatch, models: list[str]
    ) -> None:
        def stub(*_args: Any, **_kwargs: Any) -> list[str]:
            return list(models)

        monkeypatch.setattr(tijwt, "list_gateway_models", stub)

    def test_merge_prefers_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from deepagents_code import model_config as _mc

        self._tijwt_env(monkeypatch)
        _mc.clear_caches()
        self._stub_gateway_models(monkeypatch, ["b-model", "a-model"])
        available: dict[str, list[str]] = {"openai": ["gpt-4o"], "litellm": []}
        _mc._merge_ti_gateway_models(available, _mc.ModelConfig.load())
        assert available["openai"] == ["gpt-4o", "b-model", "a-model"]
        _mc.clear_caches()

    def test_merge_falls_back_to_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from deepagents_code import model_config as _mc

        self._tijwt_env(monkeypatch)
        _mc.clear_caches()
        self._stub_gateway_models(monkeypatch, ["claude-x"])
        available: dict[str, list[str]] = {"litellm": ["existing"]}
        _mc._merge_ti_gateway_models(available, _mc.ModelConfig.load())
        assert available["litellm"] == ["existing", "claude-x"]
        _mc.clear_caches()

    def test_merge_skipped_without_compatible_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_code import model_config as _mc

        self._tijwt_env(monkeypatch)
        _mc.clear_caches()
        self._stub_gateway_models(monkeypatch, ["m"])
        available: dict[str, list[str]] = {"ollama": ["llama3"]}
        _mc._merge_ti_gateway_models(available, _mc.ModelConfig.load())
        assert available == {"ollama": ["llama3"]}
        _mc.clear_caches()

    def test_merge_inactive_in_api_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from deepagents_code import model_config as _mc

        self._tijwt_env(monkeypatch, mode="api")
        _mc.clear_caches()
        available: dict[str, list[str]] = {"openai": ["gpt-4o"]}
        _mc._merge_ti_gateway_models(available, _mc.ModelConfig.load())
        assert available == {"openai": ["gpt-4o"]}
        _mc.clear_caches()
