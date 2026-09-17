"""Kerberos-ticket-backed JWT auth for model providers.

Implements the persistent token handler from the internal AI Knowledge Base
("Method 3", recommended for applications): a thread-safe, cached JWT manager
that exchanges a Kerberos ticket for a short-lived bearer token via an
external fetcher (by default `node get-token.js`) and refreshes it ahead of
expiry.

Auth mode toggle:

- `api` (default): existing behavior, provider API keys from env / `/auth`.
- `tijwt` (aliases `kerberos`, `ti`): fetch a TI JWT via `TIJWTManager` and
  use it as the provider `api_key` (Bearer) for model construction.

Resolution precedence for the mode is: `--auth-mode` CLI flag, then
`DEEPAGENTS_CODE_AUTH_MODE` (fallback `TI_AUTH_MODE`), then `[models] auth_mode`
in `~/.deepagents/config.toml`, then `api`.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

API_AUTH_MODE = "api"
TIJWT_AUTH_MODE = "tijwt"
KERBEROS_ALIASES: frozenset[str] = frozenset({"tijwt", "ti", "kerberos"})
"""Accepted spellings for the Kerberos-ticket JWT mode (canonical: `tijwt`)."""

DEFAULT_TOKEN_TTL_SECONDS = 3600
"""Lifetime assumed for a freshly fetched token when the fetcher reports none."""

DEFAULT_REFRESH_THRESHOLD_SECONDS = 300
"""Refresh the cached token this many seconds before `expires_at`."""

DEFAULT_FETCH_COMMAND: tuple[str, ...] = ("node", "get-token.js")
"""Default fetcher: exchanges the Kerberos ticket for a JWT on stdout."""

TI_GATEWAY_DEFAULT_BASE_URL = "https://llmgateway.itg.ti.com/v1"
"""Default LiteLLM gateway endpoint used in `tijwt` auth mode."""

TI_DEFAULT_TEAM_ID = "DAP_SG_CLAUDE_CODE"
"""Default `x-litellm-team-id` header value used in `tijwt` auth mode.

Mirrors the reference `TIJWTTokenProvider` default (`LITELLM_TEAM_ID` env with
this fallback).
"""

TI_TEAM_ID_HEADER = "x-litellm-team-id"
"""Request header carrying the LiteLLM team ID alongside the JWT bearer token."""

TI_MODELS_TIMEOUT_SECONDS = 15.0
"""HTTP timeout for the gateway `/v1/models` listing probe."""


class TIJWTError(RuntimeError):
    """Raised when a TI JWT cannot be fetched or the auth mode is invalid."""


@dataclass
class TIJWTToken:
    """A cached JWT with its absolute expiry time."""

    token: str
    expires_at: float


class TIJWTManager:
    """Thread-safe cached TI JWT handler with proactive refresh.

    Mirrors the Knowledge Base reference implementation: `get_token()` returns
    the cached token while it remains valid beyond the refresh threshold, and
    fetches a fresh one (via subprocess) otherwise. A lock guards the
    check-and-refresh so concurrent model calls fetch at most once.
    """

    def __init__(
        self,
        refresh_threshold_seconds: int = DEFAULT_REFRESH_THRESHOLD_SECONDS,
        *,
        token_ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
        fetch_command: tuple[str, ...] | list[str] | None = None,
        fetch_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the manager.

        Args:
            refresh_threshold_seconds: Refresh when fewer than this many
                seconds of validity remain.
            token_ttl_seconds: Assumed lifetime for a fetched token.
            fetch_command: External fetcher argv. Defaults to
                `("node", "get-token.js")`; override with
                `DEEPAGENTS_CODE_TI_GET_TOKEN_CMD` / `TI_GET_TOKEN_CMD`.
            fetch_timeout_seconds: Subprocess timeout for the fetcher.
        """
        self.refresh_threshold = refresh_threshold_seconds
        self.token_ttl_seconds = token_ttl_seconds
        self.fetch_command = (
            tuple(fetch_command) if fetch_command else DEFAULT_FETCH_COMMAND
        )
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self._current_token: TIJWTToken | None = None
        self._lock = threading.Lock()

    def _fetch_fresh_token(self) -> str:
        """Run the external fetcher and return the stripped JWT.

        Returns:
            The fresh token string.

        Raises:
            TIJWTError: If the fetcher fails, times out, or returns empty output.
        """
        try:
            result = subprocess.run(
                list(self.fetch_command),
                capture_output=True,
                text=True,
                check=True,
                timeout=self.fetch_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"TI token fetcher timed out: {' '.join(self.fetch_command)}"
            raise TIJWTError(msg) from exc
        except (subprocess.CalledProcessError, OSError) as exc:
            msg = f"TI token fetcher failed: {' '.join(self.fetch_command)}: {exc}"
            raise TIJWTError(msg) from exc
        token = result.stdout.strip()
        if not token:
            msg = "TI token fetcher returned empty output"
            raise TIJWTError(msg)
        return token

    def get_token(self) -> str:
        """Return a valid JWT, refreshing via the fetcher when near expiry.

        Returns:
            A valid bearer token string.

        Raises:
            TIJWTError: If a required refresh fails.
        """
        with self._lock:
            now = time.time()
            if self._current_token is None or (
                now >= self._current_token.expires_at - self.refresh_threshold
            ):
                fresh_token = self._fetch_fresh_token()
                self._current_token = TIJWTToken(
                    token=fresh_token,
                    expires_at=now + self.token_ttl_seconds,
                )
            return self._current_token.token

    def force_refresh(self) -> str:
        """Discard the cache and fetch a fresh token unconditionally.

        Returns:
            The fresh token string.

        Raises:
            TIJWTError: If the fetch fails; the previous cache is cleared.
        """
        with self._lock:
            self._current_token = None
        return self.get_token()

    def clear(self) -> None:
        """Drop the cached token so the next `get_token()` refetches."""
        with self._lock:
            self._current_token = None

    @property
    def expires_at(self) -> float | None:
        """Expiry of the cached token, or `None` when nothing is cached."""
        return self._current_token.expires_at if self._current_token else None


_manager: TIJWTManager | None = None
_manager_lock = threading.Lock()


def normalize_auth_mode(raw: str | None) -> str | None:
    """Normalize a raw auth-mode value to its canonical form.

    Args:
        raw: User-supplied value (e.g. `"Kerberos"`, `"TI"`, `"api"`).

    Returns:
        `"api"` or `"tijwt"`, or `None` when `raw` is empty/`None`.

    Raises:
        TIJWTError: If the value is not a recognized auth mode.
    """
    if raw is None:
        return None
    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    if cleaned == API_AUTH_MODE:
        return API_AUTH_MODE
    if cleaned in KERBEROS_ALIASES:
        return TIJWT_AUTH_MODE
    msg = f"Invalid auth mode {raw!r}: expected 'api' or 'tijwt' ('kerberos' alias)"
    raise TIJWTError(msg)


def _resolve_fetch_command() -> tuple[str, ...]:
    """Resolve the token fetcher argv from env or the default.

    Reads `DEEPAGENTS_CODE_TI_GET_TOKEN_CMD` (fallback `TI_GET_TOKEN_CMD`) as a
    shell-split command string; unset means the default fetcher.

    Returns:
        The fetcher argv tuple.
    """
    # Late import keeps this module import-light for the startup hot path.
    from deepagents_code import _env_vars

    raw = os.environ.get(_env_vars.TI_GET_TOKEN_CMD) or os.environ.get(
        "TI_GET_TOKEN_CMD"
    )
    if raw is None:
        try:
            from deepagents_code.config_manifest import load_config_toml

            data = load_config_toml()
            models = data.get("models")
            if isinstance(models, dict):
                configured = models.get("ti_get_token_cmd")
                if isinstance(configured, str) and configured.strip():
                    raw = configured
        except Exception:
            logger.debug("Could not read [models].ti_get_token_cmd", exc_info=True)
    if not raw or not raw.strip():
        return DEFAULT_FETCH_COMMAND
    return tuple(shlex.split(raw.strip()))


def resolve_base_url() -> str:
    """Resolve the LiteLLM gateway endpoint used in `tijwt` auth mode.

    Precedence: `DEEPAGENTS_CODE_TI_BASE_URL` (fallback `TI_BASE_URL`), then
    `[models].ti_base_url` in `config.toml`, then the default gateway.

    Returns:
        The gateway base URL (e.g. `https://llmgateway.itg.ti.com/v1`).
    """
    from deepagents_code import _env_vars

    for name in (_env_vars.TI_BASE_URL, "TI_BASE_URL"):
        raw = os.environ.get(name)
        if raw and raw.strip():
            return raw.strip()
    try:
        from deepagents_code.config_manifest import load_config_toml

        data = load_config_toml()
        models = data.get("models")
        if isinstance(models, dict):
            configured = models.get("ti_base_url")
            if isinstance(configured, str) and configured.strip():
                return configured.strip()
    except Exception:
        logger.debug("Could not read [models].ti_base_url", exc_info=True)
    return TI_GATEWAY_DEFAULT_BASE_URL


def resolve_team_id() -> str:
    """Resolve the `x-litellm-team-id` header value used in `tijwt` auth mode.

    Precedence: `DEEPAGENTS_CODE_TI_TEAM_ID` (fallbacks `TI_TEAM_ID`,
    `LITELLM_TEAM_ID` — the env name used by the reference
    `TIJWTTokenProvider`), then `[models].ti_team_id` in `config.toml`, then
    the default team.

    Returns:
        The LiteLLM team ID string.
    """
    from deepagents_code import _env_vars

    for name in (_env_vars.TI_TEAM_ID, "TI_TEAM_ID", "LITELLM_TEAM_ID"):
        raw = os.environ.get(name)
        if raw and raw.strip():
            return raw.strip()
    try:
        from deepagents_code.config_manifest import load_config_toml

        data = load_config_toml()
        models = data.get("models")
        if isinstance(models, dict):
            configured = models.get("ti_team_id")
            if isinstance(configured, str) and configured.strip():
                return configured.strip()
    except Exception:
        logger.debug("Could not read [models].ti_team_id", exc_info=True)
    return TI_DEFAULT_TEAM_ID


def team_headers() -> dict[str, str]:
    """Return the gateway team headers for the current auth mode.

    Returns:
        `{TI_TEAM_ID_HEADER: team_id}` when `tijwt` mode is active and a team
            ID resolves, otherwise `{}`. Never raises: resolution failures
            fall back to an empty mapping so model construction proceeds with
            just the bearer token.
    """
    try:
        if resolve_auth_mode() != TIJWT_AUTH_MODE:
            return {}
        return {TI_TEAM_ID_HEADER: resolve_team_id()}
    except Exception:
        logger.debug("Could not resolve TI team headers", exc_info=True)
        return {}


def resolve_verify_ssl() -> bool:
    """Resolve whether the TI gateway connection verifies TLS certificates.

    Precedence: `DEEPAGENTS_CODE_TI_VERIFY_SSL` (fallback `TI_VERIFY_SSL`),
    then `[models].ti_verify_ssl` in `config.toml`, then `True` (verify on).
    An unrecognized env value or non-bool TOML value is logged and falls
    through to the next layer — a bad value must never silently disable
    verification.

    Returns:
        `True` (default) to verify TLS, `False` to skip verification.
    """
    from deepagents_code import _env_vars
    from deepagents_code._env_vars import classify_env_bool

    for name in (_env_vars.TI_VERIFY_SSL, "TI_VERIFY_SSL"):
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        classified = classify_env_bool(raw)
        if classified is None:
            logger.warning("Ignoring %s=%r (expected bool)", name, raw)
            continue
        return classified
    try:
        from deepagents_code.config_manifest import load_config_toml

        data = load_config_toml()
        models = data.get("models")
        if isinstance(models, dict):
            configured = models.get("ti_verify_ssl")
            if isinstance(configured, bool):
                return configured
            if configured is not None:
                logger.warning(
                    "Ignoring [models].ti_verify_ssl=%r in config.toml"
                    " (expected bool)",
                    configured,
                )
    except Exception:
        logger.debug("Could not read [models].ti_verify_ssl", exc_info=True)
    return True


def list_gateway_models(*, timeout: float = TI_MODELS_TIMEOUT_SECONDS) -> list[str]:
    """List model IDs served by the TI LiteLLM gateway (`GET /v1/models`).

    Authenticates with the cached TI JWT plus the team header, mirroring the
    reference "Method 2" snippet. Proxy servers from the environment are
    honored (`urllib` reads `HTTP(S)_PROXY`/`NO_PROXY`), and the
    `ti_verify_ssl` setting controls TLS verification. Best-effort by design:
    any failure (missing ticket, unreachable gateway, auth rejection,
    malformed payload) yields an empty list with a log line instead of
    raising, so model discovery can never break the `/model` selector.

    Args:
        timeout: HTTP timeout in seconds for the listing probe.

    Returns:
        Sorted model IDs reported by the gateway, or `[]` when the probe
            fails for any reason.

    Raises:
        TIJWTError: If the auth mode is not `tijwt` (a programming error —
            callers must gate on `resolve_auth_mode` first) or no valid
            configuration resolves. Token-fetch failures are *not* raised;
            they degrade to `[]` like every other probe failure.
    """
    import json
    import ssl
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    if resolve_auth_mode() != TIJWT_AUTH_MODE:
        msg = "list_gateway_models requires tijwt auth mode"
        raise TIJWTError(msg)
    try:
        token = get_tijwt_token()
    except TIJWTError as exc:
        logger.debug("TI gateway model probe skipped: %s", exc)
        return []
    base_url = resolve_base_url().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        logger.warning(
            "Skipping TI gateway model probe: %r has no http:// or https:// scheme",
            base_url,
        )
        return []
    url = f"{base_url}/v1/models"
    headers = {
        "Authorization": f"Bearer {token}",
        TI_TEAM_ID_HEADER: resolve_team_id(),
    }
    request = Request(url, headers=headers)  # noqa: S310  # scheme guarded above
    context = None
    if url.startswith("https://") and not resolve_verify_ssl():
        # Explicit user opt-out (`ti_verify_ssl = false`): skip certificate
        # verification for self-signed corporate gateways.
        context = ssl._create_unverified_context()  # noqa: S323
    try:
        with urlopen(  # noqa: S310  # scheme guarded above
            request, timeout=timeout, context=context
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        logger.debug("TI gateway model probe failed for %s: %s", url, exc)
        return []
    except Exception as exc:  # noqa: BLE001  # discovery is best-effort
        logger.warning(
            "TI gateway model probe raised unexpected %s for %s: %s",
            type(exc).__name__,
            url,
            exc,
        )
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        logger.debug("TI gateway model probe: unexpected payload shape")
        return []
    return sorted(
        entry["id"]
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )


def resolve_auth_mode(*, cli_value: str | None = None) -> str:
    """Resolve the effective model auth mode.

    Precedence: `cli_value` (`--auth-mode`), then
    `DEEPAGENTS_CODE_AUTH_MODE` (fallback `TI_AUTH_MODE`), then
    `[models] auth_mode` in `config.toml`, then `api`.

    Args:
        cli_value: Value from the `--auth-mode` CLI flag, if given.

    Returns:
        `"api"` or `"tijwt"`.

    Raises:
        TIJWTError: If a configured value is not a recognized auth mode.
    """
    from deepagents_code import _env_vars

    if cli_value is not None:
        normalized = normalize_auth_mode(cli_value)
        if normalized is not None:
            return normalized
    for name in (_env_vars.AUTH_MODE, "TI_AUTH_MODE"):
        raw = os.environ.get(name)
        if raw:
            return normalize_auth_mode(raw) or API_AUTH_MODE
    try:
        from deepagents_code.config_manifest import load_config_toml

        data = load_config_toml()
        models = data.get("models")
        if isinstance(models, dict):
            configured = models.get("auth_mode")
            if isinstance(configured, str) and configured.strip():
                return normalize_auth_mode(configured) or API_AUTH_MODE
    except TIJWTError:
        raise
    except Exception:
        logger.debug("Could not read [models].auth_mode", exc_info=True)
    return API_AUTH_MODE


def is_tijwt_auth_mode(*, cli_value: str | None = None) -> bool:
    """Return whether the effective auth mode is the TI JWT (Kerberos) mode.

    Args:
        cli_value: Value from the `--auth-mode` CLI flag, if given.

    Returns:
        `True` when the resolved mode is `tijwt`.
    """
    try:
        return resolve_auth_mode(cli_value=cli_value) == TIJWT_AUTH_MODE
    except TIJWTError:
        raise


def get_manager() -> TIJWTManager:
    """Return the process-wide `TIJWTManager` singleton.

    Returns:
        The shared manager, built from the current fetcher env/config.
    """
    global _manager  # noqa: PLW0603
    with _manager_lock:
        if _manager is None:
            _manager = TIJWTManager(fetch_command=_resolve_fetch_command())
        return _manager


def reset_manager() -> None:
    """Drop the process-wide manager (test hook for fetcher re-resolution)."""
    global _manager  # noqa: PLW0603
    with _manager_lock:
        _manager = None


def get_tijwt_token() -> str:
    """Fetch (or reuse the cached) TI JWT for model auth.

    Returns:
        A valid bearer token string.

    Raises:
        TIJWTError: If the token cannot be fetched.
    """
    return get_manager().get_token()
