"""The `dcode models` command group: inspect the available model lineup.

`dcode models list` prints every `provider:model` spec the `/model` switcher
can offer, grouped by provider. In `tijwt` auth mode the corporate LiteLLM
gateway lineup is probed live (`GET /v1/models` with the Kerberos JWT plus
team header) and merged under the OpenAI-compatible provider entry, with a
dedicated section calling out which rows came from the gateway — handy for
checking which model IDs the gateway actually serves before setting one.

Help rendering for `dcode models -h` / `dcode models list -h` is served by
`ui.show_models_help` / `ui.show_models_list_help`, which do not import this
module, so the help path stays light.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents_code.output import write_json

if TYPE_CHECKING:
    import argparse

    from deepagents_code.output import OutputFormat

logger = logging.getLogger(__name__)


def run_models_command(args: argparse.Namespace) -> int:
    """Dispatch a `dcode models` subcommand.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Process exit code (`0`; discovery is best-effort and never fails the
            command — an unreachable gateway degrades to a printed note).
    """
    subcommand = getattr(args, "models_command", None)
    if subcommand == "list":
        return _run_models_list(args)

    # `cli_main`'s bare-group help fast path handles `dcode models` with no
    # subcommand, so this is only reached for an unexpected value.
    from deepagents_code import ui

    ui.show_models_help()
    return 0


def _run_models_list(args: argparse.Namespace) -> int:
    """List the models available to the agent, grouped by provider.

    Enumerates `get_available_models()` — the same mapping the `/model`
    switcher renders — so names never drift from what the TUI offers. Gateway
    discovery is best-effort: when `tijwt` mode is active but the probe fails
    (expired ticket, unreachable gateway), the static catalog still renders
    with a note explaining the missing gateway section.

    Args:
        args: Parsed CLI namespace. Only `output_format` is read.

    Returns:
        `0` always.
    """
    from deepagents_code.model_config import (
        get_available_models,
        get_ti_gateway_model_specs,
    )

    output_format: OutputFormat = getattr(args, "output_format", "text")
    available = get_available_models()
    gateway_specs = get_ti_gateway_model_specs()
    gateway_info = _gateway_info(gateway_specs)

    if output_format == "json":
        specs = {
            provider: [f"{provider}:{model}" for model in models]
            for provider, models in available.items()
        }
        payload: dict[str, object] = {
            "models": specs,
            "count": sum(len(models) for models in specs.values()),
            "auth_mode": gateway_info["auth_mode"],
            "gateway_base_url": gateway_info["base_url"],
            "gateway_models": sorted(gateway_specs),
        }
        write_json("models list", payload)
        return 0

    _print_models(available, gateway_specs, gateway_info)
    return 0


def _gateway_info(gateway_specs: list[str]) -> dict[str, object]:
    """Describe TI gateway state for the listing output.

    Args:
        gateway_specs: Resolved `provider:model` specs from the live probe.

    Returns:
        Mapping with `active` (tijwt mode on), `auth_mode`, `base_url`
            (`None` unless active), and `probed` (active and probe returned
            models — `False` means the section is missing because the probe
            failed, not because the gateway is model-less).
    """
    from deepagents_code import tijwt as _tijwt

    try:
        active = _tijwt.resolve_auth_mode() == _tijwt.TIJWT_AUTH_MODE
    except Exception:
        logger.debug("Could not resolve auth mode for models list", exc_info=True)
        return {"active": False, "auth_mode": "api", "base_url": None, "probed": False}
    base_url = _tijwt.resolve_base_url().rstrip("/") if active else None
    return {
        "active": active,
        "auth_mode": _tijwt.TIJWT_AUTH_MODE if active else "api",
        "base_url": base_url,
        "probed": active and bool(gateway_specs),
    }


def _print_models(
    available: dict[str, list[str]],
    gateway_specs: list[str],
    gateway_info: dict[str, object],
) -> None:
    """Render the provider-grouped model listing to the console.

    Args:
        available: Provider-to-model-IDs mapping from `get_available_models()`.
        gateway_specs: Live gateway `provider:model` specs for the callout.
        gateway_info: Gateway state from `_gateway_info`.
    """
    from deepagents_code.config import console

    total = sum(len(models) for models in available.values())
    noun = "model" if total == 1 else "models"

    console.print()
    console.print(f"{total} {noun} available", highlight=False)
    for provider, models in available.items():
        if not models:
            continue
        console.print()
        console.print(provider, style="bold", markup=False, highlight=False)
        for model in models:
            # `markup=False`/`highlight=False`: model IDs may contain brackets.
            console.print(
                f"  {provider}:{model}",
                markup=False,
                highlight=False,
                no_wrap=True,
                crop=True,
            )

    if gateway_info["active"]:
        console.print()
        if gateway_specs:
            console.print(
                f"TI gateway models (live from {gateway_info['base_url']})",
                style="bold",
                markup=False,
                highlight=False,
            )
            for spec in sorted(gateway_specs):
                console.print(
                    f"  {spec}",
                    markup=False,
                    highlight=False,
                    no_wrap=True,
                    crop=True,
                )
        else:
            console.print(
                "Note: TI gateway probe returned no models "
                f"({gateway_info['base_url']}/v1/models) — "
                "check the Kerberos ticket (`/auth-mode status`), "
                "team ID, and TLS setting.",
                style="yellow",
                highlight=False,
            )
    console.print()
