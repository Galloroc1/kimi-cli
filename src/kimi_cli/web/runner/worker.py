"""Worker module for running KimiCLI in a subprocess.

This module is the entry point for the subprocess that runs KimiCLI in wire mode.
It reads the session configuration from disk and runs KimiCLI.run_wire_stdio().

Usage:
    python -m kimi_cli.web.runner.worker <session_id>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any
from uuid import UUID

from kimi_cli import logger
from kimi_cli.app import KimiCLI, enable_logging
from kimi_cli.cli.mcp import get_global_mcp_config_file
from kimi_cli.exception import MCPConfigError
from kimi_cli.web.store.sessions import load_session_by_id


def _agent_ran_enabled() -> bool:
    env_flag = os.getenv("KIMI_SOUL", "").strip().lower() == "agent-ran"
    if env_flag:
        return True
    try:
        from kimi_cli.web.store.settings import load_web_settings

        return load_web_settings().agent_ran_enabled
    except Exception:
        return False


async def run_worker(session_id: UUID) -> None:
    """Run the KimiCLI worker for a session."""
    if _agent_ran_enabled():
        from kimi_cli.integrations.agent_ran_soul import build_agent_ran_soul
        from kimi_cli.wire.server import WireServer

        try:
            soul = build_agent_ran_soul()
        except Exception:
            logger.exception("Failed to start agent-ran soul")
            raise
        server = WireServer(soul)
        await server.serve()
        return

    # Find session by ID using the web store
    joint_session = load_session_by_id(session_id)
    if joint_session is None:
        raise ValueError(f"Session not found: {session_id}")

    # Get the kimi-cli session object
    session = joint_session.kimi_cli_session

    # Load default MCP config file if it exists
    default_mcp_file = get_global_mcp_config_file()
    mcp_configs: list[dict[str, Any]] = []
    if default_mcp_file.exists():
        raw = default_mcp_file.read_text(encoding="utf-8")
        try:
            mcp_configs = [json.loads(raw)]
        except json.JSONDecodeError:
            logger.warning(
                "Invalid JSON in MCP config file: {path}",
                path=default_mcp_file,
            )

    # Create KimiCLI instance with MCP configuration
    try:
        kimi_cli = await KimiCLI.create(session, mcp_configs=mcp_configs or None)
    except MCPConfigError as exc:
        logger.warning(
            "Invalid MCP config in {path}: {error}. Starting without MCP.",
            path=default_mcp_file,
            error=exc,
        )
        kimi_cli = await KimiCLI.create(session, mcp_configs=None)

    # Run in wire stdio mode
    await kimi_cli.run_wire_stdio()


def main() -> None:
    """Entry point for the worker subprocess."""
    from kimi_cli.utils.proctitle import set_process_title

    set_process_title("kimi-code-worker")

    if len(sys.argv) < 2:
        print("Usage: python -m kimi_cli.web.runner.worker <session_id>", file=sys.stderr)
        sys.exit(1)

    try:
        session_id = UUID(sys.argv[1])
    except ValueError:
        print(f"Invalid session ID: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    # Enable logging for the subprocess
    enable_logging(debug=False)

    # Run the async worker
    try:
        asyncio.run(run_worker(session_id))
    except Exception:
        logger.exception("Worker crashed")
        raise


if __name__ == "__main__":
    main()
