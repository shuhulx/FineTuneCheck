"""MCP server for running FineTuneCheck from an AI assistant."""

from __future__ import annotations

import logging
import traceback

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from finetunecheck.mcp.schemas import (
    COMPARE_RUNS_SCHEMA,
    DETECT_FORGETTING_SCHEMA,
    EVALUATE_FINETUNE_SCHEMA,
    GENERATE_REPORT_SCHEMA,
    GET_VERDICT_SCHEMA,
    LIST_PROFILES_SCHEMA,
    QUICK_CHECK_SCHEMA,
    RUN_PROBE_SCHEMA,
    SUGGEST_FIXES_SCHEMA,
)
from finetunecheck.mcp.tools import TOOL_HANDLERS

logger = logging.getLogger(__name__)

server = Server("finetunecheck")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="evaluate_finetune",
            description=(
                "Compare a base model with a fine-tuned model and return the scores, "
                "regressions, and samples behind the result."
            ),
            inputSchema=EVALUATE_FINETUNE_SCHEMA,
        ),
        Tool(
            name="quick_check",
            description=(
                "Run a small local check for math, classification, instruction following, "
                "and refusal behavior."
            ),
            inputSchema=QUICK_CHECK_SCHEMA,
        ),
        Tool(
            name="detect_forgetting",
            description=(
                "Show which non-target capabilities changed after fine-tuning, including "
                "retention rates and the worst regressions."
            ),
            inputSchema=DETECT_FORGETTING_SCHEMA,
        ),
        Tool(
            name="compare_runs",
            description=(
                "Compare multiple fine-tuning runs against the same base model. "
                "Shows the best target score, strongest retention, overall ROI, and "
                "Pareto frontier."
            ),
            inputSchema=COMPARE_RUNS_SCHEMA,
        ),
        Tool(
            name="get_verdict",
            description=(
                "Get a quick verdict. Returns INSUFFICIENT_EVIDENCE when important "
                "results are missing or the sample is too small."
            ),
            inputSchema=GET_VERDICT_SCHEMA,
        ),
        Tool(
            name="suggest_fixes",
            description=(
                "Suggest practical follow-up checks and training changes based on the "
                "problems found in a run."
            ),
            inputSchema=SUGGEST_FIXES_SCHEMA,
        ),
        Tool(
            name="generate_report",
            description=(
                "Write an HTML, JSON, CSV, or Markdown report with charts and sample-level details."
            ),
            inputSchema=GENERATE_REPORT_SCHEMA,
        ),
        Tool(
            name="list_profiles",
            description="Show available evaluation profiles (code, chat, safety, etc.).",
            inputSchema=LIST_PROFILES_SCHEMA,
        ),
        Tool(
            name="run_probe",
            description=(
                "Run one probe set on a model when you only want to inspect a single capability."
            ),
            inputSchema=RUN_PROBE_SCHEMA,
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )

    try:
        result = await handler(arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=result)],
            isError=False,
        )
    except Exception as exc:
        logger.error("Error running %s: %s\n%s", name, exc, traceback.format_exc())
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error running {name}: {exc}")],
            isError=True,
        )


async def main() -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
