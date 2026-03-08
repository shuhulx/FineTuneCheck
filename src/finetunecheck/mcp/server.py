"""MCP server for FineTuneCheck — exposes evaluation tools to AI assistants."""

from __future__ import annotations

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

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

server = Server("finetunecheck")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="evaluate_finetune",
            description=(
                "Run comprehensive fine-tuning evaluation comparing a base model "
                "against a fine-tuned model. Returns verdict, ROI score, category "
                "scores, forgetting analysis, and recommendations."
            ),
            inputSchema=EVALUATE_FINETUNE_SCHEMA,
        ),
        Tool(
            name="quick_check",
            description=(
                "Fast 5-minute evaluation on 4 core capabilities (reasoning, code, "
                "math, safety) with 20 samples each. Use for rapid iteration."
            ),
            inputSchema=QUICK_CHECK_SCHEMA,
        ),
        Tool(
            name="detect_forgetting",
            description=(
                "Analyze catastrophic forgetting in a fine-tuned model. Returns "
                "forgetting pattern, backward transfer score, capability retention "
                "rates, and worst regressions."
            ),
            inputSchema=DETECT_FORGETTING_SCHEMA,
        ),
        Tool(
            name="compare_runs",
            description=(
                "Compare multiple fine-tuning runs against the same base model. "
                "Finds best run by ROI, best target performance, least forgetting, "
                "and computes Pareto frontier."
            ),
            inputSchema=COMPARE_RUNS_SCHEMA,
        ),
        Tool(
            name="get_verdict",
            description=(
                "Get a quick deployment readiness assessment: EXCELLENT, GOOD, "
                "GOOD_WITH_CONCERNS, POOR, or HARMFUL."
            ),
            inputSchema=GET_VERDICT_SCHEMA,
        ),
        Tool(
            name="suggest_fixes",
            description=(
                "Get actionable recommendations for improving a fine-tuning "
                "outcome based on detected issues."
            ),
            inputSchema=SUGGEST_FIXES_SCHEMA,
        ),
        Tool(
            name="generate_report",
            description=(
                "Create a diagnostic report (HTML, JSON, CSV, or Markdown) "
                "with visualizations and detailed analysis."
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
                "Run a specific probe set on a single model. Useful for testing "
                "individual capabilities."
            ),
            inputSchema=RUN_PROBE_SCHEMA,
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        result = await handler(arguments)
        return [TextContent(type="text", text=result)]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error running {name}: {exc}")]


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
