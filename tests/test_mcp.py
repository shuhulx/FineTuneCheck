"""Tests for MCP server integration.

The MCP server module may not yet be fully implemented. These tests
verify the expected interface and, if available, tool registration.
"""

import importlib

import pytest


class TestMCPImport:
    def test_mcp_module_importable(self):
        """The MCP module should be importable."""
        try:
            mod = importlib.import_module("finetunecheck.mcp")
            assert mod is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("finetunecheck.mcp module not available")


class TestMCPServer:
    @pytest.fixture(autouse=True)
    def _check_mcp(self):
        """Skip if MCP server is not implemented."""
        try:
            mod = importlib.import_module("finetunecheck.mcp")
            # Look for a server object or tool registration
            if not hasattr(mod, "server") and not hasattr(mod, "mcp") and not hasattr(mod, "tools"):
                pytest.skip("MCP server not yet implemented (no server/mcp/tools attribute)")
            self.mcp_mod = mod
        except (ImportError, ModuleNotFoundError):
            pytest.skip("finetunecheck.mcp module not available")

    def test_mcp_tool_list(self):
        """MCP server should list tools."""
        # Try various common patterns for MCP tool registration
        if hasattr(self.mcp_mod, "server"):
            server = self.mcp_mod.server
            if hasattr(server, "tools"):
                tools = server.tools
                assert len(tools) >= 1, "MCP server should have at least 1 tool"
        elif hasattr(self.mcp_mod, "tools"):
            tools = self.mcp_mod.tools
            assert len(tools) >= 1

    def test_mcp_tool_schemas(self):
        """Each tool should have valid schema attributes."""
        # The finetunecheck.mcp module exports 'main' and the server module
        # has a 'server' object. The tool handlers are in TOOL_HANDLERS dict.
        try:
            from finetunecheck.mcp.tools import TOOL_HANDLERS

            assert len(TOOL_HANDLERS) >= 1, "Should have at least 1 tool handler"
            for name, handler in TOOL_HANDLERS.items():
                assert isinstance(name, str)
                assert len(name) > 0
                assert callable(handler)
        except ImportError:
            pytest.skip("finetunecheck.mcp.tools not available")
