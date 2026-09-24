"""Exercise the real stdio entrypoint without a running Engram deployment."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("mcp")

from mcp.types import CallToolRequest, CallToolRequestParams

from engram.integrations import mcp_server
from engram.sdk import client as sdk_client


@pytest.mark.asyncio
async def test_stdio_subprocess_initializes_and_forwards_configured_url_and_key():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            body = json.dumps([{"id": "test-context", "name": "Local fixture"}]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    env = {
        **os.environ,
        "ENGRAM_API_URL": f"http://127.0.0.1:{http_server.server_port}/fixture",
        "ENGRAM_API_KEY": "eng_test_stdio_only",
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "engram.integrations.mcp_server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    async def send(message):
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()

    async def response(request_id):
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=20)
            assert line, "MCP server exited without a protocol response"
            message = json.loads(line)
            if message.get("id") == request_id:
                assert "error" not in message, message
                return message["result"]

    try:
        await send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "engram-regression", "version": "1.0"},
            },
        })
        initialized = await response(1)
        assert initialized["serverInfo"]["name"] == "engram"
        await send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        await send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = await response(2)
        assert {tool["name"] for tool in tools["tools"]} == {
            "recall_context", "save_context", "save_bullet", "record_decision",
            "list_contexts", "get_health", "archive_bullet", "get_lifecycle",
            "re_extract_context", "get_ingestion_config", "engram_list_contexts",
            "engram_recall", "engram_commit", "engram_decide", "engram_health",
            "engram_create_context",
        }
        await send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "engram_list_contexts", "arguments": {}},
        })
        result = await response(3)
        assert "Local fixture" in result["content"][0]["text"]
        assert requests == [("/fixture/contexts", "Bearer eng_test_stdio_only")]
        process.stdin.close()
        assert await asyncio.wait_for(process.wait(), timeout=10) == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await asyncio.to_thread(http_server.shutdown)
        http_server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_stdio_closes_owned_sdk_client_on_shutdown(monkeypatch, fail):
    client = SimpleNamespace(close=AsyncMock())
    created = []

    def make_client(**kwargs):
        created.append(kwargs)
        return client

    server = SimpleNamespace(
        run=AsyncMock(side_effect=RuntimeError("transport failed") if fail else None),
        create_initialization_options=lambda: "options",
    )
    injected = []

    def make_server(*args, **kwargs):
        injected.append(kwargs.get("client"))
        return server

    @asynccontextmanager
    async def transport():
        yield "read", "write"

    monkeypatch.setenv("ENGRAM_API_URL", "http://127.0.0.1:12345")
    monkeypatch.setenv("ENGRAM_API_KEY", "eng_test_lifecycle_only")
    monkeypatch.setattr(sdk_client, "Engram", make_client)
    monkeypatch.setattr(mcp_server, "create_mcp_server", make_server)
    monkeypatch.setattr(mcp_server, "stdio_server", transport)
    if fail:
        with pytest.raises(RuntimeError, match="transport failed"):
            await mcp_server.run_mcp_server()
    else:
        await mcp_server.run_mcp_server()
    assert created == [{"url": "http://127.0.0.1:12345", "api_key": "eng_test_lifecycle_only"}]
    assert injected == [client]
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_injected_sdk_client_is_used_and_remains_caller_owned(monkeypatch):
    client = SimpleNamespace(
        list_contexts=AsyncMock(return_value=[{"id": "injected", "name": "Injected"}]),
        close=AsyncMock(),
    )

    def unexpected_client(**kwargs):
        pytest.fail("An injected client must not create another SDK client")

    monkeypatch.setattr(sdk_client, "Engram", unexpected_client)
    server = mcp_server.create_mcp_server(client=client)
    async with server.lifespan(server):
        result = await server.request_handlers[CallToolRequest](CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name="engram_list_contexts", arguments={}),
        ))
    assert "Injected" in result.root.content[0].text
    client.list_contexts.assert_awaited_once_with(owner=None)
    client.close.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["record_decision", "engram_decide"])
async def test_decision_tools_display_id_returned_by_api(tool_name):
    client = SimpleNamespace(record_decision=AsyncMock(return_value={
        "concept_id": "returned-decision-id", "status": "recorded",
    }))
    server = mcp_server.create_mcp_server(client=client)
    result = await server.request_handlers[CallToolRequest](CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=tool_name, arguments={
            "context_id": "context", "decision": "Use SQLite", "rationale": "Local prototype",
        }),
    ))
    assert "returned-decision-id" in result.root.content[0].text
