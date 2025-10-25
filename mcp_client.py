#!/usr/bin/env python3
"""
MCP Client for Pipe Inspector
Flask Backend가 MCP 서버와 통신하기 위한 클라이언트
"""

import json
import asyncio
import sys
from typing import Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """MCP 서버 클라이언트 (Async Context Manager)"""

    def __init__(self, server_script_path: str):
        """
        Args:
            server_script_path: MCP 서버 스크립트 경로
        """
        self.server_script_path = server_script_path
        self.session: Optional[ClientSession] = None
        self._stdio_context = None
        self._read = None
        self._write = None

    async def __aenter__(self):
        """Async context manager 진입"""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script_path],
            env=None,
        )

        # stdio_client를 async context manager로 사용
        self._stdio_context = stdio_client(server_params)
        self._read, self._write = await self._stdio_context.__aenter__()

        self.session = ClientSession(self._read, self._write)
        await self.session.__aenter__()

        # 세션 초기화 (중요!)
        await self.session.initialize()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager 종료"""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._stdio_context:
            await self._stdio_context.__aexit__(exc_type, exc_val, exc_tb)

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        MCP 도구 호출

        Args:
            tool_name: 도구 이름
            arguments: 도구 인자

        Returns:
            도구 실행 결과 (JSON 파싱됨)
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        result = await self.session.call_tool(tool_name, arguments)

        # TextContent에서 텍스트 추출
        if result.content and len(result.content) > 0:
            text_content = result.content[0].text
            return json.loads(text_content)
        else:
            return {"success": False, "error": "No response from server"}

    async def list_tools(self) -> list:
        """사용 가능한 도구 목록 조회"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools = await self.session.list_tools()
        return tools.tools
