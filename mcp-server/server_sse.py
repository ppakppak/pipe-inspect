#!/usr/bin/env python3
"""
Pipe Inspector MCP Server (SSE Mode)
GPU 서버에서 실행되는 MCP 서버 - HTTP/SSE 기반
"""

import sys
import json
from pathlib import Path
from typing import Any

# Grounded-SAM 경로 추가
sys.path.insert(0, '/home/ppak/SynologyDrive/ykpark/linux_devel/ground_sam/Grounded-Segment-Anything')

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response

from project_manager import ProjectManager, Project


# MCP 서버 인스턴스
mcp_server = Server("pipe-inspector-mcp")


# ============================================================================
# 도구 목록 및 핸들러
# ============================================================================

@mcp_server.list_tools()
async def list_tools():
    """사용 가능한 도구 목록"""
    from mcp.types import Tool

    return [
        Tool(
            name="list_projects",
            description="모든 프로젝트 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="create_project",
            description="새로운 프로젝트 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "classes": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string", "default": ""}
                },
                "required": ["name", "classes"]
            }
        ),
        Tool(
            name="get_project",
            description="프로젝트 상세 정보 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"}
                },
                "required": ["project_id"]
            }
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any):
    """도구 실행"""
    from mcp.types import TextContent

    pm = ProjectManager()

    try:
        if name == "list_projects":
            projects = pm.list_projects()
            projects_data = []
            for p in projects:
                projects_data.append({
                    "id": p.id,
                    "name": p.name,
                    "path": str(p.project_dir),
                    "classes": p.classes
                })
            result = {"success": True, "projects": projects_data}
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        elif name == "create_project":
            project = pm.create_project(
                name=arguments["name"],
                classes=arguments["classes"],
                description=arguments.get("description", "")
            )
            result = {
                "success": True,
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "path": str(project.project_dir)
                }
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        elif name == "get_project":
            projects = pm.list_projects()
            project = None
            for p in projects:
                if p.id == arguments["project_id"]:
                    project = p
                    break

            if not project:
                result = {"success": False, "error": "Project not found"}
                return [TextContent(type="text", text=json.dumps(result))]

            stats = project.get_statistics()
            result = {
                "success": True,
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "path": str(project.project_dir),
                    "stats": stats
                }
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}
            return [TextContent(type="text", text=json.dumps(result))]

    except Exception as e:
        import traceback
        result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return [TextContent(type="text", text=json.dumps(result))]


# ============================================================================
# Starlette 앱 설정
# ============================================================================

async def handle_sse(request):
    """SSE 엔드포인트"""
    async with SseServerTransport("/messages") as transport:
        await mcp_server.run(
            transport[0],
            transport[1],
            mcp_server.create_initialization_options()
        )
    return Response()


async def handle_messages(request):
    """메시지 엔드포인트"""
    async with SseServerTransport("/messages") as transport:
        await mcp_server.run(
            transport[0],
            transport[1],
            mcp_server.create_initialization_options()
        )
    return Response()


app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Pipe Inspector MCP Server (SSE Mode)...", file=sys.stderr)
    print("📡 Server: http://0.0.0.0:5002", file=sys.stderr)

    uvicorn.run(app, host="0.0.0.0", port=5002)
