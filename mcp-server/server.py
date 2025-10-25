#!/usr/bin/env python3
"""
Pipe Inspector MCP Server
GPU 서버에서 실행되는 MCP 서버 - Grounded-SAM 작업 처리
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Sequence

# Grounded-SAM 경로 추가
sys.path.insert(0, '/home/ppak/SynologyDrive/ykpark/linux_devel/ground_sam/Grounded-Segment-Anything')

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from project_manager import ProjectManager, Project


# MCP 서버 인스턴스
app = Server("pipe-inspector-mcp")


# ============================================================================
# 프로젝트 관리 도구
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록"""
    return [
        Tool(
            name="create_project",
            description="새로운 프로젝트 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "프로젝트 이름"
                    },
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "클래스 목록 (예: ['pipe', 'crack', 'corrosion'])"
                    },
                    "description": {
                        "type": "string",
                        "description": "프로젝트 설명 (선택사항)",
                        "default": ""
                    }
                },
                "required": ["name", "classes"]
            }
        ),
        Tool(
            name="list_projects",
            description="모든 프로젝트 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_project",
            description="프로젝트 상세 정보 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "프로젝트 ID"
                    }
                },
                "required": ["project_id"]
            }
        ),
        Tool(
            name="delete_project",
            description="프로젝트 삭제",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "프로젝트 경로"
                    }
                },
                "required": ["project_path"]
            }
        ),
        Tool(
            name="add_video",
            description="프로젝트에 비디오 추가",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "프로젝트 ID"
                    },
                    "video_path": {
                        "type": "string",
                        "description": "비디오 파일 경로"
                    }
                },
                "required": ["project_id", "video_path"]
            }
        ),
        Tool(
            name="remove_video",
            description="프로젝트에서 비디오 제거",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "프로젝트 ID"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "비디오 ID"
                    }
                },
                "required": ["project_id", "video_id"]
            }
        ),
        Tool(
            name="get_project_statistics",
            description="프로젝트 통계 정보 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "프로젝트 ID"
                    }
                },
                "required": ["project_id"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """도구 실행"""

    pm = ProjectManager()

    try:
        if name == "create_project":
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
                    "path": str(project.project_dir),
                    "classes": project.classes
                }
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "list_projects":
            projects = pm.list_projects()
            projects_data = []
            for p in projects:
                projects_data.append({
                    "id": p.id,
                    "name": p.name,
                    "path": str(p.project_dir),
                    "classes": p.classes
                })
            result = {
                "success": True,
                "projects": projects_data
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

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
            videos_data = []
            for video in project.videos:
                videos_data.append({
                    "id": video.get("video_id", ""),
                    "filename": video.get("filename", ""),
                    "path": video.get("path", ""),
                    "total_frames": video.get("total_frames", 0),
                    "status": video.get("status", ""),
                })

            result = {
                "success": True,
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "path": str(project.project_dir),
                    "classes": project.classes,
                    "stats": stats,
                    "videos": videos_data
                }
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "delete_project":
            pm.delete_project(arguments["project_path"])
            result = {"success": True}
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "add_video":
            projects = pm.list_projects()
            project = None
            for p in projects:
                if p.id == arguments["project_id"]:
                    project = p
                    break

            if not project:
                result = {"success": False, "error": "Project not found"}
                return [TextContent(type="text", text=json.dumps(result))]

            video_id = project.add_video(arguments["video_path"])
            result = {"success": True, "video_id": video_id}
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "remove_video":
            projects = pm.list_projects()
            project = None
            for p in projects:
                if p.id == arguments["project_id"]:
                    project = p
                    break

            if not project:
                result = {"success": False, "error": "Project not found"}
                return [TextContent(type="text", text=json.dumps(result))]

            project.remove_video(arguments["video_id"])
            result = {"success": True}
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "get_project_statistics":
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
            result = {"success": True, "statistics": stats}
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}
            return [TextContent(type="text", text=json.dumps(result))]

    except Exception as e:
        import traceback
        error_info = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return [TextContent(type="text", text=json.dumps(error_info))]


async def main():
    """MCP 서버 시작"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    print("🚀 Starting Pipe Inspector MCP Server...", file=sys.stderr)
    print("📡 Server ready for connections", file=sys.stderr)
    asyncio.run(main())
