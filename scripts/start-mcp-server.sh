#!/bin/bash

# MCP Server 시작 스크립트 (GPU 서버용)

cd /home/ppak/pipe-inspector-electron

echo "🚀 Starting MCP Server (GPU)..."
echo "📡 MCP Server ready for stdio connections"

# MCP 서버 실행 (stdio 모드)
conda run -n mcp-server python3 mcp-server/server.py
